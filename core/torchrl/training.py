"""Training loop, evaluation, and advantage computation for TorchRL PPO/GRPO.

Supports two advantage estimation methods:
- **PPO**: Generalized Advantage Estimation (GAE) via TorchRL's GAE module.
- **GRPO**: Group Relative Policy Optimization — trajectory-level group
  normalization without a critic.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    policy,
    env_name: str,
    n_episodes: int = 10,
    seed: int = 1042,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run evaluation episodes and return mean/std of raw episode returns.

    Creates a fresh single environment (no weights, no normalization) and
    runs the policy greedily for *n_episodes* complete episodes.  Returns
    metrics compatible with the SB3 ``evaluate_policy`` contract.

    Args:
        policy: Trained ``ProbabilisticActor`` (TorchRL actor network).
        env_name: Name from ``ENV_REGISTRY`` (e.g. ``"lunarlander"``).
        n_episodes: Number of evaluation episodes.
        seed: RNG seed for the eval environment.
        device: Torch device for policy inference.

    Returns:
        Dict with ``"eval/mean_reward"`` and ``"eval/std_reward"``.
    """
    from .env import build_env

    # Use build_env (not make_single_env) so eval gets the same transforms
    # as training (obs clipping, etc.). Without this, the policy sees raw
    # observations it was never trained on → garbage outputs.
    env = build_env(env_name, num_envs=1, render_mode=None, weights=None)

    if device is None:
        device = next(policy.parameters()).device

    episode_returns: list[float] = []

    for ep in range(n_episodes):
        # Reset environment — seed each episode for reproducibility
        td = env.reset(seed=seed + ep)
        episode_return = 0.0
        done = False

        while not done:
            # Policy forward pass (deterministic — no exploration)
            with torch.no_grad():
                td_device = td.to(device)
                td_out = policy(td_device)

            # Step environment
            td = env.step(td_out.to(env.device))

            # Accumulate raw scalar reward (unweighted — no weights passed)
            reward = td["next", "reward"]
            if reward.numel() == 1:
                episode_return += reward.item()
            else:
                episode_return += reward.sum().item()

            # Check termination
            done_flag = td["next", "done"]
            truncated_flag = td["next"].get("truncated", done_flag.new_zeros(done_flag.shape))
            done = bool((done_flag | truncated_flag).any())

            # Advance: next obs becomes current obs
            td = td["next"]

        episode_returns.append(episode_return)

    env.close()

    returns_arr = np.array(episode_returns, dtype=np.float64)
    return {
        "eval/mean_reward": float(returns_arr.mean()),
        "eval/std_reward": float(returns_arr.std()),
    }


# ---------------------------------------------------------------------------
# GRPO Advantage Computation
# ---------------------------------------------------------------------------


def compute_grpo_advantages(batch, group_size: int):
    """Compute GRPO advantages via trajectory-level group normalization.

    Segments trajectories per environment, computes total returns per
    trajectory, groups them sequentially, and normalizes within each group.

    Args:
        batch: TensorDict from collector with shape [N, T].
        group_size: Number of trajectories per normalization group.

    Returns:
        The batch with ``"advantage"`` key set.
    """
    rewards = batch["next", "reward_vec"]  # [N, T, D]
    dones = batch["next", "done"]
    truncated = batch["next"].get("truncated", torch.zeros_like(dones))

    dones = dones | truncated

    # Shape normalization
    if rewards.ndim == 2:
        rewards = rewards.unsqueeze(-1)
    if dones.ndim == 3:
        dones = dones.squeeze(-1)

    N, T, D = rewards.shape
    device = rewards.device

    # --- Trajectory segmentation and returns ---
    traj_returns = []
    traj_ids_all = torch.empty((N, T), dtype=torch.long, device=device)
    offset = 0

    for n in range(N):
        r = rewards[n]  # [T, D]
        d = dones[n].to(torch.int64)  # [T]

        traj_ids = torch.cumsum(d, dim=0) - 1
        traj_ids = traj_ids.clamp(min=0)
        num_traj = int(traj_ids.max().item()) + 1

        returns = torch.zeros((num_traj, D), device=device)
        returns.index_add_(0, traj_ids, r)
        traj_returns.append(returns)

        traj_ids_all[n] = traj_ids + offset
        offset += num_traj

    traj_returns = torch.cat(traj_returns, dim=0)  # [num_traj_total, D]
    scalar_returns = traj_returns.sum(dim=1)  # [num_traj_total]
    num_traj_total = scalar_returns.shape[0]

    # --- Sequential grouping and normalization ---
    num_groups = num_traj_total // group_size
    if num_groups == 0:
        batch.set("advantage", torch.zeros((N, T, 1), device=device))
        return batch

    valid_traj = num_groups * group_size
    grouped = scalar_returns[:valid_traj].view(num_groups, group_size)

    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True) + 1e-8
    adv_per_traj = ((grouped - mean) / std).view(-1)  # [valid_traj]

    # --- Map advantages back to timesteps ---
    adv = torch.zeros((N, T), device=device)
    valid_mask = traj_ids_all < valid_traj
    adv[valid_mask] = adv_per_traj[traj_ids_all[valid_mask]]
    adv[~valid_mask] = 0.0

    batch.set("advantage", adv.unsqueeze(-1))
    return batch


# ---------------------------------------------------------------------------
# Training Infrastructure
# ---------------------------------------------------------------------------


def build_training(env, policy, value, cfg, device):
    """Create loss module, advantage estimator, optimizer, and data collector.

    Args:
        env: TorchRL environment (single or parallel).
        policy: Actor network (ProbabilisticActor).
        value: Critic network (ValueOperator).
        cfg: TorchRLConfig.
        device: Torch device.

    Returns:
        (loss_module, advantage_module, optimizer, collector) tuple.
        ``advantage_module`` is None for GRPO.
    """
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    try:
        from torchrl.collectors import Collector
    except ImportError:
        from torchrl.collectors import SyncDataCollector as Collector

    if cfg.algo == "ppo":
        advantage_module = GAE(
            gamma=cfg.gamma,
            lmbda=cfg.gae_lambda,
            value_network=value,
        )
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=value,
            clip_epsilon=cfg.clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=cfg.entropy_coeff,
            normalize_advantage=cfg.normalize_advantage,
        )
    elif cfg.algo == "grpo":
        advantage_module = None
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=None,
            clip_epsilon=cfg.clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=cfg.entropy_coeff,
            normalize_advantage=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algo}. Must be 'ppo' or 'grpo'.")

    loss_module = loss_module.to(device)
    adam_eps = getattr(cfg, "adam_eps", 1e-5)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr, eps=adam_eps)

    collector = Collector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
    )

    return loss_module, advantage_module, optimizer, collector


def train_loop(
    collector,
    loss_module,
    advantage_module,
    optimizer,
    cfg,
    policy,
    value,
    save_fn: Callable | None = None,
    advantage_fn: Callable | None = None,
    log_fn: Callable | None = None,
    auxiliary_loss_fn: Callable | None = None,
):
    """Main training loop: collect → compute advantages → PPO update.

    Args:
        collector: TorchRL data collector.
        loss_module: ClipPPOLoss module.
        advantage_module: GAE module (None for GRPO).
        optimizer: Torch optimizer.
        cfg: TorchRLConfig.
        policy: Actor network (for checkpointing).
        value: Critic network (for checkpointing).
        save_fn: Optional checkpoint callback ``(policy, value, optimizer, step)``.
        advantage_fn: Optional custom advantage function ``(batch) -> batch``.
            If provided, overrides the default GAE/GRPO logic.
        log_fn: Optional metrics callback ``(metrics_dict, step)`` for external
            logging (e.g. MLflow).
        auxiliary_loss_fn: Optional ``(minibatch) -> loss_tensor`` added to the
            PPO loss each minibatch step.  Used by multi-head PPO for
            per-component value supervision.

    Returns:
        Total number of frames collected.
    """
    total_frames = 0
    t0 = time.time()

    # Entropy schedule
    ent_schedule = cfg.entropy_coeff_schedule  # (start, end) or None
    ent_schedule_type = cfg.entropy_coeff_schedule_type

    # Log sampling: cap at ~200 data points to avoid bloating MLflow/memory
    est_total_iters = max(1, cfg.total_frames // cfg.frames_per_batch)
    log_every = max(1, est_total_iters // 200)

    for i, batch in enumerate(collector):
        batch_frames = batch.numel()
        total_frames += batch_frames

        # LR annealing
        if getattr(cfg, "lr_anneal", False):
            progress = min(total_frames / cfg.total_frames, 1.0)
            new_lr = cfg.lr * (1.0 - progress)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

        # Update entropy coefficient if schedule is active
        if ent_schedule is not None:
            progress = min(total_frames / cfg.total_frames, 1.0)
            start, end = ent_schedule
            if ent_schedule_type == "cosine":
                new_coeff = end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))
            else:  # linear
                new_coeff = start + (end - start) * progress
            loss_module.entropy_coeff.fill_(new_coeff)

        # Save raw reward before normalization modifies it
        raw_reward = batch["next", "reward"].sum().item() / max(batch_frames, 1)

        # Compute advantages
        with torch.no_grad():
            if advantage_fn is not None:
                batch = advantage_fn(batch)
            elif cfg.algo == "ppo":
                batch = advantage_module(batch)
            else:
                batch = compute_grpo_advantages(batch, cfg.group_size)

        # Flatten for minibatch updates
        flat_batch = batch.reshape(-1)
        n_samples = flat_batch.shape[0]

        # PPO update with random minibatches
        loss_val = torch.tensor(0.0)
        _last_grad_norm = 0.0
        _last_loss_vals = None  # TensorDict or None
        for _epoch in range(cfg.num_epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, cfg.minibatch_size):
                end = min(start + cfg.minibatch_size, n_samples)
                mb = flat_batch[indices[start:end]]

                loss_vals = loss_module(mb)
                loss_val = loss_vals["loss_objective"]
                if "loss_entropy" in loss_vals:
                    loss_val = loss_val + loss_vals["loss_entropy"]
                if cfg.algo == "ppo" and "loss_critic" in loss_vals:
                    loss_val = loss_val + loss_vals["loss_critic"]
                if auxiliary_loss_fn is not None:
                    loss_val = loss_val + auxiliary_loss_fn(mb)

                # NaN guard: skip update if loss is NaN to prevent
                # weight corruption (common with MuJoCo + z-norm)
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    continue

                optimizer.zero_grad()
                loss_val.backward()
                # Clip gradients for ALL parameters in the optimizer,
                # not just loss_module — this covers auxiliary parameters
                # (e.g. multi-head value heads) added via param groups
                all_params = []
                for pg in optimizer.param_groups:
                    all_params.extend(pg["params"])
                # Capture the pre-clip grad norm for diagnostics.
                # clip_grad_norm_ returns the total L2 norm computed before clipping.
                _grad_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
                _last_grad_norm = float(_grad_norm.item()) if hasattr(_grad_norm, "item") else float(_grad_norm)
                _last_loss_vals = loss_vals
                optimizer.step()

        # Logging
        elapsed = time.time() - t0
        fps = total_frames / elapsed if elapsed > 0 else 0
        reward = raw_reward

        print(
            f"  Iter {i:4d} | "
            f"frames {total_frames:>8,} | "
            f"reward {reward:>8.2f} | "
            f"loss {loss_val.item():>8.4f} | "
            f"fps {fps:>6.0f}"
        )

        # External logging (MLflow, etc.) — sampled every log_every iterations
        if log_fn is not None and i % log_every == 0:
            metrics = {
                "rollout/reward_mean": reward,
                "train/loss": loss_val.item(),
                "train/fps": fps,
                "train/entropy_coeff": loss_module.entropy_coeff.item()
                if hasattr(loss_module.entropy_coeff, "item")
                else float(loss_module.entropy_coeff),
            }
            # Capture σ and μ of the
            # scalar reward signal that enters GAE *after* normalization.  For
            # PCZ-PPO-running this is the weighted sum of per-component
            # z-scores; for PPO-znorm it is the aggregate-z-scored scalar; for
            # raw PPO it is the unnormalized weighted scalar.  Comparing this
            # series across algorithms at the same seed+env answers: does
            # per-component z-norm amplify or attenuate the noise floor
            # relative to aggregate z-norm?  See paper §6.
            try:
                r_postnorm = batch["next", "reward"]
                metrics["rollout/reward_postnorm_std"] = r_postnorm.std().item()
                metrics["rollout/reward_postnorm_mean"] = r_postnorm.mean().item()
            except Exception:
                pass
            # Log per-component reward stats if reward_vec available
            if "reward_vec" in batch or ("next" in batch and "reward_vec" in batch["next"]):
                try:
                    rvec = batch["next", "reward_vec"]
                    for ci, cname in enumerate(cfg.component_names):
                        comp = rvec[..., ci]
                        metrics[f"reward_components/{cname}_mean"] = comp.mean().item()
                        metrics[f"reward_components/{cname}_std"] = comp.std().item()
                except Exception:
                    pass
            # Grad norm + approx KL from the final minibatch.
            # _last_loss_vals is a TensorDict. `bool(td)` raises; `key in td` is safe.
            metrics["train/grad_norm"] = _last_grad_norm
            if _last_loss_vals is not None:
                for k_key in ("approx_kl", "kl", "entropy"):
                    try:
                        if k_key in _last_loss_vals:
                            v = _last_loss_vals[k_key]
                            metrics[f"train/{k_key}"] = float(v.item()) if hasattr(v, "item") else float(v)
                    except Exception:
                        pass
            log_fn(metrics, total_frames)

        # Checkpoint
        if save_fn is not None:
            save_fn(policy, value, optimizer, total_frames)

    return total_frames
