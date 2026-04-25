"""torchrl/ppo_multihead.py: TorchRL PPO-MultiHead — separate value heads per component.

Each reward component gets its own value head V^(k)(s) sharing the MLP
backbone. The combined value V(s) = sum_k w_k * V^(k)(s) is used for GAE.
This lets each head specialise in predicting one component's value,
implicitly handling scale differences.

Per-component heads are trained with an auxiliary MSE loss on per-component
discounted returns, computed alongside the standard PPO critic loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import TorchRLAlgorithm


class MultiHeadValueModule(nn.Module):
    """Value network that uses per-component heads sharing a backbone.

    Forward pass computes V(s) = sum_k w_k * V^(k)(s).
    Also provides access to per-component values for auxiliary training.
    """

    def __init__(self, backbone: nn.Module, heads: nn.ModuleList, weights: list[float]):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self._weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute aggregate value: sum_k w_k * V^(k)(s)."""
        features = self.backbone(x)
        comp_vals = torch.cat([h(features) for h in self.heads], dim=-1)
        w = self._weights.to(comp_vals.device)
        return (comp_vals * w).sum(dim=-1, keepdim=True)

    def component_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-component values: [batch, K]."""
        features = self.backbone(x)
        return torch.cat([h(features) for h in self.heads], dim=-1)


class TorchRLPPOMultiHead(TorchRLAlgorithm):
    """TorchRL PPO with separate value heads per reward component.

    Each component head V^(k)(s) shares the value MLP hidden layers.
    The aggregate value V(s) = sum_k w_k * V^(k)(s) replaces the
    standard single-head value for GAE and policy training.

    Per-component heads receive auxiliary supervision: MSE loss against
    per-component discounted returns computed from the reward_vec.
    """

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, component_value_coeff: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_value_coeff = component_value_coeff
        self._multihead_module = None

    def _setup(self):
        """Override setup to replace single value head with multi-head."""
        super()._setup()

        hidden = self.config.hidden_size
        D = len(self._component_names)
        device = next(self.value.parameters()).device

        backbone = self.value.module[0]

        heads = nn.ModuleList()
        for _ in range(D):
            head = nn.Linear(hidden, 1)
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)
            heads.append(head)
        heads = heads.to(device)

        self._multihead_module = MultiHeadValueModule(backbone, heads, self._component_weights).to(device)

        self.value.module = self._multihead_module

    def _compute_advantages(self, batch, advantage_module):
        """Compute advantages and per-component return targets."""
        # Standard GAE with multi-head aggregate value
        batch = advantage_module(batch)

        # Compute per-component discounted returns for auxiliary loss
        if "reward_vec" in batch["next"]:
            reward_vec = batch["next", "reward_vec"]  # [..., D]
            D = reward_vec.shape[-1]
            gamma = self.config.gamma

            # Compute per-component discounted returns via backward pass
            # Shape: same as reward_vec
            shape = reward_vec.shape
            comp_returns = torch.zeros_like(reward_vec)

            # Get next-state component values for bootstrap
            with torch.no_grad():
                next_obs = batch["next", "observation"]
                # Reshape for component_values
                flat_next = next_obs.reshape(-1, next_obs.shape[-1])
                comp_v_next = self._multihead_module.component_values(flat_next)
                comp_v_next = comp_v_next.reshape(*shape[:-1], D)

            # Backward pass: compute discounted returns per component
            # For the last timestep, bootstrap with V^(k)(s_next)
            # Squeeze done to remove trailing dim, then unsqueeze for broadcast
            dones_raw = batch["next", "done"].float()
            if dones_raw.shape[-1] == 1:
                dones_raw = dones_raw.squeeze(-1)  # [N, T] or [T]

            if reward_vec.dim() == 3:
                # [N, T, D] — N envs, T timesteps
                T = shape[1]
                not_done = (1 - dones_raw).unsqueeze(-1)  # [N, T, 1] for broadcast
                comp_returns[:, -1, :] = reward_vec[:, -1, :] + gamma * not_done[:, -1, :] * comp_v_next[:, -1, :]
                for t in range(T - 2, -1, -1):
                    comp_returns[:, t, :] = reward_vec[:, t, :] + gamma * not_done[:, t, :] * comp_returns[:, t + 1, :]
            else:
                # [T, D] — single env
                T = shape[0]
                not_done = (1 - dones_raw).unsqueeze(-1)  # [T, 1]
                comp_returns[-1, :] = reward_vec[-1, :] + gamma * not_done[-1, :] * comp_v_next[-1, :]
                for t in range(T - 2, -1, -1):
                    comp_returns[t, :] = reward_vec[t, :] + gamma * not_done[t, :] * comp_returns[t + 1, :]

            # Store targets for auxiliary loss
            batch["_comp_return_targets"] = comp_returns.detach()

        return batch

    def _make_auxiliary_loss_fn(self):
        """Create auxiliary loss function for per-component value heads.

        Normalizes targets per-component to prevent large-scale returns
        (e.g. landing ±100) from dominating the PPO objective loss.
        """
        multihead = self._multihead_module
        coeff = self._component_value_coeff

        def auxiliary_loss(mb):
            if "_comp_return_targets" not in mb:
                return torch.tensor(0.0, device=mb.device)

            obs = mb["observation"]
            targets = mb["_comp_return_targets"]  # [batch, D]

            # Normalize targets per-component to prevent scale domination
            t_mean = targets.mean(dim=0, keepdim=True)
            t_std = targets.std(dim=0, keepdim=True).clamp(min=1e-8)
            targets_norm = (targets - t_mean) / t_std

            # Compute per-component value predictions (with gradients)
            comp_v = multihead.component_values(obs)  # [batch, D]

            # Normalize predictions with same stats (targets are the reference)
            comp_v_norm = (comp_v - t_mean) / t_std

            # MSE loss on normalized values
            loss = F.mse_loss(comp_v_norm, targets_norm)
            return coeff * loss

        return auxiliary_loss

    def learn(self, total_frames=None, render=False, log_fn=None):
        """Override learn to add per-component auxiliary loss."""
        import subprocess
        import sys
        import time

        from ....torchrl.checkpoint import save_checkpoint
        from ....torchrl.training import build_training, train_loop

        if not self._is_setup:
            self._setup()

        if total_frames is not None:
            self.config.total_frames = total_frames

        device = torch.device(self.config.device)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        loss_module, advantage_module, optimizer, collector = build_training(
            self.env, self.policy, self.value, self.config, device
        )
        self._loss_module = loss_module
        self._optimizer = optimizer

        # Ensure component heads are in the optimizer
        head_params = list(self._multihead_module.heads.parameters())
        existing_params = set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                existing_params.add(id(p))
        new_params = [p for p in head_params if id(p) not in existing_params]
        if new_params:
            optimizer.add_param_group({"params": new_params})

        def _save_fn(pol, val, opt, step):
            save_checkpoint(pol, val, opt, step, self.config.checkpoint_dir)

        def _adv_fn(batch):
            return self._compute_advantages(batch, advantage_module)

        # Create auxiliary loss for per-component supervision
        aux_loss_fn = self._make_auxiliary_loss_fn()

        algo_name = type(self).__name__
        D = len(self._component_names)
        print(f"\n{'=' * 60}")
        print(f"  {algo_name} (TorchRL PPO Multi-Head)")
        print(f"  Environment: {self.config.env_name} ({self.config.env_id})")
        print(f"  Components:  {self._component_names} (K={D})")
        print(f"  Value heads: {D} per-component + aggregate")
        print(f"  Aux loss:    MSE per-component (coeff={self._component_value_coeff})")
        if any(w != 1.0 for w in self._component_weights):
            weight_str = ", ".join(f"{n}={w:.1f}" for n, w in zip(self._component_names, self._component_weights))
            print(f"  Weights:     {weight_str}")
        print(f"  Self-norm:   {self.is_self_normalizing}")
        print(f"  Parallel:    {self.config.num_envs} envs")
        print(f"  Total frames: {self.config.total_frames:,}")
        print(f"  Device:      {device}")
        print(f"{'=' * 60}\n")

        render_proc = None
        if render:
            render_cmd = [
                sys.executable,
                "-m",
                "core.torchrl.render",
                f"--env={self.config.env_name}",
                f"--checkpoint-dir={self.config.checkpoint_dir}",
                f"--hidden-size={self.config.hidden_size}",
            ]
            render_proc = subprocess.Popen(render_cmd)

        try:
            t0 = time.time()
            total = train_loop(
                collector,
                loss_module,
                advantage_module,
                optimizer,
                self.config,
                self.policy,
                self.value,
                _save_fn,
                advantage_fn=_adv_fn,
                log_fn=log_fn,
                auxiliary_loss_fn=aux_loss_fn,
            )
            elapsed = time.time() - t0
            print(f"\nTraining complete in {elapsed:.1f}s")
            print(f"Total frames: {total:,}")
            print(f"Avg FPS: {total / elapsed:.0f}")
            return total

        except KeyboardInterrupt:
            print("\nTraining interrupted.")
            _save_fn(self.policy, self.value, optimizer, -1)
            return 0

        finally:
            collector.shutdown()
            try:
                self.env.close()
            except RuntimeError:
                pass
            if render_proc is not None:
                render_proc.terminate()
                try:
                    render_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    render_proc.kill()
                    render_proc.wait(timeout=2)


if __name__ == "__main__":
    model = TorchRLPPOMultiHead(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
