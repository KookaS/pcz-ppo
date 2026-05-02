"""Base class for TorchRL-based algorithms.

Provides a common interface for TorchRL PPO/GRPO variants, handling
environment creation, model building, training loop execution, and
checkpointing.  Subclasses override ``_compute_advantages()`` to implement
custom reward normalization strategies (z-norm, GDPO, etc.).
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable


class TorchRLAlgorithm:
    """Base class for TorchRL-based RL algorithms.

    Provides a common interface compatible with the PCZ project's
    ``ALGORITHM_REGISTRY``.  Not an SB3 class — uses TorchRL's collector
    and loss modules internally.

    Subclasses set ``_algo_type`` ("ppo" or "grpo") and optionally override
    ``_compute_advantages()`` for custom normalization.

    Args:
        env_name: Environment name from ``ENV_REGISTRY``.
        reward_component_names: List of reward component names.
        component_weights: Optional per-component weights.
        **config_kwargs: Forwarded to ``TorchRLConfig``.
    """

    is_torchrl = True
    is_self_normalizing = False
    _algo_type = "ppo"

    def __init__(
        self,
        env_name: str,
        *,
        reward_component_names: list[str],
        component_weights: list[float] | None = None,
        **config_kwargs,
    ):
        from ...torchrl.config import TorchRLConfig

        self._component_names = list(reward_component_names)
        self._component_weights = (
            list(component_weights) if component_weights is not None else [1.0] * len(reward_component_names)
        )

        self.config = TorchRLConfig(
            algo=self._algo_type,
            env_name=env_name,
            reward_dim=len(reward_component_names),
            component_names=self._component_names,
            component_weights=self._component_weights,
            **config_kwargs,
        )

        self.env = None
        self.policy = None
        self.value = None
        self._loss_module = None
        self._optimizer = None
        self._is_setup = False

    def _setup(self):
        """Create environment and models."""
        import torch

        from ...torchrl.env import build_env
        from ...torchrl.models import build_models

        device = torch.device(self.config.device)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        torch.manual_seed(self.config.seed)

        self.env = build_env(
            self.config.env_name,
            num_envs=self.config.num_envs,
            weights=self._component_weights,
        )
        self.policy, self.value = build_models(self.env, self.config, device)
        self._is_setup = True

    def _compute_advantages(self, batch, advantage_module):
        """Compute advantages for the batch.

        Override in subclasses for custom normalization.  The default
        implementation delegates to ``advantage_module`` (GAE for PPO)
        or falls back to GRPO group normalization.

        Args:
            batch: TensorDict from the collector.
            advantage_module: GAE module (None for GRPO variants).

        Returns:
            Modified batch with ``"advantage"`` key set.
        """
        if advantage_module is not None:
            return advantage_module(batch)
        from ...torchrl.training import compute_grpo_advantages

        return compute_grpo_advantages(batch, self.config.group_size)

    def learn(
        self,
        total_frames: int | None = None,
        render: bool = False,
        log_fn: Callable | None = None,
        eval_fn: Callable | None = None,
        eval_every_n_frames: int | None = None,
    ) -> int:
        """Train the agent.

        Args:
            total_frames: Override ``config.total_frames``.
            render: Launch decoupled renderer subprocess.
            log_fn: Optional callback ``(metrics_dict, step)`` for MLflow etc.
            eval_fn: Optional callback ``(step) -> dict`` returning eval
                metrics (``eval/mean_reward``, etc.); merged into the same
                metrics dict that ``log_fn`` receives.
            eval_every_n_frames: Trigger ``eval_fn`` every N training frames.
                ``None`` disables periodic eval (legacy single end-of-training
                eval still happens via the caller).

        Returns:
            Total number of frames collected.
        """
        import torch

        from ...torchrl.checkpoint import save_checkpoint
        from ...torchrl.training import build_training, train_loop

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

        def _save_fn(pol, val, opt, step):
            save_checkpoint(pol, val, opt, step, self.config.checkpoint_dir)

        def _adv_fn(batch):
            return self._compute_advantages(batch, advantage_module)

        # Print setup
        algo_name = type(self).__name__
        print(f"\n{'=' * 60}")
        print(f"  {algo_name} (TorchRL {self._algo_type.upper()})")
        print(f"  Environment: {self.config.env_name} ({self.config.env_id})")
        print(f"  Components:  {self._component_names}")
        if any(w != 1.0 for w in self._component_weights):
            weight_str = ", ".join(f"{n}={w:.1f}" for n, w in zip(self._component_names, self._component_weights))
            print(f"  Weights:     {weight_str}")
        print(f"  Self-norm:   {self.is_self_normalizing}")
        if self.is_self_normalizing and self.config.component_gating:
            print(f"  Gating:      ON (var_floor={self.config.variance_floor})")
        print(f"  Parallel:    {self.config.num_envs} envs")
        print(f"  Total frames: {self.config.total_frames:,}")
        print(f"  Device:      {device}")
        print(f"{'=' * 60}\n")

        # Decoupled renderer
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
            print(f"  Renderer PID: {render_proc.pid}")

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
                eval_fn=eval_fn,
                eval_every_n_frames=eval_every_n_frames,
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

    def save(self, path: str):
        """Save checkpoint to directory."""
        from ...torchrl.checkpoint import save_checkpoint

        save_checkpoint(self.policy, self.value, self._optimizer, -1, path)

    def load(self, path: str):
        """Load policy weights from checkpoint."""
        import os

        from ...torchrl.checkpoint import load_checkpoint

        if not self._is_setup:
            self._setup()
        checkpoint_path = os.path.join(path, "latest.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = path
        load_checkpoint(self.policy, checkpoint_path)
