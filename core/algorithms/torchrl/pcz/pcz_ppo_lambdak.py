"""torchrl/pcz_ppo_lambdak.py: PCZ-PPO-LambdaK — per-component GAE lambda.

Runs GAE K times (once per component) with a component-specific lambda_k,
then sums weighted per-component advantages:  A(s,a) = Σ w_k · A_k(s,a).

Rationale: sparse/terminal components (e.g. landing) benefit from long
credit propagation (λ≈0.99); dense per-step components (e.g. fuel) benefit
from short windows (λ≈0.90). The critic V(s) is shared across components —
per-component λ affects only the temporal integration of per-component TD
error.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


class TorchRLPCZPPOLambdaK(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with per-component GAE lambda_k."""

    is_self_normalizing = True
    _algo_type = "ppo"

    _lambda_defaults = {
        "lunarlander": [0.99, 0.95, 0.90, 0.90],
        "lunarlander-k2": [0.99, 0.90],
        "lunarlander-k8": [0.99, 0.97, 0.95, 0.95, 0.93, 0.93, 0.90, 0.90],
        "bipedalwalker": [0.95, 0.90, 0.99],
        "halfcheetah": [0.95, 0.90],
    }

    def __init__(self, *args, ema_alpha: float = 0.01, lambda_per_component=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._running_mean = None
        self._running_var = None
        self._user_lambda = lambda_per_component

    def _resolve_lambdas(self, D, device, default_lmbda):
        if self._user_lambda is not None:
            lmbdas = list(self._user_lambda)
        else:
            lmbdas = self._lambda_defaults.get(self.config.env_name)
            if lmbdas is None:
                lmbdas = [default_lmbda] * D
        if len(lmbdas) != D:
            raise ValueError(f"lambda_per_component len {len(lmbdas)} != D={D}")
        return lmbdas

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]
        device = reward_vec.device

        if self._running_mean is None:
            self._running_mean = torch.zeros(D, device=device)
            self._running_var = torch.ones(D, device=device)

        flat = reward_vec.reshape(-1, D)
        a = self._ema_alpha
        self._running_mean = (1 - a) * self._running_mean + a * flat.mean(dim=0)
        self._running_var = (1 - a) * self._running_var + a * flat.var(dim=0)

        var_floor = self.config.variance_floor
        safe_std = torch.sqrt(torch.clamp(self._running_var, min=var_floor))
        normalized = (reward_vec - self._running_mean) / safe_std
        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        default_lmbda = (
            float(advantage_module.lmbda.item())
            if torch.is_tensor(advantage_module.lmbda)
            else float(advantage_module.lmbda)
        )
        lmbdas = self._resolve_lambdas(D, device, default_lmbda)
        original_lmbda = (
            advantage_module.lmbda.detach().clone()
            if torch.is_tensor(advantage_module.lmbda)
            else advantage_module.lmbda
        )

        total_adv = None
        total_vt = None
        try:
            for k in range(D):
                w_k = self._component_weights[k]
                batch["next", "reward"] = (normalized[..., k] * w_k).unsqueeze(-1)
                advantage_module.lmbda.fill_(lmbdas[k]) if torch.is_tensor(advantage_module.lmbda) else setattr(
                    advantage_module, "lmbda", lmbdas[k]
                )
                out_k = advantage_module(batch.clone(False))
                adv_k = out_k["advantage"]
                vt_k = out_k["value_target"]
                if total_adv is None:
                    total_adv = adv_k.clone()
                    total_vt = vt_k.clone() * (w_k / sum(self._component_weights))
                else:
                    total_adv = total_adv + adv_k
                    total_vt = total_vt + vt_k * (w_k / sum(self._component_weights))
        finally:
            if torch.is_tensor(advantage_module.lmbda):
                advantage_module.lmbda.copy_(original_lmbda)
            else:
                advantage_module.lmbda = original_lmbda

        batch["advantage"] = total_adv
        batch["value_target"] = total_vt
        # Set a scalar reward field for downstream logging consistency
        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return batch


if __name__ == "__main__":
    model = TorchRLPCZPPOLambdaK(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=20_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
