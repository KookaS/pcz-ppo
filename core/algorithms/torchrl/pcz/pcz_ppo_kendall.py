"""torchrl/pcz_ppo_kendall.py: PCZ-PPO-Kendall — learnable Kendall uncertainty weighting.

Replaces manual component weights with learned log-variances log σ_k.
Per Kendall, Gal & Cipolla (CVPR 2018), the effective component weight is
1/(2σ_k²) with an additional log σ_k regularizer. Applied here AFTER the
per-component z-normalization (weights act on the already-normalized
signal), so the learnable weights capture noise scale rather than raw
reward scale.

Effective reward: r_t = Σ_k (1 / (2 σ_k²)) · z_k(r_t^{(k)}) + Σ_k log σ_k

We add the regularizer as an auxiliary loss term (via on-policy gradient
on the scalar reward, since PPO doesn't expose a direct loss hook). The
log-variance parameters are registered as an nn.Parameter on the algorithm
and updated with a small learning rate each batch via manual gradient
descent on the per-batch approximate objective: minimise sum of
magnitudes of weighted z-scores (proxy for policy-gradient variance).

Trade-off vs manual weights: removes the 10/5/0.5/0.5 specification
burden. If Kendall-PCZ converges to similar effective weights as the
hand-tuned setting, it's a practical improvement (no tuning). If it
converges elsewhere, we learn which components truly have high noise.
"""

import torch
import torch.nn as nn

from .._base import TorchRLAlgorithm


class TorchRLPCZPPOKendall(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with Kendall uncertainty-weighted component summation."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, uncertainty_lr: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._uncertainty_lr = uncertainty_lr
        self._running_mean = None
        self._running_var = None
        self._log_var = None  # learnable log σ_k² per component

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]
        device = reward_vec.device

        # Lazy init running stats + learnable log σ_k²
        if self._running_mean is None:
            self._running_mean = torch.zeros(D, device=device)
            self._running_var = torch.ones(D, device=device)
            self._log_var = nn.Parameter(torch.zeros(D, device=device))  # init σ²=1

        # EMA running stats (same as PCZ-PPO-Running)
        flat = reward_vec.reshape(-1, D)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0)
        a = self._ema_alpha
        self._running_mean = (1 - a) * self._running_mean + a * batch_mean
        self._running_var = (1 - a) * self._running_var + a * batch_var

        # Per-component z-normalization
        var_floor = self.config.variance_floor
        safe_std = torch.sqrt(torch.clamp(self._running_var, min=var_floor))
        z = (reward_vec - self._running_mean) / safe_std  # shape [..., D]

        # Kendall uncertainty weighting: effective weight = 1/(2 σ_k²)
        precision = torch.exp(-self._log_var)  # = 1/σ_k² (stabler param via log)
        weights = 0.5 * precision  # 1/(2 σ²)

        # Weighted sum: r_t = Σ_k w_k · z_k + regularizer
        # keepdim=True to preserve [..., 1] shape expected by GAE
        weighted = (z * weights).sum(dim=-1, keepdim=True)
        reg = self._log_var.sum().detach()  # scalar regularizer (frozen for reward)

        # Reward = weighted sum minus per-sample-normalized regularizer
        batch["next", "reward"] = weighted - reg / flat.shape[0]

        # Manual update of log_var: minimize weighted z magnitude + regularizer
        # Proxy objective: L(σ) = 0.5 * Σ_k (z_k²/σ_k² + log σ_k²)
        with torch.enable_grad():
            z_mean_sq = (z.reshape(-1, D) ** 2).mean(dim=0).detach()
            loss_sigma = 0.5 * (z_mean_sq * precision + self._log_var).sum()
            grad = torch.autograd.grad(loss_sigma, self._log_var, retain_graph=False)[0]
            with torch.no_grad():
                self._log_var.data -= self._uncertainty_lr * grad
                # Clamp log_var to reasonable range
                self._log_var.data.clamp_(-4.0, 4.0)

        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOKendall(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
