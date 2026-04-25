"""Shared normalization helpers for TorchRL algorithms.

All functions operate on PyTorch tensors (not numpy arrays).
"""

from __future__ import annotations

import torch


def znorm(x: torch.Tensor, min_std: float = 1e-8) -> torch.Tensor:
    """Z-normalize tensor globally: (x - mean) / std."""
    mean = x.mean()
    std = x.std()
    if std > min_std:
        return (x - mean) / std
    return x - mean


def znorm_per_env(x: torch.Tensor, dim: int = 0, min_std: float = 1e-8) -> torch.Tensor:
    """Z-normalize per environment (along first dimension)."""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return torch.where(std > min_std, (x - mean) / std, x - mean)


def minmax_per_env(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """Min-max scale to [0, 1] per environment."""
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    denom = x_max - x_min
    return torch.where(denom > eps, (x - x_min) / denom, torch.full_like(x, 0.5))


def log_compress(x: torch.Tensor) -> torch.Tensor:
    """Log compression: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def clip_norm(x: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """Clip values to [-clip, clip]."""
    return torch.clamp(x, -clip, clip)


def weighted_sum(
    components: torch.Tensor,
    weights: list[float],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute weighted sum of components along last dimension.

    Args:
        components: [..., D] tensor of per-component values.
        weights: list of D weights.
        device: target device for weights tensor.

    Returns:
        [..., 1] tensor of weighted sums.
    """
    w = torch.tensor(weights, device=device or components.device, dtype=components.dtype)
    return (components * w).sum(dim=-1, keepdim=True)
