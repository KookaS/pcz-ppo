"""Reward normalization strategies for PPO.

Re-exports algorithm classes, registry, and shared infrastructure
from the ``algorithms`` subpackage for backwards-compatible access::

    from core import ALGORITHM_REGISTRY, PCZPPO
"""

from .algorithms import (
    ALGORITHM_REGISTRY,
    # Competitors
    PCZGRPO,
    # PCZ-PPO variants
    PCZPPO,
    # Infrastructure
    ComponentRolloutBuffer,
    PCZPPOClip,
    PCZPPOGlobal,
    PCZPPOLog,
    PCZPPOMinmax,
    # Hybrids
    PCZPPOPopArt,
    PCZPPORunning,
    PCZPPOVecnorm,
    PCZPPOWeighted,
    PopArtMixin,
    PPOAdvOnly,
    PPOMultiHead,
    PPONoNorm,
    PPOPopArt,
    PPOZnorm,
    PPOZnormPost,
    # Baselines
    StandardPPO,
    _znorm,
)

# TorchRL re-exports (optional — torchrl may not be installed)
try:
    from .algorithms.torchrl import (
        TorchRLAlgorithm,
        TorchRLGRPO,
        TorchRLPCZGRPO,
        TorchRLPCZPPO,
        TorchRLPPO,
    )
except ImportError:
    pass

__all__ = [
    "ALGORITHM_REGISTRY",
    # Baselines
    "StandardPPO",
    "PPONoNorm",
    "PPOAdvOnly",
    "PPOZnorm",
    "PPOZnormPost",
    # PCZ-PPO variants
    "PCZPPO",
    "PCZPPOGlobal",
    "PCZPPORunning",
    "PCZPPOWeighted",
    "PCZPPOVecnorm",
    "PCZPPOClip",
    "PCZPPOMinmax",
    "PCZPPOLog",
    # Hybrids
    "PCZPPOPopArt",
    # Competitors
    "PCZGRPO",
    "PPOPopArt",
    "PPOMultiHead",
    # Infrastructure
    "ComponentRolloutBuffer",
    "PopArtMixin",
    "_znorm",
    # TorchRL variants
    "TorchRLAlgorithm",
    "TorchRLPPO",
    "TorchRLGRPO",
    "TorchRLPCZPPO",
    "TorchRLPCZGRPO",
]
