"""Algorithm implementations for PCZ-PPO reward normalization comparison.

All algorithms work with any Gymnasium environment that provides
``info["reward_components"]`` as a dict of component-name → float.
"""

# Baselines
# Shared infrastructure
from ._common import (
    ComponentRolloutBuffer,
    PopArtMixin,
    _init_component_weights,
    _weighted_component_sum,
    _znorm,
)
from .baselines.mpc_lq import LQMPCAgent
from .baselines.ppo import StandardPPO
from .baselines.ppo_adv_only import PPOAdvOnly
from .baselines.ppo_multihead import PPOMultiHead
from .baselines.ppo_no_norm import PPONoNorm
from .baselines.ppo_popart import PPOPopArt
from .baselines.ppo_znorm import PPOZnorm
from .baselines.ppo_znorm_post import PPOZnormPost
from .baselines.qlearning import TabularQLearning
from .baselines.random_agent import RandomAgent
from .baselines.static_weight_agent import StaticWeightAgent
from .pcz.pcz_grpo import PCZGRPO

# PCZ-PPO variants (per-component normalization + GAE)
from .pcz.pcz_ppo import PCZPPO
from .pcz.pcz_ppo_clip import PCZPPOClip
from .pcz.pcz_ppo_global import PCZPPOGlobal
from .pcz.pcz_ppo_log import PCZPPOLog
from .pcz.pcz_ppo_minmax import PCZPPOMinmax
from .pcz.pcz_ppo_popart import PCZPPOPopArt
from .pcz.pcz_ppo_running import PCZPPORunning
from .pcz.pcz_ppo_vecnorm import PCZPPOVecnorm
from .pcz.pcz_ppo_weighted import PCZPPOWeighted

# TorchRL variants (lazy import — torchrl is optional)
try:
    from .torchrl import (
        TORCHRL_ALGORITHM_REGISTRY,
        TorchRLAlgorithm,
        TorchRLGRPO,
        TorchRLPCZGRPO,
        TorchRLPCZPPO,
        TorchRLPPO,
        TorchRLTabularQLearning,
    )

    _TORCHRL_AVAILABLE = True
except ImportError:
    _TORCHRL_AVAILABLE = False
    TORCHRL_ALGORITHM_REGISTRY = {}

__all__ = [
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
    # Tabular
    "TabularQLearning",
    # Planners (classical baselines, no gradient learning)
    "LQMPCAgent",
    "RandomAgent",
    "StaticWeightAgent",
    # Infrastructure
    "ComponentRolloutBuffer",
    "PopArtMixin",
    "_znorm",
    "_init_component_weights",
    "_weighted_component_sum",
    # TorchRL variants
    "TorchRLAlgorithm",
    "TorchRLPPO",
    "TorchRLGRPO",
    "TorchRLPCZPPO",
    "TorchRLPCZGRPO",
    "TorchRLTabularQLearning",
]

# Algorithm registry: maps CLI-style names to classes
ALGORITHM_REGISTRY = {
    "ppo": StandardPPO,
    "ppo-no-norm": PPONoNorm,
    "ppo-adv-only": PPOAdvOnly,
    "ppo-znorm": PPOZnorm,
    "ppo-znorm-post": PPOZnormPost,
    "pcz-ppo": PCZPPO,
    "pcz-ppo-global": PCZPPOGlobal,
    "pcz-ppo-running": PCZPPORunning,
    "pcz-ppo-weighted": PCZPPOWeighted,
    "pcz-ppo-vecnorm": PCZPPOVecnorm,
    "pcz-ppo-clip": PCZPPOClip,
    "pcz-ppo-minmax": PCZPPOMinmax,
    "pcz-ppo-log": PCZPPOLog,
    "pcz-ppo-popart": PCZPPOPopArt,
    "grpo-pcz": PCZGRPO,
    "ppo-popart": PPOPopArt,
    "ppo-multihead": PPOMultiHead,
    "qlearning": TabularQLearning,
    "mpc-lq": LQMPCAgent,
    "random-action": RandomAgent,
    "static-weight": StaticWeightAgent,
}

# Merge TorchRL algorithms into the main registry
ALGORITHM_REGISTRY.update(TORCHRL_ALGORITHM_REGISTRY)
