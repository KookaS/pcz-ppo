"""TorchRL-based algorithm implementations for PCZ-PPO.

Provides 17 TorchRL algorithm variants mirroring the 17 SB3 algorithms.
All are registered in ``ALGORITHM_REGISTRY`` with a ``torchrl-`` prefix.
"""

from ._base import TorchRLAlgorithm
from .baselines.grpo import TorchRLGRPO

# Baselines
from .baselines.ppo import TorchRLPPO
from .baselines.ppo_adv_only import TorchRLPPOAdvOnly
from .baselines.ppo_multihead import TorchRLPPOMultiHead
from .baselines.ppo_no_norm import TorchRLPPONoNorm
from .baselines.ppo_popart import TorchRLPPOPopArt
from .baselines.ppo_weighted_running import TorchRLPPOWeightedRunning
from .baselines.ppo_znorm import TorchRLPPOZnorm
from .baselines.ppo_znorm_post import TorchRLPPOZnormPost
from .baselines.qlearning import TorchRLTabularQLearning
from .pcz.pcz_grpo import TorchRLPCZGRPO

# PCZ-PPO variants
from .pcz.pcz_ppo import TorchRLPCZPPO
from .pcz.pcz_ppo_asym import TorchRLPCZPPOAsym
from .pcz.pcz_ppo_clip import TorchRLPCZPPOClip
from .pcz.pcz_ppo_cosw import TorchRLPCZPPOCosW
from .pcz.pcz_ppo_global import TorchRLPCZPPOGlobal
from .pcz.pcz_ppo_kendall import TorchRLPCZPPOKendall
from .pcz.pcz_ppo_lambdak import TorchRLPCZPPOLambdaK
from .pcz.pcz_ppo_log import TorchRLPCZPPOLog
from .pcz.pcz_ppo_mc import TorchRLPCZPPOMC
from .pcz.pcz_ppo_mi import TorchRLPCZPPOMI
from .pcz.pcz_ppo_minmax import TorchRLPCZPPOMinmax
from .pcz.pcz_ppo_popart import TorchRLPCZPPOPopArt
from .pcz.pcz_ppo_quantile import TorchRLPCZPPOQuantile
from .pcz.pcz_ppo_rlw import TorchRLPCZPPORlw
from .pcz.pcz_ppo_running import TorchRLPCZPPORunning
from .pcz.pcz_ppo_running_no_whiten import TorchRLPCZPPORunningNoWhiten
from .pcz.pcz_ppo_symznorm import TorchRLPCZPPOSymZnorm
from .pcz.pcz_ppo_tcheby import TorchRLPCZPPOTcheby
from .pcz.pcz_ppo_vecnorm import TorchRLPCZPPOVecnorm
from .pcz.pcz_ppo_weighted import TorchRLPCZPPOWeighted
from .pcz.pcz_ppo_zca import TorchRLPCZPPOZca

__all__ = [
    "TorchRLAlgorithm",
    # Core 4
    "TorchRLPPO",
    "TorchRLGRPO",
    "TorchRLPCZPPO",
    "TorchRLPCZGRPO",
    # Baselines
    "TorchRLPPONoNorm",
    "TorchRLPPOAdvOnly",
    "TorchRLPPOZnorm",
    "TorchRLPPOZnormPost",
    "TorchRLPPOWeightedRunning",
    # PCZ-PPO variants
    "TorchRLPCZPPOGlobal",
    "TorchRLPCZPPORunning",
    "TorchRLPCZPPOWeighted",
    "TorchRLPCZPPOVecnorm",
    "TorchRLPCZPPOClip",
    "TorchRLPCZPPOMinmax",
    "TorchRLPCZPPOLog",
    "TorchRLPCZPPOMC",
    "TorchRLPCZPPORunningNoWhiten",
    "TorchRLPCZPPOKendall",
    "TorchRLPCZPPOLambdaK",
    "TorchRLPCZPPOAsym",
    "TorchRLPCZPPOTcheby",
    "TorchRLPCZPPOMI",
    "TorchRLPCZPPOSymZnorm",
    "TorchRLPCZPPOCosW",
    "TorchRLPCZPPORlw",
    "TorchRLPCZPPOZca",
    "TorchRLPCZPPOQuantile",
    # Complex
    "TorchRLPCZPPOPopArt",
    "TorchRLPPOPopArt",
    "TorchRLPPOMultiHead",
    # Tabular
    "TorchRLTabularQLearning",
]

# TorchRL algorithm registry (merged into ALGORITHM_REGISTRY)
TORCHRL_ALGORITHM_REGISTRY = {
    # Core
    "torchrl-ppo": TorchRLPPO,
    "torchrl-grpo": TorchRLGRPO,
    "torchrl-pcz-ppo": TorchRLPCZPPO,
    "torchrl-pcz-grpo": TorchRLPCZGRPO,
    # Baselines
    "torchrl-ppo-no-norm": TorchRLPPONoNorm,
    "torchrl-ppo-adv-only": TorchRLPPOAdvOnly,
    "torchrl-ppo-znorm": TorchRLPPOZnorm,
    "torchrl-ppo-znorm-post": TorchRLPPOZnormPost,
    "torchrl-ppo-weighted-running": TorchRLPPOWeightedRunning,
    # PCZ-PPO variants
    "torchrl-pcz-ppo-global": TorchRLPCZPPOGlobal,
    "torchrl-pcz-ppo-running": TorchRLPCZPPORunning,
    "torchrl-pcz-ppo-weighted": TorchRLPCZPPOWeighted,
    "torchrl-pcz-ppo-vecnorm": TorchRLPCZPPOVecnorm,
    "torchrl-pcz-ppo-clip": TorchRLPCZPPOClip,
    "torchrl-pcz-ppo-minmax": TorchRLPCZPPOMinmax,
    "torchrl-pcz-ppo-log": TorchRLPCZPPOLog,
    "torchrl-pcz-ppo-mc": TorchRLPCZPPOMC,
    "torchrl-pcz-ppo-running-no-whiten": TorchRLPCZPPORunningNoWhiten,
    "torchrl-pcz-ppo-kendall": TorchRLPCZPPOKendall,
    "torchrl-pcz-ppo-lambdak": TorchRLPCZPPOLambdaK,
    "torchrl-pcz-ppo-asym": TorchRLPCZPPOAsym,
    "torchrl-pcz-ppo-tcheby": TorchRLPCZPPOTcheby,
    "torchrl-pcz-ppo-mi": TorchRLPCZPPOMI,
    "torchrl-pcz-ppo-symznorm": TorchRLPCZPPOSymZnorm,
    "torchrl-pcz-ppo-cosw": TorchRLPCZPPOCosW,
    "torchrl-pcz-ppo-rlw": TorchRLPCZPPORlw,
    "torchrl-pcz-ppo-zca": TorchRLPCZPPOZca,
    "torchrl-pcz-ppo-quantile": TorchRLPCZPPOQuantile,
    # Complex
    "torchrl-pcz-ppo-popart": TorchRLPCZPPOPopArt,
    "torchrl-ppo-popart": TorchRLPPOPopArt,
    "torchrl-ppo-multihead": TorchRLPPOMultiHead,
    # Tabular
    "torchrl-qlearning": TorchRLTabularQLearning,
}
