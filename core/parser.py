"""parser.py: CLI argument parser for PCZ-PPO training and evaluation.

Provides ``parse_args()`` for standalone use and ``add_train_args()`` for
embedding into a parent parser (used by ``compare.py``).
"""

import argparse

from . import ALGORITHM_REGISTRY
from .env_config import ENV_REGISTRY, get_env_config

# Algorithms that handle their own reward normalisation (no VecNormalize reward).
# Derived from each algorithm class's ``is_self_normalizing`` attribute.
SELF_NORMALIZING_ALGORITHMS: set[str] = {
    name for name, cls in ALGORITHM_REGISTRY.items() if getattr(cls, "is_self_normalizing", False)
}


def _env_choices_help() -> str:
    """Build help text listing available environments and their components."""
    lines = []
    for name, cfg in sorted(ENV_REGISTRY.items()):
        comps = ", ".join(cfg.reward_components)
        extra = f" (requires: uv sync --extra {cfg.extra})" if cfg.extra else ""
        lines.append(f"  {name}: {cfg.description} [{comps}]{extra}")
    return "\n".join(lines)


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all training-related arguments to *parser*."""

    # ── Algorithm & environment ──────────────────────────────────────
    parser.add_argument(
        "--algorithm",
        type=str,
        default="pcz-ppo",
        choices=sorted(ALGORITHM_REGISTRY.keys()),
        help="RL algorithm variant (default: pcz-ppo).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=sorted(ENV_REGISTRY.keys()),
        help="Environment name (default: cartpole). Available:\n" + _env_choices_help(),
    )

    # ── Visualization ─────────────────────────────────────────────────
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Enable live visualization during training (opens game window).",
    )

    # ── Environment-specific options ─────────────────────────────────
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override max episode steps (TimeLimit). Default: env-specific.",
    )
    parser.add_argument(
        "--mario-rom-mode",
        type=str,
        default="downsample",
        choices=["vanilla", "downsample", "pixel", "rectangle"],
        help="Super Mario Bros ROM mode (default: vanilla). "
        "vanilla=full RGB, downsample=reduced detail, pixel=simplified, "
        "rectangle=geometric. Only used with --env=supermario.",
    )

    # ── Training hyperparameters ─────────────────────────────────────
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50_000,
        help="Total training timesteps (default: 50000).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=2,
        help="Number of parallel environments (default: 2).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Rollout buffer size per env (default: 2048).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO updates (default: 64).",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of PPO epochs per rollout (default: 10).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99).",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda (default: 0.95).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Entropy coefficient. Default: 0.01 for self-normalizing algorithms (PCZ variants), 0.0 for standard PPO.",
    )
    parser.add_argument(
        "--ent-coef-schedule",
        type=str,
        default=None,
        help="Entropy coefficient schedule 'start:end' (e.g. '0.1:0.01'). "
        "Interpolates ent_coef from start to end over training. "
        "Overrides --ent-coef when set. See --ent-coef-schedule-type.",
    )
    parser.add_argument(
        "--ent-coef-schedule-type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Schedule curve type for --ent-coef-schedule. "
        "'linear' decays linearly; 'cosine' uses cosine annealing "
        "(stays higher longer, decays smoothly at end). Default: linear.",
    )
    parser.add_argument(
        "--policy-gain",
        type=float,
        default=None,
        help="Orthogonal init gain for the policy head (TorchRL only). "
        "Default: 0.01. Larger values (0.1, 1.0) may help PCZ variants.",
    )
    parser.add_argument(
        "--znorm-clip",
        type=float,
        default=None,
        help="Clip z-normalized rewards to [-clip, clip] (TorchRL PCZ only). "
        "E.g. 3.0 for [-3, 3] clipping matching SB3 behavior.",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=None,
        help="Adam optimizer epsilon (TorchRL only). Default: 1e-5 (SB3 default).",
    )
    parser.add_argument(
        "--lr-anneal",
        action="store_true",
        default=False,
        help="Enable linear LR annealing to 0 over training (TorchRL only).",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["leaky_relu", "tanh"],
        help="Activation function for hidden layers (TorchRL only). Default: leaky_relu.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden layer size for MLP networks (TorchRL only). Default: 64.",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=None,
        help="Minimum variance for z-norm denominator (TorchRL PCZ only). "
        "Default: 1e-8. Higher values (e.g. 0.01) prevent NaN from near-constant components.",
    )
    parser.add_argument(
        "--component-gating",
        action="store_true",
        default=False,
        help="Zero-out near-constant components in z-normalization (TorchRL PCZ only). "
        "Components with running variance below --variance-floor are excluded from "
        "the normalized reward signal. Fixes envs with constant rewards (e.g. Humanoid alive).",
    )
    parser.add_argument(
        "--reward-component-weights",
        type=str,
        default=None,
        help="Post-normalization weights for reward components (comma-separated "
        "floats, e.g. '5.0,3.0,0.5,0.5'). Overrides env defaults. "
        "Length must match the number of reward components.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device (default: cpu).",
    )

    # ── Evaluation ───────────────────────────────────────────────────
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip training; evaluate a checkpoint only.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.zip) for --eval-only or resume.",
    )
    parser.add_argument(
        "--eval-freq-episodes",
        type=int,
        default=10,
        help="Evaluate every N episodes during training (default: 10).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Episodes per evaluation (default: 10).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip post-training evaluation.",
    )

    parser.add_argument(
        "--no-reward-norm",
        action="store_true",
        default=False,
        help="Disable VecNormalize reward normalization even for non-self-normalizing algorithms (ablation).",
    )

    # ── MLflow ───────────────────────────────────────────────────────
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help="Disable MLflow logging entirely.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (e.g. http://127.0.0.1:5050).",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name. Default: 'pcz-ppo'.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name. Default: '<algorithm>_<timestamp>'.",
    )

    # ── Output ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Base directory for training outputs (default: runs/).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING).",
    )

    # ── Config regression gate (G3) ──────────────────────────────────
    parser.add_argument(
        "--confirm-config-changes",
        action="store_true",
        help=(
            "Acknowledge that TorchRLConfig defaults differ from the committed "
            "config_defaults.json baseline. Without this flag, drift causes "
            "training to refuse to launch (G3)."
        ),
    )

    return parser


def parse_args(argv: list[str] = None) -> argparse.Namespace:
    """Parse CLI arguments for ``train.py``.

    After parsing, resolves the env config and attaches it to the namespace
    as ``env_config`` for use by train.py.
    """
    parser = argparse.ArgumentParser(
        description="PCZ-PPO: Train and evaluate reward normalization algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_train_args(parser)
    args = parser.parse_args(argv)

    # Resolve env config from the --env name
    args.env_config = get_env_config(args.env)
    args.reward_components = args.env_config.reward_components

    return args
