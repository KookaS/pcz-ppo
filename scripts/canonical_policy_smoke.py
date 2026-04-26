"""Canonical-policy pre-training gate for RL environments.

Runs a trivial policy (hold_flat / no-op) and an oracle policy (env-specific,
informed) for N episodes each, and asserts that
``oracle_mean - trivial_mean > max(oracle_std, trivial_std)``.

Motivation: training on a degenerate environment where the
trivial policy already sits at optimum wastes compute; worse, both algorithms
converge to the trivial policy and the resulting "tie" gets mis-interpreted
as evidence about algorithm differences rather than evidence about the env.

Currently supports ``trading-k{2,4,6,8}``. Extend by adding a new case to
``_oracle_for`` / ``_trivial_for``.

Usage::

    uv run python scripts/canonical_policy_smoke.py --env trading-k4 --episodes 30
    # Exits 0 if learnable, 1 if degenerate (with a clear message).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import fmean, stdev

import numpy as np

# Make `core` importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _trading_oracle(obs: np.ndarray, threshold: float = 0.5) -> int:
    """Threshold policy on z-score feature (index 0 by env convention)."""
    z = float(obs[0])
    if z < -threshold:
        return 2  # long
    if z > threshold:
        return 0  # short
    return 1  # flat


def _trivial_hold_flat(_obs) -> int:
    return 1  # Discrete(3) middle action = hold flat


def _run_policy(env, policy, episodes: int) -> tuple[float, float]:
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        done = trunc = False
        while not (done or trunc):
            a = policy(obs)
            obs, r, done, trunc, _ = env.step(a)
            total += r
        rewards.append(total)
    return fmean(rewards), stdev(rewards) if len(rewards) > 1 else 0.0


def _build_env(name: str):
    from core.env_config import ENV_REGISTRY, make_env_factory

    if name not in ENV_REGISTRY:
        raise SystemExit(f"env '{name}' not in ENV_REGISTRY. Registered: {sorted(ENV_REGISTRY)}")
    return make_env_factory(name)()


def _policies_for(name: str):
    if name.startswith("trading-k"):
        return _trivial_hold_flat, _trading_oracle
    raise SystemExit(f"canonical policies not yet defined for env '{name}'; extend scripts/canonical_policy_smoke.py")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()

    env = _build_env(args.env)
    trivial_policy, oracle_policy = _policies_for(args.env)

    tm, ts = _run_policy(env, trivial_policy, args.episodes)
    om, os_ = _run_policy(env, oracle_policy, args.episodes)

    gap = om - tm
    bar = max(os_, ts)
    print(f"{args.env} ({args.episodes} eps):")
    print(f"  trivial_policy: {tm:+.3f} ± {ts:.3f}")
    print(f"  oracle_policy:  {om:+.3f} ± {os_:.3f}")
    print(f"  gap = {gap:+.3f}  threshold (max std) = {bar:.3f}")

    if gap <= bar:
        print("DEGENERATE — trivial and oracle are within noise. Do not train here.")
        return 1
    print("LEARNABLE — oracle clears trivial by more than one std. OK to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
