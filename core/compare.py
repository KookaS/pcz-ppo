"""compare.py: Run multiple PCZ-PPO algorithms for side-by-side comparison.

Runs each algorithm sequentially, all logged to the same MLflow experiment
for side-by-side comparison.  Each algorithm becomes a separate MLflow run.

Usage::

    # Compare 3 algorithms on CartPole
    python -m core.compare \\
        --algorithms ppo pcz-ppo ppo-znorm \\
        --total-timesteps=50000 --n-envs=2 \\
        --mlflow-tracking-uri http://127.0.0.1:5050

    # Full comparison with seeds
    python -m core.compare \\
        --algorithms ppo pcz-ppo grpo-pcz ppo-znorm \\
        --seeds 42 43 44 \\
        --total-timesteps=100000 \\
        --mlflow-tracking-uri http://127.0.0.1:5050 \\
        --mlflow-experiment-name=norm-baselines
"""

import argparse
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from itertools import product

# Enforce per-process memory limit to prevent OOM crashes.
# Caps at 10GB, raises MemoryError instead of OOM kill.
# Override with MEMORY_LIMIT_GB env var.
_max = int(os.environ.get("MEMORY_LIMIT_GB", "10")) * 1024**3
try:
    resource.setrlimit(resource.RLIMIT_AS, (_max, resource.RLIM_INFINITY))
except (ValueError, OSError):
    pass

from . import ALGORITHM_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple PCZ-PPO algorithms. All runs are logged to the same MLflow experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Grid dimensions
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["ppo", "pcz-ppo"],
        choices=sorted(ALGORITHM_REGISTRY.keys()),
        help="Algorithms to compare.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for each run (default: 42). Each (algorithm, seed) pair becomes a separate run.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name. Default: 'compare_YYYYMMDD_HHMMSS'.",
    )
    # Parse known args; rest passed through to train.py
    args, passthrough = parser.parse_known_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.mlflow_experiment_name or f"compare_{timestamp}"

    # Build grid: (algorithm, seed) pairs
    grid = list(product(args.algorithms, args.seeds))

    print("=" * 60)
    print(f"Comparison: {len(grid)} run(s)")
    print(f"Algorithms: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    print(f"MLflow experiment: {experiment_name}")
    print("=" * 60)
    for i, (algo, seed) in enumerate(grid, 1):
        print(f"  {i}. {algo} (seed={seed})")
    print()

    # Filter out args we control per-variant from the passthrough
    filtered = []
    skip_next = False
    for arg in passthrough:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith(
            (
                "--algorithm=",
                "--seed=",
                "--mlflow-run-name=",
                "--mlflow-experiment-name=",
            )
        ):
            continue
        if arg in (
            "--algorithm",
            "--seed",
            "--mlflow-run-name",
            "--mlflow-experiment-name",
        ):
            skip_next = True
            continue
        filtered.append(arg)
    passthrough = filtered

    # Run each variant
    results: list[tuple[str, int, int, float]] = []
    for i, (algo, seed) in enumerate(grid, 1):
        run_name = f"{algo}_s{seed}"
        cmd = [
            sys.executable,
            "-m",
            "core.train",
            f"--algorithm={algo}",
            f"--seed={seed}",
            f"--mlflow-run-name={run_name}",
            f"--mlflow-experiment-name={experiment_name}",
            *passthrough,
        ]

        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(grid)}] {run_name}")
        print(f"{'─' * 60}")

        start = time.time()
        # Ensure src/ (parent of the core package) is on PYTHONPATH so
        # subprocess can resolve `python -m core.train`.
        env = os.environ.copy()
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(cmd, cwd=src_dir, env=env)
        elapsed = time.time() - start
        results.append((algo, seed, result.returncode, elapsed))

        status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
        print(f"\n  {run_name}: {status} ({elapsed:.0f}s)")

    # Summary
    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")
    print(f"{'Run':<30} {'Status':<10} {'Time':>8}")
    print(f"{'─' * 30} {'─' * 10} {'─' * 8}")
    for algo, seed, rc, elapsed in results:
        name = f"{algo}_s{seed}"
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{name:<30} {status:<10} {elapsed:>7.0f}s")

    print(f"\nMLflow experiment: {experiment_name}")

    if any(rc != 0 for _, _, rc, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
