"""IQM (Interquartile Mean) analysis for PCZ-PPO headline claims.

Uses `rliable` (Agarwal et al., NeurIPS 2021) to compute:
- IQM point estimates with stratified bootstrap 95% CIs
- Performance profiles
- Probability of improvement

Run:
    cd /workspace && uv run python -m core.plot.iqm_analysis

Emits results to stdout; fragments wired via render_claims.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "artifacts" / "pcz-ppo" / "paper"))
from fig_data import RESULTS_CSV, load_results, query

LL_PRIMARY_WEIGHTS = "10.00,5.00,0.50,0.50"
# Canonical K=6 weights prefix (env default, proportional to K=4/K=8).  Without
# this filter one empty-weights seed-42 row silently wins chronologically-latest
# dedupe and contaminates K=6 stats — see render_claims.py ``LL_K6_WEIGHTS``
LL_K6_WEIGHTS = "10.00,3.00"


def compute_iqm_comparison(
    rows: list[dict],
    env: str,
    total_timesteps: int = 500000,
    weights: str | None = None,
) -> dict:
    """Compute IQM + bootstrap CI for PCZ-PPO vs PPO on given env."""
    from rliable import library as rly
    from rliable import metrics

    pcz = query(rows, algorithm="torchrl-pcz-ppo-running", env=env, total_timesteps=total_timesteps, weights=weights)
    ppo = query(rows, algorithm="torchrl-ppo", env=env, total_timesteps=total_timesteps, weights=weights)

    pcz_scores = np.array([float(r["eval_mean"]) for r in pcz["runs"]])
    ppo_scores = np.array([float(r["eval_mean"]) for r in ppo["runs"]])

    if len(pcz_scores) < 3 or len(ppo_scores) < 3:
        return {"pcz_n": len(pcz_scores), "ppo_n": len(ppo_scores), "skip": True}

    # rliable expects shape (n_runs, n_tasks) — we have 1 task
    pcz_arr = pcz_scores.reshape(-1, 1)
    ppo_arr = ppo_scores.reshape(-1, 1)

    algo_dict = {"PCZ-PPO": pcz_arr, "PPO": ppo_arr}

    # IQM with bootstrap CIs
    def iqm_func(x):
        return np.array([metrics.aggregate_iqm(x)])

    iqm_scores, iqm_cis = rly.get_interval_estimates(algo_dict, iqm_func, reps=10000)

    # Probability of improvement (PCZ > PPO)
    # Use all pairwise comparisons
    n_better = sum(1 for p in pcz_scores for q in ppo_scores if p > q)
    n_total = len(pcz_scores) * len(ppo_scores)
    prob_improvement = n_better / n_total

    return {
        "pcz_iqm": float(iqm_scores["PCZ-PPO"][0]),
        "pcz_iqm_ci": (float(iqm_cis["PCZ-PPO"][0][0]), float(iqm_cis["PCZ-PPO"][1][0])),
        "ppo_iqm": float(iqm_scores["PPO"][0]),
        "ppo_iqm_ci": (float(iqm_cis["PPO"][0][0]), float(iqm_cis["PPO"][1][0])),
        "pcz_n": len(pcz_scores),
        "ppo_n": len(ppo_scores),
        "prob_improvement": prob_improvement,
        "skip": False,
    }


def main() -> None:
    rows = load_results()
    print(f"Loaded {len(rows)} runs from {RESULTS_CSV}\n")

    configs = [
        ("lunarlander", 4, "LunarLander K=4", LL_PRIMARY_WEIGHTS),
        ("lunarlander-k6", 6, "LunarLander K=6", LL_K6_WEIGHTS),
        ("lunarlander-k8", 8, "LunarLander K=8", None),
        ("lunarlander-k2", 2, "LunarLander K=2", None),
        ("bipedalwalker", 3, "BipedalWalker K=3", None),
        ("halfcheetah", 2, "HalfCheetah K=2", None),
    ]

    print("=== IQM Analysis (rliable, 10k bootstrap) ===\n")
    for env, _k, label, weights in configs:
        result = compute_iqm_comparison(rows, env, weights=weights)
        if result.get("skip"):
            print(f"{label}: skipped (PCZ n={result['pcz_n']}, PPO n={result['ppo_n']})")
            continue
        print(f"{label} (PCZ n={result['pcz_n']}, PPO n={result['ppo_n']}):")
        print(
            f"  PCZ-PPO IQM: {result['pcz_iqm']:+.1f}  CI [{result['pcz_iqm_ci'][0]:+.1f}, {result['pcz_iqm_ci'][1]:+.1f}]"
        )
        print(
            f"  PPO     IQM: {result['ppo_iqm']:+.1f}  CI [{result['ppo_iqm_ci'][0]:+.1f}, {result['ppo_iqm_ci'][1]:+.1f}]"
        )
        print(f"  P(PCZ > PPO): {result['prob_improvement']:.2%}")
        print()


if __name__ == "__main__":
    main()
