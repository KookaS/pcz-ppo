"""diagnostics.py: Automated post-training analysis from parquet metrics.

Reads per-run parquet files, computes structured diagnostics, optionally
generates plots, and produces a summary report.  Output goes to a sibling
``diagnostics/`` directory next to ``metrics/`` (gitignored, but visible
in the IDE).  Only the insights extracted from these belong in the journal.

Usage::

    # Diagnose all runs in a metrics directory
    python -m core.plot.diagnostics \
        --metrics-dir artifacts/pcz-ppo/data/metrics

    # Diagnose specific runs
    python -m core.plot.diagnostics \
        --metrics-dir artifacts/pcz-ppo/data/metrics \
        --runs 4c82ae6af43f 5cba0dc849c7

    # With plots (to artifacts/pcz-ppo/data/diagnostics/)
    python -m core.plot.diagnostics \
        --metrics-dir artifacts/pcz-ppo/data/metrics \
        --plots

    # Cross-seed analysis for a specific algorithm+env
    python -m core.plot.diagnostics \
        --metrics-dir artifacts/pcz-ppo/data/metrics \
        --cross-seed --algo torchrl-pcz-ppo-running --env lunarlander
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Spike:
    """A detected anomaly in the reward time-series."""

    step: int
    value: float
    local_mean: float
    z_score: float
    recovered: bool  # Did reward return to pre-spike level within 10 steps?


@dataclass
class RunDiagnostics:
    """Structured diagnostics for a single training run."""

    run_id: str
    algorithm: str
    env: str
    seed: str
    total_steps: int
    n_data_points: int

    # Convergence
    convergence_class: str  # "converged", "learning", "diverging", "collapsed"
    stability_ratio: float  # final 20% std / overall std (<0.5 = converged)
    final_reward: float
    peak_reward: float
    peak_step: int
    peak_to_final_drop: float  # How much reward dropped from peak to end

    # Learning dynamics
    improving_pct: float  # % of step transitions that are upward
    early_mean: float  # First-half mean reward
    late_mean: float  # Second-half mean reward
    improvement: float  # late - early

    # Sample efficiency
    steps_to_50pct: int | None  # Step at which reward first crosses 50% of final
    steps_to_80pct: int | None  # Step at which reward first crosses 80% of final

    # Anomalies
    spikes: list[Spike] = field(default_factory=list)
    n_spikes: int = 0
    worst_spike_z: float = 0.0
    all_spikes_recovered: bool = True

    # Component analysis
    dominant_component: str = ""  # Component with highest final absolute mean
    component_summary: dict = field(default_factory=dict)

    # Trust score (0-1, composite)
    trust_score: float = 0.0
    trust_flags: list[str] = field(default_factory=list)


@dataclass
class CrossSeedDiagnostics:
    """Cross-seed analysis for an algorithm+env combination."""

    algorithm: str
    env: str
    n_seeds: int
    seeds: list[str]

    # Agreement
    final_mean: float
    final_std: float
    cv: float  # coefficient of variation (std/|mean|)

    # Failure detection
    n_failures: int  # Seeds with final reward < 0 or trust_score < 0.5
    failure_seeds: list[str]
    failure_rate: float

    # Trajectory shape agreement
    shape_agreement: float  # Correlation of reward curves across seeds (0-1)

    # Per-seed trust
    per_seed_trust: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def _parse_filename(fname: str) -> dict:
    """Extract run_id, algorithm, env, seed from parquet filename."""
    # Pattern: {run_id}_{algo}_{env}_s{seed}.parquet
    m = re.match(r"^([a-f0-9]+)_(.+?)_([a-z][a-z0-9-]*)_s(\d+)\.parquet$", fname)
    if not m:
        return {"run_id": fname[:12], "algorithm": "", "env": "", "seed": ""}
    return {
        "run_id": m.group(1),
        "algorithm": m.group(2),
        "env": m.group(3),
        "seed": m.group(4),
    }


def _detect_spikes(steps: np.ndarray, values: np.ndarray, window: int = 15, threshold: float = 3.0) -> list[Spike]:
    """Detect reward spikes as >threshold std deviations from rolling mean.

    RL reward curves are inherently noisy.  A 2σ threshold on a 10-step
    window flags normal exploration noise.  We use 3σ on a 15-step window
    so only genuine anomalies (policy collapse, reward hacking) trigger.
    Recovery is checked over the next 20% of window size.
    """
    if len(values) < window + 5:
        return []

    spikes = []
    recovery_horizon = max(5, window // 3)
    # Minimum gap between spikes to avoid double-counting the same event
    min_gap = window // 2

    last_spike_idx = -min_gap - 1
    for i in range(window, len(values)):
        if i - last_spike_idx < min_gap:
            continue
        local = values[i - window : i]
        local_mean = local.mean()
        local_std = local.std()
        if local_std < 1e-8:
            continue
        z = (values[i] - local_mean) / local_std
        if abs(z) > threshold:
            # Check recovery: does reward return to within 1σ of pre-spike mean?
            future = values[i + 1 : i + 1 + recovery_horizon]
            recovered = len(future) > 0 and any(abs(v - local_mean) < 1.5 * local_std for v in future)
            spikes.append(
                Spike(
                    step=int(steps[i]),
                    value=float(values[i]),
                    local_mean=float(local_mean),
                    z_score=float(z),
                    recovered=bool(recovered),
                )
            )
            last_spike_idx = i
    return spikes


def _convergence_class(stability_ratio: float, improving_pct: float, late_mean: float, early_mean: float) -> str:
    """Classify the training run's convergence behavior."""
    if stability_ratio < 0.4 and late_mean > early_mean:
        return "converged"
    if stability_ratio < 0.7 and improving_pct > 45:
        return "learning"
    if late_mean < early_mean * 0.5 and early_mean > 0:
        return "collapsed"
    if stability_ratio > 1.2:
        return "diverging"
    if improving_pct > 45:
        return "learning"
    return "stagnant"


def _steps_to_threshold(steps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    """Find the first step where reward crosses a threshold."""
    for i, v in enumerate(values):
        if v >= threshold:
            return int(steps[i])
    return None


def _compute_trust(diag: RunDiagnostics) -> tuple[float, list[str]]:
    """Compute a 0-1 trust score with explanatory flags."""
    score = 1.0
    flags = []

    # Convergence penalty
    if diag.convergence_class == "collapsed":
        score -= 0.5
        flags.append("COLLAPSED: reward degraded severely")
    elif diag.convergence_class == "diverging":
        score -= 0.4
        flags.append("DIVERGING: increasing instability")
    elif diag.convergence_class == "stagnant":
        score -= 0.2
        flags.append("STAGNANT: no clear learning trend")

    # Spike penalty — only significant with tighter 3σ detection
    unrecovered = [s for s in diag.spikes if not s.recovered]
    if diag.n_spikes > 5:
        score -= 0.15
        flags.append(f"SPIKY: {diag.n_spikes} anomalies at 3\u03c3 threshold")
    elif diag.n_spikes > 3:
        score -= 0.05
        flags.append(f"MODERATE_SPIKES: {diag.n_spikes} anomalies")

    # Unrecovered spikes are the real concern
    if len(unrecovered) >= 2:
        score -= 0.15
        flags.append(f"UNRECOVERED: {len(unrecovered)} spikes did not recover")
    elif len(unrecovered) == 1:
        score -= 0.05
        flags.append("UNRECOVERED: 1 spike did not recover")

    # Peak regression penalty
    if diag.peak_reward > 0 and diag.peak_to_final_drop > 0.3 * abs(diag.peak_reward):
        score -= 0.15
        flags.append(
            f"REGRESSION: peak {diag.peak_reward:.1f} -> final {diag.final_reward:.1f} "
            f"({diag.peak_to_final_drop:.1f} drop)"
        )

    # Stability penalty
    if diag.stability_ratio > 1.0:
        score -= 0.1
        flags.append(f"UNSTABLE_TAIL: final variance > overall (ratio {diag.stability_ratio:.2f})")

    if not flags:
        flags.append("CLEAN: smooth convergence, no anomalies")

    return max(0.0, min(1.0, score)), flags


def diagnose_run(filepath: str) -> RunDiagnostics:
    """Run full diagnostics on a single parquet file."""
    fname = os.path.basename(filepath)
    meta = _parse_filename(fname)

    df = pq.read_table(filepath).to_pandas()

    # Extract reward time-series
    reward_df = df[df["metric"] == "rollout/reward_mean"].sort_values("step")
    if reward_df.empty:
        # Fallback: maybe different metric name
        for candidate in ["rollout/ep_rew_mean", "eval/mean_reward"]:
            reward_df = df[df["metric"] == candidate].sort_values("step")
            if not reward_df.empty:
                break

    if reward_df.empty:
        return RunDiagnostics(
            run_id=meta["run_id"],
            algorithm=meta["algorithm"],
            env=meta["env"],
            seed=meta["seed"],
            total_steps=0,
            n_data_points=0,
            convergence_class="no_data",
            stability_ratio=0,
            final_reward=0,
            peak_reward=0,
            peak_step=0,
            peak_to_final_drop=0,
            improving_pct=0,
            early_mean=0,
            late_mean=0,
            improvement=0,
            steps_to_50pct=None,
            steps_to_80pct=None,
            trust_score=0,
            trust_flags=["NO_DATA: no reward metric found"],
        )

    steps = reward_df["step"].values
    vals = reward_df["value"].values
    n = len(vals)

    # Basic stats
    total_steps = int(steps[-1]) if n > 0 else 0
    final_reward = float(vals[-1]) if n > 0 else 0
    peak_idx = int(np.argmax(vals))
    peak_reward = float(vals[peak_idx])
    peak_step = int(steps[peak_idx])

    # Convergence
    n20 = max(1, n // 5)
    final_std = float(vals[-n20:].std()) if n > 1 else 0
    overall_std = float(vals.std()) if n > 1 else 1
    stability_ratio = final_std / overall_std if overall_std > 1e-8 else 0

    # Improvement direction
    diffs = np.diff(vals)
    improving_pct = float(np.sum(diffs > 0) / len(diffs) * 100) if len(diffs) > 0 else 0

    midpoint = n // 2
    early_mean = float(vals[:midpoint].mean()) if midpoint > 0 else 0
    late_mean = float(vals[midpoint:].mean()) if midpoint < n else 0

    convergence = _convergence_class(stability_ratio, improving_pct, late_mean, early_mean)

    # Sample efficiency
    if final_reward > 0:
        steps_50 = _steps_to_threshold(steps, vals, final_reward * 0.5)
        steps_80 = _steps_to_threshold(steps, vals, final_reward * 0.8)
    else:
        steps_50 = None
        steps_80 = None

    # Spikes
    spikes = _detect_spikes(steps, vals)
    all_recovered = all(s.recovered for s in spikes)
    worst_z = max((abs(s.z_score) for s in spikes), default=0)

    # Component analysis
    comp_summary = {}
    dominant = ""
    max_abs = 0
    for metric_name in df["metric"].unique():
        if metric_name.startswith("reward_components/") and metric_name.endswith("_mean"):
            comp_name = metric_name.removeprefix("reward_components/").removesuffix("_mean")
            comp_df = df[df["metric"] == metric_name].sort_values("step")
            if comp_df.empty:
                continue
            comp_vals = comp_df["value"].values
            final_val = float(comp_vals[-1])
            comp_summary[comp_name] = {
                "final_mean": round(final_val, 4),
                "overall_mean": round(float(comp_vals.mean()), 4),
                "trend": round(
                    float(
                        comp_vals[-max(1, len(comp_vals) // 5) :].mean()
                        - comp_vals[: max(1, len(comp_vals) // 5)].mean()
                    ),
                    4,
                ),
            }
            if abs(final_val) > max_abs:
                max_abs = abs(final_val)
                dominant = comp_name

    diag = RunDiagnostics(
        run_id=meta["run_id"],
        algorithm=meta["algorithm"],
        env=meta["env"],
        seed=meta["seed"],
        total_steps=total_steps,
        n_data_points=n,
        convergence_class=convergence,
        stability_ratio=round(stability_ratio, 3),
        final_reward=round(final_reward, 2),
        peak_reward=round(peak_reward, 2),
        peak_step=peak_step,
        peak_to_final_drop=round(peak_reward - final_reward, 2),
        improving_pct=round(improving_pct, 1),
        early_mean=round(early_mean, 2),
        late_mean=round(late_mean, 2),
        improvement=round(late_mean - early_mean, 2),
        steps_to_50pct=steps_50,
        steps_to_80pct=steps_80,
        spikes=spikes,
        n_spikes=len(spikes),
        worst_spike_z=round(worst_z, 2),
        all_spikes_recovered=all_recovered,
        dominant_component=dominant,
        component_summary=comp_summary,
    )

    diag.trust_score, diag.trust_flags = _compute_trust(diag)
    diag.trust_score = round(diag.trust_score, 2)

    return diag


def diagnose_cross_seed(
    diagnostics: list[RunDiagnostics],
    parquet_dir: str,
) -> CrossSeedDiagnostics | None:
    """Cross-seed analysis for runs sharing algorithm+env."""
    if len(diagnostics) < 2:
        return None

    algo = diagnostics[0].algorithm
    env = diagnostics[0].env
    seeds = [d.seed for d in diagnostics]
    finals = [d.final_reward for d in diagnostics]

    final_mean = float(np.mean(finals))
    final_std = float(np.std(finals))
    cv = final_std / abs(final_mean) if abs(final_mean) > 1e-8 else float("inf")

    # Failure detection
    failures = [d for d in diagnostics if d.final_reward < 0 or d.trust_score < 0.5]
    failure_seeds = [d.seed for d in failures]

    # Trajectory shape agreement (pairwise correlation of reward curves)
    reward_series = []
    for d in diagnostics:
        # Find the parquet for this run
        pattern = f"{d.run_id}_"
        for fname in os.listdir(parquet_dir):
            if fname.startswith(pattern) and fname.endswith(".parquet"):
                df = pq.read_table(os.path.join(parquet_dir, fname)).to_pandas()
                rdf = df[df["metric"] == "rollout/reward_mean"].sort_values("step")
                if not rdf.empty:
                    reward_series.append(rdf["value"].values)
                break

    # Interpolate to common length for correlation
    shape_agreement = 0.0
    if len(reward_series) >= 2:
        min_len = min(len(s) for s in reward_series)
        if min_len > 5:
            # Resample all to same length
            resampled = []
            for s in reward_series:
                indices = np.linspace(0, len(s) - 1, min_len).astype(int)
                resampled.append(s[indices])
            # Average pairwise correlation
            corrs = []
            for i in range(len(resampled)):
                for j in range(i + 1, len(resampled)):
                    std_i = np.std(resampled[i])
                    std_j = np.std(resampled[j])
                    if std_i > 1e-8 and std_j > 1e-8:
                        corr = np.corrcoef(resampled[i], resampled[j])[0, 1]
                        if np.isfinite(corr):
                            corrs.append(corr)
            if corrs:
                shape_agreement = float(np.mean(corrs))

    per_seed_trust = {d.seed: d.trust_score for d in diagnostics}

    return CrossSeedDiagnostics(
        algorithm=algo,
        env=env,
        n_seeds=len(diagnostics),
        seeds=seeds,
        final_mean=round(final_mean, 2),
        final_std=round(final_std, 2),
        cv=round(cv, 3),
        n_failures=len(failures),
        failure_seeds=failure_seeds,
        failure_rate=round(len(failures) / len(diagnostics), 2),
        shape_agreement=round(shape_agreement, 3),
        per_seed_trust=per_seed_trust,
    )


# ---------------------------------------------------------------------------
# Plot generation (optional, to diagnostics/ dir)
# ---------------------------------------------------------------------------


def _plot_run(diag: RunDiagnostics, filepath: str, output_dir: str) -> list[str]:
    """Generate diagnostic plots for a single run. Returns list of plot paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pq.read_table(filepath).to_pandas()
    plots = []

    run_dir = os.path.join(output_dir, f"{diag.run_id}_{diag.algorithm}_{diag.env}_s{diag.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Reward curve with spikes marked
    reward_df = df[df["metric"] == "rollout/reward_mean"].sort_values("step")
    if not reward_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = reward_df["step"].values / 1000  # ksteps
        vals = reward_df["value"].values
        ax.plot(steps, vals, linewidth=0.8, color="#2196F3", label="reward")

        # Rolling mean
        window = max(1, len(vals) // 20)
        if len(vals) > window:
            rolling = np.convolve(vals, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1 :], rolling, linewidth=1.5, color="#1565C0", label=f"rolling avg (w={window})")

        # Mark spikes
        for spike in diag.spikes:
            color = "#4CAF50" if spike.recovered else "#F44336"
            marker = "v" if spike.z_score < 0 else "^"
            ax.scatter(
                spike.step / 1000,
                spike.value,
                color=color,
                marker=marker,
                s=60,
                zorder=5,
                label=f"spike {'(recovered)' if spike.recovered else '(unrecovered)'}"
                if spike == diag.spikes[0]
                else "",
            )

        ax.set_xlabel("Steps (k)")
        ax.set_ylabel("Reward")
        ax.set_title(
            f"{diag.algorithm} | {diag.env} | s{diag.seed} | trust={diag.trust_score:.2f} | {diag.convergence_class}"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(run_dir, "reward_curve.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        plots.append(path)

    # 2. Component balance over time
    comp_metrics = [m for m in df["metric"].unique() if m.startswith("reward_components/") and m.endswith("_mean")]
    if comp_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        for metric in sorted(comp_metrics):
            comp_name = metric.removeprefix("reward_components/").removesuffix("_mean")
            cdf = df[df["metric"] == metric].sort_values("step")
            ax.plot(cdf["step"].values / 1000, cdf["value"].values, linewidth=0.8, label=comp_name)
        ax.set_xlabel("Steps (k)")
        ax.set_ylabel("Component Mean")
        ax.set_title(f"Component Balance | {diag.algorithm} | {diag.env} | s{diag.seed}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(run_dir, "component_balance.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        plots.append(path)

    # 3. Loss + entropy
    loss_df = df[df["metric"] == "train/loss"].sort_values("step")
    ent_df = df[df["metric"] == "train/entropy_coeff"].sort_values("step")
    if not loss_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(loss_df["step"].values / 1000, loss_df["value"].values, color="#FF9800", linewidth=0.8, label="loss")
        ax1.set_xlabel("Steps (k)")
        ax1.set_ylabel("Loss", color="#FF9800")
        if not ent_df.empty:
            ax2 = ax1.twinx()
            ax2.plot(
                ent_df["step"].values / 1000,
                ent_df["value"].values,
                color="#9C27B0",
                linewidth=0.8,
                label="entropy coeff",
            )
            ax2.set_ylabel("Entropy Coeff", color="#9C27B0")
        ax1.set_title(f"Training Dynamics | {diag.algorithm} | {diag.env} | s{diag.seed}")
        fig.tight_layout()
        path = os.path.join(run_dir, "training_dynamics.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        plots.append(path)

    return plots


def _plot_cross_seed(
    cross: CrossSeedDiagnostics,
    diagnostics: list[RunDiagnostics],
    parquet_dir: str,
    output_dir: str,
) -> list[str]:
    """Generate cross-seed comparison plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = []
    group_dir = os.path.join(output_dir, f"cross_{cross.algorithm}_{cross.env}")
    os.makedirs(group_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for d in sorted(diagnostics, key=lambda x: x.seed):
        pattern = f"{d.run_id}_"
        for fname in os.listdir(parquet_dir):
            if fname.startswith(pattern) and fname.endswith(".parquet"):
                df = pq.read_table(os.path.join(parquet_dir, fname)).to_pandas()
                rdf = df[df["metric"] == "rollout/reward_mean"].sort_values("step")
                if not rdf.empty:
                    trust_marker = "*" if d.trust_score < 0.5 else ""
                    ax.plot(
                        rdf["step"].values / 1000,
                        rdf["value"].values,
                        linewidth=0.8,
                        alpha=0.7,
                        label=f"s{d.seed} (trust={d.trust_score:.2f}{trust_marker})",
                    )
                break

    ax.set_xlabel("Steps (k)")
    ax.set_ylabel("Reward")
    ax.set_title(
        f"Cross-Seed | {cross.algorithm} | {cross.env} | "
        f"{cross.n_seeds} seeds | shape_agree={cross.shape_agreement:.2f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(group_dir, "seed_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    plots.append(path)

    return plots


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_run_summary(d: RunDiagnostics) -> str:
    """One-paragraph summary of a run for the digest."""
    lines = [
        f"**{d.run_id}** | {d.algorithm} | {d.env} | s{d.seed} | trust={d.trust_score:.2f} | {d.convergence_class}",
    ]
    lines.append(
        f"  Final: {d.final_reward:.1f} | Peak: {d.peak_reward:.1f} @ {d.peak_step:,} | "
        f"Improvement: {d.improvement:+.1f} | Stability: {d.stability_ratio:.2f}"
    )
    if d.n_spikes > 0:
        recovered = sum(1 for s in d.spikes if s.recovered)
        lines.append(f"  Spikes: {d.n_spikes} ({recovered} recovered, worst z={d.worst_spike_z:.1f})")
    if d.steps_to_50pct is not None:
        lines.append(f"  Sample efficiency: 50% @ {d.steps_to_50pct:,} | 80% @ {d.steps_to_80pct or 'N/A':,}")
    if d.dominant_component:
        lines.append(f"  Dominant component: {d.dominant_component}")
    for flag in d.trust_flags:
        lines.append(f"  [{flag}]")
    return "\n".join(lines)


def _format_cross_seed_summary(c: CrossSeedDiagnostics) -> str:
    """Summary of cross-seed analysis."""
    lines = [
        f"**{c.algorithm} | {c.env}** | {c.n_seeds} seeds",
        f"  Final: {c.final_mean:.1f} +/- {c.final_std:.1f} (CV={c.cv:.2f})",
        f"  Shape agreement: {c.shape_agreement:.2f} (1.0 = identical curves)",
        f"  Failures: {c.n_failures}/{c.n_seeds} ({c.failure_rate:.0%})",
    ]
    if c.failure_seeds:
        lines.append(f"  Failed seeds: {', '.join(c.failure_seeds)}")
    lines.append(f"  Per-seed trust: {c.per_seed_trust}")
    return "\n".join(lines)


def generate_summary(
    run_diagnostics: list[RunDiagnostics],
    cross_diagnostics: list[CrossSeedDiagnostics],
    output_dir: str,
) -> str:
    """Write summary.md and return its content."""
    sections = ["# Diagnostics Summary\n"]

    # Cross-seed summaries first (higher-level)
    if cross_diagnostics:
        sections.append("## Cross-Seed Analysis\n")
        for c in sorted(cross_diagnostics, key=lambda x: (x.env, x.algorithm)):
            sections.append(_format_cross_seed_summary(c))
            sections.append("")

    # Per-run summaries, grouped by trust
    low_trust = [d for d in run_diagnostics if d.trust_score < 0.7]
    high_trust = [d for d in run_diagnostics if d.trust_score >= 0.7]

    if low_trust:
        sections.append("## Runs Requiring Attention (trust < 0.7)\n")
        for d in sorted(low_trust, key=lambda x: x.trust_score):
            sections.append(_format_run_summary(d))
            sections.append("")

    if high_trust:
        sections.append(f"## Clean Runs ({len(high_trust)} runs, trust >= 0.7)\n")
        for d in sorted(high_trust, key=lambda x: (x.env, x.algorithm, x.seed)):
            sections.append(_format_run_summary(d))
            sections.append("")

    content = "\n".join(sections)
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(content)

    return content


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_diagnostics(
    metrics_dir: str,
    run_ids: list[str] | None = None,
    with_plots: bool = False,
    cross_seed_filter: tuple[str, str] | None = None,  # (algo, env)
    output_dir: str | None = None,
) -> tuple[list[RunDiagnostics], list[CrossSeedDiagnostics], str]:
    """Run full diagnostic pipeline.

    Args:
        metrics_dir: Path to directory containing parquet files.
        run_ids: If set, only diagnose these run_ids (prefix match).
        with_plots: Generate plots to output_dir.
        cross_seed_filter: If set, only cross-seed for this (algo, env).
        output_dir: Output directory for plots/summary
            (default: sibling ``diagnostics/`` dir next to metrics).

    Returns:
        (run_diagnostics, cross_seed_diagnostics, summary_text)
    """
    if output_dir is None:
        # Derive output path from metrics_dir:
        # artifacts/pcz-ppo/data/metrics -> artifacts/pcz-ppo/data/diagnostics
        # The diagnostics/ dir is gitignored by the blanket /artifacts/*/**
        # rule — only parquet and results.csv are allowed through.
        parent = os.path.dirname(metrics_dir)  # artifacts/pcz-ppo/data
        output_dir = os.path.join(parent, "diagnostics")

    os.makedirs(output_dir, exist_ok=True)

    # Find parquet files
    parquet_files = sorted(f for f in os.listdir(metrics_dir) if f.endswith(".parquet"))

    if run_ids:
        parquet_files = [f for f in parquet_files if any(f.startswith(rid) for rid in run_ids)]

    if not parquet_files:
        print("No parquet files found.")
        return [], [], ""

    print(f"Diagnosing {len(parquet_files)} runs...")

    # Per-run diagnostics
    all_diags = []
    for fname in parquet_files:
        filepath = os.path.join(metrics_dir, fname)
        diag = diagnose_run(filepath)
        all_diags.append(diag)

        if with_plots:
            _plot_run(diag, filepath, output_dir)

    # Cross-seed grouping
    from collections import defaultdict

    groups = defaultdict(list)
    for d in all_diags:
        key = (d.algorithm, d.env)
        groups[key].append(d)

    cross_diags = []
    for (algo, env), diags in sorted(groups.items()):
        if cross_seed_filter and (algo, env) != cross_seed_filter:
            continue
        if len(diags) < 2:
            continue
        cross = diagnose_cross_seed(diags, metrics_dir)
        if cross:
            cross_diags.append(cross)
            if with_plots:
                _plot_cross_seed(cross, diags, metrics_dir, output_dir)

    # Write JSON diagnostics per run
    for d in all_diags:
        run_dir = os.path.join(output_dir, f"{d.run_id}_{d.algorithm}_{d.env}_s{d.seed}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "diagnostics.json"), "w") as f:
            json.dump(asdict(d), f, indent=2, default=str)

    # Write cross-seed JSON
    for c in cross_diags:
        group_dir = os.path.join(output_dir, f"cross_{c.algorithm}_{c.env}")
        os.makedirs(group_dir, exist_ok=True)
        with open(os.path.join(group_dir, "diagnostics.json"), "w") as f:
            json.dump(asdict(c), f, indent=2, default=str)

    # Summary
    summary = generate_summary(all_diags, cross_diags, output_dir)

    # Print key stats
    n_clean = sum(1 for d in all_diags if d.trust_score >= 0.7)
    n_flagged = len(all_diags) - n_clean
    print(f"\nResults: {n_clean} clean, {n_flagged} flagged (trust < 0.7)")
    print(f"Output: {output_dir}/")

    return all_diags, cross_diags, summary


def main():
    parser = argparse.ArgumentParser(
        description="Automated post-training diagnostics from parquet metrics.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="artifacts/pcz-ppo/data/metrics",
        help="Directory containing per-run parquet files.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help="Only diagnose these run IDs (prefix match).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate diagnostic plots (to diagnostics/ dir next to metrics/).",
    )
    parser.add_argument(
        "--cross-seed",
        action="store_true",
        help="Include cross-seed analysis.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Filter cross-seed analysis to this algorithm.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Filter cross-seed analysis to this environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/<project>/data/diagnostics).",
    )
    args = parser.parse_args()

    cross_filter = None
    if args.algo and args.env:
        cross_filter = (args.algo, args.env)

    diags, _cross, summary = run_diagnostics(
        metrics_dir=args.metrics_dir,
        run_ids=args.runs,
        with_plots=args.plots,
        cross_seed_filter=cross_filter,
        output_dir=args.output_dir,
    )

    if not diags:
        sys.exit(1)

    # Print summary to stdout
    print("\n" + summary)


if __name__ == "__main__":
    main()
