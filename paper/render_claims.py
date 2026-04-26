"""Render numerical claims to per-claim LaTeX fragments.

NAMING CONVENTION:
  k{N}_*              — K-scaling rows (e.g. k4_pcz_stat, k8_ratio)
  ablX_*              — ablation row X (e.g. ablA1_stat, ablB2_mean)
  abl_*               — cross-row derived ablation quantities (e.g. abl_decomp_delta)
  grpo{K}_{T}_*       — GRPO at K components, T timesteps
  ce{ENV}_*           — cross-environment supporting evidence
  se_{T}_*            — sample-efficiency time points
  LLM_k{K}_{mode}_*  — LLM alignment results
  w{N}_*              — long-horizon (4M) results at N weight-config
  {prefix}_prob_improvement — P(PCZ > PPO) pairwise probability

Every fragment is a tiny .tex file under ``generated/`` whose contents come
directly from ``results.csv`` via ``fig_data.query``.  The paper includes
them with ``\\input{generated/<name>.tex}`` (or the ``\\cnum{<name>}``
shorthand defined in the preamble).

Usage::

    cd /workspace && uv run python artifacts/pcz-ppo/paper/render_claims.py
    cd /workspace && uv run python artifacts/pcz-ppo/paper/render_claims.py --check
      # --check: render to tempdir and fail if content differs from committed
      # files.  Used in CI and the pre-commit hook.

Design notes
------------
* One fragment per *semantic* claim.  If the paper says ``$+157.7 \\pm 27.0$``
  (10 seeds), that becomes three fragments: ``<name>_stat``, ``<name>_mean``,
  ``<name>_seeds``.  Over-decomposition is deliberate: fragments are cheap,
  and individual values are reused across tables.
* Formatting is done once, here, so sig-figs and sign conventions are
  centralised.  Tables no longer carry formatting decisions.
* Every registered claim MUST produce a non-empty fragment.  A missing row
  in ``results.csv`` fails the render rather than silently emitting ``0.0``.
* Fragment files end with a single trailing newline for readable git diffs
  and no extra whitespace after ``\\input`` (the ``\\cnum`` macro uses
  ``\\unskip`` to strip any stray space).
* Writes are atomic (tempfile + rename) so a crash mid-render never leaves
  a truncated fragment that latexmk would pick up.

Weight-filter discipline (MANDATORY for every K-emit site)
----------------------------------------------------------
Every headline K-scaling emit MUST pass an explicit ``weights=`` argument
that uniquely identifies the canonical reward-weight configuration.  The
``fig_data.query`` filter is a ``startswith`` prefix match, so the filter
string should be as long as needed to exclude any off-canonical rows that
share the same leading components.  ``weights=None`` is ONLY safe when
``results.csv`` contains zero off-canonical rows for that (algorithm, env,
total_timesteps) triple — verify manually before relying on it.

Why this matters: a single empty-weights seed row can win chronologically-latest
dedupe and drive K=6 PCZ from +167 down to +155 with doubled SD (Welch p
0.002→0.214, var-ratio 0.9×→0.4×).  ``--check`` catches fragment drift but not
filter drift.  The test ``TestWeightConsistency`` in
``tests/test_render_claims.py`` enforces the invariant structurally: every
resolved seed in every headline K-row must share a single canonical weight
string.  If you add a new K-row, add it to ``HEADLINE_CONFIGS`` in that test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_data import load_results, query

_PAPER_DIR = Path(__file__).parent
_GEN_DIR = _PAPER_DIR / "generated"
_MANIFEST = _GEN_DIR / "_manifest.json"

LL_PRIMARY_WEIGHTS = "10.00,5.00,0.50,0.50"
# Canonical K=6 weights (env default; proportional to K=4/K=8). Filter prevents
# an off-canonical seed from silently contaminating the K=6 stats —
# without this filter the dedupe picks an equal-weights accidental run and
# drives the mean from +167 down to +155 with doubled std.
LL_K6_WEIGHTS = "10.00,3.00"

# Weight-config sensitivity matrix (LL K=4 500k, n=5 each)
K4_WEIGHTS_MODERATE = "5.00,3.00,1.00,1.00"
K4_WEIGHTS_FLAT = "3.00,3.00,3.00,3.00"
K4_WEIGHTS_EQUAL = "1.00,1.00,1.00,1.00"

# K=8 4M weight-allocation experiment weights.
# H7 = "headline" weights (analogous to K=6 canonical structure: high primary,
# moderate per-component, halved fuel). PCZ +212.9 vs PPO +175.7 — PCZ wins.
# H8 = SNR-informed weights with velocity ZEROED (CV=177 → "pure noise"
# auto-demoted to 0). Outcome OPPOSITE to hypothesis: zeroing any declared
# component is pathological for PCZ (per-component variance accounting still
# uses the zeroed channel, but it contributes 0 to the scalar reward → degenerate
# normalisation). Documented in §Limitations as the "non-zero weights required"
# usage constraint.
K8_WEIGHTS_HETEROG = "10.00,1.00,1.00,1.00,1.00,1.00,0.50,0.50"
K8_WEIGHTS_ZEROED_VEL = "10.00,5.00,0.00,1.00,0.50,0.50,0.50,0.50"
# K=8/4M additional weight-sensitivity configs beyond H7,
# matching the K=4 sensitivity pattern. Three additional configs beyond H7
# (heterogeneous 20x spread, the only config where PCZ wins):
#   moderate: 5,3,1,1,1,1,0.5,0.5 (10x spread) — PCZ collapses +2.7 vs PPO +189.8
#   flat:     3,3,3,3,3,3,3,3     (1x, uniform) — PCZ collapses +1.8 vs PPO +213.5
K8_WEIGHTS_MODERATE = "5.00,3.00,1.00,1.00,1.00,1.00,0.50,0.50"
K8_WEIGHTS_FLAT = "3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00"

# --- formatting helpers ---------------------------------------------------


def _check_finite(x: float, ctx: str) -> float:
    if x != x or x == float("inf") or x == float("-inf"):
        raise ValueError(f"non-finite value for {ctx}: {x!r}")
    return x


def fmt_signed(x: float, decimals: int = 1) -> str:
    """Format a float with explicit sign: ``+157.7`` / ``-67.8``.

    Zero is formatted as ``+0.0`` (no special-case for negative zero).
    """
    _check_finite(x, "fmt_signed")
    s = f"{x:+.{decimals}f}"
    # -0.0 -> +0.0 (symmetric)
    if s == f"-{'0.' + '0' * decimals}":
        s = s.replace("-", "+")
    return s


def fmt_plain(x: float, decimals: int = 1) -> str:
    _check_finite(x, "fmt_plain")
    return f"{x:.{decimals}f}"


def fmt_ratio(x: float, decimals: int = 2) -> str:
    _check_finite(x, "fmt_ratio")
    return f"{x:.{decimals}f}"


def fmt_int(x: int) -> str:
    if not isinstance(x, int):
        raise TypeError(f"expected int, got {type(x).__name__}")
    return str(x)


def fmt_stat(mean: float, std: float, decimals: int = 1, *, signed: bool = True) -> str:
    """Format ``mean \\pm std`` (LaTeX math-mode body, without $ delimiters)."""
    m = fmt_signed(mean, decimals) if signed else fmt_plain(mean, decimals)
    s = fmt_plain(std, decimals)
    return f"{m} \\pm {s}"


# --- Registry ------------------------------------------------------------


@dataclass
class Claim:
    """One emitted fragment.

    Attributes
    ----------
    name : str
        Filename stem; ``<name>.tex`` lives in ``generated/``.
    content : str
        The already-formatted fragment body.
    source : str
        Human-readable description of the query used (for audit trail).
    """

    name: str
    content: str
    source: str


class Registry:
    """Collect claims, enforce name uniqueness, write atomically.

    Supports an optional ``used_filter`` set.  If set, ``add()`` silently
    skips claims whose name isn't in the filter — so registrations can be
    "defensive" (the catalog offers every shape) while only referenced
    fragments are materialized on disk.  Names added but filtered out are
    tracked in ``_registered_but_filtered`` for audit.
    """

    def __init__(self, used_filter: set[str] | None = None) -> None:
        self._claims: dict[str, Claim] = {}
        self._used_filter = used_filter
        self._registered_but_filtered: set[str] = set()

    def add(self, name: str, content: str, source: str) -> None:
        if name in self._claims:
            raise KeyError(f"duplicate claim name: {name!r}")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid claim name {name!r}: alnum + '_' + '-' only")
        if "\n" in content:
            raise ValueError(f"claim {name!r}: fragment contains newline")
        if not content.strip():
            raise ValueError(f"claim {name!r}: empty fragment")
        if self._used_filter is not None and name not in self._used_filter:
            self._registered_but_filtered.add(name)
            return
        self._claims[name] = Claim(name=name, content=content, source=source)

    def as_dict(self) -> dict[str, Claim]:
        return dict(self._claims)

    def render_files(self, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Remove any stale fragments no longer in the registry (so diff is honest)
        for p in target_dir.glob("*.tex"):
            if p.stem not in self._claims:
                p.unlink()
        for c in self._claims.values():
            _atomic_write(target_dir / f"{c.name}.tex", c.content + "\n")
        manifest = {
            c.name: {
                "sha256": hashlib.sha256((c.content + "\n").encode()).hexdigest(),
                "source": c.source,
            }
            for c in self._claims.values()
        }
        _atomic_write(target_dir / "_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    # Write to tempfile in same dir (so rename is atomic on same filesystem)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --- Query wrappers with loud failures -----------------------------------


def q_required(
    rows: list[dict],
    *,
    algorithm: str,
    env: str,
    total_timesteps: int,
    weights: str | None = None,
    min_seeds: int = 1,
    label: str,
    ent_coef_schedule: str | None = None,
    learning_rate: str | None = None,
) -> dict:
    """Query that raises if fewer than ``min_seeds`` runs match.

    Augments ``fig_data.query``'s result with ``mean_raw`` / ``std_raw``
    (unrounded) so derived quantities such as ratios and variance ratios
    don't accumulate rounding error.  ``label`` is used in error messages.
    ``ent_coef_schedule`` (optional) restricts to runs with an exact-match
    schedule string — used to isolate canonical-schedule headline runs from
    fixed-entropy tuning-audit runs that share seeds.
    ``learning_rate`` (optional) restricts to an exact LR — use for canonical
    K-scaling rows when HP-sweep data at non-canonical LRs shares seeds, so that
    non-canonical LR cells don't displace canonical seeds via chrono-latest dedupe.
    """
    import numpy as np

    q = query(
        rows,
        algorithm=algorithm,
        env=env,
        total_timesteps=total_timesteps,
        weights=weights,
        ent_coef_schedule=ent_coef_schedule,
        learning_rate=learning_rate,
    )
    if q["seeds"] < min_seeds:
        raise LookupError(
            f"[{label}] query({algorithm=}, {env=}, {total_timesteps=}, {weights=}, "
            f"{ent_coef_schedule=}, {learning_rate=}) returned {q['seeds']} seeds, "
            f"expected >= {min_seeds}"
        )
    evals = [float(r["eval_mean"]) for r in q["runs"] if r.get("eval_mean")]
    q["mean_raw"] = float(np.mean(evals))
    ddof = 1 if len(evals) > 1 else 0
    q["std_raw"] = float(np.std(evals, ddof=ddof))
    return q


# --- Claim definitions ----------------------------------------------------


def _emit_pcz_ppo_pair(
    reg: Registry,
    rows: list[dict],
    prefix: str,
    *,
    env: str,
    total_timesteps: int,
    weights: str | None,
    min_seeds: int = 1,
    ent_coef_schedule: str | None = None,
    learning_rate: str | None = None,
) -> None:
    """Emit standard fragments for a (PCZ vs PPO) comparison at one (env, ts, weights).

    Emits, for both PCZ-PPO and PPO:
      <prefix>_{pcz,ppo}_mean, _std, _seeds, _stat
    And for the pair:
      <prefix>_delta, _ratio, _var_ratio

    ``ent_coef_schedule`` restricts to a specific entropy schedule (e.g.
    ``"0.1:0.01"``) — use this for headline K-scaling queries so that seeds
    shared with fixed-entropy tuning audits don't override the dedupe.
    ``learning_rate`` (optional) restricts to exact LR — use when HP-sweep data
    lands in results.csv to prevent non-canonical LR cells from displacing
    canonical seeds via chrono-latest.
    """
    pcz = q_required(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env=env,
        total_timesteps=total_timesteps,
        weights=weights,
        min_seeds=min_seeds,
        label=f"{prefix}_pcz",
        ent_coef_schedule=ent_coef_schedule,
        learning_rate=learning_rate,
    )
    ppo = q_required(
        rows,
        algorithm="torchrl-ppo",
        env=env,
        total_timesteps=total_timesteps,
        weights=weights,
        min_seeds=min_seeds,
        label=f"{prefix}_ppo",
        ent_coef_schedule=ent_coef_schedule,
        learning_rate=learning_rate,
    )
    src = f"env={env} ts={total_timesteps} weights={weights or '*'}"

    for tag, q in [("pcz", pcz), ("ppo", ppo)]:
        reg.add(f"{prefix}_{tag}_mean", fmt_signed(q["mean"]), src)
        reg.add(f"{prefix}_{tag}_std", fmt_plain(q["std"]), src)
        reg.add(f"{prefix}_{tag}_seeds", fmt_int(q["seeds"]), src)
        reg.add(f"{prefix}_{tag}_stat", fmt_stat(q["mean"], q["std"]), src)
        reg.add(f"{prefix}_{tag}_confidence", _seeds_confidence(q["seeds"]), src)
    # Pair-level confidence: floor of the two arms, since a comparison is
    # only as strong as its weakest arm.
    reg.add(
        f"{prefix}_confidence",
        _seeds_confidence(min(pcz["seeds"], ppo["seeds"])),
        src,
    )

    # Derived quantities use raw (unrounded) mean/std to avoid double-rounding
    # error.  The displayed means/stds are still 1-decimal (fmt_signed/plain).
    delta = pcz["mean_raw"] - ppo["mean_raw"]
    reg.add(f"{prefix}_delta", fmt_signed(delta), src)
    if ppo["mean_raw"] > 0:
        reg.add(f"{prefix}_ratio", fmt_ratio(pcz["mean_raw"] / ppo["mean_raw"]), src)
    if pcz["std_raw"] > 0:
        reg.add(
            f"{prefix}_var_ratio",
            fmt_ratio((ppo["std_raw"] / pcz["std_raw"]) ** 2, decimals=1),
            src,
        )


def _emit_pcz_ppo_pair_exact_weights(
    reg: Registry,
    rows: list[dict],
    prefix: str,
    *,
    env: str,
    total_timesteps: int,
    weights_exact: str,
    ent_coef_schedule: str | None = None,
    min_seeds: int = 1,
) -> None:
    """Same shape as ``_emit_pcz_ppo_pair`` but matches ``component_weights``
    by EXACT string equality, not prefix.

    Required for the empty-weights regime (env default, all-1.0 weights at K=8
    4M): ``query`` filters via ``startswith``, so ``weights=""`` would admit
    every run.  Used by the K=8/4M equal-weights negative-control fragments.
    """
    import numpy as np

    def _exact_subset(algo: str) -> list[dict]:
        return [
            r
            for r in rows
            if r["algorithm"] == algo
            and r["env"] == env
            and r["total_timesteps"] == str(total_timesteps)
            and r.get("component_weights", "") == weights_exact
            and (ent_coef_schedule is None or r.get("ent_coef_schedule", "") == ent_coef_schedule)
            and r.get("eval_mean", "")
        ]

    def _summarize(matching: list[dict]) -> dict:
        # chrono-latest dedupe per seed (matches fig_data.query)
        by_seed: dict[str, dict] = {}
        for r in sorted(matching, key=lambda r: r.get("date", "")):
            by_seed[r["seed"]] = r
        evals = [float(r["eval_mean"]) for r in by_seed.values()]
        ddof = 1 if len(evals) > 1 else 0
        return {
            "mean": round(float(np.mean(evals)), 1),
            "std": round(float(np.std(evals, ddof=ddof)), 1),
            "mean_raw": float(np.mean(evals)),
            "std_raw": float(np.std(evals, ddof=ddof)),
            "seeds": len(evals),
        }

    pcz_rows = _exact_subset("torchrl-pcz-ppo-running")
    ppo_rows = _exact_subset("torchrl-ppo")
    pcz = _summarize(pcz_rows)
    ppo = _summarize(ppo_rows)
    if pcz["seeds"] < min_seeds or ppo["seeds"] < min_seeds:
        raise LookupError(
            f"[{prefix}] exact-weights query (env={env}, ts={total_timesteps}, "
            f"weights='{weights_exact}') returned PCZ {pcz['seeds']} / PPO "
            f"{ppo['seeds']}, expected >= {min_seeds}"
        )

    src = f"env={env} ts={total_timesteps} weights='{weights_exact}' (exact)"
    for tag, q in [("pcz", pcz), ("ppo", ppo)]:
        reg.add(f"{prefix}_{tag}_mean", fmt_signed(q["mean"]), src)
        reg.add(f"{prefix}_{tag}_std", fmt_plain(q["std"]), src)
        reg.add(f"{prefix}_{tag}_seeds", fmt_int(q["seeds"]), src)
        reg.add(f"{prefix}_{tag}_stat", fmt_stat(q["mean"], q["std"]), src)
        reg.add(f"{prefix}_{tag}_confidence", _seeds_confidence(q["seeds"]), src)
    reg.add(
        f"{prefix}_confidence",
        _seeds_confidence(min(pcz["seeds"], ppo["seeds"])),
        src,
    )
    delta = pcz["mean_raw"] - ppo["mean_raw"]
    reg.add(f"{prefix}_delta", fmt_signed(delta), src)
    if pcz["std_raw"] > 0:
        reg.add(
            f"{prefix}_var_ratio",
            fmt_ratio((ppo["std_raw"] / pcz["std_raw"]) ** 2, decimals=1),
            src,
        )


# --- Seed-count confidence labels (G6 Multi-Seed Statistical Enforcement) -
#
# Maps the seed count behind a claim to a confidence tier so the paper can
# surface (and a future linter can enforce) honest qualifiers like
# "PRELIMINARY" or "SIGNAL" instead of letting under-powered claims land
# unmarked. Tier thresholds match the convention documented in the
# the project statistical conventions:
#
#   n  < 3   -> "preliminary"
#   3 <= n < 10 -> "signal"
#   n >= 10  -> "confirmed"
#
# Available via ``\cnum{<prefix>_confidence}`` — emitted automatically by
# ``_emit_pcz_ppo_pair``, ``_emit_pcz_ppo_pair_exact_weights``, and
# ``_emit_single``. The Registry's ``used_filter`` means only fragments the
# paper actually references materialize on disk.
def _seeds_confidence(n: int) -> str:
    # Reject bool explicitly (bool is an int subclass in Python). Catches the
    # accidental ``_seeds_confidence(True)`` that would silently map to
    # "preliminary".
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"_seeds_confidence: expected non-negative int, got {n!r}")
    if n < 3:
        return "preliminary"
    if n < 10:
        return "signal"
    return "confirmed"


# --- Comparator descriptors (Paper Semantic Integrity v1) -----------------
#
# Maps a numeric ratio to an English adjective phrase so the prose
# automatically tracks the data. Use via ``_emit_comparator()`` for any
# ratio claim where the words around the number ("roughly", "more than",
# "comparable") would otherwise be hand-typed and risk going stale when
# results.csv is re-exported.
#
# Ranges are tuples of (low_inclusive, high_exclusive, phrase). The first
# match wins. Overlapping ranges are a registration error.
# Phrases use LaTeX math mode (``$...$``) for the times symbol so they render
# correctly when ``\cnum{}`` is dropped into prose. Pure-adjective tiers have
# no math; magnitude tiers wrap the number in ``$...\times$``.
COMPARATOR_TIERS_DEFAULT: tuple[tuple[float, float, str], ...] = (
    # ratios near 1 → effectively tied
    (0.95, 1.05, "comparable"),
    # mild advantages (5–25%)
    (1.05, 1.25, "marginally above"),
    (0.80, 0.95, "marginally below"),
    # moderate (25–75%)
    (1.25, 1.75, "noticeably above"),
    (0.55, 0.80, "noticeably below"),
    # large (1.75–3×)
    (1.75, 3.00, "more than $1.5\\times$"),
    (0.33, 0.55, "less than two-thirds of"),
    # very large (≥3×)
    (3.00, float("inf"), "more than $3\\times$"),
    (0.0, 0.33, "less than one-third of"),
)


def _comparator_phrase(
    ratio: float,
    tiers: tuple[tuple[float, float, str], ...] = COMPARATOR_TIERS_DEFAULT,
) -> str:
    _check_finite(ratio, "_comparator_phrase")
    if ratio < 0:
        raise ValueError(f"_comparator_phrase: negative ratio {ratio!r}")
    for lo, hi, phrase in tiers:
        if lo <= ratio < hi:
            return phrase
    raise ValueError(
        f"_comparator_phrase: ratio {ratio!r} matched no tier; widen "
        f"COMPARATOR_TIERS_DEFAULT or pass a custom tier list"
    )


def _emit_comparator(
    reg: Registry,
    name: str,
    *,
    ratio: float,
    src: str,
    tiers: tuple[tuple[float, float, str], ...] = COMPARATOR_TIERS_DEFAULT,
) -> None:
    """Emit a comparator descriptor fragment derived from a numeric ratio.

    Pair this with the corresponding ``_ratio`` fragment so the paper can
    say e.g. ``\\cnum{name_phrase} (\\cnum{name_ratio}$\\times$)`` and have
    *both* the words and the number track the data.
    """
    reg.add(name, _comparator_phrase(ratio, tiers), src + " | comparator")


def _emit_single(
    reg: Registry,
    rows: list[dict],
    prefix: str,
    *,
    algorithm: str,
    env: str,
    total_timesteps: int,
    weights: str | None = None,
    min_seeds: int = 1,
    ent_coef_schedule: str | None = None,
    learning_rate: str | None = None,
) -> None:
    q = q_required(
        rows,
        algorithm=algorithm,
        env=env,
        total_timesteps=total_timesteps,
        weights=weights,
        min_seeds=min_seeds,
        label=prefix,
        ent_coef_schedule=ent_coef_schedule,
        learning_rate=learning_rate,
    )
    src = f"algo={algorithm} env={env} ts={total_timesteps} weights={weights or '*'}"
    reg.add(f"{prefix}_mean", fmt_signed(q["mean"]), src)
    reg.add(f"{prefix}_std", fmt_plain(q["std"]), src)
    reg.add(f"{prefix}_seeds", fmt_int(q["seeds"]), src)
    reg.add(f"{prefix}_stat", fmt_stat(q["mean"], q["std"]), src)
    reg.add(f"{prefix}_confidence", _seeds_confidence(q["seeds"]), src)


def _load_llm_results(k: int) -> dict[str, list[float]]:
    """Aggregate held-out eval ``total_reward`` across seeds for standard/pcz modes at K.

    Data source: artifacts/pcz-ppo/llm_alignment/runs/result_k{K}_{mode}_s{seed}.json
    (separate data pipeline from results.csv; LLM experiments use TRL/RLOO).

    We intentionally use ``total_reward`` (eval-time raw composite) rather than
    ``final_train_reward`` (TRL's logged reward). The latter is NOT comparable across
    modes: in "standard" mode TRL logs the raw weighted sum, while in "pcz" mode TRL
    logs the z-normalised composite (mean ≈ 0 by construction, because the wrapper
    subtracts the running EMA mean before returning). Comparing raw reward to
    z-scores produced a spurious ~2.7-point "PCZ collapse" signal in earlier results;
    the eval reward (raw functions, held-out prompts) is the only apples-to-apples
    measure when standard runs and a backfilled raw-logged PCZ signal are not
    both available.
    """
    runs_dir = _PAPER_DIR.parent / "llm_alignment" / "runs"
    by_mode: dict[str, list[float]] = {"standard": [], "pcz": []}
    for mode in ["standard", "pcz"]:
        for seed in [42, 43, 44]:
            path = runs_dir / f"result_k{k}_{mode}_s{seed}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            r = data.get("total_reward")
            if r is not None:
                by_mode[mode].append(float(r))
    return by_mode


def _load_llm_k2_results() -> dict[str, list[float]]:
    """Backwards-compat alias for K=2."""
    return _load_llm_results(2)


def _emit_llm_claims_for_k(reg: Registry, k: int) -> None:
    """Emit LLM K={k} Standard vs PCZ-RLOO fragments from JSON results.

    Naming convention: LLM_k{K}_{standard,pcz}_{stat,seeds}, LLM_k{K}_delta.
    """
    import statistics

    data = _load_llm_results(k)
    src = f"llm_alignment/runs/result_k{k}_{{mode}}_s{{42,43,44}}.json"
    for mode in ["standard", "pcz"]:
        vals = data[mode]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0  # sample SD (ddof=1)
        reg.add(f"LLM_k{k}_{mode}_stat", fmt_stat(mean, std, decimals=3), src)
        reg.add(f"LLM_k{k}_{mode}_seeds", fmt_int(len(vals)), src)
    if data["standard"] and data["pcz"]:
        mean_std = statistics.fmean(data["standard"])
        mean_pcz = statistics.fmean(data["pcz"])
        reg.add(
            f"LLM_k{k}_delta",
            fmt_signed(mean_pcz - mean_std, decimals=3),
            f"PCZ mean - Standard mean (LLM K={k})",
        )


def _emit_llm_k2_claims(reg: Registry) -> None:
    """Backwards-compat alias for K=2."""
    _emit_llm_claims_for_k(reg, 2)


def _load_llm_ppo_results(k: int, seeds: list[int] | None = None, suffix: str = "") -> dict[str, list[float]]:
    """Aggregate held-out eval ``total_reward`` for PPO-critic runs (Phase A/B/C/D).

    Data source: ``artifacts/pcz-ppo/llm_alignment/runs/result_ppo_k{K}_{mode}_s{seed}{suffix}.json``.

    suffix="" reads live files (500-step for K=6).
    suffix="_200step" reads 200-step backup files (K=6 s43-46 preserved via backup script).

    Distinct from ``_load_llm_results`` which reads ``result_k{K}_*`` (RLOO legacy).
    """
    runs_dir = _PAPER_DIR.parent / "llm_alignment" / "runs"
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    by_mode: dict[str, list[float]] = {"standard": [], "pcz": []}
    for mode in ["standard", "pcz"]:
        for seed in seeds:
            path = runs_dir / f"result_ppo_k{k}_{mode}_s{seed}{suffix}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            r = data.get("total_reward")
            if r is not None:
                by_mode[mode].append(float(r))
    return by_mode


def _emit_llm_ppo_claims_for_k(
    reg: Registry,
    k: int,
    seeds: list[int] | None = None,
    suffix: str = "",
    prefix: str = "LLMppo",
) -> None:
    """Emit LLM PPO K={k} Standard vs PCZ fragments from JSON results.

    Naming convention: ``{prefix}_k{K}_{standard,pcz}_{mean,std,seeds,stat}``,
    ``{prefix}_k{K}_{delta,var_ratio}``. Default prefix=``LLMppo`` keeps existing
    fragments. ``suffix`` selects live (``""``) vs 200-step backup (``"_200step"``).
    """
    import statistics

    data = _load_llm_ppo_results(k, seeds=seeds, suffix=suffix)
    seed_list = seeds if seeds is not None else [42, 43, 44, 45, 46]
    src = f"llm_alignment/runs/result_ppo_k{k}_{{mode}}_s{{{','.join(str(s) for s in seed_list)}}}{suffix}.json"
    for mode in ["standard", "pcz"]:
        vals = data[mode]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        reg.add(f"{prefix}_k{k}_{mode}_mean", fmt_signed(mean, decimals=3), src)
        reg.add(f"{prefix}_k{k}_{mode}_std", fmt_plain(std, decimals=3), src)
        reg.add(f"{prefix}_k{k}_{mode}_stat", fmt_stat(mean, std, decimals=3), src)
        reg.add(f"{prefix}_k{k}_{mode}_seeds", fmt_int(len(vals)), src)
    if data["standard"] and data["pcz"]:
        mean_std = statistics.fmean(data["standard"])
        mean_pcz = statistics.fmean(data["pcz"])
        std_std = statistics.stdev(data["standard"]) if len(data["standard"]) > 1 else 0.0
        std_pcz = statistics.stdev(data["pcz"]) if len(data["pcz"]) > 1 else 0.0
        reg.add(
            f"{prefix}_k{k}_delta",
            fmt_signed(mean_pcz - mean_std, decimals=3),
            f"PCZ mean - Standard mean (LLM PPO K={k})",
        )
        if std_pcz > 0:
            reg.add(
                f"{prefix}_k{k}_var_ratio",
                fmt_ratio((std_std / std_pcz) ** 2, decimals=1),
                f"Standard / PCZ variance ratio (LLM PPO K={k})",
            )
        # Welch p for mean difference
        try:
            from scipy import stats as _stats_llm_p

            _t_llm, _p_llm = _stats_llm_p.ttest_ind(data["pcz"], data["standard"], equal_var=False)
            reg.add(
                f"{prefix}_k{k}_welch_p",
                fmt_plain(float(_p_llm), decimals=3),
                f"Welch p (LLM PPO K={k})",
            )
        except Exception:
            pass


def _emit_k_inference(reg: Registry, rows: list[dict]) -> None:
    """Emit Welch's t / Cohen's d / CI / permutation-p / Holm-corrected fragments
    for the K=4,6,8 headline family on LunarLander.

    All statistics are computed under the single chronologically-latest seed
    rule defined in ``fig_data.query``.  Seeds for K=4 additionally require
    the primary weight config ``10.00,5.00,0.50,0.50``.  The K=6 inline stats
    in ``tab:kscaling`` previously came from a different subset and were
    inconsistent with the rendered mean/std fragments; this function derives
    everything from the same dedup'd rows that feed ``k{4,6,8}_{pcz,ppo}_*``.
    """
    import numpy as np
    from scipy import stats as sp

    configs = [
        ("k4", "lunarlander", LL_PRIMARY_WEIGHTS),
        ("k6", "lunarlander-k6", LL_K6_WEIGHTS),
        ("k8", "lunarlander-k8", None),
    ]
    raw_ps: list[tuple[str, float]] = []
    stats_by_k: dict[str, dict] = {}
    for prefix, env, weights in configs:
        pcz = q_required(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env=env,
            total_timesteps=500000,
            weights=weights,
            min_seeds=10,
            label=f"{prefix}_pcz_inference",
            ent_coef_schedule="0.1:0.01",
            learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep cells at non-canonical LRs
        )
        ppo = q_required(
            rows,
            algorithm="torchrl-ppo",
            env=env,
            total_timesteps=500000,
            weights=weights,
            min_seeds=10,
            label=f"{prefix}_ppo_inference",
            ent_coef_schedule="0.1:0.01",
            learning_rate="0.0003",  # guard: canonical LR only
        )
        v_p = np.array([float(r["eval_mean"]) for r in pcz["runs"]])
        v_o = np.array([float(r["eval_mean"]) for r in ppo["runs"]])
        # Welch's t-test
        _t, p_welch = sp.ttest_ind(v_p, v_o, equal_var=False)
        # Cohen's d with pooled sample SD (ddof=1)
        s_p = v_p.std(ddof=1)
        s_o = v_o.std(ddof=1)
        d = float((v_p.mean() - v_o.mean()) / np.sqrt((s_p**2 + s_o**2) / 2.0))
        # Bootstrap 95% CI on the mean difference (10 000 resamples, fixed seed)
        rng = np.random.default_rng(20260416)
        deltas = np.array(
            [
                rng.choice(v_p, size=len(v_p), replace=True).mean()
                - rng.choice(v_o, size=len(v_o), replace=True).mean()
                for _ in range(10000)
            ]
        )
        ci_lo, ci_hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))

        # BCa 95% CI on the mean difference (10 000 resamples, fixed seed).
        # Percentile CI undercovers at n=10 with heavy tails (K=8 especially)
        # — BCa corrects for bias and acceleration per Efron (1987).  See
        # BCa is appropriate for small-n heavy-tailed distributions (Efron 1987).
        def _delta_stat(p_sample, o_sample, axis=-1):
            return np.mean(p_sample, axis=axis) - np.mean(o_sample, axis=axis)

        bca_res = sp.bootstrap(
            (v_p, v_o),
            _delta_stat,
            n_resamples=10000,
            method="BCa",
            confidence_level=0.95,
            rng=np.random.default_rng(20260416),
            vectorized=True,
        )
        bca_lo = float(bca_res.confidence_interval.low)
        bca_hi = float(bca_res.confidence_interval.high)

        # IQM (interquartile mean) with BCa bootstrap CI via scipy.  We use
        # scipy rather than rliable here because rliable's ``random_state``
        # does not fully seed its internal bootstrap (three calls with the
        # same seed return different CIs to the 2nd decimal; see FutureWarning
        # in arch.bootstrap).  At n=10 IQM drops the two most extreme scores
        # and averages the middle 6 — resistant to single-seed outliers that
        # flip the mean.  See Agarwal et al. 2021 for the methodology.
        def _iqm(x, axis=-1):
            lo = np.quantile(x, 0.25, axis=axis, keepdims=True)
            hi = np.quantile(x, 0.75, axis=axis, keepdims=True)
            mask = (x >= lo) & (x <= hi)
            x_masked = np.where(mask, x, np.nan)
            return np.nanmean(x_masked, axis=axis)

        pcz_iqm = float(_iqm(v_p))
        ppo_iqm = float(_iqm(v_o))
        pcz_iqm_res = sp.bootstrap(
            (v_p,),
            _iqm,
            n_resamples=10000,
            method="BCa",
            confidence_level=0.95,
            rng=np.random.default_rng(20260416),
            vectorized=True,
        )
        ppo_iqm_res = sp.bootstrap(
            (v_o,),
            _iqm,
            n_resamples=10000,
            method="BCa",
            confidence_level=0.95,
            rng=np.random.default_rng(20260416),
            vectorized=True,
        )
        pcz_iqm_lo = float(pcz_iqm_res.confidence_interval.low)
        pcz_iqm_hi = float(pcz_iqm_res.confidence_interval.high)
        ppo_iqm_lo = float(ppo_iqm_res.confidence_interval.low)
        ppo_iqm_hi = float(ppo_iqm_res.confidence_interval.high)
        # Probability of superiority (A12 / stochastic dominance) and its
        # Mann–Whitney U two-sided p-value.  A12 = P(X > Y) over all pairs,
        # with 0.5 weight on ties.  Reported as a formal test (previously
        # formally tested rather than read off the IQM figure).
        n_pairs = len(v_p) * len(v_o)
        n_gt = int(np.sum(v_p[:, None] > v_o[None, :]))
        n_eq = int(np.sum(v_p[:, None] == v_o[None, :]))
        prob_sup = (n_gt + 0.5 * n_eq) / n_pairs
        _u_stat, p_mwu = sp.mannwhitneyu(v_p, v_o, alternative="two-sided")
        # Exact two-sided permutation test (20 000 perms, fixed seed)
        obs = abs(v_p.mean() - v_o.mean())
        pool = np.concatenate([v_p, v_o])
        rng2 = np.random.default_rng(20260416)
        n_exceed = 0
        N_PERM = 20000
        for _ in range(N_PERM):
            rng2.shuffle(pool)
            if abs(pool[: len(v_p)].mean() - pool[len(v_p) :].mean()) >= obs:
                n_exceed += 1
        p_perm = (n_exceed + 1) / (N_PERM + 1)

        src = f"env={env} ts=500000 weights={weights or '*'} rule=chrono-latest"
        reg.add(f"{prefix}_welch_p", fmt_ratio(float(p_welch), decimals=3), src)
        reg.add(f"{prefix}_welch_d", fmt_ratio(d, decimals=2), src)
        reg.add(
            f"{prefix}_welch_ci",
            f"[{fmt_signed(ci_lo, 1)}, {fmt_signed(ci_hi, 1)}]",
            src,
        )
        reg.add(
            f"{prefix}_bca_ci",
            f"[{fmt_signed(bca_lo, 1)}, {fmt_signed(bca_hi, 1)}]",
            src,
        )
        reg.add(
            f"{prefix}_pcz_iqm_ci",
            f"[{fmt_signed(pcz_iqm_lo, 1)}, {fmt_signed(pcz_iqm_hi, 1)}]",
            src,
        )
        reg.add(f"{prefix}_pcz_iqm", fmt_signed(pcz_iqm, 1), src)
        reg.add(
            f"{prefix}_ppo_iqm_ci",
            f"[{fmt_signed(ppo_iqm_lo, 1)}, {fmt_signed(ppo_iqm_hi, 1)}]",
            src,
        )
        reg.add(f"{prefix}_ppo_iqm", fmt_signed(ppo_iqm, 1), src)
        reg.add(f"{prefix}_prob_sup", fmt_ratio(prob_sup * 100, decimals=0) + "\\%", src)
        reg.add(f"{prefix}_mwu_p", fmt_ratio(float(p_mwu), decimals=3), src)
        reg.add(f"{prefix}_perm_p", fmt_ratio(p_perm, decimals=3), src)
        raw_ps.append((prefix, float(p_welch)))
        stats_by_k[prefix] = {"p_welch": float(p_welch), "d": d, "n_p": len(v_p), "n_o": len(v_o)}

    # Holm–Bonferroni step-down across the three headline K-scaling tests
    sorted_ps = sorted(raw_ps, key=lambda x: x[1])
    m = len(sorted_ps)
    for i, (prefix, p) in enumerate(sorted_ps):
        p_holm = min(1.0, p * (m - i))
        reg.add(
            f"{prefix}_holm_p",
            fmt_ratio(p_holm, decimals=3),
            f"Holm–Bonferroni across {{{','.join(p for p, _ in sorted_ps)}}} family of m={m}",
        )


def _emit_k2_basic_inference(reg: Registry, rows: list[dict]) -> None:
    """Emit k2_welch_p and k2_welch_d at n=5.

    Both arms use the canonical paired weight "10.00,6.00" (Apr-11/13 batch)
    so the comparison is balanced.  Earlier versions used weights=None which
    caused PPO to resolve to equal-weight Apr-18 runs (PCZ had no equal-weight
    data), inflating the spurious PCZ/PPO ratio from 0.83→1.53 and flipping d
    from -0.83→+1.01.  This is the correct paired comparison; d is negative
    (PPO wins at K=2, consistent with the homogeneous-component hypothesis).
    """
    import numpy as np
    from scipy import stats as sp

    _K2_WEIGHTS = "10.00,6.00"
    pcz = q_required(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander-k2",
        total_timesteps=500000,
        weights=_K2_WEIGHTS,
        min_seeds=5,
        label="k2_pcz_inference",
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR — prevents HP-sweep seeds from displacing canonical
    )
    ppo = q_required(
        rows,
        algorithm="torchrl-ppo",
        env="lunarlander-k2",
        total_timesteps=500000,
        weights=_K2_WEIGHTS,
        min_seeds=5,
        label="k2_ppo_inference",
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR only
    )
    v_p = np.array([float(r["eval_mean"]) for r in pcz["runs"]])
    v_o = np.array([float(r["eval_mean"]) for r in ppo["runs"]])
    _t, p_welch = sp.ttest_ind(v_p, v_o, equal_var=False)
    s_p = v_p.std(ddof=1)
    s_o = v_o.std(ddof=1)
    d = float((v_p.mean() - v_o.mean()) / np.sqrt((s_p**2 + s_o**2) / 2.0))
    src = "env=lunarlander-k2 ts=500000 weights=* rule=chrono-latest n=5"
    reg.add("k2_welch_p", fmt_ratio(float(p_welch), decimals=3), src)
    reg.add("k2_welch_d", fmt_ratio(d, decimals=2), src)


def _emit_rr4_noise_amp(reg: Registry) -> None:
    """Emit noise-amplification fragments from the signal-tier experiment.

    Measures σ of the scalar reward *after normalization* that enters GAE
    under three variants on identical seed+env (LunarLander K={4,6,8}, seed
    42, 50k frames, canonical per-K weights, ent_coef cosine 0.1→0.01):
      - torchrl-pcz-ppo-running  (per-component z-norm + weighted sum)
      - torchrl-ppo-znorm        (aggregate scalar z-norm: σ=1 by construction)
      - torchrl-ppo              (raw, no reward normalization)

    Reports the final-3 batches (so EMA has warmed up) and the PCZ/znorm
    amplification ratio that the paper's §3 mechanism claim asserts.  The
    K-scaling extension tests whether the amplification factor is K-invariant
    (prediction: ratio ≈ sqrt(Σw_i²) × correlation inflation; theoretical bound
    is K-invariant at ~11, observed growth reflects component correlation).
    Data from ``artifacts/pcz-ppo/experiments/E46.1_noise_amp/``.
    """
    import json as _json

    path = _PAPER_DIR.parent / "experiments" / "E46.1_noise_amp" / "eval" / "eval_summary.json"
    if not path.is_file():
        return  # artifact not present; fragment is optional
    with open(path) as f:
        data = _json.load(f)

    # Support both legacy flat keys (algorithm name) and K-keyed schema
    # (K{N}_{algorithm}).  Emit backward-compat "rr4_*" for K=4 always, plus
    # K-specific fragments where data is present.
    for K in (4, 6, 8):
        pcz_key = f"K{K}_torchrl-pcz-ppo-running"
        znm_key = f"K{K}_torchrl-ppo-znorm"
        raw_key = f"K{K}_torchrl-ppo"
        pcz = data.get(pcz_key)
        znm = data.get(znm_key)
        raw = data.get(raw_key)
        if not (pcz and znm and raw):
            # Legacy fallback for K=4 only
            if K == 4:
                pcz = data.get("torchrl-pcz-ppo-running")
                znm = data.get("torchrl-ppo-znorm")
                raw = data.get("torchrl-ppo")
            if not (pcz and znm and raw):
                continue
        src = f"noise-amp: LL K={K} s42 50k frames canonical weights"
        prefix = f"rr4_k{K}"
        reg.add(f"{prefix}_pcz_postnorm_std", fmt_ratio(pcz["postnorm_std_final3"], decimals=1), src)
        reg.add(f"{prefix}_znorm_postnorm_std", fmt_ratio(znm["postnorm_std_final3"], decimals=1), src)
        reg.add(f"{prefix}_raw_postnorm_std", fmt_ratio(raw["postnorm_std_final3"], decimals=1), src)
        pcz_v = pcz["postnorm_std_final3"]
        znm_v = znm["postnorm_std_final3"]
        raw_v = raw["postnorm_std_final3"]
        if znm_v > 0:
            reg.add(f"{prefix}_pcz_over_znorm_ratio", fmt_ratio(pcz_v / znm_v, decimals=1), src)
        if pcz_v > 0:
            reg.add(f"{prefix}_raw_over_pcz_ratio", fmt_ratio(raw_v / pcz_v, decimals=2), src)
        # Backward-compat aliases for K=4 (no K prefix) used by §3 paragraph
        if K == 4:
            reg.add("rr4_pcz_postnorm_std", fmt_ratio(pcz_v, decimals=1), src)
            reg.add("rr4_znorm_postnorm_std", fmt_ratio(znm_v, decimals=1), src)
            reg.add("rr4_raw_postnorm_std", fmt_ratio(raw_v, decimals=1), src)
            if znm_v > 0:
                reg.add("rr4_pcz_over_znorm_ratio", fmt_ratio(pcz_v / znm_v, decimals=1), src)
            if pcz_v > 0:
                reg.add("rr4_raw_over_pcz_ratio", fmt_ratio(raw_v / pcz_v, decimals=2), src)


def _emit_ca11_weight_sensitivity(reg: Registry, rows: list[dict]) -> None:
    """Emit weight-config sensitivity fragments (LL K=4 500k, n=5).

    Shows that the PCZ advantage is monotone in weight heterogeneity:
      canonical (10,5,0.5,0.5) d=+1.04 (paper headline)
      moderate (5,3,1,1) d=+0.09 (tied)
      uniform (3,3,3,3) d=-7.04 (PPO dominates)
      uniform-unit (1,1,1,1) d=-3.45 (PPO dominates)

    Emits for each non-canonical config:
      ca11_{tag}_{pcz,ppo}_{mean,std,seeds,stat} + _delta/_ratio/_var_ratio
      ca11_{tag}_cohen_d, _welch_p
    """
    import numpy as np
    from scipy import stats as _stats

    specs = [
        ("ca11_w5311", K4_WEIGHTS_MODERATE),
        ("ca11_w3333", K4_WEIGHTS_FLAT),
        ("ca11_w1111", K4_WEIGHTS_EQUAL),
    ]
    for prefix, weights in specs:
        _emit_pcz_ppo_pair(reg, rows, prefix, env="lunarlander", total_timesteps=500000, weights=weights, min_seeds=5)
        # Compute d + Welch p from the same dedup'd rows.  query() is the
        # single source of truth; we pull per-seed evals here for the test.
        pcz_q = query(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env="lunarlander",
            total_timesteps=500000,
            weights=weights,
        )
        ppo_q = query(
            rows,
            algorithm="torchrl-ppo",
            env="lunarlander",
            total_timesteps=500000,
            weights=weights,
        )
        pcz_vals = np.asarray([float(r["eval_mean"]) for r in pcz_q["runs"]], dtype=float)
        ppo_vals = np.asarray([float(r["eval_mean"]) for r in ppo_q["runs"]], dtype=float)
        if len(pcz_vals) < 2 or len(ppo_vals) < 2:
            continue
        pcz_sd = pcz_vals.std(ddof=1)
        ppo_sd = ppo_vals.std(ddof=1)
        pooled = ((len(pcz_vals) - 1) * pcz_sd**2 + (len(ppo_vals) - 1) * ppo_sd**2) / (
            len(pcz_vals) + len(ppo_vals) - 2
        )
        d = (pcz_vals.mean() - ppo_vals.mean()) / float(np.sqrt(pooled)) if pooled > 0 else 0.0
        _t, p = _stats.ttest_ind(pcz_vals, ppo_vals, equal_var=False)
        src = f"weight-sensitivity weights={weights} n={len(pcz_vals)}+{len(ppo_vals)}"
        reg.add(f"{prefix}_cohen_d", fmt_signed(d, decimals=2), src)
        reg.add(f"{prefix}_welch_p", fmt_plain(float(p), decimals=3), src)


def _emit_w7_baseline_audit(reg: Registry, rows: list[dict]) -> None:
    """Emit PPO baseline-strength audit fragments (LL K=4 500k, n=2 each cell).

    Paper-framing: the canonical comparison uses cosine entropy schedule 0.1→0.01
    for BOTH PPO and PCZ-PPO.  This audit asks "what if we give PPO its own best
    fixed-entropy tuning?" and answers: PPO's peak config is lr=3e-4 ent=0.0.  At that
    config, PPO achieves 167.3 ± 15.6 (n=2) — higher than its canonical 112.0 ±
    56.2 (n=10).  PCZ-PPO at the same config achieves 180.8 ± 41.6 (n=2).  So
    even at PPO's peak tuning, PCZ-PPO matches or slightly exceeds PPO; the
    canonical-config gap (+45, n=10) narrows to +13 (n=2, peak-vs-peak).

    Emits:
      w7_ppo_best_{stat,mean,std,seeds}   — PPO at lr=3e-4 ent=0.0 (best cell)
      w7_pcz_best_{stat,mean,std,seeds}   — PCZ at lr=3e-4 ent=0.0
      w7_delta_best                        — PCZ−PPO at peak (n=2 each)
    """
    import numpy as np

    # Filter directly on rows (bypassing query's dedupe which keeps latest-per-seed
    # and would miss fixed-entropy runs that share seeds with canonical headline runs).
    # Dedupe within this lr+ent+ent_schedule-empty subset ourselves.
    def _grab(algo: str) -> tuple[list[float], list[str]]:
        matching = [
            r
            for r in rows
            if r.get("algorithm") == algo
            and r.get("env") == "lunarlander"
            and r.get("total_timesteps") == "500000"
            and r.get("learning_rate") == "0.0003"
            and r.get("ent_coef") == "0.0"
            and not r.get("ent_coef_schedule")
            and r.get("component_weights", "").startswith("10.00,5.00,0.50,0.50")
            and r.get("eval_mean")
        ]
        matching.sort(key=lambda r: r.get("date", ""))
        by_seed: dict[str, dict] = {}
        for r in matching:
            by_seed[r["seed"]] = r
        vals = [float(r["eval_mean"]) for r in by_seed.values()]
        seeds = [r["seed"] for r in by_seed.values()]
        return vals, seeds

    for tag, algo in [("ppo", "torchrl-ppo"), ("pcz", "torchrl-pcz-ppo-running")]:
        vals, _seeds = _grab(algo)
        if len(vals) < 2:
            continue
        arr = np.asarray(vals, dtype=float)
        src = f"PPO-baseline-audit {algo} lr=3e-4 ent=0.0 weights={LL_PRIMARY_WEIGHTS} n={len(arr)}"
        reg.add(f"w7_{tag}_best_mean", fmt_signed(float(arr.mean())), src)
        reg.add(f"w7_{tag}_best_std", fmt_plain(float(arr.std(ddof=1))), src)
        reg.add(f"w7_{tag}_best_seeds", fmt_int(len(arr)), src)
        reg.add(f"w7_{tag}_best_stat", fmt_stat(float(arr.mean()), float(arr.std(ddof=1))), src)

    # Delta (peak-vs-peak)
    pcz_vals, _ = _grab("torchrl-pcz-ppo-running")
    ppo_vals, _ = _grab("torchrl-ppo")
    if len(pcz_vals) >= 2 and len(ppo_vals) >= 2:
        delta = float(np.mean(pcz_vals)) - float(np.mean(ppo_vals))
        reg.add(
            "w7_delta_best",
            fmt_signed(delta),
            "PPO-baseline-audit peak-vs-peak PCZ−PPO at lr=3e-4 ent=0.0",
        )


def _emit_rr6_entropy_sweep(reg: Registry, rows: list[dict]) -> None:
    """Emit entropy-coupling sweep fragments (LL K=4 500k, lr=3e-4).

    Paired PCZ and PPO runs at five fixed entropy coefficients (no cosine schedule)
    to measure the tuning envelope.  Finding: PCZ tolerates entropy up to 0.05 and
    collapses at 0.10; PPO starts collapsing at 0.03.  ent=0.03 is the sharpest
    differential cell — both at n=3, PCZ=145.7 vs PPO=21.3 (delta +124.4).

    Emits per (tag, ent):
      rr6_{pcz|ppo}_e{00|01|03|05|10}_{stat,mean,std,seeds}
      rr6_delta_e{00|01|03|05|10}   (PCZ−PPO)
    """
    import numpy as np

    def _grab(algo: str, ent: str) -> list[float]:
        matching = [
            r
            for r in rows
            if r.get("algorithm") == algo
            and r.get("env") == "lunarlander"
            and r.get("total_timesteps") == "500000"
            and r.get("learning_rate") == "0.0003"
            and r.get("ent_coef") == ent
            and not r.get("ent_coef_schedule")
            and r.get("component_weights", "").startswith("10.00,5.00,0.50,0.50")
            and r.get("eval_mean")
        ]
        matching.sort(key=lambda r: r.get("date", ""))
        by_seed: dict[str, dict] = {}
        for r in matching:
            by_seed[r["seed"]] = r
        return [float(r["eval_mean"]) for r in by_seed.values()]

    ent_pairs = [("00", "0.0"), ("01", "0.01"), ("03", "0.03"), ("05", "0.05"), ("10", "0.1")]
    for slug, ent in ent_pairs:
        for tag, algo in [("ppo", "torchrl-ppo"), ("pcz", "torchrl-pcz-ppo-running")]:
            vals = _grab(algo, ent)
            if len(vals) < 2:
                continue
            arr = np.asarray(vals, dtype=float)
            src = f"entropy-sweep {algo} lr=3e-4 ent={ent} n={len(arr)}"
            reg.add(f"rr6_{tag}_e{slug}_mean", fmt_signed(float(arr.mean())), src)
            reg.add(f"rr6_{tag}_e{slug}_std", fmt_plain(float(arr.std(ddof=1))), src)
            reg.add(f"rr6_{tag}_e{slug}_seeds", fmt_int(len(arr)), src)
            reg.add(f"rr6_{tag}_e{slug}_stat", fmt_stat(float(arr.mean()), float(arr.std(ddof=1))), src)
        pcz_vals = _grab("torchrl-pcz-ppo-running", ent)
        ppo_vals = _grab("torchrl-ppo", ent)
        if len(pcz_vals) >= 2 and len(ppo_vals) >= 2:
            delta = float(np.mean(pcz_vals)) - float(np.mean(ppo_vals))
            reg.add(
                f"rr6_delta_e{slug}",
                fmt_signed(delta),
                f"entropy-sweep PCZ−PPO at lr=3e-4 ent={ent}",
            )


def _emit_a15_vs_a1_k(
    reg: Registry,
    rows: list[dict],
    prefix: str,
    *,
    env: str,
    weights: str | None,
    min_seeds: int = 10,
) -> None:
    """Emit A15 (symznorm) vs A1 (running) inferential fragments at one K.

    Generalised to K=4 (champion arbitration), K=6, and K=8. Fragments:
      ``{prefix}_delta, _welch_p, _welch_d, _bca_ci, _paired_p, _var_ratio``
    """
    import numpy as np
    from scipy import stats as sp

    a15 = q_required(
        rows,
        algorithm="torchrl-pcz-ppo-symznorm",
        env=env,
        total_timesteps=500000,
        weights=weights,
        min_seeds=min_seeds,
        label=f"{prefix}_A15",
        ent_coef_schedule="0.1:0.01",
    )
    a1 = q_required(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env=env,
        total_timesteps=500000,
        weights=weights,
        min_seeds=min_seeds,
        label=f"{prefix}_A1",
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
    )
    by_seed_a15 = {r["seed"]: float(r["eval_mean"]) for r in a15["runs"]}
    by_seed_a1 = {r["seed"]: float(r["eval_mean"]) for r in a1["runs"]}
    common = sorted(set(by_seed_a15) & set(by_seed_a1), key=lambda s: int(s))
    v15 = np.array([by_seed_a15[s] for s in common])
    v1 = np.array([by_seed_a1[s] for s in common])
    v15_all = np.array([float(r["eval_mean"]) for r in a15["runs"]])
    v1_all = np.array([float(r["eval_mean"]) for r in a1["runs"]])

    delta = float(v15_all.mean() - v1_all.mean())
    _t, p_welch = sp.ttest_ind(v15_all, v1_all, equal_var=False)
    s15 = v15_all.std(ddof=1)
    s1 = v1_all.std(ddof=1)
    d = float(delta / np.sqrt((s15**2 + s1**2) / 2.0))
    var_ratio = float((s1 / s15) ** 2) if s15 > 0 else 0.0

    # Paired BCa CI on the per-seed difference (shared seeds only)
    rng = np.random.default_rng(20260418)
    if len(common) >= 2:
        pair_diff = v15 - v1
        bca = sp.bootstrap(
            (pair_diff,),
            np.mean,
            n_resamples=10000,
            method="BCa",
            random_state=rng,
            confidence_level=0.95,
        )
        ci_lo = float(bca.confidence_interval.low)
        ci_hi = float(bca.confidence_interval.high)
        _tp, p_paired = sp.ttest_rel(v15, v1)
    else:
        ci_lo = ci_hi = 0.0
        p_paired = 1.0

    src = f"A15 vs A1 {env} 500k n={len(v15_all)}/{len(v1_all)}, paired seeds={len(common)}, weights={weights}"
    reg.add(f"{prefix}_delta", fmt_signed(delta), src)
    reg.add(f"{prefix}_welch_p", fmt_ratio(float(p_welch), decimals=3), src)
    reg.add(f"{prefix}_welch_d", fmt_ratio(d, decimals=2), src)
    reg.add(f"{prefix}_paired_p", fmt_ratio(float(p_paired), decimals=3), src)
    reg.add(f"{prefix}_var_ratio", fmt_ratio(var_ratio, decimals=2), src)
    reg.add(
        f"{prefix}_bca_ci",
        f"[{fmt_signed(ci_lo)}, {fmt_signed(ci_hi)}]",
        src,
    )


def _emit_component_stats(reg: Registry, rows: list[dict]) -> None:
    """Emit run-level σ/|μ| fragments for Table 12 (tab:app_comp_stats).

    The table's historical per-step column was hand-typed from a mixed subset
    and reviewers flagged a discrepancy (BW ``crash`` was reported at
    per-step σ/|μ| = 52.6×, while a reviewer's run-level recomputation gave
    14.2×).  We add a run-level column alongside the per-step column and pin
    the run-level values to ``results.csv`` so they cannot drift.

    Subset: all runs with non-empty comp/{name}_mean column for the (env, comp)
    pair.  No algorithm or weight filter — component magnitudes are an
    environmental property and should be reported across the full dataset.
    """
    import statistics as st

    # (env, comp, frag_env_tag) — frag_env_tag keeps frag names short
    comps = [
        ("lunarlander", "landing", "ll"),
        ("lunarlander", "shaping", "ll"),
        ("lunarlander", "fuel_main", "ll"),
        ("lunarlander", "fuel_side", "ll"),
        ("bipedalwalker", "shaping", "bw"),
        ("bipedalwalker", "energy", "bw"),
        ("bipedalwalker", "crash", "bw"),
        ("halfcheetah", "run", "hc"),
        ("halfcheetah", "ctrl_cost", "hc"),
    ]
    for env, comp, tag in comps:
        means = []
        for r in rows:
            if r.get("env") != env:
                continue
            m = r.get(f"comp/{comp}_mean", "").strip()
            if not m:
                continue
            try:
                means.append(float(m))
            except ValueError:
                continue
        if len(means) < 2:
            continue
        mu_rl = st.fmean(means)
        sig_rl = st.stdev(means)
        if mu_rl == 0:
            continue
        cv_rl = abs(sig_rl / mu_rl)
        src = f"results.csv comp/{comp}_mean across {len(means)} runs on env={env}"
        # Render as `X.YY` so the table just says "2.27" with the ×-suffix in LaTeX
        reg.add(f"compstat_{tag}_{comp}_cv_rl", fmt_ratio(cv_rl, decimals=2), src)


def _emit_hp_sweep_claims(
    reg: Registry,
    rows: list[dict],
    *,
    env: str,
    total_timesteps: int,
    prefix: str,
    weight_tiers: dict[str, str],
    lrs: list[str] = ("0.0001", "0.0003", "0.001"),
    min_n: int = 2,
) -> None:
    """Emit HP-fairness sweep claims: best cell per algo and peak delta.

    Fragment naming:
      {prefix}_pcz_best_{mean,std,seeds,stat,tier,lr,ent}
      {prefix}_ppo_best_{mean,std,seeds,stat,tier,lr,ent}
      {prefix}_peak_delta   (PCZ-best − PPO-best means)
      {prefix}_peak_a12     (Vargha-Delaney A12, PCZ-best vs PPO-best cells)
      {prefix}_ncells_{pcz,ppo}  (n cells with n>=min_n)

    Skips silently if insufficient data for either algo (fragment stays
    unreferenced until data arrives, matching the existing no-op pattern
    for optional extensions above).
    """
    import itertools

    import numpy as np

    ts_str = str(total_timesteps)
    seeds = {"42", "43", "44"}

    # ent-config filter specs: (slug, sched_match, coef_match)
    # cosine matches on schedule only; fixed variants match coef with empty sched.
    ent_configs = [
        ("cosine", "0.1:0.01", None),
        ("e00", "", "0.0"),
        ("e01", "", "0.01"),
    ]

    def _collect_cell(
        algo: str, w_csv: str, lr: str, ent_slug: str, ent_sched: str, ent_coef: str | None
    ) -> list[float]:
        vals: dict[str, float] = {}
        for r in rows:
            if r.get("algorithm") != algo:
                continue
            if r.get("env") != env:
                continue
            if r.get("total_timesteps") != ts_str:
                continue
            if r.get("seed") not in seeds:
                continue
            if not r.get("eval_mean"):
                continue
            # weight match
            if w_csv:
                if not r.get("component_weights", "").startswith(w_csv):
                    continue
            # lr match
            if r.get("learning_rate") != lr:
                continue
            # ent match
            r_sched = r.get("ent_coef_schedule", "")
            r_coef = r.get("ent_coef", "")
            if ent_sched:  # cosine: require matching schedule
                if r_sched != ent_sched:
                    continue
            else:  # fixed: require empty schedule + exact coef
                if r_sched:
                    continue
                if ent_coef is not None and r_coef != ent_coef:
                    continue
            seed = r["seed"]
            # latest-seed-first dedupe (consistent with query())
            if seed not in vals or r.get("date", "") > vals.get(f"_date_{seed}", ""):
                vals[seed] = float(r["eval_mean"])
                vals[f"_date_{seed}"] = r.get("date", "")
        return [v for k, v in vals.items() if not k.startswith("_date_")]

    def _a12(x: list[float], y: list[float]) -> float:
        """Vargha-Delaney A12: P(X > Y) + 0.5*P(X == Y)."""
        wins = sum(1 for xi, yi in itertools.product(x, y) if xi > yi)
        ties = sum(1 for xi, yi in itertools.product(x, y) if xi == yi)
        return (wins + 0.5 * ties) / (len(x) * len(y))

    algo_specs = [
        ("ppo", "torchrl-ppo"),
        ("pcz", "torchrl-pcz-ppo-running"),
    ]

    best_per_algo: dict[str, dict] = {}
    for algo_slug, algo in algo_specs:
        best_mean = float("-inf")
        best_vals: list[float] = []
        best_cell_label = ("?", "?", "?")
        n_cells = 0
        for tier_name, w_csv in weight_tiers.items():
            for lr in lrs:
                for ent_slug, ent_sched, ent_coef in ent_configs:
                    vals = _collect_cell(algo, w_csv, lr, ent_slug, ent_sched, ent_coef)
                    if len(vals) < min_n:
                        continue
                    n_cells += 1
                    m = float(np.mean(vals))
                    if m > best_mean:
                        best_mean = m
                        best_vals = vals
                        best_cell_label = (tier_name, lr, ent_slug)
        best_per_algo[algo_slug] = {
            "vals": best_vals,
            "label": best_cell_label,
            "n_cells": n_cells,
        }

    # Require both algos to have data before emitting anything
    for slug in ("ppo", "pcz"):
        if not best_per_algo[slug]["vals"]:
            return

    src_base = f"HP-fairness sweep env={env} ts={total_timesteps}"
    for slug in ("ppo", "pcz"):
        vals = np.asarray(best_per_algo[slug]["vals"], dtype=float)
        tier, lr, ent = best_per_algo[slug]["label"]
        n_cells = best_per_algo[slug]["n_cells"]
        src = f"{src_base} {slug} best={tier}/{lr}/{ent} n={len(vals)}"
        reg.add(f"{prefix}_{slug}_best_mean", fmt_signed(float(vals.mean())), src)
        reg.add(f"{prefix}_{slug}_best_std", fmt_plain(float(vals.std(ddof=max(1, len(vals) - 1)))), src)
        reg.add(f"{prefix}_{slug}_best_seeds", fmt_int(len(vals)), src)
        reg.add(
            f"{prefix}_{slug}_best_stat", fmt_stat(float(vals.mean()), float(vals.std(ddof=max(1, len(vals) - 1)))), src
        )
        reg.add(f"{prefix}_{slug}_best_tier", tier, src)
        reg.add(f"{prefix}_{slug}_best_lr", lr, src)
        reg.add(f"{prefix}_{slug}_best_ent", ent, src)
        reg.add(f"{prefix}_ncells_{slug}", fmt_int(n_cells), src)

    pcz_vals = best_per_algo["pcz"]["vals"]
    ppo_vals = best_per_algo["ppo"]["vals"]
    delta = float(np.mean(pcz_vals)) - float(np.mean(ppo_vals))
    reg.add(f"{prefix}_peak_delta", fmt_signed(delta), f"{src_base} peak_delta PCZ−PPO")
    a12 = _a12(pcz_vals, ppo_vals)
    reg.add(f"{prefix}_peak_a12", fmt_plain(a12, decimals=2), f"{src_base} Vargha-Delaney A12 PCZ>PPO")


def build_registry(rows: list[dict], used_filter: set[str] | None = None) -> Registry:
    reg = Registry(used_filter=used_filter)

    # --- K-scaling table (tab:kscaling) ---
    # K=2 / K=4 / K=6 / K=8 on LunarLander family.
    # ``ent_coef_schedule="0.1:0.01"`` isolates canonical cosine-schedule runs
    # so that fixed-entropy tuning audits (which share seeds 42-46 with
    # the headline) don't win the chronological dedupe.
    _emit_pcz_ppo_pair(
        reg,
        rows,
        "k2",
        env="lunarlander-k2",
        total_timesteps=500000,
        weights="10.00,6.00",  # explicit filter — equal-weight PPO runs (Apr-18) must not override paired runs
        min_seeds=5,
        ent_coef_schedule="0.1:0.01",
    )
    _emit_pcz_ppo_pair(
        reg,
        rows,
        "k4",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        min_seeds=10,
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR — prevents HP-sweep cells from displacing canonical seeds
    )
    _emit_pcz_ppo_pair(
        reg,
        rows,
        "k6",
        env="lunarlander-k6",
        total_timesteps=500000,
        weights=LL_K6_WEIGHTS,
        min_seeds=10,
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR — prevents future HP-sweep cells at K=6 from contaminating
    )
    _emit_pcz_ppo_pair(
        reg,
        rows,
        "k8",
        env="lunarlander-k8",
        total_timesteps=500000,
        weights=None,
        min_seeds=10,
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR — prevents future HP-sweep cells at K=8 from contaminating
    )
    # Inferential statistics (Welch, Cohen's d, bootstrap CI, permutation,
    # Holm) derived from the same dedup'd rows used above; see §A.3 of the paper.
    _emit_k_inference(reg, rows)
    # K=2 basic inference (Welch p, Cohen's d) at n=5.  Kept separate from
    # the Holm family (K=4,6,8) because K=2 is a supporting-evidence row,
    # not a headline claim; emitting only welch_p + welch_d keeps the
    # negative-control cell in Table 3 fragment-backed without inflating
    # the multiple-comparison family.
    _emit_k2_basic_inference(reg, rows)

    # --- A15 vs A1 n=10 inferential fragments ---
    # K=4 (the original champion-arbitration call)
    _emit_a15_vs_a1_k(reg, rows, "abl_a15_over_a1", env="lunarlander", weights=LL_PRIMARY_WEIGHTS)
    # K=6 and K=8 (follow-up; A15 has different behaviour at higher K)
    try:
        _emit_a15_vs_a1_k(reg, rows, "abl_a15_over_a1_k6", env="lunarlander-k6", weights=LL_K6_WEIGHTS, min_seeds=7)
    except Exception as exc:
        import warnings

        warnings.warn(f"A15 vs A1 K=6 skipped: {exc}", stacklevel=2)
    try:
        _emit_a15_vs_a1_k(reg, rows, "abl_a15_over_a1_k8", env="lunarlander-k8", weights=None, min_seeds=10)
    except Exception as exc:
        import warnings

        warnings.warn(f"A15 vs A1 K=8 skipped: {exc}", stacklevel=2)

    # --- Appendix component-statistics table (tab:app_comp_stats) run-level column ---
    _emit_component_stats(reg, rows)

    # --- Noise-amplification signal-tier measurement ---
    _emit_rr4_noise_amp(reg)

    # --- Weight-config sensitivity (LL K=4 500k, n=5) ---
    _emit_ca11_weight_sensitivity(reg, rows)

    # --- PPO baseline-strength audit (LL K=4 500k) ---
    _emit_w7_baseline_audit(reg, rows)

    # --- Entropy-coupling sweep (LL K=4 500k, lr=3e-4) ---
    _emit_rr6_entropy_sweep(reg, rows)

    # --- HP-fairness sweep (K=4/500k; K=8/4M added when data arrives) ---
    # No-ops until enough cells have n>=2; fragments become active as sweep fills.
    # Global best across all tiers (PCZ best = heterog, PPO best = flat at current data).
    _emit_hp_sweep_claims(
        reg,
        rows,
        env="lunarlander",
        total_timesteps=500000,
        prefix="hp19_k4",
        weight_tiers={
            "heterog": "10.00,5.00,0.50,0.50",
            "moderate": "5.00,3.00,1.00,1.00",
            "flat": "3.00,3.00,3.00,3.00",
            "equal": "1.00,1.00,1.00,1.00",
        },
    )
    # Per-tier: canonical heterog tier specifically (full HP search, n=3 per cell).
    _emit_hp_sweep_claims(
        reg,
        rows,
        env="lunarlander",
        total_timesteps=500000,
        prefix="hp19_k4_heterog",
        weight_tiers={"heterog": "10.00,5.00,0.50,0.50"},
    )
    # Per-tier: moderate (CA11 analog).
    _emit_hp_sweep_claims(
        reg,
        rows,
        env="lunarlander",
        total_timesteps=500000,
        prefix="hp19_k4_mod",
        weight_tiers={"moderate": "5.00,3.00,1.00,1.00"},
    )
    # Per-tier: flat (confirms Table 9 catastrophic failure with HP search).
    _emit_hp_sweep_claims(
        reg,
        rows,
        env="lunarlander",
        total_timesteps=500000,
        prefix="hp19_k4_flat",
        weight_tiers={"flat": "3.00,3.00,3.00,3.00"},
    )
    # Per-tier: equal (same failure mode, equal weights).
    _emit_hp_sweep_claims(
        reg,
        rows,
        env="lunarlander",
        total_timesteps=500000,
        prefix="hp19_k4_equal",
        weight_tiers={"equal": "1.00,1.00,1.00,1.00"},
    )
    # K=8/4M: emits once K=8/4M HP-sweep data lands; no-op until then.
    _emit_hp_sweep_claims(
        reg,
        rows,
        env="lunarlander-k8",
        total_timesteps=4000000,
        prefix="hp19_k8",
        weight_tiers={
            "heterog": "10.00,1.00,1.00,1.00,1.00,1.00,0.50,0.50",
            "moderate": "5.00,3.00,1.00,1.00,1.00,1.00,0.50,0.50",
            "flat": "3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00",
            "equal": "1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00",
        },
    )

    # --- Ablation table (tab:ablation) on LunarLander K=4, primary weights ---
    ablation_rows = [
        ("ablA1", "torchrl-pcz-ppo-running"),
        ("ablA5", "torchrl-ppo-znorm-post"),
        ("ablA6", "torchrl-pcz-ppo-running-no-whiten"),
        ("ablA7", "torchrl-ppo-weighted-running"),
        ("ablB1", "torchrl-ppo"),
        ("ablB2", "torchrl-ppo-no-norm"),
        ("ablB4", "torchrl-ppo-znorm"),
        ("ablC1", "torchrl-grpo"),
        ("ablC5", "torchrl-ppo-multihead"),
        ("ablGDPO", "torchrl-pcz-grpo"),
        ("ablGDPOPopArt", "torchrl-pcz-ppo-popart"),
        ("ablPopArt", "torchrl-ppo-popart"),
        ("ablA8", "torchrl-pcz-ppo-zca"),
        ("ablA9", "torchrl-pcz-ppo-log"),
        ("ablA10", "torchrl-pcz-ppo-quantile"),
        ("ablA11", "torchrl-pcz-ppo-lambdak"),
        ("ablA12", "torchrl-pcz-ppo-asym"),
        ("ablA13", "torchrl-pcz-ppo-tcheby"),
        ("ablA14", "torchrl-pcz-ppo-mi"),
        ("ablA15", "torchrl-pcz-ppo-symznorm"),
        ("ablA16", "torchrl-pcz-ppo-cosw"),
        ("ablD1", "torchrl-pcz-ppo-mc"),
    ]
    for prefix, algo in ablation_rows:
        _emit_single(
            reg,
            rows,
            prefix,
            algorithm=algo,
            env="lunarlander",
            total_timesteps=500000,
            weights=LL_PRIMARY_WEIGHTS,
            min_seeds=3,
            ent_coef_schedule="0.1:0.01",
            learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
        )

    # --- PopArt at K=6 and K=8, Multi-head at K=2/K=6 ---
    # These emit calls are no-ops until Batch A of the mega-chain completes
    # (paper prose using \cnum{ablPopArt_k6_stat} etc. will trigger the fragment).
    # Min_seeds relaxed to 3 to allow intermediate renders.
    for prefix, algo, env, wp in [
        ("ablPopArt_k6", "torchrl-ppo-popart", "lunarlander-k6", LL_K6_WEIGHTS),
        ("ablPopArt_k8", "torchrl-ppo-popart", "lunarlander-k8", None),
        ("k2_mh", "torchrl-ppo-multihead", "lunarlander-k2", None),
        ("k6_mh", "torchrl-ppo-multihead", "lunarlander-k6", LL_K6_WEIGHTS),
    ]:
        try:
            _emit_single(
                reg,
                rows,
                prefix,
                algorithm=algo,
                env=env,
                total_timesteps=500000,
                weights=wp,
                min_seeds=3,
                ent_coef_schedule="0.1:0.01",
            )
        except LookupError:
            pass  # Data not yet available; fragment stays unreferenced.
    # Derived ablation deltas used in "Key Finding" narrative cells
    # "Decomposition adds X" = A1 (PCZ running) - A7 (PPO weighted running)
    # ent_coef_schedule filter is MANDATORY: without it,
    # non-canonical-entropy A1 runs silently drag the mean (e.g. +157.7 → +72.4)
    # and the delta sign flips.
    ablA1 = q_required(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        min_seeds=10,
        label="decomp_delta_A1",
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
    )
    ablA7 = q_required(
        rows,
        algorithm="torchrl-ppo-weighted-running",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        min_seeds=3,
        label="decomp_delta_A7",
        ent_coef_schedule="0.1:0.01",
    )
    reg.add(
        "abl_decomp_delta",
        fmt_signed(ablA1["mean_raw"] - ablA7["mean_raw"]),
        "A1 (pcz-ppo-running) - A7 (ppo-weighted-running), lunarlander 500k primary weights",
    )
    # "PCZ hurts GRPO" delta = GDPO (C1 with PCZ) - GRPO (C1 no PCZ); both n=3
    ablC1 = q_required(
        rows,
        algorithm="torchrl-grpo",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        min_seeds=3,
        label="pcz_hurts_grpo_C1",
    )
    ablGDPO = q_required(
        rows,
        algorithm="torchrl-pcz-grpo",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        min_seeds=3,
        label="pcz_hurts_grpo_GDPO",
    )
    reg.add(
        "abl_pcz_hurts_grpo_delta",
        fmt_signed(ablGDPO["mean_raw"] - ablC1["mean_raw"]),
        "GDPO - GRPO (PCZ applied to GRPO, magnitude of harm)",
    )

    # ZCA variance ratio vs A1 (computed only if both present at min_seeds=3).
    # Stored under cross-context prefix `abl_zca_var_ratio` (CROSS_CONTEXT_PREFIXES).
    # Both q_required MUST filter ent_coef_schedule="0.1:0.01": without it,
    # non-canonical A1 runs blow up the std (+157.7 → +72.4 contamination) and
    # the ratio inflates 6×.
    try:
        ablA1_for_zca = q_required(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env="lunarlander",
            total_timesteps=500000,
            weights=LL_PRIMARY_WEIGHTS,
            min_seeds=3,
            label="zca_baseline_A1",
            ent_coef_schedule="0.1:0.01",
            learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
        )
        ablZCA_for_ratio = q_required(
            rows,
            algorithm="torchrl-pcz-ppo-zca",
            env="lunarlander",
            total_timesteps=500000,
            weights=LL_PRIMARY_WEIGHTS,
            min_seeds=3,
            label="zca_ratio",
            ent_coef_schedule="0.1:0.01",
        )
        if ablA1_for_zca["std_raw"] > 0 and ablZCA_for_ratio["std_raw"] > 0:
            reg.add(
                "abl_zca_var_ratio",
                fmt_ratio(ablA1_for_zca["std_raw"] / ablZCA_for_ratio["std_raw"]),
                "A1 std / ZCA std (variance reduction factor of ZCA vs A1)",
            )
    except Exception:
        pass  # ZCA may not have 3 seeds yet; skip the fragment

    # --- IQM / P(improvement) fragments ---
    # Probability of improvement: P(PCZ > PPO) via pairwise comparisons.
    # Both queries MUST filter ent_coef_schedule="0.1:0.01": without it,
    # non-canonical-entropy runs contaminate the per-K pools and silently
    # push P(PCZ>PPO) from 73% (canonical) down to 71% at K=4.
    for prefix, env, weights in [
        ("k4", "lunarlander", LL_PRIMARY_WEIGHTS),
        ("k6", "lunarlander-k6", LL_K6_WEIGHTS),
        ("k8", "lunarlander-k8", None),
    ]:
        try:
            pcz_q = q_required(
                rows,
                algorithm="torchrl-pcz-ppo-running",
                env=env,
                total_timesteps=500000,
                weights=weights,
                min_seeds=5,
                label=f"iqm_{prefix}_pcz",
                ent_coef_schedule="0.1:0.01",
                learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
            )
            ppo_q = q_required(
                rows,
                algorithm="torchrl-ppo",
                env=env,
                total_timesteps=500000,
                weights=weights,
                min_seeds=5,
                label=f"iqm_{prefix}_ppo",
                ent_coef_schedule="0.1:0.01",
                learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
            )
            pcz_e = [float(r["eval_mean"]) for r in pcz_q["runs"]]
            ppo_e = [float(r["eval_mean"]) for r in ppo_q["runs"]]
            n_better = sum(1 for p in pcz_e for q in ppo_e if p > q)
            prob = n_better / (len(pcz_e) * len(ppo_e))
            # Escape % as \% so the fragment is safe in text mode (bare `%`
            # starts a LaTeX comment).  In math mode either works.
            reg.add(f"{prefix}_prob_improvement", f"{prob * 100:.0f}\\%", f"P(PCZ > PPO) at {prefix}")
        except Exception:
            pass

    # --- Compute-matched GRPO table (tab:compute_matched) ---
    # GRPO at 500k and 1M across 4 envs
    _emit_single(
        reg, rows, "grpoK4_500k", algorithm="torchrl-grpo", env="lunarlander", total_timesteps=500000, min_seeds=3
    )
    _emit_single(
        reg, rows, "grpoK4_1M", algorithm="torchrl-grpo", env="lunarlander", total_timesteps=1000000, min_seeds=1
    )
    _emit_single(
        rows=rows,
        reg=reg,
        prefix="grpoK8_1M",
        algorithm="torchrl-grpo",
        env="lunarlander-k8",
        total_timesteps=1000000,
        min_seeds=1,
    )
    _emit_single(
        reg, rows, "grpoBW_1M", algorithm="torchrl-grpo", env="bipedalwalker", total_timesteps=1000000, min_seeds=3
    )
    _emit_single(
        reg, rows, "grpoHC_1M", algorithm="torchrl-grpo", env="halfcheetah", total_timesteps=1000000, min_seeds=3
    )

    # Compute-matched deltas = {baseline}_mean - PCZ_PPO_mean (negative when
    # PCZ wins).  Emit as signed fragments so the Delta column of Table 5
    # cannot silently drift from results.csv.
    def _delta(
        baseline_algo: str,
        baseline_env: str,
        baseline_ts: int,
        pcz_env: str,
        pcz_ts: int,
        pcz_w: str | None,
        out_key: str,
        baseline_w: str | None = None,
        learning_rate: str | None = None,
    ) -> None:
        # Apply the same canonical filters on the baseline side as on PCZ.
        b = query(
            rows,
            algorithm=baseline_algo,
            env=baseline_env,
            total_timesteps=baseline_ts,
            weights=baseline_w if baseline_w is not None else pcz_w,
            ent_coef_schedule="0.1:0.01" if "lunarlander" in baseline_env else None,
            learning_rate=learning_rate,
        )
        p = query(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env=pcz_env,
            total_timesteps=pcz_ts,
            weights=pcz_w,
            ent_coef_schedule="0.1:0.01" if "lunarlander" in pcz_env else None,
            learning_rate=learning_rate,
        )
        if b["seeds"] == 0 or p["seeds"] == 0:
            return
        d = b["mean_raw"] - p["mean_raw"]
        reg.add(
            f"{out_key}_delta",
            fmt_signed(d),
            f"{baseline_algo}({baseline_env},{baseline_ts}) - PCZ-PPO({pcz_env},{pcz_ts})",
        )

    _delta(
        "torchrl-ppo",
        "lunarlander",
        500000,
        "lunarlander",
        500000,
        LL_PRIMARY_WEIGHTS,
        "cm_k4_ppo",
        learning_rate="0.0003",
    )  # guard: canonical LR
    _delta(
        "torchrl-grpo",
        "lunarlander",
        500000,
        "lunarlander",
        500000,
        LL_PRIMARY_WEIGHTS,
        "cm_k4_grpo500k",
        learning_rate="0.0003",
    )  # guard: canonical LR
    _delta(
        "torchrl-grpo",
        "lunarlander",
        1000000,
        "lunarlander",
        500000,
        LL_PRIMARY_WEIGHTS,
        "cm_k4_grpo1M",
        learning_rate="0.0003",
    )  # guard: canonical LR
    _delta("torchrl-ppo", "lunarlander-k8", 500000, "lunarlander-k8", 500000, None, "cm_k8_ppo")
    _delta("torchrl-grpo", "lunarlander-k8", 1000000, "lunarlander-k8", 500000, None, "cm_k8_grpo1M")
    _delta("torchrl-grpo", "bipedalwalker", 1000000, "bipedalwalker", 500000, None, "cm_bw_grpo1M")
    _delta("torchrl-grpo", "halfcheetah", 1000000, "halfcheetah", 500000, None, "cm_hc_grpo1M")

    # --- Equal-weights ablation vs primary weights (was hand-typed prose) ---
    # Answers: "Equal weights after z-norm yield X ± Y (vs primary Z ± W), delta D".
    eq = query(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander",
        total_timesteps=500000,
        weights="1.00,1.00,1.00,1.00",
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep equal-weight cells
    )
    pr = query(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
    )
    if eq["seeds"] > 0 and pr["seeds"] > 0:
        reg.add(
            "abl_eqw_stat",
            fmt_stat(eq["mean"], eq["std"]),
            "torchrl-pcz-ppo-running lunarlander 500k weights=(1,1,1,1)",
        )
        reg.add("abl_eqw_seeds", fmt_int(eq["seeds"]), "torchrl-pcz-ppo-running lunarlander 500k weights=(1,1,1,1)")
        reg.add(
            "abl_eqw_delta",
            fmt_signed(eq["mean_raw"] - pr["mean_raw"]),
            "equal-weights - primary-weights PCZ-PPO means",
        )

    # --- A5 cross-seed SD ratio vs A1 (was hand-typed "4.4×") ---
    a5 = query(
        rows,
        algorithm="torchrl-ppo-znorm-post",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        ent_coef_schedule="0.1:0.01",
    )
    a1 = query(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        ent_coef_schedule="0.1:0.01",
        learning_rate="0.0003",  # guard: canonical LR only — excludes HP-sweep non-canonical cells
    )
    if a5["seeds"] > 0 and a1["seeds"] > 0 and a1["std_raw"] > 0:
        reg.add(
            "abl_a5_over_a1_sd_ratio",
            fmt_plain(a5["std_raw"] / a1["std_raw"]),
            "std(A5) / std(A1) — cross-seed SD ratio for decomposition-adds-robustness claim",
        )
        # paper calls this "higher cross-seed variance"; report both interpretations
        reg.add(
            "abl_a5_over_a1_var_ratio",
            fmt_plain((a5["std_raw"] / a1["std_raw"]) ** 2),
            "var(A5) / var(A1) — cross-seed VARIANCE ratio (SD^2)",
        )
        reg.add(
            "abl_a5_s42",
            fmt_signed(next((float(r["eval_mean"]) for r in a5["runs"] if r["seed"] == "42"), 0.0)),
            "A5 seed=42 eval_mean (single-seed illustration)",
        )
        reg.add(
            "abl_a5_s44",
            fmt_signed(next((float(r["eval_mean"]) for r in a5["runs"] if r["seed"] == "44"), 0.0)),
            "A5 seed=44 eval_mean (single-seed illustration)",
        )
    # GRPO K-scaling companions at 500k (used in narrative next to K-scaling table)
    _emit_single(
        reg, rows, "grpoK2_500k", algorithm="torchrl-grpo", env="lunarlander-k2", total_timesteps=500000, min_seeds=3
    )
    _emit_single(
        reg, rows, "grpoK6_500k", algorithm="torchrl-grpo", env="lunarlander-k6", total_timesteps=500000, min_seeds=3
    )
    _emit_single(
        reg, rows, "grpoK8_500k", algorithm="torchrl-grpo", env="lunarlander-k8", total_timesteps=500000, min_seeds=3
    )

    # --- 1M LL-K4 (prose: "At 1M steps, PCZ = 183.7 ± 28.1 vs PPO 154.9 ± 48.4") ---
    _emit_pcz_ppo_pair(
        reg, rows, "k4_1M", env="lunarlander", total_timesteps=1000000, weights=LL_PRIMARY_WEIGHTS, min_seeds=3
    )

    # --- Cross-environment table (tab:cross_env) --- ratios for supporting evidence
    for prefix, env in [
        ("ceHC", "halfcheetah"),
        ("ceBW", "bipedalwalker"),
    ]:
        _emit_pcz_ppo_pair(reg, rows, prefix, env=env, total_timesteps=500000, weights=None, min_seeds=5)

    # --- HC K-scaling — negative control: homogeneous-scale env ---
    # PCZ loses across all K on HalfCheetah at 500k (undertrained but directionally clear):
    #   K=2 Δ-9, K=4 Δ-30, K=6 Δ-44, K=8 Δ-238. Variance reduction 2.7-3.6× at K∈{2,4,8}.
    # Supports "heterogeneity-required" claim (HC components are all homogeneous CV ~0.8-1.9).
    for k_suffix, env in [
        ("k4", "halfcheetah-k4"),
        ("k6", "halfcheetah-k6"),
        ("k8", "halfcheetah-k8"),
    ]:
        try:
            _emit_pcz_ppo_pair(
                reg,
                rows,
                f"ceHC_{k_suffix}",
                env=env,
                total_timesteps=500000,
                weights=None,
                min_seeds=3,
            )
        except ValueError:
            pass  # Insufficient data

    # --- Cross-env validation of A15 (symznorm) on BipedalWalker (3 seeds) ---
    _emit_single(
        reg,
        rows,
        "ceBW_symznorm",
        algorithm="torchrl-pcz-ppo-symznorm",
        env="bipedalwalker",
        total_timesteps=500000,
        min_seeds=3,
    )
    # --- A15 (symznorm) K-scaling robustness across K=2/4/6/8 ---
    _emit_single(
        reg,
        rows,
        "k2_symznorm",
        algorithm="torchrl-pcz-ppo-symznorm",
        env="lunarlander-k2",
        total_timesteps=500000,
        min_seeds=3,
    )
    _emit_single(
        reg,
        rows,
        "k4_symznorm",
        algorithm="torchrl-pcz-ppo-symznorm",
        env="lunarlander",
        total_timesteps=500000,
        weights=LL_PRIMARY_WEIGHTS,
        min_seeds=3,
        ent_coef_schedule="0.1:0.01",
    )
    _emit_single(
        reg,
        rows,
        "k6_symznorm",
        algorithm="torchrl-pcz-ppo-symznorm",
        env="lunarlander-k6",
        total_timesteps=500000,
        min_seeds=3,
    )
    _emit_single(
        reg,
        rows,
        "k8_symznorm",
        algorithm="torchrl-pcz-ppo-symznorm",
        env="lunarlander-k8",
        total_timesteps=500000,
        min_seeds=3,
    )

    # --- Plateau validation (4M steps, LunarLander K=4) ---
    # Extended from n=3 to n=10. At n=10 the "converge to the same asymptote"
    # story from n=3 no longer holds: PCZ retains a directional mean advantage
    # (+29.2, d=+0.77, Welch p=0.11) plus 2.72x lower cross-seed variance.
    # Prose in §Limitations / Conclusion updated accordingly.
    _emit_single(
        reg,
        rows,
        "w10_k4_pcz",
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander",
        total_timesteps=4000000,
        min_seeds=3,
    )
    _emit_single(
        reg,
        rows,
        "w10_k4_ppo",
        algorithm="torchrl-ppo",
        env="lunarlander",
        total_timesteps=4000000,
        min_seeds=3,
    )
    # Pair-fragment block (delta, ratio, var_ratio) at LL K=4 4M canonical
    # cosine-entropy schedule. The ent_coef_schedule filter is essential —
    # without it the fixed-entropy tuning-audit seeds would bleed into the pair.
    _emit_pcz_ppo_pair(
        reg,
        rows,
        "w10_k4_4M",
        env="lunarlander",
        total_timesteps=4000000,
        weights="10.00,5.00,0.50,0.50",
        ent_coef_schedule="0.1:0.01",
        min_seeds=3,
    )
    # K=8 4M HEADLINE: under K-analogous heterogeneous weights (10,1,1,1,1,1,0.5,0.5),
    # PCZ-PPO +212.9 ± 11.8 BEATS matched-infra PPO +175.7 ± 33.0 (n=3 each) with
    # ~2.8x lower cross-seed variance. Earlier "K=8 4M collapses" result was an
    # equal-weights / quasi-equal-weights artifact — exactly the failure mode
    # (Table 9, K=4 weight sensitivity) already documented at K=4: equal-weights
    # d=-3.45, flat d=-7.04 are catastrophic for PCZ. Retain `w10_k8_pcz` /
    # `w10_k8_ppo` fragment names so existing \cnum{} citations in the paper
    # auto-update to the headline numbers.
    _emit_single(
        reg,
        rows,
        "w10_k8_pcz",
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander-k8",
        total_timesteps=4000000,
        weights=K8_WEIGHTS_HETEROG,
        min_seeds=3,
    )
    _emit_single(
        reg,
        rows,
        "w10_k8_ppo",
        algorithm="torchrl-ppo",
        env="lunarlander-k8",
        total_timesteps=4000000,
        weights=K8_WEIGHTS_HETEROG,
        min_seeds=3,
    )

    # K=8/4M heterog pair-fragment block (mean / std / seeds / stat / delta / ratio /
    # var_ratio for both algorithms). Same data as w10_k8_* above; pair shape is
    # convenient when prose wants the matched-pair stats together.
    _emit_pcz_ppo_pair(
        reg,
        rows,
        "rr16h7_k8_4M",
        env="lunarlander-k8",
        total_timesteps=4000000,
        weights=K8_WEIGHTS_HETEROG,
        min_seeds=3,
    )

    # K=8/4M H8 NEGATIVE CONTROL: zero-weighting `velocity` (CV=177, "pure noise")
    # drives PCZ +0.2 ± 7.0 vs PPO +183.5 ± 26.5 (n=3 each). The zero-weighted
    # component still contributes to per-component variance accounting but
    # contributes 0 to the scalar reward — degenerate per-component normalisation.
    # PPO is robust because it sees only the weighted sum. Documented in
    # §Limitations as the "non-zero weights required" usage constraint.
    _emit_pcz_ppo_pair(
        reg,
        rows,
        "rr16h8_k8_4M",
        env="lunarlander-k8",
        total_timesteps=4000000,
        weights=K8_WEIGHTS_ZEROED_VEL,
        min_seeds=3,
    )

    # K=8/4M EQUAL-WEIGHTS NEGATIVE CONTROL: env default (no --reward-component-weights
    # flag passed → all components weight 1.0). PCZ-running (n=6 across two re-launches)
    # = -2.3 ± 7.5 vs PPO (n=6) = +185.8 ± 17.6.  Same collapse mechanism as H8 but
    # via a different failure mode — equal weights at K=8 produce d=-3.45 / d=-7.04
    # territory at K=4, and the K=8 long-horizon picture matches.  Filter by exact
    # empty `component_weights` (env default) since `query()` does prefix-match and
    # ``weights=""`` would match every run.
    _emit_pcz_ppo_pair_exact_weights(
        reg,
        rows,
        "rr16_equal_k8_4M",
        env="lunarlander-k8",
        total_timesteps=4000000,
        weights_exact="",
        ent_coef_schedule="0.1:0.01",
        min_seeds=3,
    )

    # K=8/4M moderate + flat — completes the K=8/4M weight-
    # sensitivity table (tab:weight_sensitivity_k8). Same story as K=4 Table 9:
    # PCZ requires heterogeneous weights. Cohen's d: H7 +1.50 → moderate -8.30
    # → flat -12.42 → equal -13.90 (monotone, PCZ only wins at max heterogeneity).
    import numpy as np
    from scipy import stats as _stats

    for prefix, weights in [
        ("rr16_moderate_k8_4M", K8_WEIGHTS_MODERATE),
        ("rr16_flat_k8_4M", K8_WEIGHTS_FLAT),
    ]:
        _emit_pcz_ppo_pair(
            reg,
            rows,
            prefix,
            env="lunarlander-k8",
            total_timesteps=4000000,
            weights=weights,
            min_seeds=3,
            ent_coef_schedule="0.1:0.01",
        )

    # Cohen's d + Welch p for every K=8/4M weight-sensitivity row so the new
    # Table 9b matches the K=4 column layout (Weights / PCZ / PPO / Δ / d / p / n).
    def _d_p_from_rows(pcz_rows: list[dict], ppo_rows: list[dict]) -> tuple[float, float, int, int]:
        # chrono-latest dedupe per seed so we match the _emit_* quant path.
        def _dedup(rs):
            by_seed: dict[str, dict] = {}
            for r in sorted(rs, key=lambda r: r.get("date", "")):
                by_seed[r["seed"]] = r
            return by_seed.values()

        pcz_vals = np.asarray(
            [float(r["eval_mean"]) for r in _dedup(pcz_rows) if r.get("eval_mean", "") not in ("", "nan", None)],
            dtype=float,
        )
        ppo_vals = np.asarray(
            [float(r["eval_mean"]) for r in _dedup(ppo_rows) if r.get("eval_mean", "") not in ("", "nan", None)],
            dtype=float,
        )
        if len(pcz_vals) < 2 or len(ppo_vals) < 2:
            return 0.0, 1.0, len(pcz_vals), len(ppo_vals)
        pooled = ((len(pcz_vals) - 1) * pcz_vals.std(ddof=1) ** 2 + (len(ppo_vals) - 1) * ppo_vals.std(ddof=1) ** 2) / (
            len(pcz_vals) + len(ppo_vals) - 2
        )
        d = (pcz_vals.mean() - ppo_vals.mean()) / float(np.sqrt(pooled)) if pooled > 0 else 0.0
        _t, p = _stats.ttest_ind(pcz_vals, ppo_vals, equal_var=False)
        return d, float(p), len(pcz_vals), len(ppo_vals)

    def _filter_rows(algo: str, weights_match: str, exact: bool) -> list[dict]:
        return [
            r
            for r in rows
            if r["algorithm"] == algo
            and r["env"] == "lunarlander-k8"
            and r["total_timesteps"] == "4000000"
            and r.get("ent_coef_schedule", "") == "0.1:0.01"
            and r.get("eval_mean", "") not in ("", None)
            and (
                (exact and r.get("component_weights", "") == weights_match)
                or (not exact and r.get("component_weights", "").startswith(weights_match))
            )
        ]

    for prefix, weights_match, exact in [
        ("rr16h7_k8_4M", K8_WEIGHTS_HETEROG, False),
        ("rr16_moderate_k8_4M", K8_WEIGHTS_MODERATE, False),
        ("rr16_flat_k8_4M", K8_WEIGHTS_FLAT, False),
        ("rr16_equal_k8_4M", "", True),
    ]:
        pcz_r = _filter_rows("torchrl-pcz-ppo-running", weights_match, exact)
        ppo_r = _filter_rows("torchrl-ppo", weights_match, exact)
        d, p, np_, nppo = _d_p_from_rows(pcz_r, ppo_r)
        src = f"K=8/4M weights={weights_match!r} exact={exact} n={np_}+{nppo}"
        reg.add(f"{prefix}_cohen_d", fmt_signed(d, decimals=2), src)
        reg.add(f"{prefix}_welch_p", fmt_plain(p, decimals=3), src)

    # --- Cross-env plateau (4M steps, BipedalWalker K=3 and HalfCheetah K=2) ---
    # BW 4M extended to n=10 reveals CATASTROPHIC PPO instability — 3/10 PPO seeds
    # collapse to the -92 reward floor while all 10 PCZ seeds learn.
    # BW 4M n=10: PCZ +252.8 ± 35.4 vs PPO +156.0 ± 174.7 (Δ=+96.8, var ratio 4.93×).
    # This flips the n=5 "tied at plateau" reading.
    for prefix, algo, env in [
        ("w10_bw_pcz", "torchrl-pcz-ppo-running", "bipedalwalker"),
        ("w10_bw_ppo", "torchrl-ppo", "bipedalwalker"),
        ("w10_hc_pcz", "torchrl-pcz-ppo-running", "halfcheetah"),
        ("w10_hc_ppo", "torchrl-ppo", "halfcheetah"),
    ]:
        _emit_single(
            reg,
            rows,
            prefix,
            algorithm=algo,
            env=env,
            total_timesteps=4000000,
            min_seeds=3,
        )
    # Pair fragments for BW 4M (delta, var_ratio, seeds) — n=10.
    _emit_pcz_ppo_pair(reg, rows, "w10_bw_4M", env="bipedalwalker", total_timesteps=4000000, weights=None, min_seeds=5)
    # Pair fragments for HC 4M — will auto-update from n=5 to n=10 once HC chain completes.
    _emit_pcz_ppo_pair(reg, rows, "w10_hc_4M", env="halfcheetah", total_timesteps=4000000, weights=None, min_seeds=3)

    # --- Clean-decomposition trading ablation ---
    # trading-k3-clean = [pnl_gain, pnl_loss, txn_cost]. Tests whether dropping
    # noise-dominated residual/spread/borrow_cost flips the trading PCZ-vs-PPO null.
    # Finding: clean decomposition improves BOTH PPO (+56) and PCZ (+55) vs K=4
    # canonical (~+18-27), but PCZ-vs-PPO gap stays null (Δ=-1.2, p=0.94).
    _emit_pcz_ppo_pair_exact_weights(
        reg,
        rows,
        "rr18_4_tradeK3c_eq",
        env="trading-k3-clean",
        total_timesteps=100000,
        weights_exact="",
        min_seeds=5,
    )
    import numpy as _np_rr184
    from scipy import stats as _stats_rr184

    _pcz_rr184 = _np_rr184.array(
        [
            float(r["eval_mean"])
            for r in rows
            if r["algorithm"] == "torchrl-pcz-ppo-running"
            and r["env"] == "trading-k3-clean"
            and r["total_timesteps"] == "100000"
            and r.get("component_weights", "") == ""
            and r.get("eval_mean", "") not in ("", None)
        ]
    )
    _ppo_rr184 = _np_rr184.array(
        [
            float(r["eval_mean"])
            for r in rows
            if r["algorithm"] == "torchrl-ppo"
            and r["env"] == "trading-k3-clean"
            and r["total_timesteps"] == "100000"
            and r.get("component_weights", "") == ""
            and r.get("eval_mean", "") not in ("", None)
        ]
    )
    _t_rr184, _p_rr184 = _stats_rr184.ttest_ind(_pcz_rr184, _ppo_rr184, equal_var=False)
    reg.add(
        "rr18_4_tradeK3c_eq_welch_p",
        fmt_plain(float(_p_rr184), decimals=2),
        "env=trading-k3-clean 100k equal-weights",
    )

    # --- Trading heterogeneous-weight sweep ---
    # 100k trading-k4 and trading-k8, n=5 per cell. Tests whether canonical-style
    # heterogeneous weights flip the trading null found under equal weights.
    # All three head-to-head comparisons stay null — strengthens the trading null.
    import numpy as _np

    for prefix, env, weights in [
        ("rr18_1_tradeK4_w5505", "trading-k4", "5.00,5.00,0.50,0.50"),
        ("rr18_1_tradeK4_w101011", "trading-k4", "10.00,10.00,1.00,1.00"),
        ("rr18_1_tradeK8_w5505", "trading-k8", "5.00,5.00,5.00,5.00,0.50,0.50,0.50,0.50"),
    ]:
        _emit_pcz_ppo_pair_exact_weights(
            reg,
            rows,
            prefix,
            env=env,
            total_timesteps=100000,
            weights_exact=weights,
            min_seeds=5,
        )
        # Welch p for this cell (companion to existing _delta from the pair fragment).
        pcz_vals = _np.array(
            [
                float(r["eval_mean"])
                for r in rows
                if r["algorithm"] == "torchrl-pcz-ppo-running"
                and r["env"] == env
                and r["total_timesteps"] == "100000"
                and r.get("component_weights", "") == weights
                and r.get("eval_mean", "") not in ("", None)
            ]
        )
        ppo_vals = _np.array(
            [
                float(r["eval_mean"])
                for r in rows
                if r["algorithm"] == "torchrl-ppo"
                and r["env"] == env
                and r["total_timesteps"] == "100000"
                and r.get("component_weights", "") == weights
                and r.get("eval_mean", "") not in ("", None)
            ]
        )
        _t, p = _stats.ttest_ind(pcz_vals, ppo_vals, equal_var=False)
        reg.add(f"{prefix}_welch_p", fmt_plain(float(p), decimals=2), f"env={env} 100k weights={weights}")

    # --- LLM alignment (Qwen2.5-0.5B, QLoRA, 200 steps) ---
    # Data lives in artifacts/pcz-ppo/llm_alignment/runs/result_k{K}_{mode}_s{seed}.json,
    # not results.csv. K=2 has 3 seeds; K=4+ have 1 seed (signal check).
    for k in [2, 4, 6, 8]:
        _emit_llm_claims_for_k(reg, k)

    # --- LLM 2x2 factorial: {PPO, RLOO} x {standard, PCZ} at K=4 ---
    # RLOO data in llm_alignment/runs/result_k4_{mode}_s{seed}.json (s42-44 n=3).
    # PPO data in llm_alignment/runs/result_ppo_k4_{mode}_s{seed}.json (s42-44 n=3).
    # Finding: critic dominates (PPO >> RLOO, p<0.001); PCZ null in both (p=0.99).
    import json as _json_fact

    import numpy as _np_fact
    from scipy import stats as _stats_fact

    _base_fact = _PAPER_DIR.parent / "llm_alignment" / "runs"

    def _load_fact(critic: str, mode: str, seed: int):
        if critic == "rloo":
            p = _base_fact / f"result_k4_{mode}_s{seed}.json"
        else:
            p = _base_fact / f"result_ppo_k4_{mode}_s{seed}.json"
        if not p.exists():
            return None
        with open(p) as f:
            return _json_fact.load(f).get("total_reward")

    _cells_fact = {}
    for _c in ["rloo", "ppo"]:
        for _m in ["standard", "pcz"]:
            _vals = []
            for _s in [42, 43, 44]:
                _v = _load_fact(_c, _m, _s)
                if _v is not None:
                    _vals.append(float(_v))
            _cells_fact[(_c, _m)] = _np_fact.array(_vals)
            if len(_vals) > 0:
                _mean = float(_np_fact.mean(_vals))
                _std = float(_np_fact.std(_vals, ddof=1)) if len(_vals) > 1 else 0.0
                _prefix = f"LLMfact_k4_{_c}_{_m}"
                reg.add(
                    f"{_prefix}_stat",
                    fmt_stat(round(_mean, 3), round(_std, 3)),
                    f"LLM factorial K=4 critic={_c} mode={_m} n={len(_vals)}",
                )
                reg.add(
                    f"{_prefix}_seeds",
                    fmt_int(len(_vals)),
                    f"LLM factorial K=4 critic={_c} mode={_m}",
                )
    reg.add(
        "LLMfact_k4_seeds",
        fmt_int(min(len(v) for v in _cells_fact.values())),
        "LLM factorial K=4 min seeds per cell",
    )
    _ppo_all = _np_fact.concatenate([_cells_fact[("ppo", "standard")], _cells_fact[("ppo", "pcz")]])
    _rloo_all = _np_fact.concatenate([_cells_fact[("rloo", "standard")], _cells_fact[("rloo", "pcz")]])
    _pcz_all = _np_fact.concatenate([_cells_fact[("ppo", "pcz")], _cells_fact[("rloo", "pcz")]])
    _std_all = _np_fact.concatenate([_cells_fact[("ppo", "standard")], _cells_fact[("rloo", "standard")]])
    _, _p_crit = _stats_fact.ttest_ind(_ppo_all, _rloo_all, equal_var=False)
    _, _p_pcz = _stats_fact.ttest_ind(_pcz_all, _std_all, equal_var=False)
    reg.add(
        "LLMfact_k4_main_critic",
        fmt_signed(round(float(_np_fact.mean(_ppo_all) - _np_fact.mean(_rloo_all)), 2)),
        "LLM factorial K=4 main-effect critic",
    )
    reg.add(
        "LLMfact_k4_main_critic_p",
        fmt_plain(float(_p_crit), decimals=3) if float(_p_crit) >= 0.001 else "<0.001",
        "LLM factorial K=4 Welch p critic",
    )
    reg.add(
        "LLMfact_k4_main_pcz",
        fmt_signed(round(float(_np_fact.mean(_pcz_all) - _np_fact.mean(_std_all)), 2)),
        "LLM factorial K=4 main-effect PCZ",
    )
    reg.add(
        "LLMfact_k4_main_pcz_p",
        fmt_plain(float(_p_pcz), decimals=2),
        "LLM factorial K=4 Welch p PCZ",
    )

    # --- LLM alignment: PPO-critic runs ---
    # Separate `result_ppo_k{K}_*.json` data from RLOO `result_k{K}_*.json`.
    # K=4: n=3 (seeds 42-44) — NULL. K=6: n=5 (seeds 42-46) — variance-reduction signal.
    _emit_llm_ppo_claims_for_k(reg, 4, seeds=[42, 43, 44])
    # K=6: existing LLMppo_k6_* fragments use 200-step BACKUP files (s42 std backup
    # was lost when live s42 was overwritten by the 500-step rerun; fall back to
    # balanced n=4 for the 200-step claim).
    _emit_llm_ppo_claims_for_k(reg, 6, seeds=[43, 44, 45, 46], suffix="_200step", prefix="LLMppo")
    # 500-step K=6 extension. Reveals pattern reversal vs 200-step.
    _emit_llm_ppo_claims_for_k(reg, 6, seeds=[42, 43, 44, 45, 46], suffix="", prefix="LLMppo_500")

    # --- Sample efficiency (fig_sample_efficiency data used in prose) ---
    for ts, tag in [(100000, "100k"), (200000, "200k"), (500000, "500k"), (1000000, "1M")]:
        _emit_pcz_ppo_pair(
            reg,
            rows,
            f"se_{tag}",
            env="lunarlander",
            total_timesteps=ts,
            weights=LL_PRIMARY_WEIGHTS,
            min_seeds=3,
        )

    return reg


# --- Entry points --------------------------------------------------------


def _tex_references(tex_path: Path) -> set[str]:
    """Return the set of ``\\cnum{name}`` names referenced in the .tex,
    ignoring LaTeX comments.
    """
    import re

    text = tex_path.read_text()
    # strip comments: %...EOL unless escaped as \%
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    return set(re.findall(r"\\cnum\{([a-zA-Z0-9_\-]+)\}", text))


def _enforce_registry_matches_paper(reg: Registry, tex_path: Path) -> list[str]:
    """Every registered claim must be used, every used \\cnum must be registered.

    Returns list of error messages; empty if consistent.
    """
    registered = set(reg.as_dict().keys())
    referenced = _tex_references(tex_path)
    errors = []
    for ref in referenced - registered:
        errors.append(
            f"\\cnum{{{ref}}} appears in {tex_path.name} but is not registered in render_claims.build_registry()."
        )
    for orphan in registered - referenced:
        errors.append(
            f"claim {orphan!r} is registered but never referenced in {tex_path.name}. "
            f"Either use \\cnum{{{orphan}}} in the paper or remove the registration."
        )
    return errors


def render(target_dir: Path = _GEN_DIR, *, enforce: bool = True, tex_path: Path | None = None) -> Registry:
    """Render fragments referenced by the paper to ``target_dir``.

    Workflow:
      1. Parse ``tex_path`` (default: pcz_ppo.tex) to discover every
         ``\\cnum{name}`` reference.
      2. Build registry filtered to that reference set — only referenced
         claims materialize on disk.
      3. If ``enforce``, fail if the paper references a name the catalog
         doesn't know how to compute.
    """
    paper = tex_path or (_PAPER_DIR / "pcz_ppo.tex")
    used = _tex_references(paper)
    rows = load_results()
    reg = build_registry(rows, used_filter=used)
    if enforce:
        registered_names = set(reg.as_dict().keys())
        undefined = used - registered_names
        if undefined:
            print("render_claims: paper references undefined claim(s):", file=sys.stderr)
            for name in sorted(undefined):
                print(
                    f"  - \\cnum{{{name}}} used in {paper.name} but not registered in render_claims.build_registry()",
                    file=sys.stderr,
                )
            raise SystemExit(2)
    reg.render_files(target_dir)
    return reg


def check() -> int:
    """CI mode: render to a tempdir, compare against committed fragments."""
    paper = _PAPER_DIR / "pcz_ppo.tex"
    used = _tex_references(paper)
    rows = load_results()
    reg = build_registry(rows, used_filter=used)
    # Enforce: every \cnum{name} in paper must have a registration
    registered_names = set(reg.as_dict().keys())
    undefined = used - registered_names
    if undefined:
        print("render_claims --check FAILED (paper references undefined claims):", file=sys.stderr)
        for name in sorted(undefined):
            print(f"  - \\cnum{{{name}}} used but not registered", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        reg.render_files(tmp)
        # Compare every file
        mismatches = []
        for fname in sorted(tmp.iterdir()):
            committed = _GEN_DIR / fname.name
            if not committed.exists():
                mismatches.append(f"  [missing committed]  generated/{fname.name}")
                continue
            if committed.read_text() != fname.read_text():
                mismatches.append(f"  [content drift]      generated/{fname.name}")
        for fname in sorted(_GEN_DIR.iterdir()):
            if not (tmp / fname.name).exists():
                mismatches.append(f"  [orphan committed]   generated/{fname.name}")
        if mismatches:
            print("render_claims --check FAILED:", file=sys.stderr)
            for m in mismatches:
                print(m, file=sys.stderr)
            print(
                "\nFix: run `uv run python artifacts/pcz-ppo/paper/render_claims.py` and commit the updated fragments.",
                file=sys.stderr,
            )
            return 1
    print(f"render_claims --check OK ({len(reg.as_dict())} fragments consistent)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI mode: fail if committed fragments differ from a fresh render",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered claim names and their sources, then exit",
    )
    args = parser.parse_args()

    if args.list:
        reg = build_registry(load_results())
        for name, claim in sorted(reg.as_dict().items()):
            print(f"{name:30s}  {claim.content:25s}  # {claim.source}")
        return 0
    if args.check:
        return check()
    reg = render()
    print(
        f"Rendered {len(reg.as_dict())} fragments to {_GEN_DIR.relative_to(Path.cwd()) if _GEN_DIR.is_relative_to(Path.cwd()) else _GEN_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
