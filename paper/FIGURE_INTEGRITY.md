# Figure Integrity System

Pre-commit safeguards for figure-text consistency, the figure analog of the
existing `render_claims.py` + `lint_hardcoded_numbers.py` system that protects
table values.

## Motivation

The paper's tables are protected from drift by the data-driven `\cnum{}`
fragment system (`render_claims.py`) and the hand-typed-numbers lint
(`lint_hardcoded_numbers.py`). Together they enforce: every number in a table
or prose must be derived from `results.csv` via a registered fragment, with
periodic re-rendering to catch drift, and a CI lint that fails the commit if a
hand-typed `mean ± std` slips through.

**Figures had no equivalent protection.** During the 2026-05-01 audit pass we
caught the following figure-related failures, none of which any existing
gate would have flagged:

| Failure mode | Example |
|---|---|
| Caption claims a winner that the visual does not clearly support | Figure 9 top-right caption said "PCZ-PPO wins" but rollout curves were visually ambiguous (both volatile, PPO spikes higher in places) |
| Caption metric does not match figure y-axis metric | Figure 9 quoted episodic eval reward (`+188`) while y-axis showed per-step rollout reward (`±10`); reader couldn't reconcile the two without environment-specific scaling knowledge |
| Annotations / legends clipped by panel boundary | Figure 8 (entropy_sweep) bottom annotation at `ent=0.1` overlapping the bottom bounding box; Figure 10 K=4 `n=10` label touching the top boundary |
| Figure number drifts when the paper is restructured | After moving fig:alignment_results to App.D.4, fig:learning_curves_4M became Figure 9 (was 10). Hand-written "Figure 10" references in conversation/notes silently became wrong |
| Hand-typed direction claims drift when data updates | Bridge prose said "PCZ underperforms PPO at K∈{2,4}" referring to heterogeneous trading; reader read it as homogeneous trading (the figure variant) and saw a contradiction |

The common pattern: **every gate in the existing system is data-only**.
Figures sit in a blind spot where the prose says one thing and the visual says
another, with no mechanical check.

## Honest Scope: What This System Does NOT Catch

The biggest objection to a figure-integrity lint is that **most caption errors
are interpretive, not data-flipping**. The 2026-05-01 audit revealed roughly
the following distribution of failure modes:

| Failure class | Layer that catches it | Estimated share |
|---|---|---|
| Hand-typed direction claim contradicting data (e.g. "PCZ wins" when delta is negative) | Layer 1+2 | ~20% |
| Annotation clipped by panel boundary | Layer 3 | ~10% |
| Mixed metrics on same panel (rollout y-axis vs eval title) | Layer 4 | ~15% |
| **Caption claims a clear winner when the visual is ambiguous** | **none** | ~25% |
| **Cross-section coherence (prose in §X talks about env A, figure shows env B)** | **none** | ~20% |
| **Aesthetic / readability (color choices, font size, marker overlap)** | **none** | ~10% |

The lint system catches roughly **45% of observed failure modes** in the
mechanical sense. The remaining 55% require either (a) human or critic-agent
review, or (b) a vision-LLM-based check (Layer 5, deferred). This system is
necessary but not sufficient. It is calibrated as **a layer, not a solution**.

The system is most valuable as a *pre-commit safety net for mechanical
drift*. It makes the LLM less able to silently break figure-text consistency
during refactors, but it does not replace adversarial review for interpretive
correctness.

## Architecture

Four layers, in order of leverage:

### Layer 1: Data-driven caption claims (extension of `render_claims.py`)

Every paired (PCZ-PPO, PPO) cell whose `_pcz_stat` and `_ppo_stat` fragments
exist in the registry automatically gets:

- `<prefix>_winner` — `"PCZ-PPO"`, `"PPO"`, or `"matched"` (string)
- `<prefix>_winner_verb` — `"outperforms"`, `"matches"`, etc. (verb form)
- `<prefix>_direction` — `"+"` (PCZ ahead), `"-"` (PPO ahead), `"0"` (matched)

These fragments resolve the same way as `\cnum{}` and update automatically
when `results.csv` changes. A caption that says
`Algorithm \cnum{w10_k4_4M_winner} wins by \cnum{w10_k4_4M_delta}` cannot
silently become wrong if results change.

**Classification rule** (post-2026-05-01 Bug #1/#2 fix):

- **Zero-variance edge case**: when `pcz_std == ppo_std == 0`, classify
  by `|delta| > 1e-6` directly (a deterministic difference is a perfect
  signal, not a "matched" call).
- **General case**: a directional winner requires **both** `|Cohen's d|
  >= 0.2` (Cohen 1988 small-effect threshold) **and** `Welch p < 0.05`
  (statistical significance). Otherwise → `matched`.

The α=0.05 requirement matches the paper's framing convention:
"directional but not significant" cells (e.g. K=4/4M with d=0.77, p=0.10)
are classified as `matched` at this layer. Captions that need to disclose
a directional non-significant effect should cite `_delta` and `_welch_p`
fragments directly rather than the `_winner` label.

Thresholds are configuration constants in `render_claims.py`
(`_WINNER_D_THRESHOLD`, `_WINNER_P_THRESHOLD`, `_WINNER_DELTA_EPSILON`).

**Coverage**: catches direction-flip bugs where prose claims one winner but
data says another. Does NOT catch interpretive issues (claim is data-correct
but visual is ambiguous).

### Layer 2: Caption ↔ data consistency lint (`lint_caption_claims.py`)

Scans every `\caption{...}` block in `pcz_ppo.tex` looking for hand-typed
direction-claim patterns:

- `(PCZ|PCZ-PPO|PPO) (wins|outperforms|beats|ahead|loses|underperforms|fails)`
- `(PCZ|PPO) (above|below) (PCZ|PPO)`
- `is (a |an )?(structural |clean )?null` (when env is mentioned)

Each match must be **adjacent to** (within ~50 chars of) a `\cnum{}` reference
that justifies it (e.g., a `_winner` fragment, a `_delta` fragment with a sign
that matches the claim, or an explicit p-value). Matches without nearby
justification are flagged as drift risk and must either be wrapped in
`\cnum{}` or explicitly allow-listed in `ALLOW_PATTERNS` (same convention as
`lint_hardcoded_numbers.py`).

**Coverage**: catches hand-typed direction claims that drift from data.
Does NOT catch interpretive issues (the data-correct claim fails to match
the visual ambiguity).

### Layer 3: Geometric sanity checks (`lint_figure_geometry.py`)

Imports each `fig_*.py` module, runs its `main()` (or new `build_figure()`)
to obtain the matplotlib `Figure` object, then for each `Axes` checks:

1. **Text-in-bounds**: every `Text` artist has `(x, y)` inside
   `(xlim, ylim)` extended by zero margin. (Tolerance for axis labels and
   titles, which live outside the data area.)
2. **Legend non-overlap**: legend's `bbox` does not overlap with any
   `Line2D` or `Rectangle` data bbox by more than 10% area.
3. **Errorbar in-bounds**: every `errorbar` cap (top and bottom) is within
   `ylim`.
4. **Image coverage**: figure's `bbox_inches='tight'` post-render does not
   crop any `Text` artist (compared `Figure.bbox` before vs after tight).

Each check is independent. Failures are reported per-axes-per-check.

**Coverage**: catches the boundary-clip class of issues (Figure 8, Figure 10).
False positive risk: matplotlib's auto-tight-bbox post-processing may move
elements; the lint may flag pre-tight positions that render fine. Mitigation:
render to PDF, parse final positions, then check.

### Layer 4: Mixed-unit detection (`lint_figure_units.py`)

Each `fig_*.py` declares a `UNITS` module-level constant:

```python
UNITS = "rollout-per-step"   # rollout/reward_mean (TorchRL per-step)
# or
UNITS = "eval-episodic"      # eval/mean_reward (per-episode)
# or
UNITS = "rollout-per-step,eval-episodic"  # both, requires acknowledgment
```

The lint asserts:

1. Every `fig_*.py` has a `UNITS` declaration.
2. If `UNITS` contains a comma (mixed), the figure script must also have a
   `MIXED_UNITS_ACKNOWLEDGED = True` flag and the caption must contain
   the literal text `different units` or `different metrics`.
3. The caption's metric language matches the declared units (e.g., if
   `UNITS = "eval-episodic"`, caption should not say "rollout").

**Coverage**: catches the metric-confusion class (Figure 9 rollout-vs-eval).
Limitation: requires manual `UNITS` annotation on each script.

### Layer 5 (deferred): Vision-LLM caption check

Use a vision LLM with structured prompts to verify caption-claim ↔ visual
consistency. Prompt template:

```
Look at panel <panel_id> of <figure_path>.
The caption says: <caption>.
For each claim in the caption, decide whether the visual evidence supports
it, contradicts it, or is ambiguous. Be strict: if the visual could be
plausibly read either way, mark "ambiguous".
```

Deferred because:

- Vision LLMs have confirmation bias when given the caption text first
- Adversarial framing requires multiple invocations
- This is the system that should catch the 25% interpretive failures, but
  the API surface and reliability are not yet good enough to be a hard gate

Documented here so a future agent or human can pick it up when vision-LLM
reliability improves.

## Implementation

| File | Status | Purpose |
|---|---|---|
| `render_claims.py` (modify) | Layer 1 | Emit `_winner`, `_winner_verb`, `_direction` fragments |
| `lint_caption_claims.py` (new) | Layer 2 | Caption hand-typed-claim drift detector |
| `lint_figure_geometry.py` (new) | Layer 3 | Render-and-inspect geometric lint |
| `lint_figure_units.py` (new) | Layer 4 | Mixed-unit detector with `UNITS` declaration |
| `tests/test_figure_integrity.py` (new) | All | Unit tests + paper-state integration test |
| `paper_build.py` (modify) | All | Register Layers 2/3/4 as build stages |

All four layers run in `--check` mode in pre-commit. Failures block the
commit. The convention matches `render_claims.py --check`.

## Known False-Positive Scenarios

| Scenario | Layer | Mitigation |
|---|---|---|
| Caption mentions "K=2 PPO wins" referring to a specific *cell* (negative-control language) | Layer 2 | Allow-list pattern: `K{=}2.*PPO wins` is ALLOW (negative control disclosure) |
| Figure caption says "ent=0.05 PPO collapses" without `\cnum{}` because the collapse threshold is a domain concept, not a data point | Layer 2 | Allow-list pattern for "collapses" with surrounding env context |
| Mixed-units figure where the user explicitly wants both metrics (Figure 9 was an example before we removed the twin axis) | Layer 4 | `MIXED_UNITS_ACKNOWLEDGED = True` flag with explicit caption disclosure |
| matplotlib's `bbox_inches='tight'` produces a different rendered position than the script-level coordinate | Layer 3 | Render to PDF, parse PDF, check positions in final output (more reliable than pre-tight check) |

## Testing Strategy

1. **Unit tests** (`tests/test_figure_integrity.py`):
   - Layer 1: synthetic registry with known PCZ/PPO values → assert correct `_winner` emission
   - Layer 2: synthetic captions with/without `\cnum{}` justification → assert correct flagging
   - Layer 3: synthetic matplotlib figure with known clipped annotation → assert violation detected
   - Layer 4: synthetic `fig_*.py` files with mixed/unmixed `UNITS` → assert correct flagging

2. **Integration test**: run all four lints on the current `pcz_ppo.tex` /
   figures, fix any real issues found, then assert the lints pass on the
   real paper state. This is run by `paper_build.py --check`.

## Limitations and Future Work

- The lints catch ~45% of observed figure-text failure modes. The remaining
  55% (interpretive, cross-section coherence, aesthetic) require human
  review or vision-LLM iteration.
- Layer 3's geometric check has false-positive risk; expect to allow-list
  some matplotlib-tight-bbox quirks per figure.
- Layer 4 requires manual `UNITS` annotation on each `fig_*.py`. There's no
  current way to auto-detect units from arbitrary matplotlib code.
- A future Layer 5 (vision-LLM caption check) would catch the interpretive
  failures but is deferred until vision-LLM reliability improves.

## What "All Gates Green" Means After This System

Today (before Layer 1-4):
> Tables, fragments, and lints all pass. Figures may silently contradict
> their captions or have clipped annotations.

After Layer 1-4:
> Tables, fragments, AND figure caption-data consistency, AND figure
> geometry, AND figure unit declarations all pass. Figures cannot silently
> contradict their data-derived claims, cannot have annotations clipped at
> panel boundaries, and cannot mix metrics without explicit acknowledgment.
> Interpretive correctness still requires human or critic-agent review.

The system makes the LLM **less able to silently break figure-text
consistency**, which is the goal stated in the brief. It does not make the
LLM correct; it makes its mechanical errors loud.
