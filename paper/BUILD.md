# Paper build pipeline

One script — `paper_build.py` — owns the paper DAG. It replaces the implicit
"remember to re-run `fig_*.py` after experiments" norm with an explicit,
hash-checked pipeline.

## TL;DR

```bash
# Rebuild stale figures + fragments after new experiment data lands:
uv run python artifacts/pcz-ppo/paper/paper_build.py --build

# Also compile the PDF:
uv run python artifacts/pcz-ppo/paper/paper_build.py --build --pdf

# Inspect the DAG:
uv run python artifacts/pcz-ppo/paper/paper_build.py --dag

# Verify nothing is stale (run automatically by pre-commit):
uv run python artifacts/pcz-ppo/paper/paper_build.py --check
```

## The DAG

| Stage | Inputs | Outputs |
|---|---|---|
| `fragments` | `results.csv` + `fig_data.py` + `render_claims.py` | `generated/*.tex` |
| `fig_<name>` (×N) | `fig_<name>.py` + `fig_data.py` + declared `INPUTS` | `fig_<name>.{pdf,png}` (or declared `OUTPUTS`) |
| `pdf` (opt-in) | `pcz_ppo.tex` + all `generated/*.tex` + all `fig_*.pdf` + `references.bib` | `pcz_ppo.pdf` |

Run `paper_build.py --dag` to see the concrete input/output lists.

## How each figure declares its data dependencies

Every `fig_*.py` has a module-level `INPUTS` list with glob patterns
relative to the paper directory:

```python
# In fig_learning_curves_4M.py
INPUTS = [
    "../data/results.csv",
    "../data/metrics/*_torchrl-pcz-ppo-running_lunarlander_*.parquet",
    "../data/metrics/*_torchrl-ppo_lunarlander-k8_*.parquet",
]
```

`paper_build.py` AST-parses this without importing the script, hashes
the matching files, and marks the figure stale when the hash changes.

**Principle**: declare only what the script actually reads. A narrow
glob means unrelated parquet files don't invalidate the figure. The
repo has ~400 parquets but most figures only read `results.csv`.

### Optional: `OUTPUTS` override

By default, the output set is `fig_<name>.pdf` + `fig_<name>.png`. If
your script emits something else, declare it:

```python
# In fig_ablation_bar.py — writes to fig_ablation.pdf (legacy name)
OUTPUTS = [
    "fig_ablation.pdf",
    "fig_ablation.png",
]
```

## The lock file

`paper_build.lock.json` maps each stage name to its input hash. It is
**committed to git**. This is what `--check` compares against.

When data or a figure script changes, the next commit that touches the
paper tree will fail pre-commit until you run `--build` and commit the
updated artifacts + lockfile together. Same pattern as `package-lock.json`.

## Enforcement layers

| Layer | What | When |
|---|---|---|
| `--check` | Hash comparison against lockfile | Pre-commit hook `pcz-paper-build-check` on any change to `fig_*.py`, `fig_data.py`, `render_claims.py`, `paper_build.py`, `results.csv`, `data/metrics/*.parquet`, `llm_alignment/runs/*.json`, `generated/*.tex`, or `fig_*.{pdf,png}` |
| `--build` | Rebuild stale stages, refresh lockfile | Manual, after experiments |
| `--build --pdf` | As above + `latexmk` | Manual, before PR / review |
| CI | `--build --pdf` | PR / merge (when set up) |

**Pre-commit does NOT compile the PDF.** LaTeX is too slow and too
flaky for a commit-time gate. CI is where PDF freshness gets enforced.

## Workflow: experiment → paper update

1. Run experiments. MLflow logs to Postgres/MinIO.
2. Export to the data tier:
   ```bash
   uv run python -m core.plot.export_results \
       --tracking-uri http://<IP>:5050 \
       --output artifacts/pcz-ppo/data/results.csv --append
   uv run python -m core.plot.export_metrics \
       --tracking-uri http://<IP>:5050 \
       --output-dir artifacts/pcz-ppo/data/metrics --append
   ```
3. Rebuild the paper:
   ```bash
   uv run python artifacts/pcz-ppo/paper/paper_build.py --build --pdf
   ```
4. Commit the updated `results.csv`, parquets, `generated/*.tex`,
   `fig_*.{pdf,png}`, `paper_build.lock.json`, and `pcz_ppo.pdf` together.
   Pre-commit verifies the lock matches the data.

If you forget step 3, the next commit touching paper files will fail
with a clear message pointing at the stale stage(s).

## Adding a new figure

1. Create `artifacts/pcz-ppo/paper/fig_myplot.py`. Save as
   `fig_myplot.pdf` + `fig_myplot.png` (default naming).
2. Declare inputs:
   ```python
   INPUTS = ["../data/results.csv"]
   ```
3. Run `paper_build.py --build`. It discovers the new figure, runs it,
   and adds a lockfile entry.
4. Commit the new script, its PDF/PNG, and the updated lockfile.

## Adding a new numerical claim to the paper

No change here — the existing `render_claims.py` workflow applies. Add
the registration, run `render_claims.py`, commit the new fragments.
`paper_build.py --check` will verify the `fragments` stage after.

## Porting to another paper

`paper_build.py` is 90% project-agnostic. To port:

1. Copy `paper_build.py` to the new paper dir.
2. Edit the four constants at the top (`MAIN_TEX`, `BIB`, the
   `results.csv` path in `fragments_stage()`, `PDF_COMPILE_CMD`).
3. Add `INPUTS` declarations to every figure script.
4. Run `paper_build.py --build` to seed the lockfile.
5. Add a pre-commit hook entry pointing at the new path.

A future `paper_build.toml` manifest (planned as a project skill) will
remove steps 1–2.

## Troubleshooting

**`STALE paper artifacts detected` on commit, but I didn't change anything
paper-related.** A parquet, `results.csv`, or `fig_data.py` got touched
by another commit. Run `--build` to refresh outputs + lockfile.

**Fresh clone: every stage reports stale.** The lockfile was committed
but the LFS parquets weren't pulled yet. Run `git lfs pull` first.

**A figure regenerates every time even though nothing changed.** The
script has non-determinism (e.g., matplotlib's default random colors,
timestamps baked into the PDF). Make the script deterministic — pin
seeds, strip PDF metadata (`latexmk -pdflatex='pdflatex -output-format=pdf'`
or post-process with `qpdf`).

**Can I see which inputs changed?** `--check` prints the stage and
reason. For deeper debugging:
```bash
uv run python artifacts/pcz-ppo/paper/paper_build.py --dag | less
```

## Design notes

* **Why content hashing, not mtime?** `git checkout` scrambles mtimes.
  Make-style mtime DAGs would rebuild everything on branch switch.
* **Why one lockfile, not per-artifact sidecars?** Smaller diffs, single
  source of truth, easy to inspect.
* **Why not DVC?** For one paper with MLflow already handling training
  lineage, DVC adds a tool without removing anything. Honest tradeoff
  Revisit
  when paper #2 exists.
* **Why is PDF compile not in pre-commit?** 30–60 s runtime, occasional
  LaTeX warnings-as-errors. Pre-commit needs to be fast (<5 s total) to
  stay useful. CI is the right place for PDF enforcement.
