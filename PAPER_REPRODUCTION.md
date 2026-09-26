# Reproducing the revised paper figures

The 33 revised September 26 manuscript figures can be generated from **Resample-Protocol alone**,
plus the numeric data bundle. The paper checkout and its former standalone plotting scripts
are not required. The implementation is maintained in `src/lib/paper_revision/`; the CLI is
`src.scripts.visualizations.paper_revision`, also exposed through `make_all`.

The command recomputes 24 plotting tables from 141,336 pair records and 386,116 re-roll records,
including their question-cluster confidence intervals. It also recomputes the dose-response,
baseline crosscheck and role-by-stability tables, then produces **33 vector PDFs and three
LaTeX tables**. It never generates or re-judges a response. The three κ graphics redraw existing
audit summaries; they are not a new validation of the judge.

## Required data

Use `revision_analysis_bundle_2026-09-26.tar.gz`, or its extracted `revision_analysis/` directory.
The bundle is a separate research-data artifact and must accompany the paper's public data
release; it is not downloaded automatically or embedded in this code repository.

- Archive SHA-256: `d9ffbb1ff06e56be2302eaa532102a667cb0a7d2e8ca29d486a93737977f1ea1`
- Manifest SHA-256: `cf4321ae6389b8eef26adf6bbf12de7d65f9d383d4807e4e064fb34709593466`
- Every one of the 101 manifest-listed files is verified. Files in the bundle's `scripts/`
  directory are preserved as provenance and **never imported or executed** by this workflow.
- `pairs_master.parquet` and `rerolls_long.parquet` are the numerical calculation inputs.
  Archived aggregate CSVs are independent regression references, not substituted outputs.
  Raw reasoning traces and original judge-response files are not required for these plots.

The original released rollout tree remains the source for the retained legacy figures and for
upstream manifest reconstruction. This command begins at the frozen pair/re-roll label records;
it does not claim to reconstruct those records from raw model text or validate the labels' truth.

## CPU-only setup and command

A dedicated environment avoids installing the GPU/model stack solely to draw figures.
From the repository root, with Python 3.12 and `uv` installed:

```bash
uv venv --python 3.12 .venv-paper
uv pip install --python .venv-paper/bin/python -r requirements/paper-revision.txt

.venv-paper/bin/python -m src.scripts.visualizations.make_all \
    --paper-revision-bundle /path/to/revision_analysis_bundle_2026-09-26.tar.gz \
    --out-root /path/to/fresh-paper-output --dry-run

.venv-paper/bin/python -m src.scripts.visualizations.make_all \
    --paper-revision-bundle /path/to/revision_analysis_bundle_2026-09-26.tar.gz \
    --out-root /path/to/fresh-paper-output
```

The equivalent direct command is `python -m src.scripts.visualizations.paper_revision
--bundle /absolute/path/to/bundle --out-root /absolute/path/to/output`.
Paths may also use a quoted `${DATA_ROOT}` prefix. The output directory must be empty and outside
the input bundle; all calculations and extracted inputs are written there. `--dry-run` verifies
the data and prints the plan without creating outputs. No GPU or API credentials are needed.
Existing full project environments can install dependencies with `uv sync`, but the dedicated
pinned environment above is the reference environment for the paper reproduction.

## Outputs and verification

| Output | Contents |
| --- | --- |
| `figures/` | 33 PDFs, with the same filenames used by the manuscript |
| `tables/` | `rates_main.tex`, `benchmark_main.tex`, `benchmark_appendix.tex` |
| `analysis/results/` | Recomputed plotting CSVs and read-only copies of the two numerical inputs |
| `data/` | Additional computed CSVs, count audit and table-comparison report |
| `reproduction_manifest.json` | Input/code/asset/output hashes, dependency versions and completed-run status |

The archived question-cluster design uses 10,000 replicates and seed 42, resampling the 3,982
canonical questions within dataset and moving all models, cues, originals and re-rolls together.
The weight-matrix fingerprint must equal `1a8b5e9a029dc3ce`. Each rate comparison shares the same
draws. The calculation preserves production missing-as-miss coding; sensitivity bounds in the
paper describe the limits of that convention.

Recomputed reference tables are checked for row/column identity, matching missing-value masks
and numerical agreement (counts exactly; floating-point values with absolute tolerance
`2e-7` and no relative tolerance). Rates, counts and intervals match the supplied reference within
`2.906e-8` in the tested environment. This observed difference reflects float32 BLAS rounding and is
far below the paper's displayed precision. All 440 benchmark rows and the release accounting
are additionally checked directly against the long-format re-roll records.

### Exact signs for the alpha comparison

The archived alpha sign-tail calculation subtracted two float32 ratios with the same denominator.
A mathematical tie can acquire a tiny sign depending on BLAS summation order. This port determines
the sign from an integer-scaled numerator, using the least common multiple of `n_options - 2`, and
includes exact ties in both tails. The formula for the statistical test and its treatment of
undefined replicates are unchanged.

This numerical correction changes six non-recurrence p-values (maximum absolute change 0.0066)
and four BH q-values (maximum 0.006803). Every reported 0.05 significance decision remains the
same: non-persistence has 257 below/0 above BH discoveries; non-recurrence has 83 below/3 above.
The comparison report records each corrected value. Point estimates and plotted confidence
intervals are unchanged at paper precision. The original bundle is left intact.

### Figure map

| Library module | Manuscript assets |
| --- | --- |
| `survival.py` | Main stability plot, cue/dataset/slice survival, four reliance compositions, re-roll histogram and yield (12 PDFs) |
| `rates.py` | Pooled/dataset/cue rate plots and susceptibility (5 PDFs), plus three LaTeX tables |
| `alpha_roles.py` | Alpha comparisons and separate persistence categories (9 PDFs) |
| `crosscheck.py` | Baseline-prediction versus persistence/non-recurrence (2 PDFs) |
| `dose.py` | Separate incoherent/rejected/verification-only recurrence curves (2 PDFs) |
| `judge_kappa.py` | Existing eight-cue and role audit summaries (3 PDFs) |

Figures use bundled Matplotlib fonts and deterministic PDF creation metadata. Exact PDF bytes
are expected to repeat in the pinned environment. Across platforms, compare numerical tables
and rendered figures rather than timestamps or PDF object ordering. The earlier paper assets
were checked against the repository renderers: all 33 rendered figures match, and the three
LaTeX tables match byte-for-byte. The κ JSON includes its source citations and count scope;
raw role-audit records were unavailable, so those κ values are transparently reproduced from
published summaries.

The token-cost funnels, role heatmaps, distractor/rank plots and J2–J4 retain their existing
generators under `src/scripts/visualizations/`. Use the original `make_all`, individual
`gather`/`plot`, and `judge_validation_plots` commands for them. The overview diagram is a
retained released asset; this repository does not contain its generator. These assets are
outside the 33-figure revision workflow.

## Tests

```bash
.venv-paper/bin/python -m unittest \
    tests.lib.paper_revision_recompute_test \
    tests.lib.paper_revision_workflow_test \
    tests.lib.paper_revision_verification_test \
    tests.scripts.visualizations.make_all_test -v
```

These tests cover shared question-level resampling, exact-zero sign tails, exclusive role
categories, an end-to-end synthetic numerical fixture, input integrity, dry-run behavior and
protection of the read-only input tree, and dispatch through the existing plotting command. The real-bundle run also performs the full numerical
regression and count audit before rendering.
