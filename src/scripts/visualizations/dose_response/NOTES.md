# Dose-response

Question: as behavioural reliance on the cue rises (more of the 4 re-rolls follow it), do claims that the cue was
*not* relied on (rejected / verification only / incoherent) become rarer? If they stay frequent, the claims are
uninformative about behaviour. Weighting is **per rollout, originals and re-rolls**.

## Run

```bash
python -m src.scripts.visualizations.dose_response.gather   # tables -> <cueball>/plots/dose_response/data/
python -m src.scripts.visualizations.dose_response.plot     # figures -> <cueball>/plots/dose_response/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only <substr>` / `--models a,b`
(debug subsets of the models; without `--out-dir` they are written under `<plot dir>/_debug`), `--minus1
own_category` (the definition, default) | `exclude` (SENSITIVITY ONLY — the superseded rule; write to a separate
`--out-dir`), and `--no-cache` / `--refresh-cache` / `--cache-dir` (accepted for uniformity with the other gathers;
this one reads its two inputs directly and uses no cache). `plot` flags: `--data` (default `<out-dir>/data`),
`--out-dir`. The pool lives in `common/section3_claims.py`; its sha256, the repo git sha, the shas of the repo
code it depends on and both inputs' sha256 are recorded in `data/gather_meta.json`.

## Files

| file | role |
| --- | --- |
| `gather.py` | builds `data/` from `resample/resample_manifest.parquet` + `resample/question_reliance.csv` |
| `plot.py` | reads only `data/`; writes `dose_response.{png,pdf}` (primary: used pairs, cases pooled) and `dose_response_by_case.{png,pdf}` |
| `data/dose_by_model[_case\|_provenance\|_style].csv` | per `scope` × model (× group) × k: `n_pairs`, `n_kept`, `n_none` (non-mentions, left out), `denominator` (mentions), `numerator` (rejected + verification_only + −1), per-category counts incl. `n_incoherent`, `rate` + Wilson CI, and the same for rejected / verification_only / incoherent alone |
| `data/exclusions.csv` | per scope × model (× k / × case): on-target, kept, kept −1, excl. no role, excl. unjudged, excl. truncated |
| `data/minus1_by_role_by_model.csv` | v2 −1 rollouts by the role the judge also emitted |
| `data/gather_meta.json` | inputs (sha256), checks, `minus1_rule`, code shas |

## Definitions

- **Pool**: the claims-vs-behaviour role pool (`common/section3_claims.py`, shared with the role × reliance
  heatmap): every on-target (`to_hint`) rollout of a re-sampled (question, hint) pair — the original and the
  on-target re-rolls — 5 models, no smoke runs, 8 styles. Category = v2 role, or "incoherent" for a v2 −1
  rollout (whatever role it also carries). Excluded and counted: no v2 verdict, 0/1 verdict without a role,
  truncated rollouts.
- **x** = the pair's `k_to_hint_count` (re-rolls of 4 that answered the target; `question_reliance.csv`, every
  pair has k_n = 4). Checked: a pair at x = k has exactly k on-target re-rolls in the pool.
- **Denominator** = kept rollouts that mention the hint (category ≠ `none`: credited, verification_only, rejected,
  neutral, incoherent). **Numerator** = category ∈ {rejected, verification_only, incoherent (−1)}.
  y = numerator / denominator, Wilson 95 %.
- **Weighting**: per rollout. A used pair at x = k contributes its original + k re-rolls (so x = 0 is originals
  only, and x = 4 has five rollouts per pair); rollouts of one pair are not independent, so the Wilson whiskers
  are narrower than a question-clustered interval would be.
- **Scope** (`scope` column): `used_pairs` = pairs whose original switched (selection role used_candidate) —
  **the figures**; `all_pairs` (tables only, not drawn) adds control pairs whose re-rolls reached the target.
- Points with denominator < 20 are faded. Lines are dodged horizontally by 0.06 so whiskers do not overlap.
- Cases pooled in the primary figure; `dose_response_by_case` splits by manifest `case`.

## Exclusions

The figure footers print, per model, the used_pairs scope's kept −1 / no role / unjudged / truncated counts
(`data/exclusions.csv`, grouping `model`). Truncated rollouts are excluded everywhere.

**Population caveat (role coverage).** v2 roles are missing mostly on the ORIGINAL rollouts of two models (they
were judged before the role-emitting prompt): most of Qwen3.5-9B's on-target originals and about a third of
Qwen3-8B's have no role and are excluded; re-rolls are almost fully covered. For these two models the
original/re-roll mix is therefore skewed towards re-rolls, and the x = 0 point (originals only) rests on the
minority of originals that carry a role. The other three models are nearly fully covered.

## Caption

Non-reliance claims versus behavioural reliance. For each re-sampled (question, cue) pair whose original rollout
followed the cue, x is the number of 4 re-rolls of the same prompt that followed it again. y is the share of
on-target rollouts (the original and those re-rolls, each counted once) that disclaim reliance on the cue —
rejecting it, using it only to verify an independently reached answer, or reasoning to a different option than the
one given (judge verdict incoherent) — among all rollouts that mention the cue. Whiskers: Wilson 95 % intervals
(rollouts of a pair are not independent); faded points have n < 20. Rollouts judged without a role and truncated
rollouts are excluded (footer); roles are missing on most Qwen3.5-9B originals and on about a third of Qwen3-8B's,
so their x = 0 points rest on the covered minority.
