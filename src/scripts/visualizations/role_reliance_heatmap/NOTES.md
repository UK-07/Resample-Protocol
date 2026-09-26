# Role x reliance heatmap

Does what the CoT *claims* about the hint (the v2 judge's role) track what the model *does* with it across
re-rolls (the reliance label)? One panel per model; rows = the five v2 roles + "incoherent (−1)"; columns = the
pair's reliance label; cell = share of the column's kept rollouts.

## Run

```bash
python -m src.scripts.visualizations.role_reliance_heatmap.gather   # tables -> <cueball>/plots/role_reliance_heatmap/data/
python -m src.scripts.visualizations.role_reliance_heatmap.plot     # figures -> <cueball>/plots/role_reliance_heatmap/
```

`gather` flags: `--cueball-dir` (default `${DATA_ROOT}/cueball`), `--out-dir`, `--only <substr>` / `--models a,b`
(debug subsets of the models; written under `<plot dir>/_debug` unless `--out-dir` is given),
`--minus1 own_category|exclude` (`exclude` is the superseded sensitivity rule — −1 excluded — and must go to a
separate `--out-dir`), and the uniform cache flags `--no-cache` / `--refresh-cache` / `--cache-dir` (accepted,
unused: this gather reads the resample manifest directly and keeps no cache). `plot` flags: `--data` (default
`<out-dir>/data`), `--out-dir`. The pool logic lives in `common/section3_claims.py`; its sha256, the repo git sha,
the shas of the repo code it depends on and both inputs' sha256 are recorded in `data/gather_meta.json`.

## Files

| file | role |
| --- | --- |
| `gather.py` | builds every table in `data/` from the resample manifest + `question_reliance.csv` |
| `plot.py` | reads only `data/`; writes `role_reliance_heatmap{,_positive,_negative}.{png,pdf}` |
| `data/role_share_by_model[_case\|_provenance].csv` | long table: model (× case / provenance) × column_key × category → `n`, `n_column`, `rate` (share of column), Wilson `ci_lo`/`ci_hi` |
| `data/column_summary[_by_case].csv` | per model (× case) × column: `n_pairs`, on-target rollouts, kept rollouts |
| `data/exclusions_by_model[_case\|_column\|_case_column\|_provenance].csv` | on-target rollouts, kept (of which −1), excluded: no role, unjudged, truncated |
| `data/minus1_by_role_by_model[_provenance].csv`, `data/minus1_by_role.md` | every v2 −1 rollout of the pool by the role the judge also emitted (`no_role` if none), `n_minus1`, `n_judged`, `share_minus1` |
| `data/gather_meta.json` | inputs (sha256), code shas, join checks, `minus1_rule`, the column keys and categories |

## Definitions

- **Inputs**: `resample/resample_manifest.parquet` (the hinted-once original of every re-sampled pair and the
  k = 4 re-rolls) and `resample/question_reliance.csv`. The five binary-judged models (Qwen3.6-27B excluded), no
  smoke runs, the 8 styles.
- **Role pool** (`common/section3_claims.py::load_role_pool`): every rollout with `to_hint` True (answer = hint
  target, i.e. an on-target rollout) — the original (`provenance == hinted_once`) and every on-target re-roll
  (`resample_k4`) — each counted once (per-rollout weighting). Pair key: the original's `rollout_id` / the
  re-roll's `source_rollout_id`; joined to `question_reliance.csv` (checked: every row joins; the manifest's copy
  of `reliance_label` and selection `role` equal the CSV's; every pair has `k_n = 4`).
- **Row category** = v2 `judge_role` ∈ {credited, verification_only, rejected, neutral, none}, except that a
  rollout with v2 verdict −1 (`judge_label_final == -1`) goes to the row **incoherent (−1)** whatever role it
  also carries (with or without a role).
- **Excluded from every denominator** (and counted): verdict missing (never judged); verdict 0/1 without a role
  (judged before the role-emitting prompt); truncated on-target rollouts. `exclude_reason` is not applied
  otherwise.
- **Columns** = the pair's `reliance_label`: `robust_used` (≥ 3/4 re-rolls on target), `weak_used` (1–2/4),
  `mixed` (0/4 — a used_candidate pair: the original switched but no re-roll did; its only on-target rollout is
  the original), `robust_ignored` (control pairs, 0/4 on target and ≥ 3/4 on the modal answer — no on-target
  rollout, so never judged: the column is greyed and labelled "not judged (control)", with its pair count).
  Control pairs labelled `mixed` (the original did not switch, but 1–4 re-rolls did) are NOT in the `mixed`
  column; they are tabulated as `mixed_control` in `data/` only.
- **Cell** = n(category, column) / n(column), both over kept rollouts; `n` printed in every cell and the column
  total under each column label. Colour on a √ scale (credited dominates). Columns with total n < 20 are faded.
- Case: the pooled figure pools both manifest cases; `_positive` / `_negative` restrict to `case`.
- The figure footer counts the drawn columns only (`data/exclusions_by_model_column.csv`); the whole-pool
  counts, including the undrawn control-mixed re-rolls, are in `data/exclusions_by_model.csv`.

## Caption

Claims versus behaviour. For every (question, cue) pair that was re-sampled, each on-target rollout (the original
hinted rollout and every re-roll that answered the cue's target) is classified by the v2 judge's role for the cue:
credited, verification only, rejected, neutral mention, no mention, or incoherent. Columns group rollouts by how
often the cue was followed in 4 re-rolls of the same prompt (robust: ≥ 3/4; weak: 1–2/4; mixed: 0/4 although the
original followed it); cells give the share of each column (n rollouts below). Controls (robust ignored: 0/4 on
target) have no on-target rollout and are not judged. Rollouts the judge marked incoherent (−1: the reasoning
concludes a different option than the answer given) form their own row. Rollouts judged without a role (mostly
originals judged before the role-emitting prompt) and truncated rollouts are excluded; counts in the footer.
