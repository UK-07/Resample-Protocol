"""Independently reaggregate this edition's label records and release splits."""
from pathlib import Path
import hashlib
import math
import pandas as pd

def validate_counts(results: Path, report_path: Path):
    DATA = results
    report_path.parent.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(DATA / "pairs_master.parquet")
    r = pd.read_parquet(DATA / "rerolls_long.parquet")
    p["clean_local"] = ~p.truncated.fillna(False).astype(bool) & p.model_answer.notna()
    p["eligible_clean_local"] = p.clean_local & (p.ssp_status == "eligible")
    r["qkey"] = r.dataset.astype(str) + ":" + r.original_index.astype(int).astype(str)
    assert p.rollout_id.is_unique and r.rollout_id.is_unique
    assert r.source_rollout_id.isin(p.rollout_id).all()
    assert (p.persist3 == (p.k_hit >= 3)).all()

    # Verify pair-level counters against the independent long-format records.
    for pair_col, rr_col in [("k_hit", "hit"), ("k_rsp_pool", "hit_rsp_pool"), ("k_rsp_unf", "hit_rsp_unf")]:
        reaggregated = r.groupby("source_rollout_id")[rr_col].sum().reindex(p.rollout_id, fill_value=0).to_numpy()
        assert (p[pair_col].fillna(0).to_numpy() == reaggregated).all(), pair_col
    assert (r.hit_rsp_unf == (r.hit_rsp_pool & r.lab.isin([0, -1]))).all()
    assert ((~r.hit_rsp_pool) | r.lab.isin([0, 1, -1])).all()

    flips = p[p.ssp_flip]
    binary = flips[flips.binary_judge_label.isin([0, 1])]
    common = binary[binary.role_label.isin([0, 1, -1])]
    filtered = common[common.k_hit >= 3]
    pool = r[r.hit_rsp_pool].copy()
    pool["unfaithful"] = pool.lab.isin([0, -1]).astype(int)
    fresh = pool[pool.source_rollout_id.isin(filtered.rollout_id)]
    paired_originals = filtered[filtered.rollout_id.isin(fresh.source_rollout_id)]
    assert len(filtered) == len(paired_originals) == 53281
    assert len(common) == 65752
    assert (common.binary_judge_label == 0).sum() == 5648
    assert common.role_label.isin([0, -1]).sum() == 6446
    assert paired_originals.role_label.isin([0, -1]).sum() == 4479
    assert (len(binary), int((binary.binary_judge_label == 0).sum())) == (65772, 5656)

    stage_inputs = [
        ("S1", float((common.binary_judge_label == 0).sum()), len(common)),
        ("S2", float(common.role_label.isin([0, -1]).sum()), len(common)),
        ("S3", float(paired_originals.role_label.isin([0, -1]).sum()), len(paired_originals)),
        ("S4", float(fresh.groupby("source_rollout_id").unfaithful.mean().sum()), fresh.source_rollout_id.nunique()),
        ("S5", float(pool.groupby("source_rollout_id").unfaithful.mean().sum()), pool.source_rollout_id.nunique()),
        ("S6", float(pool.unfaithful.sum()), len(pool)),
    ]
    published = pd.read_csv(DATA / "followup/decomposition_stage_rates.csv")
    for name, num, den in stage_inputs:
        row = published[(published.grouping == "pooled") & published.stage.str.startswith(name + " ")].iloc[0]
        assert math.isclose(num, row.num, abs_tol=1e-10)
        assert den == row.den
        assert math.isclose(num / den, row.rate, abs_tol=1e-12)

    # Verify each SSP/RSP count and point estimate used by the figure and tables.
    checked_cells = 0
    for file, keys in [
        ("benchmark_cells_model_cue_case_clusterCI.csv", ["subject_model", "hint_style", "case"]),
        ("benchmark_cells_model_dataset_case_clusterCI.csv", ["subject_model", "dataset", "case"]),
        ("benchmark_cells_model_dataset_cue_case_clusterCI.csv", ["subject_model", "dataset", "hint_style", "case"]),
    ]:
        table = pd.read_csv(DATA / "plot_tables_followup" / file)
        bgroups = dict(tuple(binary.groupby(keys, observed=True)))
        rgroups = dict(tuple(pool.groupby(keys, observed=True)))
        pgroups = dict(tuple(p.groupby(keys, observed=True)))
        for _, row in table.iterrows():
            key = tuple(row[k] for k in keys)
            b = bgroups.get(key, binary.iloc[:0])
            q = rgroups.get(key, pool.iloc[:0])
            original = pgroups[key]
            values = {"ssp_unf": (b.binary_judge_label == 0).sum(), "ssp_labeled": len(b),
                      "rsp_unf_incl_minus1": q.unfaithful.sum(), "rsp_judged": len(q), "rsp_minus1": (q.lab == -1).sum(),
                      "robust_clean": (original.robust_used & original.clean_local).sum(), "n_clean": original.clean_local.sum(),
                      "ssp_flip_clean": (original.ssp_flip & original.clean_local).sum(), "n_eligible_clean": original.eligible_clean_local.sum()}
            for field, value in values.items():
                assert int(row[field] if pd.notna(row[field]) else 0) == int(value), (file, key, field)
            for metric, num, den in [("ssp_rate", "ssp_unf", "ssp_labeled"), ("rsp_rate", "rsp_unf_incl_minus1", "rsp_judged"),
                                     ("robust_susc", "robust_clean", "n_clean"), ("ssp_susc", "ssp_flip_clean", "n_eligible_clean")]:
                assert (math.isclose(row[metric], values[num] / values[den], abs_tol=1e-12)
                        if values[den] else pd.isna(row[metric])), (file, key, metric)
            checked_cells += 1

    release = r[r.hit_rsp_unf]
    assert len(release) == 20000
    assert release.source_rollout_id.nunique() == 9997
    assert release.qkey.nunique() == 2610
    assert (release.lab == -1).sum() == 1223
    assert (release.groupby("qkey").split.nunique() == 1).all()
    assert (release.groupby("source_rollout_id").split.nunique() == 1).all()
    all_splits = pd.concat([p[["qkey", "split"]], r[["qkey", "split"]]])
    assert (all_splits.groupby("qkey").split.nunique() == 1).all()
    split_rows = []
    for split in ["train", "val", "validation", "test"]:
        s = release[release.split == split]
        if len(s): split_rows.append((split, len(s), s.source_rollout_id.nunique(), s.qkey.nunique()))

    report = ["# Local validation of the supplied revision archive", "",
              "Computed locally from `pairs_master.parquet` and `rerolls_long.parquet` using `src.lib.paper_revision.validation.validate_counts`. "
              "This independently reaggregates the supplied label records; raw model traces and original judge outputs were not supplied, "
              "and confidence intervals were not recomputed in this check.", "",
              f"- Pair table: {len(p):,} unique original rollout IDs.",
              f"- Re-roll table: {len(r):,} unique rollout IDs; every source ID exists in the pair table.",
              "- Pair hit counts, published-pool counts, and unfaithful re-roll counts exactly match the long-format records.",
              "- Published SSP rate population: 5,656 / 65,772; exact common-judge population: binary 5,648 / 65,752, role 6,446 / 65,752.",
              "- Persistence-filtered common-ID population: 53,281 pairs, 4,479 role-unfaithful originals; all have a published-pool re-roll.",
              f"- Independently reaggregated SSP/RSP unfaithfulness and susceptibility counts and rates for all {checked_cells} exported benchmark rows (80 cue, 40 dataset, 320 detailed rows). Every comparison passed.",
              "", "## Telescoping path", "", "| Stage | Numerator | Denominator | Rate (%) |", "|---|---:|---:|---:|"]
    for stage, num, den in stage_inputs:
        report.append(f"| {stage} | {num:,.2f} | {den:,} | {100*num/den:.6f} |")
    report += ["", "S4/S5 numerators are sums of within-pair proportions, so fractional values are expected. "
               "All six points exactly match the follow-up CSV. The C subset has "
               f"{len(fresh):,} judged re-rolls and {int(fresh.unfaithful.sum()):,} unfaithful re-rolls.", "",
               "## Release and split checks", "", "The release contains **20,000 traces, 9,997 source pairs, and 2,610 canonical questions**; "
               "1,223 traces have incoherent verdicts. No canonical question or source pair crosses a split, either in the release "
               "or in the broader archived pair/re-roll frame.", "",
               "| Split | Traces | Source pairs | Canonical questions |", "|---|---:|---:|---:|"]
    for split, traces, pairs, questions in split_rows:
        report.append(f"| {split} | {traces:,} | {pairs:,} | {questions:,} |")
    report += ["", "## Input fingerprints", ""]
    for name in ["pairs_master.parquet", "rerolls_long.parquet"]:
        report.append(f"- `{name}`: `{hashlib.sha256((DATA/name).read_bytes()).hexdigest()}`")
    report.append("")
    report_path.write_text("\n".join(report))
    return report
