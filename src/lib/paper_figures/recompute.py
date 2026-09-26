"""Compute paper figure tables from row-level data.

Inputs are the pairs and re-roll parquet tables. All rates and intervals are
calculated here.

Every calculation shares the same stratified question-cluster bootstrap weights.
The historical ``wilson_*`` columns are retained solely for reference-table
compatibility; the renderers use the question-cluster interval columns.

Alpha sign-tail p-values use an integer-scaled contrast numerator. Subtracting two
rounded bootstrap ratios can move mathematically exact zeros to either side of
zero, making p-values depend on the BLAS implementation. The rates share one
positive denominator, so integer numerator signs preserve the intended inclusive
zero-tail definition exactly. Point estimates and percentile intervals use the
bootstrap ratios directly.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import common as rc


ALPHA_SIGN_METHOD = "LCM-scaled integer contrast numerator; exact float64 sums; inclusive zero tails"


def alpha_contrast_pvalues(bs, rows, q_counts, measured_counts, keys):
    """Evaluate alpha-minus-measured signs without floating-ratio cancellation.

    Both rates have the same positive flip denominator. Multiplying their
    numerator difference by LCM(n_options - 2) makes each row contribution an
    integer. All integer sums are exact in float64 under the explicit bound
    below. Undefined (zero-flip-denominator) replicates remain undefined.
    """
    divisors = rows.n_options.to_numpy(dtype=float) - 2
    if not np.all(np.isfinite(divisors) & (divisors > 0) & (divisors == np.rint(divisors))):
        raise ValueError("Alpha normalization requires integer option counts above two")
    integers = divisors.astype(np.int64)
    scale = math.lcm(*(int(value) for value in np.unique(integers)))
    q = np.asarray(q_counts, dtype=float)
    measured = np.asarray(measured_counts, dtype=float)
    if not np.all(np.isfinite(q) & np.isfinite(measured) & (q == np.rint(q)) & (measured == np.rint(measured))):
        raise ValueError("Alpha contrast contributions must be integer counts")
    scaled = q * (scale // integers) - measured * scale
    groups = rows[keys].astype(str).agg("|".join, axis=1) if keys else pd.Series("all", index=rows.index)
    numerators, levels = bs.per_question_groups(rows, scaled, groups)
    denominators, denominator_levels = bs.per_question_groups(rows, rows.ssp_flip.astype(float), groups)
    if levels != denominator_levels:
        raise ValueError("Alpha numerator and denominator group levels differ")
    # Every bootstrap row contains exactly Q question draws across its strata;
    # this bounds every absolute intermediate sum, not just its final value.
    if not np.all(numerators == np.rint(numerators)) or bs.Q * np.max(np.abs(numerators)) >= 2 ** 53:
        raise ValueError("Alpha contrast exceeds the exact float64 integer range")
    contrasts = bs.W @ numerators.astype(np.float64)
    denominator_totals = bs.W @ denominators.astype(np.float32)
    contrasts = np.where(denominator_totals > 0, contrasts, np.nan)
    return bs.pvalue(contrasts)


def persistence_tables(df, bs, output: Path) -> None:
    """Threshold survival and alpha contrasts with shared clustered draws."""
    OUT = output / "s1_persistence"
    OUT.mkdir(parents=True, exist_ok=True)
    flips = df[df.ssp_flip].copy()
    if not (flips.k_n == 4).all():
        raise ValueError("The paper persistence analysis requires four fresh re-rolls per SSP flip")
    pops = {"ssp_flips": flips, "ssp_unfaithful": flips[flips.ssp_unfaithful]}
    def count_table(d, keys):
        g = d.groupby(keys, observed=True) if keys else d.groupby(lambda _: "all")
        t = g.agg(n=("rollout_id", "size"), n_questions=("qkey", "nunique"),
                  k0=("k_hit", lambda s: int((s == 0).sum())), k1=("k_hit", lambda s: int((s == 1).sum())),
                  k2=("k_hit", lambda s: int((s == 2).sum())), k3=("k_hit", lambda s: int((s == 3).sum())),
                  k4=("k_hit", lambda s: int((s == 4).sum())),
                  n_missing_any=("k_missing", lambda s: int((s > 0).sum())), mean_k_eff=("k_eff", "mean")).reset_index()
        return t

    rows = []
    for pname, d in pops.items():
        for keys in ([], ["case"], ["subject_model", "case"], ["subject_model"]):
            ct = count_table(d, keys)
            for thr, col in ((1, "persist1"), (2, "persist2"), (3, "persist3")):
                rt = rc.ratio_table(bs, d, d[col].astype(float), np.ones(len(d)), keys, f"surv_ge{thr}")
                ct = ct.merge(rt[keys + [f"surv_ge{thr}", f"surv_ge{thr}_ci_lo", f"surv_ge{thr}_ci_hi"]], on=keys, how="left") if keys else \
                     rc.cat([ct.reset_index(drop=True), rt[[f"surv_ge{thr}", f"surv_ge{thr}_ci_lo", f"surv_ge{thr}_ci_hi"]]], axis=1)
            ct.insert(0, "population", pname)
            ct.insert(1, "grouping", "+".join(keys) if keys else "pooled")
            rows.append(ct)
    persist = rc.cat(rows, ignore_index=True)
    persist.to_csv(OUT / "persistence_counts_and_survival.csv", index=False)

    # detailed model x dataset x cue x case tables (point estimates + cluster CI for >=3/4)
    for pname, d in pops.items():
        ct = count_table(d, ["subject_model", "dataset", "hint_style", "case"])
        for thr, col in ((1, "persist1"), (2, "persist2"), (3, "persist3")):
            rt = rc.ratio_table(bs, d, d[col].astype(float), np.ones(len(d)), ["subject_model", "dataset", "hint_style", "case"], f"surv_ge{thr}")
            ct = ct.merge(rt[["subject_model", "dataset", "hint_style", "case", f"surv_ge{thr}", f"surv_ge{thr}_ci_lo", f"surv_ge{thr}_ci_hi"]],
                          on=["subject_model", "dataset", "hint_style", "case"], how="left")
        ct["small_n_lt_20"] = ct.n < rc.SMALL_N
        ct.to_csv(OUT / f"cells_{pname}_model_dataset_cue_case.csv", index=False)

    el = df[df.eligible_clean].copy()
    ans = el.model_answer.astype(str).str.upper()
    tg = el.target_option.astype(str).str.upper()
    b0 = el.b0.astype(str).str.upper()
    el["p_cnt"] = el.ssp_flip.astype(float)
    el["q_cnt"] = ((ans != tg) & (ans != b0)).astype(float)
    el["q_over_nm2"] = el.q_cnt / (el.n_options - 2)
    el["lit_cnt"] = (el.ssp_flip & (el.k_hit < 3)).astype(float)                        # non-persistent (<3/4), production coding
    el["strict_cnt"] = (el.ssp_flip & (el.k_hit == 0)).astype(float)                    # non-recurring, missing counted as miss (upper bound)
    el["strict_lower_cnt"] = (el.ssp_flip & (el.k_hit == 0) & (el.k_missing == 0)).astype(float)  # all four complete and off-target (lower bound)
    el["strict_obs_cnt"] = (el.ssp_flip & (el.k_hit == 0) & (el.k_eff >= 1)).astype(float)         # observed-only numerator
    el["strict_obs_den"] = (el.ssp_flip & (el.k_eff >= 1)).astype(float)
    alpha_tabs = []
    for keys in (["subject_model", "dataset", "hint_style", "case"], ["subject_model", "case"], ["subject_model"], ["case"]):
        A = rc.ratio_table(bs, el, el.q_over_nm2, el.p_cnt, keys, "one_minus_alpha")
        L = rc.ratio_table(bs, el, el.lit_cnt, el.p_cnt, keys, "nonpersistent_share")
        S = rc.ratio_table(bs, el, el.strict_cnt, el.p_cnt, keys, "nonrecurring_share_missing_as_miss")
        SL = rc.ratio_table(bs, el, el.strict_lower_cnt, el.p_cnt, keys, "nonrecurring_share_lower_complete_only")
        SO = rc.ratio_table(bs, el, el.strict_obs_cnt, el.strict_obs_den, keys, "nonrecurring_share_observed_only")
        E = el.groupby(keys).agg(n_eligible_clean=("rollout_id", "size"), n_questions=("qkey", "nunique"), p_count=("p_cnt", "sum"),
                                 q_count=("q_cnt", "sum"), n_flips_missing_any=("lit_cnt", lambda s: 0)).reset_index()
        t = E.merge(A.drop(columns=["num", "den", "n_questions", "undefined_reps"]), on=keys).merge(L.drop(columns=["num", "den", "n_questions", "undefined_reps"]), on=keys) \
             .merge(S.drop(columns=["num", "den", "n_questions", "undefined_reps"]), on=keys).merge(SL.drop(columns=["num", "den", "n_questions", "undefined_reps"]), on=keys) \
             .merge(SO.drop(columns=["num", "den", "n_questions", "undefined_reps"]), on=keys)
        t["one_minus_alpha_exceeds_1"] = t.one_minus_alpha > 1
        t["undefined_p_zero"] = t.p_count == 0
        t["small_n_lt_20_flips"] = t.p_count < rc.SMALL_N
        for lab, M, col in (("lit", L, "nonpersistent_share"), ("strict", S, "nonrecurring_share_missing_as_miss")):
            dr = A.attrs["boot"].reps - M.attrs["boot"].reps
            s = bs.summarize(dr, A.one_minus_alpha.to_numpy() - M[col].to_numpy())
            t[f"diff_alpha_minus_{lab}"] = s.point.to_numpy()
            t[f"diff_alpha_minus_{lab}_ci_lo"] = s.ci_lo.to_numpy()
            t[f"diff_alpha_minus_{lab}_ci_hi"] = s.ci_hi.to_numpy()
            measured_counts = el.lit_cnt if lab == "lit" else el.strict_cnt
            p = alpha_contrast_pvalues(bs, el, el.q_cnt, measured_counts, keys)
            t[f"p_{lab}"] = p
            t[f"q_bh_{lab}"] = rc.benjamini_hochberg(p)
        t.insert(0, "grouping", "+".join(keys))
        alpha_tabs.append(t)
    alpha = rc.cat(alpha_tabs, ignore_index=True)
    alpha.to_csv(OUT / "alpha_vs_measured_noise.csv", index=False)


def role_tables(df, bs, output: Path) -> None:
    """Exclusive original-role categories against the all-flip reference."""
    OUT2 = output / "s2_roles"
    OUT2.mkdir(parents=True, exist_ok=True)
    flips = df[df.ssp_flip].copy()
    def primary_cat(d):
        rl = d.role_label
        cat = np.where(rl == -1, "incoherent",
              np.where(rl.isna(), "unjudged",
              np.where(d.role_orig.isna(), "missing_role", "coherent_" + d.role_orig.astype(object).fillna("").astype(str))))
        return pd.Series(cat, index=d.index)
    flips["pcat"] = primary_cat(flips)
    flips["coherence"] = np.where(flips.role_label == -1, "incoherent(-1)", np.where(flips.role_label.isna(), "unjudged", "coherent(0/1)"))
    flips["emitted_role"] = flips.role_orig.astype(object).where(flips.role_orig.notna(), "no_role")
    # cross-tab emitted role x coherence per model (inclusive rejected = any flip whose emitted role is rejected)
    xt = flips.groupby(["subject_model", "emitted_role", "coherence"]).size().rename("n").reset_index()
    xt.to_csv(OUT2 / "crosstab_emitted_role_x_coherence_by_model.csv", index=False)
    incl = flips.groupby("subject_model")[["pcat", "emitted_role", "role_label"]].apply(lambda d: pd.Series({
        "ssp_flips": len(d), "role_covered(role or -1)": int(d.pcat.isin(["incoherent"]).sum() + d.pcat.str.startswith("coherent_").sum()),
        "missing_role": int((d.pcat == "missing_role").sum()), "unjudged": int((d.pcat == "unjudged").sum()),
        "incoherent(-1)": int((d.pcat == "incoherent").sum()), "coherent_rejected": int((d.pcat == "coherent_rejected").sum()),
        "coherent_verification_only": int((d.pcat == "coherent_verification_only").sum()),
        "rejected_inclusive(any coherence)": int((d.emitted_role == "rejected").sum()),
        "rejected_and_-1": int(((d.emitted_role == "rejected") & (d.role_label == -1)).sum()),
        "verification_only_inclusive": int((d.emitted_role == "verification_only").sum()),
        "verification_only_and_-1": int(((d.emitted_role == "verification_only") & (d.role_label == -1)).sum()),
        "-1_with_no_role": int(((d.role_label == -1) & (d.emitted_role == "no_role")).sum()),
    })).reset_index()
    incl.to_csv(OUT2 / "role_coverage_and_inclusive_counts_by_model.csv", index=False)

    # persistence >=3/4 by category, model, case; reference = all SSP flips of the same model x case (includes the subgroup)
    CATS = ["incoherent", "coherent_rejected", "coherent_verification_only", "coherent_credited", "coherent_neutral", "coherent_none"]
    rows = []
    for keys in (["subject_model", "case"], ["subject_model"], ["case"], []):
        ref = rc.ratio_table(bs, flips, flips.persist3.astype(float), np.ones(len(flips)), keys, "persist3")
        cov = flips[flips.pcat.isin(CATS)]
        for cat in CATS + ["covered_all"]:
            sub = cov if cat == "covered_all" else flips[flips.pcat == cat]
            if sub.empty: continue
            t = rc.ratio_table(bs, sub, sub.persist3.astype(float), np.ones(len(sub)), keys, "persist3")
            t = t.rename(columns={"num": "n_persist3", "den": "n"})
            # align to reference groups
            ref_idx = {l: i for i, l in enumerate(ref.attrs["boot"].levels)}
            lv = t.attrs["boot"].levels
            cols = [ref_idx[l] for l in lv]
            dr = t.attrs["boot"].reps - ref.attrs["boot"].reps[:, cols]
            rp = ref.persist3.to_numpy()[cols]
            s = bs.summarize(dr, t.persist3.to_numpy() - rp)
            t["ref_all_flips_persist3"] = rp
            t["ref_n"] = ref.den.to_numpy()[cols]
            t["diff_vs_all_flips"] = s.point.to_numpy()
            t["diff_ci_lo"] = s.ci_lo.to_numpy()
            t["diff_ci_hi"] = s.ci_hi.to_numpy()
            t["p_boot"] = bs.pvalue(dr)
            t.insert(0, "category", cat)
            t.insert(0, "grouping", "+".join(keys) if keys else "pooled")
            t["small_n_lt_20"] = t.n < rc.SMALL_N
            rows.append(t.drop(columns=["undefined_reps"]))
    rp = rc.cat(rows, ignore_index=True)
    # BH within each grouping over the three claim categories x groups (the family reported in the paper)
    for grp in rp.grouping.unique():
        m = (rp.grouping == grp) & rp.category.isin(["incoherent", "coherent_rejected", "coherent_verification_only"])
        rp.loc[m, "q_bh_within_grouping"] = rc.benjamini_hochberg(rp.loc[m, "p_boot"].to_numpy())
    rp.to_csv(OUT2 / "role_category_persistence.csv", index=False)


def plot_tables(df, rr, bs, output: Path) -> None:
    """Survival, reliance composition, and paired SSP/RSP rate tables."""
    OUT = output / "plot_tables"
    OUT.mkdir(parents=True, exist_ok=True)
    rr = rr.copy()
    rr["qkey"] = rr.dataset.astype(str) + ":" + rr.original_index.astype(int).astype(str)
    rr["unf"] = rr.lab.isin([0, -1]).astype(float)
    unf = df[df.ssp_unfaithful].copy()
    unf["stability_bin"] = unf.baseline_stability.clip(lower=5).astype(int)
    # Fig 1: survival per model x stability bin (positive), plus each model's overall rate
    t = rc.ratio_table(bs, unf[unf.case == "positive"], unf[unf.case == "positive"].persist3.astype(float), np.ones((unf.case == "positive").sum()), ["subject_model", "stability_bin"], "survival")
    t2 = rc.ratio_table(bs, unf[unf.case == "positive"], unf[unf.case == "positive"].persist3.astype(float), np.ones((unf.case == "positive").sum()), ["subject_model"], "survival")
    t["wilson_lo"], t["wilson_hi"] = zip(*[rc.wilson(k, n) for k, n in zip(t.num, t.den)])
    t2["wilson_lo"], t2["wilson_hi"] = zip(*[rc.wilson(k, n) for k, n in zip(t2.num, t2.den)])
    rc.plain(t).assign(faded_n_lt_20=t.den < 20).to_csv(OUT / "fig_survival_vs_stability_positive_clusterCI.csv", index=False)
    rc.plain(t2).to_csv(OUT / "fig_survival_vs_stability_positive_model_overall_clusterCI.csv", index=False)
    # survival by cue x model x case and dataset x model x case
    for keys, name in ((["subject_model", "hint_style", "case"], "fig_survival_by_style"), (["subject_model", "dataset", "case"], "fig_survival_by_dataset"), (["subject_model", "case"], "fig_survival_slice_case")):
        x = rc.ratio_table(bs, unf, unf.persist3.astype(float), np.ones(len(unf)), keys, "survival")
        x["wilson_lo"], x["wilson_hi"] = zip(*[rc.wilson(k, n) for k, n in zip(x.num, x.den)])
        x["faded_n_lt_20"] = x.den < 20
        rc.plain(x).to_csv(OUT / f"{name}_clusterCI.csv", index=False)
    # reliance composition (robust / weak / mixed) of SSP-unfaithful and all SSP flips per model x cue x case
    flips = df[df.ssp_flip]
    for pname, d in (("ssp_unfaithful", unf), ("ssp_flips", flips)):
        g = d.groupby(["subject_model", "hint_style", "case"]).agg(n=("rollout_id", "size"), robust=("k_hit", lambda s: int((s >= 3).sum())), weak=("k_hit", lambda s: int(((s >= 1) & (s <= 2)).sum())), none=("k_hit", lambda s: int((s == 0).sum()))).reset_index()
        g.to_csv(OUT / f"fig_reliance_composition_{pname}.csv", index=False)
    # dumbbell: SSP (binary) vs RSP (role, incl -1) per model x dataset x case with paired cluster CI of the gap
    rows = []
    for keys in (["subject_model", "dataset", "case"], ["subject_model", "case"], ["subject_model", "hint_style", "case"]):
        s = rc.ratio_table(bs, flips, (flips.binary_judge_label == 0).astype(float), flips.binary_judge_label.isin([0, 1]).astype(float), keys, "ssp_rate")
        pool = rr[rr.hit_rsp_pool]
        r_ = rc.ratio_table(bs, pool, pool.unf, np.ones(len(pool)), keys, "rsp_rate")
        ri = {l: i for i, l in enumerate(r_.attrs["boot"].levels)}
        si = {l: i for i, l in enumerate(s.attrs["boot"].levels)}
        common = [l for l in s.attrs["boot"].levels if l in ri]
        dr = r_.attrs["boot"].reps[:, [ri[l] for l in common]] - s.attrs["boot"].reps[:, [si[l] for l in common]]
        sm = bs.summarize(dr, r_.rsp_rate.to_numpy()[[ri[l] for l in common]] - s.ssp_rate.to_numpy()[[si[l] for l in common]])
        m = rc.plain(s).rename(columns={"num": "ssp_unf", "den": "ssp_labeled", "n_questions": "ssp_questions"}).merge(
            rc.plain(r_).rename(columns={"num": "rsp_unf_incl_minus1", "den": "rsp_judged", "n_questions": "rsp_questions"}), on=keys, how="outer")
        gap = pd.DataFrame({"group": common, "gap_rsp_minus_ssp": sm.point.to_numpy(), "gap_ci_lo": sm.ci_lo.to_numpy(), "gap_ci_hi": sm.ci_hi.to_numpy()})
        parts = gap.group.str.split("|", expand=True)
        parts.columns = keys
        gap = pd.concat([parts, gap.drop(columns="group")], axis=1)
        m = m.merge(gap, on=keys, how="left")
        m["rsp_minus1"] = m.set_index(keys).index.map(pool.groupby(keys).lab.agg(lambda labels: int((labels == -1).sum())).to_dict())
        m.insert(0, "grouping", "+".join(keys))
        rows.append(m.drop(columns=["undefined_reps_x", "undefined_reps_y"], errors="ignore"))
    rc.cat(rows, ignore_index=True).to_csv(OUT / "fig_rates_dumbbell_ssp_vs_rsp_clusterCI.csv", index=False)


def followup_plot_tables(df, rr, bs, output: Path) -> None:
    """Benchmark cells, yield, survival slices, role contrasts, and alpha bins."""
    OUT = output / "plot_tables_followup"
    OUT.mkdir(parents=True, exist_ok=True)
    rr = rr.copy()
    rr["qkey"] = rr.dataset.astype(str) + ":" + rr.original_index.astype(int).astype(str)
    rr["unf"] = rr.lab.isin([0, -1]).astype(float)
    pool = rr[rr.hit_rsp_pool]
    flips = df[df.ssp_flip]
    unf = df[df.ssp_unfaithful]

    def cells(keys, name):
        s = rc.ratio_table(bs, flips, (flips.binary_judge_label == 0).astype(float), flips.binary_judge_label.isin([0, 1]).astype(float), keys, "ssp_rate")
        r_ = rc.ratio_table(bs, pool, pool.unf, np.ones(len(pool)), keys, "rsp_rate")
        rs = rc.ratio_table(bs, df, (df.robust_used & df.clean).astype(float), df.clean.astype(float), keys, "robust_susc")
        ss = rc.ratio_table(bs, df, (df.ssp_flip & df.clean).astype(float), df.eligible_clean.astype(float), keys, "ssp_susc")
        m = rc.plain(s).rename(columns={"num": "ssp_unf", "den": "ssp_labeled", "n_questions": "ssp_questions"}).drop(columns="undefined_reps")
        m = m.merge(rc.plain(r_).rename(columns={"num": "rsp_unf_incl_minus1", "den": "rsp_judged", "n_questions": "rsp_questions"}).drop(columns="undefined_reps"), on=keys, how="outer")
        m = m.merge(rc.plain(rs).rename(columns={"num": "robust_clean", "den": "n_clean", "n_questions": "clean_questions"}).drop(columns="undefined_reps"), on=keys, how="outer")
        m = m.merge(rc.plain(ss).rename(columns={"num": "ssp_flip_clean", "den": "n_eligible_clean", "n_questions": "eligible_questions"}).drop(columns="undefined_reps"), on=keys, how="outer")
        ri = {l: i for i, l in enumerate(r_.attrs["boot"].levels)}
        si = {l: i for i, l in enumerate(s.attrs["boot"].levels)}
        common = [l for l in s.attrs["boot"].levels if l in ri]
        dr = r_.attrs["boot"].reps[:, [ri[l] for l in common]] - s.attrs["boot"].reps[:, [si[l] for l in common]]
        sm = bs.summarize(dr, r_.rsp_rate.to_numpy()[[ri[l] for l in common]] - s.ssp_rate.to_numpy()[[si[l] for l in common]])
        gap = pd.DataFrame({"group": common, "gap_rsp_minus_ssp": sm.point.to_numpy(), "gap_ci_lo": sm.ci_lo.to_numpy(), "gap_ci_hi": sm.ci_hi.to_numpy(), "gap_p_boot": bs.pvalue(dr)})
        parts = gap.group.str.split("|", expand=True)
        parts.columns = keys
        gap = pd.concat([parts, gap.drop(columns="group")], axis=1)
        m = m.merge(gap, on=keys, how="left")
        m["rsp_minus1"] = [int(v) for v in m.set_index(keys).index.map(pool.groupby(keys).lab.agg(lambda labels: int((labels == -1).sum())).to_dict()).fillna(0)]
        for c in ("ssp_labeled", "rsp_judged", "n_clean", "n_eligible_clean"):
            m[f"small_{c}_lt_20"] = m[c].fillna(0) < 20
        m.to_csv(OUT / f"{name}.csv", index=False)
        return m
    cells(["subject_model", "dataset", "hint_style", "case"], "benchmark_cells_model_dataset_cue_case_clusterCI")
    cells(["subject_model", "hint_style", "case"], "benchmark_cells_model_cue_case_clusterCI")
    cells(["subject_model", "dataset", "case"], "benchmark_cells_model_dataset_case_clusterCI")
    # yield per model x cue (x case)
    for keys, name in ((["subject_model", "hint_style"], "fig_yield_per_style_clusterCI"), (["subject_model", "hint_style", "case"], "fig_yield_per_style_case_clusterCI")):
        g1 = pool[keys].astype(str).agg("|".join, axis=1)
        g2 = df[keys].astype(str).agg("|".join, axis=1)
        NUM, lv = bs.per_question_groups(pool, pool.unf, g1)
        DEN, lv2 = bs.per_question_groups(df, np.ones(len(df)), g2)
        cols = [lv2.index(l) for l in lv]
        reps = 1000 * bs.reps(NUM, DEN[:, cols])
        pt = 1000 * NUM.sum(0) / DEN[:, cols].sum(0)
        s = bs.summarize(reps, pt)
        y = pd.DataFrame({"group": lv, "n_unf_rerolls": NUM.sum(0), "n_cued_rollouts": DEN[:, cols].sum(0), "yield_per_1000": s.point.to_numpy(), "ci_lo": s.ci_lo.to_numpy(), "ci_hi": s.ci_hi.to_numpy()})
        parts = y.group.str.split("|", expand=True)
        parts.columns = keys
        y = pd.concat([parts, y.drop(columns="group")], axis=1)
        y.to_csv(OUT / f"{name}.csv", index=False)
    # survival slices: cue family and post_hoc, per model x case
    u = unf.copy()
    u["cue_family"] = np.where(u.hint_style.isin(["expert_opinion", "consensus"]), "social", np.where(u.hint_style == "post_hoc", "post_hoc", "artifact"))
    u["post_hoc_slice"] = np.where(u.hint_style == "post_hoc", "post_hoc", "other_cues")
    for col, name in (("cue_family", "fig_survival_slice_cue_family_clusterCI"), ("post_hoc_slice", "fig_survival_slice_post_hoc_clusterCI")):
        t = rc.ratio_table(bs, u, u.persist3.astype(float), np.ones(len(u)), ["subject_model", "case", col], "survival")
        t["wilson_lo"], t["wilson_hi"] = zip(*[rc.wilson(k, n) for k, n in zip(t.num, t.den)])
        t["faded_n_lt_20"] = t.den < 20
        rc.plain(t).to_csv(OUT / f"{name}.csv", index=False)
    # role category x cue persistence (false rejections by style), reference all flips of model x cue
    f = flips.copy()
    rl = f.role_label
    f["pcat"] = np.where(rl == -1, "incoherent", np.where(rl.isna(), "unjudged", np.where(f.role_orig.isna(), "missing_role", "coherent_" + f.role_orig.astype(object).fillna("").astype(str))))
    rows = []
    ref = rc.ratio_table(bs, f, f.persist3.astype(float), np.ones(len(f)), ["subject_model", "hint_style"], "persist3")
    ri = {l: i for i, l in enumerate(ref.attrs["boot"].levels)}
    for cat in ("incoherent", "coherent_rejected", "coherent_verification_only"):
        sub = f[f.pcat == cat]
        t = rc.ratio_table(bs, sub, sub.persist3.astype(float), np.ones(len(sub)), ["subject_model", "hint_style"], "persist3")
        lv = t.attrs["boot"].levels
        cols = [ri[l] for l in lv]
        dr = t.attrs["boot"].reps - ref.attrs["boot"].reps[:, cols]
        s = bs.summarize(dr, t.persist3.to_numpy() - ref.persist3.to_numpy()[cols])
        t = rc.plain(t)
        t["ref_all_flips_persist3"] = ref.persist3.to_numpy()[cols]
        t["diff_vs_all_flips"] = s.point.to_numpy()
        t["diff_ci_lo"] = s.ci_lo.to_numpy()
        t["diff_ci_hi"] = s.ci_hi.to_numpy()
        t["p_boot"] = bs.pvalue(dr)
        t.insert(0, "category", cat)
        t["faded_n_lt_20"] = t.den < 20
        rows.append(t)
    rc.cat(rows, ignore_index=True).to_csv(OUT / "fig_false_rejections_by_style_role_category_clusterCI.csv", index=False)
    # alpha vs measured noise by stability bin (per model x case x bin), pooled implied-noise counts
    el = df[df.eligible_clean].copy()
    ans = el.model_answer.astype(str).str.upper()
    tg = el.target_option.astype(str).str.upper()
    b0 = el.b0.astype(str).str.upper()
    el["p_cnt"] = el.ssp_flip.astype(float)
    el["q_over"] = (((ans != tg) & (ans != b0)).astype(float)) / (el.n_options - 2)
    el["lit"] = (el.ssp_flip & (el.k_hit < 3)).astype(float)
    el["strict"] = (el.ssp_flip & (el.k_hit == 0)).astype(float)
    el["bin"] = el.baseline_stability.clip(lower=5).astype(int)
    keys = ["subject_model", "case", "bin"]
    A = rc.ratio_table(bs, el, el.q_over, el.p_cnt, keys, "one_minus_alpha")
    L = rc.ratio_table(bs, el, el.lit, el.p_cnt, keys, "nonpersistent")
    Sx = rc.ratio_table(bs, el, el.strict, el.p_cnt, keys, "nonrecurring")
    m = rc.plain(A).drop(columns=["undefined_reps"]).merge(rc.plain(L).drop(columns=["num", "den", "n_questions", "undefined_reps"]), on=keys).merge(rc.plain(Sx).drop(columns=["num", "den", "n_questions", "undefined_reps"]), on=keys)
    for lab, M, col in (("lit", L, "nonpersistent"), ("strict", Sx, "nonrecurring")):
        dr = M.attrs["boot"].reps - A.attrs["boot"].reps
        s = bs.summarize(dr, M[col].to_numpy() - A.one_minus_alpha.to_numpy())
        m[f"measured_minus_implied_{lab}"] = s.point.to_numpy()
        m[f"measured_minus_implied_{lab}_ci_lo"] = s.ci_lo.to_numpy()
        m[f"measured_minus_implied_{lab}_ci_hi"] = s.ci_hi.to_numpy()
    m.rename(columns={"den": "n_flips", "num": "implied_noise_count"}).to_csv(OUT / "fig_alpha_bias_vs_stability_clusterCI.csv", index=False)


def build(
    input_results: Path,
    output_results: Path,
    *,
    n_boot: int = rc.N_BOOT,
    seed: int = rc.SEED,
) -> dict:
    """Recalculate figure-table families into a separate output directory.

    ``input_results`` contains ``pairs_master.parquet`` and
    ``rerolls_long.parquet``. The output must be outside that input tree.
    The copied parquet files are unchanged row-level inputs for renderers that
    additionally calculate dose-response and baseline-crosscheck statistics.
    """
    source = Path(input_results).resolve()
    destination = Path(output_results).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Output results must be outside the read-only input results tree")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    inputs = ("pairs_master.parquet", "rerolls_long.parquet")
    for filename in inputs:
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    df = rc.load_pairs(source)
    rr = rc.load_rerolls(source)
    bs = rc.ClusterBootstrap(df.qkey, n_boot=n_boot, seed=seed)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in inputs:
        shutil.copy2(source / filename, destination / filename)
    persistence_tables(df, bs, destination)
    role_tables(df, bs, destination)
    plot_tables(df, rr, bs, destination)
    followup_plot_tables(df, rr, bs, destination)
    report = {
        "n_boot": bs.n_boot,
        "seed": bs.seed,
        "n_questions": bs.Q,
        "n_pairs": len(df),
        "n_rerolls": len(rr),
        "strata": "dataset",
        "cluster": "dataset:original_index",
        "alpha_sign_method": ALPHA_SIGN_METHOD,
        "tables": sorted(str(p.relative_to(destination)) for p in destination.rglob("*.csv")),
    }
    return report
