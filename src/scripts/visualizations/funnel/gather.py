"""Yield funnel and token cost — data. Per model, question × style pairs through the SSP → RSP chain, plus tokens.

Chain (unit = one (question, hint style) pair = one hinted-once rollout):
    hinted pairs → SSP flips (sample 0) → robust_used → original rollout's binary verdict 0 → clean
Cost in tokens (never GPU-hours): generation tokens per stage (8 baseline samples, of which the SSP needs only
sample 0; one hinted rollout per pair; 4 re-rolls per selected pair) and judge tokens from the judge caches.

Three stages, each writing into <out-dir>/data (default <cueball>/plots/funnel/data/):
    tokens   heavy: tokenizes every stored generation with the subject model's own tokenizer (offline HF cache)
             → tokens/<kind>__<file stem>.parquet, one row per CSV row (resumable: existing files are kept
             unless --force). A few worker processes at most.
    judge    reads the judge caches (per-record prompt_tokens / completion_tokens / cost_usd)
             → judge_keys.parquet + judge_sources.csv
    funnel   SSP rows from ``common.ssp_common`` → pairs.parquet, funnel_counts.csv, funnel_dropouts.csv,
             join_checks.csv; and, when tokens/ and judge_keys.parquet exist, cost_by_model.csv /
             cost_summary.csv / cost_checks.json.
    all      tokens, judge, funnel (the default).

Run:
    python -m src.scripts.visualizations.funnel.gather [all|tokens|judge|funnel] [--cueball-dir D] [--out-dir D]
        [--only SUBSTR] [--no-cache | --refresh-cache] [--cache-dir D] [--workers W] [--limit-rows N] [--force]
Debug: --only <stem substring> (never cached; written under <plot dir>/_debug unless --out-dir is given),
--limit-rows N (tokens: first N rows per file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.visualizations.common import paths as P
from src.scripts.visualizations.common import ssp_common as S
from src.scripts.visualizations.common import yield_common as yc

PLOT = "funnel"
CASE_GROUPS = ("all", "positive", "negative")
STAGES = ["hinted_pairs", "ssp_flip", "robust_used", "binary_verdict_0", "clean"]


def log(msg: str) -> None:
    print(msg, flush=True)


def noise_flip_min_votes(cueball: Path) -> int:
    """The manifest builder's noise-flip threshold, from the manifest meta sidecar."""
    return json.loads(P.manifest_meta_path(cueball).read_text())["noise_flip_min_votes"]


# ---------------------------------------------------------------------------
# Sources: the binary-judged runs of the 5 models (smoke runs skipped)
# ---------------------------------------------------------------------------

def run_sources(only: str | None, cueball: Path) -> list[dict]:
    srcs = S.sources(only, cueball=cueball)
    out = []
    for s in srcs:
        if not any(s["stem"].startswith(m + "_") for m in yc.MODELS):
            continue
        recipe_path = P.hinted_dir(cueball) / f"{s['stem']}.meta.json"
        recipe = json.loads(recipe_path.read_text())
        bl_meta = json.loads(Path(s["baseline_meta"]).read_text())
        s = dict(s)
        s["model_name"] = bl_meta["model_name"]
        if recipe.get("model_name") and recipe["model_name"] != s["model_name"]:
            raise SystemExit(f"{s['stem']}: recipe model {recipe['model_name']} != baseline model {s['model_name']}")
        s["baseline_thinking"] = bl_meta.get("thinking", True)
        s["recipe_thinking"] = recipe.get("thinking", True)
        s["hinted_csv"] = P.hinted_dir(cueball) / f"{s['stem']}.csv"
        s["reroll_csvs"] = {seed: P.rollouts_dir(cueball) / f"{s['stem']}_rs{seed}.csv" for seed in yc.SEEDS}
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Stage 1: token counts (worker side)
# ---------------------------------------------------------------------------

_TOK: dict = {}
_TC: dict = {}
TOKENIZE_BATCH = 64


def _tokenizer(model_name: str):
    if model_name not in _TOK:
        from transformers import AutoTokenizer
        _TOK[model_name] = AutoTokenizer.from_pretrained(model_name)   # offline: the HF cache
    return _TOK[model_name]


def _thinking(model_name: str, thinking):
    key = (model_name, str(thinking))
    if key not in _TC:
        from src.lib.model_utils import get_thinking_config
        _TC[key] = get_thinking_config(model_name, thinking)
    return _TC[key]


def _count(tok, texts: list[str]) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), TOKENIZE_BATCH):
        batch = texts[i:i + TOKENIZE_BATCH]
        enc = tok(batch, add_special_tokens=False, return_attention_mask=False, return_length=True)
        out.extend(int(n) if t else 0 for n, t in zip(enc["length"], batch))
    return out


def _render_counts(tok, tc, messages_list: list[list[dict]]) -> tuple[list, int]:
    from src.lib.model_utils import template_kwargs
    texts, ok, errors = [], [], 0
    for msgs in messages_list:
        try:
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 **template_kwargs(tc.enable_thinking)))
            ok.append(True)
        except Exception:  # noqa: BLE001 — a template failure only loses that row's input-token estimate
            texts.append("")
            ok.append(False)
            errors += 1
    lens = _count(tok, texts)
    return [n if good else None for n, good in zip(lens, ok)], errors


def token_task(task: dict) -> dict:
    """Tokenize one CSV; write one row per CSV row. Returns timing / volume stats."""
    from src.lib.activation_store import build_messages
    from src.lib.model_utils import build_chat_messages

    t0 = time.time()
    tok = _tokenizer(task["model_name"])
    tc = _thinking(task["model_name"], task["thinking"])
    t_load = time.time() - t0
    rows, n_chars, n_prompt_chars, render_errors = [], 0, 0, 0
    kind = task["kind"]
    if kind == "baseline":
        usecols, chunksize = ["original_index", "prompt", "sample_rollouts"], 200
    else:
        usecols, chunksize = ["original_index", "sample_type", "hint_name", "hinted_prompt", "rollout"], 1000
    reader = pd.read_csv(task["path"], usecols=usecols, chunksize=chunksize, nrows=task.get("limit_rows"),
                         keep_default_na=False, dtype=str)
    for chunk in reader:
        if kind == "baseline":
            samples = [json.loads(s) if s else [] for s in chunk["sample_rollouts"]]
            flat = [t or "" for ss in samples for t in ss]
            lens = _count(tok, flat)
            msgs = [build_chat_messages(p, tc.system_prompt) for p in chunk["prompt"]]
            plens, err = _render_counts(tok, tc, msgs)
            render_errors += err
            pos = 0
            for oi, ss, pl, p in zip(chunk["original_index"], samples, plens, chunk["prompt"]):
                ln = lens[pos:pos + len(ss)]
                pos += len(ss)
                n_chars += sum(len(t or "") for t in ss)
                n_prompt_chars += len(p)
                rows.append({"original_index": int(oi), "n_samples": len(ss),
                             "sample0_out_tokens": ln[0] if ln else 0, "all_out_tokens": int(sum(ln)),
                             "sample0_chars": len(ss[0] or "") if ss else 0,
                             "all_chars": sum(len(t or "") for t in ss),
                             "n_blank_samples": sum(1 for t in ss if not t),
                             "sample_out_tokens": json.dumps(ln), "prompt_tokens": pl})
        else:
            texts = chunk["rollout"].tolist()
            lens = _count(tok, texts)
            msgs = [build_messages(hp, tc.system_prompt) for hp in chunk["hinted_prompt"]]
            plens, err = _render_counts(tok, tc, msgs)
            render_errors += err
            n_chars += sum(map(len, texts))
            n_prompt_chars += sum(map(len, chunk["hinted_prompt"]))
            for oi, st, hn, t, ln, pl in zip(chunk["original_index"], chunk["sample_type"], chunk["hint_name"],
                                             texts, lens, plens):
                rows.append({"original_index": int(oi), "sample_type": st, "hint_name": hn,
                             "out_tokens": ln, "out_chars": len(t), "blank": not t, "prompt_tokens": pl})
    df = pd.DataFrame(rows)
    if "prompt_tokens" in df:
        df["prompt_tokens"] = df["prompt_tokens"].astype("Int64")
    out = Path(task["out"])
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    stats = {k: task[k] for k in ("kind", "stem", "seed", "model_name", "path", "limit_rows")}
    stats.update({
        "file_bytes": Path(task["path"]).stat().st_size, "rows": len(df), "gen_chars": n_chars,
        "prompt_chars": n_prompt_chars, "render_errors": render_errors,
        "out_tokens": int(df["all_out_tokens" if kind == "baseline" else "out_tokens"].sum()) if len(df) else 0,
        "seconds": round(time.time() - t0, 2), "tokenizer_load_s": round(t_load, 2), "pid": os.getpid(),
    })
    return stats


def token_tasks(srcs: list[dict], out: Path, limit_rows: int | None) -> list[dict]:
    tdir = out / "tokens"
    tdir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for s in srcs:
        tasks.append({"kind": "baseline", "stem": s["stem"], "seed": None, "model_name": s["model_name"],
                      "thinking": s["baseline_thinking"], "path": str(s["baseline_csv"])})
        tasks.append({"kind": "hinted", "stem": s["stem"], "seed": None, "model_name": s["model_name"],
                      "thinking": s["recipe_thinking"], "path": str(s["hinted_csv"])})
        for seed, p in s["reroll_csvs"].items():
            if not p.exists():
                raise SystemExit(f"missing re-roll CSV {p}")
            rmeta = json.loads(p.with_suffix(".meta.json").read_text())
            if rmeta.get("model_name") != s["model_name"]:
                raise SystemExit(f"{p}: model {rmeta.get('model_name')} != {s['model_name']}")
            tasks.append({"kind": "reroll", "stem": s["stem"], "seed": seed, "model_name": s["model_name"],
                          "thinking": rmeta.get("thinking", True), "path": str(p)})
    for t in tasks:
        t["limit_rows"] = limit_rows
        t["out"] = str(tdir / f"{t['kind']}__{Path(t['path']).stem}.parquet")
    return tasks


def run_tokens(srcs: list[dict], out: Path, limit_rows: int | None, workers: int, force: bool) -> None:
    import multiprocessing as mp
    tasks = token_tasks(srcs, out, limit_rows)
    todo = [t for t in tasks if force or not Path(t["out"]).exists()]
    # largest first, grouped by model so a worker mostly reuses one tokenizer
    todo.sort(key=lambda t: -Path(t["path"]).stat().st_size)
    log(f"tokens: {len(tasks)} files, {len(todo)} to do, {workers} worker(s), "
        f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}")
    stats_path = out / "tokens" / "token_stats.jsonl"
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool, open(stats_path, "a") as fh:
        for st in pool.imap_unordered(token_task, todo):
            fh.write(json.dumps(st) + "\n")
            fh.flush()
            rate = st["gen_chars"] / max(st["seconds"], 1e-9) / 1e6
            log(f"  [{time.time() - t0:7.0f}s] {st['kind']:8s} {Path(st['path']).name}: {st['rows']:,} rows, "
                f"{st['out_tokens']:,} out tokens, {st['seconds']:.0f}s ({rate:.2f} Mchar/s), "
                f"render_errors={st['render_errors']}")
    log(f"tokens done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Stage 2: judge-cache token usage
# ---------------------------------------------------------------------------

def read_cache(path: Path) -> pd.DataFrame:
    recs = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            r = json.loads(line)
            err = r.get("error")
            recs.append({"seq": i, "original_index": int(r["original_index"]), "hint_name": r["hint_name"],
                         "prompt_tokens": pd.to_numeric(r.get("prompt_tokens"), errors="coerce"),
                         "completion_tokens": pd.to_numeric(r.get("completion_tokens"), errors="coerce"),
                         "cost_usd": pd.to_numeric(r.get("cost_usd"), errors="coerce"),
                         "is_error": err not in (None, "", "None"),
                         "label": pd.to_numeric(r.get("label"), errors="coerce")})
    return pd.DataFrame(recs)


def judge_sources(srcs: list[dict], cueball: Path) -> list[dict]:
    """(source_kind, stem, seed, cache file, judge model, prompt hash) for every judge pass the chain uses.

    The binary judge's cache dir is the one its sidecar names; the v2 caches sit under ``${DATA_ROOT}/judge_cache/
    <stem>`` (the judged sidecar's ``cache_dir`` where a ``_judged.meta.json`` exists, the repo convention otherwise).
    """
    out = []
    cache_root = P.judge_cache_root()
    for s in srcs:
        stem = s["stem"]
        bmeta = json.loads((P.binary_dir(cueball) / f"{stem}_binary_judged.meta.json").read_text())
        out.append({"source_kind": "binary_orig", "stem": stem, "seed": None,
                    "cache_dir": Path(bmeta["cache_dir"]), "judge_model": bmeta["judge_model"],
                    "prompt_hash": bmeta["prompt"]["cache_hash"], "sidecar": f"{stem}_binary_judged.meta.json"})
        summ = json.loads((P.hinted_dir(cueball) / f"{stem}_judged_summary.json").read_text())
        meta_p = P.hinted_dir(cueball) / f"{stem}_judged.meta.json"
        cache_dir = Path(json.loads(meta_p.read_text())["cache_dir"]) if meta_p.exists() else cache_root / stem
        out.append({"source_kind": "v2_orig", "stem": stem, "seed": None, "cache_dir": cache_dir,
                    "judge_model": summ["judge_model"], "prompt_hash": summ["judge_prompt_hash"],
                    "sidecar": meta_p.name if meta_p.exists() else f"{stem}_judged_summary.json (cache dir by convention)"})
        for seed in yc.SEEDS:
            rs = f"{stem}_rs{seed}"
            summ = json.loads((P.rollouts_dir(cueball) / f"{rs}_judged_summary.json").read_text())
            out.append({"source_kind": "v2_reroll", "stem": stem, "seed": seed,
                        "cache_dir": cache_root / rs, "judge_model": summ["judge_model"],
                        "prompt_hash": summ["judge_prompt_hash"],
                        "sidecar": f"{rs}_judged_summary.json (cache dir by convention)"})
    for j in out:
        j["cache_file"] = j["cache_dir"] / f"{j['judge_model'].replace('/', '_')}_binary_{j['prompt_hash']}.jsonl"
    return out


def run_judge(srcs: list[dict], out: Path, cueball: Path) -> None:
    man = yc.load_rollout_manifest(["source_csv", "original_index", "hint_style", "judge_label"], cueball=cueball)
    res = yc.load_resample_manifest(["source_csv", "original_index", "hint_style", "judge_label", "provenance"],
                                    cueball=cueball)
    res = res[res["provenance"].astype(str) == "resample_k4"]
    keys_out, rows = [], []
    for j in judge_sources(srcs, cueball):
        stem, seed, kind = j["stem"], j["seed"], j["source_kind"]
        if kind == "binary_orig":
            uni = man[man.source_csv == f"{stem}_judged.csv"]
            b = pd.read_csv(P.binary_dir(cueball) / f"{stem}_binary_judged.csv",
                            usecols=["original_index", "hint_name", "judge_label"])
            scope = b[b.judge_label.notna()][["original_index", "hint_name", "judge_label"]]
        elif kind == "v2_orig":
            uni = man[man.source_csv == f"{stem}_judged.csv"]
            scope = uni[uni.judge_label.notna()].rename(columns={"hint_style": "hint_name"})
        else:
            uni = res[res.source_csv == f"{stem}_rs{seed}_judged.csv"]
            scope = uni[uni.judge_label.notna()].rename(columns={"hint_style": "hint_name"})
        universe = uni.rename(columns={"hint_style": "hint_name"})[["original_index", "hint_name"]].copy()
        universe["original_index"] = universe["original_index"].astype(int)
        scope = scope[["original_index", "hint_name", "judge_label"]].astype({"original_index": int})
        scope["used_label"] = pd.to_numeric(scope.pop("judge_label"), errors="coerce")
        if scope.duplicated(["original_index", "hint_name"]).any():
            raise SystemExit(f"{kind} {stem} {seed}: duplicate labelled keys")
        uset = set(map(tuple, universe.values.tolist()))
        sset = set(map(tuple, scope[["original_index", "hint_name"]].values.tolist()))
        if not sset <= uset:
            raise SystemExit(f"{kind} {stem} {seed}: judged scope has keys outside the paper tree's source rows")
        other_files = sorted(p.name for p in j["cache_dir"].glob("*.jsonl") if p != j["cache_file"])
        rec = read_cache(j["cache_file"]) if j["cache_file"].exists() else pd.DataFrame(
            columns=["seq", "original_index", "hint_name", "prompt_tokens", "completion_tokens", "cost_usd",
                     "is_error", "label"])
        rec["key"] = list(zip(rec.original_index, rec.hint_name))
        in_uni = rec.key.isin(uset)
        r_u = rec[in_uni]
        # all calls on the paper tree's rows (retries, errors, superseded re-judges included)
        all_agg = r_u.groupby(["original_index", "hint_name"]).agg(
            n_records=("seq", "size"), n_error_records=("is_error", "sum"),
            all_prompt_tokens=("prompt_tokens", "sum"), all_completion_tokens=("completion_tokens", "sum"),
            all_cost_usd=("cost_usd", "sum"))
        # one call per labelled row: the latest non-error record of each in-scope key WHOSE LABEL equals the label
        # the judged CSV / manifest carries (the verdict actually used); fallback = latest non-error record, counted
        cand = r_u[~r_u.is_error & r_u.key.isin(sset)].merge(scope, on=["original_index", "hint_name"], how="left")
        cand["label_match"] = cand.label == cand.used_label
        latest_any = cand.sort_values("seq").groupby(["original_index", "hint_name"]).tail(1)
        latest_match = cand[cand.label_match].sort_values("seq").groupby(["original_index", "hint_name"]).tail(1)
        n_latest_differs = int((~latest_any.label_match).sum())            # latest record's label != used label
        fallback = latest_any[~latest_any.set_index(["original_index", "hint_name"]).index.isin(
            latest_match.set_index(["original_index", "hint_name"]).index)]
        ok = pd.concat([latest_match, fallback])
        served = ok.set_index(["original_index", "hint_name"])[["prompt_tokens", "completion_tokens", "cost_usd",
                                                                "label_match"]] \
            .rename(columns=lambda c: f"served_{c}")
        k = scope.set_index(["original_index", "hint_name"]).join(served, how="left").join(all_agg, how="outer")
        n_mismatch_served = int((k.served_label_match == False).sum())  # noqa: E712 — no record with the used label
        k = k.reset_index()
        k["in_scope"] = list(map(lambda t: t in sset, zip(k.original_index, k.hint_name)))
        k.insert(0, "seed", seed)
        k.insert(0, "stem", stem)
        k.insert(0, "source_kind", kind)
        keys_out.append(k)
        n_scope = len(sset)
        n_served = int(k.served_prompt_tokens.notna().sum())
        rows.append({
            "source_kind": kind, "stem": stem, "seed": seed, "judge_model": j["judge_model"],
            "prompt_hash": j["prompt_hash"], "cache_file": str(j["cache_file"]), "sidecar": j["sidecar"],
            "cache_exists": j["cache_file"].exists(), "other_cache_files_ignored": ";".join(other_files),
            "universe_rows": len(uset), "labelled_rows": n_scope, "labelled_rows_with_record": n_served,
            "labelled_rows_missing_record": n_scope - n_served,
            "served_label_mismatch": n_latest_differs,
            "served_label_mismatch_note": "keys whose LATEST non-error record's label differs from the used label; "
                                          "served = the latest record with the used label instead",
            "served_no_record_with_used_label": n_mismatch_served,
            "served_median_completion_tokens": float(ok.completion_tokens.median()) if len(ok) else np.nan,
            "records_total": len(rec), "records_on_cueball_rows": int(in_uni.sum()),
            "records_outside_cueball_rows": int((~in_uni).sum()),
            "records_missing_prompt_tokens": int(rec.prompt_tokens.isna().sum()),
            "records_zero_prompt_tokens": int((rec.prompt_tokens == 0).sum()),
            "records_missing_completion_tokens": int(rec.completion_tokens.isna().sum()),
            "error_records_on_cueball_rows": int(r_u.is_error.sum()),
            "served_prompt_tokens": int(k.served_prompt_tokens.sum()),
            "served_completion_tokens": int(k.served_completion_tokens.sum()),
            "served_cost_usd": float(k.served_cost_usd.sum()),
            "all_prompt_tokens": int(k.all_prompt_tokens.sum()), "all_completion_tokens": int(k.all_completion_tokens.sum()),
            "all_cost_usd": float(k.all_cost_usd.sum()),
        })
        log(f"  judge {kind:10s} {stem} {seed or ''}: labelled {n_scope:,}, with record {n_served:,}, "
            f"records {len(rec):,} ({int((~in_uni).sum()):,} outside the paper tree's rows), others ignored: {other_files}")
    keys = pd.concat(keys_out, ignore_index=True)
    keys["seed"] = keys["seed"].astype("Int64")
    js = pd.DataFrame(rows)
    # provider usage that looks like it excludes reasoning tokens: a cache whose median served completion is < 1/4
    # of the median of the other caches of the same judge pass and subject model (flagged, never imputed)
    js["model"] = js.stem.map(lambda st: next(m for m in yc.MODELS if st.startswith(m + "_")))
    ref = js.groupby(["source_kind", "model"]).served_median_completion_tokens.transform("median")
    js["completion_tokens_flag"] = np.where(js.served_median_completion_tokens < 0.25 * ref,
                                            "LOW: median completion < 1/4 of this model+pass median; provider usage "
                                            "likely excludes reasoning tokens — understated, not imputed", "")
    rows = js.to_dict("records")
    keys.to_parquet(out / "judge_keys.parquet", index=False)
    pd.DataFrame(rows).to_csv(out / "judge_sources.csv", index=False)
    log(f"wrote {out / 'judge_keys.parquet'} and judge_sources.csv")


# ---------------------------------------------------------------------------
# Stage 3: the funnel
# ---------------------------------------------------------------------------

def stitch_pairs(only: str | None, cueball: Path, cache: str | Path | None,
                 refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """SSP rows from ``common.ssp_common`` (shared with the survival figures), plus the manifest's v2 label and
    no-hint target votes, and a check of question_reliance's label against the resample manifest's."""
    df, meta = S.load_rows(only, cache_dir=cache, refresh=refresh, cueball=cueball)
    df = df[df.subject_model.isin(yc.MODELS)].copy()
    extra = yc.load_rollout_manifest(["rollout_id", "judge_label_final", "baseline_hint_votes"], cueball=cueball)
    extra = extra[["rollout_id", "judge_label_final", "baseline_hint_votes"]]
    n0 = len(df)
    df = df.merge(extra, on="rollout_id", how="left", validate="one_to_one")
    assert len(df) == n0
    # question_reliance (via ssp_common) vs the resample manifest's own reliance_label on the original rows
    rm = yc.load_resample_manifest(["rollout_id", "provenance", "reliance_label"], cueball=cueball)
    rm = rm[rm.provenance.astype(str) == "hinted_once"].set_index("rollout_id")["reliance_label"]
    j = df.set_index("rollout_id")["reliance_label"].astype(object)
    both = j.index.intersection(rm.index)
    mism = int((j.loc[both].fillna("<NA>") != rm.loc[both].astype(object).fillna("<NA>")).sum())
    ch = pd.DataFrame(meta["join_checks"])
    ch["reliance_mismatch_vs_resample_manifest"] = mism     # pooled over runs (same number on every row)
    return df, ch, meta


def stage_flags(df: pd.DataFrame, min_votes: int) -> pd.DataFrame:
    out = df.copy()
    rl = out.reliance_label.astype(object)
    out["st_hinted_pairs"] = True
    out["st_ssp_flip"] = out.ssp_flip
    out["st_robust_used"] = out.st_ssp_flip & (rl == "robust_used")
    out["st_binary_verdict_0"] = out.st_robust_used & (out.binary_judge_label == 0)
    # v2 -1 (exclude_reason == incoherent) is NOT dropped; every other reason is. incoherent ranks before
    # noise_flip in EXCLUDE_REASONS, so a -1 original is re-tested for the noise-flip rule itself
    # (to_hint ∧ baseline_hint_votes ≥ min_votes, the manifest meta's noise_flip_min_votes).
    er = out.exclude_reason.astype(object)
    votes = pd.to_numeric(out.baseline_hint_votes, errors="coerce")
    hidden_noise = (er == "incoherent") & out.to_hint.fillna(False).astype(bool) & (votes >= min_votes)
    out["noise_flip_under_incoherent"] = hidden_noise
    out["st_clean"] = out.st_binary_verdict_0 & (er.isna() | ((er == "incoherent") & ~hidden_noise))
    out["v2_label"] = pd.to_numeric(out.judge_label_final, errors="coerce")
    return out


def _groups(df: pd.DataFrame):
    for model in yc.MODELS:
        d = df[df.subject_model == model]
        for cg in CASE_GROUPS:
            yield model, cg, (d if cg == "all" else d[d.case.astype(object) == cg])


def funnel_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts, drops = [], []
    for model, cg, d in _groups(df):
        row = {"subject_model": model, "case_group": cg}
        for st in STAGES:
            row[f"n_{st}"] = int(d[f"st_{st}"].sum())
        for s in S.SSP_STATUSES:
            row[f"ssp_status_{s}"] = int((d.ssp_status == s).sum())
        row["n_truncated_to_target_excluded"] = int(d.ssp_flip_truncated.sum())
        row["n_to_hint_manifest"] = int(d.to_hint.fillna(False).astype(bool).sum())
        row["n_clean_v2_label_-1"] = int((d.st_clean & (d.v2_label == -1)).sum())
        row["n_clean_v2_label_0"] = int((d.st_clean & (d.v2_label == 0)).sum())
        row["n_clean_v2_label_1"] = int((d.st_clean & (d.v2_label == 1)).sum())
        row["n_clean_v2_label_missing"] = int((d.st_clean & d.v2_label.isna()).sum())
        row["n_binary0_incoherent_but_noise_flip_dropped"] = int((d.st_binary_verdict_0 & d.noise_flip_under_incoherent).sum())
        counts.append(row)
        f = d[d.st_ssp_flip]
        r = d[d.st_robust_used]
        b = d[d.st_binary_verdict_0]
        dd = {
            ("ssp_flip→robust_used", "no_reliance_row (not re-rolled)"): int((f.reliance_label.isna() & f.k_n.isna()).sum()),
            ("ssp_flip→robust_used", "reliance_null (k_n<4)"): int((f.reliance_label.isna() & f.k_n.notna()).sum()),
            ("ssp_flip→robust_used", "weak_used (1-2/4)"): int((f.reliance_label.astype(object) == "weak_used").sum()),
            ("ssp_flip→robust_used", "mixed (0/4)"): int((f.reliance_label.astype(object) == "mixed").sum()),
            ("robust_used→binary_verdict_0", "binary verdict 1 (faithful)"): int((r.binary_judge_label == 1).sum()),
            ("robust_used→binary_verdict_0", "binary verdict missing (judge error / not judged)"): int(r.binary_judge_label.isna().sum()),
        }
        gone = b[~b.st_clean]
        why = gone.exclude_reason.astype(object).where(~gone.noise_flip_under_incoherent, "noise_flip (recorded as incoherent)")
        for reason, n in why.value_counts().items():
            dd[("binary_verdict_0→clean", f"exclude_reason={reason}")] = int(n)
        for (step, reason), n in dd.items():
            drops.append({"subject_model": model, "case_group": cg, "step": step, "reason": reason, "n": n})
    return pd.DataFrame(counts), pd.DataFrame(drops)


# ---------------------------------------------------------------------------
# Cost aggregation
# ---------------------------------------------------------------------------

# component → (kind, chain role, charged in the figure?)   The figure shows generation OUTPUT tokens and SERVED
# judge calls; only the funnel's own questions / SSP-flip pairs are charged ("charged" scope). The "as_run" scope
# (everything generated / judged for these runs) and reconstructed input tokens stay in the tables.
COMPONENTS = {
    "baseline_sample0": ("generation", "SSP baseline: sample 0 of 8, questions with hinted pairs", True),
    "baseline_samples_1to7": ("generation", "RSP-only: the other 7 no-hint samples, questions with hinted pairs", True),
    "hinted_rollouts": ("generation", "one hinted rollout per pair (both protocols)", True),
    "rerolls": ("generation", "k=4 re-rolls of SSP-flip pairs → robust_used", True),
    "judge_binary_orig": ("judge", "binary judge on the SSP-flip originals → SSP label (binary verdict 0)", True),
    "judge_v2_reroll": ("judge", "v2 judge on the to-target re-rolls of SSP-flip pairs → RSP label", True),
    "judge_v2_orig": ("judge", "v2 judge on the switched originals (charged scope: SSP-flip originals only) — "
                               "not used by the chain since round 3 (−1 no longer drops a pair); NOT in the "
                               "figure, table only", False),
}
SCOPES = ("charged", "as_run")


def _read_tokens(tdir: Path, kind: str) -> pd.DataFrame:
    parts = []
    for p in sorted(tdir.glob(f"{kind}__*.parquet")):
        d = pd.read_parquet(p)
        name = p.stem.split("__", 1)[1]
        if kind == "baseline":
            d["baseline_file"] = name
        else:
            d["stem"] = name.rsplit("_rs", 1)[0] if kind == "reroll" else name
            d["seed"] = int(name.rsplit("_rs", 1)[1]) if kind == "reroll" else None
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def costs(df: pd.DataFrame, srcs: list[dict], out: Path) -> None:
    tdir = out / "tokens"
    expected = token_tasks(srcs, out, None)
    missing = [t["out"] for t in expected if not Path(t["out"]).exists()]
    if missing or not (out / "judge_keys.parquet").exists():
        log(f"cost tables skipped: {len(missing)} token file(s) missing, "
            f"judge_keys.parquet {'present' if (out / 'judge_keys.parquet').exists() else 'missing'}")
        return
    stem_model = {s["stem"]: s["subject_model"] for s in srcs}
    bl_stem = {Path(s["baseline_csv"]).stem: s["stem"] for s in srcs}
    # question → case (one case per question per run; checked)
    qc = df.groupby(["stem", "original_index"]).case.agg(lambda c: sorted(set(c.astype(str))))
    if (qc.map(len) > 1).any():
        raise SystemExit("a question carries two cases within one run")
    qcase = qc.map(lambda c: c[0]).rename("case").reset_index()
    pk = df[["stem", "original_index", "hint_style", "case", "ssp_flip", "st_clean", "k_n", "to_hint"]].rename(
        columns={"hint_style": "hint_name"}).assign(case=lambda d: d.case.astype(str), _pair=True)
    checks = {}

    bl = _read_tokens(tdir, "baseline")
    bl["stem"] = bl.baseline_file.map(bl_stem)
    bl["subject_model"] = bl.stem.map(stem_model)
    bl = bl.merge(qcase, on=["stem", "original_index"], how="left", validate="one_to_one")
    bl["case"] = bl["case"].fillna("no_case_pool")
    checks["baseline_questions"] = len(bl)
    checks["baseline_questions_without_pairs"] = int((bl.case == "no_case_pool").sum())
    checks["baseline_rows_not_8_samples"] = int((bl.n_samples != 8).sum())
    checks["baseline_blank_samples"] = int(bl.n_blank_samples.sum())
    checks["baseline_prompt_render_failures"] = int(bl.prompt_tokens.isna().sum())

    def attach(d: pd.DataFrame, name: str) -> pd.DataFrame:
        d = d.drop(columns=[c for c in ("sample_type",) if c in d]).merge(
            pk, on=["stem", "original_index", "hint_name"], how="left", validate="many_to_one")
        checks[f"{name}_rows"] = len(d)
        checks[f"{name}_rows_not_in_manifest"] = int(d._pair.isna().sum())
        checks[f"{name}_prompt_render_failures"] = int(d.prompt_tokens.isna().sum())
        checks[f"{name}_blank_rollouts"] = int(d.blank.sum())
        d = d[d._pair.notna()].copy()
        d["subject_model"] = d.stem.map(stem_model)
        d["ssp_flip"] = d.ssp_flip.astype(bool)
        d["used_candidate"] = d.k_n.notna()
        return d

    hi = attach(_read_tokens(tdir, "hinted"), "hinted")
    checks["pairs_without_hinted_tokens"] = int(len(df) - len(hi))
    rr = attach(_read_tokens(tdir, "reroll"), "reroll")
    checks["reroll_rows_per_seed"] = {int(k): int(v) for k, v in rr.seed.value_counts().items()}

    jk = pd.read_parquet(out / "judge_keys.parquet")
    jk = jk.merge(pk[["stem", "original_index", "hint_name", "case"]], on=["stem", "original_index", "hint_name"],
                  how="left", validate="many_to_one")
    jk["subject_model"] = jk.stem.map(stem_model)

    jk["ssp_flip"] = jk.merge(pk[["stem", "original_index", "hint_name", "ssp_flip"]],
                              on=["stem", "original_index", "hint_name"], how="left")["ssp_flip"].fillna(False).astype(bool).values
    rows = []
    for scope in SCOPES:
        for model in yc.MODELS:
            for cg in CASE_GROUPS:
                def sel(d):
                    d = d[d.subject_model == model]
                    return d if cg == "all" else d[d.case == cg]
                b, h, r, j = sel(bl), sel(hi), sel(rr), sel(jk)
                if scope == "charged":
                    b = b[b.case != "no_case_pool"]            # baseline samples of questions in the funnel
                    r = r[r.ssp_flip]                           # re-rolls of SSP-flip pairs only
                    j = j[j.ssp_flip]                           # judge calls on SSP-flip pairs only (all passes)

                def add(comp, out_tok, in_tok, n_units, unit, note=""):
                    rows.append({"scope": scope, "subject_model": model, "case_group": cg, "component": comp,
                                 "kind": COMPONENTS[comp][0], "chain_role": COMPONENTS[comp][1],
                                 "in_figure": COMPONENTS[comp][2] and scope == "charged",
                                 "output_tokens": int(out_tok),
                                 "input_tokens_reconstructed": None if COMPONENTS[comp][0] == "judge" else int(in_tok),
                                 "judge_prompt_tokens_served": int(in_tok) if COMPONENTS[comp][0] == "judge" else None,
                                 "n_units": int(n_units), "unit": unit, "note": note})
                s0 = b.sample0_out_tokens.sum()
                add("baseline_sample0", s0, b.prompt_tokens.sum(), len(b), "baseline questions × 1 sample")
                add("baseline_samples_1to7", b.all_out_tokens.sum() - s0, (b.prompt_tokens * (b.n_samples - 1)).sum(),
                    int((b.n_samples - 1).sum()), "baseline samples 1-7",
                    "input counted per request (no prefix-cache credit)")
                add("hinted_rollouts", h.out_tokens.sum(), h.prompt_tokens.sum(), len(h), "hinted rollouts (= pairs)")
                add("rerolls", r.out_tokens.sum(), r.prompt_tokens.sum(), len(r), "re-roll rollouts",
                    "Nemotron re-rolls were generated at a 24,576-token budget (others 16,384); kept as generated")
                for comp, sk in (("judge_binary_orig", "binary_orig"), ("judge_v2_orig", "v2_orig"),
                                 ("judge_v2_reroll", "v2_reroll")):
                    q = j[j.source_kind == sk]
                    qs = q[q.in_scope]
                    add(comp, qs.served_completion_tokens.sum(), qs.served_prompt_tokens.sum(),
                        int(qs.served_prompt_tokens.notna().sum()), "served calls (one per label)",
                        f"all calls on these rows incl. retries/errors/superseded re-judges: prompt "
                        f"{int(q.all_prompt_tokens.sum()):,} + completion {int(q.all_completion_tokens.sum()):,} "
                        f"({int(q.n_records.sum()):,} records); labelled rows without a cache record: "
                        f"{int((qs.served_prompt_tokens.isna()).sum()):,}")
    cost = pd.DataFrame(rows)
    cost.to_csv(out / "cost_by_model.csv", index=False)
    counts = pd.read_csv(out / "funnel_counts.csv")
    ch = cost[(cost.scope == "charged") & (cost.case_group == "all")]
    served_total = ch.output_tokens + ch.judge_prompt_tokens_served.fillna(0)
    tot = ch.assign(t=served_total).pivot(index="subject_model", columns="component", values="t")
    n_clean = counts[counts.case_group == "all"].set_index("subject_model")["n_clean"]
    gen = ["baseline_sample0", "baseline_samples_1to7", "hinted_rollouts", "rerolls"]
    summ = pd.DataFrame({
        "gen_output_tokens_ssp": tot.baseline_sample0 + tot.hinted_rollouts,
        "gen_output_tokens_rsp": tot[gen].sum(axis=1),
        "judge_served_tokens_ssp": tot.judge_binary_orig,
        "judge_served_tokens_rsp_added": tot.judge_v2_reroll,
        "charged_tokens_total": tot[[c for c, v in COMPONENTS.items() if v[2]]].sum(axis=1),
        "n_clean": n_clean,
    })
    summ["charged_tokens_per_clean_pair"] = summ.charged_tokens_total / summ.n_clean.replace(0, np.nan)
    summ.reset_index().to_csv(out / "cost_summary.csv", index=False)
    yc.json_dump(checks, out / "cost_checks.json")
    log("cost checks: " + json.dumps(checks))
    log("\n" + cost[cost.case_group == "all"][["scope", "subject_model", "component", "output_tokens",
                                                   "judge_prompt_tokens_served", "n_units"]].to_string(index=False))


# ---------------------------------------------------------------------------

def run_funnel(srcs: list[dict], out: Path, only: str | None, cueball: Path, cache: str | Path | None,
               refresh: bool) -> dict:
    df, checks, ssp_meta = stitch_pairs(only, cueball, cache, refresh)
    stem_of = {s["source_csv"]: s["stem"] for s in srcs}
    df["stem"] = df.source_csv.map(stem_of)
    if df.stem.isna().any():
        raise SystemExit("SSP rows from a run outside the funnel's sources")
    df = stage_flags(df, noise_flip_min_votes(cueball))
    keep = ["rollout_id", "question_id", "subject_model", "run", "stem", "dataset", "original_index", "hint_style",
            "case", "target_option", "groundtruth", "baseline_stability", "b0_stored", "b0", "ssp_status",
            "model_answer", "to_hint", "truncated", "ssp_flip_truncated", "ssp_flip", "binary_judge_label",
            "ssp_unfaithful", "rel_role", "reliance_label", "k_n", "k_to_hint_count", "exclude_reason",
            "source_csv"] + [f"st_{s}" for s in STAGES]
    df[keep].to_parquet(out / "pairs.parquet", index=False)
    counts, drops = funnel_counts(df)
    counts.to_csv(out / "funnel_counts.csv", index=False)
    drops.to_csv(out / "funnel_dropouts.csv", index=False)
    checks.to_csv(out / "join_checks.csv", index=False)
    log("\n" + counts[["subject_model", "case_group"] + [f"n_{s}" for s in STAGES] +
                      [f"ssp_status_{s}" for s in S.SSP_STATUSES if s != "eligible"]
                      + ["n_truncated_to_target_excluded"]].to_string(index=False))
    costs(df, srcs, out)
    return {k: ssp_meta.get(k) for k in ("built_utc", "lib", "inputs", "flag_checks", "filters",
                                         "manifest_rows_used", "cache_key")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", nargs="?", default="all", choices=["tokens", "judge", "funnel", "all"])
    ap.add_argument("--cueball-dir", default=P.DEFAULT_CUEBALL_DIR)
    ap.add_argument("--out-dir", default=None, help="default <cueball>/plots/funnel; tables land in <out-dir>/data")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cueball = P.cueball_dir(args.cueball_dir)
    if args.out_dir is None:
        # a subset (--only) never writes into the plot's real data/ folder
        out_dir = P.plot_dir(PLOT, cueball) / "_debug" if args.only else P.plot_dir(PLOT, cueball)
        if args.only:
            log(f"--only without --out-dir: writing to the debug dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    out = out_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    if args.no_cache or args.only:
        cache: str | Path | None = None
    else:
        cache = Path(args.cache_dir) if args.cache_dir else P.cache_dir(cueball)
    srcs = run_sources(args.only, cueball)
    if not srcs:
        raise SystemExit("no sources matched")
    stem_models = yc.load_rollout_manifest(["source_csv"], cueball=cueball).drop_duplicates("source_csv")
    sm = dict(zip(stem_models.source_csv, stem_models.subject_model))
    for s in srcs:
        s["subject_model"] = sm[s["source_csv"]]
    log(f"{len(srcs)} run(s): " + ", ".join(s["stem"] for s in srcs))
    if args.stage in ("tokens", "all"):
        run_tokens(srcs, out, args.limit_rows, args.workers, args.force)
    if args.stage in ("judge", "all"):
        run_judge(srcs, out, cueball)
    ssp_meta = None
    if args.stage in ("funnel", "all"):
        ssp_meta = run_funnel(srcs, out, args.only, cueball, cache, args.refresh_cache)
    meta_path = out / "gather_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if ssp_meta is not None:
        meta["ssp_common_build"] = ssp_meta
    meta.setdefault("runs", {})[args.stage] = {"utc": yc.now_utc(), "only": args.only, "limit_rows": args.limit_rows,
                                               "workers": args.workers,
                                               "rayon_threads": os.environ.get("RAYON_NUM_THREADS")}
    meta.update({
        "plot": PLOT, "script": f"src.scripts.visualizations.{PLOT}.gather",
        "script_sha256": yc.sha256_file(Path(__file__).resolve()),
        "git_sha": S.repo_git_sha(), "shared_code": yc.shared_code_meta(),
        "cueball_dir": str(cueball), "sources": [s["stem"] for s in srcs], "models": yc.MODELS,
        "inputs": {"manifest": yc.file_info(P.manifest_path(cueball), sha=True),
                   "manifest_meta": yc.file_info(P.manifest_meta_path(cueball), sha=True),
                   "resample_manifest": yc.file_info(P.resample_manifest_path(cueball), sha=True),
                   "question_reliance": yc.file_info(P.reliance_path(cueball), sha=True)},
        "generation_csvs": {t["kind"] + "__" + Path(t["path"]).stem: yc.file_info(Path(t["path"]))
                            for t in token_tasks(srcs, out, None)},
        "tokenizers": sorted({s["model_name"] for s in srcs}),
        "hf_home": os.environ.get("HF_HOME"),
    })
    yc.json_dump(meta, meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
