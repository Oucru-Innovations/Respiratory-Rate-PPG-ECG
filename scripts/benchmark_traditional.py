"""Benchmark the signal-processing RR estimators against the impedance reference.

For every usable window: extract BW/AM/FM once, apply every method, fuse with
smart fusion and median fusion. Methods changed by the very-low-frequency
corrections (see ``estimate_rr``) are also scored as published (``pub-*``).
Use ``--split-json`` to restrict to the same test patients as a trained DL
model so the comparison is like-for-like.

Usage::

    python -m scripts.benchmark_traditional --windows dataset/01NVa_monitor/windows_w60_s15.parquet \
        --signal ppg --split-json runs/ppg_mod_w60_f0/split.json --out runs/traditional_ppg_f0
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common.metrics import rr_metrics
from data_prep.build_dataset import TARGET_FS
from estimate_rr import (FREQ_METHODS, FUSIONS, METHODS, MODULATIONS, PREWHITEN,
                         extract_resp_signals, rate_from_resp_signal)
from estimate_rr.resp_signals import EXTRA_SIGNALS

CORRECTED_METHODS = FREQ_METHODS | set(PREWHITEN)  # CtO is used as published

RATIO_BINS = [0, 0.2, 0.3, 0.4, 0.5, np.inf]


def _segment(seg_id, rows, sig_dir, signal_type, window_s):
    x_all = np.load(sig_dir / f"{seg_id}.npz")[signal_type]
    fs = TARGET_FS[signal_type]
    out = []
    for r in rows:
        a = int(round(r["start_s"] * fs))
        rs = extract_resp_signals(x_all[a:a + int(round(window_s * fs))].astype(float), fs,
                                  signal_type, modulations=MODULATIONS + EXTRA_SIGNALS)
        res = {"window_row": r["window_row"], "hr": rs.hr}
        for method in METHODS:
            per = {m: rate_from_resp_signal(rs.signals[m], rs.fs, method) if m in rs.signals else np.nan
                   for m in MODULATIONS + EXTRA_SIGNALS}
            for m, v in per.items():
                res[f"{method}|{m}"] = v
            # fusion follows the published definitions: BW/AM/FM only
            for fname, fuse in FUSIONS.items():
                res[f"{method}|{fname}"] = fuse([per[m] for m in MODULATIONS])
            if method in CORRECTED_METHODS:  # the same method exactly as published, for comparison
                pub = [rate_from_resp_signal(rs.signals[m], rs.fs, method, corrected=False)
                       if m in rs.signals else np.nan for m in MODULATIONS]
                for fname, fuse in FUSIONS.items():
                    res[f"{method}|pub-{fname}"] = fuse(pub)
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--signal", choices=["ppg", "ecg"], required=True)
    ap.add_argument("--split-json", help="restrict to the patients of one split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=0, help="random subsample of windows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    wpath = Path(args.windows)
    w = pd.read_parquet(wpath)
    w = w[w["ref_ok"] & w[f"{args.signal}_ok"]].reset_index().rename(columns={"index": "window_row"})
    if args.split_json:
        w = w[w["patient_id"].isin(json.loads(Path(args.split_json).read_text())[args.split])]
    if args.sample and len(w) > args.sample:
        w = w.sample(args.sample, random_state=0)
    window_s = float(w["window_s"].iloc[0])
    print(f"{len(w)} windows, {w['patient_id'].nunique()} patients")

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_segment, seg, g[["window_row", "start_s"]].to_dict("records"),
                          wpath.parent / "signals", args.signal, window_s)
                for seg, g in w.groupby("segment_id")]
        for f in futs:
            rows.extend(f.result())
    pred = w[["window_row", "segment_id", "patient_id", "start_s", "ref_rr"]].merge(
        pd.DataFrame(rows), on="window_row")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    table = []
    for col in pred.columns:
        if "|" in col:
            method, sig = col.split("|")
            table.append({"method": method, "abbrev": METHODS[method][1], "signal": sig,
                          **rr_metrics(pred["ref_rr"], pred[col], pred["patient_id"])})
    table = pd.DataFrame(table).sort_values(["signal", "mae"])
    table.to_csv(out / "metrics.csv", index=False)
    cols = ["method", "abbrev", "signal", "coverage", "mae", "patient_macro_mae", "bias",
            "loa_low", "loa_high", "within_2bpm"]
    print(table[table["signal"].isin(list(FUSIONS) + list(EXTRA_SIGNALS))][cols].round(2).to_string(index=False))

    # Beat-sampled respiratory signals cannot represent RR > HR/2: stratify by RR/HR
    pred["rr_hr_ratio"] = pred["ref_rr"] / pred["hr"]
    bins = pd.cut(pred["rr_hr_ratio"], RATIO_BINS)
    strat = []
    for col in [c for c in pred.columns if "|" in c]:
        for b, g in pred.groupby(bins, observed=True):
            strat.append({"column": col, "rr_hr_ratio": str(b), **rr_metrics(g["ref_rr"], g[col])})
    pd.DataFrame(strat).to_csv(out / "metrics_by_rr_hr_ratio.csv", index=False)


if __name__ == "__main__":
    main()
