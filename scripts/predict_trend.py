"""Score every window of the test patients' recordings for trend analysis.

Unlike the benchmark, windows whose impedance reference failed its quality
checks are kept (flagged by ``ref_ok``), so a patient's whole recording can be
plotted as a time series. Writes ``<out>/trend_<signal>.parquet`` with the DL
estimate and three signal-processing estimates (CtO, ARM, WCH; smart fusion,
corrected rules) per window.

Usage::

    python -m scripts.predict_trend --signal ppg --run runs/ppg_mod_w60_f0 --out runs/trends_f0
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_prep.build_dataset import TARGET_FS
from dl.features import SpecConfig, make_features
from dl.model import LabelDistribution, SpectroCNN
from estimate_rr import FUSIONS, MODULATIONS, extract_resp_signals, rate_from_resp_signal

TRAD = ("count_orig", "ar_median", "welch")


def _segment(seg_id, rows, sig_dir, signal_type, window_s, spec):
    x_all = np.load(sig_dir / f"{seg_id}.npz")[signal_type]
    fs = TARGET_FS[signal_type]
    feats, out = [], []
    for r in rows:
        a = int(round(r["start_s"] * fs))
        x = x_all[a:a + int(round(window_s * fs))].astype(float)
        feats.append(make_features(x, fs, signal_type, SpecConfig(**spec)))
        rs = extract_resp_signals(x, fs, signal_type)
        res = {"window_row": r["window_row"], "hr": rs.hr}
        for method in TRAD:
            per = [rate_from_resp_signal(rs.signals[m], rs.fs, method) if m in rs.signals else np.nan
                   for m in MODULATIONS]
            res[f"{method}|smart"] = FUSIONS["smart"](per)
            res[f"{method}|median"] = FUSIONS["median"](per)
        out.append(res)
    return feats, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=["ppg", "ecg"], required=True)
    ap.add_argument("--run", required=True, help="DL run directory (best.pt + split.json)")
    ap.add_argument("--windows", default="dataset/01NVa_monitor/windows_w60_s30.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="cpu",
                    help="inference device; CPU by default so it never competes with a training run for GPU memory")
    args = ap.parse_args()

    run = Path(args.run)
    device = torch.device(args.device)
    ckpt = torch.load(run / "best.pt", map_location=device)
    patients = json.loads((run / "split.json").read_text())[args.split]
    wpath = Path(args.windows)
    w = pd.read_parquet(wpath).reset_index().rename(columns={"index": "window_row"})
    w = w[w["patient_id"].isin(patients) & w[f"{args.signal}_ok"]]
    window_s = float(w["window_s"].iloc[0])
    spec = ckpt["features"]["spec"]
    print(f"{len(w)} windows ({w['ref_ok'].mean():.0%} with a usable reference), {w['patient_id'].nunique()} patients")

    feats, rows = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_segment, seg, g[["window_row", "start_s"]].to_dict("records"),
                          wpath.parent / "signals", args.signal, window_s, spec)
                for seg, g in w.groupby("segment_id")]
        for f in futs:
            fe, ro = f.result()
            feats.extend(fe)
            rows.extend(ro)

    ldl = LabelDistribution(sigma=ckpt["ldl"]["sigma"]).to(device)
    model = SpectroCNN(**ckpt["model"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    shape = tuple(ckpt["features"]["shape"])
    ok = np.array([f is not None and f.shape == shape and np.all(np.isfinite(f)) for f in feats])
    dl = np.full(len(feats), np.nan)
    idx = np.flatnonzero(ok)
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = torch.from_numpy(np.stack([feats[j] for j in idx[i:i + 256]])).float().to(device)
            dl[idx[i:i + 256]] = ldl.decode(model(b)).cpu().numpy()

    res = pd.DataFrame(rows)
    res["dl"] = dl
    res = w[["window_row", "segment_id", "patient_id", "start_s", "t_epoch", "ref_rr", "ref_ok"]].merge(res, on="window_row")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / f"trend_{args.signal}.parquet", index=False)
    print(f"wrote {out / f'trend_{args.signal}.parquet'}: {len(res)} windows, DL on {ok.mean():.0%}")


if __name__ == "__main__":
    main()
