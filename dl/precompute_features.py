"""Precompute spectrogram features for every usable window into one on-disk array.

Output directory layout::

    <out>/features.npy   float16 memmap, shape (N, C, F, T)
    <out>/index.parquet  one row per feature row: window metadata + ref_rr
    <out>/config.json    SpecConfig, signal type, source windows file

Usage::

    python -m dl.precompute_features --windows dataset/01NVa_monitor/windows_w60_s15.parquet \
        --signal ppg --kind modulation --out dataset/01NVa_monitor/features/ppg_mod_w60
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep.build_dataset import TARGET_FS
from dl.features import SpecConfig, feature_shape, make_features


def _segment_features(seg_id, rows, sig_dir, signal_type, window_s, cfg):
    x_all = np.load(sig_dir / f"{seg_id}.npz")[signal_type]
    fs = TARGET_FS[signal_type]
    out = []
    for r in rows:
        a = int(round(r["start_s"] * fs))
        out.append(make_features(x_all[a:a + int(round(window_s * fs))].astype(float), fs, signal_type, cfg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--signal", choices=["ppg", "ecg"], required=True)
    ap.add_argument("--kind", choices=["modulation", "raw", "modwave", "modwave_hr"], default="modulation")
    ap.add_argument("--stft-window", type=float, default=16.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include-bad-reference", action="store_true",
                    help="also keep windows whose reference failed quality (not for training)")
    args = ap.parse_args()

    windows_path = Path(args.windows)
    sig_dir = windows_path.parent / "signals"
    w = pd.read_parquet(windows_path)
    keep = w[f"{args.signal}_ok"]
    if not args.include_bad_reference:
        keep &= w["ref_ok"]
    w = w[keep].reset_index().rename(columns={"index": "window_row"})
    window_s = float(w["window_s"].iloc[0])
    cfg = SpecConfig(kind=args.kind, stft_window_s=args.stft_window)
    shape = feature_shape(window_s, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                      shape=(len(w),) + shape)
    ok = np.zeros(len(w), dtype=bool)
    groups = list(w.groupby("segment_id").groups.items())
    print(f"{len(w)} windows in {len(groups)} segments -> {shape}")
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_segment_features, seg, w.loc[idx, ["start_s"]].to_dict("records"),
                          sig_dir, args.signal, window_s, cfg): idx for seg, idx in groups}
        for i, (fut, idx) in enumerate(futs.items(), 1):
            for row, f in zip(idx, fut.result()):
                if f is not None and f.shape == shape and np.all(np.isfinite(f)):
                    feats[row] = f
                    ok[row] = True
            if i % 50 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] segments", flush=True)
    feats.flush()
    w["feature_ok"] = ok
    w.to_parquet(out / "index.parquet", index=False)
    (out / "config.json").write_text(json.dumps({
        "signal": args.signal, "windows": str(windows_path), "window_s": window_s,
        "shape": list(shape), "spec": cfg.to_dict()}, indent=2))
    print(f"done: {ok.sum()}/{len(ok)} windows with valid features")


if __name__ == "__main__":
    main()
