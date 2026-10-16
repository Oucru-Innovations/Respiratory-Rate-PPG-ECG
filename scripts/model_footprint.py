"""Size and CPU cost of every trained model, plus the cost of its input features.

For each run with a ``best.pt`` under ``runs/``: parameter count, fp32 size,
and the median single-window latency on ONE CPU thread (batch 1, after
warm-up) of (a) the network and (b) computing its input from 60 s of signal
(beat detection, respiratory signals, STFT). Measured on this machine, so
compare models with each other rather than across machines.

Usage::

    python -m scripts.model_footprint --fold 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dl.features import SpecConfig, make_features
from dl.model import build_model
from tests.synthetic import synth_ecg, synth_ppg

FS = {"ppg": 100.0, "ecg": 250.0}


def _median_ms(fn, repeats, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)

    signals = {"ppg": synth_ppg(18, hr_bpm=85, duration=60), "ecg": synth_ecg(18, hr_bpm=85, duration=60)}
    feature_ms = {}
    rows = []
    for ck in sorted(Path("runs").glob(f"*_f{args.fold}/best.pt")):
        ckpt = torch.load(ck, map_location="cpu")
        sig = ckpt["features"]["signal"]
        spec = SpecConfig(**ckpt["features"]["spec"])
        key = (sig, spec.kind)
        if key not in feature_ms:
            feature_ms[key] = _median_ms(lambda: make_features(signals[sig], FS[sig], sig, spec), 20, warmup=2)
        model = build_model(**ckpt["model"]).eval()
        model.load_state_dict(ckpt["state_dict"])
        x = torch.randn(1, *ckpt["features"]["shape"])
        with torch.no_grad():
            net_ms = _median_ms(lambda: model(x), args.repeats)
        n_params = sum(p.numel() for p in model.parameters())
        rows.append({"run": ck.parent.name, "signal": sig, "features": spec.kind,
                     "arch": ckpt["model"].get("arch", "cnn"), "params": n_params,
                     "size_mb_fp32": 4 * n_params / 1e6, "net_ms": net_ms,
                     "feature_ms": feature_ms[key], "total_ms": net_ms + feature_ms[key]})
        print(f"{rows[-1]['run']:34s} {n_params:>9,} params  net {net_ms:6.2f} ms  features {feature_ms[key]:6.1f} ms")
    out = Path(args.out or f"runs/footprint_f{args.fold}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
