"""Estimate respiratory rate from one window of monitor data with every method.

Run from the repo root after building the dataset (stage 1)::

    python -m examples.main_rr_estimation dataset/01NVa_monitor/signals/<segment>.npz --start 600
"""
import argparse

import numpy as np

from data_prep.build_dataset import TARGET_FS
from data_prep.reference import reference_rr
from estimate_rr import METHODS, estimate_rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--start", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--window", type=float, default=60.0)
    args = ap.parse_args()

    data = np.load(args.npz)
    cut = {k: data[k][int(args.start * fs):int((args.start + args.window) * fs)].astype(float)
           for k, fs in TARGET_FS.items()}
    ref = reference_rr(cut["resp"], TARGET_FS["resp"])
    print(f"reference (impedance): {ref['ref_rr']:.1f} bpm   quality ok: {ref['ref_ok']}")
    print(f"{'method':16s} {'PPG':>8s} {'ECG':>8s}")
    for method, (_, abbrev) in METHODS.items():
        r = [estimate_rr(cut[s], TARGET_FS[s], s, method).rr for s in ("ppg", "ecg")]
        print(f"{method:12s} {abbrev:3s} {r[0]:8.1f} {r[1]:8.1f}")


if __name__ == "__main__":
    main()
