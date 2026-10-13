"""Summarise a windows table: quality pass rates and RR / HR / RR-HR-ratio
distributions of the usable windows.

Usage::

    python -m scripts.summarize_windows dataset/01NVa_monitor/windows_w60_s30.parquet
"""
import sys

import numpy as np
import pandas as pd


def main(path):
    w = pd.read_parquet(path)
    print(f"{len(w)} windows, {w['patient_id'].nunique()} patients, "
          f"{len(w) * w['window_s'].iloc[0] / 3600:.0f} h of window time (overlapping)")
    for k in ("ref_ok", "ppg_ok", "ecg_ok"):
        print(f"  {k:7s} {w[k].mean():6.1%}")
    for sig in ("ppg", "ecg"):
        u = w[w["ref_ok"] & w[f"{sig}_ok"]]
        ratio = u["ref_rr"] / u[f"{sig}_hr"]
        print(f"\nref_ok & {sig}_ok: {len(u)} windows, {u['patient_id'].nunique()} patients")
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        print("  ref RR   quantiles", np.round(u["ref_rr"].quantile(q).to_numpy(), 1))
        print(f"  {sig} HR   quantiles", np.round(u[f"{sig}_hr"].quantile(q).to_numpy(), 1))
        print("  RR/HR    quantiles", np.round(ratio.quantile(q).to_numpy(), 2))
        print("  share with RR/HR > 0.4:", f"{(ratio > 0.4).mean():.1%}", " > 0.5:", f"{(ratio > 0.5).mean():.1%}")
        per_patient = u.groupby("patient_id").size()
        print(f"  windows per patient: median {per_patient.median():.0f}, max {per_patient.max()}")


if __name__ == "__main__":
    main(sys.argv[1])
