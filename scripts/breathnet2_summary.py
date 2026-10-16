"""Compare the BreathNet-2 screening runs with BreathNet and the CNN on fold 0.

Every model is scored on the identical test windows; the comparison is a paired
Wilcoxon test on per-patient MAE. Also reads the physiology each variant learned
(RR/HR prior, HR ceiling, HMM step width, coherent-sum phases and lags).

    python -m scripts.breathnet2_summary [--fold 0]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from common.metrics import rr_metrics

RUNS = Path("runs")
VARIANTS = [("breathnet2-whiten", "+ whiten"), ("breathnet2-hr", "+ hr prior"), ("breathnet2-track", "+ HMM track"),
            ("breathnet2-coherent", "+ coherent"), ("breathnet2-conv", "+ conv head"),
            ("breathnet2", "BreathNet-2 (whiten+coh+track+hr)"), ("breathnet2-coh-hr", "coherent + hr"),
            ("breathnet2-coh-hr-track", "coherent + hr + track"), ("breathnet2-coh-hr-conv", "coherent + hr + conv"),
            ("breathnet2-all-conv", "whiten + coherent + hr + conv")]


def per_patient(run):
    p = pd.read_csv(run / "test_predictions.csv")
    return p.assign(err=(p["pred_rr"] - p["ref_rr"]).abs()).groupby("patient_id")["err"].mean()


def paired(a, b):
    """Patients where b beats a, mean difference and Wilcoxon p (b - a)."""
    d = (b - a).dropna()
    return int((d < 0).sum()), len(d), float(d.mean()), float(wilcoxon(d).pvalue) if len(d) > 5 else float("nan")


def n_params(run):
    from dl.model import build_model
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    return sum(p.numel() for p in build_model(**ck["model"]).parameters())


def learned(run):
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    sd, out = ck["state_dict"], {}
    if "hr_ratio_mu" in sd:
        out["RR/HR centre"] = math.exp(float(sd["hr_ratio_mu"]))
        out["RR/HR sd (log)"] = math.exp(float(sd["hr_ratio_log_sd"]))
        out["ceiling (x HR)"] = float(sd["hr_ceiling"])
        out["ceiling slope"] = math.log1p(math.exp(float(sd["hr_ceiling_slope"])))
    if "log_trans_sigma" in sd:
        out["HMM step sd (bpm/frame)"] = math.exp(float(sd["log_trans_sigma"]))
    if "coh_phase" in sd:
        out["coherent gain BW,AM,FM"] = ", ".join(f"{g:.2f}" for g in sd["coh_log_gain"].exp().tolist())
        out["coherent phase (deg)"] = ", ".join(f"{math.degrees(p):+.0f}" for p in sd["coh_phase"].tolist())
        out["coherent lag (s)"] = ", ".join(f"{l:+.2f}" for l in sd["coh_lag"].tolist())
    out["temperature"] = float(sd["temperature"])
    out["epochs"] = int(ck["epoch"]) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    rows, notes = [], {}
    for sig in ("ppg", "ecg"):
        base = {"CNN": RUNS / f"{sig}_mod_w60_f{a.fold}", "BreathNet": RUNS / f"{sig}_modwave_w60_breathnet_f{a.fold}"}
        if not all((r / "metrics.json").exists() for r in base.values()):
            continue
        pp = {k: per_patient(r) for k, r in base.items()}
        for k, r in base.items():
            m = json.loads((r / "metrics.json").read_text())["test"]
            rows.append({"signal": sig.upper(), "model": k, "params": n_params(r), "MAE": m["mae"], "patient MAE": m["patient_macro_mae"],
                         "within 2 bpm": m["within_2bpm"], "val patient MAE": json.loads((r / "metrics.json").read_text())["best_val_patient_mae"]})
        for suffix, label in VARIANTS:
            r = RUNS / f"{sig}_modwavehr_w60_{suffix}_f{a.fold}"
            if not (r / "metrics.json").exists():
                rows.append({"signal": sig.upper(), "model": label, "MAE": float("nan")})
                continue
            mj = json.loads((r / "metrics.json").read_text())
            m, q = mj["test"], per_patient(r)
            wb, nb, db, pb = paired(pp["BreathNet"], q)
            wc, nc, dc, pc = paired(pp["CNN"], q)
            rows.append({"signal": sig.upper(), "model": label, "params": n_params(r), "MAE": m["mae"], "patient MAE": m["patient_macro_mae"],
                         "within 2 bpm": m["within_2bpm"], "val patient MAE": mj["best_val_patient_mae"],
                         "vs BreathNet: better/n": f"{wb}/{nb}", "vs BreathNet: d (bpm)": db, "vs BreathNet: p": pb,
                         "vs CNN: better/n": f"{wc}/{nc}", "vs CNN: d (bpm)": dc, "vs CNN: p": pc})
            notes[(sig.upper(), label)] = learned(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = RUNS / f"breathnet2_fold{a.fold}_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}\n")
    for (sig, label), v in notes.items():
        print(f"{sig} {label}: " + "; ".join(f"{k} = {val:.3g}" if isinstance(val, float) else f"{k} = {val}" for k, val in v.items()))
    (RUNS / f"breathnet2_fold{a.fold}_learned.json").write_text(
        json.dumps([{"signal": sig, "model": label, **v} for (sig, label), v in notes.items()], indent=1))


if __name__ == "__main__":
    main()
