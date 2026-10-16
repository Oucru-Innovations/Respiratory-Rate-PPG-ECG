"""Score the monitor-trained models and the signal-processing methods on the
wearables (``dataset/01NVa_wearable``, built by ``data_prep.build_wearable``).

No retraining: every patient's wearable windows are scored by the cross-validation
model of the fold in which that patient was a *test* patient, so no model has seen
the patient. The reference is the monitor's impedance RR over the same 60 s.

Stages (from the repo root)::

    python -m scripts.wearable_eval features      # DL inputs from the wearable windows
    python -m scripts.wearable_eval dl            # every fold's model on its test patients
    python -m scripts.wearable_eval traditional   # signal-processing benchmark
    python -m scripts.wearable_eval summary       # -> runs/wearable/summary.json

``summary`` reports two views per signal: all wearable windows with a usable
reference, and the *paired* windows where the monitor's own PPG/ECG is usable
too, scored for both inputs so monitor vs wearable is a like-for-like comparison.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.metrics import rr_metrics

WEAR = Path("dataset/01NVa_wearable")
RUNS = Path("runs")
OUT = RUNS / "wearable"
FOLDS = range(5)
KIND_TAG = {"modulation": "mod", "modwave": "modwave", "modwave_hr": "modwavehr"}
# monitor-trained models to transfer: run stem (without _f<k>), feature kind, label
MODELS = [("{s}_mod_w60", "modulation", "DL CNN"),
          ("{s}_modwave_w60_breathnet", "modwave", "BreathNet"),
          ("{s}_modwavehr_w60_breathnet2-coh-hr-conv", "modwave_hr", "BreathNet + coherent + HR prior + conv head"),
          ("{s}_modwavehr_w60_breathnet2", "modwave_hr", "BreathNet-2 (690 parameters)"),
          ("{s}_modwavehr_w60_breathnet2-track", "modwave_hr", "BreathNet + HMM tracking (585 parameters)")]
TRAD = [("ar_median|smart", "ARM · smart fusion"), ("welch|smart", "WCH · smart fusion"),
        ("count_orig|smart", "CtO · smart fusion"), ("ar_median|median", "ARM · median fusion"),
        ("count_orig|median", "CtO · median fusion"), ("ar_median|pub-smart", "ARM · smart fusion, as published")]
SIGNALS = ("ppg", "ecg")
PY = sys.executable


def windows_path(sig):
    return WEAR / f"windows_{sig}_w60_s30.parquet"


def feature_dir(sig, kind):
    return WEAR / "features" / f"{sig}_{KIND_TAG[kind]}_w60"


def run(cmd, env=None):
    print(" ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env={**os.environ, **(env or {})})


def stage_features(args):
    for sig in SIGNALS:
        if not windows_path(sig).exists():
            continue
        for kind in dict.fromkeys(k for _, k, _ in MODELS):
            out = feature_dir(sig, kind)
            if (out / "config.json").exists() and not args.redo:
                continue
            run([PY, "-W", "ignore", "-m", "dl.precompute_features", "--windows", windows_path(sig), "--signal", sig,
                 "--kind", kind, "--out", out, "--workers", args.workers])


def stage_dl(args):
    for sig in SIGNALS:
        for stem_t, kind, _ in MODELS:
            stem = stem_t.format(s=sig)
            for k in FOLDS:
                ckpt = RUNS / f"{stem}_f{k}" / "best.pt"
                out = OUT / f"{stem}_f{k}"
                if not ckpt.exists() or not (feature_dir(sig, kind) / "config.json").exists():
                    continue
                if (out / "predictions.csv").exists() and not args.redo:
                    continue
                # CPU: leaves the GPU to training; default = this fold's test patients only
                run([PY, "-W", "ignore", "-m", "dl.evaluate", "--checkpoint", ckpt, "--features", feature_dir(sig, kind),
                     "--out", out, "--workers", 4], env={"CUDA_VISIBLE_DEVICES": ""})


def stage_traditional(args):
    for sig in SIGNALS:
        out = OUT / f"traditional_{sig}"
        if not windows_path(sig).exists() or ((out / "metrics.csv").exists() and not args.redo):
            continue
        run([PY, "-W", "ignore", "-m", "scripts.benchmark_traditional", "--windows", windows_path(sig),
             "--signal", sig, "--workers", args.workers, "--out", out])


# ---------------------------------------------------------------- summary
def patient_ci(ref, pred, pid, n=2000, seed=0):
    d = pd.DataFrame({"e": (pred - ref).abs(), "p": pid}).dropna()
    per = d.groupby("p")["e"].mean().to_numpy()
    if len(per) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def score(label, ref, pred, pid, extra=None):
    m = rr_metrics(ref, pred, pid)
    lo, hi = patient_ci(ref, pred, pid)
    return {"estimator": label, **m, "ci_lo": lo, "ci_hi": hi, **(extra or {})}


def monitor_dl_preds(stem):
    parts = [pd.read_csv(RUNS / f"{stem}_f{k}" / "test_predictions.csv") for k in FOLDS
             if (RUNS / f"{stem}_f{k}" / "test_predictions.csv").exists()]
    return pd.concat(parts).set_index("window_row")["pred_rr"] if parts else None


def monitor_trad_preds(sig, cols):
    parts = [pd.read_parquet(RUNS / f"traditional_{sig}_f{k}_all" / "predictions.parquet", columns=["window_row"] + cols)
             for k in FOLDS if (RUNS / f"traditional_{sig}_f{k}_all" / "predictions.parquet").exists()]
    return pd.concat(parts).set_index("window_row") if parts else None


def stage_summary(args):
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"signals": {}}
    al = pd.read_csv(WEAR / "alignment.csv")
    files = al.drop_duplicates("path")
    acc = al[al["status"] == "aligned"]
    summary["alignment"] = {
        kind: {"files": int((files["kind"] == kind).sum()),
               "files_aligned": int(acc.loc[acc["kind"] == kind, "path"].nunique()),
               "patients_aligned": int(acc.loc[acc["kind"] == kind, "patient_id"].nunique()),
               "hours_aligned": float(((acc["t_end"] - acc["t_start"])[acc["kind"] == kind]).sum() / 3600),
               "median_block_r": float(acc.loc[acc["kind"] == kind, "median_r"].median()),
               "median_resid_s": float(acc.loc[acc["kind"] == kind, "resid_sd_s"].median()),
               "drift_ppm_iqr": [float(v) for v in acc.loc[acc["kind"] == kind, "drift_ppm"].quantile([.25, .75])],
               "offset_minus_nominal_h": acc[acc["kind"] == kind].drop_duplicates("path")["offset_minus_nominal_h"].round(1).value_counts().head(6).to_dict()}
        for kind in SIGNALS}
    for sig in SIGNALS:
        if not windows_path(sig).exists():
            continue
        w = pd.read_parquet(windows_path(sig)).reset_index().rename(columns={"index": "window_row"})
        usable = w[w["ref_ok"]]
        s = {"n_windows_worn": int(len(usable)), "n_patients": int(usable["patient_id"].nunique()),
             "wearable_ok": float(usable[f"{sig}_ok"].mean()), "monitor_ok": float(usable["monitor_ok"].mean())}
        W = usable[usable[f"{sig}_ok"]].set_index("window_row")
        preds = pd.DataFrame(index=W.index)
        for stem_t, _, label in MODELS:
            stem = stem_t.format(s=sig)
            parts = [pd.read_csv(OUT / f"{stem}_f{k}" / "predictions.csv") for k in FOLDS
                     if (OUT / f"{stem}_f{k}" / "predictions.csv").exists()]
            if len(parts) == 5:
                preds[label] = pd.concat(parts).set_index("window_row")["pred_rr"]
        tp = OUT / f"traditional_{sig}" / "predictions.parquet"
        if tp.exists():
            t = pd.read_parquet(tp).set_index("window_row")
            for col, label in TRAD:
                if col in t:
                    preds[label] = t[col]
        s["all"] = [score(c, W["ref_rr"], preds[c], W["patient_id"]) for c in preds.columns]
        s["all_n"] = int(len(W))
        # paired: the monitor's own channel is usable in the same window
        P = W[W["monitor_ok"]]
        mon = pd.DataFrame(index=P.index)
        for stem_t, _, label in MODELS:
            mp = monitor_dl_preds(stem_t.format(s=sig))
            if mp is not None and label in preds:
                mon[label] = P["monitor_window_row"].map(mp)
        mt = monitor_trad_preds(sig, [c for c, _ in TRAD])
        if mt is not None:
            for col, label in TRAD:
                if label in preds:
                    mon[label] = P["monitor_window_row"].map(mt[col])
        rows = []
        for label in mon.columns:
            both = P.index[mon[label].notna() | preds.loc[P.index, label].notna()]
            mw = score(label, P["ref_rr"], preds.loc[P.index, label], P["patient_id"])
            mm = score(label, P["ref_rr"], mon[label], P["patient_id"])
            ew, em = (preds.loc[P.index, label] - P["ref_rr"]).abs(), (mon[label] - P["ref_rr"]).abs()
            per = pd.DataFrame({"w": ew, "m": em, "p": P["patient_id"]}).dropna().groupby("p").mean()
            from scipy.stats import wilcoxon
            p = float(wilcoxon(per["w"] - per["m"]).pvalue) if len(per) > 5 else np.nan
            rows.append({"estimator": label, "monitor": mm, "wearable": mw, "n_patients": int(len(per)),
                         "wearable_worse_patients": int((per["w"] > per["m"]).sum()), "p": p,
                         "pred_corr": float(np.corrcoef(*pd.DataFrame({"a": preds.loc[P.index, label], "b": mon[label]}).dropna().to_numpy().T)[0, 1])})
        s["paired"] = rows
        s["paired_n"] = int(len(P))
        summary["signals"][sig] = s
        # window-level predictions for the report figures
        preds.join(W[["patient_id", "ref_rr", "t_epoch", "monitor_window_row", "monitor_ok"]]).to_parquet(OUT / f"predictions_{sig}.parquet")
        mon.join(P[["patient_id", "ref_rr"]]).to_parquet(OUT / f"monitor_predictions_paired_{sig}.parquet")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for sig, s in summary["signals"].items():
        print(f"\n[{sig.upper()}] {s['n_windows_worn']:,} windows worn with a usable reference, {s['n_patients']} patients; "
              f"wearable usable {s['wearable_ok']:.0%}, monitor usable {s['monitor_ok']:.0%}")
        for r in sorted(s["all"], key=lambda r: r["patient_macro_mae"]):
            print(f"   {r['estimator'][:44]:44s} coverage {r['coverage']:.0%}  MAE {r['mae']:.2f}  per-patient {r['patient_macro_mae']:.2f} "
                  f"({r['ci_lo']:.2f}-{r['ci_hi']:.2f})  within 2 {r['within_2bpm']:.0%}")
        print(f"   paired ({s['paired_n']:,} windows, monitor usable too): per-patient MAE monitor -> wearable")
        for r in s["paired"]:
            print(f"   {r['estimator'][:44]:44s} {r['monitor']['patient_macro_mae']:.2f} -> {r['wearable']['patient_macro_mae']:.2f}  "
                  f"(wearable worse for {r['wearable_worse_patients']}/{r['n_patients']} patients, p = {r['p']:.2g}; r(pred) {r['pred_corr']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "dl", "traditional", "summary", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    stages = ["features", "dl", "traditional", "summary"] if a.stage == "all" else [a.stage]
    for st in stages:
        {"features": stage_features, "dl": stage_dl, "traditional": stage_traditional, "summary": stage_summary}[st](a)


if __name__ == "__main__":
    main()
