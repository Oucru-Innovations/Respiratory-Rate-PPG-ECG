"""Evaluate a trained checkpoint on a feature set.

By default only the checkpoint's held-out test patients are used; pass
``--all`` to score every window (e.g. a different dataset).

Usage::

    python -m dl.evaluate --checkpoint runs/ppg_mod_w60_f0/best.pt \
        --features dataset/01NVa_monitor/features/ppg_mod_w60 --out runs/ppg_mod_w60_f0/eval
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from common.metrics import metrics_by_range, rr_metrics
from dl.model import LabelDistribution, SpectroCNN
from dl.train import make_loader, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="score all windows, not only test patients")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    feat_dir = Path(args.features)
    feat_cfg = json.loads((feat_dir / "config.json").read_text())
    if feat_cfg["shape"] != ckpt["features"]["shape"] or feat_cfg["spec"] != ckpt["features"]["spec"]:
        raise SystemExit(f"feature config mismatch:\n{feat_cfg}\nvs checkpoint\n{ckpt['features']}")

    ldl = LabelDistribution(sigma=ckpt["ldl"]["sigma"]).to(device)
    model = SpectroCNN(**ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    index = pd.read_parquet(feat_dir / "index.parquet")
    index["feature_row"] = range(len(index))
    df = index[index["feature_ok"]]
    if not args.all:
        df = df[df["patient_id"].isin(ckpt["split"]["test"])]
    df = df.reset_index(drop=True)
    df["pred_rr"] = predict(model, ldl, make_loader(feat_dir / "features.npy", df, args.batch,
                                                    args.workers, False), device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df[["window_row", "segment_id", "patient_id", "start_s", "ref_rr", "pred_rr"]].to_csv(
        out / "predictions.csv", index=False)
    metrics = {"overall": rr_metrics(df["ref_rr"], df["pred_rr"], df["patient_id"]),
               "by_range": metrics_by_range(df["ref_rr"], df["pred_rr"])}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
