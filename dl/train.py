"""Train the spectrogram RR estimator with a patient-wise split.

Usage::

    python -m dl.train --features dataset/01NVa_monitor/features/ppg_mod_w60 --out runs/ppg_mod_w60_f0 --fold 0

Writes to ``--out``: ``best.pt`` (weights + config), ``split.json`` (patient
IDs per split), ``history.csv``, ``test_predictions.csv`` and ``metrics.json``.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common.metrics import metrics_by_range, rr_metrics
from common.splits import patient_split
from dl.model import ARCHS, WAVEFORM_ARCHS, LabelDistribution, build_model

DEFAULT_WIDTH = {"cnn": 32, "lstm": 64, "gru": 64, "unet": 16, "mlp": 128, "breathnet": 32, "breathnet2": 32}
CLIPPED = {"transformer", "lstm", "gru", "breathnet", "breathnet2"}


class FeatureDataset(Dataset):
    def __init__(self, feature_file, rows, targets, augment=False):
        self.feature_file = str(feature_file)
        self.rows = np.asarray(rows)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.augment = augment
        self._x = None  # opened lazily so each DataLoader worker has its own memmap

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._x is None:
            self._x = np.load(self.feature_file, mmap_mode="r")
        x = np.asarray(self._x[self.rows[i]], dtype=np.float32)
        if self.augment:
            x = _time_mask(x)
        return torch.from_numpy(x), torch.tensor(self.targets[i])


def _time_mask(x, max_frac=0.2):
    """Zero a random block of time frames. Frequency-axis augmentations are not
    used: shifting frequency would change the label."""
    t = x.shape[-1]
    w = np.random.randint(0, max(int(max_frac * t), 1) + 1)
    if w:
        s = np.random.randint(0, t - w + 1)
        x = x.copy()
        x[..., s:s + w] = 0.0
    return x


def load_split(features_dir, n_folds, fold, seed):
    index = pd.read_parquet(Path(features_dir) / "index.parquet")
    index["feature_row"] = np.arange(len(index))
    index = index[index["feature_ok"]].reset_index(drop=True)
    split = patient_split(index["patient_id"], n_folds, fold, seed)
    parts = {k: index[index["patient_id"].isin(v)].reset_index(drop=True) for k, v in split.items()}
    return parts, split


@torch.no_grad()
def predict(model, ldl, loader, device):
    model.eval()
    preds = []
    for x, _ in loader:
        preds.append(ldl.decode(model(x.to(device))).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def make_loader(features_file, df, batch, workers, train, balance=False):
    ds = FeatureDataset(features_file, df["feature_row"], df["ref_rr"], augment=train)
    sampler = None
    if train and balance:
        # inverse frequency of 2-bpm reference bins, so rare low/high rates are seen
        bins = np.floor(df["ref_rr"].to_numpy() / 2)
        counts = pd.Series(bins).map(pd.Series(bins).value_counts()).to_numpy()
        sampler = WeightedRandomSampler(1.0 / counts, num_samples=len(df), replacement=True)
    return DataLoader(ds, batch_size=batch, shuffle=train and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def model_kwargs(args, shape, n_out, feat_kind="modwave"):
    """Constructor arguments for ``args.arch`` given the feature shape; stored in
    the checkpoint so evaluation rebuilds the identical model."""
    waveform = len(shape) == 2
    if waveform != (args.arch in WAVEFORM_ARCHS):
        raise SystemExit(f"--arch {args.arch} needs {'waveform (modwave)' if not waveform else 'spectrogram'} "
                         f"features; got shape {shape}")
    width = args.width or DEFAULT_WIDTH.get(args.arch)
    if waveform:
        kw = {"arch": args.arch, "in_channels": shape[0], "n_samples": shape[1], "n_out": n_out,
              "width": width, "dropout": args.dropout}
        hr_channel = feat_kind == "modwave_hr"
        if args.arch == "breathnet2":
            kw.update(variant=args.variant, hr_channel=hr_channel)
            if "hr" in args.variant.split(",") and not hr_channel:
                raise SystemExit("--variant hr needs modwave_hr features (python -m dl.precompute_features --kind modwave_hr)")
        elif hr_channel:
            raise SystemExit(f"--arch {args.arch} takes modwave features; modwave_hr is for breathnet2")
        return kw
    kw = {"arch": args.arch, "in_channels": shape[0], "n_freq": shape[1], "n_out": n_out}
    if args.arch == "transformer":
        kw.update(tokens=args.tokens, d_model=args.d_model, depth=args.depth or 4,
                  heads=args.heads, dropout=args.dropout)
    elif args.arch in ("lstm", "gru"):
        kw.update(width=width, depth=args.depth or 2, dropout=args.dropout)
    elif args.arch == "mlp":
        kw.update(width=width, dropout=args.dropout)
    else:  # cnn, unet
        kw["width"] = width
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="cnn")
    ap.add_argument("--width", type=int, default=None,
                    help="CNN width / RNN hidden size / U-Net base channels / MLP and BreathNet hidden "
                         f"(defaults {DEFAULT_WIDTH})")
    ap.add_argument("--tokens", choices=["frame", "patch"], default="frame", help="transformer tokens")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=None, help="transformer layers (4) / RNN layers (2)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout (all but CNN and U-Net)")
    ap.add_argument("--variant", default="whiten,coherent,track,hr",
                    help="breathnet2 mechanisms, comma-separated subset of whiten,coherent,track,hr,conv")
    ap.add_argument("--clip-grad", type=float, default=None,
                    help="max grad norm (default: 1.0 for transformer, LSTM, GRU, BreathNet; off otherwise)")
    ap.add_argument("--sigma", type=float, default=1.5, help="label-distribution width (bpm)")
    ap.add_argument("--balance", action="store_true", help="oversample rare RR ranges")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training windows (debug)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features)
    feat_cfg = json.loads((feat_dir / "config.json").read_text())
    feat_file = feat_dir / "features.npy"

    parts, split = load_split(feat_dir, args.folds, args.fold, args.seed)
    if args.max_train and len(parts["train"]) > args.max_train:
        parts["train"] = parts["train"].sample(args.max_train, random_state=args.seed)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print({k: f"{len(v)} windows / {v['patient_id'].nunique()} patients" for k, v in parts.items()})
    if any(len(v) == 0 for v in parts.values()):
        raise SystemExit("a split is empty: need at least 3 patients with usable windows")

    ldl = LabelDistribution(sigma=args.sigma).to(device)
    model_kw = model_kwargs(args, feat_cfg["shape"], ldl.n_bins, feat_cfg["spec"]["kind"])
    clip = args.clip_grad if args.clip_grad is not None else (1.0 if args.arch in CLIPPED else 0.0)
    model = build_model(**model_kw).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * max(
        1, int(np.ceil(len(parts["train"]) / args.batch))))

    train_dl = make_loader(feat_file, parts["train"], args.batch, args.workers, True, args.balance)
    val_dl = make_loader(feat_file, parts["val"], args.batch, args.workers, False)
    test_dl = make_loader(feat_file, parts["test"], args.batch, args.workers, False)

    ckpt = {"args": vars(args), "features": feat_cfg, "split": split,
            "ldl": {"sigma": args.sigma}, "model": model_kw}
    best, bad_epochs, history = np.inf, 0, []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = ldl.loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        val_pred = predict(model, ldl, val_dl, device)
        vm = rr_metrics(parts["val"]["ref_rr"], val_pred, parts["val"]["patient_id"])
        history.append({"epoch": epoch, "train_loss": tot / n, "val_mae": vm["mae"],
                        "val_patient_mae": vm["patient_macro_mae"], "sec": time.time() - t0})
        print(f"epoch {epoch:3d}  loss {tot / n:.4f}  val MAE {vm['mae']:.2f}  "
              f"patient-MAE {vm['patient_macro_mae']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        # model selection on patient-averaged MAE so one long recording cannot dominate
        if vm["patient_macro_mae"] < best:
            best, bad_epochs = vm["patient_macro_mae"], 0
            torch.save({**ckpt, "state_dict": model.state_dict(), "epoch": epoch}, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    test = parts["test"].copy()
    test["pred_rr"] = predict(model, ldl, test_dl, device)
    test[["window_row", "segment_id", "patient_id", "start_s", "ref_rr", "pred_rr"]].to_csv(
        out / "test_predictions.csv", index=False)
    metrics = {"test": rr_metrics(test["ref_rr"], test["pred_rr"], test["patient_id"]),
               "test_by_range": metrics_by_range(test["ref_rr"], test["pred_rr"]),
               "best_val_patient_mae": best}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
