"""Build the offline benchmark report (figures + report.html) from whatever runs exist.

With several folds, every estimator is scored on the pooled out-of-fold test
predictions: each patient is a test patient in exactly one fold, so the pooled
results cover every patient once. A comparison only uses the folds in which all
of its estimators have finished; anything missing is shown as pending.

Usage::

    python -m scripts.make_report --folds 0 1 2 3 4 --out reports/cv
    python -m scripts.make_report --folds 0 --out reports/fold0
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.metrics import rr_metrics  # noqa: E402
from data_prep.build_dataset import TARGET_FS as TARGET_FS_ALL  # noqa: E402

DATASET = Path("dataset/01NVa_monitor")
WINDOWS = DATASET / "windows_w60_s30.parquet"
RUNS = Path("runs")
SIGNALS = ("ppg", "ecg")
DL_KINDS = {"mod": "DL CNN · BW/AM/FM spectrogram", "raw": "DL CNN · raw-signal STFT",
            "mod_tf": "DL transformer · BW/AM/FM spectrogram", "raw_tf": "DL transformer · raw-signal STFT"}
# kind -> run-name suffix after "<signal>_"
DL_RUNS = {"mod": "mod_w60", "raw": "raw_w60", "mod_tf": "mod_w60_tf-frame", "raw_tf": "raw_w60_tf-frame"}
# architecture comparison. Per input: (input key, label, CNN baseline run suffix,
# [(run suffix, model label), ...]); every model is compared with the CNN on the same windows.
ARCH_GROUPS = [
    ("mod", "BW/AM/FM spectrogram", "mod_w60",
     [("mod_w60", "CNN"), ("mod_w60_tf-frame", "Transformer, frame tokens"),
      ("mod_w60_tf-patch", "Transformer, patch tokens"), ("mod_w60_lstm", "LSTM"), ("mod_w60_gru", "GRU"),
      ("mod_w60_unet", "U-Net"), ("mod_w60_mlp", "MLP on time-averaged spectrum"),
      ("modwave_w60_breathnet", "BreathNet (waveform input)"),
      ("modwavehr_w60_breathnet2-whiten", "BreathNet + explicit 1/f whitening"),
      ("modwavehr_w60_breathnet2-coherent", "BreathNet + coherent pathway sum"),
      ("modwavehr_w60_breathnet2-track", "BreathNet + HMM rhythm tracking"),
      ("modwavehr_w60_breathnet2-hr", "BreathNet + heart-rate prior"),
      ("modwavehr_w60_breathnet2-conv", "BreathNet + conv decision head"),
      ("modwavehr_w60_breathnet2", "BreathNet-2 (whitening + coherent + tracking + HR prior)"),
      ("modwavehr_w60_breathnet2-coh-hr", "BreathNet + coherent + HR prior"),
      ("modwavehr_w60_breathnet2-coh-hr-track", "BreathNet + coherent + HR prior + HMM tracking"),
      ("modwavehr_w60_breathnet2-coh-hr-conv", "BreathNet + coherent + HR prior + conv head"),
      ("modwavehr_w60_breathnet2-all-conv", "BreathNet-2c (whitening + coherent + HR prior + conv head)")]),
    ("raw", "raw-signal STFT", "raw_w60",
     [("raw_w60", "CNN"), ("raw_w60_tf-frame", "Transformer, frame tokens"), ("raw_w60_lstm", "LSTM"),
      ("raw_w60_gru", "GRU"), ("raw_w60_unet", "U-Net"), ("raw_w60_mlp", "MLP on time-averaged spectrum")]),
]
SIGNAL_LABELS = {"smart": "smart fusion", "median": "median fusion", "bwf": "filtered BW",
                 "pub-smart": "smart fusion, as published", "pub-median": "median fusion, as published"}
RANGES = [(0, 12), (12, 20), (20, 30), (30, 40), (40, 100)]
RATIOS = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 10)]

# Reference palette (dataviz skill): categorical slots 1-3, ink, chrome
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "font.family": "sans-serif", "lines.linewidth": 2,
})


# ------------------------------------------------------------------ loading
def _folds(folds):
    return [folds] if isinstance(folds, int) else list(folds)


def dl_folds(sig, kind, folds):
    """Folds (of ``folds``) in which this DL run has finished."""
    stem = DL_RUNS.get(kind, kind)
    return [k for k in _folds(folds) if (RUNS / f"{sig}_{stem}_f{k}" / "test_predictions.csv").exists()]


def load_dl(sig, kind, folds):
    """Pooled out-of-fold test predictions of one DL run over ``folds`` (an int
    or a list); ``None`` when no fold has finished. Adds a ``fold`` column."""
    stem = DL_RUNS.get(kind, kind)
    parts = [pd.read_csv(RUNS / f"{sig}_{stem}_f{k}" / "test_predictions.csv").assign(fold=k)
             for k in dl_folds(sig, kind, folds)]
    return pd.concat(parts, ignore_index=True) if parts else None


def trad_path(sig, fold, single):
    """Signal-processing benchmark of one fold: every test window (``_all``), or
    the original 15k-window fold-0 sample when building a single-fold report."""
    full = RUNS / f"traditional_{sig}_f{fold}_all" / "predictions.parquet"
    if full.exists():
        return full
    sample = RUNS / f"traditional_{sig}_f{fold}" / "predictions.parquet"
    return sample if single and sample.exists() else None


def load_traditional(sig, folds):
    folds = _folds(folds)
    parts = [pd.read_parquet(p).assign(fold=k) for k in folds if (p := trad_path(sig, k, len(folds) == 1))]
    return pd.concat(parts, ignore_index=True) if parts else None


def best_traditional(trad, fusion):
    """Method with the lowest MAE among those covering >= 50 % of windows."""
    best = None
    for col in [c for c in trad.columns if c.endswith(f"|{fusion}")]:
        m = rr_metrics(trad["ref_rr"], trad[col])
        if m.get("coverage", 0) >= 0.5 and (best is None or m["mae"] < best[1]["mae"]):
            best = (col, m)
    return best[0] if best else None


# ------------------------------------------------------------------ tables
def comparison_table(sig, folds, windows):
    """All estimators scored on the SAME windows: the signal-processing benchmark
    windows of the folds in which the CNN baseline has also finished."""
    folds = _folds(folds)
    usable = [k for k in folds if trad_path(sig, k, len(folds) == 1) and k in dl_folds(sig, "mod", folds)]
    trad = load_traditional(sig, usable) if usable else None
    if trad is None:
        return None, None
    df = trad.copy()
    df["hr"] = windows.loc[df["window_row"], f"{sig}_hr"].to_numpy()
    rows = []
    for kind, label in DL_KINDS.items():
        have = dl_folds(sig, kind, usable)
        if have != usable:  # compare only on identical windows
            rows.append({"estimator": label, "pending": True, "folds_done": len(dl_folds(sig, kind, folds))})
            continue
        dl = load_dl(sig, kind, usable)
        df = df.merge(dl[["window_row", "pred_rr"]].rename(columns={"pred_rr": f"dl_{kind}"}),
                      on="window_row", how="left")
        rows.append({"estimator": label, "column": f"dl_{kind}",
                     **rr_metrics(df["ref_rr"], df[f"dl_{kind}"], df["patient_id"])})
    from estimate_rr import METHODS
    for col in sorted(c for c in trad.columns if "|" in c):
        method, s = col.split("|")
        if s not in SIGNAL_LABELS:
            continue
        label = f"{METHODS[method][1]} · {SIGNAL_LABELS[s]}"
        rows.append({"estimator": label, "column": col,
                     **rr_metrics(df["ref_rr"], df[col], df["patient_id"])})
    return pd.DataFrame(rows), df


def stratified(df, col, by, edges):
    out = []
    for lo, hi in edges:
        m = (df[by] >= lo) & (df[by] < hi)
        if m.sum():
            out.append({"bin": f"{lo:g}–{hi:g}" if hi < 100 else f"≥{lo:g}", "n": int(m.sum()),
                        **rr_metrics(df.loc[m, "ref_rr"], df.loc[m, col])})
    return pd.DataFrame(out)


def _run_info(run):
    """Test metrics, epochs, seconds/epoch and parameter count of one DL run."""
    import torch
    m = json.loads((run / "metrics.json").read_text())
    hist = pd.read_csv(run / "history.csv")
    from dl.model import build_model
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    n_params = sum(p.numel() for p in build_model(**ckpt["model"]).parameters())   # trainable only, no buffers
    return {**m["test"], "val_patient_mae": m["best_val_patient_mae"], "epochs": len(hist),
            "sec_per_epoch": float(hist["sec"].mean()), "params": int(n_params),
            "arch": ckpt["model"].get("arch", "cnn"), "tokens": ckpt["model"].get("tokens")}


def _run_dir(stem, fold):
    return RUNS / f"{stem}_f{fold}"


def _stem_folds(stem, folds):
    return [k for k in folds if (_run_dir(stem, k) / "test_predictions.csv").exists()
            and (_run_dir(stem, k) / "metrics.json").exists()]


def _pooled_preds(stem, folds):
    return pd.concat([pd.read_csv(_run_dir(stem, k) / "test_predictions.csv").assign(fold=k) for k in folds],
                     ignore_index=True)


def arch_comparison(folds):
    """Every trained architecture, pooled over the folds in which both it and its
    input's CNN baseline have finished; compared with the CNN by a paired
    per-patient Wilcoxon test on those (identical) windows. Per-fold per-patient
    MAE gives the fold-to-fold spread; size and CPU cost come from the fold-0 run."""
    from scipy.stats import wilcoxon
    folds = _folds(folds)
    foot = RUNS / "footprint_f0.csv"
    foot = pd.read_csv(foot).set_index("run") if foot.exists() else None
    rows = []
    for sig in SIGNALS:
        for key, input_label, base_suffix, models in ARCH_GROUPS:
            base_stem = f"{sig}_{base_suffix}"
            base_folds = _stem_folds(base_stem, folds)
            if not base_folds:
                continue
            for suffix, label in models:
                stem = f"{sig}_{suffix}"
                row = {"signal": sig, "input": input_label, "model": label, "run": stem,
                       "is_baseline": suffix == base_suffix}
                fk = [k for k in _stem_folds(stem, folds) if k in base_folds]
                if not fk:
                    if "tf-patch" not in suffix or sig == "ppg":  # patch tokens were only tried on PPG
                        rows.append({**row, "pending": True, "n_folds": 0})
                    continue
                info0 = _run_info(_run_dir(stem, fk[0]))
                b = _pooled_preds(stem, fk)
                per_fold = [rr_metrics(g["ref_rr"], g["pred_rr"], g["patient_id"])["patient_macro_mae"]
                            for _, g in b.groupby("fold")]
                vals = [json.loads((_run_dir(stem, k) / "metrics.json").read_text())["best_val_patient_mae"] for k in fk]
                row.update(rr_metrics(b["ref_rr"], b["pred_rr"], b["patient_id"]),
                           params=info0["params"], arch=info0["arch"], n_folds=len(fk), folds=fk,
                           fold_patient_mae=per_fold, fold_sd=float(np.std(per_fold, ddof=1)) if len(fk) > 1 else float("nan"),
                           val_patient_mae=float(np.mean(vals)))
                f0 = f"{stem}_f0"
                if foot is not None and f0 in foot.index:   # trainable parameters (the checkpoint also holds buffers)
                    row.update(params=int(foot.loc[f0, "params"]), net_ms=float(foot.loc[f0, "net_ms"]),
                               feature_ms=float(foot.loc[f0, "feature_ms"]))
                if not row["is_baseline"]:
                    a = _pooled_preds(base_stem, fk)
                    m = a.merge(b[["window_row", "pred_rr"]], on="window_row", suffixes=("_cnn", "_x"))
                    err_c, err_x = m["pred_rr_cnn"] - m["ref_rr"], m["pred_rr_x"] - m["ref_rr"]
                    per = pd.DataFrame({"c": err_c.abs(), "x": err_x.abs(), "p": m["patient_id"]}).groupby("p").mean()
                    d = per["x"] - per["c"]
                    folds_better = sum(fx < fc for fx, fc in zip(
                        per_fold, [rr_metrics(g["ref_rr"], g["pred_rr"], g["patient_id"])["patient_macro_mae"]
                                   for _, g in a.groupby("fold")]))
                    if "breathnet2" in stem:    # the parent model is BreathNet, not the CNN
                        bn_stem = f"{sig}_modwave_w60_breathnet"
                        bk = [k for k in fk if k in _stem_folds(bn_stem, folds)]
                        if bk:
                            mb = _pooled_preds(bn_stem, bk).merge(b[["window_row", "pred_rr"]], on="window_row",
                                                                  suffixes=("_bn", "_x"))
                            pb = pd.DataFrame({"b": (mb["pred_rr_bn"] - mb["ref_rr"]).abs(),
                                               "x": (mb["pred_rr_x"] - mb["ref_rr"]).abs(), "p": mb["patient_id"]}).groupby("p").mean()
                            db = pb["x"] - pb["b"]
                            row.update(better_than_bn=int((db < 0).sum()), n_patients_bn=int(len(db)),
                                       mean_diff_bn=float(db.mean()), p_bn=float(wilcoxon(db).pvalue))
                    row.update(n_common=int(len(m)), n_patients_common=int(len(d)),
                               better_than_cnn=int((d < 0).sum()), median_diff=float(d.median()), mean_diff=float(d.mean()),
                               p=float(wilcoxon(d).pvalue), err_corr=float(np.corrcoef(err_c, err_x)[0, 1]),
                               folds_better_than_cnn=int(folds_better))
                rows.append(row)
    return rows


def patient_ci(ref, pred, groups, n_boot=2000, seed=0):
    """95 % bootstrap interval of the per-patient MAE, resampling patients."""
    e = pd.Series(np.abs(np.asarray(pred, float) - np.asarray(ref, float))).groupby(np.asarray(groups)).mean().dropna()
    if len(e) < 3:
        return float("nan"), float("nan")
    v = e.to_numpy()
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_fold_table(df, cols):
    """Per-patient MAE of each estimator in each fold (same windows), with mean and SD."""
    out = []
    for col, label in cols:
        vals = {int(k): rr_metrics(g["ref_rr"], g[col], g["patient_id"]).get("patient_macro_mae", np.nan)
                for k, g in df.groupby("fold")}
        v = np.array(list(vals.values()), float)
        out.append({"estimator": label, "folds": vals, "mean": float(np.nanmean(v)),
                    "sd": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")})
    return out


def load_trends(sig, folds):
    parts = [pd.read_parquet(p).assign(fold=k) for k in _folds(folds)
             if (p := RUNS / f"trends_f{k}" / f"trend_{sig}.parquet").exists()]
    return pd.concat(parts, ignore_index=True) if parts else None


def breathnet_cv(folds):
    """BreathNet ablations and learned parameters in every fold where they were exported."""
    abl, learned = [], []
    for k in _folds(folds):
        for sig in SIGNALS:
            a = RUNS / f"breathnet_f{k}" / f"{sig}_ablation.csv"
            z = RUNS / f"breathnet_f{k}" / f"{sig}.npz"
            if a.exists():
                abl.append(pd.read_csv(a).assign(fold=k, signal=sig.upper()))
            if z.exists():
                d = np.load(z)
                learned.append({"fold": k, "signal": sig.upper(), "median_sigma_s": float(np.median(d["sigma"][6:50])),
                                "inverted_pathway": ["BW", "AM", "FM"][int(np.argmin(d["refine_corr"]))],
                                "inversion_r": float(np.min(d["refine_corr"])),
                                "half_rate_weights": d["harm"][:, 2].round(2).tolist(),
                                "strongest_harmonic_pathway": ["BW", "AM", "FM"][int(np.argmax(d["harm"][:, :2].sum(1)))]})
    return (pd.concat(abl, ignore_index=True) if abl else None), learned


def fig_breathnet_cv(abl, out):
    """Extra per-patient MAE when each component is switched off: one dot per fold, bar = mean."""
    variants = [v for v in abl["variant"].unique() if v != "full model"]
    sigs = list(dict.fromkeys(abl["signal"]))
    fig, axes = plt.subplots(1, len(sigs), figsize=(5.8 * len(sigs), 0.42 * len(variants) + 1.4), sharey=True, squeeze=False)
    for ax, sig in zip(axes[0], sigs):
        t = abl[abl["signal"] == sig]
        means = t.groupby("variant")["delta_patient_mae"].mean().reindex(variants)
        y = np.arange(len(variants))
        ax.barh(y, means.clip(lower=0.01), color=BLUE, alpha=0.35, height=0.6)
        for i, v in enumerate(variants):
            pts = t.loc[t["variant"] == v, "delta_patient_mae"].clip(lower=0.01)
            ax.plot(pts, np.full(len(pts), i), "o", color=BLUE, ms=4)
            ax.text(means[v] * 1.15 if means[v] > 0 else 0.012, i, f"{means[v]:+.2f}", va="center", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_yticks(y, variants)
        ax.invert_yaxis()
        ax.set(title=f"{sig}: extra per-patient MAE when switched off", xlabel="bpm (log scale); dots = folds, bars = mean")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = out / "fig10_breathnet_cv.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_arch_size(rows, out):
    """Per-patient MAE vs parameter count, one panel per signal."""
    done = [r for r in rows if not r.get("pending")]
    if not done:
        return None
    sigs = [s for s in SIGNALS if any(r["signal"] == s for r in done)]
    fig, axes = plt.subplots(1, len(sigs), figsize=(5.6 * len(sigs), 3.9), sharey=True, squeeze=False)
    colors = {"BW/AM/FM spectrogram": BLUE, "raw-signal STFT": AQUA}
    for ax, sig in zip(axes[0], sigs):
        for r in (r for r in done if r["signal"] == sig):
            c = ORANGE if "BreathNet" in r["model"] else colors[r["input"]]
            if r.get("n_folds", 1) > 1 and np.isfinite(r.get("fold_sd", np.nan)):
                ax.errorbar(r["params"], r["patient_macro_mae"], yerr=r["fold_sd"], color=c, alpha=0.5, lw=1, capsize=2, zorder=2)
            ax.scatter(r["params"], r["patient_macro_mae"], s=40, color=c, zorder=3,
                       marker="D" if r["is_baseline"] else "o")
            short = r["model"].split(" (")[0].replace("Transformer, ", "TF ").replace(" on time-averaged spectrum", "")
            ax.annotate(short, (r["params"], r["patient_macro_mae"]), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set(title=f"{sig.upper()}: accuracy vs size", xlabel="parameters (log scale)")
    axes[0][0].set_ylabel("per-patient MAE (bpm)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="BW/AM/FM spectrogram input"),
               Line2D([], [], marker="o", ls="", color=AQUA, label="raw-signal STFT input"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="BreathNet (waveform input)"),
               Line2D([], [], marker="D", ls="", color=INK2, label="CNN baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    path = out / "fig9_arch_size.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ figures
def fig_dataset(windows, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    ok = windows[windows["ref_ok"]]
    axes[0].hist(ok["ref_rr"], bins=np.arange(4, 61, 1), color=BLUE, rwidth=0.85)
    axes[0].set(title="Reference RR (impedance)", xlabel="breaths/min", ylabel="windows")
    u = windows[windows["ref_ok"] & windows["ppg_ok"]]
    axes[1].hist(u["ppg_hr"], bins=np.arange(30, 200, 3), color=BLUE, rwidth=0.85)
    axes[1].set(title="Heart rate (PPG windows)", xlabel="beats/min")
    ratio = u["ref_rr"] / u["ppg_hr"]
    axes[2].hist(ratio, bins=np.arange(0, 0.8, 0.02), color=BLUE, rwidth=0.85)
    axes[2].axvline(0.5, color=INK2, lw=1)
    axes[2].text(0.51, axes[2].get_ylim()[1] * 0.9, "HR/2 limit", color=INK2, fontsize=8)
    axes[2].set(title="RR / HR ratio", xlabel="ratio")
    fig.tight_layout()
    fig.savefig(out / "fig1_dataset.png", dpi=150)
    plt.close(fig)


def fig_method_bars(t, sig, out):
    """MAE of every estimator on the same windows; DL in blue, best traditional in orange."""
    t = t[~t["estimator"].str.endswith("as published")]  # before/after has its own figure
    t = t.dropna(subset=["mae"]).sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * len(t) + 1.2))
    best_trad = t[~t["estimator"].str.startswith("DL")]["mae"].idxmin()
    colors = [BLUE if e.startswith("DL") else ORANGE if i == best_trad else GRAY
              for i, e in zip(t.index, t["estimator"])]
    ax.barh(t["estimator"], t["mae"], color=colors, height=0.6)
    for y, (mae, cov) in enumerate(zip(t["mae"], t["coverage"])):
        ax.text(mae + 0.15, y, f"{mae:.1f}" + ("" if cov > 0.995 else f"  ({cov:.0%} cov.)"),
                va="center", fontsize=7.5, color=INK2)
    ax.set(title=f"{sig.upper()}: mean absolute error (bpm), same windows", xlabel="MAE (bpm)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5, colors=INK2)
    ax.set_xlim(0, t["mae"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(out / f"fig2_methods_{sig}.png", dpi=150, bbox_inches="tight")  # long labels
    plt.close(fig)


def fig_bland_altman(df, sig, dl_col, trad_col, trad_label, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, col, label, color in ((axes[0], dl_col, DL_KINDS[dl_col[3:]], BLUE),
                                  (axes[1], trad_col, trad_label, ORANGE)):
        m = df[col].notna()
        mean = (df.loc[m, col] + df.loc[m, "ref_rr"]) / 2
        diff = df.loc[m, col] - df.loc[m, "ref_rr"]
        ax.scatter(mean, diff, s=3, alpha=0.15, color=color, linewidths=0)
        b, sd = diff.mean(), diff.std()
        for v, ls in ((b, "-"), (b - 1.96 * sd, "--"), (b + 1.96 * sd, "--")):
            ax.axhline(v, color=INK2, lw=1, ls=ls)
        ax.text(0.99, 0.97, f"bias {b:+.2f}\nLoA [{b - 1.96 * sd:.1f}, {b + 1.96 * sd:.1f}]\n"
                f"coverage {m.mean():.0%}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK2)
        ax.set(title=label, xlabel="mean of estimate and reference (bpm)")
        ax.set_ylim(-20, 20)
    axes[0].set_ylabel("estimate − reference (bpm)")
    fig.suptitle(f"{sig.upper()} Bland–Altman, held-out patients", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"fig3_bland_altman_{sig}.png", dpi=150)
    plt.close(fig)


def fig_stratified(df, sig, cols, out):
    """MAE by reference-RR range and by RR/HR ratio; one panel each."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    df = df.assign(ratio=df["ref_rr"] / df["hr"])
    for ax, by, edges, title in ((axes[0], "ref_rr", RANGES, "MAE by reference RR (bpm)"),
                                 (axes[1], "ratio", RATIOS, "MAE by RR / HR ratio")):
        width = 0.8 / len(cols)
        for i, (col, label, color) in enumerate(cols):
            s = stratified(df, col, by, edges)
            x = np.arange(len(s)) + (i - (len(cols) - 1) / 2) * width
            ax.bar(x, s["mae"], width=width * 0.9, color=color, label=label)
        ax.set_xticks(np.arange(len(s)), [f"{b}\n(n={n})" for b, n in zip(s["bin"], s["n"])])
        ax.set(title=title, ylabel="MAE (bpm)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper center")
    fig.suptitle(f"{sig.upper()}: where the errors are", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"fig4_stratified_{sig}.png", dpi=150)
    plt.close(fig)


def fig_example(windows, df, sig, out):
    """One held-out window: reference impedance, PPG/ECG, derived signals, DL input."""
    from data_prep.build_dataset import TARGET_FS
    from dl.features import SpecConfig, modulation_spectrogram
    from estimate_rr.resp_signals import RESP_FS, extract_resp_signals, respiratory_waveform
    cand = df[(df["ref_rr"].between(18, 26)) & df["dl_mod"].notna()]
    if cand.empty:
        return
    r = windows.loc[cand.sample(1, random_state=3)["window_row"].iloc[0]]
    data = np.load(DATASET / "signals" / f"{r['segment_id']}.npz")
    cut = {k: data[k][int(r["start_s"] * fs):int((r["start_s"] + 60) * fs)].astype(float)
           for k, fs in TARGET_FS.items()}
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 1])
    ax0 = fig.add_subplot(gs[0, :])
    ref = respiratory_waveform(cut["resp"], TARGET_FS["resp"])
    ax0.plot(np.arange(len(ref)) / RESP_FS, ref, color=INK2)
    ax0.set(title=f"Reference: thoracic impedance, {r['ref_rr']:.1f} breaths/min", yticks=[])
    ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
    x = cut[sig]
    ax1.plot(np.arange(len(x)) / TARGET_FS[sig], x, color=INK2, lw=0.6)
    ax1.set(title=f"Input: monitor {sig.upper()}", yticks=[])
    ax2 = fig.add_subplot(gs[2, :], sharex=ax0)
    rs = extract_resp_signals(x, TARGET_FS[sig], sig)
    for name, color in (("bw", BLUE), ("am", ORANGE), ("fm", AQUA)):
        if name in rs.signals:
            y = rs.signals[name]
            ax2.plot(np.arange(len(y)) / RESP_FS, y / y.std(), color=color, label=name.upper(), lw=1.5)
    ax2.legend(ncol=3, loc="upper right")
    ax2.set(title="Derived respiratory signals", yticks=[], xlabel="time (s)")
    spec = modulation_spectrogram(x, TARGET_FS[sig], sig, SpecConfig())
    cfg = SpecConfig()
    for i, name in enumerate(("BW", "AM", "FM")):
        ax = fig.add_subplot(gs[3, i])
        if spec is not None:
            ax.imshow(spec[i], aspect="auto", origin="lower", cmap="Blues",
                      extent=[0, spec.shape[2], cfg.band_low_hz * 60, cfg.band_high_hz * 60])
            ax.axhline(r["ref_rr"], color=ORANGE, lw=1)
        ax.set(title=f"DL input: {name} spectrogram", xlabel="frame (1 s hop)",
               ylabel="breaths/min" if i == 0 else None)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out / f"fig5_example_{sig}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ fix, cases, trends
METHOD_ORDER = ["CtO", "CtA", "PKS", "ZeX", "FTS", "WCH", "ARS", "ARM", "ARP"]


def fix_rows(df):
    """Published vs corrected rule per method (smart fusion) on the benchmark windows."""
    from estimate_rr import METHODS
    rows = []
    for col in sorted(c for c in df.columns if c.endswith("|pub-smart")):
        method = col.split("|")[0]
        pub = rr_metrics(df["ref_rr"], df[col], df["patient_id"])
        cor = rr_metrics(df["ref_rr"], df[f"{method}|smart"], df["patient_id"])
        rows.append({"method": METHODS[method][1], "pub_mae": pub["mae"], "pub_bias": pub["bias"],
                     "pub_cov": pub["coverage"], "cor_mae": cor["mae"], "cor_bias": cor["bias"],
                     "cor_cov": cor["coverage"], "cor_w2": cor["within_2bpm"]})
    return sorted(rows, key=lambda r: METHOD_ORDER.index(r["method"]))


def fig_fix(rows, sig, out):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["pub_mae"] for r in rows], width=0.38, color=GRAY, label="as published")
    ax.bar(x + 0.2, [r["cor_mae"] for r in rows], width=0.38, color=BLUE, label="corrected")
    for xi, r in zip(x, rows):
        ax.text(xi + 0.2, r["cor_mae"] + 0.2, f"{r['cor_mae']:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [r["method"] for r in rows])
    ax.set(title=f"{sig.upper()}: MAE as published vs corrected (smart fusion, held-out patients)", ylabel="MAE (bpm)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"fig6_fix_{sig}.png", dpi=150)
    plt.close(fig)


def fig_case(det, case, i, out):
    """One sample window: signals with the detections each method uses, and PPG spectra."""
    from estimate_rr.resp_signals import RESP_FS
    raw = det["raw"]
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1.4], hspace=0.8, top=0.93)
    t = {k: np.arange(len(v)) / TARGET_FS_ALL[k] for k, v in raw.items()}
    ax = [fig.add_subplot(gs[r, :]) for r in range(5)]
    for a in ax[1:]:
        a.sharex(ax[0])
    ax[0].plot(t["ecg"], raw["ecg"], color=INK2, lw=0.6)
    ax[0].plot(t["ecg"][det["rpeaks"]], raw["ecg"][det["rpeaks"]], "^", color=ORANGE, ms=4)
    ax[0].set_title("ECG · R peaks (beat times, R amplitude)", fontsize=9)
    ax[1].plot(t["ppg"], raw["ppg"], color=INK2, lw=0.6)
    ax[1].plot(t["ppg"][det["ppg_peaks"]], raw["ppg"][det["ppg_peaks"]], "^", color=ORANGE, ms=4)
    ax[1].plot(t["ppg"][det["ppg_onsets"]], raw["ppg"][det["ppg_onsets"]], "v", color=BLUE, ms=4)
    ax[1].set_title("PPG · pulse peaks ▲ and onsets ▼", fontsize=9)
    tr = np.arange(len(det["resp"])) / RESP_FS
    ax[2].plot(tr, det["resp"], color=INK2, lw=1.5)
    bi = np.clip(np.round(det["ref_breaths"] * RESP_FS).astype(int), 0, len(tr) - 1)
    ax[2].plot(det["ref_breaths"], det["resp"][bi], "o", color=ORANGE, ms=5)
    ax[2].set_title(f"Impedance · reference breaths → {det['ref_rr']:.1f} bpm", fontsize=9)
    colors = {"bw": BLUE, "am": ORANGE, "fm": AQUA}
    for a, sig in ((ax[3], "ppg"), (ax[4], "ecg")):
        for m, y in det[f"{sig}_derived"].items():
            yn = y / (y.std() + 1e-12)
            a.plot(np.arange(len(y)) / RESP_FS, yn, color=colors[m], lw=1.1, label=m.upper())
            bb = det[f"{sig}_cto_breaths"][m]
            a.plot(bb, yn[np.clip(np.round(bb * RESP_FS).astype(int), 0, len(y) - 1)], "o", color=colors[m],
                   ms=4, mec=SURFACE)
        a.set_title(f"{sig.upper()}-derived respiratory signals · breaths CtO accepts", fontsize=9)
    ax[3].legend(ncol=3, fontsize=7, loc="upper right")
    for a in ax:
        a.set_yticks([])
        a.grid(False)
    ax[4].set_xlabel("time in window (s)")
    for j, m in enumerate(("bw", "am", "fm")):
        a = fig.add_subplot(gs[5, j])
        sp = det["ppg_spectra"].get(m)
        if sp is None:
            continue
        a.semilogy(sp["f_bpm"], sp["psd"], color=colors[m], lw=1.5)
        a.semilogy(sp["f_bpm"], sp["background"], color=INK2, lw=0.8, ls="--")
        a.axvspan(4, 8, color=GRAY, alpha=0.3, lw=0)
        a.axvline(det["ref_rr"], color=INK2, lw=1.2)
        a.axvline(sp["pick_published"], color=ORANGE, lw=1.5, ls=":")
        a.axvline(sp["pick_corrected"], color=BLUE, lw=1.5)
        a.set(title=f"PPG {m.upper()} spectrum", xlabel="breaths/min", yticks=[])
        a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a.grid(False)
    fig.suptitle(f"{case['case']}: {case['patient_id']} at {case['start_s']:.0f} s · reference "
                 f"{det['ref_rr']:.1f}, DL {case['dl']:.1f} bpm · HR {case['hr']:.0f}",
                 x=0.01, y=0.985, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(out / f"fig7_case{i}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def case_rates(det):
    t = det["rates"].pivot_table(index="method", columns=["signal", "rule"], values="smart fusion")
    t = t.reindex(METHOD_ORDER)
    return [{"method": m, **{f"{s} {r}": (None if not np.isfinite(t.loc[m].get((s, r), np.nan)) else
                                          float(t.loc[m][(s, r)]))
                             for s in ("PPG", "ECG") for r in ("published", "corrected")}} for m in METHOD_ORDER]


TREND_SERIES = [("dl", "DL (CNN)", BLUE), ("count_orig|smart", "CtO smart fusion", ORANGE),
                ("ar_median|smart", "ARM corrected, smart fusion", AQUA)]


def fig_trends(trend, out, n=3, sig="ppg"):
    from scripts.case_study import rolling_trend, trend_concordance
    trend = trend.sort_values(["patient_id", "t_epoch"]).copy()
    trend["hours"] = (trend["t_epoch"] - trend.groupby("patient_id")["t_epoch"].transform("min")) / 3600
    top = trend.groupby("patient_id").size().sort_values(ascending=False).index[:n]
    fig, axes = plt.subplots(len(top), 1, figsize=(11, 3.1 * len(top)))
    for ax, pid in zip(np.atleast_1d(axes), top):
        g = trend[trend["patient_id"] == pid]
        ok = g[g["ref_ok"]]
        ax.plot(g.loc[~g["ref_ok"], "hours"], g.loc[~g["ref_ok"], "ref_rr"], ".", color=GRAY, ms=2)
        ax.plot(ok["hours"], ok["ref_rr"], ".", color=INK2, ms=2.5)
        for col, label, color in TREND_SERIES:
            ax.plot(*rolling_trend(g, col), color=color, lw=1.6, label=label)
        ax.plot(*rolling_trend(ok, "ref_rr"), color=INK, lw=1.2, ls=":", label="reference")
        m = rr_metrics(ok["ref_rr"], ok["dl"])
        ax.set(title=f"{pid}: reference usable in {g['ref_ok'].mean():.0%} of {len(g)} windows; DL MAE {m['mae']:.2f}",
               ylabel="breaths/min", ylim=(0, 60))
    np.atleast_1d(axes)[0].legend(ncol=4, loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("hours since start of recording (lines = 5-min rolling medians; "
                                        "grey dots = reference failed quality)")
    fig.tight_layout()
    fig.savefig(out / f"fig8_trends_{sig}.png", dpi=140)
    plt.close(fig)
    ok = trend[trend["ref_ok"]]
    return [{"estimator": label, **trend_concordance(ok, col), **{k: v for k, v in rr_metrics(ok["ref_rr"], ok[col]).items()
                                                                  if k in ("mae", "coverage")}}
            for col, label, _ in TREND_SERIES], list(top)


# ------------------------------------------------------------------ wearables
WEARABLE = Path("dataset/01NVa_wearable")
WEARABLE_RUNS = RUNS / "wearable"
DEVICE = {"ppg": "SmartCare wrist PPG", "ecg": "Faros/ePatch chest ECG"}


def fig_wearable_paired(summary, out):
    """Per-patient MAE on the same windows: monitor input (x) vs wearable input (y),
    for the CNN and the best signal-processing method, one panel per signal."""
    sigs = [s for s in ("ppg", "ecg") if (WEARABLE_RUNS / f"predictions_{s}.parquet").exists()]
    if not sigs:
        return None
    fig, axes = plt.subplots(1, len(sigs), figsize=(5.4 * len(sigs), 4.8), squeeze=False)
    for ax, sig in zip(axes[0], sigs):
        w = pd.read_parquet(WEARABLE_RUNS / f"predictions_{sig}.parquet")
        m = pd.read_parquet(WEARABLE_RUNS / f"monitor_predictions_paired_{sig}.parquet")
        lim = 0
        for label, color, short in (("DL CNN", BLUE, "DL CNN"), ("ARM · smart fusion", ORANGE, "ARM smart fusion")):
            if label not in w or label not in m:
                continue
            j = m[["patient_id", "ref_rr"]].assign(mon=m[label], wear=w.loc[m.index, label]).dropna()
            per = j.assign(em=(j["mon"] - j["ref_rr"]).abs(), ew=(j["wear"] - j["ref_rr"]).abs()).groupby("patient_id")[["em", "ew"]].mean()
            ax.scatter(per["em"], per["ew"], s=14, color=color, alpha=0.7, label=f"{short} ({len(per)} patients)",
                       edgecolor="white", linewidth=0.4)
            lim = max(lim, float(np.nanpercentile(per.to_numpy(), 98)))
        lim = max(lim * 1.1, 2)
        ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="per-patient MAE, monitor input (bpm)",
               ylabel="per-patient MAE, wearable input (bpm)", title=f"{sig.upper()}: monitor vs {DEVICE[sig]}")
        ax.legend(fontsize=8, loc="upper left")
        ax.text(0.97, 0.04, "above the line = worse on the wearable", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    path = out / "fig11_wearable_paired.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_wearable_alignment(out):
    """Ten seconds of the monitor's ECG / Pleth and the wearable's at the aligned time,
    and how the device clocks relate to the monitor's."""
    segs = pd.read_csv(WEARABLE / "segments.csv") if (WEARABLE / "segments.csv").exists() else None
    al = pd.read_csv(WEARABLE / "alignment.csv")
    if segs is None or segs.empty:
        return None
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 2.2, 1.6])
    for row, kind in enumerate(("ecg", "ppg")):
        sk = segs[segs["kind"] == kind]
        wp = WEARABLE / f"windows_{kind}_w60_s30.parquet"
        if sk.empty or not wp.exists():
            continue
        w = pd.read_parquet(wp)
        w = w[w[f"{kind}_ok"] & w["monitor_ok"] & w["ref_ok"]]
        if w.empty:
            continue
        r = w.sample(1, random_state=3).iloc[0]
        d = np.load(WEARABLE / "signals" / f"{r['segment_id']}.npz")
        fs = 250.0 if kind == "ecg" else 100.0
        a = int(r["start_s"] * fs)
        x_w, x_m = d[kind][a:a + int(10 * fs)].astype(float), d[f"monitor_{kind}"][a:a + int(10 * fs)].astype(float)
        t = np.arange(len(x_w)) / fs
        z = lambda v: (v - np.nanmedian(v)) / (np.nanpercentile(v, 99) - np.nanpercentile(v, 1) + 1e-9)
        ax = fig.add_subplot(gs[row, :2])
        ax.plot(t, z(x_m) + 0.7, color=INK2, lw=0.9, label="monitor")
        ax.plot(t, z(x_w) - 0.7, color=BLUE if kind == "ecg" else ORANGE, lw=0.9, label=DEVICE[kind])
        ax.set(yticks=[], xlabel="seconds", title=f"{'ECG' if kind == 'ecg' else 'PPG'}: the same 10 s on both devices "
               f"({r['patient_id']}, after alignment)")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax = fig.add_subplot(gs[:, 2])
    acc = al[al["status"] == "aligned"].drop_duplicates("path")
    for kind, color in (("ppg", ORANGE), ("ecg", BLUE)):
        v = acc.loc[acc["kind"] == kind, "offset_minus_nominal_h"].dropna()
        if len(v):
            ax.hist(v, bins=np.arange(-12, 36, 0.5), color=color, alpha=0.75, label=f"{DEVICE[kind]} ({len(v)} files)")
    ax.set(xlabel="monitor clock minus device clock (h)", ylabel="files", title="Device clocks vs the monitor")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    path = out / "fig12_wearable_alignment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def wearable_ctx(out):
    p = WEARABLE_RUNS / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    s["figs"] = {"paired": fig_wearable_paired(s, out), "alignment": fig_wearable_alignment(out)}
    return s


# ------------------------------------------------------------------ report
def dataset_summary(windows):
    ok = windows["ref_ok"]
    u = windows[ok & windows["ppg_ok"]]
    ratio = u["ref_rr"] / u["ppg_hr"]
    segs = pd.read_csv(DATASET / "segments.csv")
    return {"n_windows": len(windows), "n_patients": int(windows["patient_id"].nunique()),
            "hours": float(segs["duration_s"].sum() / 3600), "ref_ok": float(ok.mean()),
            "ppg_usable": float((ok & windows["ppg_ok"]).mean()),
            "ecg_usable": float((ok & windows["ecg_ok"]).mean()),
            "rr_median": float(windows.loc[ok, "ref_rr"].median()),
            "rr_p5": float(windows.loc[ok, "ref_rr"].quantile(.05)),
            "rr_p95": float(windows.loc[ok, "ref_rr"].quantile(.95)),
            "ratio_median": float(ratio.median()), "ratio_gt04": float((ratio > 0.4).mean()),
            **locked_rate_patients(windows[ok])}


def locked_rate_patients(w, tol=0.15, min_frac=0.3, min_windows=240):
    """Patients whose reference stays within +-tol bpm of one value for at least min_frac of their
    usable windows (and >= 2 h of them, 30-s stride), as a ventilator set rate would."""
    out, fracs = [], []
    for pid, g in w.groupby("patient_id"):
        mode = float(g["ref_rr"].round(1).mode().iloc[0])
        frac = float((np.abs(g["ref_rr"] - mode) <= tol).mean())
        fracs.append(frac)
        if frac >= min_frac and len(g) >= min_windows:
            out.append({"patient_id": pid, "rate": mode, "frac": frac, "hours": len(g) * 30 / 3600})
    return {"locked": sorted(out, key=lambda r: -r["frac"]), "locked_median_frac": float(np.median(fracs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="folds to pool (default: 0)")
    ap.add_argument("--fold", type=int, default=None, help="single fold (same as --folds N)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    folds = args.folds or [args.fold if args.fold is not None else 0]
    cv = len(folds) > 1
    out = Path(args.out or ("reports/cv" if cv else f"reports/fold{folds[0]}"))
    out.mkdir(parents=True, exist_ok=True)

    windows = pd.read_parquet(WINDOWS)
    fig_dataset(windows, out)
    ctx = {"folds": folds, "cv": cv, "fold": folds[0], "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           "dataset": dataset_summary(windows), "figs": {"dataset": out / "fig1_dataset.png"},
           "signals": {}, "notebooks": []}
    for sig in SIGNALS:
        t, df = comparison_table(sig, folds, windows)
        if t is None or "dl_mod" not in df:
            continue  # this signal's DL model or benchmark has not finished
        trad_col = best_traditional(df, "smart") or best_traditional(df, "median")
        trad_label = t.loc[t["column"] == trad_col, "estimator"].iloc[0]
        cols = [("dl_mod", DL_KINDS["mod"], BLUE), (trad_col, trad_label, ORANGE)]
        if "dl_raw" in df:
            cols.insert(1, ("dl_raw", DL_KINDS["raw"], AQUA))
        fig_method_bars(t, sig, out)
        fig_bland_altman(df, sig, "dl_mod", trad_col, trad_label, out)
        fig_stratified(df, sig, cols, out)
        fig_example(windows, df, sig, out)
        dfr = df.assign(ratio=df["ref_rr"] / df["hr"])
        for r in t.to_dict("records"):  # patient-level 95 % intervals for every finished estimator
            if isinstance(r.get("column"), str) and r["column"] in df:
                lo, hi = patient_ci(df["ref_rr"], df[r["column"]], df["patient_id"])
                t.loc[t["column"] == r["column"], ["ci_lo", "ci_hi"]] = lo, hi
        rows = t.drop(columns=["column"], errors="ignore").to_dict("records")
        per_patient = (dfr.assign(e_dl=(dfr["dl_mod"] - dfr["ref_rr"]).abs(), e_tr=(dfr[trad_col] - dfr["ref_rr"]).abs())
                       .groupby("patient_id")[["e_dl", "e_tr"]].mean())
        dl_all = load_dl(sig, "mod", folds)  # every test window of every finished fold, not only the benchmark ones
        dl_full = rr_metrics(dl_all["ref_rr"], dl_all["pred_rr"], dl_all["patient_id"])
        dl_full["ci_lo"], dl_full["ci_hi"] = patient_ci(dl_all["ref_rr"], dl_all["pred_rr"], dl_all["patient_id"])
        dl_full["n_folds"] = int(dl_all["fold"].nunique())
        fold_cols = [("dl_mod", DL_KINDS["mod"])] + ([("dl_raw", DL_KINDS["raw"])] if "dl_raw" in df else [])
        fold_cols.append((trad_col, trad_label))
        published = trad_col.replace("|smart", "|pub-smart")
        if published in df:  # the same method as published, for scale
            fold_cols.append((published, trad_label + ", as published"))
        ctx["signals"][sig] = {
            "rows": rows, "dl": True, "n_windows": len(df), "n_patients": int(df["patient_id"].nunique()),
            "folds_used": sorted(int(k) for k in df["fold"].unique()),
            "dl_full": dl_full,
            "best": next(r for r in rows if r["estimator"] == trad_label),
            "dl_better_patients": float((per_patient["e_dl"] < per_patient["e_tr"]).mean()),
            "per_fold": per_fold_table(df, fold_cols) if cv else None,
            "range_dl": {r["bin"]: r for r in stratified(dfr, "dl_mod", "ref_rr", RANGES).to_dict("records")},
            "by_range": {c: stratified(dfr, c, "ref_rr", RANGES).to_dict("records") for c, _, _ in cols},
            "by_ratio": {c: stratified(dfr, c, "ratio", RATIOS).to_dict("records") for c, _, _ in cols},
            "figs": {"methods": out / f"fig2_methods_{sig}.png", "ba": out / f"fig3_bland_altman_{sig}.png",
                     "strat": out / f"fig4_stratified_{sig}.png", "example": out / f"fig5_example_{sig}.png",
                     "fix": out / f"fig6_fix_{sig}.png"},
        }
        fix = fix_rows(df)
        if fix:
            fig_fix(fix, sig, out)
            ctx["signals"][sig]["fix"] = fix
            if cv and 0 in df["fold"].unique():  # fold 0 alone: its test patients played no part in choosing the fix
                ctx["signals"][sig]["fix_fold0"] = fix_rows(df[df["fold"] == 0])

    # sample windows: chosen from fold 0's PPG test predictions (the captions and caveats describe those
    # windows, so they must not change as other folds finish); all signals shown
    from scripts.case_study import pick_cases, window_detail
    test = load_dl("ppg", "mod", 0)
    ctx["cases"] = []
    if test is not None:
        test = test.rename(columns={"pred_rr": "dl_mod"})
        test["hr"] = windows.loc[test["window_row"], "ppg_hr"].to_numpy()
        for i, case in enumerate(pick_cases(test, windows, trad_col="none").to_dict("records")):
            det = window_detail(case["segment_id"], case["start_s"], dataset=DATASET)
            fig_case(det, case, i, out)
            ctx["cases"].append({**case, "fig": out / f"fig7_case{i}.png", "rates": case_rates(det),
                                 "ratio": case["ref_rr"] / case["hr"]})

    ctx["arch"] = arch_comparison(folds)
    ctx["figs"]["arch_size"] = fig_arch_size(ctx["arch"], out)

    ctx["trend"] = {}
    for sig in SIGNALS:
        tr = load_trends(sig, folds)
        if tr is not None:
            conc, patients = fig_trends(tr, out, sig=sig)
            ctx["trend"][sig] = {"fig": out / f"fig8_trends_{sig}.png", "concordance": conc, "patients": patients,
                                 "n_folds": int(tr["fold"].nunique()), "n_patients": int(tr["patient_id"].nunique())}

    abl, learned = breathnet_cv(folds)
    if abl is not None:
        ctx["breathnet_cv"] = {"ablation": abl.to_dict("records"), "learned": learned,
                               "n_folds": int(abl["fold"].nunique()), "fig": fig_breathnet_cv(abl, out)}

    if cv:   # the wearable evaluation uses every fold's model (each patient scored by the fold that tested it)
        ctx["wearable"] = wearable_ctx(out)

    for nb in sorted(out.glob("[0-9][0-9]_*.html")):
        ctx["notebooks"].append((f"Notebook: {nb.stem[3:].replace('_', ' ')}", nb.name))

    from scripts.report_html import render
    (out / "report.html").write_text(render(ctx), encoding="utf-8")
    summary = {k: v for k, v in ctx.items() if k != "figs"}
    for s in summary["signals"].values():
        s.pop("figs", None)
    summary["cases"] = [{k: v for k, v in c.items() if k != "fig"} for c in ctx["cases"]]
    summary["trend"] = {sig: {k: v for k, v in t.items() if k != "fig"} for sig, t in ctx["trend"].items()}
    if ctx.get("wearable"):
        summary["wearable"] = {k: v for k, v in ctx["wearable"].items() if k != "figs"}
    if "breathnet_cv" in summary:
        summary["breathnet_cv"] = {k: v for k, v in ctx["breathnet_cv"].items() if k != "fig"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"report written to {out}/report.html")


if __name__ == "__main__":
    main()
