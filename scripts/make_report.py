"""Build the offline benchmark report (figures + report.html) from whatever runs exist.

Re-run any time; missing runs are listed as pending.

Usage::

    python -m scripts.make_report --fold 0 --out reports/fold0
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
DL_KINDS = {"mod": "DL · BW/AM/FM spectrogram", "raw": "DL · raw-signal STFT"}
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
def load_dl(sig, kind, fold):
    run = RUNS / f"{sig}_{kind}_w60_f{fold}"
    if not (run / "test_predictions.csv").exists():
        return None
    return pd.read_csv(run / "test_predictions.csv")


def load_traditional(sig, fold):
    p = RUNS / f"traditional_{sig}_f{fold}" / "predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


def best_traditional(trad, fusion):
    """Method with the lowest MAE among those covering >= 50 % of windows."""
    best = None
    for col in [c for c in trad.columns if c.endswith(f"|{fusion}")]:
        m = rr_metrics(trad["ref_rr"], trad[col])
        if m.get("coverage", 0) >= 0.5 and (best is None or m["mae"] < best[1]["mae"]):
            best = (col, m)
    return best[0] if best else None


# ------------------------------------------------------------------ tables
def comparison_table(sig, fold, windows):
    """All estimators scored on the SAME windows (the traditional-benchmark sample)."""
    trad = load_traditional(sig, fold)
    if trad is None:
        return None, None
    df = trad.copy()
    df["hr"] = windows.loc[df["window_row"], f"{sig}_hr"].to_numpy()
    rows = []
    for kind, label in DL_KINDS.items():
        dl = load_dl(sig, kind, fold)
        if dl is None:
            rows.append({"estimator": label, "pending": True})
            continue
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
    fig.savefig(out / f"fig2_methods_{sig}.png", dpi=150)
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


TREND_SERIES = [("dl", "DL (PPG)", BLUE), ("count_orig|smart", "CtO smart fusion", ORANGE),
                ("ar_median|smart", "ARM corrected, smart fusion", AQUA)]


def fig_trends(trend, out, n=3):
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
    fig.savefig(out / "fig8_trends_ppg.png", dpi=140)
    plt.close(fig)
    ok = trend[trend["ref_ok"]]
    return [{"estimator": label, **trend_concordance(ok, col), **{k: v for k, v in rr_metrics(ok["ref_rr"], ok[col]).items()
                                                                  if k in ("mae", "coverage")}}
            for col, label, _ in TREND_SERIES], list(top)


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
            "ratio_median": float(ratio.median()), "ratio_gt04": float((ratio > 0.4).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="reports/fold0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    windows = pd.read_parquet(WINDOWS)
    fig_dataset(windows, out)
    ctx = {"fold": args.fold, "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           "dataset": dataset_summary(windows), "figs": {"dataset": out / "fig1_dataset.png"},
           "signals": {}, "notebooks": []}
    for sig in SIGNALS:
        t, df = comparison_table(sig, args.fold, windows)
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
        rows = t.drop(columns=["column"], errors="ignore").to_dict("records")
        per_patient = (dfr.assign(e_dl=(dfr["dl_mod"] - dfr["ref_rr"]).abs(), e_tr=(dfr[trad_col] - dfr["ref_rr"]).abs())
                       .groupby("patient_id")[["e_dl", "e_tr"]].mean())
        ctx["signals"][sig] = {
            "rows": rows, "dl": True, "n_windows": len(df), "n_patients": int(df["patient_id"].nunique()),
            "dl_full": json.loads((RUNS / f"{sig}_mod_w60_f{args.fold}" / "metrics.json").read_text())["test"],
            "best": next(r for r in rows if r["estimator"] == trad_label),
            "dl_better_patients": float((per_patient["e_dl"] < per_patient["e_tr"]).mean()),
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

    # sample windows: chosen from the PPG model's test predictions, all signals shown
    from scripts.case_study import pick_cases, window_detail
    test = load_dl("ppg", "mod", args.fold)
    ctx["cases"] = []
    if test is not None:
        test = test.rename(columns={"pred_rr": "dl_mod"})
        test["hr"] = windows.loc[test["window_row"], "ppg_hr"].to_numpy()
        for i, case in enumerate(pick_cases(test, windows, trad_col="none").to_dict("records")):
            det = window_detail(case["segment_id"], case["start_s"], dataset=DATASET)
            fig_case(det, case, i, out)
            ctx["cases"].append({**case, "fig": out / f"fig7_case{i}.png", "rates": case_rates(det),
                                 "ratio": case["ref_rr"] / case["hr"]})

    trend_file = RUNS / f"trends_f{args.fold}" / "trend_ppg.parquet"
    if trend_file.exists():
        conc, patients = fig_trends(pd.read_parquet(trend_file), out)
        ctx["trend"] = {"fig": out / "fig8_trends_ppg.png", "concordance": conc, "patients": patients}

    for nb in sorted(out.glob("[0-9][0-9]_*.html")):
        ctx["notebooks"].append((f"Notebook: {nb.stem[3:].replace('_', ' ')}", nb.name))

    from scripts.report_html import render
    (out / "report.html").write_text(render(ctx), encoding="utf-8")
    summary = {k: v for k, v in ctx.items() if k != "figs"}
    for s in summary["signals"].values():
        s.pop("figs", None)
    summary["cases"] = [{k: v for k, v in c.items() if k != "fig"} for c in ctx["cases"]]
    if "trend" in summary:
        summary["trend"] = {k: v for k, v in ctx["trend"].items() if k != "fig"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"report written to {out}/report.html")


if __name__ == "__main__":
    main()
