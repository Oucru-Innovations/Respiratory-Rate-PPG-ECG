"""Per-window detail and trend analysis shared by the report and notebook 03.

``window_detail`` recomputes, for one window, everything the estimators look
at: R peaks, PPG pulse peaks, the impedance breaths that define the
reference, the derived BW/AM/FM signals, the breaths CtO accepts on them, the
spectra with their power-law background and the peak each rule picks, and
every method's rate (published and corrected). ``trend_concordance`` scores
whether an estimator follows *changes* in RR (four-quadrant concordance,
Critchley et al. 2010, Anesth Analg 111:1180).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from data_prep.build_dataset import TARGET_FS
from estimate_rr import FUSIONS, METHODS, MODULATIONS, rate_from_resp_signal
from estimate_rr.freq_domain import aperiodic_residual
from estimate_rr.resp_signals import (DEFAULT_BAND, RESP_FS, extract_resp_signals,
                                      respiratory_waveform)
from estimate_rr.time_domain import count_adv, count_orig
from preprocess.beats import detect_ecg_rpeaks, detect_ppg_beats

DATASET = Path("dataset/01NVa_monitor")


def load_window(segment_id, start_s, window_s=60.0, dataset=DATASET):
    data = np.load(Path(dataset) / "signals" / f"{segment_id}.npz")
    return {k: data[k][int(round(start_s * fs)):int(round((start_s + window_s) * fs))].astype(float)
            for k, fs in TARGET_FS.items()}


def spectrum_detail(y, fs=RESP_FS, band=DEFAULT_BAND):
    """Welch spectrum, its power-law background over ``band`` and both picks."""
    f, p = signal.welch(y - y.mean(), fs=fs, nperseg=len(y) // 2, noverlap=len(y) // 4, nfft=4096)
    m = (f >= band[0]) & (f <= band[1])
    fb, resid = aperiodic_residual(f, p, band)
    background = np.exp(np.log(p[m] + 1e-300) - resid)
    pk, _ = signal.find_peaks(np.log(p[m] + 1e-300))
    pk_w, _ = signal.find_peaks(resid)
    return {"f_bpm": 60 * f[m], "psd": p[m], "background": background,
            "pick_published": 60 * fb[pk[np.argmax(p[m][pk])]] if len(pk) else np.nan,
            "pick_corrected": 60 * fb[pk_w[np.argmax(resid[pk_w])]] if len(pk_w) else np.nan}


def window_detail(segment_id, start_s, window_s=60.0, dataset=DATASET):
    w = load_window(segment_id, start_s, window_s, dataset)
    out = {"segment_id": segment_id, "start_s": start_s, "raw": w}
    out["rpeaks"], _, out["ecg_polarity"] = detect_ecg_rpeaks(w["ecg"], TARGET_FS["ecg"])
    out["ppg_peaks"], out["ppg_onsets"], _ = detect_ppg_beats(w["ppg"], TARGET_FS["ppg"])
    resp = respiratory_waveform(w["resp"], TARGET_FS["resp"])
    out["resp"] = resp
    out["ref_rr"], out["ref_breaths"] = count_adv(resp, RESP_FS)
    rows = []
    for sig in ("ppg", "ecg"):
        rs = extract_resp_signals(w[sig], TARGET_FS[sig], sig)
        out[f"{sig}_hr"] = rs.hr
        out[f"{sig}_derived"] = rs.signals
        out[f"{sig}_cto_breaths"] = {m: count_orig(y, RESP_FS)[1] for m, y in rs.signals.items()}
        out[f"{sig}_spectra"] = {m: spectrum_detail(y) for m, y in rs.signals.items()}
        for method, (_, abbrev) in METHODS.items():
            for corrected in (False, True):
                per = [rate_from_resp_signal(rs.signals[m], RESP_FS, method, corrected=corrected)
                       if m in rs.signals else np.nan for m in MODULATIONS]
                rows.append({"signal": sig.upper(), "method": abbrev,
                             "rule": "corrected" if corrected else "published",
                             **dict(zip([m.upper() for m in MODULATIONS], per)),
                             "smart fusion": FUSIONS["smart"](per), "median fusion": FUSIONS["median"](per)})
    out["rates"] = pd.DataFrame(rows)
    return out


def pick_cases(df, windows, pred_col="dl_mod", trad_col="count_orig|smart", seed=0):
    """Four illustrative held-out windows: typical agreement, fast breathing
    near the HR/2 limit, the slowest breathing, and the largest DL error.
    ``df`` holds test windows with ``ref_rr``, ``hr``, predictions and ``window_row``."""
    d = df.dropna(subset=[pred_col]).assign(err=lambda x: (x[pred_col] - x["ref_rr"]).abs(),
                                            ratio=lambda x: x["ref_rr"] / x["hr"])
    rng = np.random.default_rng(seed)
    typical = d[(d["err"] < 1) & d["ref_rr"].between(18, 28)]
    picks = {
        "Typical window": typical.iloc[rng.integers(len(typical))] if len(typical) else None,
        "Fast breathing near HR/2": d.sort_values("ratio").iloc[-max(1, len(d) // 500)],
        "Slowest breathing": d.sort_values("ref_rr").iloc[0],
        "Largest DL error": d.sort_values("err").iloc[-1],
    }
    rows = []
    for label, r in picks.items():
        if r is None:
            continue
        wr = windows.loc[int(r["window_row"])]
        rows.append({"case": label, "window_row": int(r["window_row"]), "segment_id": wr["segment_id"],
                     "patient_id": wr["patient_id"], "start_s": float(wr["start_s"]),
                     "ref_rr": float(r["ref_rr"]), "dl": float(r[pred_col]),
                     "trad": float(r[trad_col]) if trad_col in r and np.isfinite(r[trad_col]) else np.nan,
                     "hr": float(r["hr"])})
    return pd.DataFrame(rows)


def trend_concordance(df, pred_col, lag_windows=10, exclusion_bpm=2.0):
    """Four-quadrant trending: changes over ``lag_windows`` consecutive windows
    (10 x 30 s = 5 min) of the reference vs the estimate, within one segment.
    Pairs whose reference change is below ``exclusion_bpm`` are excluded (noise
    zone). Returns concordance rate (same direction of change), the number of
    pairs, and the correlation of the changes."""
    pairs = []
    for _, g in df.sort_values("start_s").groupby("segment_id"):
        g = g.set_index("start_s")
        step = 30.0 * lag_windows
        common = g.index.intersection(g.index + step)
        if len(common) == 0:
            continue
        later = g.loc[common]
        earlier = g.loc[common - step]
        pairs.append(pd.DataFrame({"d_ref": later["ref_rr"].to_numpy() - earlier["ref_rr"].to_numpy(),
                                   "d_pred": later[pred_col].to_numpy() - earlier[pred_col].to_numpy()}))
    if not pairs:
        return {"concordance": np.nan, "n_pairs": 0, "r_change": np.nan}
    p = pd.concat(pairs).dropna()
    p = p[p["d_ref"].abs() >= exclusion_bpm]
    if len(p) < 3:
        return {"concordance": np.nan, "n_pairs": len(p), "r_change": np.nan}
    return {"concordance": float((np.sign(p["d_ref"]) == np.sign(p["d_pred"])).mean()),
            "n_pairs": int(len(p)), "r_change": float(np.corrcoef(p["d_ref"], p["d_pred"])[0, 1])}


def rolling_trend(g, col, window=10, gap_h=0.1):
    """Centred rolling median of ``col`` per segment, as (hours, values) arrays
    with a NaN inserted wherever consecutive windows are more than ``gap_h``
    hours apart, so plotted lines break at recording gaps instead of bridging
    them. ``g`` needs columns ``hours``, ``segment_id`` and ``col``."""
    parts = []
    for _, s in g.sort_values("hours").groupby("segment_id", sort=False):
        r = s.set_index("hours")[col].rolling(window, min_periods=max(2, window // 2 - 1), center=True).median()
        parts.append(r)
    if not parts:
        return np.array([]), np.array([])
    r = pd.concat(parts).sort_index()
    h, v = r.index.to_numpy(), r.to_numpy(dtype=float)
    cut = np.flatnonzero(np.diff(h) > gap_h) + 1
    return np.insert(h, cut, np.nan), np.insert(v, cut, np.nan)
