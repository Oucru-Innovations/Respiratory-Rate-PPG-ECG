"""Wearable recordings of the 01NVa_Dengue study and their alignment to the GE monitor.

Each patient wore, besides the bedside monitor, a SmartCare wrist PPG (``PPG/*.csv``,
100 Hz) and/or a Faros/ePatch chest ECG (``ECG/*.edf``, 256 Hz). The monitor's
impedance gives the reference breathing rate; to score the wearables against it,
every wearable sample needs a monitor time stamp.

Clocks:

* SmartCare: the file name is the start time (``20210112T101204.930+0700.csv``);
  ``TIMESTAMP_MS`` counts from 0. Six early files (``<patient> Smartcare.csv``)
  carry no start time.
* Faros/ePatch EDF: the header start date is real in about half the files; the
  rest show 01.01.2000 (the device clock was never set).
* GE monitor: its header epoch differs from its header wall clock by 7 h.

None of these is trusted. The offset between a wearable and the monitor is
measured from the heart itself: instantaneous heart rate from the wearable's beats
and from the monitor's ECG/PPG beats is cross-correlated (coarse, over the whole
search range, at 1 Hz), then refined block by block at 4 Hz, which also measures
clock drift. Alignment is accepted only if the blocks agree.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess.beats import detect_ecg_rpeaks, detect_ppg_beats, valid_beat_mask

ISO_NAME = re.compile(r"(\d{8}T\d{6}\.\d{3}[+-]\d{4})")


# ---------------------------------------------------------------- readers
@dataclass
class WearableRecording:
    path: str
    kind: str                     # "ppg" or "ecg"
    fs: float
    signal: np.ndarray            # one channel, float32
    t0_nominal: float = None      # epoch seconds from the device clock / file name, if any
    extra: dict = field(default_factory=dict)

    @property
    def duration_s(self):
        return len(self.signal) / self.fs


def read_smartcare(path):
    """SmartCare wrist PPG export. Gaps in ``TIMESTAMP_MS`` are filled with NaN so
    sample k is always at k / 100 s after the first sample."""
    import pyarrow.csv as pacsv
    path = Path(path)
    # some exports have truncated or merged lines (device write errors): skip them, keep the rest
    tbl = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(invalid_row_handler=lambda row: "skip"),
                         convert_options=pacsv.ConvertOptions(include_columns=["TIMESTAMP_MS", "PLETH", "PULSE_BPM", "SPO2_STATUS"]))
    df = tbl.to_pandas().dropna(subset=["TIMESTAMP_MS"])
    df = df[np.isfinite(pd.to_numeric(df["TIMESTAMP_MS"], errors="coerce"))]
    t_ms = df["TIMESTAMP_MS"].to_numpy(np.int64)
    fs = 100.0
    idx = np.round((t_ms - t_ms[0]) / (1000.0 / fs)).astype(np.int64)
    keep = np.r_[True, np.diff(idx) > 0]                          # drop duplicated / backwards stamps
    idx, df = idx[keep], df.loc[keep]
    sig = np.full(idx[-1] + 1, np.nan, dtype=np.float32)
    sig[idx] = df["PLETH"].to_numpy(np.float32)
    m = ISO_NAME.search(path.name)
    t0 = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S.%f%z").timestamp() + t_ms[0] / 1000.0 if m else None
    pulse = np.full(len(sig), np.nan, dtype=np.float32)
    pulse[idx] = df["PULSE_BPM"].to_numpy(np.float32)
    return WearableRecording(str(path), "ppg", fs, sig, t0,
                             {"gap_frac": float(1 - keep.sum() / len(sig)), "device_pulse": pulse})


def read_edf_ecg(path, channel=0):
    """Minimal EDF/EDF+ reader for the Faros/ePatch exports (16-bit samples). Returns
    the first ECG channel in microvolts. Device clocks left at 2000-01-01 give
    ``t0_nominal = None``."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(256).decode("latin1")
        ns = int(head[252:256])
        sh = fh.read(256 * ns).decode("latin1")
        n_rec, rec_dur = int(head[236:244]), float(head[244:252])

        def fld(offset, width):
            return [sh[offset * ns + i * width: offset * ns + (i + 1) * width].strip() for i in range(ns)]
        labels = fld(0, 16)
        o = 16 + 80 + 8
        pmin, pmax = np.array(fld(o, 8), float), np.array(fld(o + 8, 8), float)
        dmin, dmax = np.array(fld(o + 16, 8), float), np.array(fld(o + 24, 8), float)
        nsamp = np.array(fld(o + 32 + 80, 8), int)
        raw = np.fromfile(fh, dtype="<i2")
    rec_len = int(nsamp.sum())
    n_rec = min(n_rec if n_rec > 0 else len(raw) // rec_len, len(raw) // rec_len)
    raw = raw[:n_rec * rec_len].reshape(n_rec, rec_len)
    ecg_idx = [i for i, l in enumerate(labels) if "annotation" not in l.lower()]
    i = ecg_idx[channel]
    a = int(nsamp[:i].sum())
    x = raw[:, a:a + nsamp[i]].reshape(-1).astype(np.float32)
    x = (x - dmin[i]) * (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) + pmin[i]
    fs = nsamp[i] / rec_dur
    d, t = head[168:176], head[176:184]
    t0 = None
    if not d.endswith("00"):        # dd.mm.yy; year 2000 = clock never set
        # EDF start time is the device's local wall clock; the study ran in Vietnam (UTC+7)
        t0 = datetime.strptime(f"{d} {t} +0700", "%d.%m.%y %H.%M.%S %z").timestamp()
    return WearableRecording(str(path), "ecg", float(fs), x, t0, {"labels": labels, "n_channels": len(ecg_idx)})


# ---------------------------------------------------------------- heart-rate series
def beat_times(x, fs, kind, chunk_s=600.0):
    """Beat times (s) over a long recording, detected in overlapping chunks; NaN
    stretches (gaps) are skipped."""
    detect = detect_ppg_beats if kind == "ppg" else detect_ecg_rpeaks
    n, step, pad = len(x), int(chunk_s * fs), int(5 * fs)
    out = []
    for a in range(0, n, step):
        lo, hi = max(a - pad, 0), min(a + step + pad, n)
        seg = x[lo:hi]
        if np.mean(np.isfinite(seg)) < 0.9 or np.nanstd(seg) == 0:
            continue
        seg = np.where(np.isfinite(seg), seg, np.nanmedian(seg)).astype(float)
        try:
            peaks = detect(seg, fs)[0]
        except Exception:
            continue
        t = (lo + np.asarray(peaks)) / fs
        out.append(t[(t >= a / fs) & (t < min(a + step, n) / fs)])
    return np.sort(np.concatenate(out)) if out else np.array([])


def hr_series(beats, t_end, step=1.0, t_start=0.0):
    """Instantaneous heart rate (bpm) sampled every ``step`` s on [t_start, t_end);
    NaN where no plausible beat interval is available."""
    grid = np.arange(t_start, t_end, step)
    out = np.full(len(grid), np.nan)
    if len(beats) < 3:
        return grid, out
    ok = valid_beat_mask(beats)
    ibi = np.diff(beats)
    t_mid, hr = beats[1:][ok[1:]], 60.0 / ibi[ok[1:]]
    good = (hr > 25) & (hr < 230)
    t_mid, hr = t_mid[good], hr[good]
    if len(hr) < 3:
        return grid, out
    k = np.searchsorted(t_mid, grid)
    vals = np.interp(grid, t_mid, hr)
    near = np.minimum(np.abs(grid - t_mid[np.clip(k - 1, 0, len(t_mid) - 1)]),
                      np.abs(t_mid[np.clip(k, 0, len(t_mid) - 1)] - grid))
    out = np.where(near <= 3.0, vals, np.nan)
    return grid, out


# ---------------------------------------------------------------- alignment
def _masked_xcorr(a, b, max_lag):
    """Normalised cross-correlation of two NaN-gapped series at integer lags
    -max_lag..max_lag (b shifted against a); returns (lags, r, n_overlap)."""
    ma, mb = np.isfinite(a), np.isfinite(b)
    a0, b0 = np.where(ma, a, 0.0), np.where(mb, b, 0.0)
    n = len(a) + len(b)
    nfft = 1 << int(np.ceil(np.log2(n)))
    F = np.fft.rfft

    def xc(u, v):
        return np.fft.irfft(F(u, nfft) * np.conj(F(v, nfft)), nfft)
    cnt = xc(ma.astype(float), mb.astype(float))
    sa, sb = xc(a0, mb.astype(float)), xc(ma.astype(float), b0)
    saa, sbb = xc(a0 ** 2, mb.astype(float)), xc(ma.astype(float), b0 ** 2)
    sab = xc(a0, b0)
    lags = np.arange(-max_lag, max_lag + 1)
    idx = lags % nfft
    c, xa, xb, xaa, xbb, xab = (v[idx] for v in (cnt, sa, sb, saa, sbb, sab))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = xab - xa * xb / c
        var = (xaa - xa ** 2 / c) * (xbb - xb ** 2 / c)
        r = cov / np.sqrt(var)
    r[c < 600] = np.nan
    return lags, r, np.round(c).astype(int)


def _detrend(h, win):
    """Remove a running median (``win`` samples) so slow HR trends shared by every
    patient do not dominate; keeps the beat-to-beat and minute-scale variability."""
    s = pd.Series(h)
    return (s - s.rolling(win, center=True, min_periods=win // 4).median()).to_numpy()


def coarse_offset(hr_wear, hr_mon, max_lag, detrend_s=300):
    """Lag L (samples of the 1-Hz grid) maximising the correlation of
    hr_mon[i + L] with hr_wear[i]; with the peak r, the runner-up r outside +-60 s,
    and the overlap."""
    c = coarse_candidates(hr_wear, hr_mon, max_lag, detrend_s, k=2)
    if not c:
        return None
    second = c[1]["r"] if len(c) > 1 else np.nan
    return {**c[0], "r_second": second}


def coarse_candidates(hr_wear, hr_mon, max_lag, detrend_s=300, k=5, min_sep=300, drifts_ppm=(0.0,),
                      lag_windows=None):
    """Best (lag, drift) pairs of the correlation between detrended 1-Hz heart-rate
    series, the wearable time axis stretched by each candidate drift (monitor time =
    lag + (1 + drift) * t): the global top ``k`` (lags at least ``min_sep`` apart),
    plus the best pair inside each of ``lag_windows`` (lo, hi) if given. The cross-
    correlation is computed once per drift."""
    a = _detrend(hr_mon, detrend_s)
    glob, local = {}, [None] * len(lag_windows or [])
    for d in drifts_ppm:
        f = 1.0 + d * 1e-6
        u = np.arange(0, (len(hr_wear) - 1) * f, 1.0)
        src = u / f
        i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, len(hr_wear) - 1); w = src - i0
        y = (1 - w) * hr_wear[i0] + w * hr_wear[i1]          # NaN propagates: gaps stay gaps
        lags, r, c = _masked_xcorr(a, _detrend(y, detrend_s), max_lag)
        r = np.where(np.isfinite(r), r, -np.inf)
        for n, (lo, hi) in enumerate(lag_windows or []):
            rw = np.where((lags >= lo) & (lags <= hi), r, -np.inf)
            j = int(np.argmax(rw))
            if np.isfinite(rw[j]) and (local[n] is None or local[n]["r"] < rw[j]):
                local[n] = {"lag": int(lags[j]), "r": float(rw[j]), "overlap": int(c[j]), "drift_ppm": float(d), "window": n}
        for _ in range(k):
            j = int(np.argmax(r))
            if not np.isfinite(r[j]):
                break
            key = int(lags[j]) // min_sep
            if key not in glob or glob[key]["r"] < r[j]:
                glob[key] = {"lag": int(lags[j]), "r": float(r[j]), "overlap": int(c[j]), "drift_ppm": float(d)}
            r[np.abs(lags - lags[j]) < min_sep] = -np.inf
    out = []
    for cand in sorted(glob.values(), key=lambda v: -v["r"]):
        if all(abs(cand["lag"] - o["lag"]) >= min_sep for o in out):
            out.append(cand)
        if len(out) == k:
            break
    return out + [c for c in local if c is not None]


def hr_fine(beats, t0, t1, fs=4.0, hp_s=30.0):
    """Instantaneous HR on a ``fs``-Hz grid over [t0, t1) with slow changes removed
    (running median over ``hp_s``): what remains is beat-to-beat variability, mostly
    respiratory sinus arrhythmia, which is specific to each moment."""
    _, h = hr_series(beats, t1, step=1.0 / fs, t_start=t0)
    return _detrend(h, int(hp_s * fs))


def _block_lags(beats_wear, beats_mon, line, block_s, search_s, fs, min_valid, t_range=None):
    """Per-block best lag (s) of the monitor HR against the wearable HR, searched
    within +-search_s of ``line(t)`` (monitor time minus wearable time)."""
    t_lo, t_hi = t_range or (beats_wear[0], beats_wear[-1])
    out = []
    for a in np.arange(t_lo, t_hi - block_s / 2, block_s):
        b = min(a + block_s, t_hi)
        centre = line((a + b) / 2)
        hw = hr_fine(beats_wear, a, b, fs)
        hm = hr_fine(beats_mon, a + centre - search_s, b + centre + search_s, fs)
        if np.isfinite(hw).mean() < min_valid or np.isfinite(hm).mean() < min_valid:
            continue
        lags, r, c = _masked_xcorr(hm, hw, int(2 * search_s * fs))
        keep = (lags >= 0) & (lags <= 2 * search_s * fs) & (c >= min_valid * len(hw)) & np.isfinite(r)
        if not keep.any():
            continue
        j = int(np.argmax(np.where(keep, r, -np.inf)))
        out.append({"t": (a + b) / 2, "t0": a, "t1": b, "lag_s": centre + lags[j] / fs - search_s, "r": float(r[j])})
    return pd.DataFrame(out, columns=["t", "t0", "t1", "lag_s", "r"])


def _theil_sen(t, l):
    if len(t) < 2:
        return 0.0, float(np.median(l)) if len(l) else np.nan
    i, j = np.triu_indices(len(t), 1)
    ok = t[j] != t[i]
    slope = float(np.median((l[j] - l[i])[ok] / (t[j] - t[i])[ok])) if ok.any() else 0.0
    return slope, float(np.median(l - slope * t))


def _pieces(bl, tol_s=2.0, min_blocks=3):
    """Split time-ordered block lags into pieces that each follow one straight line
    (constant offset + drift); a clock reset or recording restart starts a new piece.
    Greedy: grow a piece while the next block lies within ``tol_s`` of the line
    fitted so far; an outlier block is skipped unless the next one disagrees too."""
    bl = bl.sort_values("t").reset_index(drop=True)
    pieces, cur, miss = [], [], 0
    for k in range(len(bl)):
        if len(cur) < 2:
            if cur and abs(bl.lag_s[k] - bl.lag_s[cur[-1]]) > tol_s + 0.01 * (bl.t[k] - bl.t[cur[-1]]):
                cur = [k]
            else:
                cur.append(k)
            continue
        slope, inter = _theil_sen(bl.t[cur].to_numpy(), bl.lag_s[cur].to_numpy())
        if abs(bl.lag_s[k] - (inter + slope * bl.t[k])) <= tol_s:
            cur.append(k); miss = 0
        else:
            miss += 1
            if miss >= 2:
                pieces.append(cur); cur, miss = [k - 1, k] if abs(bl.lag_s[k] - bl.lag_s[k - 1]) <= tol_s else [k], 0
    pieces.append(cur)
    return [bl.loc[p] for p in pieces if len(p) >= min_blocks]


def fine_align(beats_wear, beats_mon, offset, drift_ppm=0.0, coarse_block_s=600.0, coarse_search_s=90.0,
               fine_block_s=300.0, fine_search_s=8.0, fs=4.0, min_valid=0.3, min_r=0.4,
               max_drift_ppm=6000.0):
    """Turn a candidate offset (monitor time = wearable time + offset) into a
    verified, piecewise-linear time map.

    Pass 1: 10-min blocks searched +-90 s around the candidate line (offset plus the
    coarse drift estimate; the SmartCare sample clock runs up to ~0.4 % off, which
    would smear longer blocks). Blocks with r >= ``min_r`` are split into
    straight-line pieces (a clock reset or recording restart starts a new piece).
    Pass 2: within each piece, 5-min blocks searched +-8 s around its line; the
    piece is kept when at least 3 of them agree to within 1 s (median r >= 0.5).
    Returns the pieces as dicts (wearable-time span, lag0_s, drift_ppm, evidence)."""
    if len(beats_wear) < 100 or len(beats_mon) < 100:
        return {"pieces": [], "blocks": pd.DataFrame()}
    b1 = _block_lags(beats_wear, beats_mon, lambda t: offset + drift_ppm * 1e-6 * t, coarse_block_s,
                     coarse_search_s, fs, min_valid)
    good = b1[b1["r"] >= min_r]
    pieces = []
    for pc in _pieces(good):
        slope, inter = _theil_sen(pc["t"].to_numpy(), pc["lag_s"].to_numpy())
        if abs(slope) * 1e6 > max_drift_ppm:
            continue
        span = (pc["t0"].min(), pc["t1"].max())
        b2 = _block_lags(beats_wear, beats_mon, lambda t: inter + slope * t, fine_block_s, fine_search_s, fs,
                         min_valid, span)
        g2 = b2[b2["r"] >= min_r]
        if len(g2) < 3:
            continue
        s2, i2 = _theil_sen(g2["t"].to_numpy(), g2["lag_s"].to_numpy())
        resid = g2["lag_s"].to_numpy() - (i2 + s2 * g2["t"].to_numpy())
        inl = np.abs(resid) <= 1.0
        piece = {"t_start": float(span[0]), "t_end": float(span[1]), "lag0_s": i2, "drift_ppm": s2 * 1e6,
                 "n_blocks": int(len(b2)), "n_good": int(len(g2)), "n_inliers": int(inl.sum()),
                 "inlier_frac": float(inl.mean()), "median_r": float(np.median(g2["r"][inl])) if inl.any() else np.nan,
                 "resid_sd_s": float(np.std(resid[inl])) if inl.sum() > 1 else np.nan}
        piece["accepted"] = bool(piece["n_inliers"] >= 3 and piece["inlier_frac"] >= 0.6 and piece["median_r"] >= 0.5)
        pieces.append(piece)
    return {"pieces": pieces, "blocks": b1}
