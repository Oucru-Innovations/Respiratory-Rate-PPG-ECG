"""Window-level signal-quality checks shared by reference and input signals."""
import numpy as np

from estimate_rr.resp_signals import extract_resp_signals


def saturation_fraction(x, lo, hi, tol=1e-3):
    """Fraction of samples sitting at the ADC rails ``lo``/``hi`` (the recording's
    min/max). GE exports clip e.g. Pleth at +/-7.5 and Resp at +/-9."""
    span = max(hi - lo, 1e-12)
    return float(np.mean((x <= lo + tol * span) | (x >= hi - tol * span)))


def flat_fraction(x, fs, min_run_s=1.0):
    """Fraction of the window covered by runs of identical consecutive samples
    lasting at least ``min_run_s`` (lead-off / sensor-off)."""
    same = np.concatenate([[False], np.diff(x) == 0])
    if not same.any():
        return 0.0
    d = np.diff(np.concatenate([[0], same.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = ends - starts
    return float(runs[runs >= min_run_s * fs].sum() / len(x))


def amplitude_burst_ratio(x, fs, block_s=5.0):
    """Max/median of the interquartile range over ``block_s`` blocks; large
    values indicate motion artefact bursts."""
    n = int(block_s * fs)
    blocks = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    if len(blocks) < 2:
        return 1.0
    iqr = np.array([np.subtract(*np.percentile(b, [75, 25])) for b in blocks])
    med = np.median(iqr)
    return float(iqr.max() / med) if med > 0 else np.inf


def cardiac_quality(x, fs, signal_type, rails=None):
    """Quality summary for a PPG/ECG window. ``ok`` requires no NaNs, < 1 %
    saturation, < 10 % flat-line, plausible HR and <= 20 % rejected beats."""
    q = {"nan_frac": float(np.mean(~np.isfinite(x)))}
    if q["nan_frac"] > 0:
        return {**q, "sat_frac": np.nan, "flat_frac": np.nan, "hr": np.nan,
                "beat_reject_frac": 1.0, "ok": False}
    q["sat_frac"] = saturation_fraction(x, *rails) if rails else 0.0
    q["flat_frac"] = flat_fraction(x, fs)
    rs = extract_resp_signals(x, fs, signal_type)
    q["hr"] = rs.hr
    q["beat_reject_frac"] = rs.beat_reject_frac
    q["ok"] = bool(q["sat_frac"] < 0.01 and q["flat_frac"] < 0.1
                   and np.isfinite(rs.hr) and 30 <= rs.hr <= 240
                   and rs.beat_reject_frac <= 0.2 and rs.ok)
    return q
