"""Beat detection for ECG (R peaks) and PPG (systolic peaks + pulse onsets).

Both detectors use locally adaptive thresholds so they work on long recordings
whose amplitude drifts, and both return sample indices into the input array.

References
----------
ECG: gradient-based QRS detector as used by NeuroKit2 (Makowski et al., 2021,
     Behav Res Methods 53:1689) -- QRS complexes are segments where the smoothed
     absolute gradient exceeds 1.5x its 0.75-s moving average.
PPG: two-moving-average detector of Elgendi et al. (2013), PLoS ONE 8(10):e76585.
"""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from preprocess.filters import bandpass

# Physiological limits used to discard implausible beats (seconds).
MIN_IBI = 60.0 / 240  # 240 bpm (paediatric tachycardia)
MAX_IBI = 60.0 / 25   # 25 bpm


def _moving_average(x, n):
    n = max(int(n), 1)
    return uniform_filter1d(x, size=n, mode="nearest")


def _segments(mask):
    """Return (start, end) index pairs of contiguous True runs in ``mask``."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def detect_ecg_rpeaks(ecg, fs):
    """Detect R peaks. Returns ``(rpeaks, ecg_filtered, polarity)``.

    ``ecg_filtered`` is the 0.5-40 Hz ECG multiplied by ``polarity`` (+1/-1) so
    that R waves point upwards; apply the same polarity to any other version of
    the signal that features are read from.
    """
    ecg = np.asarray(ecg, dtype=float)
    ecg_f = bandpass(ecg, fs, 0.5, min(40.0, 0.45 * fs), order=3)

    # QRS energy: 5-15 Hz band, smoothed absolute gradient
    qrs_band = bandpass(ecg, fs, 5.0, min(15.0, 0.45 * fs), order=3)
    grad = np.abs(np.gradient(qrs_band))
    smooth = _moving_average(grad, 0.1 * fs)
    avg = _moving_average(grad, 0.75 * fs)
    starts, ends = _segments(smooth > 1.5 * avg)
    if len(starts) == 0:
        return np.array([], dtype=int), ecg_f, 1
    widths = ends - starts
    keep = widths > 0.4 * np.mean(widths)
    starts, ends = starts[keep], ends[keep]

    # Polarity: R should be the dominant positive deflection
    pos = np.median([ecg_f[s:e].max() for s, e in zip(starts, ends)])
    neg = np.median([-ecg_f[s:e].min() for s, e in zip(starts, ends)])
    polarity = -1 if neg > 1.5 * pos else 1
    ecg_f = polarity * ecg_f

    peaks = np.array([s + np.argmax(ecg_f[s:e]) for s, e in zip(starts, ends)], dtype=int)
    peaks = _enforce_refractory(peaks, ecg_f, fs)
    return peaks, ecg_f, polarity


def detect_ppg_beats(ppg, fs):
    """Detect PPG systolic peaks and pulse onsets (preceding troughs).

    Returns ``(peaks, onsets, ppg_filtered)``; ``onsets[i]`` is the trough before
    ``peaks[i]`` (the first peak without a preceding trough is dropped).
    """
    ppg = np.asarray(ppg, dtype=float)
    # Elgendi 2013: 0.5-8 Hz band-pass, clip negatives, square
    y = bandpass(ppg, fs, 0.5, min(8.0, 0.45 * fs), order=2)
    z = np.clip(y, 0, None) ** 2
    ma_peak = _moving_average(z, 0.111 * fs)
    ma_beat = _moving_average(z, 0.667 * fs)
    thr = ma_beat + 0.02 * np.mean(z)
    starts, ends = _segments(ma_peak > thr)
    keep = (ends - starts) >= int(0.111 * fs)
    starts, ends = starts[keep], ends[keep]
    peaks = np.array([s + np.argmax(y[s:e]) for s, e in zip(starts, ends)], dtype=int)
    peaks = _enforce_refractory(peaks, y, fs)

    # Onset = minimum of the band-passed signal between consecutive peaks
    onsets, kept = [], []
    for i in range(1, len(peaks)):
        a, b = peaks[i - 1], peaks[i]
        onsets.append(a + np.argmin(y[a:b]))
        kept.append(b)
    return np.array(kept, dtype=int), np.array(onsets, dtype=int), y


def _enforce_refractory(peaks, x, fs):
    """When two peaks are closer than MIN_IBI keep the larger one."""
    if len(peaks) < 2:
        return peaks
    out = [peaks[0]]
    for p in peaks[1:]:
        if (p - out[-1]) < MIN_IBI * fs:
            if x[p] > x[out[-1]]:
                out[-1] = p
        else:
            out.append(p)
    return np.asarray(out, dtype=int)


def valid_beat_mask(beat_times, max_rel_dev=0.3, window=7):
    """Flag beats whose preceding inter-beat interval is physiologically
    implausible or deviates more than ``max_rel_dev`` from the local median IBI
    (ectopic beats / missed detections). The first beat is always valid.
    """
    t = np.asarray(beat_times, dtype=float)
    ok = np.ones(len(t), dtype=bool)
    if len(t) < 3:
        return ok
    ibi = np.diff(t)
    med = median_filter(ibi, size=window, mode="nearest")
    bad = (ibi < MIN_IBI) | (ibi > MAX_IBI) | (np.abs(ibi - med) > max_rel_dev * med)
    ok[1:] = ~bad
    return ok


def heart_rate_summary(beat_times):
    """Median HR (bpm) and fraction of beats rejected by ``valid_beat_mask``."""
    t = np.asarray(beat_times, dtype=float)
    if len(t) < 3:
        return np.nan, 1.0
    ok = valid_beat_mask(t)
    ibi = np.diff(t)[ok[1:]]
    hr = 60.0 / np.median(ibi) if len(ibi) else np.nan
    return hr, 1.0 - ok.mean()
