"""Time-domain breath detection on a conditioned respiratory signal.

Every function takes a uniformly sampled, band-passed respiratory signal
(see :mod:`estimate_rr.resp_signals`) and returns ``(rr_bpm, breath_times_s)``.
``rr_bpm`` is NaN when fewer than ``MIN_BREATHS`` breath intervals are found.
The rate is ``60 / mean(breath interval)``, which -- unlike counting breaths and
dividing by the window length -- is not biased by partial breaths at the window
edges or by rejected cycles.
"""
import numpy as np
from scipy.signal import argrelextrema, find_peaks

MIN_BREATHS = 2  # minimum number of breath intervals for a valid estimate


def _rate(breath_times):
    breath_times = np.asarray(breath_times, dtype=float)
    if len(breath_times) < MIN_BREATHS + 1:
        return np.nan, breath_times
    return 60.0 / np.mean(np.diff(breath_times)), breath_times


def _extrema(x):
    maxima = argrelextrema(x, np.greater)[0]
    minima = argrelextrema(x, np.less)[0]
    return maxima, minima


def count_orig(x, fs, q_factor=0.2):
    """Count-orig (Schäfer & Kratky 2008, Ann Biomed Eng 36:2061).

    Valid peaks are local maxima above ``0.2 x Q3`` of all local maxima; valid
    troughs are local minima below zero. A breath is the interval between two
    consecutive valid peaks that contains exactly one valid trough.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    maxima, minima = _extrema(x)
    if len(maxima) < 2:
        return np.nan, np.array([])
    thr = q_factor * np.percentile(x[maxima], 75)
    peaks = maxima[x[maxima] > thr]
    troughs = minima[x[minima] < 0]
    intervals, times = [], []
    for a, b in zip(peaks[:-1], peaks[1:]):
        if np.count_nonzero((troughs > a) & (troughs < b)) == 1:
            intervals.append((b - a) / fs)
            times.append(b / fs)
    if len(intervals) < MIN_BREATHS:
        return np.nan, np.asarray(times)
    return 60.0 / np.mean(intervals), np.asarray(times)


def count_adv(x, fs, q_factor=0.3):
    """Count-adv (Schäfer & Kratky 2008).

    Starting from all local extrema, repeatedly remove the pair of consecutive
    extrema with the smallest absolute amplitude difference while that
    difference is below ``0.3 x Q3`` of all consecutive differences. The
    remaining maxima mark breaths.
    """
    x = np.asarray(x, dtype=float)
    maxima, minima = _extrema(x)
    ext = np.sort(np.concatenate([maxima, minima]))
    if len(ext) < 4:
        return np.nan, np.array([])
    is_max = np.isin(ext, maxima)
    diffs = np.abs(np.diff(x[ext]))
    thr = q_factor * np.percentile(diffs, 75)
    while len(ext) > 3:
        diffs = np.abs(np.diff(x[ext]))
        i = int(np.argmin(diffs))
        if diffs[i] >= thr:
            break
        ext = np.delete(ext, [i, i + 1])
        is_max = np.delete(is_max, [i, i + 1])
    return _rate(ext[is_max] / fs)


def peak_detection(x, fs, max_rate_bpm=60.0, rel_prominence=0.3):
    """Simple peak detection: maxima separated by at least one breath at
    ``max_rate_bpm`` and with prominence above ``rel_prominence`` x the signal's
    interquartile range."""
    x = np.asarray(x, dtype=float)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    peaks, _ = find_peaks(x, distance=max(1, int(fs * 60.0 / max_rate_bpm)),
                          prominence=rel_prominence * iqr)
    return _rate(peaks / fs)


def zero_crossing(x, fs, min_interval_s=1.0):
    """Positive-going zero crossings of the mean-removed signal; crossings closer
    than ``min_interval_s`` to the previous one are treated as noise."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    idx = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    # sub-sample crossing time by linear interpolation
    t = (idx + (-x[idx]) / (x[idx + 1] - x[idx])) / fs
    kept = []
    for ti in t:
        if not kept or ti - kept[-1] >= min_interval_s:
            kept.append(ti)
    return _rate(kept)
