"""Zero-phase filtering and resampling helpers.

All filters use second-order sections with ``sosfiltfilt`` so that very low
cut-offs (e.g. 0.05 Hz at 300 Hz sampling) stay numerically stable and no
phase lag is introduced (breath timing matters for the time-domain methods).
"""
from fractions import Fraction

import numpy as np
from scipy import signal


def _sos(order, cutoff, btype, fs):
    return signal.butter(order, cutoff, btype=btype, fs=fs, output="sos")


def _filt(sos, x):
    x = np.asarray(x, dtype=float)
    # sosfiltfilt needs a minimum length; fall back to the input when too short
    padlen = 3 * (2 * len(sos) + 1)
    if len(x) <= padlen:
        return x - np.mean(x) if len(x) else x
    return signal.sosfiltfilt(sos, x)


def bandpass(x, fs, low, high, order=4):
    """Zero-phase Butterworth band-pass between ``low`` and ``high`` Hz."""
    return _filt(_sos(order, [low, high], "bandpass", fs), x)


def lowpass(x, fs, cutoff, order=4):
    return _filt(_sos(order, cutoff, "lowpass", fs), x)


def highpass(x, fs, cutoff, order=4):
    return _filt(_sos(order, cutoff, "highpass", fs), x)


def resample(x, fs_in, fs_out):
    """Polyphase resampling (includes the anti-aliasing filter)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(np.asarray(x, dtype=float), frac.numerator, frac.denominator)


def interpolate_uniform(t, y, fs, t_start=None, t_end=None, kind="linear"):
    """Interpolate an irregularly sampled series (e.g. one value per beat) onto a
    uniform grid at ``fs`` Hz. Returns ``(t_uniform, y_uniform)``.

    Values outside the observed range are held at the nearest observation rather
    than extrapolated, so a missing first/last beat does not create a ramp.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t_start = t[0] if t_start is None else t_start
    t_end = t[-1] if t_end is None else t_end
    tu = np.arange(t_start, t_end, 1.0 / fs)
    if kind == "cubic" and len(t) >= 4:
        from scipy.interpolate import CubicSpline
        yu = CubicSpline(t, y, extrapolate=False)(tu)
        yu[tu < t[0]] = y[0]
        yu[tu > t[-1]] = y[-1]
    else:
        yu = np.interp(tu, t, y)
    return tu, yu
