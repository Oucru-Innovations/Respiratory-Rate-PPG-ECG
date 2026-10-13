"""Frequency-domain respiratory-rate estimation.

Every function takes a uniformly sampled, band-passed respiratory signal and
returns ``(rr_bpm, (freqs_hz, psd))`` where ``rr_bpm = 60 x`` the frequency of
the dominant spectral peak inside ``band`` (Hz). ``rr_bpm`` is NaN when no
peak exists inside the band.

Peak selection (``whiten``)
---------------------------
Respiratory signals derived from the ECG/PPG carry a 1/f-shaped
very-low-frequency background (vasomotion, Mayer waves, posture). On the
01NVa training patients ~20 % of the 4-60 bpm power lies below 8 bpm, so the
*largest* peak in the band is usually that background, not breathing, and the
published "maximum spectral power" rule under-estimates by ~10 bpm.
With ``whiten=True`` (default) the background is modelled as a power law,
``log P = a + b log f`` fitted over the band, and the selected peak is the one
standing highest above it -- the aperiodic-component correction used for
neural power spectra (Donoghue et al. 2020, Nat Neurosci 23:1655). The band
keeps its 4-bpm floor, so slow breathing can still be detected.
``whiten=False`` reproduces the published rule.
"""
import numpy as np
from scipy import signal

from estimate_rr.resp_signals import DEFAULT_BAND

NFFT = 4096  # zero-padding only interpolates the spectrum; it adds no resolution


def aperiodic_residual(f, p, band):
    """``log p`` minus a power-law fit over ``band``; returns ``(f_band, residual)``."""
    m = (f > 0) & (f >= band[0]) & (f <= band[1])
    lf, lp = np.log(f[m]), np.log(p[m] + 1e-300)
    return f[m], lp - np.polyval(np.polyfit(lf, lp, 1), lf)


def _peak_rate(f, p, band, whiten=True):
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3 or not np.all(np.isfinite(p[m])):
        return np.nan
    if whiten:
        fb, score = aperiodic_residual(f, p, band)
    else:
        fb, score = f[m], np.log(p[m] + 1e-300)
    peaks, _ = signal.find_peaks(score)
    if len(peaks) == 0:  # monotonic inside band -> no respiratory peak
        return np.nan
    best = peaks[np.argmax(score[peaks])]
    # parabolic interpolation around the peak bin (on the log scale)
    if 0 < best < len(score) - 1:
        a, b, c = score[best - 1:best + 2]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return 60.0 * (fb[best] + shift * (fb[1] - fb[0]))
    return 60.0 * fb[best]


def fft_peak(x, fs, band=DEFAULT_BAND, nfft=NFFT, whiten=True):
    """Hann-windowed periodogram of the whole window (FTS in Charlton et al.)."""
    x = np.asarray(x, dtype=float)
    f, p = signal.periodogram(x - np.mean(x), fs=fs, window="hann", nfft=max(nfft, len(x)))
    return _peak_rate(f, p, band, whiten), (f, p)


def welch_peak(x, fs, band=DEFAULT_BAND, segment_s=None, nfft=NFFT, whiten=True):
    """Welch periodogram (WCH). Default segment = half the window, 50 % overlap.

    Frequency resolution is ~1/segment_s Hz, so short segments blur nearby
    rates; with a 60-s window the default 30-s segment resolves ~2 bpm.
    """
    x = np.asarray(x, dtype=float)
    nperseg = len(x) // 2 if segment_s is None else min(len(x), int(segment_s * fs))
    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=max(nperseg, 8),
                        noverlap=max(nperseg, 8) // 2, nfft=max(nfft, nperseg))
    return _peak_rate(f, p, band, whiten), (f, p)


def ar_burg(x, order):
    """Burg estimate of AR coefficients.

    Returns ``(a, noise_var)`` with ``a = [1, a1, ..., ap]`` such that
    ``x[n] + a1 x[n-1] + ... + ap x[n-p] = e[n]``.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    f = x[1:].copy()   # forward prediction errors  f_{m-1}(n),   n = m..N-1
    b = x[:-1].copy()  # backward prediction errors b_{m-1}(n-1), n = m..N-1
    a = np.array([1.0])
    e = np.dot(x, x) / len(x)
    for _ in range(order):
        den = np.dot(f, f) + np.dot(b, b)
        k = -2.0 * np.dot(f, b) / den if den > 0 else 0.0
        a_ext = np.concatenate([a, [0.0]])
        a = a_ext + k * a_ext[::-1]
        e *= 1.0 - k * k
        f, b = f[1:] + k * b[1:], b[:-1] + k * f[:-1]
    return a, e


def ar_psd(a, noise_var, fs, nfft=NFFT):
    """One-sided power spectral density of an AR model."""
    w, h = signal.freqz([1.0], a, worN=nfft // 2 + 1, fs=fs)
    return w, noise_var / fs * np.abs(h) ** 2


def ar_spectral(x, fs, band=DEFAULT_BAND, order=8, nfft=NFFT, whiten=True):
    """Peak of the Burg AR spectrum at a single model order (ARS)."""
    a, e = ar_burg(x, order)
    f, p = ar_psd(a, e, fs, nfft)
    return _peak_rate(f, p, band, whiten), (f, p)


def ar_median(x, fs, band=DEFAULT_BAND, orders=range(2, 21), nfft=NFFT, whiten=True):
    """Peak of the median of AR spectra over model orders 2-20 (ARM), which
    removes the sensitivity of a single AR spectrum to the order choice.
    Spectra are normalised to unit area before taking the median."""
    spectra = []
    for p in orders:
        if p >= len(x) // 2:
            break
        a, e = ar_burg(x, p)
        f, psd = ar_psd(a, e, fs, nfft)
        area = np.trapezoid(psd, f)
        if np.isfinite(area) and area > 0:  # flat input -> zero spectrum
            spectra.append(psd / area)
    if not spectra:
        return np.nan, (None, None)
    p_med = np.median(np.vstack(spectra), axis=0)
    return _peak_rate(f, p_med, band, whiten), (f, p_med)


def ar_pole(x, fs, band=DEFAULT_BAND, order=8, whiten=True):
    """AR pole method (ARP): the rate is the frequency of one AR pole inside
    ``band``; a pole ``z = r e^{i w}`` corresponds to ``w fs / (2 pi)`` Hz.

    Published rule (``whiten=False``): the pole of highest magnitude. That
    pole is usually the very-low-frequency background (see module docstring),
    so by default the pole whose frequency stands highest above the power-law
    fit of the model's own spectrum is chosen instead.
    Returns ``(rr_bpm, poles)``.
    """
    a, e = ar_burg(x, order)
    poles = np.roots(a)
    freqs = np.angle(poles) * fs / (2 * np.pi)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(m):
        return np.nan, poles
    if not whiten:
        return 60.0 * freqs[m][np.argmax(np.abs(poles[m]))], poles
    f, p = ar_psd(a, e, fs)
    fb, resid = aperiodic_residual(f, p, band)
    score = np.interp(freqs[m], fb, resid)
    return 60.0 * freqs[m][np.argmax(score)], poles
