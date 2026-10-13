"""Extraction of respiratory signals from the ECG and PPG.

Breathing modulates both signals in three ways (Charlton et al. 2016, Physiol
Meas 37:610; Karlen et al. 2013, IEEE TBME 60:1946):

* BW  - baseline wander (RIIV for PPG): mean of each beat's peak and trough
* AM  - amplitude modulation (RIAV): peak minus trough amplitude per beat
* FM  - frequency modulation (RIFV, i.e. respiratory sinus arrhythmia):
        inter-beat intervals

Each beat-wise series is interpolated onto a uniform grid (4 Hz by default),
linearly detrended and band-passed to the respiratory band. A direct
respiratory waveform (e.g. thoracic impedance) only needs the last two steps,
see :func:`respiratory_waveform`.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import detrend

from preprocess.beats import (detect_ecg_rpeaks, detect_ppg_beats,
                              heart_rate_summary, valid_beat_mask)
from preprocess.filters import bandpass, interpolate_uniform, lowpass, resample

RESP_FS = 4.0
# 4-60 breaths/min; the upper limit covers infants and tachypnoeic children.
DEFAULT_BAND = (4 / 60.0, 60 / 60.0)
MODULATIONS = ("bw", "am", "fm")
# "bwf": baseline wander obtained by band-pass filtering the raw signal instead
# of sampling it once per beat. Beat-wise series can only represent breathing
# slower than HR/2; the filtered version has no such limit but must stay below
# the heart rate, so its upper cut-off is set to BWF_HR_FRACTION x HR.
EXTRA_SIGNALS = ("bwf",)
BWF_HR_FRACTION = 0.8
MIN_BEATS = 8


@dataclass
class RespSignals:
    fs: float
    signals: dict = field(default_factory=dict)  # name -> uniform respiratory signal
    n_beats: int = 0
    hr: float = np.nan                 # median heart rate, bpm
    beat_reject_frac: float = 1.0      # fraction of beats rejected as implausible

    @property
    def ok(self):
        return bool(self.signals)


def _hz(hr_bpm):
    return hr_bpm / 60.0 if np.isfinite(hr_bpm) else 0.0


def _finish(y, fs, band):
    y = detrend(np.asarray(y, dtype=float))
    return bandpass(y, fs, band[0], band[1], order=2)


def prewhiten(y, fs, band=DEFAULT_BAND, strength=0.5):
    """Partially flatten the 1/f background of a respiratory signal.

    The power spectrum over ``band`` is fitted with ``P ~ f^b``; multiplying
    the spectrum by ``f^(-strength * b / 2)`` (zero outside ``band``) removes a
    fraction ``strength`` of that slope (1 = fully white). Very-low-frequency
    swings then no longer dominate thresholds and zero crossings, while a
    strength below 1 avoids amplifying high-frequency noise into spurious
    breaths. Used by the CtA / PKS / ZeX detectors (see ``estimate_rr``).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3:
        return y
    slope = np.polyfit(np.log(f[m]), np.log(np.abs(spec[m]) ** 2 + 1e-300), 1)[0]
    gain = np.zeros_like(f)
    gain[m] = (f[m] / f[m][0]) ** (-strength * slope / 2)
    return np.fft.irfft(spec * gain, n)


def respiratory_waveform(resp, fs, band=DEFAULT_BAND, resp_fs=RESP_FS):
    """Condition a direct respiratory waveform (impedance, flow, capnogram)."""
    resp = np.asarray(resp, dtype=float)
    y = lowpass(resp, fs, min(2.0, 0.45 * fs), order=4) if fs > 2 * resp_fs else resp
    return _finish(resample(y, fs, resp_fs), resp_fs, band)


def beat_features(x, fs, signal_type):
    """Locate beats and read per-beat features.

    Returns ``(beat_times, features)`` where ``features`` maps modulation name to
    a ``(times, values)`` pair. Invalid beats are already removed.
    """
    signal_type = signal_type.lower()
    x = np.asarray(x, dtype=float)
    if signal_type == "ppg":
        peaks, onsets, _ = detect_ppg_beats(x, fs)
        # Features are read from a signal that still holds the baseline (BW)
        xs = lowpass(x, fs, min(8.0, 0.45 * fs))
        troughs = onsets
    elif signal_type == "ecg":
        peaks, _, polarity = detect_ecg_rpeaks(x, fs)
        xs = polarity * lowpass(x, fs, min(40.0, 0.45 * fs))
        # Q/S-wave trough: minimum in the 100 ms before each R peak
        w = max(int(0.1 * fs), 1)
        troughs = np.array([max(p - w, 0) + np.argmin(xs[max(p - w, 0):p + 1]) for p in peaks], dtype=int)
    else:
        raise ValueError(f"signal_type must be 'ppg' or 'ecg', got {signal_type!r}")

    t = peaks / fs
    if len(peaks) < MIN_BEATS:
        return t, {}
    ok = valid_beat_mask(t)
    feats = {
        "bw": (t[ok], 0.5 * (xs[peaks] + xs[troughs])[ok]),
        "am": (t[ok], (xs[peaks] - xs[troughs])[ok]),
    }
    # FM: interval ending at beat i, valid only if both beats i-1 and i are valid
    ibi_ok = ok[1:] & ok[:-1]
    feats["fm"] = (t[1:][ibi_ok], np.diff(t)[ibi_ok])
    return t, feats


def extract_resp_signals(x, fs, signal_type, band=DEFAULT_BAND, resp_fs=RESP_FS,
                         modulations=MODULATIONS):
    """Extract respiratory signals (any of ``MODULATIONS + EXTRA_SIGNALS``) from
    a PPG or ECG segment.

    All returned signals share the same uniform time grid covering the segment.
    """
    beat_t, feats = beat_features(x, fs, signal_type)
    hr, rej = heart_rate_summary(beat_t)
    out = RespSignals(fs=resp_fs, n_beats=len(beat_t), hr=hr, beat_reject_frac=rej)
    if not feats:
        return out
    duration = len(x) / fs
    for name in modulations:
        if name == "bwf":
            high = min(band[1], BWF_HR_FRACTION * _hz(hr))
            if high > band[0] * 1.5:
                y = resample(lowpass(np.asarray(x, dtype=float), fs, min(2.0, 0.45 * fs)), fs, resp_fs)
                out.signals[name] = _finish(y[:int(duration * resp_fs)], resp_fs, (band[0], high))
            continue
        t, v = feats[name]
        if len(t) < MIN_BEATS:
            continue
        _, yu = interpolate_uniform(t, v, resp_fs, t_start=0.0, t_end=duration)
        out.signals[name] = _finish(yu, resp_fs, band)
    return out
