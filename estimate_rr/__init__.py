"""Respiratory-rate estimation from ECG, PPG or a direct respiratory waveform.

Pipeline (Charlton et al. 2016/2018):

1. extract respiratory signals (BW, AM, FM) from beat-detected ECG/PPG, or
   condition a direct respiratory waveform        -> :mod:`resp_signals`
2. estimate RR from each signal with a time-domain breath detector or a
   frequency-domain spectral method              -> :mod:`time_domain`,
                                                     :mod:`freq_domain`
3. fuse the per-signal estimates                 -> :mod:`fusion`

Corrections to the published methods (``corrected=True``, the default)
------------------------------------------------------------------------
ECG/PPG-derived respiratory signals carry a 1/f very-low-frequency background
that the published rules mistake for breathing (on 01NVa training patients the
spectral methods under-estimated by ~10 bpm, CtA/PKS/ZeX by 2-5 bpm). Chosen
on fold-0 *training* patients only, identically for PPG and ECG:

* FTS, WCH, ARS, ARM: peak picked above a fitted power-law background
  (:func:`freq_domain.aperiodic_residual`);
* ARP: pole with the strongest whitened power instead of the largest modulus;
* CtA, PKS, ZeX: signal half pre-whitened first (:func:`resp_signals.prewhiten`);
* CtO: unchanged -- its 0.2 x Q3 peak threshold is already robust.

``corrected=False`` runs the methods exactly as published.

Example
-------
>>> from estimate_rr import estimate_rr
>>> res = estimate_rr(ppg, fs=100, signal_type="ppg", method="count_adv")
>>> res.rr, res.per_signal
"""
from dataclasses import dataclass, field

import numpy as np

from estimate_rr import freq_domain, time_domain
from estimate_rr.fusion import median_fusion, smart_fusion
from estimate_rr.resp_signals import (DEFAULT_BAND, MODULATIONS, RESP_FS,
                                      extract_resp_signals, prewhiten,
                                      respiratory_waveform)

# name -> (function, abbreviation used in Charlton et al.)
METHODS = {
    "count_orig": (time_domain.count_orig, "CtO"),
    "count_adv": (time_domain.count_adv, "CtA"),
    "peak_detection": (time_domain.peak_detection, "PKS"),
    "zero_crossing": (time_domain.zero_crossing, "ZeX"),
    "fft": (freq_domain.fft_peak, "FTS"),
    "welch": (freq_domain.welch_peak, "WCH"),
    "ar_spectral": (freq_domain.ar_spectral, "ARS"),
    "ar_median": (freq_domain.ar_median, "ARM"),
    "ar_pole": (freq_domain.ar_pole, "ARP"),
}
FUSIONS = {"smart": smart_fusion, "median": median_fusion}
FREQ_METHODS = {"fft", "welch", "ar_spectral", "ar_median", "ar_pole"}
# pre-whitening strength applied before each time-domain detector when corrected
PREWHITEN = {"count_adv": 0.5, "peak_detection": 0.5, "zero_crossing": 0.5}


@dataclass
class RRResult:
    rr: float                                        # fused estimate, bpm (NaN = rejected)
    per_signal: dict = field(default_factory=dict)   # e.g. {"bw": 14.8, "am": 15.1, "fm": 15.3}
    hr: float = np.nan
    beat_reject_frac: float = np.nan


def rate_from_resp_signal(y, fs, method, band=DEFAULT_BAND, corrected=True):
    """Apply one estimation method to an already conditioned respiratory signal."""
    fn, _ = METHODS[method]
    if method in FREQ_METHODS:
        return fn(y, fs, band=band, whiten=corrected)[0]
    if corrected and method in PREWHITEN:
        y = prewhiten(y, fs, band, PREWHITEN[method])
    return fn(y, fs)[0]


def estimate_rr(x, fs, signal_type="ppg", method="count_adv", fusion="smart",
                band=DEFAULT_BAND, modulations=MODULATIONS, corrected=True):
    """Estimate respiratory rate (breaths/min) from one window of signal.

    Parameters
    ----------
    x : array-like
        Raw signal window. Literature commonly uses 32-60 s windows; shorter
        windows hold too few breaths at low rates.
    fs : float
        Sampling frequency of ``x`` in Hz.
    signal_type : {"ppg", "ecg", "resp"}
        ``"resp"`` means ``x`` is itself a respiratory waveform (impedance,
        flow, capnography) and no modulation extraction is done.
    method : str
        One of :data:`METHODS`.
    fusion : {"smart", "median"}
        How BW/AM/FM estimates are combined (ignored for ``"resp"``).
    corrected : bool
        Apply the very-low-frequency corrections (module docstring); False
        reproduces the published methods.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    if signal_type == "resp":
        y = respiratory_waveform(x, fs, band=band)
        rr = rate_from_resp_signal(y, RESP_FS, method, band, corrected)
        return RRResult(rr=rr, per_signal={"resp": rr})

    rs = extract_resp_signals(x, fs, signal_type, band=band, modulations=modulations)
    per = {name: rate_from_resp_signal(y, rs.fs, method, band, corrected) for name, y in rs.signals.items()}
    fused = FUSIONS[fusion]([per.get(m, np.nan) for m in modulations]) if per else np.nan
    return RRResult(rr=fused, per_signal=per, hr=rs.hr, beat_reject_frac=rs.beat_reject_frac)


__all__ = ["estimate_rr", "rate_from_resp_signal", "extract_resp_signals",
           "respiratory_waveform", "prewhiten", "METHODS", "FUSIONS", "RRResult",
           "DEFAULT_BAND", "RESP_FS", "MODULATIONS"]
