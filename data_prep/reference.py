"""Reference respiratory rate from a direct respiratory waveform (thoracic
impedance on the GE monitor).

The label for a window is the breath-by-breath rate ``60 / mean(breath
interval)`` from count-adv breath detection on the conditioned impedance
signal -- the same definition as the estimates being evaluated, applied to the
gold-standard channel over *exactly* the input window.

Impedance is not perfect (motion artefact, cardiogenic oscillation, apnoea), so
several quality indicators are stored and ``ref_ok`` combines them. Thresholds
are heuristics; all components are kept so they can be tuned without
rebuilding the dataset.
"""
import numpy as np

from data_prep.quality import amplitude_burst_ratio, flat_fraction, saturation_fraction
from estimate_rr.freq_domain import ar_median
from estimate_rr.resp_signals import DEFAULT_BAND, RESP_FS, respiratory_waveform
from estimate_rr.time_domain import count_adv

MIN_BREATH_INTERVALS = 3
MAX_INTERVAL_CV = 0.5
MAX_TIME_FREQ_DIFF = 3.0  # bpm, agreement between count-adv and AR spectrum
MAX_BURST_RATIO = 5.0


def reference_rr(resp, fs, rails=None, band=DEFAULT_BAND):
    """Return a dict with the reference RR and its quality indicators."""
    resp = np.asarray(resp, dtype=float)
    out = {"ref_rr": np.nan, "ref_rr_spec": np.nan, "ref_n_breaths": 0,
           "ref_interval_cv": np.nan, "ref_sat_frac": np.nan, "ref_flat_frac": np.nan,
           "ref_burst_ratio": np.nan, "ref_ok": False}
    if not np.all(np.isfinite(resp)):
        return out
    out["ref_sat_frac"] = saturation_fraction(resp, *rails) if rails else 0.0
    out["ref_flat_frac"] = flat_fraction(resp, fs)
    out["ref_burst_ratio"] = amplitude_burst_ratio(resp, fs)

    y = respiratory_waveform(resp, fs, band=band)
    rr, breaths = count_adv(y, RESP_FS)
    intervals = np.diff(breaths)
    out["ref_rr"] = rr
    out["ref_n_breaths"] = len(breaths)
    if len(intervals) >= 2:
        out["ref_interval_cv"] = float(np.std(intervals) / np.mean(intervals))
    # published (unwhitened) ARM: the impedance has a clear breathing peak, and
    # the stored dataset was built with this rule
    out["ref_rr_spec"] = ar_median(y, RESP_FS, band=band, whiten=False)[0]

    out["ref_ok"] = bool(
        np.isfinite(rr)
        and len(intervals) >= MIN_BREATH_INTERVALS
        and out["ref_interval_cv"] <= MAX_INTERVAL_CV
        and np.isfinite(out["ref_rr_spec"])
        and abs(rr - out["ref_rr_spec"]) <= MAX_TIME_FREQ_DIFF
        and out["ref_sat_frac"] < 0.01
        and out["ref_flat_frac"] < 0.1
        and out["ref_burst_ratio"] <= MAX_BURST_RATIO
    )
    return out
