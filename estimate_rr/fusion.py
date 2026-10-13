"""Fusion of respiratory-rate estimates from several respiratory signals."""
import numpy as np

SMART_FUSION_MAX_SD = 4.0  # bpm, Karlen et al. 2013


def smart_fusion(estimates, max_sd=SMART_FUSION_MAX_SD):
    """Smart fusion (Karlen et al. 2013, IEEE TBME 60:1946).

    The estimates from BW, AM and FM are averaged only if they agree (standard
    deviation <= ``max_sd`` bpm); otherwise the window is rejected (NaN). All
    three estimates are required, as in the original method.
    """
    est = np.asarray([e for e in estimates], dtype=float)
    if len(est) < 3 or np.any(~np.isfinite(est)):
        return np.nan
    return float(np.mean(est)) if np.std(est) <= max_sd else np.nan


def median_fusion(estimates):
    """Median of the available (finite) estimates."""
    est = np.asarray(estimates, dtype=float)
    est = est[np.isfinite(est)]
    return float(np.median(est)) if len(est) else np.nan
