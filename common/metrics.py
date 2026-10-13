"""Agreement metrics for respiratory-rate estimates (breaths/min).

Reported as in the RR-estimation literature (e.g. Charlton et al. 2016):
MAE, RMSE, bias and 95 % limits of agreement (Bland-Altman), share of
estimates within +/-2 and +/-3 bpm, and coverage (share of windows for which
the method produced an estimate -- methods with quality rejection such as
smart fusion trade coverage for accuracy, so both must be reported).
"""
import numpy as np
import pandas as pd


def rr_metrics(y_true, y_pred, groups=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    out = {"n": int(np.isfinite(y_true).sum()), "coverage": float(m.mean()) if len(m) else np.nan}
    if m.sum() < 2:
        return out
    err = y_pred[m] - y_true[m]
    sd = float(np.std(err, ddof=1))
    out.update({
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "loa_low": float(np.mean(err) - 1.96 * sd),
        "loa_high": float(np.mean(err) + 1.96 * sd),
        "within_2bpm": float(np.mean(np.abs(err) <= 2)),
        "within_3bpm": float(np.mean(np.abs(err) <= 3)),
        "pearson_r": float(np.corrcoef(y_true[m], y_pred[m])[0, 1]),
    })
    if groups is not None:
        # every patient counts once, however long their recording
        g = pd.Series(np.abs(err)).groupby(np.asarray(groups)[m]).mean()
        out["patient_macro_mae"] = float(g.mean())
        out["n_patients"] = int(len(g))
    return out


def metrics_by_range(y_true, y_pred, edges=(0, 12, 20, 30, 40, 1000)):
    """Metrics per reference-RR range; low and high rates are the clinically
    important ones and are rare, so overall MAE hides failures there."""
    y_true = np.asarray(y_true, dtype=float)
    rows = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_true >= lo) & (y_true < hi)
        if m.any():
            rows[f"{lo}-{hi}"] = rr_metrics(y_true[m], np.asarray(y_pred, dtype=float)[m])
    return rows
