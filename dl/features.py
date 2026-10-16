"""Spectrogram inputs for the deep-learning RR estimator.

Four input kinds (``modwave`` = the BW/AM/FM waveforms themselves, for BreathNet;
``modwave_hr`` adds the instantaneous heart rate as a fourth channel, for BreathNet-2's
heart-rate prior):

``modulation`` (default)
    BW, AM and FM respiratory signals are extracted from the PPG/ECG (see
    :mod:`estimate_rr.resp_signals`), and the log-power STFT of each one,
    cropped to the respiratory band, forms one image channel -> (3, F, T).
    The breathing frequency appears directly as a horizontal ridge.

``raw``
    Log-power STFT of the band-limited raw signal from 0 to 3 Hz -> (1, F, T).
    Respiration shows as baseline wander below 1 Hz and as AM/FM sidebands at
    f_heart +/- f_resp; a long STFT window (16 s by default) is needed to
    resolve those sidebands, so the time axis is coarse.

Why not the previous design: a 2.56-s STFT over 0-50 Hz puts respiration in
~2 % of the image rows, the 1.28-s hop aliases rates above 23 bpm, and the
0.5-3.5 Hz pre-filter removes the baseline-wander component altogether.

Every channel is z-scored per sample (log power), so absolute signal scale
does not matter and no image conversion / clipping is involved.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from estimate_rr.resp_signals import DEFAULT_BAND, MODULATIONS, RESP_FS, extract_resp_signals
from preprocess.filters import bandpass, resample

EPS = 1e-12


@dataclass
class SpecConfig:
    kind: str = "modulation"     # "modulation", "raw", "modwave" or "modwave_hr"
    stft_window_s: float = 16.0  # frequency resolution ~ 1 / stft_window_s Hz
    hop_s: float = 1.0
    nfft_modulation: int = 256   # at 4 Hz -> 0.94 bpm bin spacing
    band_low_hz: float = DEFAULT_BAND[0]
    band_high_hz: float = DEFAULT_BAND[1]
    raw_fs: float = 10.0
    raw_fmax_hz: float = 3.0
    nfft_raw: int = 512          # at 10 Hz -> 1.17 bpm bin spacing

    def to_dict(self):
        return asdict(self)


def _log_spec(y, fs, cfg, nfft, fmin, fmax):
    nper = int(round(cfg.stft_window_s * fs))
    hop = max(int(round(cfg.hop_s * fs)), 1)
    f, _, s = signal.spectrogram(y, fs=fs, window="hann", nperseg=nper,
                                 noverlap=nper - hop, nfft=nfft,
                                 detrend="constant", scaling="density", mode="psd")
    keep = (f >= fmin) & (f <= fmax)
    ls = np.log(s[keep] + EPS)
    return f[keep], (ls - ls.mean()) / (ls.std() + 1e-6)


def modulation_spectrogram(x, fs, signal_type, cfg=SpecConfig()):
    """(3, F, T) array or ``None`` when any of BW/AM/FM could not be extracted."""
    rs = extract_resp_signals(x, fs, signal_type, band=(cfg.band_low_hz, cfg.band_high_hz))
    if any(m not in rs.signals for m in MODULATIONS):
        return None
    chans = [_log_spec(rs.signals[m], RESP_FS, cfg, cfg.nfft_modulation,
                       cfg.band_low_hz, cfg.band_high_hz)[1] for m in MODULATIONS]
    return np.stack(chans).astype(np.float32)


def raw_spectrogram(x, fs, cfg=SpecConfig()):
    """(1, F, T) array of the raw signal's log-power STFT from ~0.05 to 3 Hz."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.std(x) == 0:
        return None
    y = resample(bandpass(x, fs, 0.05, min(4.0, 0.45 * cfg.raw_fs), order=2), fs, cfg.raw_fs)
    _, s = _log_spec(y, cfg.raw_fs, cfg, cfg.nfft_raw, 0.0, cfg.raw_fmax_hz)
    return s[None].astype(np.float32)


def modulation_waveforms(x, fs, signal_type, cfg=SpecConfig()):
    """(3, N) array: the BW, AM and FM respiratory signals themselves at 4 Hz,
    each z-scored, or ``None`` when any could not be extracted. Input of the
    physiology-inspired :class:`dl.model.BreathNet`, which does its own
    frequency analysis with a learnable resonator bank."""
    rs = extract_resp_signals(x, fs, signal_type, band=(cfg.band_low_hz, cfg.band_high_hz))
    if any(m not in rs.signals for m in MODULATIONS):
        return None
    sig = np.stack([rs.signals[m] for m in MODULATIONS])
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-9)
    if cfg.kind == "modwave_hr":
        # heart rate in bpm / 100, *not* normalised: its level is the information
        if rs.hr_series is None or len(rs.hr_series) < sig.shape[1]:
            return None
        sig = np.concatenate([sig, rs.hr_series[None, :sig.shape[1]] / 100.0])
    return sig.astype(np.float32)


def make_features(x, fs, signal_type, cfg=SpecConfig()):
    if cfg.kind in ("modwave", "modwave_hr"):
        return modulation_waveforms(x, fs, signal_type, cfg)
    if cfg.kind == "modulation":
        return modulation_spectrogram(x, fs, signal_type, cfg)
    if cfg.kind == "raw":
        return raw_spectrogram(x, fs, cfg)
    raise ValueError(f"unknown feature kind {cfg.kind!r}")


def feature_shape(window_s, cfg=SpecConfig()):
    """Shape produced for a window of ``window_s`` seconds (used to allocate
    the on-disk array before computing)."""
    if cfg.kind in ("modwave", "modwave_hr"):
        return (len(MODULATIONS) + (cfg.kind == "modwave_hr"), int(window_s * RESP_FS))
    fs = RESP_FS if cfg.kind == "modulation" else cfg.raw_fs
    probe = np.random.default_rng(0).standard_normal(int(window_s * fs))
    if cfg.kind == "modulation":
        _, s = _log_spec(probe, fs, cfg, cfg.nfft_modulation, cfg.band_low_hz, cfg.band_high_hz)
        return (len(MODULATIONS),) + s.shape
    _, s = _log_spec(probe, fs, cfg, cfg.nfft_raw, 0.0, cfg.raw_fmax_hz)
    return (1,) + s.shape
