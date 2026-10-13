"""Synthetic ECG/PPG/respiration with known respiratory rate, for tests."""
import numpy as np


def _beat_times(duration, hr_bpm, rr_bpm, rsa=0.06, rng=None):
    """Beat times with respiratory sinus arrhythmia (instantaneous HR modulated
    at the respiratory frequency)."""
    t, times = 0.0, []
    fr = rr_bpm / 60.0
    while t < duration:
        inst_hr = hr_bpm * (1 + rsa * np.sin(2 * np.pi * fr * t))
        t += 60.0 / inst_hr
        if rng is not None:
            t += rng.normal(0, 0.003)
        times.append(t)
    return np.array(times[:-1])


def synth_resp(rr_bpm, fs=25, duration=60, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    phase = 2 * np.pi * rr_bpm / 60 * t + 0.2 * np.sin(2 * np.pi * 0.01 * t)
    return np.sin(phase) + 0.3 * np.sin(2 * phase) * 0.3 + noise * rng.standard_normal(len(t))


def synth_ppg(rr_bpm, hr_bpm=80, fs=100, duration=60, noise=0.02, seed=0,
              bw=0.3, am=0.15, rsa=0.06):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    fr = rr_bpm / 60.0
    resp = np.sin(2 * np.pi * fr * t)
    x = np.zeros_like(t)
    for tb in _beat_times(duration, hr_bpm, rr_bpm, rsa, rng):
        amp = 1 + am * np.sin(2 * np.pi * fr * tb)
        x += amp * (np.exp(-0.5 * ((t - tb - 0.15) / 0.06) ** 2)
                    + 0.4 * np.exp(-0.5 * ((t - tb - 0.4) / 0.08) ** 2))
    return x + bw * resp + noise * rng.standard_normal(len(t))


def synth_ecg(rr_bpm, hr_bpm=75, fs=250, duration=60, noise=0.02, seed=0,
              bw=0.2, am=0.15, rsa=0.06):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    fr = rr_bpm / 60.0
    x = np.zeros_like(t)
    for tb in _beat_times(duration, hr_bpm, rr_bpm, rsa, rng):
        amp = 1 + am * np.sin(2 * np.pi * fr * tb)
        x += amp * np.exp(-0.5 * ((t - tb) / 0.012) ** 2)              # R
        x -= 0.15 * np.exp(-0.5 * ((t - tb + 0.03) / 0.01) ** 2)       # Q
        x += 0.25 * np.exp(-0.5 * ((t - tb - 0.25) / 0.05) ** 2)       # T
    return x + bw * np.sin(2 * np.pi * fr * t) + noise * rng.standard_normal(len(t))
