"""Run from the repo root:  python -m unittest discover -s tests -t ."""
import unittest

import numpy as np
from scipy import signal

from estimate_rr import METHODS, estimate_rr
from estimate_rr.freq_domain import ar_burg
from estimate_rr.resp_signals import RESP_FS, respiratory_waveform
from preprocess.beats import detect_ecg_rpeaks, detect_ppg_beats
from tests.synthetic import synth_ecg, synth_ppg, synth_resp

RATES = [6, 10, 15, 20, 30, 45]


class TestBurg(unittest.TestCase):
    def test_recovers_ar2_coefficients(self):
        rng = np.random.default_rng(1)
        a_true = [1.0, -1.5, 0.8]
        x = signal.lfilter([1.0], a_true, rng.standard_normal(20000))
        a, _ = ar_burg(x, 2)
        np.testing.assert_allclose(a, a_true, atol=0.02)


class TestRespWaveform(unittest.TestCase):
    def test_every_method_on_clean_resp(self):
        for method in METHODS:
            for rr in RATES:
                with self.subTest(method=method, rr=rr):
                    est = estimate_rr(synth_resp(rr), 25, "resp", method).rr
                    self.assertTrue(np.isfinite(est))
                    self.assertAlmostEqual(est, rr, delta=1.0)

    def test_conditioning_output_rate(self):
        y = respiratory_waveform(synth_resp(15, fs=300, duration=30), 300)
        self.assertEqual(len(y), int(30 * RESP_FS))


class TestBeatDetection(unittest.TestCase):
    def test_ppg_beats(self):
        x = synth_ppg(15, hr_bpm=90, duration=60)
        peaks, onsets, _ = detect_ppg_beats(x, 100)
        self.assertAlmostEqual(60 / np.median(np.diff(peaks / 100)), 90, delta=4)
        self.assertTrue(np.all(onsets < peaks))

    def test_ecg_rpeaks_inverted_lead(self):
        for sign in (1, -1):
            x = sign * synth_ecg(15, hr_bpm=120, duration=60)
            peaks, _, polarity = detect_ecg_rpeaks(x, 250)
            self.assertEqual(polarity, sign)
            self.assertAlmostEqual(60 / np.median(np.diff(peaks / 250)), 120, delta=4)


class TestModulations(unittest.TestCase):
    def test_ppg_fused(self):
        for rr in [8, 12, 18, 24]:
            for method in ("count_adv", "welch", "ar_median"):
                with self.subTest(rr=rr, method=method):
                    res = estimate_rr(synth_ppg(rr, hr_bpm=80), 100, "ppg", method)
                    for name, est in res.per_signal.items():
                        self.assertAlmostEqual(est, rr, delta=1.5, msg=name)
                    self.assertAlmostEqual(res.rr, rr, delta=1.5)

    def test_ecg_fused(self):
        for rr in [8, 12, 18, 24]:
            with self.subTest(rr=rr):
                res = estimate_rr(synth_ecg(rr, hr_bpm=75), 250, "ecg", "count_adv")
                self.assertAlmostEqual(res.rr, rr, delta=1.5)

    def test_filtered_bw_beyond_half_heart_rate(self):
        # RR/HR = 0.67: beat-wise BW is aliased, filter-based BW is not
        from estimate_rr import rate_from_resp_signal
        from estimate_rr.resp_signals import extract_resp_signals
        rs = extract_resp_signals(synth_ppg(40, hr_bpm=60, am=0.0, rsa=0.0), 100, "ppg",
                                  modulations=("bw", "bwf"))
        self.assertAlmostEqual(rate_from_resp_signal(rs.signals["bwf"], rs.fs, "welch"), 40, delta=1.5)
        self.assertGreater(abs(rate_from_resp_signal(rs.signals["bw"], rs.fs, "welch") - 40), 5)

    def test_corrections_ignore_very_low_frequency_background(self):
        # breathing at 20 bpm under a stronger Mayer-wave (~6 bpm) + pink-noise
        # background, as in ECG/PPG-derived signals: the published "largest peak"
        # rules lock onto the background, the corrected ones find the breathing
        from estimate_rr import rate_from_resp_signal
        from estimate_rr.resp_signals import _finish

        def derived(seed, fs=4.0, n=240):
            rng = np.random.default_rng(seed)
            t = np.arange(n) / fs
            f = np.maximum(np.fft.rfftfreq(n, 1 / fs), 1 / 60)
            pink = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / f ** 0.5, n)
            mayer = 0.9 * np.sin(2 * np.pi * (0.1 + 0.01 * rng.standard_normal()) * t + rng.uniform(0, 6.28))
            x = 0.5 * np.sin(2 * np.pi * 20 / 60 * t) + mayer + 0.5 * pink / pink.std()
            return _finish(x + 0.1 * rng.standard_normal(n), fs, (4 / 60, 1.0))

        for method in ("fft", "welch", "ar_median", "zero_crossing"):
            with self.subTest(method=method):
                err = {c: np.nanmedian([abs(rate_from_resp_signal(derived(s), 4.0, method, corrected=c) - 20)
                                        for s in range(20)]) for c in (True, False)}
                self.assertLess(err[True], 1.5)
                self.assertGreater(err[False], 4.0)

    def test_smart_fusion_rejects_disagreement(self):
        from estimate_rr.fusion import smart_fusion
        self.assertTrue(np.isnan(smart_fusion([10, 15, 25])))
        self.assertAlmostEqual(smart_fusion([14, 15, 16]), 15)
        self.assertTrue(np.isnan(smart_fusion([14, 15])))


if __name__ == "__main__":
    unittest.main()
