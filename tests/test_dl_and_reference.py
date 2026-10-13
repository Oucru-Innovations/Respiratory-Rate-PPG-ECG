"""Run from the repo root:  python -m unittest discover -s tests -t ."""
import unittest

import numpy as np

from common.metrics import rr_metrics
from common.splits import patient_split
from data_prep.reference import reference_rr
from tests.synthetic import synth_ecg, synth_ppg, synth_resp

try:
    import torch
except ImportError:  # torch-free environments still run the signal tests
    torch = None


class TestReference(unittest.TestCase):
    def test_clean_impedance(self):
        for rr in (8, 15, 26, 40):
            out = reference_rr(synth_resp(rr, fs=25, duration=60), 25)
            self.assertTrue(out["ref_ok"], out)
            self.assertAlmostEqual(out["ref_rr"], rr, delta=1.0)

    def test_flat_channel_rejected(self):
        self.assertFalse(reference_rr(np.zeros(1500), 25)["ref_ok"])

    def test_nan_rejected(self):
        x = synth_resp(15, fs=25)
        x[100] = np.nan
        self.assertFalse(reference_rr(x, 25)["ref_ok"])


class TestSegmentIds(unittest.TestCase):
    def test_exports_in_same_day_folder_get_distinct_ids(self):
        from data_prep.build_dataset import segment_id
        day = "RAW/01NVa-003-2014/18082020/Monitor/"
        names = ["01NVa-003-2014waves.asc", "01NVa-003-2014 2waves.asc", "01NVa-003-2014-1waves.asc"]
        ids = {segment_id(day + n, 0) for n in names}
        self.assertEqual(len(ids), 3)


class TestSplitsAndMetrics(unittest.TestCase):
    def test_patients_disjoint(self):
        ids = [f"P{i}" for i in range(23)] * 3
        s = patient_split(ids, n_folds=5, test_fold=2)
        self.assertFalse(set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"])
                         or set(s["val"]) & set(s["test"]))
        self.assertEqual(len(s["train"]) + len(s["val"]) + len(s["test"]), 23)

    def test_metrics(self):
        m = rr_metrics([10, 20, 30, 40], [11, 19, 30, np.nan])
        self.assertAlmostEqual(m["coverage"], 0.75)
        self.assertAlmostEqual(m["mae"], 2 / 3)


@unittest.skipIf(torch is None, "torch not installed")
class TestFeaturesAndModel(unittest.TestCase):
    def test_modulation_spectrogram_peak_at_rr(self):
        from dl.features import SpecConfig, feature_shape, modulation_spectrogram
        cfg = SpecConfig()
        for rr in (10, 20, 30):
            x = modulation_spectrogram(synth_ppg(rr, hr_bpm=85), 100, "ppg", cfg)
            self.assertEqual(x.shape, feature_shape(60, cfg))
            freqs = np.fft.rfftfreq(cfg.nfft_modulation, 1 / 4.0)
            freqs = freqs[(freqs >= cfg.band_low_hz) & (freqs <= cfg.band_high_hz)]
            peak_bpm = 60 * freqs[np.argmax(x.mean(axis=(0, 2)))]
            self.assertAlmostEqual(peak_bpm, rr, delta=2.0)

    def test_raw_spectrogram_shape(self):
        from dl.features import SpecConfig, feature_shape, raw_spectrogram
        cfg = SpecConfig(kind="raw")
        x = raw_spectrogram(synth_ecg(15), 250, cfg)
        self.assertEqual(x.shape, feature_shape(60, cfg))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_label_distribution(self):
        from dl.model import LabelDistribution, SpectroCNN
        ldl = LabelDistribution()
        y = torch.tensor([6.0, 18.3, 45.0, 80.0])
        t = ldl.target(y)
        torch.testing.assert_close(t.sum(1), torch.ones(4))
        # decoding a (log-)target recovers in-range labels ...
        dec = ldl.decode(torch.log(t + 1e-12))
        torch.testing.assert_close(dec[:3], torch.tensor([6.0, 18.3, 45.0]), atol=0.3, rtol=0)
        # ... while at the range edge the truncated Gaussian pulls the mean inward
        self.assertTrue(58.5 < dec[3] < 60.0)
        model = SpectroCNN(3, 60, ldl.n_bins)
        self.assertEqual(model(torch.randn(2, 3, 60, 45)).shape, (2, ldl.n_bins))


if __name__ == "__main__":
    unittest.main()
