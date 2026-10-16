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


@unittest.skipIf(torch is None, "torch not installed")
class TestBreathNet2(unittest.TestCase):
    def test_variants_forward(self):
        import torch
        from dl.model import BreathNet, BreathNet2, LabelDistribution
        torch.manual_seed(0)
        ldl = LabelDistribution()
        x = torch.randn(4, 4, 240)
        x[:, 3] = 0.85                                   # heart-rate channel: 85 bpm
        for variant in ("", "whiten", "coherent", "track", "hr", "conv", "whiten,coherent,track,hr"):
            m = BreathNet2(4, 240, 57, variant=variant, hr_channel=True)
            logits, d = m(x, return_details=True)
            self.assertEqual(tuple(logits.shape), (4, 57), variant)
            ldl.loss(logits, torch.full((4,), 18.0)).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), variant)
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(n(BreathNet2(3, 240, 57, variant="")), n(BreathNet(3, 240, 57)))
        self.assertLess(n(BreathNet2(3, 240, 57, variant="track")), 1000)   # attention gone
        with self.assertRaises(ValueError):
            BreathNet2(4, 240, 57, variant="track,conv", hr_channel=True)
        with self.assertRaises(ValueError):
            BreathNet2(3, 240, 57, variant="hr")                           # needs the HR channel

    def test_whitening_removes_power_law(self):
        import torch
        from dl.model import BreathNet2
        m = BreathNet2(3, 240, 57, variant="whiten")
        e = 0.7 - 1.3 * m.log_rate_c[None, None, :, None].expand(2, 3, 57, 5)
        self.assertLess(m._whiten(e).abs().max().item(), 1e-4)

    def test_tracking_posterior(self):
        import torch
        from dl.model import BreathNet2
        m = BreathNet2(3, 240, 57, variant="track")
        s = torch.zeros(1, 57, 60)
        s[0, 20] = 6.0                                   # evidence ridge at bin 20 in every frame
        out = m._track(s)
        self.assertEqual(out.argmax().item(), 20)
        self.assertAlmostEqual(torch.logsumexp(out, 1).item(), 0.0, places=4)   # a log-distribution

    def test_hr_prior_ceiling(self):
        import torch
        from dl.model import BreathNet2
        m = BreathNet2(4, 240, 57, variant="hr", hr_channel=True)
        prior = m._hr_prior(torch.tensor([60.0]))[0]     # rates 4..60 bpm; ceiling at HR/2 = 30
        self.assertGreater(prior[14], prior[50])         # 18 bpm plausible, 54 bpm above the ceiling


class TestBreathNet(unittest.TestCase):
    def test_untrained_prior_finds_the_rate(self):
        # the resonator bank + harmonic summation alone should already peak at
        # the breathing rate: output bin k is resonator k
        from dl.features import SpecConfig, make_features
        from dl.model import BreathNet, LabelDistribution
        ldl, m = LabelDistribution(), BreathNet(3, 240, 57).eval()
        for rr in (8, 18, 30, 45):
            x = make_features(synth_ppg(rr, hr_bpm=90), 100, "ppg", SpecConfig(kind="modwave"))
            with torch.no_grad():
                logits, det = m(torch.from_numpy(x)[None], return_details=True)
            self.assertLessEqual(abs(float(ldl.bins[logits.argmax()]) - rr), 1.5, rr)
            torch.testing.assert_close(det["pathway_weight"].sum(1), torch.ones(1))

    def test_shapes_and_gradients(self):
        from dl.model import BreathNet
        m = BreathNet(3, 240, 57)
        out = m(torch.randn(4, 3, 240))
        self.assertEqual(out.shape, (4, 57))
        out.sum().backward()
        self.assertIsNotNone(m.bank.log_sigma.grad)  # resonator bandwidths are learned
        self.assertEqual(m(torch.randn(2, 3, 120)).shape, (2, 57))  # 30-s windows


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

    def test_transformer_shapes_and_attention(self):
        from dl.model import SpectroTransformer, build_model
        for c, f, tokens in [(3, 60, "frame"), (1, 154, "frame"), (3, 60, "patch"), (1, 154, "patch")]:
            with self.subTest(c=c, f=f, tokens=tokens):
                m = SpectroTransformer(c, f, 57, tokens=tokens, d_model=32, depth=1).eval()
                out, attn = m(torch.randn(2, c, f, 45), return_attention=True)
                self.assertEqual(out.shape, (2, 57))
                torch.testing.assert_close(attn.sum(dim=(1, 2)), torch.ones(2))
                # 30-s windows give fewer frames; the same model must accept them
                self.assertEqual(m(torch.randn(2, c, f, 15)).shape, (2, 57))
        # every small architecture: right output shape on 60- and 30-s windows
        for arch in ("lstm", "gru", "unet", "mlp"):
            for c, f in ((3, 60), (1, 154)):
                with self.subTest(arch=arch, c=c, f=f):
                    m = build_model(arch, in_channels=c, n_freq=f, n_out=57).eval()
                    self.assertEqual(m(torch.randn(2, c, f, 45)).shape, (2, 57))
                    self.assertEqual(m(torch.randn(2, c, f, 15)).shape, (2, 57))
        # checkpoints written before `arch` existed still build a CNN
        self.assertEqual(type(build_model(in_channels=3, n_freq=60, n_out=57, width=8)).__name__, "SpectroCNN")


if __name__ == "__main__":
    unittest.main()
