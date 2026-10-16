"""Wearable alignment: recover a known offset, clock drift and restart from beats."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_prep.wearable import coarse_candidates, fine_align, hr_series, read_edf_ecg


def synthetic_beats(duration_s, seed=0):
    """Beat times with slow HR wander and respiratory sinus arrhythmia whose rate
    itself wanders, so every stretch is distinguishable from every other."""
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    hr_walk, rr = 75.0, 15.0
    while t < duration_s:
        hr_walk = np.clip(hr_walk + rng.normal(0, 0.15), 55, 110)
        rr = np.clip(rr + rng.normal(0, 0.05), 8, 30)
        hr = hr_walk + 4 * np.sin(2 * np.pi * rr / 60 * t)
        t += 60.0 / hr + rng.normal(0, 0.004)
        out.append(t)
    return np.array(out)


class TestAlignment(unittest.TestCase):
    def test_recovers_offset_and_drift(self):
        mon = 1.6e9 + synthetic_beats(6 * 3600)                   # monitor beats, epoch seconds
        offset, drift = 1.6e9 + 1234.5, 1500e-6                   # monitor = offset + (1 + drift) * wearable
        sel = (mon > 1.6e9 + 1500) & (mon < 1.6e9 + 5.5 * 3600)
        wear = (mon[sel] - offset) / (1 + drift)
        T0 = np.floor(mon[0])
        hm = hr_series(mon - T0, mon[-1] - T0)[1]
        hw = hr_series(wear, wear[-1])[1]
        cands = coarse_candidates(hw, hm, len(hm) + len(hw), k=3, drifts_ppm=np.arange(-3000, 3001, 250))
        self.assertLess(abs(T0 + cands[0]["lag"] - offset), 30)
        res = fine_align(wear, mon, T0 + cands[0]["lag"], cands[0]["drift_ppm"])
        acc = [p for p in res["pieces"] if p["accepted"]]
        self.assertEqual(len(acc), 1)
        self.assertLess(abs(acc[0]["lag0_s"] - offset), 0.5)
        self.assertLess(abs(acc[0]["drift_ppm"] - 1500), 50)

    def test_wrong_offset_is_rejected(self):
        mon = 1.6e9 + synthetic_beats(4 * 3600, seed=1)
        wear = mon[(mon > 1.6e9 + 600) & (mon < 1.6e9 + 3.5 * 3600)] - 1.6e9 - 100.0
        res = fine_align(wear, mon, 1.6e9 + 100.0 + 1800.0)     # 30 min off: outside the search window
        self.assertFalse(any(p["accepted"] for p in res["pieces"]))

    def test_restart_gives_two_pieces(self):
        mon = 1.6e9 + synthetic_beats(8 * 3600, seed=2)
        a = mon[(mon > 1.6e9 + 600) & (mon < 1.6e9 + 3.5 * 3600)] - 1.6e9 - 100.0
        b = mon[(mon > 1.6e9 + 3.6 * 3600) & (mon < 1.6e9 + 7.5 * 3600)] - 1.6e9 - 100.0 - 360.0   # 6 min lost
        res = fine_align(np.r_[a, b], mon, 1.6e9 + 100.0)
        lags = sorted(round(p["lag0_s"]) for p in res["pieces"] if p["accepted"])
        self.assertIn(round(1.6e9 + 100.0), lags)


class TestEdfReader(unittest.TestCase):
    def test_round_trip(self):
        fs, n_rec, spr = 256, 8, 64
        x = (1000 * np.sin(2 * np.pi * 1.2 * np.arange(n_rec * spr) / fs)).astype(np.int16)
        def field(v, w):
            return str(v).ljust(w)[:w]
        head = (field("0", 8) + field("X", 80) + field("Startdate 01-JAN-2000", 80) + "01.01.00" + "00.00.46"
                + field(256 * 3, 8) + field("EDF+C", 44) + field(n_rec, 8) + field("0.25", 8) + field(2, 4))
        sigs = ["ECG Channel 1", "EDF Annotations"]
        sh = "".join(field(l, 16) for l in sigs) + field("", 80) * 2 + field("uV", 8) * 2
        sh += field(-32768, 8) * 2 + field(32767, 8) * 2 + field(-32768, 8) * 2 + field(32767, 8) * 2
        sh += field("", 80) * 2 + field(spr, 8) * 2 + field("", 32) * 2
        recs = np.stack([np.r_[x[i * spr:(i + 1) * spr], np.zeros(spr, np.int16)] for i in range(n_rec)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.edf"
            p.write_bytes(head.encode("latin1") + sh.encode("latin1") + recs.astype("<i2").tobytes())
            rec = read_edf_ecg(p)
        self.assertEqual(rec.fs, 256.0)
        self.assertIsNone(rec.t0_nominal)                         # year-2000 clock = not set
        np.testing.assert_allclose(rec.signal, x, atol=1)


if __name__ == "__main__":
    unittest.main()
