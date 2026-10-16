"""Wearable PPG / ECG windows scored against the monitor's impedance reference.

The models and signal-processing methods were developed on the bedside monitor's
PPG and ECG. This builds the same 60-s windows from the patients' *wearables*
(SmartCare wrist PPG, Faros/ePatch chest ECG) at the same moments, so each method
can be scored on the wearable against the same impedance reference.

Stages (from the repo root; ``dataset/01NVa_monitor`` must exist)::

    python -m data_prep.build_wearable align   --workers 8   # heart-rate alignment of every file
    python -m data_prep.build_wearable extract --workers 8   # wearable signal on the monitor clock
    python -m data_prep.build_wearable windows --workers 8   # windows_{ppg,ecg}_w60_s30.parquet

Output ``dataset/01NVa_wearable/``:

* ``alignment.csv``: one row per wearable file x straight-line piece of its time
  map (monitor time = wearable time + lag0 + drift * wearable time), with the
  evidence (blocks, correlation, residual); ``accepted`` pieces are used.
* ``signals/<monitor segment>_<kind>.npz``: the wearable channel resampled onto the
  monitor segment's time base (NaN where the wearable has no aligned data), the
  monitor's impedance (``resp``) and the monitor's own channel of the same kind
  (``monitor_ppg`` / ``monitor_ecg``) for paired comparison.
* ``windows_<kind>_w60_s30.parquet``: the monitor windows (same ``start_s``,
  ``t_epoch`` and reference RR columns; ``monitor_window_row`` joins them 1:1 with
  the monitor's results) that the wearable covers completely, with the wearable's
  quality flags and ``monitor_ok`` (the monitor's own channel usable).
"""
import argparse
import json
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep.build_dataset import SEGMENT_DTYPES, TARGET_FS
from data_prep.quality import cardiac_quality
from data_prep.wearable import (beat_times, coarse_candidates, fine_align, hr_series, read_edf_ecg,
                                read_smartcare)

RAW = Path("/mnt/ssd/Workspace/Dengue/01NVa_Dengue/RAW_DATA")
MONITOR = Path("dataset/01NVa_monitor")
OUT = Path("dataset/01NVa_wearable")
# Device clock -> monitor clock offsets seen on verified alignments (hours); tried as
# extra candidates, each still verified beat by beat like any other candidate.
PRIOR_OFFSETS_H = {"ppg": (9.1,), "ecg": (9.86, 14.63, 16.4)}


def patient_id(path):
    m = re.search(r"(\d{3})-(\d{4})", Path(path).relative_to(RAW).parts[0])
    return f"01NVA-{m.group(1)}-{m.group(2)}" if m else None


def inventory():
    rows = []
    for f in sorted(RAW.glob("*/*/PPG/*.csv")):
        rows.append({"kind": "ppg", "path": str(f)})
    for f in sorted(list(RAW.glob("*/*/ECG/*.edf")) + list(RAW.glob("*/*/ECG/*.EDF"))):
        rows.append({"kind": "ecg", "path": str(f)})
    df = pd.DataFrame(rows)
    df["patient_id"] = df["path"].map(patient_id)
    df["day"] = df["path"].map(lambda p: Path(p).relative_to(RAW).parts[1])
    return df


def monitor_segments():
    s = pd.read_csv(MONITOR / "segments.csv", dtype=SEGMENT_DTYPES)
    return s[s["error"].fillna("") == ""].dropna(subset=["segment_id"])


# ---------------------------------------------------------------- stage 1: align
def monitor_beats(patient, segs, cache_dir):
    """Monitor beat times (epoch s) of one patient: ECG R peaks, or the monitor PPG
    where the ECG gives too few plausible beats. Cached per patient."""
    cache = cache_dir / f"{patient}_beats.npy"
    if cache.exists():
        return np.load(cache)
    out = []
    for _, r in segs[segs["patient_id"] == patient].sort_values("t0_epoch").iterrows():
        d = np.load(MONITOR / "signals" / f"{r['segment_id']}.npz")
        best = None
        for kind in ("ecg", "ppg"):
            b = beat_times(d[kind].astype(float), TARGET_FS[kind], kind)
            _, h = hr_series(b, r["duration_s"])
            if best is None or np.isfinite(h).mean() > best[1] + 0.2:
                best = (b, np.isfinite(h).mean())
        out.append(r["t0_epoch"] + best[0])
    beats = np.sort(np.concatenate(out)) if out else np.array([])
    np.save(cache, beats)
    return beats


def wearable_hr_1hz(rec, beats):
    """1-Hz heart rate for the coarse search: the SmartCare's own pulse rate (cleaner
    than wrist-PPG beats during movement) or the ECG's beats."""
    if rec.kind == "ppg":
        p = rec.extra["device_pulse"].astype(float)
        p[(p >= 250) | (p < 25)] = np.nan
        n = int(rec.duration_s)
        p = p[:n * 100].reshape(n, 100)
        with np.errstate(all="ignore"):
            return np.nanmedian(p, axis=1)
    return hr_series(beats, rec.duration_s)[1]


def align_file(row, segs, cache_dir):
    t = time.time()
    base = {"kind": row["kind"], "path": row["path"], "patient_id": row["patient_id"], "day": row["day"]}
    rec = read_smartcare(row["path"]) if row["kind"] == "ppg" else read_edf_ecg(row["path"])
    base.update(duration_h=rec.duration_s / 3600, t0_nominal=rec.t0_nominal)
    if rec.duration_s < 1800:
        return [{**base, "status": "shorter than 30 min"}]
    bm = monitor_beats(row["patient_id"], segs, cache_dir)
    if len(bm) < 1000:
        return [{**base, "status": "no monitor beats"}]
    bw = beat_times(rec.signal, rec.fs, rec.kind)
    T0, T1 = np.floor(bm[0]), np.ceil(bm[-1])
    hm = hr_series(bm - T0, T1 - T0)[1]
    hw = wearable_hr_1hz(rec, bw)
    drifts = np.arange(-6000, 6001, 250) if rec.kind == "ppg" else np.arange(-500, 501, 100)
    priors = PRIOR_OFFSETS_H[rec.kind] if rec.t0_nominal else ()
    windows = [(rec.t0_nominal + h * 3600 - T0 - 1800, rec.t0_nominal + h * 3600 - T0 + 1800) for h in priors]
    cands = coarse_candidates(hw, hm, len(hm) + len(hw), k=5, drifts_ppm=drifts, lag_windows=windows)
    cands = [{**c, "prior_h": priors[c["window"]]} if "window" in c else c for c in cands]
    if not cands:
        return [{**base, "status": "not aligned", "seconds": time.time() - t}]
    best, best_key = None, (-1, -1)
    for c in cands:
        v = fine_align(bw, bm, T0 + c["lag"], c["drift_ppm"])
        acc = [p for p in v["pieces"] if p["accepted"]]
        key = (sum(p["t_end"] - p["t_start"] for p in acc), np.median([p["median_r"] for p in acc]) if acc else 0)
        if key > best_key:
            best, best_key = (c, v), key
    out = []
    c, v = best
    for i, p in enumerate(v["pieces"]):
        out.append({**base, "status": "aligned" if p["accepted"] else "rejected piece", "piece": i,
                    "coarse_r": c["r"], "coarse_drift_ppm": c["drift_ppm"], "prior_h": c.get("prior_h", np.nan),
                    "offset_minus_nominal_h": ((T0 + c["lag"]) - rec.t0_nominal) / 3600 if rec.t0_nominal else np.nan,
                    **p, "seconds": time.time() - t})
    if not out:
        out.append({**base, "status": "not aligned", "coarse_r": c["r"] if c else np.nan, "seconds": time.time() - t})
    return out


def align_patient(patient, rows, segs, cache_dir):
    out = []
    for row in rows:
        try:
            out += align_file(row, segs, cache_dir)
        except Exception as e:
            out.append({**{k: row[k] for k in ("kind", "path", "patient_id", "day")}, "status": f"error: {e!r}"[:300],
                        "traceback": traceback.format_exc()[-1500:]})
    return out


def run_align(args):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    segs = monitor_segments()
    inv = inventory()
    inv = inv[inv["patient_id"].isin(set(segs["patient_id"]))]
    done = set()
    out_csv = OUT / "alignment.csv"
    if out_csv.exists() and not args.redo:
        done = set(pd.read_csv(out_csv)["path"])
    todo = inv[~inv["path"].isin(done)]
    if args.patients:
        todo = todo[todo["patient_id"].isin(args.patients)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(inv)} wearable files of monitored patients, {len(todo)} to align", flush=True)
    groups = list(todo.groupby("patient_id"))
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(align_patient, p, g.to_dict("records"), segs, cache) for p, g in groups]
        for i, f in enumerate(as_completed(futs), 1):
            rows = f.result()
            df = pd.DataFrame(rows)
            if out_csv.exists():
                df = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
            df.to_csv(out_csv, index=False)
            ok = sum(r.get("status") == "aligned" for r in rows)
            print(f"[{i}/{len(groups)}] {rows[0]['patient_id']}: {len(rows)} rows, {ok} aligned pieces", flush=True)


# ---------------------------------------------------------------- stage 2: extract
def extract_patient(patient, pieces, segs, out):
    """Resample each aligned wearable piece onto the monitor segments it overlaps."""
    my = segs[segs["patient_id"] == patient]
    out_rows = []
    by_kind = {}
    for kind, g in pieces.groupby("kind"):
        recs = {}
        for path, gp in g.groupby("path"):
            rec = read_smartcare(path) if kind == "ppg" else read_edf_ecg(path)
            sig = rec.signal.astype(np.float64)
            for _, p in gp.iterrows():
                d = p["drift_ppm"] * 1e-6
                e0, e1 = p["t_start"] * (1 + d) + p["lag0_s"], p["t_end"] * (1 + d) + p["lag0_s"]
                recs.setdefault(kind, []).append((sig, rec.fs, p["lag0_s"], d, e0, e1, path, float(np.nanmin(sig)), float(np.nanmax(sig))))
        by_kind[kind] = recs.get(kind, [])
    for _, s in my.iterrows():
        t0, dur = float(s["t0_epoch"]), float(s["duration_s"])
        mon = None
        for kind, items in by_kind.items():
            fs = TARGET_FS[kind]
            n = int(round(dur * fs))
            x = np.full(n, np.nan)
            sources, rails = [], [np.inf, -np.inf]
            for sig, wfs, lag0, d, e0, e1, path, lo, hi in items:
                a, b = max(e0, t0), min(e1, t0 + dur)
                if b - a < 60:
                    continue
                i0, i1 = int(np.ceil((a - t0) * fs)), int(np.floor((b - t0) * fs))
                e = t0 + np.arange(i0, i1) / fs
                tw = (e - lag0) / (1 + d) * wfs                  # wearable sample index
                k = np.floor(tw).astype(np.int64)
                ok = (k >= 0) & (k + 1 < len(sig))
                w = tw - k
                vals = np.full(len(e), np.nan)
                vals[ok] = (1 - w[ok]) * sig[k[ok]] + w[ok] * sig[k[ok] + 1]
                fill = np.isnan(x[i0:i1])
                x[i0:i1][fill] = vals[fill]
                sources.append(Path(path).name)
                rails = [min(rails[0], lo), max(rails[1], hi)]
            if not sources or np.isfinite(x).sum() < 60 * fs:
                continue
            if mon is None:
                mon = np.load(MONITOR / "signals" / f"{s['segment_id']}.npz")
            seg_id = f"{s['segment_id']}_{kind.upper()}W"
            np.savez_compressed(out / "signals" / f"{seg_id}.npz", **{
                kind: x.astype(np.float32), f"{kind}_rails": np.array(rails, dtype=np.float32),
                "resp": mon["resp"], "resp_rails": mon["resp_rails"],
                f"monitor_{kind}": mon[kind], f"monitor_{kind}_rails": mon[f"{kind}_rails"]})
            out_rows.append({"segment_id": seg_id, "monitor_segment_id": s["segment_id"], "kind": kind,
                             "patient_id": patient, "day": s["day"], "t0_epoch": t0, "duration_s": dur,
                             "wearable_coverage": float(np.isfinite(x).mean()), "sources": ";".join(sorted(set(sources)))})
    return out_rows


def run_extract(args):
    (OUT / "signals").mkdir(parents=True, exist_ok=True)
    al = pd.read_csv(OUT / "alignment.csv")
    pieces = al[al["status"] == "aligned"]
    segs = monitor_segments()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(extract_patient, p, g, segs, OUT): p for p, g in pieces.groupby("patient_id")}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception as e:
                print(f"{futs[f]} failed: {e!r}", flush=True)
            if i % 20 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] patients extracted", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "segments.csv", index=False)
    print(f"{len(rows)} wearable segments written", flush=True)


# ---------------------------------------------------------------- stage 3: windows
def windows_segment(seg, mon_windows, window_s, out):
    d = np.load(out / "signals" / f"{seg['segment_id']}.npz")
    kind = seg["kind"]
    x, fs = d[kind], TARGET_FS[kind]
    rails = tuple(d[f"{kind}_rails"].tolist()) if kind == "ppg" else None
    rows = []
    for _, w in mon_windows.iterrows():
        a = int(round(w["start_s"] * fs))
        cut = x[a:a + int(round(window_s * fs))].astype(float)
        cov = float(np.isfinite(cut).mean()) if len(cut) else 0.0
        if cov < 1.0:          # wearable gaps are rare (a few per recording) and only at piece edges otherwise
            continue
        q = cardiac_quality(cut, fs, kind, rails)
        row = {**w.to_dict(), "segment_id": seg["segment_id"], "monitor_segment_id": seg["monitor_segment_id"],
               "wearable_coverage": cov}
        row.update({f"{kind}_{k}": v for k, v in q.items()})
        rows.append(row)
    return rows


def run_windows(args):
    segs = pd.read_csv(OUT / "segments.csv", dtype=SEGMENT_DTYPES)
    mon = pd.read_parquet(MONITOR / f"windows_w{args.window}_s{args.stride}.parquet")
    mon["monitor_window_row"] = np.arange(len(mon))       # = window_row of the monitor results (CV predictions)
    ref_cols = [c for c in mon.columns if c.startswith("ref_")]
    keep = ["segment_id", "patient_id", "day", "start_s", "window_s", "t_epoch", "monitor_window_row"] + ref_cols
    for kind in ("ppg", "ecg"):
        sk = segs[segs["kind"] == kind]
        if sk.empty:
            continue
        mk = mon[keep + [f"{kind}_ok"]].rename(columns={"segment_id": "monitor_segment_id", f"{kind}_ok": "monitor_ok"})
        rows = []
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(windows_segment, s, mk[mk["monitor_segment_id"] == s["monitor_segment_id"]].drop(
                        columns=["monitor_segment_id"]), args.window, OUT)
                    for _, s in sk.iterrows()]
            for f in as_completed(futs):
                rows += f.result()
        w = pd.DataFrame(rows)
        for other in ("ppg", "ecg"):          # benchmark/feature scripts filter on <signal>_ok
            if other != kind:
                w[f"{other}_ok"] = False
        name = f"windows_{kind}_w{args.window}_s{args.stride}.parquet"
        w.to_parquet(OUT / name, index=False)
        print(f"{kind}: {len(w)} windows, {w['patient_id'].nunique()} patients, "
              f"reference ok {w['ref_ok'].mean():.0%}, wearable ok {w[f'{kind}_ok'].mean():.0%}, "
              f"both {(w['ref_ok'] & w[f'{kind}_ok']).mean():.0%} -> {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["align", "extract", "windows"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="align: only the first N files (testing)")
    ap.add_argument("--redo", action="store_true", help="align: redo files already in alignment.csv")
    ap.add_argument("--patients", nargs="*", default=None, help="align: only these patient ids (testing)")
    ap.add_argument("--out", default=None, help="output folder (default dataset/01NVa_wearable)")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--stride", type=int, default=30)
    args = ap.parse_args()
    if args.out:
        global OUT
        OUT = Path(args.out)     # worker functions get it as an argument (forkserver/spawn re-import the module)
    {"align": run_align, "extract": run_extract, "windows": run_windows}[args.stage](args)


if __name__ == "__main__":
    main()
