"""Build a windowed respiratory-rate dataset from GE monitor exports.

Stage 1 (``extract``): read every ``*waves.asc`` under ``--raw-root``, split
into gap-free segments, resample (ECG 250 Hz, PPG 100 Hz, Resp 25 Hz) and save
one compressed ``.npz`` per segment. Slow (148 GB of text) but done once.

Stage 2 (``windows``): slide a window over every segment and store, per
window, the reference RR from the impedance channel and quality indicators
for the PPG and ECG. Cheap to re-run with another window length / stride.

Usage (from the repo root)::

    python -m data_prep.build_dataset extract --raw-root /mnt/ssd/Workspace/Dengue/01NVa_Dengue/RAW_DATA --out dataset/01NVa_monitor --workers 4
    python -m data_prep.build_dataset windows --out dataset/01NVa_monitor --window 60 --stride 15 --workers 8

All windows of one patient share the same ``patient_id``; split train/val/test
by that column, never by window.
"""
import argparse
import json
import re
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep.quality import cardiac_quality
from data_prep.readers import CorruptFileError, read_ge_waves
from data_prep.reference import reference_rr
from preprocess.filters import resample

TARGET_FS = {"ecg": 250.0, "ppg": 100.0, "resp": 25.0}
# day folders are ddmmyyyy; read as text so leading zeros survive
SEGMENT_DTYPES = {"day": str, "patient_id": str, "segment_id": str}


def _ids(path):
    """RAW_DATA/<patient>/<ddmmyyyy>/Monitor/<file> -> (patient_id, day)."""
    p = Path(path)
    patient = p.parents[2].name.upper()
    m = re.search(r"(\d{3})-(\d{4})$", patient)
    if m:  # normalise case/typo variants such as 01Nva-003-2187
        patient = f"01NVA-{m.group(1)}-{m.group(2)}"
    return patient, p.parents[1].name


def segment_id(path, k):
    """Unique per source file: one day folder can hold several exports
    (``...2131waves.asc``, ``...2131-1waves.asc``, ``...2014 2waves.asc``)."""
    p = Path(path)
    stem = re.sub(r"[^0-9A-Za-z-]+", "-", p.name.removesuffix("waves.asc")).strip("-").upper()
    return f"{stem}_{p.parents[1].name}_seg{k:02d}"


# ----------------------------------------------------------------- stage 1
def extract_one(path, out_dir):
    patient, day = _ids(path)
    rows = []
    try:
        rec = read_ge_waves(path)
    except (CorruptFileError, pd.errors.ParserError, UnicodeDecodeError) as err:
        return [{"source": str(path), "patient_id": patient, "day": day, "error": str(err)}]
    for k, seg in enumerate(rec.segments):
        seg_id = segment_id(path, k)
        arrays = {}
        for name, x in seg.channels.items():
            # rails = recording-level min/max, used to detect clipping later
            arrays[name] = resample(x, seg.fs, TARGET_FS[name]).astype(np.float32)
            arrays[f"{name}_rails"] = np.array([np.nanmin(x), np.nanmax(x)], dtype=np.float32)
        np.savez_compressed(out_dir / f"{seg_id}.npz", **arrays)
        rows.append({"segment_id": seg_id, "patient_id": patient, "day": day,
                     "source": str(path), "source_fs": seg.fs,
                     "t0_epoch": seg.t0_epoch, "duration_s": seg.duration,
                     "header_local_time": rec.header_local_time,
                     "header_epoch": rec.header_epoch, "error": ""})
    if not rows:
        rows.append({"source": str(path), "patient_id": patient, "day": day,
                     "error": "no segment >= 30 s"})
    return rows


def run_extract(args):
    out = Path(args.out)
    sig_dir = out / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.raw_root).glob("*/*/Monitor/*waves.asc"))
    if args.limit:
        files = files[:args.limit]
    done = set()
    seg_csv = out / "segments.csv"
    if seg_csv.exists() and not args.overwrite:
        done = set(pd.read_csv(seg_csv)["source"])
    todo = [f for f in files if str(f) not in done]
    print(f"{len(files)} files, {len(todo)} to extract")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(extract_one, f, sig_dir): f for f in todo}
        for i, fut in enumerate(futs, 1):
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001 -- log and continue with other files
                res = [{"source": str(futs[fut]), "error": traceback.format_exc(limit=1)}]
            rows.extend(res)
            print(f"[{i}/{len(todo)}] {futs[fut].name}: {res[0].get('error') or f'{len(res)} segment(s)'}", flush=True)
            if i % 10 == 0 or i == len(todo):  # checkpoint so a crash loses little
                _append_csv(seg_csv, rows)
                rows = []


def _append_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path, dtype=SEGMENT_DTYPES), df], ignore_index=True)
    df.drop_duplicates(subset=["source", "segment_id"], keep="last").to_csv(path, index=False)


# ----------------------------------------------------------------- stage 2
def windows_one(seg_row, sig_dir, window_s, stride_s):
    data = np.load(sig_dir / f"{seg_row['segment_id']}.npz")
    sig = {k: data[k] for k in ("ecg", "ppg", "resp")}
    rails = {k: tuple(data[f"{k}_rails"].tolist()) for k in ("ecg", "ppg", "resp")}
    duration = len(sig["resp"]) / TARGET_FS["resp"]
    rows = []
    for start in np.arange(0, duration - window_s + 1e-9, stride_s):
        cut = {k: sig[k][int(round(start * TARGET_FS[k])):int(round((start + window_s) * TARGET_FS[k]))].astype(float)
               for k in sig}
        row = {"segment_id": seg_row["segment_id"], "patient_id": seg_row["patient_id"],
               "day": seg_row["day"], "start_s": float(start), "window_s": window_s,
               "t_epoch": seg_row["t0_epoch"] + start}
        row.update(reference_rr(cut["resp"], TARGET_FS["resp"], rails["resp"]))
        for k in ("ppg", "ecg"):
            q = cardiac_quality(cut[k], TARGET_FS[k], k, rails[k] if k == "ppg" else None)
            row.update({f"{k}_{name}": v for name, v in q.items()})
        rows.append(row)
    return rows


def run_windows(args):
    out = Path(args.out)
    segs = pd.read_csv(out / "segments.csv", dtype=SEGMENT_DTYPES)
    segs = segs[segs["error"].fillna("") == ""].dropna(subset=["segment_id"])
    if args.limit:
        segs = segs.head(args.limit)
    if segs["segment_id"].duplicated().any():
        raise SystemExit("duplicate segment_id in segments.csv; re-run extract")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(windows_one, r, out / "signals", args.window, args.stride): r["segment_id"]
                for r in segs.to_dict("records")}
        for i, fut in enumerate(futs, 1):
            try:
                rows.extend(fut.result())
            except Exception as err:  # noqa: BLE001 -- one bad file must not lose the run
                print(f"SKIPPED {futs[fut]}: {type(err).__name__}: {err}", flush=True)
            if i % 20 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {len(rows)} windows", flush=True)
    df = pd.DataFrame(rows)
    name = f"windows_w{int(args.window)}_s{int(args.stride)}.parquet"
    df.to_parquet(out / name, index=False)
    summary = {
        "windows": len(df), "patients": int(df["patient_id"].nunique()),
        "ref_ok": float(df["ref_ok"].mean()),
        "ref_ok_and_ppg_ok": float((df["ref_ok"] & df["ppg_ok"]).mean()),
        "ref_ok_and_ecg_ok": float((df["ref_ok"] & df["ecg_ok"]).mean()),
        "ref_rr_quantiles": df.loc[df["ref_ok"], "ref_rr"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_dict(),
    }
    (out / name.replace(".parquet", "_summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--raw-root", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--limit", type=int, default=0, help="only the first N files (testing)")
    e.add_argument("--overwrite", action="store_true")
    w = sub.add_parser("windows")
    w.add_argument("--out", required=True)
    w.add_argument("--window", type=float, default=60.0)
    w.add_argument("--stride", type=float, default=15.0)
    w.add_argument("--workers", type=int, default=8)
    w.add_argument("--limit", type=int, default=0, help="only the first N segments (testing)")
    args = ap.parse_args()
    run_extract(args) if args.cmd == "extract" else run_windows(args)


if __name__ == "__main__":
    main()
