"""Five-fold cross-validation of every fold-0 experiment.

Two lanes, each a long-running, resumable process (finished outputs are skipped):

* ``gpu``: replays every fold-0 DL run on folds 1-4 with *exactly* its fold-0
  arguments (read from the fold-0 checkpoint), in priority tiers so the core
  results arrive first: (1) CNN and BreathNet on BW/AM/FM input, (2) the main
  BreathNet-2 variants, (3) the other architectures on BW/AM/FM input, (4) the
  raw-signal-STFT runs. Each job waits for enough free GPU memory (the GPU is
  shared), so other GPU work only delays the lane instead of crashing it.
* ``cpu``: patient splits, the signal-processing benchmark on *every* test
  window of folds 0-4, trend scoring and BreathNet ablations per fold.

Status of every job is kept in ``runs/cv_status.json``. Run detached::

    setsid nohup python -m scripts.run_cv --lane gpu > runs/cv_gpu.log 2>&1 &
    setsid nohup python -m scripts.run_cv --lane cpu > runs/cv_cpu.log 2>&1 &
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RUNS = Path("runs")
FEATURES = Path("dataset/01NVa_monitor/features")
WINDOWS = "dataset/01NVa_monitor/windows_w60_s30.parquet"
FOLDS = range(5)
STATUS = RUNS / "cv_status.json"
PY = sys.executable
TIER1 = ("ppg_mod_w60", "ecg_mod_w60", "ppg_modwave_w60_breathnet", "ecg_modwave_w60_breathnet")
# BreathNet-2 models that mattered on fold 0 (per-signal best, the 690/585-parameter ones, the conv head)
TIER2 = tuple(f"{s}_modwavehr_w60_{v}" for s in ("ppg", "ecg")
              for v in ("breathnet2-coh-hr-conv", "breathnet2", "breathnet2-track", "breathnet2-conv"))


def set_status(job, state, **extra):
    """Atomic read-modify-write of the shared status file (both lanes write it)."""
    STATUS.touch()
    with open(STATUS, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        txt = fh.read()
        status = json.loads(txt) if txt.strip() else {}
        status[job] = {"state": state, "time": time.strftime("%Y-%m-%d %H:%M"), **extra}
        fh.seek(0)
        fh.truncate()
        json.dump(status, fh, indent=1, sort_keys=True)


def gpu_free_mib():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(job, min_free_mib):
    """Block until the GPU has ``min_free_mib`` free (the GPU is shared with other work;
    starting a job into a full GPU only produces an out-of-memory crash)."""
    waited = False
    while (free := gpu_free_mib()) is not None and free < min_free_mib:
        if not waited:
            print(f"{time.strftime('%H:%M')} waiting for {min_free_mib} MiB of GPU memory ({free} free)", flush=True)
            set_status(job, f"waiting for GPU memory ({min_free_mib} MiB)")
        waited = True
        time.sleep(60)


def run(job, cmd, log, retries_on_oom=3, env=None, min_free_mib=0):
    """Run ``cmd`` (list) to ``log``; retry on CUDA out-of-memory. Returns True on success."""
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **(env or {})}
    for attempt in range(1, retries_on_oom + 2):
        if min_free_mib:
            wait_for_gpu(job, min_free_mib)
        set_status(job, "running", attempt=attempt)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            set_status(job, "done")
            return True
        oom = "OutOfMemoryError" in Path(log).read_text(errors="ignore")
        if not oom or attempt > retries_on_oom:
            set_status(job, "failed", returncode=rc, oom=oom)
            return False
        set_status(job, "waiting (GPU out of memory, retrying)", attempt=attempt)
        time.sleep(300)
    return False


# ---------------------------------------------------------------- GPU lane
def fold0_runs():
    """Every finished fold-0 DL run: name stem (without _f0) and its saved args."""
    import torch
    out = []
    for d in sorted(RUNS.glob("*_f0")):
        if (d / "metrics.json").exists() and (d / "best.pt").exists():
            args = torch.load(d / "best.pt", map_location="cpu", weights_only=False)["args"]
            out.append((d.name[:-3], args))
    return out


def train_cmd(args, fold, out):
    cmd = [PY, "-W", "ignore", "-m", "dl.train"]
    for k, v in args.items():
        if k in ("out", "fold") or v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        cmd += [flag] if v is True else [flag, str(v)]
    return cmd + ["--out", str(out), "--fold", str(fold)]


def tier(stem):
    if stem in TIER1:
        return 1
    if stem in TIER2:
        return 2
    return 3 if "_mod" in stem else 4


def gpu_lane():
    jobs = sorted(((tier(stem), fold, stem, args) for stem, args in fold0_runs() for fold in FOLDS if fold > 0),
                  key=lambda j: (j[0], j[1], j[2]))
    print(f"{len(jobs)} training jobs", flush=True)
    for t, fold, stem, args in jobs:
        out = RUNS / f"{stem}_f{fold}"
        job = f"train {out.name}"
        if (out / "metrics.json").exists():
            set_status(job, "done")
            continue
        print(f"{time.strftime('%H:%M')} tier {t} {out.name}", flush=True)
        # measured peaks on this machine: CNN/U-Net/transformer ~1.9 GB (raw-STFT U-Net more), BreathNet ~0.6 GB
        need = 1000 if "breathnet" in stem else (2800 if "raw" in stem and "unet" in stem else 2300)
        run(job, train_cmd(args, fold, out), f"{out}.log", retries_on_oom=10, min_free_mib=need)
    print("GPU LANE DONE", time.strftime("%Y-%m-%d %H:%M"), flush=True)


# ---------------------------------------------------------------- CPU lane
def write_splits():
    """Patient splits per fold, from the same function and patient set as dl.train."""
    from common.splits import patient_split
    idx = pd.read_parquet(FEATURES / "ppg_mod_w60" / "index.parquet")
    pats = idx.loc[idx["feature_ok"], "patient_id"]
    (RUNS / "splits").mkdir(exist_ok=True)
    for k in FOLDS:
        s = patient_split(pats, 5, k, 0)
        (RUNS / "splits" / f"fold{k}.json").write_text(json.dumps(s, indent=1))
    ref = json.loads((RUNS / "ppg_mod_w60_f0" / "split.json").read_text())
    assert ref == json.loads((RUNS / "splits" / "fold0.json").read_text()), "fold-0 split differs from the trained runs"
    tests = [set(json.loads((RUNS / "splits" / f"fold{k}.json").read_text())["test"]) for k in FOLDS]
    assert sum(len(t) for t in tests) == len(set().union(*tests)), "test folds overlap"
    print(f"splits written: {[len(t) for t in tests]} test patients per fold", flush=True)


def cpu_jobs():
    jobs = []
    for sig in ("ppg", "ecg"):
        for k in FOLDS:
            out = RUNS / f"traditional_{sig}_f{k}_all"
            jobs.append((f"traditional {sig} f{k}", out / "metrics.csv", None,
                         [PY, "-W", "ignore", "-m", "scripts.benchmark_traditional", "--windows", WINDOWS, "--signal", sig,
                          "--split-json", str(RUNS / "splits" / f"fold{k}.json"), "--workers", "8", "--out", str(out)],
                         f"{out}.log"))
            cnn = RUNS / f"{sig}_mod_w60_f{k}"
            jobs.append((f"trend {sig} f{k}", RUNS / f"trends_f{k}" / f"trend_{sig}.parquet", cnn / "metrics.json",
                         [PY, "-W", "ignore", "-m", "scripts.predict_trend", "--signal", sig, "--run", str(cnn),
                          "--out", str(RUNS / f"trends_f{k}"), "--workers", "6"], RUNS / f"trends_{sig}_f{k}.log"))
            bn = RUNS / f"{sig}_modwave_w60_breathnet_f{k}"
            jobs.append((f"breathnet analysis {sig} f{k}", RUNS / f"breathnet_f{k}" / f"{sig}.npz", bn / "metrics.json",
                         [PY, "-W", "ignore", "-c", f"from scripts.breathnet_analysis import export; export('{sig}', fold={k})"],
                         RUNS / f"breathnet_{sig}_f{k}.log"))
    return jobs


def cpu_lane():
    write_splits()
    pending = cpu_jobs()
    while pending:
        # cheap dependent jobs first once their model exists, then the next benchmark
        ready = [j for j in pending if j[2] is None or Path(j[2]).exists()]
        ready.sort(key=lambda j: j[0].startswith("traditional"))
        if not ready:
            time.sleep(300)
            continue
        name, done_file, _, cmd, log = ready[0]
        pending.remove(ready[0])
        if Path(done_file).exists():
            set_status(name, "done")
            continue
        print(f"{time.strftime('%H:%M')} {name}", flush=True)
        run(name, cmd, log, retries_on_oom=0)
    print("CPU LANE DONE", time.strftime("%Y-%m-%d %H:%M"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the job list only")
    a = ap.parse_args()
    if a.dry_run:
        if a.lane == "gpu":
            for t, fold, stem, args in sorted(((tier(s), f, s, ar) for s, ar in fold0_runs() for f in FOLDS if f > 0)):
                print(t, " ".join(train_cmd(args, fold, RUNS / f"{stem}_f{fold}")[4:]))
        else:
            for j in cpu_jobs():
                print(j[0], "| waits for", j[2])
        return
    gpu_lane() if a.lane == "gpu" else cpu_lane()


if __name__ == "__main__":
    main()
