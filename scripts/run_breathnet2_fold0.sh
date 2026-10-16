#!/usr/bin/env bash
# BreathNet-2 screening on fold 0 (PPG and ECG): each mechanism added to BreathNet on
# its own, then all four together (~12 runs, <0.6 GB and ~5 min each on the GPU).
#
# GPU sharing: the cross-validation lane (scripts/run_cv.py --lane gpu) must not start a
# new job while a screening run holds memory, or it would hit CUDA OOM. So the lane's
# *scheduler* process is paused (SIGSTOP) for the duration: its current training job
# finishes normally, no new one starts, and the lane resumes (SIGCONT) when this script
# exits, however it exits. Finished runs are skipped.
#   setsid nohup bash scripts/run_breathnet2_fold0.sh > runs/run_breathnet2_fold0.log 2>&1 &
set -uo pipefail
PY=${KAGGLE_PY:-~/anaconda3/envs/kaggle/bin/python}
WINDOWS=dataset/01NVa_monitor/windows_w60_s30.parquet
FEAT=dataset/01NVa_monitor/features

# BW/AM/FM waveforms + instantaneous heart rate (needed by the hr prior; the others ignore the channel)
for sig in ppg ecg; do
  [ -f $FEAT/${sig}_modwavehr_w60/config.json ] || \
    $PY -W ignore -m dl.precompute_features --windows $WINDOWS --signal $sig --kind modwave_hr \
        --out $FEAT/${sig}_modwavehr_w60 --workers 6 > runs/precompute_${sig}_modwavehr.log 2>&1 &
done
wait

LANE=$(pgrep -f "scripts.run_cv --lane gp[u]" || true)
resume() { [ -n "$LANE" ] && kill -CONT $LANE 2>/dev/null && echo "$(date +%H:%M) CV lane resumed"; }
trap resume EXIT
if [ -n "$LANE" ]; then kill -STOP $LANE && echo "$(date +%H:%M) CV lane scheduler paused (pid $LANE); its current job continues"; fi

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
train() {      # signal, variant, run-name suffix
  out=runs/${1}_modwavehr_w60_${3}_f0
  [ -f "$out/metrics.json" ] && return
  while [ "$(gpu_free)" -lt 900 ]; do sleep 30; done     # the lane's last job may still be finishing
  echo "$(date +%H:%M) start $out"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore -m dl.train --features $FEAT/${1}_modwavehr_w60 \
      --arch breathnet2 --variant "$2" --lr 3e-3 --workers 2 --out "$out" --fold 0 > "$out.log" 2>&1 \
      && echo "$(date +%H:%M) done  $out" || echo "FAILED $out"
}
for sig in ppg ecg; do
  train $sig whiten    breathnet2-whiten
  train $sig hr        breathnet2-hr
  train $sig track     breathnet2-track
  train $sig coherent  breathnet2-coherent
  train $sig conv      breathnet2-conv
  train $sig whiten,coherent,track,hr breathnet2
done
echo "BREATHNET2 SCREENING DONE $(date +%F\ %H:%M)"
