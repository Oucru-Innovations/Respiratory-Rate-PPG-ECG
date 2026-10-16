#!/usr/bin/env bash
# Second BreathNet-2 pass on fold 0: the combinations the single-mechanism screening
# pointed at (coherent sum and heart-rate prior help most; the conv head beats HMM
# tracking but excludes it). Waits for scripts/run_breathnet2_fold0.sh to finish, then
# pauses the cross-validation lane's scheduler the same way (see that script).
#   setsid nohup bash scripts/run_breathnet2_combos_fold0.sh > runs/run_breathnet2_combos_fold0.log 2>&1 &
set -uo pipefail
PY=${KAGGLE_PY:-~/anaconda3/envs/kaggle/bin/python}
FEAT=dataset/01NVa_monitor/features
until grep -q "SCREENING DONE" runs/run_breathnet2_fold0.log 2>/dev/null; do sleep 60; done

LANE=$(pgrep -f "scripts.run_cv --lane gp[u]" || true)
resume() { [ -n "$LANE" ] && kill -CONT $LANE 2>/dev/null && echo "$(date +%H:%M) CV lane resumed"; }
trap resume EXIT
if [ -n "$LANE" ]; then kill -STOP $LANE && echo "$(date +%H:%M) CV lane scheduler paused (pid $LANE); its current job continues"; fi

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
train() {      # signal, variant, run-name suffix
  out=runs/${1}_modwavehr_w60_${3}_f0
  [ -f "$out/metrics.json" ] && return
  while [ "$(gpu_free)" -lt 900 ]; do sleep 30; done
  echo "$(date +%H:%M) start $out"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore -m dl.train --features $FEAT/${1}_modwavehr_w60 \
      --arch breathnet2 --variant "$2" --lr 3e-3 --workers 2 --out "$out" --fold 0 > "$out.log" 2>&1 \
      && echo "$(date +%H:%M) done  $out" || echo "FAILED $out"
}
for sig in ppg ecg; do
  train $sig coherent,hr               breathnet2-coh-hr
  train $sig coherent,hr,conv          breathnet2-coh-hr-conv
  train $sig coherent,hr,track         breathnet2-coh-hr-track
  train $sig whiten,coherent,hr,conv   breathnet2-all-conv
done
echo "BREATHNET2 COMBOS DONE $(date +%F\ %H:%M)"
