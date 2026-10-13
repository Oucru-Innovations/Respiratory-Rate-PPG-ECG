#!/usr/bin/env bash
# Fold-0 experiment: features -> DL models (one GPU job at a time) and the
# signal-processing benchmark on the same test patients.
# Run detached from the repo root:  setsid nohup bash scripts/run_fold0.sh > runs/run_fold0.log 2>&1 &
set -u
PY=${PY:-~/anaconda3/envs/kaggle/bin/python}
D=dataset/01NVa_monitor
W=$D/windows_w60_s30.parquet
mkdir -p runs

features() {  # signal kind
  local short; short=$([ "$2" = modulation ] && echo mod || echo raw)
  local out=$D/features/${1}_${short}_w60
  if grep -qs "^done" "$D/features_${1}_${short}.log"; then return; fi
  $PY -W ignore -m dl.precompute_features --windows $W --signal "$1" --kind "$2" --out "$out" \
      --workers 12 > "$D/features_${1}_${short}.log" 2>&1
}

train() {  # name batch
  if [ -f "runs/$1_f0/metrics.json" ]; then return; fi
  until grep -qs "^done" "$D/features_$(echo "$1" | sed 's/_w60//').log"; do sleep 30; done
  $PY -W ignore -m dl.train --features "$D/features/$1" --out "runs/$1_f0" --fold 0 \
      --batch "$2" --workers 4 > "runs/$1_f0.log" 2>&1
}

benchmark() {  # signal
  if [ -f "runs/traditional_$1_f0/metrics.csv" ]; then return; fi
  until [ -f "runs/$1_mod_w60_f0/split.json" ]; do sleep 30; done
  $PY -W ignore -m scripts.benchmark_traditional --windows $W --signal "$1" \
      --split-json "runs/$1_mod_w60_f0/split.json" --sample 15000 --workers 8 \
      --out "runs/traditional_$1_f0" > "runs/traditional_$1_f0.log" 2>&1
}

( features ecg modulation; features ecg raw; benchmark ppg; benchmark ecg ) &
( train ppg_mod_w60 256; train ppg_raw_w60 128; train ecg_mod_w60 256; train ecg_raw_w60 128 ) &
wait
echo "ALL DONE $(date)"
