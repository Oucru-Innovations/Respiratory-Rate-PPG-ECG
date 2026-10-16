#!/usr/bin/env bash
# Fold-0 sweep of small architectures on the same split / labels / loss as the
# CNN baseline. Run names: runs/<features>_<arch>_f0.
#   1. LSTM, GRU, U-Net, MLP on the BW/AM/FM spectrograms (PPG, ECG)
#   2. BreathNet (physiology-inspired, waveform input) on PPG and ECG
#   3. LSTM, GRU, U-Net, MLP on the raw-signal STFT (PPG, ECG)
# Run detached from the repo root:
#   setsid nohup bash scripts/run_arch_sweep_fold0.sh > runs/run_arch_sweep_fold0.log 2>&1 &
set -u
PY=${PY:-~/anaconda3/envs/kaggle/bin/python}
D=dataset/01NVa_monitor

train() {  # features arch [extra args...]
  local feat=$1 arch=$2; shift 2
  local out=runs/${feat}_${arch}_f0
  if [ -f "$out/metrics.json" ]; then return; fi
  until [ -f "$D/features/$feat/config.json" ] && [ -f "$D/features/$feat/index.parquet" ]; do sleep 60; done
  echo "$(date +%H:%M) $out"
  $PY -W ignore -m dl.train --features "$D/features/$feat" --out "$out" --fold 0 --arch "$arch" \
      --workers 4 "$@" > "$out.log" 2>&1 || echo "FAILED $out"
}

for feat in ppg_mod_w60 ecg_mod_w60; do
  for arch in lstm gru unet mlp; do train $feat $arch; done
done
# BreathNet has ~6 k parameters, so a higher learning rate
train ppg_modwave_w60 breathnet --lr 3e-3
train ecg_modwave_w60 breathnet --lr 3e-3
for feat in ppg_raw_w60 ecg_raw_w60; do
  for arch in lstm gru unet mlp; do train $feat $arch; done
done
echo "ALL DONE $(date)"
