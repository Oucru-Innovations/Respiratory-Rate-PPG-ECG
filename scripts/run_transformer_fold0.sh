#!/usr/bin/env bash
# Fold-0 transformer runs on the same features / split / loss as the CNN runs.
# Stage A: frame vs patch tokens on PPG BW/AM/FM; the tokenisation with the
# better *validation* patient-MAE (never test) is then trained on the other
# three feature sets.
# Run detached from the repo root:
#   setsid nohup bash scripts/run_transformer_fold0.sh > runs/run_transformer_fold0.log 2>&1 &
set -u
PY=${PY:-~/anaconda3/envs/kaggle/bin/python}
D=dataset/01NVa_monitor/features

train() {  # features tokens
  local out=runs/$1_tf-$2_f0
  if [ -f "$out/metrics.json" ]; then return; fi
  $PY -W ignore -m dl.train --features "$D/$1" --out "$out" --fold 0 --arch transformer \
      --tokens "$2" --lr 5e-4 --batch 256 --workers 4 > "$out.log" 2>&1
}

val_mae() { $PY -c "import json;print(json.load(open('runs/$1/metrics.json'))['best_val_patient_mae'])"; }

train ppg_mod_w60 frame
train ppg_mod_w60 patch
best=$($PY -c "
a=$(val_mae ppg_mod_w60_tf-frame_f0); b=$(val_mae ppg_mod_w60_tf-patch_f0)
print('frame' if a <= b else 'patch')")
echo "stage A: frame=$(val_mae ppg_mod_w60_tf-frame_f0) patch=$(val_mae ppg_mod_w60_tf-patch_f0) -> $best"
for f in ppg_raw_w60 ecg_mod_w60 ecg_raw_w60; do train "$f" "$best"; done
echo "ALL DONE $(date)"
