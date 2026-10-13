# Archived legacy code

Folder map:

| Path | What |
|---|---|
| `estimate_rr/`, `preprocess/`, `common/`, `dl/`, `data_prep/`, `examples/` | retired code (defects below) |
| `notebooks/` | 2023 exploration notebooks for the deleted 24EIb data; `utils.ipynb` / `debug.py` use the retired DL code |
| `legacy_outputs/` | outputs of the retired pipeline: training plots, `results/*.csv`, `study.pkl` (Optuna), `mlruns/`, `saved_models/`, `artifacts/`, scratch files. Results here were produced with the bugs below and should not be cited |
| `legacy_data/bk/` | 7 GB of CSVs built by the leaky `compose_rr_data*` scripts (git-ignored) |
| `requirements_legacy.txt` | pinned environment of the old code |

Code moved here after the October 2026 audit. It is kept for reference only. Do not import it from
the active code; most of it no longer runs (it imports the non-existent `RRest`
package, `spectrum`, `mne`, `efficientnet_pytorch`, …). Replacements are listed per
file. The numbered findings below were reproduced by running the code.

## `estimate_rr/` → replaced by `estimate_rr/` (new)

Every method ran directly on raw PPG/ECG band-passed to 0.1–0.5 Hz, so only
baseline wander was ever used and rates above 30 bpm were cut off. On clean
synthetic signals at 8–30 bpm only `CtA` returned correct rates.

| File | Defect |
|---|---|
| `ARM.py` | Frequency axis built as 0–6 Hz for 6 Hz data (should be 0–3); returned a *count of PSD peaks* divided by duration instead of a frequency. |
| `ARP.py` | Returned a pole *magnitude* (~0.9), not a rate; used the pre-resampling fs; compared radians to bpm; `np.roots` without the leading 1. |
| `AR_RR.py`, `freq_rr.py` | Treated AR coefficients (or `AutoReg.predict` output) as roots; "resample to 2 Hz" never happened; 2–30 Hz ECG filter applied to a respiratory signal. |
| `WCH.py` | Returned the *number* of Welch peaks, not the peak frequency. |
| `PKS.py` | Counted maxima of the peak-interval series, not breaths. |
| `CtO.py` | Peak threshold 0.95×Q3 instead of 0.2×Q3 (Schäfer & Kratky 2008); returned 0 on clean signals. |
| `CtA.py` | Threshold Q3 instead of 0.3×Q3; crashed when < 2 extrema remained. |
| `PZX.py` | Alternation logic reset its state and skipped checks; ~21–30 bpm regardless of input. |
| `Interpolate_RR.py`, `time_rr.py` | Low-passed to 0.5 Hz *before* R-peak detection; 4 Hz EDR filtered with fs = 100; returned `max(60/diff)` in samples. |

## `preprocess/`

* `band_filter.py`, `preprocess_signal.py`: low-pass used causal `lfilter`
  (phase lag) while high-pass used `filtfilt`; transfer-function form is unstable
  at 0.1 Hz cut-offs. → `preprocess/filters.py` (SOS + `sosfiltfilt`).
* `synthesize_data.py`: imports `RRest`, runs code at import time.

## `common/`

* `rpeak_detection.py`, `utils.py`: import `RRest.*` / `ecgdetectors`; `get_nn`
  referenced a commented-out import. → `preprocess/beats.py`, `common/metrics.py`.

## `data_prep/` (the `compose_rr_data*.py` scripts) → `data_prep/build_dataset.py`

* Label window ≠ input window: label from 60 s of reference, input = "last 3000
  samples" (assumed 100 Hz; Philips PLETH is typically 125 Hz).
* `MPDataExport.csv` rows have one more field than the header, so
  `NOM_RESP_RATE` read by name returns the **heart rate** (median 137).
* Train/validation leakage: patients 001 and 004 used the identical recording
  for both; the other "validation" files were other days of the same patients.
  Random sampling with replacement + repeated appends produced duplicates.
* Label clipped to [5, 40] but `num_classes = 40` → label 40 crashes `one_hot`.
* Vivalink ECG: every sample of a packet got the same timestamp
  (`timedelta(ms=delta)` never multiplied by the sample index).

## `dl/` → `dl/` (new)

* `wearable_data.py`: 0.5–3.5 Hz band-pass + 2nd derivative removed 99.9 % of
  the respiratory-band power; STFT hop 1.28 s aliases RR > 23 bpm; 0–50 Hz axis
  put respiration in 2 % of rows; dB/max normalisation gave values in
  [-75, 1], which `ToPILImage` wrapped modulo 256.
* `model_utils.ClosenessLoss`: index logic applied to one-hot targets (always 0),
  one-hot of a one-hot, mutable class state; effectively plain cross-entropy
  with no ordinal ("closeness") penalty.
* `run_pipeline_classifier_ppg.py`: AUC computed on shuffled predictions vs
  unshuffled targets; L2 penalty added on top of Adam weight decay; only
  1000 random windows.
* `model_pool.py`: `RREfficientNetGMClassifierPPG` discards its mixture output;
  `RREfficientNetClassifier.compute_RR` runs `find_peaks` on filtered ECG (every local maximum,
  not R peaks) and usually falls back to 20.
* `eval_segment.py`: evaluates a different architecture (64 classes) from the one
  trained (40); `mode='test'` returns 2 items but 3 are unpacked; no metrics.
