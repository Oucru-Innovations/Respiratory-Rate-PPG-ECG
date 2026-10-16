# Respiratory-Rate-PPG-ECG Repository

[![DOI](https://zenodo.org/badge/659031505.svg)](https://doi.org/10.5281/zenodo.14613763)

The **Respiratory-Rate-PPG-ECG** repository offers methods to estimate respiratory rate using signals from wearable devices, specifically Electrocardiogram (ECG) and Photoplethysmogram (PPG). It includes both signal processing techniques and deep learning models to predict respiratory rates.

## Features
- **Signal processing methods:** respiratory signals (baseline wander, amplitude and frequency modulation) extracted from detected beats, nine estimators (CtO, CtA, PKS, ZeX, FTS, WCH, ARS, ARM, ARP) and smart / median fusion, following Charlton et al. (2016) and Karlen et al. (2013).
- **Deep learning:** a small CNN on spectrograms of the respiratory signals, trained and evaluated patient-wise; transformer, LSTM/GRU, U-Net and MLP comparisons; BreathNet, a 6k-parameter physiology-inspired model on the BW/AM/FM waveforms (resonator bank, harmonic summation, attention over time, learned smart fusion), and BreathNet-2, which adds explicit 1/f whitening, a coherent pathway sum, HMM rhythm tracking and a heart-rate prior (`--arch breathnet2 --variant ...`, feature kind `modwave_hr`).
- **Benchmark and report:** every method scored against a time-aligned impedance reference, with an offline HTML report and analysis notebooks.

## Layout

| Path | Contents |
|---|---|
| `preprocess/` | zero-phase SOS filters, resampling, ECG R-peak and PPG pulse detection |
| `estimate_rr/` | BW/AM/FM respiratory-signal extraction; CtO, CtA, PKS, ZeX, FTS, WCH, ARS, ARM, ARP estimators; smart / median fusion |
| `data_prep/` | GE monitor reader, impedance reference RR + quality flags, windowed dataset builder; wearable readers (SmartCare CSV, EDF) and heart-rate alignment to the monitor |
| `dl/` | spectrogram features, small CNN with label-distribution output, patient-wise training / evaluation |
| `scripts/` | traditional-method benchmark, trend scoring, report builder |
| `common/` | metrics (MAE, RMSE, Bland-Altman, coverage) and patient-wise splits |
| `tests/` | unit tests on synthetic signals: `python -m unittest discover -s tests -t .` |
| `notebooks/`, `reports/` | analysis notebooks and the generated offline report (see below) |
| `archive/` | retired code (2023 pipeline, 2024 refactor) and the reason each piece was retired (see `archive/README.md`) |

## Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Oucru-Innovations/Respiratory-Rate-PPG-ECG.git
   ```

2. **Navigate to the Directory:**
   ```bash
   cd Respiratory-Rate-PPG-ECG
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (numpy, scipy, pandas, pyarrow, torch, matplotlib; the notebooks also need jupyter and plotly.)

4. **Estimate RR from one window:**
   ```python
   from estimate_rr import estimate_rr
   res = estimate_rr(ppg_window, fs=100, signal_type="ppg", method="ar_median", fusion="smart")
   res.rr, res.per_signal   # fused rate, and the BW/AM/FM estimates
   ```

## Workflow (01NVa_Dengue GE monitor data)

The GE monitor export (`*waves.asc`) holds ECG, Pleth and thoracic impedance on
one clock at 300 Hz, so the impedance gives a time-aligned reference RR for
every PPG/ECG window without cross-device synchronisation.

```bash
# 1. raw text -> resampled segments (ECG 250 Hz, PPG 100 Hz, Resp 25 Hz)
python -m data_prep.build_dataset extract --raw-root /mnt/ssd/Workspace/Dengue/01NVa_Dengue/RAW_DATA --out dataset/01NVa_monitor --workers 4
# 2. windows + reference RR + quality flags (re-run for other window/stride)
python -m data_prep.build_dataset windows --out dataset/01NVa_monitor --window 60 --stride 30
# 3. spectrogram features (kind: modulation = BW/AM/FM spectrograms, raw = STFT of the signal)
python -m dl.precompute_features --windows dataset/01NVa_monitor/windows_w60_s30.parquet --signal ppg --kind modulation --out dataset/01NVa_monitor/features/ppg_mod_w60
# 4. train / evaluate (patient-wise 5-fold; --fold 0..4)
python -m dl.train --features dataset/01NVa_monitor/features/ppg_mod_w60 --out runs/ppg_mod_w60_f0 --fold 0
# 5. signal-processing baseline on the same test patients
python -m scripts.benchmark_traditional --windows dataset/01NVa_monitor/windows_w60_s30.parquet --signal ppg --split-json runs/ppg_mod_w60_f0/split.json --out runs/traditional_ppg_f0
```

`scripts/run_fold0.sh` runs steps 3–5 for PPG and ECG in one go; `scripts/run_arch_sweep_fold0.sh` trains the other architectures and `scripts/run_breathnet2_fold0.sh` the BreathNet-2 mechanisms one at a time and together.

**Five-fold cross-validation.** `scripts/run_cv.py` replays every fold-0 DL run on
folds 1–4 with the arguments saved in its checkpoint, and runs the
signal-processing benchmark on every test window of every fold, the trends and the
BreathNet read-out per fold. It has two resumable lanes (finished outputs are skipped;
status in `runs/cv_status.json`):

```bash
setsid nohup python -m scripts.run_cv --lane gpu > runs/cv_gpu.log 2>&1 &
setsid nohup python -m scripts.run_cv --lane cpu > runs/cv_cpu.log 2>&1 &
python -m scripts.run_cv --lane gpu --dry-run   # list the jobs
```

**Wearables.** The same patients wore a SmartCare wrist PPG and a Faros/ePatch chest
ECG. Their clocks cannot be trusted (start times in file names, unset device clocks,
a monitor clock off by hours, SmartCare sample clocks drifting up to ~0.4 %), so each
file is aligned to the monitor from the heart itself: heart-rate cross-correlation
over offset and drift, then beat-to-beat (respiratory sinus arrhythmia) checks in
5-min blocks; only stretches where the blocks agree to within 1 s are used
(`data_prep/wearable.py`). The monitor-trained models are then scored, without
retraining, on the wearable windows against the same impedance reference, each patient
by the fold model that held them out:

```bash
python -m data_prep.build_wearable align   --workers 12   # -> dataset/01NVa_wearable/alignment.csv
python -m data_prep.build_wearable extract --workers 8    # wearable signals on the monitor clock
python -m data_prep.build_wearable windows --workers 12   # windows_{ppg,ecg}_w60_s30.parquet
python -m scripts.wearable_eval all                       # -> runs/wearable/summary.json
```

## Report (offline)

```bash
bash scripts/build_report.sh     # 5-fold cross-validation -> reports/cv
bash scripts/build_report.sh 0   # fold 0 only            -> reports/fold0
```

Writes `report.html` (self-contained summary: results pooled over folds with
patient-bootstrap confidence intervals and per-fold spread, the method
corrections, network architectures, BreathNet across folds, sample windows, RR
trends, audit, caveats) and the executed notebooks `01_dataset_and_reference.html`,
`02_benchmark.html`, `03_case_studies_and_trends.html` (interactive plotly charts:
synchronized ECG/PPG/impedance with every method's detections and spectral picks,
trends and trending concordance) and `04_breathnet_explained.html` (what each part
of the 6k-parameter BreathNet learned, ablations, single-window anatomy, and whether
the same mechanism appears in every fold; its inputs come from
`python -m scripts.breathnet_analysis`); sources in `notebooks/` (they read the
folds from `REPORT_FOLDS`), `fig*.png` for slides and `summary.json` with every
number. Nothing is loaded from the network, so the folder can be zipped and shared
as is. Runs that have not finished show as pending; re-run the script when they do.

**Corrections to the published methods.** ECG/PPG-derived respiratory signals
carry a 1/f very-low-frequency background that the published rules mistake for
breathing (spectral methods under-estimated by ~13 bpm on held-out patients).
By default (`corrected=True`) spectral peaks are picked above a fitted
power-law background, ARP picks the pole with the strongest whitened power, and
CtA/PKS/ZeX run on a half pre-whitened signal; CtO is unchanged. On fold-0 test
patients (PPG, smart fusion) this takes ARM from MAE 13.2 to 1.2 bpm and ZeX
from 6.2 to 2.7. The correction hurts when breathing itself is below ~8 bpm.
`corrected=False` reproduces the published methods; details in
`estimate_rr/__init__.py` and the report.

**Physiological limit.** BW/AM/FM are sampled once per heartbeat, so they can
only represent breathing slower than half the heart rate. Windows with
RR/HR approaching 0.5 cannot be estimated reliably by any modulation-based method.
Report results stratified by RR/HR ratio.

## License
This project is licensed under the GNU General Public License v3.0. For more details, refer to the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Disclaimer
The information produced by this tool is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely based on information produced by this tool. Oucru does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced, please consult a healthcare professional.

For more information and detailed documentation, visit the [Respiratory-Rate-PPG-ECG GitHub repository](https://github.com/Oucru-Innovations/Respiratory-Rate-PPG-ECG).
