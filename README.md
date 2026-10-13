<<<<<<< HEAD
# RespiratoryRate
Derive respiratory rate from wearable device signals (ECG &amp; PPG). The repo contains signal processing methods and deep learning methods for predicting respiratory rate.

## Layout

| Path | Contents |
|---|---|
| `preprocess/` | zero-phase SOS filters, resampling, ECG R-peak and PPG pulse detection |
| `estimate_rr/` | BW/AM/FM respiratory-signal extraction; CtO, CtA, PKS, ZeX, FTS, WCH, ARS, ARM, ARP estimators; smart / median fusion |
| `data_prep/` | GE monitor reader, impedance reference RR + quality flags, windowed dataset builder |
| `dl/` | spectrogram features, small CNN with label-distribution output, patient-wise training / evaluation |
| `scripts/benchmark_traditional.py` | every signal-processing method on the same windows and patients as the DL model |
| `common/` | metrics (MAE, RMSE, Bland-Altman, coverage) and patient-wise splits |
| `tests/` | unit tests on synthetic signals: `python -m unittest discover -s tests -t .` |
| `notebooks/`, `reports/` | analysis notebooks and the generated offline report (see below) |
| `archive/` | retired code and the reason each piece was retired (see `archive/README.md`) |

## Workflow (01NVa_Dengue GE monitor data)

The GE monitor export (`*waves.asc`) holds ECG, Pleth and thoracic impedance on
one clock at 300 Hz, so the impedance gives a time-aligned reference RR for
every PPG/ECG window without cross-device synchronisation.

```bash
# 1. raw text -> resampled segments (ECG 250 Hz, PPG 100 Hz, Resp 25 Hz)
python -m data_prep.build_dataset extract --raw-root /mnt/ssd/Workspace/Dengue/01NVa_Dengue/RAW_DATA --out dataset/01NVa_monitor --workers 4
# 2. windows + reference RR + quality flags (re-run for other window/stride)
python -m data_prep.build_dataset windows --out dataset/01NVa_monitor --window 60 --stride 15
# 3. spectrogram features (kind: modulation = BW/AM/FM spectrograms, raw = STFT of the signal)
python -m dl.precompute_features --windows dataset/01NVa_monitor/windows_w60_s15.parquet --signal ppg --kind modulation --out dataset/01NVa_monitor/features/ppg_mod_w60
# 4. train / evaluate (patient-wise 5-fold; --fold 0..4)
python -m dl.train --features dataset/01NVa_monitor/features/ppg_mod_w60 --out runs/ppg_mod_w60_f0 --fold 0
# 5. signal-processing baseline on the same test patients
python -m scripts.benchmark_traditional --windows dataset/01NVa_monitor/windows_w60_s15.parquet --signal ppg --split-json runs/ppg_mod_w60_f0/split.json --out runs/traditional_ppg_f0
```

Dependencies: `requirements.txt` (numpy, scipy, pandas, pyarrow, torch,
matplotlib; the `kaggle` conda env has all of them). The old pinned list is
`archive/requirements_legacy.txt`.

## Report (offline)

```bash
bash scripts/build_report.sh 0   # fold 0
```

Writes `reports/fold0/`: `report.html` (self-contained summary: results, the
method corrections, sample windows, RR trends, audit, caveats) and the executed
notebooks `01_dataset_and_reference.html`, `02_benchmark_fold0.html`,
`03_case_studies_and_trends.html` (interactive plotly charts: synchronized
ECG/PPG/impedance with every method's detections and spectral picks, trends and
trending concordance; sources in `notebooks/`),
`fig*.png` for slides and `summary.json` with every number. Nothing is loaded
from the network, so the folder can be zipped and shared as is. Runs that have
not finished show as pending; re-run the script when they do.

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
=======
>>>>>>> 6dff71e2045db392cb79ed6bb8996c199f53b90f

# Respiratory-Rate-PPG-ECG Repository

[![DOI](https://zenodo.org/badge/659031505.svg)](https://doi.org/10.5281/zenodo.14613763)

The **Respiratory-Rate-PPG-ECG** repository offers methods to estimate respiratory rate using signals from wearable devices, specifically Electrocardiogram (ECG) and Photoplethysmogram (PPG). It includes both signal processing techniques and deep learning models to predict respiratory rates.

## Features
- **Signal Processing Methods:** Techniques to extract respiratory rate from ECG and PPG signals.
- **Machine Learning Methods:** Include deep learning pre-trained models & other machine learning approaches for respiratory rate estimation.

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

## License
This project is licensed under the GNU General Public License v3.0. For more details, refer to the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Disclaimer
The information produced by this tool is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely based on information produced by this tool. Oucru does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced, please consult a healthcare professional.

For more information and detailed documentation, visit the [Respiratory-Rate-PPG-ECG GitHub repository](https://github.com/Oucru-Innovations/Respiratory-Rate-PPG-ECG).
