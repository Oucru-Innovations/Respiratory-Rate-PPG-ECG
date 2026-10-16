#!/usr/bin/env bash
# Rebuild the offline report: figures + report.html (kaggle env, has torch) and the
# executed notebooks exported to HTML (base env, has Jupyter + plotly).
# Run from the repo root:
#   bash scripts/build_report.sh        # 5-fold cross-validation -> reports/cv
#   bash scripts/build_report.sh 0      # one fold                -> reports/fold0
set -euo pipefail
if [ $# -ge 1 ]; then FOLDS="$1"; OUT=reports/fold$1; else FOLDS="0 1 2 3 4"; OUT=reports/cv; fi
KAGGLE_PY=${KAGGLE_PY:-~/anaconda3/envs/kaggle/bin/python}
JUPYTER=${JUPYTER:-~/anaconda3/bin/jupyter}
mkdir -p "$OUT"

# BreathNet internals for notebook 04 (needs torch): refresh when a BreathNet model is newer than the export
for k in $FOLDS; do
  for sig in ppg ecg; do
    run=runs/${sig}_modwave_w60_breathnet_f${k}/best.pt
    out=runs/breathnet_f${k}/${sig}.npz
    if [ -f "$run" ] && { [ ! -f "$out" ] || [ "$run" -nt "$out" ]; }; then
      $KAGGLE_PY -W ignore -c "from scripts.breathnet_analysis import export; export('$sig', fold=$k)" \
        || echo "warning: BreathNet export failed for $sig fold $k (needs the fold's CNN too)"
    fi
  done
done

# BreathNet-2 screening table + learned physiology for notebook 04 (fold 0; needs torch)
$KAGGLE_PY -W ignore -m scripts.breathnet2_summary --fold 0 > runs/breathnet2_fold0_summary.log 2>&1 || echo "warning: BreathNet-2 summary failed"

for nb in notebooks/0*.ipynb; do
  REPORT_FOLDS="$FOLDS" $JUPYTER nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
  # no CDN scripts: MathJax is unused and require.js would capture the embedded plotly bundle
  # --no-input: the pages show text, tables and charts; the code stays in notebooks/*.ipynb
  $JUPYTER nbconvert --to html "$nb" --output-dir "$OUT" --embed-images --no-input \
      --HTMLExporter.mathjax_url="" --HTMLExporter.require_js_url=""
  sed -i -E '/<script[^>]+src=""[^>]*>[[:space:]]*<\/script>/d' "$OUT/$(basename "${nb%.ipynb}").html"
done
# report.html last, so it links the notebook pages that now exist
$KAGGLE_PY -W ignore -m scripts.make_report --folds $FOLDS --out "$OUT"
echo "open $OUT/report.html"
