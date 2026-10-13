#!/usr/bin/env bash
# Rebuild the offline report: figures + report.html (kaggle env, has torch) and the
# executed notebooks exported to HTML (base env, has Jupyter + plotly).
# Run from the repo root:  bash scripts/build_report.sh [fold]
set -euo pipefail
FOLD=${1:-0}
OUT=reports/fold${FOLD}
KAGGLE_PY=${KAGGLE_PY:-~/anaconda3/envs/kaggle/bin/python}
JUPYTER=${JUPYTER:-~/anaconda3/bin/jupyter}

for nb in notebooks/0*.ipynb; do
  $JUPYTER nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$nb"
  # no CDN scripts: MathJax is unused and require.js would capture the embedded plotly bundle
  $JUPYTER nbconvert --to html "$nb" --output-dir "$OUT" --embed-images \
      --HTMLExporter.mathjax_url="" --HTMLExporter.require_js_url=""
  sed -i -E '/<script[^>]+src=""[^>]*>[[:space:]]*<\/script>/d' "$OUT/$(basename "${nb%.ipynb}").html"
done
# report.html last, so it links the notebook pages that now exist
$KAGGLE_PY -W ignore -m scripts.make_report --fold "$FOLD" --out "$OUT"
echo "open $OUT/report.html"
