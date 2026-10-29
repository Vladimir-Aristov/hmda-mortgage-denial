#!/bin/bash
# Trains the final CatBoost models A and B on all 2024 training rows on this Mac.
# Usage (in Terminal, from the repository root):  bash run_on_mac.sh
# Safe to re-run: finished steps are skipped, an interrupted model resumes from its snapshot.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results

# --- Python: prefer a version catboost has wheels for ---------------------------------
BASE=""
for v in python3.12 python3.11 python3.10 python3.13 python3; do
  if command -v "$v" >/dev/null 2>&1; then BASE="$(command -v "$v")"; break; fi
done
[ -n "$BASE" ] || { echo "No python3 found. Install it: brew install python@3.12"; exit 1; }
echo "== base python: $BASE ($("$BASE" --version))"

PY="$PWD/.venv/bin/python"
# (re)create the venv if it is missing or broken
if ! "$PY" -c "import sys" >/dev/null 2>&1; then
  rm -rf .venv
  "$BASE" -m venv .venv
fi
# always call pip as "python -m pip": the pip launcher breaks on paths with spaces ("big data")
if ! "$PY" -c "import pandas, pyarrow, numpy, sklearn, catboost" >/dev/null 2>&1; then
  echo "== installing packages into .venv"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install pandas pyarrow numpy scikit-learn catboost
fi
"$PY" -c "import pandas, sklearn, catboost; print('== packages ok: pandas', pandas.__version__, '| sklearn', sklearn.__version__, '| catboost', catboost.__version__)"

# --- data ------------------------------------------------------------------------------
# The download goes to data/hmda_2024.csv.part and resumes after an interruption; only a
# file of the exact expected size is renamed to data/hmda_2024.csv. A finished CSV or an
# already built sample is never downloaded again.
URL="https://files.ffiec.cfpb.gov/data-browser/datasets/2024/filtered-queries/one-year/9717a697fbe75ba0a2b00a5e0c8fb005.csv"
EXPECTED=1649130871
CSV=data/hmda_2024.csv
size() { stat -f%z "$1" 2>/dev/null || echo 0; }

if [ -f data/sample_2024.parquet ]; then
  echo "== sample already built, skipping download"
else
  # an old full download under the final name: keep it if the size is right
  if [ -f "$CSV" ] && [ "$(size "$CSV")" -ne "$EXPECTED" ]; then
    mv "$CSV" "$CSV.part"              # unfinished file from the old script -> resume it
  fi
  if [ ! -f "$CSV" ]; then
    echo "== downloading HMDA 2024 (~1.6 GB); already have $(( $(size "$CSV.part") / 1000000 )) MB"
    for attempt in 1 2 3 4 5; do
      if curl -L --fail --retry 5 --retry-delay 5 -C - -o "$CSV.part" "$URL"; then break; fi
      # server refused to resume (or a broken partial file): start that part again
      if [ "$(size "$CSV.part")" -gt "$EXPECTED" ]; then rm -f "$CSV.part"; fi
      echo "   download interrupted, retrying ($attempt)"; sleep 5
    done
    if [ "$(size "$CSV.part")" -ne "$EXPECTED" ]; then
      echo "Download incomplete ($(size "$CSV.part") of $EXPECTED bytes). Run the script again to continue."
      exit 1
    fi
    mv "$CSV.part" "$CSV"
  fi
  echo "== building the sample"
  "$PY" 01_build_sample.py 2024
fi

# --- models ----------------------------------------------------------------------------
for M in B A; do
  if [ ! -f "results/final_catboost_${M}_full.json" ]; then
    echo "== training model $M on all rows  ($(date +%H:%M))"
    caffeinate -i env THREADS=8 RAMLIM=12gb ITERS=4000 "$PY" 17_final_catboost.py "$M"
    cp "results/final_catboost_${M}.json" "results/final_catboost_${M}_full.json"
  fi
done
echo "== done ($(date +%H:%M)). Results: results/final_catboost_A_full.json and _B_full.json"
