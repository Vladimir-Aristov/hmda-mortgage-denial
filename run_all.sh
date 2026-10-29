#!/bin/sh
# Re-runs every model result in order (feature sets, logit, tuning, CatBoost CV and final models).
# Each step leaves a marker in results/done/, so the chain can be restarted after an interruption.
cd "$(dirname "$0")"
mkdir -p results/done
step() { name=$1; shift; [ -f results/done/$name ] && return; echo "== $name $(date +%H:%M)"; "$@" && touch results/done/$name; }
step fe06    python3 06_feature_engineering.py B0 B1 B1-ratios B1-lender B1-program B1-interactions B2
step sets04  python3 04_cv_feature_sets.py
step logit09 python3 09_logit.py B
step lc10    python3 10_learning_curve.py B
step tune11  sh -c "rm -f results/tuning_checkpoint.json; python3 11_tune_hgb.py"
step eval11cv   python3 11b_eval_tuned.py cv
step eval11full python3 11b_eval_tuned.py full
step tunecb14 sh -c "rm -f results/tuning_catboost.json; python3 14_tune_catboost.py"
step cbcv12  python3 12_catboost.py B
step finalB  sh -c "rm -rf results/catboost_info_B results/final_catboost_B.json; ITERS=4000 THREADS=8 RAMLIM=12gb python3 17_final_catboost.py B"
step finalA  sh -c "ITERS=4000 THREADS=8 RAMLIM=12gb python3 17_final_catboost.py A"
step desc15  python3 15_describe_all.py
echo "== all done $(date +%H:%M)"
