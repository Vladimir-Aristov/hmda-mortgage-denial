# Interpretable baseline: penalised logistic regression, same 400k subsample and folds as 04/06.
import json, sys
import numpy as np
from sklearn.model_selection import cross_validate
from models import COLS, CV, load_2024, logit_pipeline, subsample

X, y = load_2024()
idx = subsample(y, 400_000)
S, ys = X.iloc[idx].reset_index(drop=True), y[idx]
try:
    out = json.load(open("results/cv_logit.json"))
except FileNotFoundError:
    out = {}
for m in (sys.argv[1] if len(sys.argv) > 1 else "AB"):
    for C in [0.01, 0.1, 1.0]:
        sc = cross_validate(logit_pipeline(m, C=C), S[COLS[m]], ys, cv=CV,
                            scoring=["roc_auc", "average_precision"])
        r = {"auc": float(sc["test_roc_auc"].mean()), "auc_sd": float(sc["test_roc_auc"].std()),
             "pr_auc": float(sc["test_average_precision"].mean()),
             "pr_auc_sd": float(sc["test_average_precision"].std())}
        out[f"{m}_C{C}"] = r
        print(m, C, {k: round(v, 4) for k, v in r.items()}, flush=True)
        json.dump(out, open("results/cv_logit.json", "w"), indent=1)
