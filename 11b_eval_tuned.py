# Tuned boosting in the standard 5-fold CV (same 400k subsample as 04/06/09/12),
# and on the full 2024 training pool with the fixed validation set of the learning curve.
import json, sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_validate
from models import COLS, CV, hgb_pipeline, load_2024, subsample

best = json.load(open("results/tuning_best.json"))["best_params"]
kw = dict(max_iter=600, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, **best)
X, y = load_2024()
try:
    out = json.load(open("results/cv_tuned.json"))
except FileNotFoundError:
    out = {}
what = sys.argv[1] if len(sys.argv) > 1 else "cv"
if what == "cv":
    idx = subsample(y, 400_000)
    S, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    for m in "BA":
        sc = cross_validate(hgb_pipeline(m, **kw), S[COLS[m]], ys, cv=CV, scoring=["roc_auc", "average_precision"])
        out[f"{m}_cv400k"] = {"auc": float(sc["test_roc_auc"].mean()), "auc_sd": float(sc["test_roc_auc"].std()),
                              "pr_auc": float(sc["test_average_precision"].mean()),
                              "pr_auc_sd": float(sc["test_average_precision"].std())}
        print(m, out[f"{m}_cv400k"], flush=True)
        json.dump(out, open("results/cv_tuned.json", "w"), indent=1)
else:   # full training pool, same split as 10_learning_curve.py
    perm = np.random.default_rng(1).permutation(len(y))
    val, pool = perm[:300_000], perm[300_000:]
    for m in "BA":
        pipe = hgb_pipeline(m, **kw).fit(X.iloc[pool][COLS[m]], y[pool])
        p = pipe.predict_proba(X.iloc[val][COLS[m]])[:, 1]
        out[f"{m}_full"] = {"auc": float(roc_auc_score(y[val], p)),
                            "pr_auc": float(average_precision_score(y[val], p)),
                            "n_iter": int(pipe.named_steps["hgb"].n_iter_)}
        print(m, out[f"{m}_full"], flush=True)
        json.dump(out, open("results/cv_tuned.json", "w"), indent=1)
