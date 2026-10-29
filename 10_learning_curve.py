# Learning curve: does the boosting model keep improving with more training rows?
# Fixed validation set (300k random 2024 rows), nested training subsets from the rest.
import json, sys, time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from models import COLS, hgb_pipeline, load_2024

X, y = load_2024()
rng = np.random.default_rng(1)
perm = rng.permutation(len(y))
val, pool = perm[:300_000], perm[300_000:]
sizes = [100_000, 200_000, 400_000, 800_000, 1_600_000, len(pool)]
try:
    out = json.load(open("results/learning_curve.json"))
except FileNotFoundError:
    out = {}
for m in sys.argv[1:] or "BA":
    for n in sizes:
        key = f"{m}_{n}"
        if key in out:
            continue
        tr = pool[:n]
        t = time.time()
        pipe = hgb_pipeline(m).fit(X.iloc[tr][COLS[m]], y[tr])
        p = pipe.predict_proba(X.iloc[val][COLS[m]])[:, 1]
        out[key] = {"n": n, "auc": float(roc_auc_score(y[val], p)),
                    "pr_auc": float(average_precision_score(y[val], p)), "sec": round(time.time() - t)}
        print(key, out[key], flush=True)
        json.dump(out, open("results/learning_curve.json", "w"), indent=1)
