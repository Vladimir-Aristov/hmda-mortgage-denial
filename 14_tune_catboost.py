# CatBoost tuning. One split drawn from the 2024 training pool (the fixed 300k validation set
# of 10_learning_curve.py is left alone): 100k rows to train, 100k rows for early stopping
# and scoring. 8 random configurations, each saved as soon as it finishes (resumable).
import json, time
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

from fe import FeatureBuilder
from features import Neutraliser
from models import COLS, load_2024
from cat_utils import CatPrep

GRID = {"depth": [6, 8, 10], "learning_rate": [0.05, 0.1], "l2_leaf_reg": [1, 3, 10],
        "max_ctr_complexity": [1, 4], "random_strength": [0.5, 1, 2]}
CKPT = "results/tuning_catboost.json"

X, y = load_2024()
perm = np.random.default_rng(1).permutation(len(y))
pool = perm[300_000:]                       # same pool as the learning curve
rng = np.random.default_rng(7)
pick = rng.permutation(pool)[:200_000]
tr, es = pick[:100_000], pick[100_000:]

cols = COLS["B"]
neutral = Neutraliser().fit(X.iloc[tr][cols])
fb = FeatureBuilder(underwriting=True, interactions=False).fit(neutral.transform(X.iloc[tr][cols]))
prep = CatPrep()
Xtr = prep.fit_transform(fb.transform(neutral.transform(X.iloc[tr][cols])))
Xes = prep.transform(fb.transform(neutral.transform(X.iloc[es][cols])))

try:
    ck = json.load(open(CKPT))
except FileNotFoundError:
    ck = {}
configs = [{"iterations": 800, "learning_rate": 0.1, "depth": 8}]   # the default used so far
configs += list(ParameterSampler(GRID, n_iter=7, random_state=0))
for i, cfg in enumerate(configs):
    key = str(i)
    if key in ck:
        continue
    t = time.time()
    p = dict(iterations=3000, loss_function="Logloss", thread_count=2, random_seed=0, verbose=0,
             od_type="Iter", od_wait=100, use_best_model=True)
    p.update(cfg)
    m = CatBoostClassifier(**p)
    m.fit(Xtr, y[tr], cat_features=prep.cat_, eval_set=(Xes, y[es]))
    pr = m.predict_proba(Xes)[:, 1]
    ck[key] = {"config": cfg, "pr_auc": float(average_precision_score(y[es], pr)),
               "auc": float(roc_auc_score(y[es], pr)), "best_iter": int(m.get_best_iteration()),
               "sec": round(time.time() - t)}
    print(key, ck[key], flush=True)
    json.dump(ck, open(CKPT, "w"), indent=1)

# selection rule: among configurations within 0.002 PR-AUC of the best (differences that
# small are noise), take the fastest one -- the final model trains on 27x more rows
top = max(v["pr_auc"] for v in ck.values())
best = min((v for v in ck.values() if v["pr_auc"] >= top - 0.002), key=lambda v: v["sec"])
best["rule"] = "fastest configuration within 0.002 PR-AUC of the best"
json.dump(best, open("results/tuning_catboost_best.json", "w"), indent=1)
print("best", best)
