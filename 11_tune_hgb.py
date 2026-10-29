# Hyper-parameter search for boosting model B, successive halving written by hand so that
# every single fit is saved (the run can be killed and resumed).
# 16 random configurations on 50k rows -> best half on 100k -> 200k -> 400k, 3-fold CV,
# PR-AUC as the criterion.
import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from models import COLS, hgb_pipeline, load_2024, subsample

GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_leaf_nodes": [31, 63, 127, 255],
    "min_samples_leaf": [20, 50, 200, 1000],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "max_features": [0.5, 0.75, 1.0],
}
ROUNDS = [50_000, 100_000, 200_000, 400_000]
CKPT = "results/tuning_checkpoint.json"

cands = list(ParameterSampler(GRID, n_iter=16, random_state=0))
try:
    ck = json.load(open(CKPT))
except FileNotFoundError:
    ck = {}

X, y = load_2024()
alive = list(range(len(cands)))
for r, n in enumerate(ROUNDS):
    idx = subsample(y, n, seed=100 + r)
    S, ys = X.iloc[idx].reset_index(drop=True), y[idx]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(S, ys))
    for c in alive:
        for k, (tr, te) in enumerate(folds):
            key = f"r{r}_c{c}_f{k}"
            if key in ck:
                continue
            pipe = hgb_pipeline("B", max_iter=600, early_stopping=True, validation_fraction=0.1,
                                n_iter_no_change=30, **cands[c])
            pipe.fit(S.iloc[tr][COLS["B"]], ys[tr])
            p = pipe.predict_proba(S.iloc[te][COLS["B"]])[:, 1]
            ck[key] = {"pr_auc": float(average_precision_score(ys[te], p)),
                       "auc": float(roc_auc_score(ys[te], p)),
                       "n_iter": int(pipe.named_steps["hgb"].n_iter_)}
            json.dump(ck, open(CKPT, "w"), indent=1)
    score = {c: np.mean([ck[f"r{r}_c{c}_f{k}"]["pr_auc"] for k in range(3)]) for c in alive}
    ranked = sorted(alive, key=lambda c: -score[c])
    print(f"round {r} ({n} rows):", [(c, round(score[c], 4)) for c in ranked], flush=True)
    if r < len(ROUNDS) - 1:
        alive = ranked[:max(2, len(ranked) // 2)] if len(ranked) > 2 else ranked

best = ranked[0]
res = {"best_params": cands[best], "best_candidate": best,
       "pr_auc_400k_3fold": float(score[best]),
       "auc_400k_3fold": float(np.mean([ck[f"r{len(ROUNDS)-1}_c{best}_f{k}"]["auc"] for k in range(3)])),
       "n_iter": [ck[f"r{len(ROUNDS)-1}_c{best}_f{k}"]["n_iter"] for k in range(3)],
       "candidates": cands}
json.dump(res, open("results/tuning_best.json", "w"), indent=1, default=float)
print(res["best_params"], round(res["pr_auc_400k_3fold"], 4))
