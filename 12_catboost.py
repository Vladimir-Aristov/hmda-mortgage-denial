# CatBoost on the same features, subsample and folds as the other models.
# CatBoost encodes categorical columns itself (ordered target statistics), so lender, MSA and
# state go in as raw strings instead of our target encoding.
import json, sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fe import FeatureBuilder
from features import Neutraliser
from models import COLS, CV, load_2024, subsample
from cat_utils import CatPrep


def cat_pipeline(m, **params):
    p = dict(iterations=800, learning_rate=0.1, depth=8, loss_function="Logloss",
             thread_count=2, random_seed=0, verbose=0)
    p.update(params)
    return Pipeline([("neutral", Neutraliser()),
                     ("fe", FeatureBuilder(underwriting=(m == "B"), interactions=False)),
                     ("cat", CatPrep())]), CatBoostClassifier(**p)


X, y = load_2024()
idx = subsample(y, 400_000)
S, ys = X.iloc[idx].reset_index(drop=True), y[idx]
try:
    out = json.load(open("results/cv_catboost.json"))
except FileNotFoundError:
    out = {}
for m in sys.argv[1:] or "B":
    for k, (tr, te) in enumerate(CV.split(S, ys)):
        key = f"{m}_fold{k}"
        if key in out:
            continue
        t = time.time()
        prep, model = cat_pipeline(m)
        Xtr = prep.fit_transform(S.iloc[tr][COLS[m]], ys[tr])
        Xte = prep.transform(S.iloc[te][COLS[m]])
        model.fit(Xtr, ys[tr], cat_features=prep.named_steps["cat"].cat_)
        p = model.predict_proba(Xte)[:, 1]
        out[key] = {"auc": float(roc_auc_score(ys[te], p)),
                    "pr_auc": float(average_precision_score(ys[te], p)), "sec": round(time.time() - t)}
        print(key, out[key], flush=True)
        json.dump(out, open("results/cv_catboost.json", "w"), indent=1)
for m in "AB":
    f = [v for k, v in out.items() if k.startswith(m + "_fold")]
    if len(f) == 5:
        print(m, "mean AUC %.4f (%.4f)  PR-AUC %.4f (%.4f)" % (
            np.mean([v["auc"] for v in f]), np.std([v["auc"] for v in f]),
            np.mean([v["pr_auc"] for v in f]), np.std([v["pr_auc"] for v in f])))
