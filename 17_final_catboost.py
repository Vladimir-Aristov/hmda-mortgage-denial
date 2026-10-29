# Final models: CatBoost with the tuned settings on the whole 2024 training pool.
# Split (same as 10_learning_curve.py): 300k fixed validation rows, the rest is the pool;
# 100k rows of the pool are kept aside for early stopping, the model trains on the remainder.
# Snapshots are written every 5 minutes, so a killed run continues where it stopped.
# Usage: python 17_final_catboost.py B [n_train]   (then A)
# On a bigger machine, e.g. M3 / 16 GB, all training rows:
#   THREADS=8 RAMLIM=12gb ITERS=4000 python 17_final_catboost.py B
import gc, json, os, sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fe import FeatureBuilder
from features import Neutraliser
from models import COLS

m = sys.argv[1] if len(sys.argv) > 1 else "B"
n_train = int(sys.argv[2]) if len(sys.argv) > 2 else None   # cap on training rows (memory)
probe = len(sys.argv) > 3 and sys.argv[3] == "probe"        # 5 iterations, to measure memory
cfg = json.load(open("results/tuning_catboost_best.json"))["config"]

# lean loading: only the columns the model needs (the full frame with controls ran out of memory)
cols = COLS[m]
raw_cols = [c for c in cols if c != "has_coapp"]
X = pd.read_parquet("data/sample_2024.parquet", columns=raw_cols + ["denied"])
X["has_coapp"] = (X["co-applicant_age"].astype(str) != "9999").astype("int8")
y = X.pop("denied").to_numpy()
perm = np.random.default_rng(1).permutation(len(y))
val, pool = perm[:300_000], perm[300_000:]
es, tr = pool[:100_000], pool[100_000:]
if n_train:
    tr = tr[:n_train]

neutral = Neutraliser().fit(X.iloc[tr][cols])
fb = FeatureBuilder(underwriting=(m == "B"), interactions=False).fit(neutral.transform(X.iloc[tr][cols]))


def build(rows):
    return fb.transform(neutral.transform(X.iloc[rows][cols]))


Ftr = build(tr)
cats = [c for c in Ftr.columns if not pd.api.types.is_numeric_dtype(Ftr[c])]
levels = {c: pd.Index(Ftr[c].astype(str).unique()) for c in cats}


def codes(F):
    # categorical columns -> integer codes (unseen level -> -1), numbers -> float32
    for c in F.columns:
        if c in cats:
            F[c] = levels[c].get_indexer(F[c].astype(str)).astype("int32")
        else:
            F[c] = F[c].astype("float32")
    return F


Ftr = codes(Ftr)
feat_names = list(Ftr.columns)
P_tr = Pool(Ftr, y[tr], cat_features=cats)
del Ftr
gc.collect()
P_es = Pool(codes(build(es)), y[es], cat_features=cats)
Fval = codes(build(val))
rules_raw = X.iloc[val][["debt_to_income_ratio", "loan_to_value_ratio"]].copy() if m == "B" else None
del X
gc.collect()

params = dict(iterations=int(os.environ.get("ITERS", 5000)),   # 4000 in the final runs; early stopping ends earlier
              loss_function="Logloss", thread_count=int(os.environ.get("THREADS", 2)), random_seed=0, verbose=200,
              od_type="Iter", od_wait=100, use_best_model=True,
              save_snapshot=True, snapshot_file=f"results/snap_final_{m}.cbsnapshot", snapshot_interval=300,
              train_dir=f"results/catboost_info_{m}", used_ram_limit=os.environ.get("RAMLIM", "3gb"))
params.update({k: v for k, v in cfg.items() if k != "iterations"})
if probe:
    params.update(iterations=5, save_snapshot=False)
t = time.time()
model = CatBoostClassifier(**params)
model.fit(P_tr, eval_set=P_es)
if probe:
    sys.exit(0)
model.save_model(f"results/final_catboost_{m}.cbm")

p = model.predict_proba(Pool(Fval, cat_features=cats))[:, 1]
yv = y[val]
pd.DataFrame({"row": val, "y": yv, "p": p}).to_parquet(f"results/val_pred_{m}.parquet", index=False)


def prec_at(rec_target):
    order = np.argsort(-p)
    tp = np.cumsum(yv[order])
    k = np.searchsorted(tp, rec_target * yv.sum())
    return {"precision": float(tp[k] / (k + 1)), "flag_share": float((k + 1) / len(p))}


res = {"model": m, "config": {**cfg, "iterations": params["iterations"]},  # the tuned 800 is only a CV budget
       "n_train": len(tr), "best_iter": int(model.get_best_iteration()),
       "minutes": round((time.time() - t) / 60, 1),
       "auc": float(roc_auc_score(yv, p)), "pr_auc": float(average_precision_score(yv, p)),
       "brier": float(brier_score_loss(yv, p)), "log_loss": float(log_loss(yv, p)),
       "mean_pred": float(p.mean()), "base_rate": float(yv.mean()),
       "prec_at_recall": {r: prec_at(r) for r in (0.3, 0.49, 0.7)}}
imp = pd.Series(model.get_feature_importance(), index=feat_names).sort_values(ascending=False)
res["importance_top15"] = imp.head(15).round(2).to_dict()

if m == "B":   # the simple rules from the first submission, on the same validation rows
    raw = rules_raw
    dti = raw["debt_to_income_ratio"].astype(str)
    dnum = pd.to_numeric(dti, errors="coerce")
    high = dti.isin(["50%-60%", ">60%"]) | (dnum >= 50)
    ltv = raw["loan_to_value_ratio"]
    rules = {"DTI>=50 or LTV>100": high | (ltv > 100), "DTI>43 or LTV>97": high | (dnum > 43) | (ltv > 97)}
    res["rules"] = {}
    for k, r in rules.items():
        r = r.to_numpy()
        tp = (r & (yv == 1)).sum()
        res["rules"][k] = {"recall": float(tp / yv.sum()), "precision": float(tp / r.sum()),
                           "flag_share": float(r.mean())}

json.dump(res, open(f"results/final_catboost_{m}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
