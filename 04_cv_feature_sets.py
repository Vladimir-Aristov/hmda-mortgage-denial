# How much do the questionable features add? 5-fold CV, HistGradientBoosting, 2024 only.
#  A  application-time fields only (missing loan term neutralised)
#  B  A + underwriting-time values (DTI, LTV, property value, score model), missingness neutralised
#  C  A + U with raw missing / "not applicable" codes and the AUS field
#  D  everything incl. post-decision fields -- what a naive model would report
# The leakage auditor sits inside the pipeline, so it is re-fitted on every training fold.

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from features import TIERS, Neutraliser, Prep, build_X
from leakage_audit import LeakageAuditor

df = pd.read_parquet("data/sample_2024.parquet")
ctrl = pd.read_parquet("data/controls_2024.parquet")
X = build_X(df, ctrl)
y = df["denied"].to_numpy()

# 400k stratified subsample to keep the run time reasonable
rng = np.random.default_rng(42)
idx = np.concatenate([rng.choice(np.where(y == k)[0], int(400_000 * (y == k).mean()), replace=False)
                      for k in (0, 1)])
S = X.iloc[idx].reset_index(drop=True)
ys = y[idx]

A = [c for c, t in TIERS.items() if t == "A"]
U = ["debt_to_income_ratio", "loan_to_value_ratio", "property_value", "applicant_credit_score_type"]

specs = {   # columns, neutralise missingness?, auditor in the pipeline?
    "A: application-time only": (A, True, False),
    "B: + underwriting values, missingness neutralised": (A + U, True, False),
    "C: + raw missing/N-A codes and AUS": (A + U + ["aus-1"], False, False),
    "D: + post-decision fields (naive)": (list(X.columns), False, False),
    "E: D with the leakage auditor in the pipeline": (list(X.columns), False, True),
}

cv = StratifiedKFold(5, shuffle=True, random_state=0)
res = {}
for name, (cols, neutral, audit) in specs.items():
    steps = []
    if neutral:
        steps.append(("neutral", Neutraliser()))
    if audit:
        steps.append(("audit", LeakageAuditor(n_splits=3)))
    steps += [("prep", Prep()),
              ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                                     categorical_features="from_dtype", random_state=0))]
    sc = cross_validate(Pipeline(steps), S[cols], ys, cv=cv,
                        scoring=["roc_auc", "average_precision"], return_estimator=audit)
    res[name] = {"n_features": len(cols),
                 "auc": float(sc["test_roc_auc"].mean()), "auc_sd": float(sc["test_roc_auc"].std()),
                 "pr_auc": float(sc["test_average_precision"].mean()),
                 "pr_auc_sd": float(sc["test_average_precision"].std())}
    if audit:
        res[name]["dropped_per_fold"] = [e.named_steps["audit"].dropped_ for e in sc["estimator"]]
    print(name, {k: v for k, v in res[name].items() if k != "dropped_per_fold"}, flush=True)

res["n_cv"] = len(idx)
res["base_rate"] = float(ys.mean())
with open("results/cv_feature_sets.json", "w") as f:
    json.dump(res, f, indent=1)
with open("results/t_cv.tex", "w") as f:
    for name, r in res.items():
        if isinstance(r, dict):
            f.write(f"{name} & {r['n_features']} & {r['auc']:.3f} ({r['auc_sd']:.3f}) & "
                    f"{r['pr_auc']:.3f} ({r['pr_auc_sd']:.3f}) \\\\\n")
