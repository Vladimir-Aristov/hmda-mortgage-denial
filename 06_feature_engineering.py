# Does feature engineering help? Same subsample, folds and model as 04_cv_feature_sets.py.
# Order in the pipeline: Neutraliser -> FeatureBuilder -> Prep -> boosting,
# so the new features are built from already neutralised values (no missingness signal).

import json
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from features import TIERS, Neutraliser, Prep, build_X
from fe import FeatureBuilder
from leakage_audit import LeakageAuditor

df = pd.read_parquet("data/sample_2024.parquet")
ctrl = pd.read_parquet("data/controls_2024.parquet")
X = build_X(df, ctrl)
X["county_code"] = df["county_code"]
X["co-applicant_age"] = df["co-applicant_age"]
X["activity_year"] = df["activity_year"]
y = df["denied"].to_numpy()

rng = np.random.default_rng(42)   # same 400k subsample as before
idx = np.concatenate([rng.choice(np.where(y == k)[0], int(400_000 * (y == k).mean()), replace=False)
                      for k in (0, 1)])
S = X.iloc[idx].reset_index(drop=True)
ys = y[idx]

A = [c for c, t in TIERS.items() if t == "A"] + ["county_code", "co-applicant_age", "activity_year"]
U = ["debt_to_income_ratio", "loan_to_value_ratio", "property_value", "applicant_credit_score_type"]


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          categorical_features="from_dtype", random_state=0)


specs = {
    "A0": (A, None),
    "A1": (A, FeatureBuilder(underwriting=False)),
    "B0": (A + U, None),
    "B1": (A + U, FeatureBuilder(underwriting=True)),
    # leave-one-group-out on B1
    "B1-ratios": (A + U, FeatureBuilder(underwriting=True, drop_group="ratios")),
    "B1-lender": (A + U, FeatureBuilder(underwriting=True, drop_group="lender")),
    "B1-interactions": (A + U, FeatureBuilder(underwriting=True, drop_group="interactions")),
    "B1-program": (A + U, FeatureBuilder(underwriting=True, drop_group="program")),
    # final: no interactions, plus county data (07_external_data.py)
    "A1-noint": (A, FeatureBuilder(underwriting=False, interactions=False)),
    "A2": (A, FeatureBuilder(underwriting=False, interactions=False, external=True)),
    "B2": (A + U, FeatureBuilder(underwriting=True, interactions=False, external=True)),
}
only = sys.argv[1:] or ["A0", "A1", "B0", "B1"]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
out = {}
try:
    out = json.load(open("results/cv_fe.json"))
except FileNotFoundError:
    pass
for name in only:
    cols, fb = specs[name]
    steps = [("neutral", Neutraliser())]
    if fb is not None:
        steps.append(("fe", fb))
        cols_used = cols
    else:
        # baselines: exactly the columns used in 04_cv_feature_sets.py
        cols_used = [c for c in cols if c not in ("county_code", "co-applicant_age", "activity_year")]
    steps += [("prep", Prep()), ("hgb", model())]
    sc = cross_validate(Pipeline(steps), S[cols_used], ys, cv=cv, scoring=["roc_auc", "average_precision"])
    out[name] = {"auc": float(sc["test_roc_auc"].mean()), "auc_sd": float(sc["test_roc_auc"].std()),
                 "pr_auc": float(sc["test_average_precision"].mean()),
                 "pr_auc_sd": float(sc["test_average_precision"].std())}
    print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
    json.dump(out, open("results/cv_fe.json", "w"), indent=1)
