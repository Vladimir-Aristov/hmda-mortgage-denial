# Can a derived feature tell which rows had LTV / property value imputed?
# If it can, the "missing" signal (which marks early rejections) sneaks back into the model.
# Compare the old imputation (each field gets its own median) with the consistent one.
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fe import FeatureBuilder
from features import Neutraliser
from models import COLS

d = pd.read_parquet("data/sample_2024.parquet").sample(600_000, random_state=3)
d["has_coapp"] = (d["co-applicant_age"].astype(str) != "9999").astype(int)
X = d[COLS["B"]]
imp = (d.loan_to_value_ratio.isna() | d.property_value.isna()).to_numpy()


def old_neutral(X):
    Z = Neutraliser().fit(X).transform(X.drop(columns=["loan_amount"])).join(X[["loan_amount"]])
    return Z[X.columns]   # without loan_amount the consistent branch is skipped -> plain medians


out = {"imputed_share": float(imp.mean())}
for name, Z in [("independent medians (old)", old_neutral(X)), ("consistent (new)", Neutraliser().fit(X).transform(X))]:
    F = FeatureBuilder(underwriting=True, interactions=False).fit(Z).transform(Z)
    r = {}
    for c in ["subordinate_gap", "down_payment_to_income", "ltv_over_max"]:
        a = roc_auc_score(imp, F[c].fillna(-999))
        r[c] = round(max(a, 1 - a), 3)
    out[name] = r
    print(name, r)
for c in ["loan_amount", "income"]:   # never imputed: how different the imputed rows are anyway
    a = roc_auc_score(imp, d[c].fillna(-999))
    out.setdefault("never imputed", {})[c] = round(max(a, 1 - a), 3)
print(out)
json.dump(out, open("results/imputation_check.json", "w"), indent=1)
