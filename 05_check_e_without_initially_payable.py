# Check: is variant E above C because "initially payable" (flagged review, not drop) is still in?
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from features import Prep, build_X
from leakage_audit import LeakageAuditor

df = pd.read_parquet("data/sample_2024.parquet")
ctrl = pd.read_parquet("data/controls_2024.parquet")
X = build_X(df, ctrl)
y = df["denied"].to_numpy()
rng = np.random.default_rng(42)   # same subsample as 04
idx = np.concatenate([rng.choice(np.where(y == k)[0], int(400_000 * (y == k).mean()), replace=False)
                      for k in (0, 1)])
S = X.iloc[idx].reset_index(drop=True)
ys = y[idx]
cols = [c for c in X.columns if c != "initially_payable_to_institution"]
pipe = Pipeline([("audit", LeakageAuditor(n_splits=3)), ("prep", Prep()),
                 ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                                        categorical_features="from_dtype", random_state=0))])
sc = cross_validate(pipe, S[cols], ys, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                    scoring=["roc_auc", "average_precision"])
r = {"auc": float(sc["test_roc_auc"].mean()), "auc_sd": float(sc["test_roc_auc"].std()),
     "pr_auc": float(sc["test_average_precision"].mean()), "pr_auc_sd": float(sc["test_average_precision"].std()),
     "n_features": len(cols)}
print(r)
json.dump(r, open("results/cv_e_minus_initially_payable.json", "w"), indent=1)
with open("results/t_cv_extra.tex", "w") as f:
    f.write(f"E without ``initially payable'' & {r['n_features']} & {r['auc']:.3f} ({r['auc_sd']:.3f}) & "
            f"{r['pr_auc']:.3f} ({r['pr_auc_sd']:.3f}) \\\\\n")
