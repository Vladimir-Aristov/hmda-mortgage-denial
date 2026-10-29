# Fix the hold-out sample once and for all (2nd submission requirement).
# Hold-out = every 2025 application in the sample (out-of-time test).
# It is not used for EDA, feature engineering, model selection or tuning; it is opened
# exactly once, for the final validation. The hash lets anyone check it was not changed.
import hashlib, json
import pandas as pd

path = "data/sample_2025.parquet"
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
n = len(pd.read_parquet(path, columns=["denied"]))
spec = {
    "holdout": "all 2025 HMDA home-purchase applications in the analysis sample",
    "file": path, "sha256": sha, "n_rows": n,
    "rule": "not used for anything except the final, one-time validation",
    "training_data": "data/sample_2024.parquet (all model selection by CV inside 2024)",
    "fixed_on": "2026-10-10",
}
json.dump(spec, open("results/holdout_spec.json", "w"), indent=1)
print(json.dumps(spec, indent=1))
