# Leakage audit of the engineered features (built after neutralisation), grouped by lender.
import pandas as pd
from features import Neutraliser, TIERS
from fe import FeatureBuilder
from leakage_audit import LeakageAuditor

d = pd.read_parquet("data/sample_2024.parquet")
d["has_coapp"] = (d["co-applicant_age"].astype(str) != "9999").astype(int)
A = [c for c, t in TIERS.items() if t == "A"] + ["county_code", "co-applicant_age", "activity_year"]
U = ["debt_to_income_ratio", "loan_to_value_ratio", "property_value", "applicant_credit_score_type"]
Z = Neutraliser().fit_transform(d[A + U])
F = FeatureBuilder(underwriting=True, interactions=False).fit(Z).transform(Z)
new = [c for c in F.columns if c not in Z.columns]
a = LeakageAuditor().fit(F[new], d["denied"], group=d["lei"].astype(str).rename("lei"))
r = a.report_[["oof_auc", "miss_lift", "max_level_rate", "within_group_lift", "flag"]]
print(r.round(3).to_string())
r.to_csv("results/audit_fe.csv")
