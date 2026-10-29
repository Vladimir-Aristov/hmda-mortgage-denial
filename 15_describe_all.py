# Descriptive statistics for every variable used by models A / B (raw + engineered), 2024,
# and a small-multiples figure: denial rate across deciles of the engineered features.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from fe import FeatureBuilder
from features import Neutraliser, TIERS
from models import COLS, load_2024

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "pdf.fonttype": 42, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

X, y = load_2024()
raw = X[COLS["B"]]
F = FeatureBuilder(underwriting=True, interactions=False).fit(Neutraliser().fit_transform(raw))
E = F.transform(Neutraliser().fit_transform(raw))
new = [c for c in E.columns if c not in raw.columns]

NAMES = {
    "loan_type": "Loan type", "loan_amount": "Loan amount (\\$)", "loan_term": "Loan term (months)",
    "intro_rate_period": "ARM intro period (months)", "interest_only_payment": "Interest-only flag",
    "balloon_payment": "Balloon flag", "negative_amortization": "Negative amortisation flag",
    "other_nonamortizing_features": "Other non-amortising flag", "conforming_loan_limit": "Conforming status",
    "total_units": "Number of units", "income": "Income (\\$000)", "applicant_age": "Applicant age band",
    "has_coapp": "Co-applicant", "submission_of_application": "Channel", "lei": "Lender (LEI)",
    "derived_msa-md": "MSA", "state_code": "State", "county_code": "County",
    "tract_minority_population_percent": "Tract minority share (\\%)",
    "tract_to_msa_income_percentage": "Tract / MSA income (\\%)",
    "ffiec_msa_md_median_family_income": "MSA median family income (\\$)",
    "tract_owner_occupied_units": "Tract owner-occupied units",
    "tract_one_to_four_family_homes": "Tract 1--4 family homes",
    "tract_median_age_of_housing_units": "Tract median housing age (yrs)",
    "co-applicant_age": "Co-applicant age band", "activity_year": None,
    "debt_to_income_ratio": "DTI (reported bins)", "loan_to_value_ratio": "Combined LTV (\\%)",
    "property_value": "Property value (\\$)", "applicant_credit_score_type": "Credit score model",
    "loan_to_income": "Loan / income", "income_rel_area": "Income / area median",
    "loan_rel_area": "Loan / area median income", "pti_new_loan": "New payment / income",
    "n_applicants": "Number of applicants", "lender_log_apps": "Lender size (log apps)",
    "lender_fha_share": "Lender FHA share", "lender_broker_share": "Lender broker share",
    "lender_median_lti": "Lender median loan/income", "dti_num": "DTI (numeric)",
    "dti_over_limit": "DTI minus program limit", "ltv_over_max": "LTV minus program max",
    "down_payment_to_income": "Down payment / income", "subordinate_gap": "CLTV minus first-lien LTV",
}


def fmt(v):
    if pd.isna(v):
        return "--"
    if abs(v) < 0.005:
        return "0.00"
    a = abs(v)
    if a >= 1e5:
        return f"{v / 1e3:,.0f}k"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


rows = []
for c in list(raw.columns) + new:
    if NAMES.get(c) is None:
        continue
    s = (raw[c] if c in raw.columns else E[c])
    tier = TIERS.get(c, "A" if c in ("county_code", "co-applicant_age") else "FE")
    miss = 100 * s.isna().mean()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        desc = f"{fmt(s.quantile(.01))} / {fmt(s.median())} / {fmt(s.quantile(.99))}"
        kind = "num"
    else:
        vc = s.astype(str).value_counts(normalize=True)
        desc = f"{s.nunique()} levels; top {100 * vc.iloc[0]:.0f}\\%"
        kind = "cat"
    rows.append((NAMES[c], tier, kind, f"{miss:.1f}", desc))
with open("results/t_all_vars.tex", "w") as f:
    last = None
    for r in rows:
        if last is not None and r[1] != last:
            f.write("\\midrule\n")
        last = r[1]
        f.write(" & ".join(r) + " \\\\\n")
print(len(rows), "variables")

# denial rate by decile of engineered features
show = ["loan_to_income", "pti_new_loan", "income_rel_area", "down_payment_to_income",
        "lender_fha_share", "dti_over_limit"]
fig, ax = plt.subplots(2, 3, figsize=(6.6, 3.4))
for a, c in zip(ax.ravel(), show):
    q = pd.qcut(E[c], 10, labels=False, duplicates="drop")
    g = pd.Series(y).groupby(q.to_numpy()).mean()
    a.plot(g.index + 1, g.values, color="#2a78d6", lw=2, marker="o", ms=3)
    a.set_title(NAMES[c].replace("\\", ""), loc="left", fontsize=8.5)
    a.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    a.set_xticks([1, 5, 10])
    a.grid(color="#e6e5e0", lw=0.6)
    a.set_axisbelow(True)
for a in ax[1]:
    a.set_xlabel("decile")
fig.tight_layout(h_pad=1.0, w_pad=1.2)
fig.savefig("figs/fig6_engineered.pdf", bbox_inches="tight")
