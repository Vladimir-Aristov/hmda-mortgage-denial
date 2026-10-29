# Feature timing / leakage checks (comments from the first submission):
#  1) AUS "not applicable", LTV and missing values -- are they known at application time?
#  2) automatic leakage audit (leakage_audit.py) + positive controls
# (the CV comparison of feature sets is in 04_cv_feature_sets.py)
# Only 2024 is used here, 2025 stays untouched as the test year.

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "pdf.fonttype": 42, "font.size": 8.5})

from features import TIERS, build_X, dti_valid
from leakage_audit import LeakageAuditor

df = pd.read_parquet("data/sample_2024.parquet")
ctrl = pd.read_parquet("data/controls_2024.parquet")   # post-decision fields, same rows
y = df["denied"].to_numpy()
res = {}

# %% helper columns
reasons = {"1": "Debt-to-income", "2": "Employment history", "3": "Credit history",
           "4": "Collateral", "5": "Insufficient cash", "6": "Unverifiable info",
           "7": "Application incomplete", "9": "Other"}
df["outcome"] = df["denial_reason-1"].astype(str).map(reasons).where(df.denied == 1, "Approved")
df["aus_na"] = df["aus-1"].astype(str) == "6"
df["cs_na"] = df["applicant_credit_score_type"].astype(str) == "9"
df["ltv_na"] = df["loan_to_value_ratio"].isna()
df["pv_na"] = df["property_value"].isna()
df["dti_na"] = ~dti_valid(df["debt_to_income_ratio"])
df["term_na"] = df["loan_term"].isna()

# %% 1a. missing / N/A shares by outcome and denial reason
flags = ["aus_na", "cs_na", "ltv_na", "pv_na", "dti_na", "term_na"]
order = ["Approved", "Debt-to-income", "Collateral", "Credit history", "Application incomplete",
         "Insufficient cash", "Unverifiable info", "Employment history", "Other"]
tab = df.groupby("outcome")[flags].mean().mul(100).reindex(order)
tab["n"] = df.groupby("outcome").size().reindex(order)
res["miss_by_reason"] = tab.round(2).to_dict(orient="index")
with open("results/t_miss_by_reason.tex", "w") as f:
    for name, r in tab.iterrows():
        f.write(f"{name} & {r.n:,.0f} & " + " & ".join(f"{r[c]:.1f}" for c in flags) + " \\\\\n")

# %% 1b. AUS N/A inside lenders that run AUS on (almost) every approved loan
g = df.groupby(["lei", "denied"], observed=True)["aus_na"].agg(["mean", "size"]).unstack()
g.columns = ["na_app", "na_den", "n_app", "n_den"]
g = g.dropna()
g = g[(g.n_app >= 100) & (g.n_den >= 20)]
users = g[g.na_app < 0.02].index
never = g[g.na_app > 0.90].index
u = df[df.lei.isin(users)]
nv = df[df.lei.isin(never)]
res["aus"] = {
    "n_lenders_checked": len(g), "n_users": len(users), "users_app_share": len(u) / len(df),
    "users_na_approved": float(u.loc[u.denied == 0, "aus_na"].mean()),
    "users_na_denied": float(u.loc[u.denied == 1, "aus_na"].mean()),
    "users_denial_if_na": float(u.loc[u.aus_na, "denied"].mean()),
    "users_denial_if_aus": float(u.loc[~u.aus_na, "denied"].mean()),
    "n_never": len(never), "never_app_share": len(nv) / len(df),
    "never_denial": float(nv.denied.mean()),
    "pooled_denial_if_na": float(df.loc[df.aus_na, "denied"].mean()),
    "pooled_denial_if_aus": float(df.loc[~df.aus_na, "denied"].mean()),
}

# %% 1c. LTV above 100%: appraisal coming in low?
ltv = df["loan_to_value_ratio"]
lt_names = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA"}
t = df.assign(over=ltv > 100).groupby([df.loan_type.astype(str), "denied"], observed=True)["over"].mean().unstack()
res["ltv_over100"] = {lt_names[k]: {"approved": float(v[0]), "denied": float(v[1])} for k, v in t.iterrows()}
hi = df[(ltv > 105) & (df.denied == 1)]
res["ltv105_denied_collateral_share"] = float((hi["denial_reason-1"].astype(str) == "4").mean())
res["ltv105_n"] = int((ltv > 105).sum())
res["ltv105_denial"] = float(df.loc[ltv > 105, "denied"].mean())
col = df["denial_reason-1"].astype(str) == "4"
res["ltv_over100_collateral_denials"] = float((ltv[col] > 100).mean())
res["ltv_over100_approved"] = float((ltv[df.denied == 0] > 100).mean())

# %% figure: N/A and missing shares by outcome
fig, ax = plt.subplots(figsize=(6.6, 2.4))
show = ["Approved", "Debt-to-income", "Collateral", "Credit history", "Application incomplete"]
lab = {"aus_na": "AUS = not applicable", "ltv_na": "LTV missing", "dti_na": "DTI missing"}
w = 0.26
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
for i, (c, color) in enumerate(zip(lab, colors)):
    vals = tab.loc[show, c] / 100
    ax.bar(np.arange(len(show)) + (i - 1) * w, vals, width=w - 0.02, color=color, label=lab[c])
ax.set_xticks(range(len(show)))
ax.set_xticklabels(["Approved", "Denied:\nDTI", "Denied:\ncollateral", "Denied:\ncredit history",
                    "Denied:\nincomplete"], fontsize=7.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.legend(fontsize=7, loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figs/fig4_missing_by_reason.pdf", bbox_inches="tight")
plt.close(fig)

# %% 2. leakage auditor on all candidate columns + known post-decision controls
X = build_X(df, ctrl)
aud = LeakageAuditor().fit(X, y, group=df["lei"].astype(str).rename("lei"))
rep = aud.report_.copy()
rep["tier"] = pd.Categorical(rep.index.map(TIERS), ["A", "U", "P"], ordered=True)
rep = rep.sort_values(["tier", "oof_auc"], ascending=[True, False])
rep.to_csv("results/leakage_audit.csv")
res["audit"] = rep.reset_index().to_dict(orient="records")
print(rep.round(3).to_string())

pretty = {"debt_to_income_ratio": "DTI", "loan_to_value_ratio": "Combined LTV",
          "property_value": "Property value", "aus-1": "AUS used",
          "applicant_credit_score_type": "Credit score model",
          "co-applicant_credit_score_type": "Co-app. score model", "preapproval": "Preapproval flag",
          "initially_payable_to_institution": "Initially payable", "interest_rate": "Interest rate",
          "rate_spread": "Rate spread", "total_loan_costs": "Total loan costs",
          "lender_credits": "Lender credits", "purchaser_type": "Purchaser type",
          "hoepa_status": "HOEPA status", "denial_reason_ctrl": "Denial reason",
          "lei": "Lender (LEI)", "derived_msa-md": "MSA", "loan_type": "Loan type",
          "income": "Income", "loan_amount": "Loan amount", "submission_of_application": "Channel",
          "applicant_age": "Applicant age", "state_code": "State", "loan_term": "Loan term",
          "total_units": "Number of units", "has_coapp": "Co-applicant"}
keep = [k for k in pretty if k in rep.index]
with open("results/t_audit.tex", "w") as f:
    last = None
    for k, r in rep.loc[keep].sort_values(["tier", "oof_auc"], ascending=[True, False]).iterrows():
        if last is not None and r.tier != last:
            f.write("\\midrule\n")
        last = r.tier
        lift = "--" if r.miss_share in (0, 1) else ("$\\infty$" if np.isinf(r.miss_lift) else f"{r.miss_lift:.1f}")
        wg = "--" if pd.isna(r.within_group_lift) else f"{r.within_group_lift:.1f}"
        f.write(f"{pretty[k]} & {r.tier} & {r.oof_auc:.3f} & {lift} & "
                f"{100 * r.max_level_rate:.1f} & {wg} & \\texttt{{{r.flag}}} \\\\\n")

with open("results/feature_timing.json", "w") as f:
    json.dump(res, f, indent=1, default=float)
