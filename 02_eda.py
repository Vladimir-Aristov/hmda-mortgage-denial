# EDA for the first submission: tables 1-2, figures 1-3, rule benchmark.
# 2025 is our hold-out year, so from it we only take N and the denial rate.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

Path("figs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
DARK, MID = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "pdf.fonttype": 42,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MID, "axes.labelcolor": DARK,
    "xtick.color": MID, "ytick.color": MID,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})
pct = mtick.PercentFormatter(1.0, decimals=0)

# %% data
df = pd.read_parquet("data/sample_2024.parquet")
df25 = pd.read_parquet("data/sample_2025.parquet", columns=["denied"])
N = len(df)

res = {
    "n_2024": N, "n_2025": len(df25),
    "denial_2024": float(df.denied.mean()), "denial_2025": float(df25.denied.mean()),
    "n_lenders": int(df.lei.nunique()), "n_msa": int(df["derived_msa-md"].nunique()),
    "n_states": int(df.state_code.nunique()),
    "action_shares": df.action_taken.value_counts(normalize=True).sort_index().to_dict(),
    "action_counts": df.action_taken.value_counts().sort_index().to_dict(),
}

inc = df["income"].where(df["income"] > 0)   # income is in $000, some values <= 0
df["loan_to_income"] = df["loan_amount"] / (1000 * inc)
df["has_coapp"] = (df["co-applicant_age"].astype(str) != "9999").astype(int)

# %% table 1 - numeric variables
# using percentiles, the means are useless here (max LTV is ~2e9...)
num_vars = {
    "loan_amount": "Loan amount (\\$000)",
    "property_value": "Property value (\\$000)",
    "income": "Applicant income (\\$000)",
    "loan_to_value_ratio": "Combined LTV (\\%)",
    "loan_to_income": "Loan-to-income ratio",
    "loan_term": "Loan term (months)",
    "tract_minority_population_percent": "Tract minority pop. (\\%)",
    "tract_to_msa_income_percentage": "Tract/MSA income (\\%)",
}
rows = []
for col, label in num_vars.items():
    x = df[col] / 1000 if col in ("loan_amount", "property_value") else df[col]
    rows.append([label, x.quantile(.01), x.quantile(.1), x.median(), x.quantile(.9),
                 x.quantile(.99), x.max(), x.isna().mean() * 100,
                 x[df.denied == 0].median(), x[df.denied == 1].median()])
res["table1"] = rows


def fmt(v):
    if pd.isna(v):
        return "--"
    if abs(v) >= 1e6:
        return f"{v:.1e}".replace("e+0", "e").replace("e+", "e")
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


with open("results/table1.tex", "w") as f:
    for r in rows:
        f.write(r[0] + " & " + " & ".join(fmt(v) for v in r[1:]) + " \\\\\n")

# %% table 2 - categorical variables, share and denial rate
dti_order = ["<20%", "20%-<30%", "30%-<36%"] + [str(i) for i in range(36, 50)] + ["50%-60%", ">60%"]
dti_str = df["debt_to_income_ratio"].astype(str)
df["dti_cat"] = dti_str.where(dti_str.isin(dti_order), "Missing/NA")
res["dti_missing_share"] = float((df.dti_cat == "Missing/NA").mean())
res["dti_missing_denial"] = float(df.loc[df.dti_cat == "Missing/NA", "denied"].mean())

# 36..49 are reported one by one, group them into two bins for the table
dti_num = pd.to_numeric(dti_str, errors="coerce")
dti_grp = pd.cut(dti_num, [35, 43, 49], labels=["36-43", "44-49"]).astype(object)
dti_grp = dti_grp.where(dti_grp.notna(), df["dti_cat"])

cat_rows = []


def add_cat(feature, s, labels, order):
    g = df.groupby(s.astype(str), observed=True)["denied"].agg(["size", "mean"])
    g = g.reindex([o for o in order if o in g.index])
    for k, r in g.iterrows():
        cat_rows.append([feature, labels.get(k, k), r["size"] / N * 100, r["mean"] * 100])


add_cat("Loan type", df["loan_type"],
        {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA/RHS"}, ["1", "2", "3", "4"])
add_cat("DTI", dti_grp,
        {"<20%": "$<$20\\%", "20%-<30%": "20--30\\%", "30%-<36%": "30--36\\%",
         "36-43": "36--43\\%", "44-49": "44--49\\%", "50%-60%": "50--60\\%",
         ">60%": "$>$60\\%", "Missing/NA": "Missing / NA"},
        ["<20%", "20%-<30%", "30%-<36%", "36-43", "44-49", "50%-60%", ">60%", "Missing/NA"])
add_cat("Co-applicant", df["has_coapp"], {"0": "No", "1": "Yes"}, ["0", "1"])
add_cat("Channel", df["submission_of_application"],
        {"1": "Direct to lender", "2": "Via broker/other"}, ["1", "2"])
add_cat("AUS used", df["aus-1"],
        {"1": "DU (Fannie Mae)", "2": "LPA (Freddie Mac)", "3": "TOTAL (FHA)",
         "6": "Not applicable"}, ["1", "2", "3", "6"])
add_cat("Conforming limit", df["conforming_loan_limit"],
        {"C": "Conforming", "NC": "Jumbo"}, ["C", "NC"])
t2 = pd.DataFrame(cat_rows, columns=["Feature", "Level", "Share", "Denial"])
res["table2"] = t2.to_dict(orient="records")


def write_table2(path, features):
    lines = []
    for feat in features:
        if lines:
            lines.append("\\addlinespace[2pt]\n")
        for i, (_, r) in enumerate(t2[t2.Feature == feat].iterrows()):
            name = feat if i == 0 else ""
            lines.append(f"{name} & {r.Level} & {r.Share:.1f} & {r.Denial:.1f} \\\\\n")
    with open(path, "w") as f:
        f.writelines(lines)


# split in two halves so it fits side by side in the report
write_table2("results/table2a.tex", ["Loan type", "DTI"])
write_table2("results/table2b.tex", ["Co-applicant", "Channel", "AUS used", "Conforming limit"])

# %% leakage check
# preapproval=1 -> 0% denied, initially_payable=3 -> 84% denied: both basically encode y
res["leak"] = {c: df.groupby(df[c].astype(str))["denied"].agg(["size", "mean"]).to_dict(orient="index")
               for c in ["preapproval", "initially_payable_to_institution"]}

# %% figure 1 - target and denial reasons
reasons = {"1": "Debt-to-income", "2": "Employment history", "3": "Credit history",
           "4": "Collateral", "5": "Insufficient cash", "6": "Unverifiable info",
           "7": "Application incomplete", "8": "Mortgage ins. denied", "9": "Other"}
rs = df.loc[df.denied == 1, "denial_reason-1"].astype(str).map(reasons).value_counts(normalize=True)
res["denial_reasons"] = rs.to_dict()

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.25), gridspec_kw={"width_ratios": [1, 1.45]})
shares = pd.Series(res["action_shares"]).rename(
    {1: "Originated", 2: "Approved,\nnot accepted", 3: "Denied"})
bars = ax[0].bar(shares.index, shares.values, color=[BLUE, BLUE, ORANGE], width=0.6)
for b, v in zip(bars, shares.values):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.1%}",
               ha="center", va="bottom", fontsize=7.5)
ax[0].yaxis.set_major_formatter(pct)
ax[0].set_ylim(0, 1)
ax[0].set_title("(a) Outcome of application, 2024", loc="left")
ax[0].text(0.98, 0.62, "target: denied = 1\n(orange)", transform=ax[0].transAxes,
           ha="right", fontsize=7, color=MID)

rs = rs.sort_values()
ax[1].barh(rs.index, rs.values, color=ORANGE, height=0.62)
for i, v in enumerate(rs.values):
    ax[1].text(v + 0.005, i, f"{v:.0%}" if v >= 0.005 else "$<$0.1%", va="center", fontsize=7)
ax[1].xaxis.set_major_formatter(pct)
ax[1].set_xlim(0, rs.max() * 1.18)
ax[1].set_title("(b) Principal denial reason (share of denials)", loc="left")
ax[1].grid(axis="y", visible=False)
fig.tight_layout(w_pad=2)
fig.savefig("figs/fig1_target.pdf")
plt.close(fig)

# %% figure 2 - denial rate vs DTI, LTV, income, lender
fig, ax = plt.subplots(2, 2, figsize=(6.6, 3.75))

a = ax[0, 0]
g = df[df.dti_cat != "Missing/NA"].groupby("dti_cat", observed=True)["denied"].mean().reindex(dti_order)
a.plot(range(len(g)), g.values, color=BLUE, lw=2, marker="o", ms=3.5)
a.set_xticks(range(len(g)))
a.set_xticklabels(["$<$20", "20-30", "30-36"] + [str(i) for i in range(36, 50)] + ["50-60", "$>$60"],
                  rotation=90, fontsize=6.5)
a.set_yscale("log")
a.yaxis.set_major_formatter(pct)
a.set_yticks([0.05, 0.1, 0.2, 0.4, 0.8])
a.yaxis.set_minor_locator(mtick.NullLocator())
a.axvline(dti_order.index("43"), color=GREY, lw=0.8, ls="--")
a.text(dti_order.index("43") - 0.3, 0.3, "43% (old QM\ncap)", ha="right", fontsize=6.5, color=MID)
a.set_title("(a) Denial rate by debt-to-income ratio (%, log scale)", loc="left")

a = ax[0, 1]
ltv_bins = pd.cut(df["loan_to_value_ratio"].clip(upper=110),
                  [0, 60, 70, 75, 80, 85, 90, 95, 97, 100, 105, 110])
g = df.groupby(ltv_bins, observed=True)["denied"].mean()
a.plot(range(len(g)), g.values, color=BLUE, lw=2, marker="o", ms=3.5)
xl = [f"{int(i.left)}-{int(i.right)}" for i in g.index]
xl[-1] = "$>$105"
a.set_xticks(range(len(g)))
a.set_xticklabels(xl, rotation=90, fontsize=6.5)
a.yaxis.set_major_formatter(pct)
a.set_title("(b) Denial rate by combined LTV (%)", loc="left")

a = ax[1, 0]
decile = pd.qcut(inc, 10, labels=False, duplicates="drop") + 1
for code, color, name in [("1", BLUE, "Conventional"), ("2", ORANGE, "FHA"), ("3", GREEN, "VA")]:
    m = df["loan_type"].astype(str) == code
    g = df[m].groupby(decile[m])["denied"].mean()
    a.plot(g.index, g.values, color=color, lw=2, marker="o", ms=3)
    a.text(g.index[-1] + 0.2, g.values[-1], name, fontsize=6.5, va="center")
a.set_xticks(range(1, 11))
a.set_xlim(0.7, 11.9)
a.yaxis.set_major_formatter(pct)
a.set_xlabel("Applicant income decile")
a.set_title("(c) Denial rate by income decile and loan type", loc="left")

a = ax[1, 1]
lenders = df.groupby("lei", observed=True)["denied"].agg(["mean", "size"])
lenders = lenders[lenders["size"] >= 500]   # drop tiny lenders, their rates are noise
res["lenders_500"] = len(lenders)
res["lender_share_500"] = float(lenders["size"].sum() / N)
res["lender_denial_p10_p90"] = [float(lenders["mean"].quantile(.1)), float(lenders["mean"].quantile(.9))]
a.hist(lenders["mean"], bins=np.linspace(0, 0.5, 41),
       weights=lenders["size"] / lenders["size"].sum(),
       color=BLUE, edgecolor="white", linewidth=0.6)
a.axvline(res["denial_2024"], color=ORANGE, lw=1.2)
a.text(res["denial_2024"] + 0.01, a.get_ylim()[1] * 0.85, f"pooled {res['denial_2024']:.1%}", fontsize=6.5)
a.xaxis.set_major_formatter(pct)
a.yaxis.set_major_formatter(pct)
a.set_xlabel("Lender-level denial rate")
a.set_ylabel("Share of applications")
a.set_title(f"(d) Across lenders ({len(lenders):,} with $\\geq$500 apps)", loc="left")

fig.tight_layout(h_pad=1.2, w_pad=1.5)
fig.savefig("figs/fig2_predictors.pdf")
plt.close(fig)

# %% figure 3 - distributions by outcome, raw gap by race
fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.1))
for a, (col, xlabel, rng) in zip(ax[:2], [("income", "Applicant income ($000)", (0, 400)),
                                          ("loan_to_income", "Loan-to-income ratio", (0, 8))]):
    for k, color, name in [(0, BLUE, "Approved"), (1, ORANGE, "Denied")]:
        x = df.loc[df.denied == k, col].dropna()
        x = x[(x > rng[0]) & (x < rng[1])]
        a.hist(x, bins=40, range=rng, density=True, histtype="step", color=color, lw=1.6, label=name)
    a.set_xlabel(xlabel)
    a.set_yticks([])
    a.grid(axis="y", visible=False)
ax[0].legend(loc="upper right", fontsize=6.5)
ax[0].set_title("(a) Income", loc="left")
ax[1].set_title("(b) Loan-to-income", loc="left")

race_names = {"White": "White", "Black or African American": "Black", "Asian": "Asian",
              "Joint": "Joint", "Race Not Available": "Not available",
              "American Indian or Alaska Native": "Am. Indian/AK"}
rr = df.groupby(df["derived_race"].astype(str))["denied"].mean()
rr = rr[list(race_names)].rename(race_names).sort_values()
res["race_denial"] = rr.to_dict()
ax[2].barh(rr.index, rr.values, color=BLUE, height=0.6)
for i, v in enumerate(rr.values):
    ax[2].text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=6.5)
ax[2].xaxis.set_major_formatter(pct)
ax[2].set_xlim(0, rr.max() * 1.3)
ax[2].grid(axis="y", visible=False)
ax[2].set_title("(c) Raw denial rate by race", loc="left")

fig.tight_layout(w_pad=1.2)
fig.savefig("figs/fig3_distributions.pdf")
plt.close(fig)

# %% simple rule-based benchmark (what a non-ML underwriter could do)
dti_high = df["dti_cat"].isin(["50%-60%", ">60%"]) | (dti_num >= 50)
ltv = df["loan_to_value_ratio"]
rules = {
    "DTI>=50": dti_high,
    "DTI>=50 or LTV>100": dti_high | (ltv > 100),
    "DTI>43 or LTV>97": dti_high | (dti_num > 43) | (ltv > 97),
}
res["rules"] = {}
for name, flag in rules.items():
    tp = (flag & (df.denied == 1)).sum()
    res["rules"][name] = {"flag_share": float(flag.mean()),
                          "recall": float(tp / df.denied.sum()),
                          "precision": float(tp / flag.sum())}

# %% save numbers used in the text
res["median_loan"] = float(df.loan_amount.median())
res["median_income"] = float(df.income.median())
for k, cat in [("lt20", "<20%"), ("30_36", "30%-<36%"), ("50_60", "50%-60%"), ("gt60", ">60%")]:
    res[f"dti_{k}_denial"] = float(df.loc[df.dti_cat == cat, "denied"].mean())

with open("results/eda_numbers.json", "w") as f:
    json.dump(res, f, indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("table1", "table2")}, indent=1, default=float))
