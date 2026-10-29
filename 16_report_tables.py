# Tables and numbers for the 2nd-submission report, read straight from the result files.
import json
import numpy as np

J = lambda f: json.load(open(f"results/{f}"))
imp_chk = J("imputation_check.json")
sets = J("cv_feature_sets.json")
fe, logit, tuned, cat, lc = J("cv_fe.json"), J("cv_logit.json"), J("cv_tuned.json"), J("cv_catboost.json"), J("learning_curve.json")
fin = {}
for m in "AB":
    try:
        fin[m] = J(f"final_catboost_{m}.json")
    except FileNotFoundError:   # final models not trained yet -> placeholders in the report
        fin[m] = None


def cat_mean(m):
    f = [v for k, v in cat.items() if k.startswith(m + "_fold")]
    return {"auc": np.mean([v["auc"] for v in f]), "pr_auc": np.mean([v["pr_auc"] for v in f])}


def c(r):
    return "-- & --" if r is None else f"{r['auc']:.3f} & {r['pr_auc']:.3f}"


# feature engineering / selection
fe_rows = [
    ("Raw features", fe["A0"], fe["B0"]),
    ("+ engineered features (all groups)", fe["A1"], fe["B1"]),
    ("\\quad without affordability ratios", None, fe["B1-ratios"]),
    ("\\quad without lender profile", None, fe["B1-lender"]),
    ("\\quad without program limits", None, fe["B1-program"]),
    ("\\quad without lender$\\times$type, county encodings", fe["A1-noint"], fe["B1-interactions"]),
    ("Selected set + external county data", fe["A2"], fe["B2"]),
]
with open("results/t_fe.tex", "w") as f:
    for name, a, b in fe_rows:
        f.write(f"{name} & {c(a)} & {c(b)} \\\\\n")

# model comparison, CV
rows = [
    ("Logistic regression (L2, splines)", logit["A_C1.0"], logit["B_C1.0"]),
    ("Gradient boosting, raw features", fe["A0"], fe["B0"]),
    ("Gradient boosting + engineered features", fe["A1-noint"], fe["B1-interactions"]),
    ("\\quad tuned (successive halving)", tuned["A_cv400k"], tuned["B_cv400k"]),
    ("CatBoost + engineered features", cat_mean("A"), cat_mean("B")),
]
with open("results/t_models.tex", "w") as f:
    for name, a, b in rows:
        f.write(f"{name} & {c(a)} & {c(b)} \\\\\n")

full = [
    ("Gradient boosting, all 2024 training rows", lc["A_2840857"], lc["B_2840857"]),
    ("\\quad tuned", tuned["A_full"], tuned["B_full"]),
    ("\\textbf{CatBoost, tuned, all 2.74m training rows}", fin["A"], fin["B"]),
]
with open("results/t_models_full.tex", "w") as f:
    for name, a, b in full:
        f.write(f"{name} & {c(a)} & {c(b)} \\\\\n")

sb, sc = sets["B: + underwriting values, missingness neutralised"], sets["C: + raw missing/N-A codes and AUS"]
pre = {
    "SetsCBauc": f"{sc['auc'] - sb['auc']:+.3f}", "SetsCBpr": f"{sc['pr_auc'] - sb['pr_auc']:+.3f}",
    "FEgainA": f"{fe['A1-noint']['pr_auc'] - fe['A0']['pr_auc']:+.3f}",
    "FEgainB": f"{fe['B1-interactions']['pr_auc'] - fe['B0']['pr_auc']:+.3f}",
    "TuneGainA": f"{tuned['A_cv400k']['pr_auc'] - fe['A1-noint']['pr_auc']:+.3f}",
    "TuneGainB": f"{tuned['B_cv400k']['pr_auc'] - fe['B1-interactions']['pr_auc']:+.3f}",
    "LogitBpr": f"{logit['B_C1.0']['pr_auc']:.3f}", "CatBpr": f"{cat_mean('B')['pr_auc']:.3f}",
    "ImpOld": f"{imp_chk['independent medians (old)']['subordinate_gap']:.2f}",
    "ImpNew": f"{imp_chk['consistent (new)']['subordinate_gap']:.2f}",
}
pre_tex = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in pre.items())
if fin["A"] is None or fin["B"] is None:
    open("results/t_prec.tex", "w").write("(pending) & & & \\\\\n")
    open("results/final_macros.tex", "w").write(pre_tex + "".join(
        f"\\newcommand{{\\{k}}}{{??}}\n" for k in
        ["FinBauc", "FinBpr", "FinAauc", "FinApr", "FinBmean", "FinBbase", "FinBbrier", "PrecRatio",
         "FinAiter", "FinBiter"]))
    raise SystemExit("final models missing: placeholders written")
# precision at recall vs rules (validation set)
B, A = fin["B"], fin["A"]
r1, r2 = B["rules"]["DTI>=50 or LTV>100"], B["rules"]["DTI>43 or LTV>97"]
pr = lambda d, r: d["prec_at_recall"][r]
prec_rows = [
    ("Rule: DTI $\\geq$ 50\\% or LTV $>$ 100\\%", r1["recall"], r1["precision"], r1["flag_share"]),
    ("Model A (application time)", 0.49, pr(A, "0.49")["precision"], pr(A, "0.49")["flag_share"]),
    ("Model B (CatBoost)", 0.49, pr(B, "0.49")["precision"], pr(B, "0.49")["flag_share"]),
    ("Rule: DTI $>$ 43\\% or LTV $>$ 97\\%", r2["recall"], r2["precision"], r2["flag_share"]),
    ("Model B (CatBoost)", 0.70, pr(B, "0.7")["precision"], pr(B, "0.7")["flag_share"]),
    ("Model B (CatBoost)", 0.30, pr(B, "0.3")["precision"], pr(B, "0.3")["flag_share"]),
]
with open("results/t_prec.tex", "w") as f:
    for i, (name, rec, p, fl) in enumerate(prec_rows):
        if i == 3:
            f.write("\\addlinespace[3pt]\n")
        f.write(f"{name} & {100 * rec:.0f}\\% & {100 * p:.0f}\\% & {100 * fl:.0f}\\% \\\\\n")

ratio = pr(B, "0.49")["precision"] / r1["precision"]
mac = {
    "FinBauc": f"{B['auc']:.3f}", "FinBpr": f"{B['pr_auc']:.3f}",
    "FinAauc": f"{A['auc']:.3f}", "FinApr": f"{A['pr_auc']:.3f}",
    "FinBmean": f"{B['mean_pred']:.3f}", "FinBbase": f"{B['base_rate']:.3f}",
    "FinBbrier": f"{B['brier']:.3f}", "PrecRatio": f"{ratio:.1f}",
    "FinAiter": f"{A['best_iter'] + 1:,}", "FinBiter": f"{B['best_iter'] + 1:,}",
}
with open("results/final_macros.tex", "w") as f:
    f.write(pre_tex)
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(mac)
print(open("results/t_prec.tex").read())
