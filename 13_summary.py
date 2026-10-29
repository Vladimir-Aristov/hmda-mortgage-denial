# Collect model results into one table (+ learning-curve figure) for the 2nd submission.
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "pdf.fonttype": 42, "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False})
J = lambda f: json.load(open(f"results/{f}"))
fe, logit, tuned, cat, lc = J("cv_fe.json"), J("cv_logit.json"), J("cv_tuned.json"), J("cv_catboost.json"), J("learning_curve.json")


def cat_mean(m):
    f = [v for k, v in cat.items() if k.startswith(m + "_fold")]
    if len(f) < 5:
        return None
    return {"auc": np.mean([v["auc"] for v in f]), "auc_sd": np.std([v["auc"] for v in f]),
            "pr_auc": np.mean([v["pr_auc"] for v in f]), "pr_auc_sd": np.std([v["pr_auc"] for v in f])}


rows = [  # name, A result, B result  (5-fold CV, 400k subsample of 2024)
    ("Logistic regression (L2, splines, C=1)", logit["A_C1.0"], logit["B_C1.0"]),
    ("Gradient boosting, raw features", fe["A0"], fe["B0"]),
    ("Gradient boosting + engineered features", fe["A1-noint"], fe["B1-interactions"]),
    ("  ... tuned (successive halving)", tuned["A_cv400k"], tuned["B_cv400k"]),
    ("CatBoost + engineered features", cat_mean("A"), cat_mean("B")),
]


def cell(r):
    return "--" if r is None else f"{r['auc']:.3f} & {r['pr_auc']:.3f}"


with open("results/t_models.tex", "w") as f:
    for name, a, b in rows:
        f.write(f"{name} & {cell(a)} & {cell(b)} \\\\\n")
with open("results/t_models.md", "w") as f:
    f.write("| Model | A ROC-AUC | A PR-AUC | B ROC-AUC | B PR-AUC |\n|---|---|---|---|---|\n")
    for name, a, b in rows:
        f.write(f"| {name.strip()} | {cell(a).replace(' & ', ' | ')} | {cell(b).replace(' & ', ' | ')} |\n")
print(open("results/t_models.md").read())

# learning curve figure
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
for m, color, name in [("A", "#2a78d6", "A: application time"), ("B", "#eb6834", "B: + underwriting")]:
    pts = sorted((v["n"], v["auc"], v["pr_auc"]) for k, v in lc.items() if k.startswith(m + "_"))
    n, auc, pr = zip(*pts)
    ax[0].plot(n, auc, marker="o", ms=3.5, lw=2, color=color)
    ax[1].plot(n, pr, marker="o", ms=3.5, lw=2, color=color)
    ax[1].text(n[-1] * 1.08, pr[-1], name.split(":")[0], va="center", fontsize=8)
    ax[0].text(n[-1] * 1.08, auc[-1], name.split(":")[0], va="center", fontsize=8)
for a, t in zip(ax, ["ROC-AUC", "PR-AUC"]):
    a.set_xscale("log")
    a.set_xlabel("training rows (2024)")
    a.set_title(t, loc="left")
    a.grid(color="#e6e5e0", lw=0.6)
    a.set_axisbelow(True)
    a.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v / 1e6:g}m"))
fig.tight_layout(w_pad=2.5)
fig.savefig("figs/fig5_learning_curve.pdf", bbox_inches="tight")
