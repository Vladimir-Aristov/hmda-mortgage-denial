# Leakage audit as a pipeline step.
# LeakageAuditor looks at every column on the TRAINING data only and flags the ones
# that look like they carry information about the decision itself:
#   - out-of-fold AUC of the feature alone is too high
#   - some level (or NaN) has an almost deterministic outcome
#   - the fact that the value is missing predicts y much better than the value
#   - (if a group, e.g. lender, is given) a code that is rare at a lender but comes
#     with a much higher denial rate than that lender's usual one -- typical for
#     "not applicable" codes that some lenders only fill in for files they reject
# "drop" flags are removed in transform(), "review" flags are only reported --
# a human still has to decide using knowledge of when the field is recorded.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _as_groups(s, n_bins=20):
    """Turn a column into discrete groups; NaN is its own group."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > n_bins:
        # bin on the values themselves so equal values always share a bin
        # (ranking with ties broken by row order made fake "pure" bins on sorted data)
        g = pd.qcut(s, n_bins, labels=False, duplicates="drop").astype("float")
    else:
        g = s.astype(str).where(s.notna(), None)
        g = pd.Series(pd.factorize(g, use_na_sentinel=True)[0], index=s.index).astype(float)
        g[g < 0] = np.nan
    return g.fillna(-1).astype(int)


def _oof_auc(g, y, n_splits=5, prior=100, seed=0):
    """Out-of-fold AUC of a smoothed target-mean encoding of the groups."""
    score = np.zeros(len(y))
    base = y.mean()
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(g, y):
        stats = pd.DataFrame({"g": g[tr], "y": y[tr]}).groupby("g")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + prior * base) / (stats["count"] + prior)
        score[te] = pd.Series(g[te]).map(enc).fillna(base).to_numpy()
    auc = roc_auc_score(y, score)
    return max(auc, 1 - auc)


class LeakageAuditor(BaseEstimator, TransformerMixin):
    def __init__(self, auc_drop=0.90, auc_review=0.75, min_support=0.001,
                 pure_low=0.001, pure_high=0.99, pure_support=0.01, lift_review=4.0,
                 rare_share=0.05, n_splits=5):
        self.auc_drop = auc_drop
        self.auc_review = auc_review
        self.min_support = min_support
        self.pure_low = pure_low
        self.pure_high = pure_high
        self.pure_support = pure_support
        self.lift_review = lift_review
        self.rare_share = rare_share
        self.n_splits = n_splits

    def fit(self, X, y, group=None):
        y = np.asarray(y)
        base = y.mean()
        if group is not None:
            group = pd.Series(np.asarray(group))
            expected = pd.Series(y).groupby(group).transform("mean").to_numpy()
        rows = []
        for col in X.columns:
            s = X[col].reset_index(drop=True)
            g = _as_groups(s).to_numpy()
            auc = _oof_auc(g, y, self.n_splits)

            miss = s.isna().to_numpy()
            miss_share = miss.mean()
            if 0 < miss_share < 1:
                miss_auc = roc_auc_score(y, miss)
                miss_auc = max(miss_auc, 1 - miss_auc)
                miss_lift = y[miss].mean() / y[~miss].mean() if y[~miss].mean() > 0 else np.inf
            else:
                miss_auc, miss_lift = 0.5, 1.0

            # levels with large support and (almost) deterministic outcome
            lv = pd.DataFrame({"g": g, "y": y}).groupby("g")["y"].agg(["mean", "size"])
            lv = lv[lv["size"] >= self.min_support * len(y)]
            # a pure level only counts if it is big: a single small lender with no denials
            # is a lender property, not a leak
            big = lv[lv["size"] >= self.pure_support * len(y)]
            pure = big[(big["mean"] <= self.pure_low) | (big["mean"] >= self.pure_high)]
            strong = lv[(lv["mean"] / base >= self.lift_review) | (lv["mean"] <= base / self.lift_review)]

            # within-group check: only rows where the level is rare for that group
            wg_lift = np.nan
            if group is not None and col != getattr(group, "name", None):
                t = pd.DataFrame({"g": g, "grp": group, "y": y, "e": expected})
                freq = t.groupby(["grp", "g"]).g.transform("size") / t.groupby("grp").g.transform("size")
                t = t[freq < self.rare_share]
                lv2 = t.groupby("g")[["y", "e"]].sum()
                lv2 = lv2[t.groupby("g").size() >= self.min_support * len(y)]
                if len(lv2):
                    wg_lift = (lv2["y"] / lv2["e"].clip(lower=1e-9)).max()

            if auc >= self.auc_drop or len(pure) > 0 or miss_auc >= self.auc_drop:
                flag = "drop"
            elif (auc >= self.auc_review or miss_lift >= self.lift_review or len(strong) > 0
                  or wg_lift >= self.lift_review):
                flag = "review"
            else:
                flag = "ok"
            rows.append({
                "feature": col, "oof_auc": auc,
                "miss_share": miss_share, "miss_auc": miss_auc, "miss_lift": miss_lift,
                "max_level_rate": lv["mean"].max() if len(lv) else np.nan,
                "min_level_rate": lv["mean"].min() if len(lv) else np.nan,
                "n_pure_levels": len(pure), "within_group_lift": wg_lift, "flag": flag,
            })
        self.report_ = pd.DataFrame(rows).set_index("feature")
        self.dropped_ = list(self.report_.index[self.report_["flag"] == "drop"])
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])
