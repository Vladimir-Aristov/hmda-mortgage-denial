# Shared pieces for the model experiments (09-12): data, feature sets, pipelines.
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler, TargetEncoder

from features import TIERS, Neutraliser, Prep, build_X
from fe import FeatureBuilder

A = [c for c, t in TIERS.items() if t == "A"] + ["county_code", "co-applicant_age", "activity_year"]
U = ["debt_to_income_ratio", "loan_to_value_ratio", "property_value", "applicant_credit_score_type"]
COLS = {"A": A, "B": A + U}
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def load_2024():
    df = pd.read_parquet("data/sample_2024.parquet")
    ctrl = pd.read_parquet("data/controls_2024.parquet")
    X = build_X(df, ctrl)
    X["county_code"] = df["county_code"]
    X["co-applicant_age"] = df["co-applicant_age"]
    X["activity_year"] = df["activity_year"]
    return X, df["denied"].to_numpy()


def subsample(y, n, seed=42):
    """stratified subsample; seed 42 and n=400k give the subsample used in 04/06"""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.where(y == k)[0], int(n * (y == k).mean()), replace=False)
                           for k in (0, 1)])


def hgb_pipeline(model_set, **params):
    p = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0,
             categorical_features="from_dtype")
    p.update(params)
    return Pipeline([("neutral", Neutraliser()),
                     ("fe", FeatureBuilder(underwriting=(model_set == "B"), interactions=False)),
                     ("prep", Prep()),
                     ("hgb", HistGradientBoostingClassifier(**p))])


# ---------------------------------------------------------------- logit ----
class Winsor(BaseEstimator, TransformerMixin):
    """clip numeric columns at the training 1st / 99th percentiles"""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_, self.hi_ = X.quantile(0.01), X.quantile(0.99)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1)


class LogitPrep(BaseEstimator, TransformerMixin):
    """After FeatureBuilder: choose columns by type at fit time and build the linear design:
    logs of money variables, median-imputed + winsorised (P1/P99) + standardised numbers,
    splines for DTI / LTV / loan-to-income / payment-to-income, one-hot for small
    categoricals, target encoding for lender / MSA / state."""
    LOG = ["loan_amount", "income", "property_value"]
    SPLINE = ["dti_num", "loan_to_value_ratio", "loan_to_income", "pti_new_loan"]
    HIGH = ["lei", "derived_msa-md", "state_code"]
    DROP = ["debt_to_income_ratio"]   # replaced by dti_num

    def _setup(self, X):
        self.cat_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
                     and c not in self.HIGH]
        self.high_ = [c for c in self.HIGH if c in X.columns]
        self.spl_ = [c for c in self.SPLINE if c in X.columns]
        self.num_ = [c for c in X.columns if c not in self.cat_ + self.high_ + self.spl_]
        self.med_ = X[self.num_ + self.spl_].median()
        self.ct_ = ColumnTransformer([
            ("num", make_pipeline(Winsor(), StandardScaler()), self.num_),
            ("spl", make_pipeline(Winsor(), SplineTransformer(n_knots=6, degree=3)), self.spl_),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200, sparse_output=False), self.cat_),
            ("te", make_pipeline(TargetEncoder(random_state=0), StandardScaler()), self.high_),
        ])

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        # fit_transform so that the target encoder cross-fits on the training rows
        X = self._base(X)
        self._setup(X)
        return self.ct_.fit_transform(self._fill(X), y)

    def transform(self, X):
        return self.ct_.transform(self._fill(self._base(X)))

    def _base(self, X):
        X = X.drop(columns=[c for c in self.DROP if c in X.columns]).copy()
        for c in self.LOG:
            if c in X:
                X[c] = np.log1p(X[c].clip(lower=0))
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype(str)
        return X

    def _fill(self, X):
        X = X.copy()
        cols = [c for c in self.med_.index if c in X]
        X[cols] = X[cols].fillna(self.med_[cols])
        return X


def logit_pipeline(model_set, C=1.0):
    return Pipeline([("neutral", Neutraliser()),
                     ("fe", FeatureBuilder(underwriting=(model_set == "B"), interactions=False)),
                     ("prep", LogitPrep()),
                     ("lr", LogisticRegression(C=C, max_iter=2000))])
