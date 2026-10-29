# Feature lists and timing tiers shared by 03_feature_timing.py and 04_cv_feature_sets.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# A: known when the application is filed
# U: produced during underwriting (credit pull, appraisal, AUS run) -- "relied on" values
# P: recorded only after / because of the decision
TIERS = {
    "loan_type": "A", "loan_amount": "A", "loan_term": "A", "intro_rate_period": "A",
    "interest_only_payment": "A", "balloon_payment": "A", "negative_amortization": "A",
    "other_nonamortizing_features": "A", "conforming_loan_limit": "A", "total_units": "A",
    "income": "A", "applicant_age": "A", "has_coapp": "A", "submission_of_application": "A",
    "lei": "A", "derived_msa-md": "A", "state_code": "A",
    "tract_minority_population_percent": "A", "tract_to_msa_income_percentage": "A",
    "ffiec_msa_md_median_family_income": "A", "tract_owner_occupied_units": "A",
    "tract_one_to_four_family_homes": "A", "tract_median_age_of_housing_units": "A",
    "debt_to_income_ratio": "U", "loan_to_value_ratio": "U", "property_value": "U",
    "aus-1": "U", "applicant_credit_score_type": "U", "co-applicant_credit_score_type": "U",
    "preapproval": "P", "initially_payable_to_institution": "P",
    "interest_rate": "P", "rate_spread": "P", "total_loan_costs": "P", "lender_credits": "P",
    "purchaser_type": "P", "hoepa_status": "P", "denial_reason_ctrl": "P",
}

DTI_PATTERN = r"^(<20%|20%-<30%|30%-<36%|\d\d|50%-60%|>60%)$"


def dti_valid(s):
    return s.astype(str).str.match(DTI_PATTERN).fillna(False).astype(bool)


def build_X(df, ctrl):
    """All audited columns: sample columns + post-decision controls."""
    df = df.copy()
    df["has_coapp"] = (df["co-applicant_age"].astype(str) != "9999").astype(int)
    X = df[[c for c in TIERS if c in df.columns]].copy()
    for c in ctrl.columns:
        X[c] = ctrl[c].to_numpy()
    for c in ["interest_rate", "rate_spread", "total_loan_costs", "lender_credits"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")   # "NA"/"Exempt" -> NaN
    return X


class Neutraliser(BaseEstimator, TransformerMixin):
    """Removes the 'missing / not applicable' signal from underwriting-time fields:
    NaN -> training median, invalid DTI codes and 'no score' -> training mode."""

    def fit(self, X, y=None):
        self.medians_ = {c: X[c].median() for c in ["loan_to_value_ratio", "property_value", "loan_term"]
                         if c in X}
        self.modes_ = {}
        if "debt_to_income_ratio" in X:
            dti = X["debt_to_income_ratio"].astype(str)
            self.modes_["debt_to_income_ratio"] = dti[dti_valid(dti)].mode()[0]
        if "applicant_credit_score_type" in X:
            cs = X["applicant_credit_score_type"].astype(str)
            self.modes_["applicant_credit_score_type"] = cs[cs != "9"].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        if {"loan_to_value_ratio", "property_value", "loan_amount"} <= set(X.columns):
            # fill LTV and property value *consistently* with the loan amount. Filling each with
            # its own median produced impossible LTV / value pairs, and features built from both
            # (e.g. CLTV minus first-lien LTV) then pointed straight at the imputed rows
            ltv = X["loan_to_value_ratio"].where(X["loan_to_value_ratio"] > 0)
            pv = X["property_value"].where(X["property_value"] > 0)
            ltv = ltv.fillna(100 * X["loan_amount"] / pv)              # value known, LTV missing
            ltv = ltv.fillna(self.medians_["loan_to_value_ratio"])     # both missing
            implied = 100 * X["loan_amount"] / ltv
            implied = (implied // 10_000) * 10_000 + 5_000             # public data: $10k mid-points
            X["loan_to_value_ratio"] = ltv
            X["property_value"] = pv.fillna(implied)
        for c, m in self.medians_.items():
            X[c] = X[c].fillna(m)
        if "debt_to_income_ratio" in self.modes_:
            dti = X["debt_to_income_ratio"].astype(str)
            X["debt_to_income_ratio"] = dti.where(dti_valid(dti), self.modes_["debt_to_income_ratio"])
        if "applicant_credit_score_type" in self.modes_:
            cs = X["applicant_credit_score_type"].astype(str)
            X["applicant_credit_score_type"] = cs.where(cs != "9", self.modes_["applicant_credit_score_type"])
        return X


HIGH_CARD = ["lei", "derived_msa-md", "state_code", "lei_x_type", "county"]


class Prep(BaseEstimator, TransformerMixin):
    """Encoding for gradient boosting, decided at fit time from the columns that are there
    (so it still works after the auditor has dropped some):
    high-cardinality ids -> cross-fitted target encoding, other strings -> pandas category,
    numbers as they are."""

    def _split(self, X):
        cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        return [c for c in cats if c in HIGH_CARD], [c for c in cats if c not in HIGH_CARD]

    def _cats(self, X):
        out = X.drop(columns=self.hc_).copy()
        for c in self.lc_:
            out[c] = pd.Categorical(X[c].astype(str), categories=self.levels_[c])
        return out

    def fit_transform(self, X, y=None):
        from sklearn.preprocessing import TargetEncoder
        self.hc_, self.lc_ = self._split(X)
        # keep the 250 most frequent levels, the rest become NaN
        self.levels_ = {c: X[c].astype(str).value_counts().index[:250].tolist() for c in self.lc_}
        out = self._cats(X)
        if self.hc_:
            self.te_ = TargetEncoder(random_state=0)
            te = self.te_.fit_transform(X[self.hc_].astype(str), y)   # cross-fitted on train
            for i, c in enumerate(self.hc_):
                out[c + "_te"] = te[:, i]
        return out

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        out = self._cats(X)
        if self.hc_:
            te = self.te_.transform(X[self.hc_].astype(str))
            for i, c in enumerate(self.hc_):
                out[c + "_te"] = te[:, i]
        return out
