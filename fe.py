# Feature engineering for the 2nd submission. Everything is computed either row by row
# or from the training data inside fit(), so it can sit in the CV pipeline without leakage.
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# approximate program guidelines (used as reference points, the model still learns the rest)
DTI_LIMIT = {"1": 50, "2": 57, "3": 41, "4": 44}        # conv. (DU max), FHA, VA guideline, USDA (GUS)
LTV_MAX = {"1": 97, "2": 98.25, "3": 102.15, "4": 101}  # incl. financed MIP / funding / guarantee fee
RATE_2024 = 0.068   # ~ average 30y fixed rate in 2024, only to turn loan + term into a payment

DTI_MID = {"<20%": 15, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}


def dti_numeric(s):
    s = s.astype(str)
    out = pd.to_numeric(s, errors="coerce")
    return out.fillna(s.map(DTI_MID))


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """underwriting=False -> only application-time features (model A)
    underwriting=True  -> also features built from DTI / LTV / property value (model B)
    lender_stats      -> lender profile computed on training rows (no target used)"""

    GROUPS = ("ratios", "lender", "interactions", "program")

    def __init__(self, underwriting=False, lender_stats=True, drop_group=None,
                 interactions=True, external=False):
        self.underwriting = underwriting
        self.lender_stats = lender_stats
        self.drop_group = drop_group     # for ablation: leave one group out
        self.interactions = interactions
        self.external = external         # county data from 07_external_data.py

    def fit(self, X, y=None):
        if self.external:
            self.ext_ = pd.read_parquet("data/external_county.parquet").set_index(["county", "year"])
        if self.lender_stats and self.drop_group != "lender":
            t = pd.DataFrame({
                "lei": X["lei"].astype(str),
                "fha": (X["loan_type"].astype(str) == "2").astype(float),
                "broker": (X["submission_of_application"].astype(str) == "2").astype(float),
                "lti": self._lti(X),
            })
            g = t.groupby("lei")
            self.lender_ = pd.DataFrame({
                "lender_log_apps": np.log(g.size()),
                "lender_fha_share": g["fha"].mean(),
                "lender_broker_share": g["broker"].mean(),
                "lender_median_lti": g["lti"].median(),
            })
        return self

    @staticmethod
    def _lti(X):
        inc = X["income"].where(X["income"] > 0)
        return X["loan_amount"] / (1000 * inc)

    def transform(self, X):
        X = X.copy()
        lt = X["loan_type"].astype(str)
        inc = X["income"].where(X["income"] > 0)          # $000
        area = X["ffiec_msa_md_median_family_income"].where(lambda v: v > 0)

        # affordability at application time
        X["loan_to_income"] = self._lti(X)
        X["income_rel_area"] = 1000 * inc / area
        X["loan_rel_area"] = X["loan_amount"] / area
        n = X["loan_term"].fillna(360).clip(lower=12)
        r = RATE_2024 / 12
        payment = X["loan_amount"] * r / (1 - (1 + r) ** (-n))
        X["pti_new_loan"] = 12 * payment / (1000 * inc)     # new mortgage payment / income
        X["n_applicants"] = 1 + (X["co-applicant_age"].astype(str) != "9999").astype(int)
        if self.interactions:
            X["lei_x_type"] = X["lei"].astype(str) + "_" + lt  # target-encoded later (Prep)
            X["county"] = X["state_code"].astype(str) + "_" + X["county_code"].astype(str)

        if self.external:
            key = pd.MultiIndex.from_arrays([X["county_code"].astype(str),
                                             pd.to_numeric(X["activity_year"]).astype(int)])
            e = self.ext_.reindex(key).set_axis(X.index)
            units = pd.to_numeric(X["total_units"].astype(str), errors="coerce").clip(1, 4).fillna(1).astype(int)
            limit = np.choose(units.to_numpy() - 1, [e["cll_1"], e["cll_2"], e["cll_3"], e["cll_4"]])
            X["loan_to_conforming_limit"] = X["loan_amount"] / limit
            for c in ["hpi_growth_1y", "hpi_growth_3y", "unemp_rate", "unemp_change"]:
                X[c] = e[c]

        if self.lender_stats and self.drop_group != "lender":
            # lenders unseen in training get NaN, which the boosting model handles
            stats = self.lender_.reindex(X["lei"].astype(str)).set_axis(X.index)
            X = pd.concat([X, stats], axis=1)

        if self.underwriting:
            dti = dti_numeric(X["debt_to_income_ratio"])
            X["dti_num"] = dti
            X["dti_over_limit"] = dti - lt.map(DTI_LIMIT)
            ltv = X["loan_to_value_ratio"]
            X["ltv_over_max"] = ltv - lt.map(LTV_MAX)
            # cash the borrower has to bring, relative to income
            down = (X["property_value"] - X["loan_amount"]).clip(lower=0)
            X["down_payment_to_income"] = down / (1000 * inc)
            # CLTV above LTV of the first lien -> there is a second (e.g. down-payment assistance) loan
            X["subordinate_gap"] = ltv - 100 * X["loan_amount"] / X["property_value"]
        groups = {
            "ratios": ["loan_to_income", "income_rel_area", "loan_rel_area", "pti_new_loan", "n_applicants",
                       "down_payment_to_income", "subordinate_gap"],
            "interactions": ["lei_x_type", "county"],
            "program": ["dti_num", "dti_over_limit", "ltv_over_max"],
        }
        drop = [c for c in groups.get(self.drop_group, []) if c in X]
        drop += [c for c in ["county_code", "activity_year"] if c in X]
        return X.drop(columns=drop)
