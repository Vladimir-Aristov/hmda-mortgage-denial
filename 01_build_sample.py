# Builds the analysis sample from the raw HMDA files (data/hmda_<year>.csv).
# Run: python 01_build_sample.py 2024 2025

import sys
import pandas as pd

years = [int(y) for y in sys.argv[1:]] or [2024, 2025]

# rate, spread, costs, points, purchaser type etc. are only filled for approved
# loans -> leakage, so we don't even load them
cols = [
    "activity_year", "lei", "derived_msa-md", "state_code", "county_code",
    "conforming_loan_limit", "derived_dwelling_category",
    "derived_ethnicity", "derived_race", "derived_sex",
    "action_taken", "preapproval", "loan_type", "loan_purpose", "lien_status",
    "reverse_mortgage", "open-end_line_of_credit", "business_or_commercial_purpose",
    "loan_amount", "loan_to_value_ratio", "loan_term", "intro_rate_period",
    "negative_amortization", "interest_only_payment", "balloon_payment",
    "other_nonamortizing_features", "property_value", "construction_method",
    "occupancy_type", "total_units", "income", "debt_to_income_ratio",
    "applicant_credit_score_type", "co-applicant_credit_score_type",
    "applicant_age", "co-applicant_age",
    "submission_of_application", "initially_payable_to_institution",
    "aus-1", "aus-2", "denial_reason-1", "denial_reason-2",
    "tract_population", "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]

num_cols = ["loan_amount", "loan_to_value_ratio", "loan_term", "intro_rate_period",
            "property_value", "income", "tract_population",
            "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
            "tract_to_msa_income_percentage", "tract_owner_occupied_units",
            "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]

filter_cols = ["action_taken", "loan_purpose", "lien_status", "occupancy_type",
               "construction_method", "open-end_line_of_credit",
               "reverse_mortgage", "business_or_commercial_purpose"]


def filter_chunk(df):
    # home purchase, 1st lien, owner-occupied, site-built 1-4 units, closed-end,
    # no reverse / business loans
    mask = (
        df["action_taken"].isin([1, 2, 3])
        & (df["loan_purpose"] == 1)
        & (df["lien_status"] == 1)
        & (df["occupancy_type"] == 1)
        & (df["construction_method"] == 1)
        & (df["derived_dwelling_category"] == "Single Family (1-4 Units):Site-Built")
        & (df["open-end_line_of_credit"] == 2)
        & (df["reverse_mortgage"] == 2)
        & (df["business_or_commercial_purpose"] == 2)
    )
    df = df[mask].copy()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # "Exempt", "NA" -> NaN
    df["denied"] = (df["action_taken"] == 3).astype("int8")
    return df.drop(columns=[c for c in filter_cols if c != "action_taken"]
                   + ["derived_dwelling_category"])


def build(year):
    chunks = []
    n_raw = 0
    # file is ~1.6GB, so read in chunks
    for ch in pd.read_csv(f"data/hmda_{year}.csv", usecols=cols, dtype=str,
                          chunksize=500_000, low_memory=False):
        n_raw += len(ch)
        for c in filter_cols:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        chunks.append(filter_chunk(ch))

    df = pd.concat(chunks, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    df.to_parquet(f"data/sample_{year}.parquet", index=False)
    print(f"{year}: raw rows {n_raw:,} -> sample {len(df):,}; "
          f"denial rate {df['denied'].mean():.4f}")


if __name__ == "__main__":
    for y in years:
        build(y)
