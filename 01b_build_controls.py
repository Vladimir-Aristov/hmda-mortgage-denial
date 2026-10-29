# Post-decision fields for the same rows as data/sample_2024.parquet.
# They are used ONLY as positive controls for the leakage auditor (a leak detector must
# flag them) and never as model inputs.
# Run after 01_build_sample.py:  python 01b_build_controls.py
import pandas as pd

from importlib import import_module

bs = import_module("01_build_sample")

ctrl_cols = ["interest_rate", "rate_spread", "total_loan_costs", "lender_credits",
             "purchaser_type", "hoepa_status", "denial_reason-1"]


def build(year=2024):
    out = []
    for ch in pd.read_csv(f"data/hmda_{year}.csv", usecols=bs.cols + ctrl_cols, dtype=str,
                          chunksize=500_000, low_memory=False):
        for c in bs.filter_cols:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        ch = bs.filter_chunk(ch)   # same filter -> same rows in the same order
        out.append(ch[ctrl_cols])
    df = pd.concat(out, ignore_index=True).rename(columns={"denial_reason-1": "denial_reason_ctrl"})
    df.to_parquet(f"data/controls_{year}.parquet", index=False)
    print(f"controls_{year}: {len(df):,} rows")


if __name__ == "__main__":
    build(2024)
