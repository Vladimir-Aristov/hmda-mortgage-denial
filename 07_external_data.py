# County-level data from public sources listed on data.gov, lagged so that for an
# application in year t we only use what was public at that time:
#   FHFA conforming loan limits for year t   (announced in November of t-1)
#   FHFA county house price index, growth in t-1 and over t-4..t-1
#   BLS LAUS county unemployment rate, annual average t-1 and change vs t-2
# Files (downloaded to data/external/):
#   https://www.fhfa.gov/document/d/cll/fullcountyloanlimitlist{2024,2025}_hera-based_final_flat.csv
#   https://www.fhfa.gov/hpi/download/annual/hpi_at_county.xlsx
#   https://www.bls.gov/lau/laucnty{22,23,24}.xlsx
import pandas as pd

P = "data/external/"
YEARS = [2024, 2025]

# conforming loan limits by number of units
cll = []
for t in YEARS:
    c = pd.read_csv(P + f"fullcountyloanlimitlist{t}_hera-based_final_flat.csv", encoding="utf-8-sig", dtype=str)
    c["county"] = c["FIPS_State_Code"].str.zfill(2) + c["FIPS_County_Code"].str.zfill(3)
    c = c.rename(columns={"One_Unit_Limit": "cll_1", "Two_Unit_Limit": "cll_2",
                          "Three_Unit_Limit": "cll_3", "Four_Unit_Limit": "cll_4"})
    for k in ["cll_1", "cll_2", "cll_3", "cll_4"]:
        c[k] = pd.to_numeric(c[k])
    c["year"] = t
    cll.append(c[["county", "year", "cll_1", "cll_2", "cll_3", "cll_4"]])
cll = pd.concat(cll)

# house prices
h = pd.read_excel(P + "hpi_at_county.xlsx", skiprows=5, dtype={"FIPS code": str})
h = h.rename(columns={"FIPS code": "county", "Year": "y", "Annual Change (%)": "chg", "HPI": "hpi"})
h["chg"] = pd.to_numeric(h["chg"], errors="coerce")
h["hpi"] = pd.to_numeric(h["hpi"], errors="coerce")
h = h[["county", "y", "chg", "hpi"]].set_index(["county", "y"])
hpi = []
for t in YEARS:
    lag1 = h.xs(t - 1, level="y")
    lag4 = h.xs(t - 4, level="y")
    hpi.append(pd.DataFrame({"year": t, "hpi_growth_1y": lag1["chg"],
                             "hpi_growth_3y": 100 * (lag1["hpi"] / lag4["hpi"] - 1)}).reset_index())
hpi = pd.concat(hpi)

# unemployment
def laus(yy):
    u = pd.read_excel(P + f"laucnty{yy}.xlsx", skiprows=1, dtype=str)
    u.columns = ["laus", "st", "cty", "name", "year", "lf", "emp", "unemp", "rate"]
    u = u[u["st"].str.fullmatch(r"\d{2}", na=False)]
    u["county"] = u["st"] + u["cty"].str.zfill(3)
    return u.set_index("county")["rate"].astype(float)

rates = {2000 + yy: laus(yy) for yy in (22, 23, 24)}
une = pd.concat([pd.DataFrame({"year": t, "unemp_rate": rates[t - 1],
                               "unemp_change": rates[t - 1] - rates[t - 2]}).reset_index()
                 for t in YEARS])

ext = cll.merge(hpi, on=["county", "year"], how="outer").merge(une, on=["county", "year"], how="outer")
ext.to_parquet("data/external_county.parquet", index=False)
print(ext.describe().round(2).to_string())
print(ext.groupby("year").size())
