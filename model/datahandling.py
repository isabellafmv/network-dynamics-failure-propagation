from pathlib import Path
import pandas as pd

# root = .../model/data
DATA_DIR = Path(__file__).resolve().parent / "data"


def combine_mx_5x5() -> pd.DataFrame:
    """
    Read all STATS/Mx_5x5.txt files under DATA_DIR,
    add a country column, and return one big long dataframe.
    """
    paths = list(DATA_DIR.rglob("STATS/Mx_5x5.txt"))
    if not paths:
        raise FileNotFoundError(f"No Mx_5x5.txt files found under {DATA_DIR}")

    dfs = []
    for p in paths:
        # country code is the folder directly under data (DNK, ESP, USA, ...)
        country = p.relative_to(DATA_DIR).parts[0]

        df = pd.read_csv(p, sep="\t", comment="#")

        # Typical HMD layout: Year, Age, Female, Male, Total
        # Melt to long form: one row per (Year, Age, Sex)
        df_long = df.melt(
            id_vars=["Year", "Age"],
            value_vars=["Female", "Male", "Total"],
            var_name="Sex",
            value_name="mx",
        )
        df_long["country"] = country
        dfs.append(df_long)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


if __name__ == "__main__":
    combined = combine_mx_5x5()
    out_path = DATA_DIR / "HMD_Mx_5x5_allcountries.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved combined data to {out_path}")
