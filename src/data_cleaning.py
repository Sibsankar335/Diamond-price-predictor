"""
data_cleaning.py
-----------------
Loads the raw Diamonds dataset (~54,000 real diamonds, via seaborn) and
performs cleaning: removes duplicates, drops physically-impossible rows
(zero dimensions), removes extreme outliers, and encodes categorical
quality grades (cut/color/clarity) as ordered numeric ranks so models can
use their natural ordering. Saves a clean CSV to data/diamonds_clean.csv.

Run:
    python src/data_cleaning.py
"""

import os
import pandas as pd
import seaborn as sns

RAW_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diamonds_raw.csv")
CLEAN_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diamonds_clean.csv")

# Natural quality ordering (worst -> best) for each categorical grade
CUT_ORDER = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
COLOR_ORDER = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}  # D = best, J = worst
CLARITY_ORDER = {
    "I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8,
}


def load_raw_data() -> pd.DataFrame:
    """Load the Diamonds dataset from seaborn's sample-data repo."""
    df = sns.load_dataset("diamonds")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning + encoding steps to the raw dataframe."""
    initial_rows = len(df)

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Drop physically impossible rows (zero width/height/depth mm)
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

    # 3. Remove extreme outliers using IQR on price and carat
    for col in ["price", "carat"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # 4. Sanity-check dimensions vs. table/depth percentages (drop obvious data errors)
    df = df[(df["depth"] > 40) & (df["depth"] < 80)]
    df = df[(df["table"] > 40) & (df["table"] < 90)]

    # 5. Encode categorical quality grades as ordered ranks
    df["cut_encoded"] = df["cut"].map(CUT_ORDER)
    df["color_encoded"] = df["color"].map(COLOR_ORDER)
    df["clarity_encoded"] = df["clarity"].map(CLARITY_ORDER)

    df = df.dropna().reset_index(drop=True)

    print(f"Rows before cleaning : {initial_rows}")
    print(f"Rows after cleaning  : {len(df)}")
    print(f"Rows removed         : {initial_rows - len(df)}")

    return df


def main():
    os.makedirs(os.path.dirname(RAW_OUT_PATH), exist_ok=True)

    raw_df = load_raw_data()
    raw_df.to_csv(RAW_OUT_PATH, index=False)
    print(f"Raw data saved to {RAW_OUT_PATH}")

    clean_df = clean_data(raw_df)
    clean_df.to_csv(CLEAN_OUT_PATH, index=False)
    print(f"Clean data saved to {CLEAN_OUT_PATH}")


if __name__ == "__main__":
    main()
