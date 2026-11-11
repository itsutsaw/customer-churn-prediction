"""Load the raw CSV and fix data-quality problems found during EDA."""

import pandas as pd

from churn.config import DATA_PATH, ID_COL, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy. Works on training data and on new customers (no Churn column)."""
    df = df.copy()

    # TotalCharges is stored as text and has 11 blank values. All 11 rows have
    # tenure == 0: brand-new customers who have not been billed yet. So the
    # correct value is 0, not the median (that would invent money they never paid).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0

    # An ID is unique per row, so it carries no pattern a model could learn.
    df = df.drop(columns=[ID_COL], errors="ignore")

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    return df


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]
