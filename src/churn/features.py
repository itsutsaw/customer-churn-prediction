"""Feature engineering and the preprocessing pipeline.

Everything here goes INSIDE the scikit-learn Pipeline, so:
- imputing, scaling and encoding are learned from training data only (no leakage),
- the saved model takes raw customer rows and does all the work itself.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS, SERVICE_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """New columns built from domain knowledge. Row-by-row only, so it cannot leak."""
    df = df.copy()

    # How many services the customer pays for. More services = more "locked in".
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (
        df["InternetService"] != "No"
    ).astype(int)

    # Average bill over their lifetime. tenure 0 would divide by zero, so use this month's bill.
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, 1), df["MonthlyCharges"]
    )

    # Is the current bill higher than what they used to pay on average? Price rises drive churn.
    df["ChargeIncrease"] = df["MonthlyCharges"] - df["AvgMonthlySpend"]

    # EDA shows churn is very high in the first year, so give the model tenure bands too.
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, np.inf],
        labels=["0-1y", "1-2y", "2-4y", "4y+"],
    ).astype(str)

    return df


def build_preprocessor() -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a category never seen in training becomes all zeros
        # instead of crashing the model in production.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("columns", columns),
    ])


def build_model_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])
