"""Score new customers with the saved model.

Run: python -m churn.predict path/to/customers.csv [--out scored.csv]

The input has the same columns as the raw dataset (Churn column optional).
"""

import argparse

import joblib
import pandas as pd

from churn.config import ID_COL, MODEL_PATH, TARGET
from churn.data import clean


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(df: pd.DataFrame, bundle) -> pd.DataFrame:
    X = clean(df).drop(columns=[TARGET], errors="ignore")
    prob = bundle["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame(index=df.index)
    if ID_COL in df.columns:
        out[ID_COL] = df[ID_COL]
    out["churn_probability"] = prob.round(4)
    out["will_churn"] = (prob >= bundle["threshold"]).astype(int)
    return out.sort_values("churn_probability", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--out", help="write results here instead of printing")
    args = parser.parse_args()

    result = predict(pd.read_csv(args.csv), load_model())
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Scored {len(result)} customers -> {args.out}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
