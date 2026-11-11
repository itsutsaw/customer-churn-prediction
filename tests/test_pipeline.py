"""Offline tests: no saved model needed, they run on a few hand-made customers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.data import clean, split_xy
from churn.features import add_features, build_model_pipeline
from churn.predict import predict
from churn.train import best_f1_threshold, churners_caught_in_top

BASE = {
    "customerID": "X", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": "900.0",
    "Churn": "Yes",
}


def make_df(n=40):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = dict(BASE, customerID=f"C{i}")
        churn = i % 2 == 0
        row["Churn"] = "Yes" if churn else "No"
        row["Contract"] = "Month-to-month" if churn else "Two year"
        row["tenure"] = int(rng.integers(1, 10) if churn else rng.integers(30, 72))
        row["TotalCharges"] = str(row["tenure"] * row["MonthlyCharges"])
        rows.append(row)
    return pd.DataFrame(rows)


def test_clean_fills_blank_total_charges_for_new_customers_with_zero():
    df = pd.DataFrame([dict(BASE, tenure=0, TotalCharges=" ")])
    out = clean(df)
    assert out.loc[0, "TotalCharges"] == 0.0
    assert "customerID" not in out.columns
    assert out.loc[0, "Churn"] == 1


def test_clean_works_without_target_column():
    df = pd.DataFrame([BASE]).drop(columns=["Churn"])
    assert "Churn" not in clean(df).columns


def test_add_features_handles_zero_tenure_and_counts_services():
    df = clean(pd.DataFrame([dict(BASE, tenure=0, TotalCharges=" ")]))
    out = add_features(df)
    assert out.loc[0, "AvgMonthlySpend"] == 80.0  # falls back to this month's bill
    assert np.isfinite(out.loc[0, "ChargeIncrease"])
    # PhoneService, OnlineBackup, StreamingTV = 3 "Yes" + internet = 4
    assert out.loc[0, "NumServices"] == 4
    assert out.loc[0, "TenureGroup"] == "0-1y"


def test_pipeline_learns_and_tolerates_unseen_category():
    X, y = split_xy(clean(make_df()))
    pipe = build_model_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)

    prob = pipe.predict_proba(X)[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()
    assert pipe.score(X, y) > 0.9  # the toy data is easy to separate

    new = X.head(1).copy()
    new["PaymentMethod"] = "Crypto"  # never seen in training
    pipe.predict_proba(new)  # must not raise


def test_predict_uses_saved_threshold_and_keeps_ids():
    df = make_df()
    X, y = split_xy(clean(df))
    pipe = build_model_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)

    out = predict(df.drop(columns=["Churn"]), {"pipeline": pipe, "threshold": 1.01})
    assert set(out["customerID"]) == set(df["customerID"])
    assert out["will_churn"].sum() == 0  # nobody passes an impossible threshold
    assert out["churn_probability"].is_monotonic_decreasing


def test_threshold_and_top_share_helpers():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < best_f1_threshold(y, prob) <= 0.8
    assert churners_caught_in_top(y, prob, share=0.5) == pytest.approx(1.0)
