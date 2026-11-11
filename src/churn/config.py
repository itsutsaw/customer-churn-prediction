"""One place for paths, column names and settings, so every script agrees."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# The add-on services a customer can subscribe to. Used to count services per customer.
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns after feature engineering. Listed explicitly (not picked by dtype)
# so a change in pandas dtypes can never silently move a column to the wrong pipe.
NUMERIC_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "NumServices", "AvgMonthlySpend", "ChargeIncrease",
]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
]
