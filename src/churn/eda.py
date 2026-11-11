"""Exploratory data analysis: prints a summary and saves charts to reports/figures.

Run: python -m churn.eda
"""

import matplotlib

matplotlib.use("Agg")  # save files, do not open windows
import matplotlib.pyplot as plt
import seaborn as sns

from churn.config import FIGURES_DIR, TARGET
from churn.data import clean, load_raw


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()

    print("Shape:", raw.shape)
    print("\nBlank TotalCharges:", (raw["TotalCharges"].astype(str).str.strip() == "").sum())
    print("Duplicate customer IDs:", raw["customerID"].duplicated().sum())

    df = clean(raw)
    print("\nMissing values after cleaning:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nClass balance (1 = churned):\n", df[TARGET].value_counts(normalize=True).round(3))
    print("\nNumeric summary:\n", df.describe().round(2))

    for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
        print(f"\nChurn rate by {col}:\n", df.groupby(col)[TARGET].mean().sort_values().round(3))

    charts = {
        "01_class_balance.png": lambda ax: sns.countplot(x=TARGET, data=df, ax=ax),
        "02_churn_by_contract.png": lambda ax: sns.barplot(x="Contract", y=TARGET, data=df, ax=ax),
        "03_tenure_by_churn.png": lambda ax: sns.histplot(
            data=df, x="tenure", hue=TARGET, bins=36, multiple="stack", ax=ax),
        "04_monthly_charges_by_churn.png": lambda ax: sns.boxplot(
            x=TARGET, y="MonthlyCharges", data=df, ax=ax),
        "05_correlation.png": lambda ax: sns.heatmap(
            df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax),
    }
    for name, draw in charts.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        draw(ax)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / name, dpi=120)
        plt.close(fig)
    print(f"\nSaved {len(charts)} charts to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
