# Customer Churn Prediction

An end-to-end machine learning pipeline that predicts which telecom customers are about to cancel, so a retention team knows who to call first. Built with pandas and scikit-learn on the IBM Telco Customer Churn dataset (7,043 customers, 26.5% churned).

## Results (held-out test set, 1,409 customers)

| | Value |
|---|---|
| Selected model | Histogram Gradient Boosting |
| ROC-AUC | 0.845 |
| PR-AUC | 0.663 (random guessing = 0.265) |
| Recall on churners | 0.79 (catches 295 of 374) |
| Precision on churners | 0.52 |
| F1 (churn class) | 0.63 |
| Top 20% riskiest customers | contain 50.8% of all churners |
| "Always predict stays" baseline accuracy | 0.735 |

Cross-validated comparison (5 folds, training set only):

| Model | CV ROC-AUC | CV F1 | CV recall |
|---|---|---|---|
| Logistic Regression | 0.846 ± 0.012 | 0.629 | 0.797 |
| Random Forest | 0.847 ± 0.010 | 0.629 | 0.777 |
| Gradient Boosting | 0.848 ± 0.011 | 0.634 | 0.803 |

The three models are within one standard deviation of each other, so the gap is noise. Logistic regression would be an equally valid pick, and easier to explain to a business team.

Exact numbers are in [reports/metrics.json](reports/metrics.json) and charts in [reports/figures](reports/figures).

## What the data says

- **Contract type and tenure dominate.** Month-to-month customers churn at 42.7%, two-year customers at 2.8%. Shuffling either column drops test ROC-AUC by about 0.075; nothing else comes close.
- Fiber optic customers (41.9%) and electronic check payers (45.3%) churn far more than the rest.
- Customers without tech support churn at 41.6% vs 15.2% with it, which suggests a retention offer.

## How it works

1. **Problem and metric.** Binary classification. Accuracy is misleading here: predicting "stays" for everyone already scores 73.5%. Models are compared on ROC-AUC (ranking quality), and recall/precision/F1 on the churn class are reported.
2. **EDA** ([src/churn/eda.py](src/churn/eda.py)). Class balance, churn rate per category, distributions, correlations.
3. **Cleaning** ([src/churn/data.py](src/churn/data.py)). `TotalCharges` is text with 11 blanks; all 11 have `tenure == 0` (not billed yet), so they become 0 rather than a median. Customer ID is dropped. 22 rows share all attributes but have different IDs; they are different customers, so they are kept.
4. **Feature engineering** ([src/churn/features.py](src/churn/features.py)). Number of services, average monthly spend, current bill minus average (price rise), tenure bands.
5. **Split.** Stratified 80/20, so both sets keep the 26.5% churn rate. The test set is used once, at the end.
6. **Models.** Logistic regression, random forest and gradient boosting, all with `class_weight="balanced"` for the imbalance. A dummy majority-class model is the baseline.
7. **Tuning and evaluation** ([src/churn/train.py](src/churn/train.py)). `GridSearchCV` with stratified 5-fold CV. The decision threshold is chosen to maximise F1 on out-of-fold training predictions, never on the test set.
8. **Interpretation and deployment.** Permutation importance on the raw columns, measured on the test set. The whole pipeline plus its threshold is saved with joblib, and [src/churn/predict.py](src/churn/predict.py) scores new customers from a CSV.

**No data leakage:** feature engineering, imputation, scaling and one-hot encoding all live inside one scikit-learn `Pipeline`, so in every CV fold they are fitted on that fold's training part only. The saved model takes raw customer rows, and unseen categories are ignored instead of crashing it.

## Run it

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m churn.eda          # summary + charts in reports/figures
.venv/bin/python -m churn.train        # compare, tune, evaluate, save model (~15 s)
.venv/bin/python -m churn.predict data/telco_churn.csv --out scored.csv
.venv/bin/python -m pytest             # 6 offline tests
```

## Limits

- One dataset, one snapshot in time. A real deployment would retrain on fresh data and watch for drift in the input columns and the churn rate.
- The engineered features did not move the score much; contract and tenure carry most of the signal.
- The F1-optimal threshold (0.53) treats a missed churner and a wasted call as equally bad. With real costs (retention offer price vs. customer lifetime value) the threshold should be set from those instead.

## Project layout

```
data/telco_churn.csv     raw data (IBM Telco Customer Churn)
src/churn/config.py      paths, column lists, settings
src/churn/data.py        loading and cleaning
src/churn/features.py    feature engineering + preprocessing pipeline
src/churn/eda.py         exploratory analysis and charts
src/churn/train.py       model comparison, tuning, evaluation, saving
src/churn/predict.py     score new customers
tests/                   offline tests
reports/                 metrics.json and figures
```
