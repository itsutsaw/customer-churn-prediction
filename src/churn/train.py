"""Train, compare, tune, evaluate and save the churn model.

Run: python -m churn.train

Order matters and is the whole point of the script:
1. Split once. The test set is locked away until the very end.
2. Compare models with cross-validation on the TRAINING set only.
3. Pick the decision threshold from out-of-fold predictions (still training set only).
4. Touch the test set exactly once, for the final honest numbers.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    average_precision_score, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)

from churn.config import (
    CV_FOLDS, FIGURES_DIR, MODEL_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE,
)
from churn.data import clean, load_raw, split_xy
from churn.features import build_model_pipeline

# class_weight="balanced": only ~27% of customers churn, so without it the models
# lean towards predicting "stays" and miss churners.
CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", max_iter=2000),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Random Forest": (
        # no n_jobs here: GridSearchCV already runs folds in parallel, nesting both oversubscribes CPUs
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "model__n_estimators": [300],
            "model__max_depth": [6, 10, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
    ),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [3, 5],
            "model__max_iter": [200],
        },
    ),
}


def metrics_at(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])  # last P/R point has no threshold


def churners_caught_in_top(y_true, y_prob, share=0.2):
    """Business view: if retention calls only the riskiest 20%, what share of churners do they reach?"""
    n = int(len(y_prob) * share)
    top = np.argsort(y_prob)[::-1][:n]
    return float(np.asarray(y_true)[top].sum() / np.asarray(y_true).sum())


def rounded(d):
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y = split_xy(clean(load_raw()))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"Train {X_train.shape}, test {X_test.shape}, churn rate {y.mean():.3f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Baseline: always predict the majority class. Any real model must beat this.
    baseline = build_model_pipeline(DummyClassifier(strategy="most_frequent")).fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"\nBaseline (always 'stays') test accuracy: {baseline_acc:.3f}  <- why accuracy is not our metric")

    # Compare models with cross-validated grid search. ROC-AUC is the selection metric
    # because it judges ranking quality and does not depend on the 0.5 cut-off.
    comparison, searches = [], {}
    for name, (model, grid) in CANDIDATES.items():
        search = GridSearchCV(
            build_model_pipeline(model), grid, cv=cv, n_jobs=-1,
            scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"},
            refit="roc_auc",
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        row = {
            "model": name,
            "cv_roc_auc": res["mean_test_roc_auc"][i],
            "cv_roc_auc_std": res["std_test_roc_auc"][i],
            "cv_f1": res["mean_test_f1"][i],
            "cv_recall": res["mean_test_recall"][i],
            "cv_precision": res["mean_test_precision"][i],
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        }
        comparison.append(row)
        searches[name] = search
        print(f"{name:20s} CV ROC-AUC {row['cv_roc_auc']:.3f} +/- {row['cv_roc_auc_std']:.3f}  "
              f"F1 {row['cv_f1']:.3f}  recall {row['cv_recall']:.3f}  {row['best_params']}")

    best_name = max(comparison, key=lambda r: r["cv_roc_auc"])["model"]
    best = searches[best_name].best_estimator_
    print(f"\nSelected: {best_name}")

    # Choose the threshold on out-of-fold training predictions, never on the test set.
    oof_prob = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof_prob)
    print(f"Threshold that maximises out-of-fold F1: {threshold:.3f}")

    # Final, one-time evaluation on the untouched test set.
    test_prob = best.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    at_default = metrics_at(y_test, test_prob, 0.5)
    at_tuned = metrics_at(y_test, test_prob, threshold)
    top20 = churners_caught_in_top(y_test, test_prob, 0.2)

    print("\nTest set, threshold 0.5:  ", {k: round(v, 3) for k, v in at_default.items()})
    print("Test set, tuned threshold:", {k: round(v, 3) for k, v in at_tuned.items()})
    print(f"Top 20% riskiest customers contain {top20:.1%} of all churners")
    print("\nConfusion matrix (tuned):\n", confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, target_names=["stays", "churns"]))

    # Interpretation: permutation importance on the RAW input columns. Shuffle one column,
    # see how much test ROC-AUC drops. Works for any model and is measured on unseen data.
    perm = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("Top features (drop in ROC-AUC when shuffled):\n", importance.head(8).round(4))

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, test_prob, ax=axes[0], name=best_name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    PrecisionRecallDisplay.from_predictions(y_test, test_prob, ax=axes[1], name=best_name)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "06_roc_pr_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, display_labels=["stays", "churns"], cmap="Blues", ax=ax)
    ax.set_title(f"Threshold {threshold:.2f}")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "07_confusion_matrix.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.head(10)[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Drop in test ROC-AUC when column is shuffled")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "08_feature_importance.png", dpi=120); plt.close(fig)

    # Save the model together with its threshold: one without the other gives wrong predictions.
    joblib.dump({"pipeline": best, "threshold": threshold, "model_name": best_name}, MODEL_PATH)

    report = {
        "data": {"rows": int(len(X)), "train": int(len(X_train)), "test": int(len(X_test)),
                 "churn_rate": round(float(y.mean()), 4)},
        "baseline_test_accuracy": round(baseline_acc, 4),
        "cv_comparison": [rounded(r) for r in comparison],
        "selected_model": best_name,
        "test_at_0.5": rounded(at_default),
        "test_at_tuned_threshold": rounded(at_tuned),
        "top20_share_of_churners": round(top20, 4),
        "permutation_importance": importance.round(4).to_dict(),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model to {MODEL_PATH} and metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
