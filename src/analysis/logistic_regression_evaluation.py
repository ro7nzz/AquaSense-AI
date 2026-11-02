from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "fault_d7"
NON_FEATURE_COLUMNS = {"timestamp", TARGET}


def evaluate_split(name, y_true, predictions, probabilities):
    print(f"\n{name} RESULTS")
    print("-" * 45)
    print(f"Rows: {len(y_true)}")
    print(f"Positive rate: {y_true.mean() * 100:.2f}%")
    print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, predictions):.4f}")
    print(f"Positive-class F1: {f1_score(y_true, predictions, zero_division=0):.4f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_true, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["No leak-proximity label", "Leak-proximity label"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"timestamp", TARGET}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%d-%m-%Y %H:%M", errors="raise"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values; resolve before training.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Expected binary target labels 0 and 1.")

    feature_columns = [
        column for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]
    X = df[feature_columns].apply(pd.to_numeric, errors="raise")
    y = df[TARGET].astype(int)

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:validation_end], y.iloc[train_end:validation_end]
    X_test, y_test = X.iloc[validation_end:], y.iloc[validation_end:]

    # All learned preprocessing is fitted on training rows only.
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    print("\nAquaSenseAI - Logistic Regression Evaluation")
    print("=" * 50)
    print(f"Dataset rows: {n}")
    print(f"Predictor count: {len(feature_columns)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Training period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[train_end - 1]}")
    print(f"Validation period: {df['timestamp'].iloc[train_end]} to {df['timestamp'].iloc[validation_end - 1]}")
    print(f"Test period: {df['timestamp'].iloc[validation_end]} to {df['timestamp'].iloc[-1]}")
    print("\nPredictors:")
    print("\n".join(feature_columns))
    print("\nMissing values in training predictors:")
    print(X_train.isna().sum()[X_train.isna().sum() > 0].to_string() or "None")

    validation_probabilities = model.predict_proba(X_val)[:, 1]
    validation_predictions = (validation_probabilities >= 0.5).astype(int)
    evaluate_split("VALIDATION (for model assessment)", y_val, validation_predictions, validation_probabilities)

    print(
        "\nThe test split has NOT been evaluated. "
        "Review validation results and methodology before final testing."
    )
    print(
        "\nCaution: supplied anomaly-score generation and feature-selection timing "
        "are not fully documented; predictive performance may be affected by "
        "information-timing or label-association risks."
    )


if __name__ == "__main__":
    main()
