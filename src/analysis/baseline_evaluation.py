from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%d-%m-%Y %H:%M", errors="raise"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    y_train = df.iloc[:train_end]["fault_d7"]
    y_test = df.iloc[validation_end:]["fault_d7"]

    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0]] * len(y_train), y_train)
    predictions = model.predict([[0]] * len(y_test))

    print("\nAquaSenseAI - Majority-Class Baseline")
    print("=" * 45)
    print(f"Training rows: {len(y_train)}")
    print(f"Test rows: {len(y_test)}")
    print(f"Training majority class: {model.classes_[model.class_prior_.argmax()]}")
    print(f"Test positive rate: {(y_test == 1).mean() * 100:.2f}%")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
    print(f"Positive-class F1: {f1_score(y_test, predictions, zero_division=0):.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["No leak-proximity label", "Leak-proximity label"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
