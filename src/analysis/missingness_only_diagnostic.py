from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)
df = df.sort_values("timestamp").reset_index(drop=True)

target = "fault_d7"
features = [c for c in df.columns if c not in ("timestamp", target)]
X = df[features].iloc[:, :]
y = df[target].astype(int)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

# Each feature becomes 1 if missing and 0 if present.
X_train_missing = X_train.isna().astype(int)
X_val_missing = X_val.isna().astype(int)

models = {
    "Missingness-only Logistic Regression": Pipeline([
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ]),
    "Most-frequent baseline": DummyClassifier(strategy="most_frequent"),
}

print("\nMISSINGNESS-ONLY VALIDATION DIAGNOSTIC")
print("=" * 55)
print("This diagnostic uses training and validation only.")
print("Final test split remains unevaluated.")

for name, model in models.items():
    model.fit(X_train_missing, y_train)
    pred = model.predict(X_val_missing)

    print(f"\n{name}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_val, pred):.4f}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val_missing)[:, list(model.classes_).index(1)]
        print(f"PR-AUC: {average_precision_score(y_val, probs):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_val, probs):.4f}")
