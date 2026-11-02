from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)
df = df.sort_values("timestamp").reset_index(drop=True)

target = "fault_d7"
excluded = [
    "timestamp",
    target,
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
]

X = df.drop(columns=excluded)
y = df[target].astype(int)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

print("\nAquaSenseAI - Validation Threshold Analysis")
print("=" * 65)
print(f"Validation rows: {len(X_val)}")
print(f"Validation positive rate: {y_val.mean():.4f}")
print(f"PR-AUC (threshold-independent): {average_precision_score(y_val, probabilities):.4f}")
print(f"ROC-AUC (threshold-independent): {roc_auc_score(y_val, probabilities):.4f}")
print("\nThreshold  Accuracy  BalancedAcc  Precision  Recall  F1")

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    predictions = (probabilities >= threshold).astype(int)
    print(
        f"{threshold:9.2f}  "
        f"{accuracy_score(y_val, predictions):8.4f}  "
        f"{balanced_accuracy_score(y_val, predictions):12.4f}  "
        f"{precision_score(y_val, predictions, zero_division=0):9.4f}  "
        f"{recall_score(y_val, predictions, zero_division=0):6.4f}  "
        f"{f1_score(y_val, predictions, zero_division=0):.4f}"
    )

print("\nConfusion matrices at selected thresholds")
for threshold in [0.40, 0.50, 0.60, 0.70]:
    predictions = (probabilities >= threshold).astype(int)
    print(f"\nThreshold: {threshold:.2f}")
    print("Rows = actual [0, 1]; columns = predicted [0, 1]")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

print("\nFinal test split: NOT evaluated")
