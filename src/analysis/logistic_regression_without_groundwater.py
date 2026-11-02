from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
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
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

print("\nAquaSenseAI - Logistic Regression Without Groundwater Features")
print("=" * 65)
print(f"Predictor count: {X.shape[1]}")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print("Final test split: NOT evaluated")
print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_val, predictions):.4f}")
print(f"PR-AUC: {average_precision_score(y_val, probabilities):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")

print("\nClassification report:")
print(classification_report(
    y_val,
    predictions,
    labels=[0, 1],
    target_names=["No leak-proximity label", "Leak-proximity label"],
    zero_division=0,
))

print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_val, predictions, labels=[0, 1]))
