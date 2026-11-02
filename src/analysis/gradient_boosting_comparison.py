from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

X = df.drop(columns=[
    "timestamp", "fault_d7",
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
])
y = df["fault_d7"].astype(int)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train, X_val = X.iloc[:train_end], X.iloc[train_end:val_end]
y_train, y_val = y.iloc[:train_end], y.iloc[train_end:val_end]

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight=None, max_iter=2000, random_state=42
        )),
    ]),
    "Histogram Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=42,
        )),
    ]),
}

print("\nMODEL COMPARISON — CHRONOLOGICAL VALIDATION")
print("=" * 60)
print(f"Validation positive rate: {y_val.mean():.4f}")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    prob = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

    print(f"\n{name}")
    print(f"Accuracy:          {accuracy_score(y_val, pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_val, pred):.4f}")
    print(f"ROC-AUC:           {roc_auc_score(y_val, prob):.4f}")
    print(f"PR-AUC:            {average_precision_score(y_val, prob):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_val, pred, labels=[0, 1]))
    print(classification_report(
        y_val, pred, labels=[0, 1],
        target_names=["No proximity label", "Proximity label"],
        digits=4, zero_division=0
    ))

print("\nFinal test period untouched.")
