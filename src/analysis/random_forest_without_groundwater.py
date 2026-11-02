from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

excluded = [
    "timestamp", "fault_d7",
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
]
X = df.drop(columns=excluded)
y = df["fault_d7"].astype(int)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train, X_val = X.iloc[:train_end], X.iloc[train_end:val_end]
y_train, y_val = y.iloc[:train_end], y.iloc[train_end:val_end]

print("\nRANDOM FOREST COMPARISON — VALIDATION ONLY")
print("=" * 55)

for weight in [None, "balanced"]:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight=weight,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

    print(f"\nClass weight: {weight}")
    print(f"ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")
    print(f"PR-AUC: {average_precision_score(y_val, probabilities):.4f}")

    for threshold in [0.40, 0.50, 0.60, 0.70]:
        pred = (probabilities >= threshold).astype(int)
        print(
            f"Threshold {threshold:.2f} | "
            f"Accuracy {accuracy_score(y_val, pred):.4f} | "
            f"Balanced accuracy {balanced_accuracy_score(y_val, pred):.4f}"
        )
        print("Confusion [[TN, FP], [FN, TP]]:",
              confusion_matrix(y_val, pred, labels=[0, 1]).tolist())

print("\nFinal test set remains untouched.")
