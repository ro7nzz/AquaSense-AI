from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

target = "fault_d7"
excluded = [
    "timestamp", target,
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
]
X = df.drop(columns=excluded)
y = df[target].astype(int)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train, X_val = X.iloc[:train_end], X.iloc[train_end:val_end]
y_train, y_val = y.iloc[:train_end], y.iloc[train_end:val_end]

val = df.iloc[train_end:val_end].copy()
outage = val[["326417_ws_temp", "326416_ws_vigor"]].isna().all(axis=1).to_numpy()

print("\nVALIDATION-ONLY CLASS-WEIGHT COMPARISON")
print("=" * 65)

for weight in [None, "balanced"]:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight=weight, max_iter=2000, random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

    print(f"\nCLASS WEIGHT: {weight}")
    print(f"ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")

    for threshold in [0.40, 0.50, 0.60, 0.70]:
        pred = (probabilities >= threshold).astype(int)
        print(
            f"threshold={threshold:.2f} | "
            f"accuracy={accuracy_score(y_val, pred):.4f} | "
            f"balanced_acc={balanced_accuracy_score(y_val, pred):.4f} | "
            f"precision={precision_score(y_val, pred, zero_division=0):.4f} | "
            f"recall={recall_score(y_val, pred, zero_division=0):.4f} | "
            f"F1={f1_score(y_val, pred, zero_division=0):.4f}"
        )
        print("  Confusion [[TN, FP], [FN, TP]]:")
        print(" ", confusion_matrix(y_val, pred, labels=[0, 1]).tolist())

    pred50 = (probabilities >= 0.50).astype(int)
    print("\nAt threshold 0.50:")
    print(f"  Outage hours predicted positive: {pred50[outage].sum()}/{outage.sum()}")
    print(f"  Available-sensor hours predicted positive: "
          f"{pred50[~outage].sum()}/{(~outage).sum()}")

print("\nValidation only. Final test set untouched.")
