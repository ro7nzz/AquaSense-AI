from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix

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

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    )),
])

model.fit(X_train, y_train)
pred = model.predict(X_val)

val = df.iloc[train_end:val_end].copy()
val["actual"] = y_val.to_numpy()
val["predicted"] = pred
val["outage"] = val[["326417_ws_temp", "326416_ws_vigor"]].isna().all(axis=1)

print("\nVALIDATION PERFORMANCE BY SENSOR AVAILABILITY")
print("=" * 65)

for name, group in [
    ("Sensor outage", val[val["outage"]]),
    ("Sensors available", val[~val["outage"]]),
]:
    actual = group["actual"]
    predicted = group["predicted"]

    print(f"\n{name}")
    print(f"Hours: {len(group)}")
    print(f"Balanced accuracy: {balanced_accuracy_score(actual, predicted):.4f}")
    print(f"Positive recall: {recall_score(actual, predicted, pos_label=1, zero_division=0):.4f}")
    print(f"Negative recall: {recall_score(actual, predicted, pos_label=0, zero_division=0):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

print("\nValidation only. Final test untouched; no model decisions made from test data.")
