from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
REPORT_PATH = ROOT / "reports" / "VALIDATION_ERROR_ANALYSIS.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
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

errors = df.iloc[train_end:val_end][["timestamp", target]].copy()
errors["predicted_label"] = predictions
errors["probability_label_1"] = probabilities

errors["error_type"] = "Correct"
errors.loc[
    (errors[target] == 0) & (errors["predicted_label"] == 1),
    "error_type",
] = "False Positive"
errors.loc[
    (errors[target] == 1) & (errors["predicted_label"] == 0),
    "error_type",
] = "False Negative"

errors = errors[errors["error_type"] != "Correct"].copy()
errors["actual_label"] = errors[target]
errors = errors.drop(columns=[target])
errors = errors.sort_values("timestamp")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
errors.to_csv(REPORT_PATH, index=False)

print("\nVALIDATION ERROR ANALYSIS")
print("=" * 55)
print(f"Validation rows: {len(X_val)}")
print(f"False positives: {(errors['error_type'] == 'False Positive').sum()}")
print(f"False negatives: {(errors['error_type'] == 'False Negative').sum()}")
print(f"Total misclassified: {len(errors)}")
print(f"Saved error details to: {REPORT_PATH}")
print("\nError counts by type:")
print(errors["error_type"].value_counts().to_string())

print("\nFirst 10 false positives:")
print(
    errors[errors["error_type"] == "False Positive"]
    .head(10)
    .to_string(index=False)
)

print("\nFirst 10 false negatives:")
print(
    errors[errors["error_type"] == "False Negative"]
    .head(10)
    .to_string(index=False)
)

print("\nFinal test split: NOT evaluated.")
