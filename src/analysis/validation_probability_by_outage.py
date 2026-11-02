from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    )),
])

model.fit(X.iloc[:train_end], y.iloc[:train_end])

val = df.iloc[train_end:val_end].copy()
val["probability"] = model.predict_proba(X.iloc[train_end:val_end])[
    :, list(model.classes_).index(1)
]
val["outage"] = val[["326417_ws_temp", "326416_ws_vigor"]].isna().all(axis=1)

print("\nVALIDATION PREDICTED-PROBABILITY DISTRIBUTION")
print("=" * 55)

for name, group in [
    ("Sensor outage", val[val["outage"]]),
    ("Sensors available", val[~val["outage"]]),
]:
    print(f"\n{name}: {len(group)} hours")
    print(group["probability"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).round(4).to_string())
    print("Below 0.5:", int((group["probability"] < 0.5).sum()))
    print("At least 0.5:", int((group["probability"] >= 0.5).sum()))

print("\nValidation only. Final test set untouched.")
