from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(path)
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
dates = df["timestamp"].iloc[train_end:val_end]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])
model.fit(X_train, y_train)

prob = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
pred = (prob >= 0.5).astype(int)

results = pd.DataFrame({
    "timestamp": dates.to_numpy(),
    "actual": y_val.to_numpy(),
    "predicted": pred,
    "probability": prob,
})
results["week"] = results["timestamp"].dt.to_period("W").astype(str)
results["correct"] = results["actual"] == results["predicted"]

weekly = results.groupby("week").agg(
    hours=("actual", "size"),
    actual_positive_rate=("actual", "mean"),
    predicted_positive_rate=("predicted", "mean"),
    accuracy=("correct", "mean"),
    mean_probability=("probability", "mean"),
)

print("\nWEEKLY VALIDATION DIAGNOSTIC")
print(weekly.round(4).to_string())
print("\nFinal test period remains untouched.")
