from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]

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

model_path = MODEL_DIR / "aquasense_logistic_regression.joblib"
features_path = MODEL_DIR / "feature_schema.json"

joblib.dump(model, model_path)
features_path.write_text(
    json.dumps({
        "features": X.columns.tolist(),
        "excluded_features": excluded,
        "target": target,
        "model_type": "Logistic Regression",
        "training_rows": len(X_train),
        "training_end": str(df.iloc[train_end - 1]["timestamp"]),
        "intended_output": "Experimental leak-proximity risk indicator",
        "final_test_used_for_training": False,
    }, indent=2),
    encoding="utf-8",
)

print("AquaSenseAI model saved successfully.")
print(f"Training rows: {len(X_train)}")
print(f"Features: {X.shape[1]}")
print(f"Model: {model_path}")
print(f"Feature schema: {features_path}")
print("Validation and final test rows were not used for fitting.")
