from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
ERROR_PATH = ROOT / "reports" / "VALIDATION_ERROR_ANALYSIS.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)
df = df.sort_values("timestamp").reset_index(drop=True)

errors = pd.read_csv(ERROR_PATH, parse_dates=["timestamp"])
start = int(len(df) * 0.70)
end = int(len(df) * 0.85)

val = df.iloc[start:end].copy()
val = val.merge(errors[["timestamp", "error_type"]], on="timestamp", how="left")
val["error_type"] = val["error_type"].fillna("Correct")

features = ["326417_ws_temp", "326416_ws_vigor"]

for feature in features:
    missing = val[feature].isna()
    print(f"\n{feature}")
    print("=" * 55)
    print(f"Missing hours: {int(missing.sum())} / {len(val)}")
    print(f"Missing during false negatives: {int((missing & val['error_type'].eq('False Negative')).sum())}")
    print(f"Missing during correct predictions: {int((missing & val['error_type'].eq('Correct')).sum())}")

    blocks = val.loc[missing, ["timestamp"]].copy()
    if blocks.empty:
        print("No missing periods.")
        continue

    blocks["gap"] = blocks["timestamp"].diff().dt.total_seconds().div(3600)
    blocks["episode"] = blocks["gap"].ne(1).cumsum()

    episodes = blocks.groupby("episode").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        hours=("timestamp", "size"),
    ).reset_index(drop=True)

    print("Longest missing periods:")
    print(episodes.sort_values("hours", ascending=False).head(10).to_string(index=False))

print("\nValidation only; no model retrained and final test untouched.")
