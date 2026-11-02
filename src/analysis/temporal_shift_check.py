from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)
df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "TRAIN": df.iloc[:train_end],
    "VALIDATION": df.iloc[train_end:val_end],
    "TEST (not used for modeling)": df.iloc[val_end:],
}

features = [
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
    "temp_site1_anomaly_score",
    "sfc_temp_site1_anomaly_score",
]

print("\nTEMPORAL DISTRIBUTION SHIFT CHECK")
print("=" * 75)

for name, part in splits.items():
    print(f"\n{name}")
    print("-" * 55)
    print(f"Rows: {len(part)}")
    print(f"Period: {part['timestamp'].min()} to {part['timestamp'].max()}")
    print(f"Positive label rate: {part['fault_d7'].mean():.4f}")

    for feature in features:
        values = part[feature]
        print(
            f"{feature}: median={values.median():.3f}, "
            f"missing={values.isna().mean():.1%}"
        )

print("\nNote: Test statistics are descriptive only; no test-based model decisions.")
