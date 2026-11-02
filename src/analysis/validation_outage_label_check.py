from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)
df = df.sort_values("timestamp").reset_index(drop=True)

start = int(len(df) * 0.70)
end = int(len(df) * 0.85)
val = df.iloc[start:end].copy()

features = ["326417_ws_temp", "326416_ws_vigor"]
outage = val[features].isna().all(axis=1)

print("\nVALIDATION LABEL RATE BY SENSOR AVAILABILITY")
print("=" * 60)

for label, mask in [
    ("Sensor outage", outage),
    ("Sensors available", ~outage),
]:
    group = val.loc[mask]
    print(f"\n{label}")
    print(f"Hours: {len(group)}")
    print(f"Positive labels: {int(group['fault_d7'].sum())}")
    print(f"Positive-label rate: {group['fault_d7'].mean():.4f}")
    print(f"Period: {group['timestamp'].min()} to {group['timestamp'].max()}")

print("\nValidation only. Final test remains untouched.")
