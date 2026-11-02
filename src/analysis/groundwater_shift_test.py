from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[train_end:val_end]

features = [
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
]

print("\nCLASS-CONDITIONAL GROUNDWATER SHIFT CHECK")
print("=" * 75)
print("Training vs validation only; final test set excluded.")

for feature in features:
    print(f"\nFEATURE: {feature}")
    print("-" * 65)

    for label in (0, 1):
        train_values = train.loc[train["fault_d7"] == label, feature].dropna()
        val_values = validation.loc[
            validation["fault_d7"] == label, feature
        ].dropna()

        print(f"\nClass {label}:")
        print(f"  Training observed:   {len(train_values)}")
        print(f"  Validation observed: {len(val_values)}")
        print(f"  Training median:     {train_values.median():.3f}")
        print(f"  Validation median:   {val_values.median():.3f}")

        if len(train_values) and len(val_values):
            result = ks_2samp(train_values, val_values)
            print(f"  KS statistic:        {result.statistic:.4f}")
            print(f"  KS p-value:          {result.pvalue:.6g}")
        else:
            print("  KS comparison unavailable: insufficient observed values.")

print("\nInterpretation:")
print("- Compare KS statistics across classes to assess distribution shifts.")
print("- Large shifts in both classes suggest broader temporal distribution change.")
print("- A shift in only one class may indicate a changing feature-target relationship.")
print("- Time dependence affects statistical-test assumptions.")
print("- This is diagnostic only; no model fitted and no final test evaluation.")
