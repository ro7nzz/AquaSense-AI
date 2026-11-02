from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
ERROR_PATH = ROOT / "reports" / "VALIDATION_ERROR_ANALYSIS.csv"
OUTPUT_PATH = ROOT / "reports" / "VALIDATION_ERROR_FEATURE_DIAGNOSTIC.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M"
)
df = df.sort_values("timestamp").reset_index(drop=True)

errors = pd.read_csv(ERROR_PATH, parse_dates=["timestamp"])

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)
validation = df.iloc[train_end:val_end].copy()

validation = validation.merge(
    errors[["timestamp", "error_type"]],
    on="timestamp",
    how="left",
)
validation["error_type"] = validation["error_type"].fillna("Correct")

features = [
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
    "326417_ws_temp",
    "326416_ws_vigor",
    "4408411600_kW",
    "4408411600_power_hour",
]

rows = []

for error_type, group in validation.groupby("error_type"):
    row = {
        "error_type": error_type,
        "hours": len(group),
        "positive_label_rate": group["fault_d7"].mean(),
    }

    for feature in features:
        row[f"{feature}__missing_pct"] = round(
            group[feature].isna().mean() * 100, 2
        )
        row[f"{feature}__median"] = round(
            group[feature].median(), 3
        )

    rows.append(row)

report = pd.DataFrame(rows).sort_values("error_type")
report.to_csv(OUTPUT_PATH, index=False)

print("\nVALIDATION ERROR / FEATURE DIAGNOSTIC")
print("=" * 65)
print("Validation period only; final test remains untouched.")
print(report.to_string(index=False))
print(f"\nSaved report: {OUTPUT_PATH}")
