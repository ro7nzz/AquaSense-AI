import pandas as pd

path = "data/raw/input_model_potenc_predXfault7_A.csv"
df = pd.read_csv(path)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)
splits = {
    "TRAIN": df.iloc[:train_end],
    "VALIDATION": df.iloc[train_end:val_end],
}

features = [
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
]

for split_name, part in splits.items():
    print(f"\n{'=' * 65}\n{split_name}")
    print(f"Period: {part['timestamp'].min()} to {part['timestamp'].max()}")

    for feature in features:
        print(f"\nFeature: {feature}")
        print(f"Overall missing: {part[feature].isna().mean():.1%}")

        for label in (0, 1):
            values = part.loc[part["fault_d7"] == label, feature]
            missing_rate = values.isna().mean()
            present = values.dropna()

            print(
                f"  Class {label}: rows={len(values)}, "
                f"missing={missing_rate:.1%}, "
                f"median={present.median():.2f}, "
                f"Q1={present.quantile(0.25):.2f}, "
                f"Q3={present.quantile(0.75):.2f}"
                if not present.empty else
                f"  Class {label}: rows={len(values)}, missing={missing_rate:.1%}, no observed values"
            )

print("\nDiagnostic complete. No model fitted; final test set not evaluated.")
