import pandas as pd

path = "data/raw/input_model_potenc_predXfault7_A.csv"
df = pd.read_csv(path)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)
train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]

features = [c for c in df.columns if c not in ("timestamp", "fault_d7")]

print("\nFEATURE RELATIONSHIP CHECK")
print("=" * 95)
print(f"{'Feature':38} {'Train corr':>12} {'Val corr':>12} {'Train means (0 / 1)':>24}")
print("-" * 95)

for feature in features:
    train_corr = train[feature].corr(train["fault_d7"])
    val_corr = val[feature].corr(val["fault_d7"])
    means = train.groupby("fault_d7")[feature].mean()
    mean_0 = means.get(0, float("nan"))
    mean_1 = means.get(1, float("nan"))
    print(f"{feature:38} {train_corr:12.3f} {val_corr:12.3f} {mean_0:10.2f} / {mean_1:.2f}")
