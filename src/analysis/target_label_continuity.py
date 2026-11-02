from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

print("TARGET LABEL CONTINUITY CHECK")
print("=" * 55)

for name, part in [
    ("TRAIN", df.iloc[:train_end].copy()),
    ("VALIDATION", df.iloc[train_end:val_end].copy()),
]:
    labels = part["fault_d7"].astype(int)
    changes = int(labels.ne(labels.shift()).sum() - 1)

    part["run_id"] = labels.ne(labels.shift()).cumsum()
    runs = part.groupby("run_id").agg(
        label=("fault_d7", "first"),
        hours=("fault_d7", "size"),
        start=("timestamp", "first"),
        end=("timestamp", "last"),
    )

    print(f"\n{name}")
    print(f"Rows: {len(part)}")
    print(f"Label changes: {changes}")
    print(f"Positive rate: {labels.mean():.4f}")
    print("Run lengths in hours:")
    print(runs["hours"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).round(2).to_string())

    for label in [0, 1]:
        subset = runs[runs["label"] == label]
        print(
            f"Class {label}: {len(subset)} continuous runs; "
            f"longest run = {int(subset['hours'].max()) if len(subset) else 0} hours"
        )

    print("First 5 label transitions:")
    transitions = part.loc[labels.ne(labels.shift()), ["timestamp", "fault_d7"]].head(5)
    print(transitions.to_string(index=False))

print("\nTraining and validation only. Final test set untouched.")
