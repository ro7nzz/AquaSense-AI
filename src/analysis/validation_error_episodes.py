from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ERROR_PATH = ROOT / "reports" / "VALIDATION_ERROR_ANALYSIS.csv"

errors = pd.read_csv(ERROR_PATH, parse_dates=["timestamp"])

print("\nVALIDATION ERROR EPISODE ANALYSIS")
print("=" * 65)

if errors.empty:
    print("No misclassified validation rows found.")
    raise SystemExit(0)

all_episodes = []

for error_type, subset in errors.groupby("error_type"):
    subset = subset.sort_values("timestamp").copy()
    gaps = subset["timestamp"].diff().dt.total_seconds().div(3600)
    subset["episode_id"] = gaps.ne(1).cumsum()

    episodes = (
        subset.groupby("episode_id")
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            hours=("timestamp", "size"),
        )
        .reset_index(drop=True)
    )
    episodes["error_type"] = error_type
    all_episodes.append(episodes)

episodes = pd.concat(all_episodes, ignore_index=True)

for error_type in ["False Positive", "False Negative"]:
    subset = episodes[episodes["error_type"] == error_type]
    total_hours = int(subset["hours"].sum())

    print(f"\n{error_type}")
    print(f"Error episodes: {len(subset)}")
    print(f"Misclassified hours: {total_hours}")
    print(
        "Longest episode: "
        f"{int(subset['hours'].max()) if not subset.empty else 0} hour(s)"
    )
    print("Five longest episodes:")
    print(
        subset.sort_values("hours", ascending=False)
        .head(5)[["start", "end", "hours"]]
        .to_string(index=False)
    )

print("\nValidation data only. No model retrained; final test untouched.")
