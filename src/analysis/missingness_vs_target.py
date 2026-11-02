from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "missingness_vs_target.txt"

FEATURES = [
    "326417_ws_temp",
    "326416_ws_vigor",
    "4408411600_kW",
    "4408411600_power_hour",
    "279625_ws_temp",
    "279623_ws_vigor",
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
    "temp_site1_anomaly_score",
    "sfc_temp_site1_anomaly_score",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "fault_d7" not in df.columns:
        raise ValueError("Required target column 'fault_d7' is missing.")

    lines = [
        "AquaSenseAI - Missingness vs Target Analysis",
        "=" * 52,
        f"Dataset rows: {len(df)}",
        "",
        "Missingness rate and target distribution",
        "-" * 52,
    ]

    for feature in FEATURES:
        if feature not in df.columns:
            lines.extend([f"{feature}: column not found", ""])
            continue

        missing = df[feature].isna()
        lines.append(f"Feature: {feature}")
        lines.append(f"Missing rows: {int(missing.sum())} ({missing.mean() * 100:.2f}%)")

        for group_name, mask in [
            ("Missing", missing),
            ("Present", ~missing),
        ]:
            subset = df.loc[mask, "fault_d7"]
            if len(subset) == 0:
                lines.append(f"  {group_name}: no rows")
                continue

            counts = subset.value_counts(dropna=False)
            positive_rate = (subset == 1).mean() * 100
            lines.append(
                f"  {group_name}: rows={len(subset)}, "
                f"fault_d7=0: {int((subset == 0).sum())}, "
                f"fault_d7=1: {int((subset == 1).sum())}, "
                f"positive rate={positive_rate:.2f}%"
            )

        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
