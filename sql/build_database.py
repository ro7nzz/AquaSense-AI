from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "input_model_potenc_predXfault7_A.csv"
DB_PATH = ROOT / "data" / "aquasense.db"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("water_sensor_records", conn, if_exists="replace", index=False)

        row_count = conn.execute(
            "SELECT COUNT(*) FROM water_sensor_records"
        ).fetchone()[0]

        target_summary = conn.execute("""
            SELECT fault_d7, COUNT(*) AS records
            FROM water_sensor_records
            GROUP BY fault_d7
            ORDER BY fault_d7
        """).fetchall()

        print("AquaSenseAI SQL database created successfully.")
        print(f"Database: {DB_PATH}")
        print(f"Rows imported: {row_count}")
        print(f"Columns imported: {len(df.columns)}")
        print("Historical target distribution:")
        for label, count in target_summary:
            print(f"  fault_d7={label}: {count}")

if __name__ == "__main__":
    main()
