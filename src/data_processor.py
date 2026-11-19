"""
data_processor.py
Reads readings.csv (produced by data_generator.py), enriches each row with:
  - delta_litres : consumption since the previous reading for that meter
                   (first reading per meter is 0; negative values clipped to 0)
  - hour         : 0-23, extracted from timestamp
  - day_of_week  : 0=Monday … 6=Sunday, extracted from timestamp

Writes the result to data/processed_readings.csv.

Usage (from project root):
    python src/data_processor.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths – resolve relative to this file so the script works from any cwd.
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR     = os.path.join(_PROJECT_ROOT, "data")

INPUT_PATH  = os.path.join(_DATA_DIR, "readings.csv")
OUTPUT_PATH = os.path.join(_DATA_DIR, "processed_readings.csv")


def process(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Load, enrich, and persist the readings data.

    Parameters
    ----------
    input_path  : path to readings.csv
    output_path : path where processed_readings.csv will be written

    Returns
    -------
    The processed DataFrame (useful for testing without hitting the filesystem).
    """
    # 1. Load CSV; parse timestamp column as datetime immediately.
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    # 2. Sort by meter_id then timestamp so diffs are always forward-in-time.
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # 3. Compute delta_litres per meter group.
    df["delta_litres"] = (
        df.groupby("meter_id")["meter_reading"]
        .diff()          # NaN for the first row of each group
        .fillna(0.0)     # first reading of each meter → 0
    )

    # 4. Extract time features from the timestamp column.
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week   # 0=Monday, 6=Sunday

    # 5. Clip delta_litres to 0 to guard against floating-point anomalies.
    df["delta_litres"] = df["delta_litres"].clip(lower=0.0)

    # 6. Write output CSV with the full column set.
    output_cols = [
        "timestamp", "hostel_id", "floor_id", "meter_id",
        "meter_reading", "delta_litres", "hour", "day_of_week",
    ]
    df[output_cols].to_csv(output_path, index=False)
    print(f"Processed {len(df)} rows -> {output_path}")

    return df[output_cols]


if __name__ == "__main__":
    process()
