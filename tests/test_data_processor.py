"""
test_data_processor.py
Pytest suite for data_processor.py.

Tests:
  1. All 8 expected columns are present in the output CSV.
  2. delta_litres >= 0 for every row.
  3. hour is in [0, 23] for every row.
  4. day_of_week is in [0, 6] for every row.
  5. Row count of output matches row count of input readings.csv.
  6. The first delta_litres per meter is 0.
  7. hostel_id, floor_id, meter_id columns are non-null.
"""

import os
import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths relative to this test file so pytest can be run from anywhere.
# ---------------------------------------------------------------------------
_TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_DATA_DIR     = os.path.join(_PROJECT_ROOT, "data")

INPUT_CSV  = os.path.join(_DATA_DIR, "readings.csv")
OUTPUT_CSV = os.path.join(_DATA_DIR, "processed_readings.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def processed_df():
    """Run the processor once and return the resulting DataFrame."""
    import sys
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    from data_processor import process  # noqa: PLC0415

    df = process(input_path=INPUT_CSV, output_path=OUTPUT_CSV)
    return df


@pytest.fixture(scope="module")
def input_df():
    """Load the raw readings.csv for row-count comparison."""
    return pd.read_csv(INPUT_CSV)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "timestamp", "hostel_id", "floor_id", "meter_id",
    "meter_reading", "delta_litres", "hour", "day_of_week",
}


def test_all_columns_present(processed_df):
    """Test 1: output CSV must contain all 8 required columns."""
    assert EXPECTED_COLUMNS.issubset(set(processed_df.columns)), (
        f"Missing columns: {EXPECTED_COLUMNS - set(processed_df.columns)}"
    )


def test_delta_litres_non_negative(processed_df):
    """Test 2: delta_litres must be >= 0 for every row."""
    assert (processed_df["delta_litres"] >= 0).all(), (
        "Found negative delta_litres values:\n"
        f"{processed_df[processed_df['delta_litres'] < 0]}"
    )


def test_hour_in_valid_range(processed_df):
    """Test 3: hour must be in [0, 23]."""
    assert processed_df["hour"].between(0, 23).all(), (
        f"Out-of-range hour values: {processed_df['hour'].unique()}"
    )


def test_day_of_week_in_valid_range(processed_df):
    """Test 4: day_of_week must be in [0, 6]."""
    assert processed_df["day_of_week"].between(0, 6).all(), (
        f"Out-of-range day_of_week values: {processed_df['day_of_week'].unique()}"
    )


def test_row_count_matches_input(processed_df, input_df):
    """Test 5: output row count must equal input row count."""
    assert len(processed_df) == len(input_df), (
        f"Row count mismatch: input={len(input_df)}, output={len(processed_df)}"
    )


def test_first_delta_per_meter_is_zero(processed_df):
    """Test 6: first delta_litres for each meter must be 0."""
    first_deltas = (
        processed_df
        .sort_values(["meter_id", "timestamp"])
        .groupby("meter_id")["delta_litres"]
        .first()
    )
    non_zero = first_deltas[first_deltas != 0.0]
    assert non_zero.empty, (
        f"Non-zero first deltas found for meters: {non_zero.to_dict()}"
    )


def test_identifier_columns_non_null(processed_df):
    """Test 7: hostel_id, floor_id, meter_id must have no null values."""
    for col in ("hostel_id", "floor_id", "meter_id"):
        assert processed_df[col].notna().all(), f"Null values found in column {col!r}"
