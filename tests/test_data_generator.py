"""
test_data_generator.py
Pytest tests for data_generator.py.

Run from the project root:
    pytest tests/test_data_generator.py
"""

import os
import sys
import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure both config.py and src/ are importable from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import METERS, METER_IDS, START_DATE, END_DATE, READING_INTERVAL_MINS
from data_generator import build_timestamps, generate_readings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def readings_df():
    """Generate the readings DataFrame once for all tests in this module."""
    return generate_readings()


@pytest.fixture(scope="module")
def expected_timestamps():
    """Return the DatetimeIndex that the generator uses."""
    return build_timestamps(START_DATE, END_DATE, READING_INTERVAL_MINS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_columns(readings_df):
    """CSV must contain exactly the required columns."""
    expected = ["timestamp", "hostel_id", "floor_id", "meter_id", "meter_reading"]
    assert list(readings_df.columns) == expected, (
        f"Unexpected columns: {list(readings_df.columns)}"
    )


def test_all_meters_present(readings_df):
    """Every meter_id from config.METER_IDS must appear in the data."""
    meters_in_data = set(readings_df["meter_id"].unique())
    for meter_id in METER_IDS:
        assert meter_id in meters_in_data, f"Meter {meter_id!r} missing from readings"


def test_row_count(readings_df, expected_timestamps):
    """Total rows must equal len(METERS) × number_of_hourly_timestamps."""
    expected_rows = len(METERS) * len(expected_timestamps)
    assert len(readings_df) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(readings_df)}"
    )


def test_meter_reading_monotonic(readings_df):
    """meter_reading must be monotonically non-decreasing per meter."""
    for meter_id, group in readings_df.groupby("meter_id"):
        readings = group["meter_reading"].values
        assert all(readings[i] <= readings[i + 1] for i in range(len(readings) - 1)), (
            f"meter_reading is NOT monotonically non-decreasing for meter {meter_id!r}"
        )


def test_meter_reading_dtype(readings_df):
    """meter_reading column must be numeric."""
    assert pd.api.types.is_numeric_dtype(readings_df["meter_reading"]), (
        "meter_reading column is not numeric"
    )


def test_timestamp_format(readings_df):
    """timestamp column must be parseable as ISO 8601 datetimes."""
    try:
        pd.to_datetime(readings_df["timestamp"])
    except Exception as exc:
        pytest.fail(f"timestamp column is not valid ISO 8601: {exc}")


def test_all_12_meters(readings_df):
    """Exactly 12 unique meters must be present."""
    assert readings_df["meter_id"].nunique() == 12, (
        f"Expected 12 meters, got {readings_df['meter_id'].nunique()}"
    )


def test_hostel_ids_present(readings_df):
    """All three hostel IDs (H1, H2, H3) must be present."""
    hostels = set(readings_df["hostel_id"].unique())
    assert hostels == {"H1", "H2", "H3"}, f"Unexpected hostel IDs: {hostels}"
