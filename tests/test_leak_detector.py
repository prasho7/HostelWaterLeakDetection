"""
test_leak_detector.py — Pytest suite for leak_detector.py

Runs the full detection pipeline against the real processed_readings.csv
and verifies the output anomalies.csv satisfies all specified constraints.
"""

import os
import sys
import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so config and src are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import LEAK_INJECT_METERS          # noqa: E402
import src.leak_detector as ld                 # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed_readings.csv")
ANOMALIES_CSV = os.path.join(PROJECT_ROOT, "data", "anomalies.csv")

# ---------------------------------------------------------------------------
# Session-scoped fixture: run the detector once, load both CSVs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def run_detector():
    """Run the full detection pipeline and return (anomalies_df, processed_df)."""
    ld.run()
    anomalies_df = pd.read_csv(ANOMALIES_CSV)
    processed_df = pd.read_csv(PROCESSED_CSV)
    return anomalies_df, processed_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_columns_present(run_detector):
    """Assert all required output columns exist in anomalies.csv."""
    anomalies_df, _ = run_detector
    required_cols = {
        "timestamp", "hostel_id", "floor_id", "meter_id", "delta_litres",
        "median_baseline", "mad", "z_score_mad",
        "is_anomaly", "persistence_count",
    }
    missing = required_cols - set(anomalies_df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_persistence_count_non_negative(run_detector):
    """Assert persistence_count >= 0 for every row."""
    anomalies_df, _ = run_detector
    assert (anomalies_df["persistence_count"] >= 0).all(), \
        "Found negative persistence_count values"


def test_is_anomaly_boolean_dtype(run_detector):
    """Assert is_anomaly is boolean dtype (True/False, not 0/1 int)."""
    anomalies_df, _ = run_detector
    assert anomalies_df["is_anomaly"].dtype == bool, \
        f"Expected bool dtype, got {anomalies_df['is_anomaly'].dtype}"


def test_known_leak_meters_have_anomalies(run_detector):
    """Assert each known leak-injected meter has at least one is_anomaly=True row."""
    anomalies_df, _ = run_detector
    for meter_id in LEAK_INJECT_METERS:
        meter_rows = anomalies_df[anomalies_df["meter_id"] == meter_id]
        assert meter_rows["is_anomaly"].any(), \
            f"Leak meter {meter_id} has no anomaly detected"


def test_row_count_matches_processed(run_detector):
    """Assert anomalies.csv has the same number of rows as processed_readings.csv."""
    anomalies_df, processed_df = run_detector
    assert len(anomalies_df) == len(processed_df), (
        f"Row count mismatch: anomalies={len(anomalies_df)}, "
        f"processed={len(processed_df)}"
    )


def test_all_12_meters_in_anomalies(run_detector):
    """Assert all 12 meters appear in anomalies.csv."""
    anomalies_df, _ = run_detector
    assert anomalies_df["meter_id"].nunique() == 12, (
        f"Expected 12 meters in anomalies.csv, got {anomalies_df['meter_id'].nunique()}"
    )
