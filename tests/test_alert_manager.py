"""
test_alert_manager.py — pytest suite for alert_manager.py

Tests run against the live alerts.csv produced by alert_manager.py.
If alerts.csv does not yet exist the test session will auto-generate it.
"""

import os
import sys
import subprocess

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Resolve project root so config.py and src/ are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PERSISTENCE_MIN, LEAK_INJECT_METERS  # noqa: E402

ALERTS_PATH          = os.path.join(PROJECT_ROOT, "data", "alerts.csv")
ALERT_MANAGER_SCRIPT = os.path.join(PROJECT_ROOT, "src", "alert_manager.py")

EXPECTED_COLUMNS = {
    "alert_id",
    "hostel_id",
    "floor_id",
    "meter_id",
    "first_seen",
    "last_seen",
    "severity",
    "avg_z_score",
    "persistence_count",
    "status",
    "resolved_at",
}

KNOWN_LEAK_METERS = set(LEAK_INJECT_METERS)
VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}
VALID_STATUSES   = {"OPEN", "RESOLVED"}


# ---------------------------------------------------------------------------
# Session-scoped fixture: generate alerts.csv once for the whole test session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def alerts_df():
    """Run alert_manager.py if needed and return the resulting DataFrame."""
    result = subprocess.run(
        [sys.executable, ALERT_MANAGER_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"alert_manager.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert os.path.exists(ALERTS_PATH), "alerts.csv was not created by alert_manager.py"

    df = pd.read_csv(ALERTS_PATH)
    return df


# ---------------------------------------------------------------------------
# 1. Output CSV has all required columns
# ---------------------------------------------------------------------------
def test_output_has_all_columns(alerts_df):
    missing = EXPECTED_COLUMNS - set(alerts_df.columns)
    assert not missing, f"alerts.csv is missing columns: {missing}"


# ---------------------------------------------------------------------------
# 2. Every alert_id is unique (no duplicate UUIDs)
# ---------------------------------------------------------------------------
def test_alert_ids_are_unique(alerts_df):
    if alerts_df.empty:
        pytest.skip("No alerts generated — uniqueness check skipped.")
    assert alerts_df["alert_id"].nunique() == len(alerts_df), (
        "Duplicate alert_id values found in alerts.csv"
    )


# ---------------------------------------------------------------------------
# 3. severity values are only in {LOW, MEDIUM, HIGH}
# ---------------------------------------------------------------------------
def test_severity_values_valid(alerts_df):
    if alerts_df.empty:
        pytest.skip("No alerts generated — severity check skipped.")
    invalid = set(alerts_df["severity"].unique()) - VALID_SEVERITIES
    assert not invalid, f"Unexpected severity values: {invalid}"


# ---------------------------------------------------------------------------
# 4. status values are valid (OPEN or RESOLVED)
# ---------------------------------------------------------------------------
def test_status_values_valid(alerts_df):
    if alerts_df.empty:
        pytest.skip("No alerts generated — status check skipped.")
    invalid = set(alerts_df["status"].unique()) - VALID_STATUSES
    assert not invalid, f"Unexpected status values: {invalid}"


# ---------------------------------------------------------------------------
# 5. persistence_count >= PERSISTENCE_MIN for all rows
# ---------------------------------------------------------------------------
def test_persistence_count_meets_minimum(alerts_df):
    if alerts_df.empty:
        pytest.skip("No alerts generated — persistence check skipped.")
    below_min = alerts_df[alerts_df["persistence_count"] < PERSISTENCE_MIN]
    assert below_min.empty, (
        f"Found {len(below_min)} alert(s) with persistence_count < {PERSISTENCE_MIN}:\n"
        f"{below_min}"
    )


# ---------------------------------------------------------------------------
# 6. Known leak meters each have at least one alert
# ---------------------------------------------------------------------------
def test_known_leak_meters_have_alerts(alerts_df):
    assert not alerts_df.empty, "alerts.csv is empty — no alerts generated at all."
    meters_with_alerts = set(alerts_df["meter_id"].unique())
    missing_meters = KNOWN_LEAK_METERS - meters_with_alerts
    assert not missing_meters, (
        f"Known leak meter(s) have no alerts: {missing_meters}"
    )


# ---------------------------------------------------------------------------
# 7. hostel_id and floor_id are populated on every alert row
# ---------------------------------------------------------------------------
def test_hostel_and_floor_populated(alerts_df):
    if alerts_df.empty:
        pytest.skip("No alerts generated.")
    for col in ("hostel_id", "floor_id"):
        assert alerts_df[col].notna().all(), f"Null values found in alert column {col!r}"
