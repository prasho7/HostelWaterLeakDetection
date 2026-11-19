"""
test_dashboard.py — Unit tests for dashboard helper functions.

Tests target the three pure functions defined at module level in dashboard.py:
  - load_alerts(path)
  - load_readings(path)
  - filter_alerts(df, hostel_id, floor_id, severity)

These tests do NOT launch Streamlit; they exercise the data-loading and
filtering logic directly using temporary CSV files.
"""

import os
import sys
import textwrap
import tempfile

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure src/ is on the path so we can import dashboard without installing
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.join(_TESTS_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Import pure helper functions — do NOT call main()
from dashboard import (  # noqa: E402
    DEFAULT_REFRESH_SECS,
    STALE_MULTIPLIER,
    get_alert_counts,
    get_latest_reading_time,
    get_stale_meters,
    load_alerts,
    load_readings,
    load_anomalies,
    filter_alerts,
    get_hostel_status,
    get_floor_status,
    get_alert_card_style,
)


# ---------------------------------------------------------------------------
# Fixtures — write minimal CSV files into a temp directory
# ---------------------------------------------------------------------------

ALERTS_CSV_CONTENT = textwrap.dedent("""\
    alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
    aaaa-0001,H1,H1-F1,M-H1-F1,2024-01-01T03:00:00,2024-01-02T08:00:00,HIGH,34.98,30,OPEN,
    bbbb-0002,H1,H1-F2,M-H1-F2,2024-01-05T10:00:00,2024-01-05T18:00:00,MEDIUM,12.50,10,OPEN,
    cccc-0003,H2,H2-F1,M-H2-F1,2024-01-10T06:00:00,2024-01-10T12:00:00,LOW,3.20,5,OPEN,
    dddd-0004,H2,H2-F3,M-H2-F3,2024-01-15T00:00:00,2024-01-15T06:00:00,HIGH,40.00,25,RESOLVED,2024-01-16T09:00:00
""")

READINGS_CSV_CONTENT = textwrap.dedent("""\
    timestamp,hostel_id,floor_id,meter_id,meter_reading,delta_litres,hour,day_of_week
    2024-01-01 00:00:00,H1,H1-F1,M-H1-F1,18.5,0.0,0,0
    2024-01-01 01:00:00,H1,H1-F1,M-H1-F1,33.4,14.9,1,0
    2024-01-01 00:00:00,H1,H1-F2,M-H1-F2,20.1,0.0,0,0
    2024-01-01 01:00:00,H1,H1-F2,M-H1-F2,38.6,18.5,1,0
    2024-01-01 00:00:00,H2,H2-F1,M-H2-F1,15.3,0.0,0,0
""")


@pytest.fixture(scope="module")
def tmp_data_dir():
    """Create a temporary directory with sample CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        alerts_path   = os.path.join(tmpdir, "alerts.csv")
        readings_path = os.path.join(tmpdir, "processed_readings.csv")

        with open(alerts_path,   "w", newline="") as f:
            f.write(ALERTS_CSV_CONTENT)
        with open(readings_path, "w", newline="") as f:
            f.write(READINGS_CSV_CONTENT)

        yield {"alerts": alerts_path, "readings": readings_path}


# ---------------------------------------------------------------------------
# Tests for load_alerts
# ---------------------------------------------------------------------------

EXPECTED_ALERT_COLUMNS = {
    "alert_id", "hostel_id", "floor_id", "meter_id",
    "first_seen", "last_seen", "severity",
    "avg_z_score", "persistence_count", "status", "resolved_at",
}


class TestLoadAlerts:
    def test_returns_dataframe(self, tmp_data_dir):
        df = load_alerts(tmp_data_dir["alerts"])
        assert isinstance(df, pd.DataFrame)

    def test_has_all_11_columns(self, tmp_data_dir):
        df = load_alerts(tmp_data_dir["alerts"])
        assert set(df.columns) == EXPECTED_ALERT_COLUMNS

    def test_first_seen_is_datetime(self, tmp_data_dir):
        df = load_alerts(tmp_data_dir["alerts"])
        assert pd.api.types.is_datetime64_any_dtype(df["first_seen"])

    def test_last_seen_is_datetime(self, tmp_data_dir):
        df = load_alerts(tmp_data_dir["alerts"])
        assert pd.api.types.is_datetime64_any_dtype(df["last_seen"])

    def test_row_count(self, tmp_data_dir):
        df = load_alerts(tmp_data_dir["alerts"])
        assert len(df) == 4

    def test_missing_file_returns_empty_df(self):
        df = load_alerts("/nonexistent/path/alerts.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert set(df.columns) == EXPECTED_ALERT_COLUMNS


# ---------------------------------------------------------------------------
# Tests for load_readings
# ---------------------------------------------------------------------------

EXPECTED_READING_COLUMNS = {
    "timestamp", "hostel_id", "floor_id", "meter_id",
    "meter_reading", "delta_litres", "hour", "day_of_week",
}


class TestLoadReadings:
    def test_returns_dataframe(self, tmp_data_dir):
        df = load_readings(tmp_data_dir["readings"])
        assert isinstance(df, pd.DataFrame)

    def test_has_all_8_columns(self, tmp_data_dir):
        df = load_readings(tmp_data_dir["readings"])
        assert set(df.columns) == EXPECTED_READING_COLUMNS

    def test_timestamp_is_datetime(self, tmp_data_dir):
        df = load_readings(tmp_data_dir["readings"])
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_row_count(self, tmp_data_dir):
        df = load_readings(tmp_data_dir["readings"])
        assert len(df) == 5

    def test_missing_file_returns_empty_df(self):
        df = load_readings("/nonexistent/path/processed_readings.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert set(df.columns) == EXPECTED_READING_COLUMNS


EXPECTED_ANOMALY_COLUMNS = {
    "timestamp", "hostel_id", "floor_id", "meter_id",
    "delta_litres", "median_baseline", "mad", "z_score_mad",
    "is_anomaly", "persistence_count",
}


class TestLoadAnomalies:
    def test_missing_file_returns_empty_df(self):
        df = load_anomalies("/nonexistent/path/anomalies.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert set(df.columns) == EXPECTED_ANOMALY_COLUMNS


# ---------------------------------------------------------------------------
# Tests for filter_alerts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alerts_df(tmp_data_dir):
    return load_alerts(tmp_data_dir["alerts"])


class TestFilterAlerts:
    def test_no_filters_returns_all_rows(self, alerts_df):
        result = filter_alerts(alerts_df)
        assert len(result) == len(alerts_df)

    def test_filter_by_hostel_id(self, alerts_df):
        result = filter_alerts(alerts_df, hostel_id="H1")
        assert len(result) == 2
        assert set(result["hostel_id"].unique()) == {"H1"}

    def test_filter_by_floor_id(self, alerts_df):
        result = filter_alerts(alerts_df, floor_id="H2-F3")
        assert len(result) == 1
        assert result.iloc[0]["floor_id"] == "H2-F3"

    def test_filter_by_severity_high(self, alerts_df):
        result = filter_alerts(alerts_df, severity="HIGH")
        assert len(result) == 2
        assert all(result["severity"] == "HIGH")

    def test_filter_by_severity_medium(self, alerts_df):
        result = filter_alerts(alerts_df, severity="MEDIUM")
        assert len(result) == 1
        assert all(result["severity"] == "MEDIUM")

    def test_filter_by_severity_low(self, alerts_df):
        result = filter_alerts(alerts_df, severity="LOW")
        assert len(result) == 1
        assert all(result["severity"] == "LOW")

    def test_filter_severity_all_returns_all_rows(self, alerts_df):
        result = filter_alerts(alerts_df, severity="All")
        assert len(result) == len(alerts_df)

    def test_filter_hostel_all_returns_all_rows(self, alerts_df):
        result = filter_alerts(alerts_df, hostel_id="All")
        assert len(result) == len(alerts_df)

    def test_filter_by_hostel_and_severity(self, alerts_df):
        result = filter_alerts(alerts_df, hostel_id="H2", severity="HIGH")
        assert len(result) == 1
        assert result.iloc[0]["hostel_id"] == "H2"
        assert result.iloc[0]["severity"] == "HIGH"

    def test_filter_nonexistent_hostel_returns_empty(self, alerts_df):
        result = filter_alerts(alerts_df, hostel_id="H9")
        assert result.empty

    def test_original_df_not_mutated(self, alerts_df):
        original_len = len(alerts_df)
        filter_alerts(alerts_df, hostel_id="H1", severity="HIGH")
        assert len(alerts_df) == original_len


# ---------------------------------------------------------------------------
# Tests for Visual States (OPEN vs RESOLVED)
# ---------------------------------------------------------------------------

class TestVisualStates:
    def test_open_high_alert_card_has_red_styling(self):
        row = {"status": "OPEN", "severity": "HIGH"}
        style = get_alert_card_style(row)
        assert style["is_resolved"] is False
        assert style["border_color"] == "#dc2626"
        assert style["bg_color"] == "#fef2f2"
        assert "HIGH" in style["badge_html"]
        assert "#dc2626" in style["badge_html"]

    def test_resolved_alert_card_has_neutral_green_styling(self):
        row = {"status": "RESOLVED", "severity": "HIGH"}
        style = get_alert_card_style(row)
        assert style["is_resolved"] is True
        assert style["border_color"] == "#16a34a"
        assert style["bg_color"] == "#f0fdf4"
        assert "RESOLVED" in style["badge_html"]
        assert "#dc2626" not in style["border_color"]
        assert "#dc2626" not in style["bg_color"]

    def test_hostel_status_with_open_high_alert(self):
        df = pd.DataFrame([
            {"status": "OPEN", "severity": "HIGH"},
            {"status": "RESOLVED", "severity": "HIGH"},
        ])
        status, brd, bg = get_hostel_status(df)
        assert status == "⚠️ Leak Detected"
        assert brd == "#dc2626"

    def test_hostel_status_when_all_alerts_resolved(self):
        df = pd.DataFrame([
            {"status": "RESOLVED", "severity": "HIGH"},
            {"status": "RESOLVED", "severity": "MEDIUM"},
        ])
        status, brd, bg = get_hostel_status(df)
        assert status == "✅ Normal"
        assert brd == "#16a34a"
        assert bg == "#f0fdf4"

    def test_floor_status_with_open_high_alert(self):
        df = pd.DataFrame([
            {"status": "OPEN", "severity": "HIGH"},
        ])
        status, brd, bg = get_floor_status(df)
        assert status == "⚠️ LEAK ALERT"
        assert brd == "#dc2626"

    def test_floor_status_when_all_alerts_resolved(self):
        df = pd.DataFrame([
            {"status": "RESOLVED", "severity": "HIGH"},
        ])
        status, brd, bg = get_floor_status(df)
        assert status == "✅ Normal"
        assert brd == "#16a34a"
        assert bg == "#f0fdf4"

    def test_empty_alerts_status(self):
        status, brd, bg = get_hostel_status(pd.DataFrame())
        assert status == "✅ Normal"
        assert brd == "#16a34a"


# ---------------------------------------------------------------------------
# Tests for Phase 4 — Auto-refresh helpers
# ---------------------------------------------------------------------------

READINGS_WITH_TIMESTAMPS_CONTENT = textwrap.dedent("""\
    timestamp,hostel_id,floor_id,meter_id,meter_reading,delta_litres,hour,day_of_week
    2024-01-15 10:00:00,H1,H1-F1,M-H1-F1,100.0,10.0,10,0
    2024-01-15 10:00:00,H1,H1-F2,M-H1-F2,200.0,20.0,10,0
    2024-01-15 10:00:00,H2,H2-F1,M-H2-F1,150.0,15.0,10,0
    2024-01-15 09:00:00,H2,H2-F2,M-H2-F2,90.0,9.0,9,0
""")


@pytest.fixture(scope="module")
def readings_with_ts(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("phase4")
    path   = str(tmpdir / "processed_readings.csv")
    with open(path, "w", newline="") as f:
        f.write(READINGS_WITH_TIMESTAMPS_CONTENT)
    return load_readings(path)


@pytest.fixture(scope="module")
def alerts_mixed(tmp_path_factory):
    """4 alerts: 2 OPEN (1 HIGH, 1 MEDIUM), 2 RESOLVED."""
    tmpdir = tmp_path_factory.mktemp("phase4a")
    path   = str(tmpdir / "alerts.csv")
    content = textwrap.dedent("""\
        alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
        a001,H1,H1-F1,M-H1-F1,2024-01-01T00:00:00,2024-01-01T10:00:00,HIGH,35.0,10,OPEN,
        a002,H1,H1-F2,M-H1-F2,2024-01-02T00:00:00,2024-01-02T05:00:00,MEDIUM,8.5,5,OPEN,
        a003,H2,H2-F1,M-H2-F1,2024-01-03T00:00:00,2024-01-03T04:00:00,LOW,4.0,3,RESOLVED,2024-01-04T08:00:00
        a004,H2,H2-F3,M-H2-F3,2024-01-05T00:00:00,2024-01-05T12:00:00,HIGH,40.0,12,RESOLVED,2024-01-06T10:00:00
    """)
    with open(path, "w", newline="") as f:
        f.write(content)
    return load_alerts(path)


class TestGetLatestReadingTime:
    def test_returns_latest_timestamp(self, readings_with_ts):
        ts = get_latest_reading_time(readings_with_ts)
        assert ts is not None
        assert ts == pd.Timestamp("2024-01-15 10:00:00").to_pydatetime()

    def test_empty_df_returns_none(self):
        empty = pd.DataFrame(columns=["timestamp", "hostel_id", "floor_id",
                                      "meter_id", "meter_reading",
                                      "delta_litres", "hour", "day_of_week"])
        ts = get_latest_reading_time(empty)
        assert ts is None

    def test_missing_timestamp_column_returns_none(self):
        df = pd.DataFrame({"meter_id": ["M-H1-F1"]})
        ts = get_latest_reading_time(df)
        assert ts is None


class TestGetStaleMeters:
    def test_all_meters_present_no_stale(self, tmp_path_factory):
        """All 12 meters at the same timestamp → none are stale."""
        from config import METERS
        rows = []
        for m in METERS:
            rows.append({
                "timestamp":     pd.Timestamp("2024-01-15 10:00:00"),
                "hostel_id":     m["hostel_id"],
                "floor_id":      m["floor_id"],
                "meter_id":      m["meter_id"],
                "meter_reading": 100.0,
                "delta_litres":  10.0,
                "hour":          10,
                "day_of_week":   0,
            })
        df    = pd.DataFrame(rows)
        stale = get_stale_meters(df, stale_threshold_seconds=3600)
        assert stale == []

    def test_lagging_meter_is_stale(self, readings_with_ts):
        """M-H2-F2 is 1 hour behind the latest reading — stale at 1800s threshold."""
        stale = get_stale_meters(readings_with_ts, stale_threshold_seconds=1800)
        assert "M-H2-F2" in stale

    def test_lagging_meter_not_stale_at_loose_threshold(self, readings_with_ts):
        """At 7200s threshold the 1-hour gap should not be flagged."""
        stale = get_stale_meters(readings_with_ts, stale_threshold_seconds=7200)
        assert "M-H2-F2" not in stale

    def test_missing_meter_always_stale(self, readings_with_ts):
        """Meters not present in readings are always stale."""
        stale = get_stale_meters(readings_with_ts, stale_threshold_seconds=1)
        # readings_with_ts only has 4 of the 12 meters
        from config import METER_IDS
        present = set(readings_with_ts["meter_id"].unique())
        absent  = set(METER_IDS) - present
        for mid in absent:
            assert mid in stale, f"Missing meter {mid} should be stale"

    def test_empty_readings_all_meters_stale(self):
        empty = pd.DataFrame(columns=["timestamp", "hostel_id", "floor_id",
                                      "meter_id", "meter_reading",
                                      "delta_litres", "hour", "day_of_week"])
        stale = get_stale_meters(empty, stale_threshold_seconds=60)
        from config import METER_IDS
        for mid in METER_IDS:
            assert mid in stale


class TestGetAlertCounts:
    def test_mixed_alerts(self, alerts_mixed):
        counts = get_alert_counts(alerts_mixed)
        assert counts["open_total"]    == 2
        assert counts["open_high"]     == 1
        assert counts["open_medium"]   == 1
        assert counts["open_low"]      == 0
        assert counts["resolved_total"] == 2

    def test_empty_alerts(self):
        counts = get_alert_counts(pd.DataFrame())
        assert counts["open_total"]     == 0
        assert counts["resolved_total"] == 0

    def test_all_open(self, tmp_data_dir):
        df     = load_alerts(tmp_data_dir["alerts"])
        open_  = df[df["status"] == "OPEN"]
        counts = get_alert_counts(open_)
        assert counts["open_total"]     == 3
        assert counts["resolved_total"] == 0

    def test_all_resolved(self, tmp_data_dir):
        df       = load_alerts(tmp_data_dir["alerts"])
        resolved = df[df["status"] == "RESOLVED"]
        counts   = get_alert_counts(resolved)
        assert counts["open_total"]     == 0
        assert counts["resolved_total"] == 1

    def test_counts_sum_matches_total(self, alerts_mixed):
        counts = get_alert_counts(alerts_mixed)
        assert (counts["open_high"] + counts["open_medium"] + counts["open_low"]
                == counts["open_total"])


class TestAutoRefreshConfig:
    def test_default_refresh_secs_positive(self):
        assert DEFAULT_REFRESH_SECS > 0

    def test_stale_multiplier_positive(self):
        assert STALE_MULTIPLIER > 0

    def test_load_readings_parses_mixed_timestamps(self, tmp_path_factory):
        """
        load_readings must handle both ISO-T and space-format timestamps in the
        same CSV (produced by mixed pipeline runs).
        """
        tmpdir = tmp_path_factory.mktemp("mixed_ts")
        path   = str(tmpdir / "processed_readings.csv")
        content = textwrap.dedent("""\
            timestamp,hostel_id,floor_id,meter_id,meter_reading,delta_litres,hour,day_of_week
            2024-01-01T00:00:00,H1,H1-F1,M-H1-F1,100.0,10.0,0,0
            2024-01-01 01:00:00,H1,H1-F1,M-H1-F1,120.0,20.0,1,0
        """)
        with open(path, "w", newline="") as f:
            f.write(content)
        df = load_readings(path)
        assert len(df) == 2
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load_readings_no_pipeline_side_effects(self, tmp_data_dir):
        """
        Calling load_readings repeatedly must not change the CSV file or
        trigger any pipeline logic (pure read).
        """
        import os
        path   = tmp_data_dir["readings"]
        mtime1 = os.path.getmtime(path)
        load_readings(path)
        load_readings(path)
        mtime2 = os.path.getmtime(path)
        assert mtime1 == mtime2, "load_readings must not write to the file"

    def test_new_alerts_appear_after_csv_update(self, tmp_path_factory):
        """
        When the CSV is updated on disk, the next load_alerts call must return
        the new data (no stale cache).
        """
        tmpdir = tmp_path_factory.mktemp("alert_update")
        path   = str(tmpdir / "alerts.csv")

        # Write initial CSV
        initial = textwrap.dedent("""\
            alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
            x001,H1,H1-F1,M-H1-F1,2024-01-01T00:00:00,2024-01-01T10:00:00,HIGH,35.0,10,OPEN,
        """)
        with open(path, "w", newline="") as f:
            f.write(initial)
        df1 = load_alerts(path)
        assert len(df1) == 1

        # Append a new alert
        with open(path, "a", newline="") as f:
            f.write("x002,H2,H2-F1,M-H2-F1,2024-01-02T00:00:00,2024-01-02T05:00:00,"
                    "MEDIUM,8.5,5,OPEN,\n")
        df2 = load_alerts(path)
        assert len(df2) == 2, (
            "load_alerts must return fresh data on each call (no stale cache)"
        )

    def test_resolved_alert_status_reflected_on_reload(self, tmp_path_factory):
        """
        After alert status changes to RESOLVED on disk, the next load_alerts
        must return the updated status.
        """
        tmpdir = tmp_path_factory.mktemp("resolved_update")
        path   = str(tmpdir / "alerts.csv")

        content = textwrap.dedent("""\
            alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
            y001,H1,H1-F1,M-H1-F1,2024-01-01T00:00:00,2024-01-01T10:00:00,HIGH,35.0,10,OPEN,
        """)
        with open(path, "w", newline="") as f:
            f.write(content)

        df_open = load_alerts(path)
        assert df_open.iloc[0]["status"] == "OPEN"

        # Simulate pipeline resolving the alert
        resolved_content = content.replace(",OPEN,", ",RESOLVED,2024-01-02T09:00:00")
        with open(path, "w", newline="") as f:
            f.write(resolved_content)

        df_resolved = load_alerts(path)
        assert df_resolved.iloc[0]["status"] == "RESOLVED", (
            "Updated status must be visible after re-reading the file"
        )

