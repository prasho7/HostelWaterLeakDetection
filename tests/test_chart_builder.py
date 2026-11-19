"""
test_chart_builder.py — Unit tests for interactive Plotly chart generators.
"""

import os
import sys
import pandas as pd
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_TESTS_DIR, "..", "src")
for _p in (_TESTS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chart_builder import (
    filter_by_date_range,
    build_floor_telemetry_chart,
    build_hostel_telemetry_chart,
)


@pytest.fixture
def sample_telemetry_data():
    """Create sample readings, anomalies, and alerts for testing."""
    times = pd.date_range("2024-01-01 00:00:00", periods=50, freq="1h")
    readings = pd.DataFrame({
        "timestamp": times,
        "hostel_id": "H1",
        "floor_id": "H1-F1",
        "meter_id": "M-H1-F1",
        "delta_litres": [20.0 + (i % 10) for i in range(50)],
        "meter_reading": [100.0 + i * 25.0 for i in range(50)],
    })
    # Add an anomaly at index 15 (isolated) and a sustained anomaly at indices 30-34
    anomalies = pd.DataFrame({
        "timestamp": times,
        "hostel_id": "H1",
        "floor_id": "H1-F1",
        "meter_id": "M-H1-F1",
        "delta_litres": readings["delta_litres"].copy(),
        "median_baseline": [25.0] * 50,
        "mad": [5.0] * 50,
        "z_score_mad": [0.5] * 50,
        "is_anomaly": [False] * 50,
        "persistence_count": [0] * 50,
    })
    # Set isolated anomaly
    anomalies.loc[15, "is_anomaly"] = True
    anomalies.loc[15, "z_score_mad"] = 8.5
    anomalies.loc[15, "delta_litres"] = 95.0
    readings.loc[15, "delta_litres"] = 95.0

    # Set sustained anomaly (indices 30 to 34)
    for i in range(30, 35):
        anomalies.loc[i, "is_anomaly"] = True
        anomalies.loc[i, "z_score_mad"] = 12.0
        anomalies.loc[i, "delta_litres"] = 120.0
        readings.loc[i, "delta_litres"] = 120.0

    alerts = pd.DataFrame([{
        "alert_id": "alert-test-1",
        "hostel_id": "H1",
        "floor_id": "H1-F1",
        "meter_id": "M-H1-F1",
        "first_seen": times[30],
        "last_seen": times[34],
        "severity": "HIGH",
        "avg_z_score": 12.0,
        "persistence_count": 5,
        "status": "OPEN",
        "resolved_at": "",
    }])

    return readings, anomalies, alerts


def test_filter_by_date_range(sample_telemetry_data):
    readings, _, _ = sample_telemetry_data

    # All Time
    all_df = filter_by_date_range(readings, "All Time")
    assert len(all_df) == 50

    # Last 24 Hours
    df_24h = filter_by_date_range(readings, "Last 24 Hours")
    assert len(df_24h) == 25  # 24 hours back from max includes 25 points at 1h interval

    # Last 7 Days (all 50 points are within ~2 days)
    df_7d = filter_by_date_range(readings, "Last 7 Days")
    assert len(df_7d) == 50

    # Custom Range
    t_min = readings["timestamp"].min()
    t_mid = readings["timestamp"].iloc[20]
    df_custom = filter_by_date_range(readings, "Custom Range", custom_dates=(t_min.date(), t_mid.date()))
    assert len(df_custom) > 0 and len(df_custom) <= 50


def test_build_floor_telemetry_chart(sample_telemetry_data):
    readings, anomalies, alerts = sample_telemetry_data
    fig = build_floor_telemetry_chart(
        readings,
        anomalies,
        alerts,
        meter_id="M-H1-F1",
        time_range="All Time",
    )

    trace_names = [t.name for t in fig.data]
    assert "Water Flow (L/hr)" in trace_names
    assert "Baseline (Median)" in trace_names
    assert "Threshold (Z=3.5)" in trace_names
    assert "Unconfirmed Anomalies (Spikes < 3h)" in trace_names
    assert "Confirmed Leak Alert Readings" in trace_names

    # Check that alert window shape exists
    assert len(fig.layout.shapes) > 0


def test_build_floor_telemetry_chart_empty():
    fig = build_floor_telemetry_chart(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        meter_id="M-H1-F1",
    )
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) > 0


def test_build_hostel_telemetry_chart(sample_telemetry_data):
    readings, _, _ = sample_telemetry_data
    # Replicate for 4 floors
    r1 = readings.copy()
    r2 = readings.copy()
    r2["floor_id"] = "H1-F2"
    r2["meter_id"] = "M-H1-F2"
    hostel_r = pd.concat([r1, r2], ignore_index=True)

    fig = build_hostel_telemetry_chart(hostel_r, hostel_id="H1")
    trace_names = [t.name for t in fig.data]
    assert any("H1-F1" in name for name in trace_names)
    assert any("H1-F2" in name for name in trace_names)
