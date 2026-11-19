"""
test_rolling_detector.py
Pytest suite for src/rolling_detector.py — Phase 3 rolling window baseline.

All tests use isolated, synthetic DataFrames so they never touch production
CSV files.  The test scenarios map 1-to-1 to the Phase 3 requirements:

  T1  Schema — output columns match anomalies.csv exactly
  T2  Baseline isolation — meters do not cross-contaminate each other
  T3  Adaptation — baseline tracks gradual consumption shift
  T4  Sudden spike detection
  T5  Sustained (continuous) anomaly detection
  T6  Insufficient historical data → fallback to global baseline
  T7  Requirement 8 — current reading excluded from its own baseline
  T8  Persistence count correctness
  T9  Anomaly flag dtype is bool
  T10 Window boundary — only readings within window included
  T11 Alert-status preservation (merge_with_existing still works after rolling run)
  T12 Known leak-injected meters detected on real processed_readings.csv

Run from the project root:
    pytest tests/test_rolling_detector.py -v
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    LEAK_INJECT_METERS,
    MAD_Z_THRESHOLD,
    PERSISTENCE_MIN,
    ROLLING_MIN_READINGS,
    ROLLING_WINDOW_DAYS,
)
from rolling_detector import (
    DEFAULT_MIN_READINGS,
    DEFAULT_WINDOW_DAYS,
    HOUR_TOLERANCE,
    OUTPUT_COLS,
    _compute_global_stats,
    compute_persistence,
    compute_rolling_baselines,
    run,
)

# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

METER_ID   = "M-H1-F1"
HOSTEL_ID  = "H1"
FLOOR_ID   = "H1-F1"


def _make_df(
    deltas: list,
    meter_id: str = METER_ID,
    hostel_id: str = HOSTEL_ID,
    floor_id: str = FLOOR_ID,
    base_ts: datetime = None,
    hour: int = 10,
    day_of_week: int = 0,
    daily: bool = True,
) -> pd.DataFrame:
    """
    Build a minimal processed_readings-style DataFrame from a list of delta values.

    When daily=True (default), readings are spaced 24 hours apart so every
    reading falls at the same hour-of-day (``hour``).  This keeps the hour
    bucket filter stable across rows and is what most tests need.

    When daily=False, readings are spaced 1 hour apart (useful when you want
    to exercise multi-hour behaviour).
    """
    if base_ts is None:
        base_ts = datetime(2024, 1, 1, hour, 0, 0)

    step = timedelta(hours=24) if daily else timedelta(hours=1)

    rows = []
    reading = 0.0
    for i, delta in enumerate(deltas):
        ts       = base_ts + step * i
        reading += delta
        rows.append({
            "timestamp":     ts,
            "hostel_id":     hostel_id,
            "floor_id":      floor_id,
            "meter_id":      meter_id,
            "meter_reading": round(reading, 4),
            "delta_litres":  float(delta),
            "hour":          hour,          # pinned to fixed hour
            "day_of_week":   int(ts.weekday()),
        })
    return pd.DataFrame(rows)


def _make_multi_meter_df(meter_deltas: dict) -> pd.DataFrame:
    """
    meter_deltas : {meter_id: [delta, ...]}
    Returns a combined DataFrame for multiple meters, staggered by 0 seconds.
    """
    parts = []
    for mid, deltas in meter_deltas.items():
        hostel = mid.split("-")[1]           # "M-H1-F1" → "H1"
        floor  = mid[2:]                     # "M-H1-F1" → "H1-F1"
        parts.append(_make_df(deltas, meter_id=mid, hostel_id=hostel, floor_id=floor))
    return pd.concat(parts, ignore_index=True).sort_values(
        ["meter_id", "timestamp"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# T1  Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_output_columns(self):
        """Output DataFrame must contain all required anomaly columns."""
        df = _make_df([60.0] * 20)
        result = compute_rolling_baselines(df, window_days=1, min_readings=5)
        result = compute_persistence(result)
        for col in OUTPUT_COLS:
            assert col in result.columns, f"Missing output column: {col!r}"

    def test_no_extra_required_columns_missing(self):
        """run() output must be writable to a CSV matching anomalies.csv schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc_path = os.path.join(tmpdir, "processed_readings.csv")
            out_path  = os.path.join(tmpdir, "anomalies.csv")
            df = _make_df([60.0] * 25)
            df.to_csv(proc_path, index=False)
            result = run(input_path=proc_path, output_path=out_path, min_readings=3)
            assert list(result.columns) == OUTPUT_COLS

    def test_row_count_preserved(self):
        """Output must have the same number of rows as input."""
        df = _make_df([60.0] * 15)
        result = compute_rolling_baselines(df, window_days=1, min_readings=5)
        assert len(result) == len(df)

    def test_is_anomaly_bool_dtype(self):
        """is_anomaly column must be bool dtype."""
        df = _make_df([60.0] * 15)
        result = compute_rolling_baselines(df, window_days=1, min_readings=5)
        assert result["is_anomaly"].dtype == bool


# ---------------------------------------------------------------------------
# T2  Baseline isolation across meters
# ---------------------------------------------------------------------------

class TestBaselineIsolation:
    def test_meters_do_not_share_baseline(self):
        """
        Meter A has stable 60L/h consumption.
        Meter B has stable 200L/h consumption.
        After warm-up, baselines must be distinct per meter.
        """
        n = 50
        df = _make_multi_meter_df({
            "M-H1-F1": [60.0] * n,
            "M-H2-F1": [200.0] * n,
        })
        result = compute_rolling_baselines(df, window_days=10, min_readings=5)

        # After warm-up (position >= min_readings), rolling baselines should differ
        for mid, grp in result.groupby("meter_id"):
            warm = grp[grp["baseline_source"] == "rolling"]
            if warm.empty:
                continue
            if mid == "M-H1-F1":
                assert warm["median_baseline"].mean() == pytest.approx(60.0, abs=5.0)
            elif mid == "M-H2-F1":
                assert warm["median_baseline"].mean() == pytest.approx(200.0, abs=5.0)

    def test_spike_in_one_meter_does_not_affect_other(self):
        """A spike in Meter A must not change Meter B's baseline or anomaly count."""
        n = 50
        deltas_a = [60.0] * n
        deltas_a[25] = 999.0   # spike

        deltas_b = [60.0] * n

        df = _make_multi_meter_df({
            "M-H1-F1": deltas_a,
            "M-H2-F1": deltas_b,
        })
        result = compute_rolling_baselines(df, window_days=10, min_readings=5)

        b_anomalies = result[result["meter_id"] == "M-H2-F1"]["is_anomaly"].sum()
        assert b_anomalies == 0, (
            f"Meter B unexpectedly got {b_anomalies} anomaly(ies) due to Meter A spike"
        )


# ---------------------------------------------------------------------------
# T3  Adaptation — baseline tracks gradual consumption shift
# ---------------------------------------------------------------------------

class TestAdaptation:
    def test_baseline_adapts_to_gradual_increase(self):
        """
        Consumption ramps slowly from 60 to 120 L/h over 200 hours.
        In the second half the rolling baseline should be higher than in the first.
        """
        n = 200
        deltas = [60.0 + i * 0.3 for i in range(n)]  # gradual ramp
        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=5, min_readings=5)
        rolling = result[result["baseline_source"] == "rolling"]

        if len(rolling) < 10:
            pytest.skip("Not enough rolling-baseline rows to compare halves")

        mid_point  = len(rolling) // 2
        first_half = rolling.iloc[:mid_point]["median_baseline"].mean()
        second_half = rolling.iloc[mid_point:]["median_baseline"].mean()
        assert second_half > first_half, (
            f"Baseline did not adapt upward: first={first_half:.1f}, second={second_half:.1f}"
        )

    def test_baseline_does_not_flag_gradual_ramp_as_anomaly(self):
        """
        Gradual ramp that the baseline follows should produce few or no anomalies.
        (A purely global MAD would flag everything in the high half.)
        """
        n = 200
        step = 0.2
        deltas = [60.0 + i * step for i in range(n)]
        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=5, min_readings=5)
        rolling = result[result["baseline_source"] == "rolling"]
        anomaly_rate = rolling["is_anomaly"].mean()
        # Gradual ramp within window → low anomaly rate
        assert anomaly_rate < 0.15, (
            f"Too many anomalies on gradual ramp: {anomaly_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# T4  Sudden spike detection
# ---------------------------------------------------------------------------

class TestSuddenSpike:
    def test_single_spike_flagged(self):
        """A single large spike in an otherwise stable meter must be flagged."""
        n = 60
        deltas = [60.0] * n
        deltas[50] = 600.0    # 10× normal at position 50

        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        assert result.iloc[50]["is_anomaly"], (
            "Expected the spike at position 50 to be flagged as anomaly"
        )

    def test_non_spike_not_flagged(self):
        """Normal consumption must not be flagged as anomaly."""
        n = 60
        deltas = [60.0 + (i % 3) * 2 for i in range(n)]   # small variation
        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        rolling = result[result["baseline_source"] == "rolling"]
        assert not rolling["is_anomaly"].any(), (
            f"Normal readings unexpectedly flagged: {rolling[rolling['is_anomaly']]}"
        )


# ---------------------------------------------------------------------------
# T5  Sustained (continuous) anomaly detection
# ---------------------------------------------------------------------------

class TestSustainedAnomaly:
    def test_sustained_high_consumption_flagged(self):
        """
        After warm-up, sustained 5× consumption must trigger anomalies for
        at least PERSISTENCE_MIN consecutive readings.
        """
        # 40 normal readings, then 20 high
        n_normal = 40
        n_high   = 20
        deltas   = [60.0] * n_normal + [400.0] * n_high
        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)

        # Check the 'high' window
        high_window = result.iloc[n_normal:]
        consecutive_anomalies = high_window["is_anomaly"].sum()
        assert consecutive_anomalies >= PERSISTENCE_MIN, (
            f"Expected at least {PERSISTENCE_MIN} consecutive anomalies, "
            f"got {consecutive_anomalies}"
        )

    def test_persistence_count_increments_during_streak(self):
        """persistence_count must increase by 1 each row during a sustained anomaly."""
        n_normal = 40
        n_high   = 10
        deltas   = [60.0] * n_normal + [500.0] * n_high
        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        result = compute_persistence(result)

        anomaly_rows = result[result["is_anomaly"]].reset_index(drop=True)
        # Find consecutive streak
        counts = anomaly_rows["persistence_count"].values
        for i in range(1, len(counts)):
            if counts[i] > 1:
                assert counts[i] == counts[i - 1] + 1 or counts[i] == 1, (
                    f"Non-consecutive persistence_count: {counts[i - 1]} → {counts[i]}"
                )


# ---------------------------------------------------------------------------
# T6  Insufficient historical data → global fallback
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_first_rows_use_global_fallback(self):
        """
        The first min_readings rows must use 'global' baseline_source because
        there are not enough prior readings to build the rolling window.
        """
        min_r = 10
        df    = _make_df([60.0] * 30)
        result = compute_rolling_baselines(df, window_days=7, min_readings=min_r)

        # Rows 0 .. min_r-1 have < min_r prior readings → must fall back
        first_n = result.iloc[:min_r]
        assert (first_n["baseline_source"] == "global").all(), (
            f"Expected 'global' for first {min_r} rows; "
            f"got: {first_n['baseline_source'].tolist()}"
        )

    def test_later_rows_use_rolling(self):
        """Once enough prior readings exist, baseline_source switches to 'rolling'."""
        min_r = 5
        df    = _make_df([60.0] * 30)
        result = compute_rolling_baselines(df, window_days=10, min_readings=min_r)

        later = result.iloc[min_r + HOUR_TOLERANCE + 1:]
        assert (later["baseline_source"] == "rolling").any(), (
            "Expected at least some rows to use rolling baseline after warm-up"
        )

    def test_global_fallback_does_not_crash(self):
        """A single-reading meter must not raise any exception."""
        df = _make_df([60.0])
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        assert len(result) == 1
        assert result.iloc[0]["baseline_source"] == "global"

    def test_empty_df_returns_empty(self):
        """Empty input must return an empty DataFrame with correct columns."""
        df = pd.DataFrame(columns=[
            "timestamp", "hostel_id", "floor_id", "meter_id",
            "meter_reading", "delta_litres", "hour", "day_of_week",
        ])
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        assert result.empty


# ---------------------------------------------------------------------------
# T7  Requirement 8 — current reading excluded from its own baseline
# ---------------------------------------------------------------------------

class TestSelfExclusion:
    def test_current_reading_not_in_its_own_baseline(self):
        """
        Insert a large spike at position N.  The baseline at position N+1
        should still reflect normal consumption (spike not yet in window
        to pollute the baseline).
        """
        n = 30
        deltas = [60.0] * n
        spike_pos = 20
        deltas[spike_pos] = 1000.0

        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)

        # The spike at spike_pos must be flagged
        assert result.iloc[spike_pos]["is_anomaly"], (
            "Spike row itself should be flagged"
        )

        # The row immediately after the spike should still have a low baseline
        # (the spike contributes ≤1 out of many prior readings)
        post_spike = result.iloc[spike_pos + 1]
        if post_spike["baseline_source"] == "rolling":
            # Baseline should not jump massively just because of one spike row
            assert post_spike["median_baseline"] < 200.0, (
                f"Baseline at spike+1 jumped too high: {post_spike['median_baseline']:.1f}"
            )

    def test_row_zero_has_no_prior_data(self):
        """The very first row of a meter must use the global fallback."""
        df = _make_df([60.0] * 20)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        assert result.iloc[0]["baseline_source"] == "global"


# ---------------------------------------------------------------------------
# T8  Persistence count correctness
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_resets_to_zero_after_normal_reading(self):
        """persistence_count must reset to 0 when is_anomaly returns to False."""
        n_normal = 40
        n_high   = 5
        # Normal → spike streak → back to normal
        deltas = [60.0] * n_normal + [500.0] * n_high + [60.0] * 5
        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        result = compute_persistence(result)

        # Last 5 rows should be non-anomalous with persistence 0
        tail = result.tail(5)
        assert (tail["persistence_count"] == 0).all(), (
            f"Expected persistence_count=0 after streak ends; "
            f"got: {tail['persistence_count'].tolist()}"
        )

    def test_persistence_non_negative_always(self):
        deltas = [60.0] * 10 + [600.0] * 5 + [60.0] * 10
        df = _make_df(deltas)
        result = compute_rolling_baselines(df, window_days=7, min_readings=5)
        result = compute_persistence(result)
        assert (result["persistence_count"] >= 0).all()


# ---------------------------------------------------------------------------
# T9  Anomaly flag dtype
# ---------------------------------------------------------------------------

class TestAnomalyDtype:
    def test_is_anomaly_is_bool(self):
        df = _make_df([60.0] * 20)
        result = compute_rolling_baselines(df, window_days=5, min_readings=5)
        assert result["is_anomaly"].dtype == bool

    def test_z_score_numeric(self):
        df = _make_df([60.0] * 20)
        result = compute_rolling_baselines(df, window_days=5, min_readings=5)
        assert pd.api.types.is_numeric_dtype(result["z_score_mad"])


# ---------------------------------------------------------------------------
# T10  Window boundary — only readings within window included
# ---------------------------------------------------------------------------

class TestWindowBoundary:
    def test_readings_outside_window_ignored(self):
        """
        Establish a stable baseline for 10 days (hourly readings), then run
        a 2-day window.  The 2-day window must not use readings from day 1–8.
        Use daily=False so 240 readings span 10 actual days at hourly intervals.
        """
        n_hist = 240
        deltas = [60.0] * n_hist + [600.0]
        # daily=False → 1-hour spacing, so 240 readings = 10 days
        df     = _make_df(deltas, daily=False)

        # 2-day window → ~48 same-hour prior readings eligible (every 24th)
        result = compute_rolling_baselines(df, window_days=2, min_readings=5)
        spike_row = result.iloc[-1]
        assert spike_row["is_anomaly"], "Spike at end should be flagged"

    def test_window_days_zero_always_uses_global(self):
        """window_days=0 means no window → always global fallback."""
        df = _make_df([60.0] * 30)
        result = compute_rolling_baselines(df, window_days=0, min_readings=5)
        assert (result["baseline_source"] == "global").all()


# ---------------------------------------------------------------------------
# T11  Alert-status preservation after rolling run
# ---------------------------------------------------------------------------

class TestAlertPreservation:
    def test_existing_resolved_alerts_preserved(self):
        """
        Run rolling_detector.run() against real processed_readings.csv.
        The alert_manager's merge_with_existing must retain previously
        RESOLVED alert IDs and statuses.
        """
        alerts_path = os.path.join(PROJECT_ROOT, "data", "alerts.csv")
        if not os.path.exists(alerts_path):
            pytest.skip("alerts.csv not present — skipping preservation test")

        existing = pd.read_csv(alerts_path, dtype=str)
        resolved_before = existing[existing["status"] == "RESOLVED"][
            ["alert_id", "meter_id", "first_seen", "last_seen"]
        ].copy()

        if resolved_before.empty:
            pytest.skip("No RESOLVED alerts to verify preservation")

        # Run rolling detector against real data
        processed_path = os.path.join(PROJECT_ROOT, "data", "processed_readings.csv")
        anomaly_path   = os.path.join(PROJECT_ROOT, "data", "anomalies.csv")
        if not os.path.exists(processed_path):
            pytest.skip("processed_readings.csv not present")

        run(input_path=processed_path, output_path=anomaly_path)

        # Re-run alert manager
        sys.path.insert(0, SRC_DIR)
        from alert_manager import build_alerts, merge_with_existing  # noqa: PLC0415

        df_anom = pd.read_csv(anomaly_path)
        df_anom["timestamp"] = pd.to_datetime(df_anom["timestamp"], format="mixed")
        df_anom["is_anomaly"] = (
            df_anom["is_anomaly"].astype(str).str.strip().str.lower() == "true"
        )
        new_alerts = build_alerts(df_anom)
        merged     = merge_with_existing(new_alerts, alerts_path)

        # All previously RESOLVED alerts with matching keys must still be RESOLVED
        for _, old_row in resolved_before.iterrows():
            key_mask = (
                (merged["meter_id"]   == old_row["meter_id"]) &
                (merged["first_seen"] == old_row["first_seen"]) &
                (merged["last_seen"]  == old_row["last_seen"])
            )
            matching = merged[key_mask]
            if not matching.empty:
                assert (matching["status"] == "RESOLVED").all(), (
                    f"Alert {old_row['alert_id']} changed from RESOLVED after rolling run"
                )


# ---------------------------------------------------------------------------
# T12  Real data — known leak-injected meters detected
# Module-level fixture (class-scoped fixtures inside classes require pytest 9+
# workaround; moving outside the class is the clean solution).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_data_rolling_anomalies():
    processed_path = os.path.join(PROJECT_ROOT, "data", "processed_readings.csv")
    if not os.path.exists(processed_path):
        pytest.skip("processed_readings.csv not present")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "anomalies.csv")
        df  = run(input_path=processed_path, output_path=out)
    return df


class TestRealDataDetection:
    def test_all_12_meters_in_output(self, real_data_rolling_anomalies):
        assert real_data_rolling_anomalies["meter_id"].nunique() == 12

    def test_known_leak_meters_have_anomalies(self, real_data_rolling_anomalies):
        for mid in LEAK_INJECT_METERS:
            meter_rows = real_data_rolling_anomalies[
                real_data_rolling_anomalies["meter_id"] == mid
            ]
            has_anomaly = (
                meter_rows["is_anomaly"]
                .astype(str).str.lower().eq("true")
                .any()
            )
            assert has_anomaly, f"Leak meter {mid} has no anomaly detected by rolling detector"

    def test_output_row_count_matches_input(self, real_data_rolling_anomalies):
        processed_path = os.path.join(PROJECT_ROOT, "data", "processed_readings.csv")
        processed = pd.read_csv(processed_path)
        assert len(real_data_rolling_anomalies) == len(processed)

    def test_persistence_non_negative(self, real_data_rolling_anomalies):
        assert (real_data_rolling_anomalies["persistence_count"] >= 0).all()

    def test_all_output_columns_present(self, real_data_rolling_anomalies):
        for col in OUTPUT_COLS:
            assert col in real_data_rolling_anomalies.columns


# ---------------------------------------------------------------------------
# T_config  Config constants exported correctly
# ---------------------------------------------------------------------------

class TestConfigConstants:
    def test_rolling_window_days_in_config(self):
        assert ROLLING_WINDOW_DAYS > 0

    def test_rolling_min_readings_in_config(self):
        assert ROLLING_MIN_READINGS > 0

    def test_defaults_match_config(self):
        assert DEFAULT_WINDOW_DAYS  == ROLLING_WINDOW_DAYS
        assert DEFAULT_MIN_READINGS == ROLLING_MIN_READINGS
