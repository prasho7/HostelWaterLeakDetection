"""
test_live_simulator.py
Pytest tests for src/live_simulator.py  — Phase 1 live monitoring.

Run from the project root:
    pytest tests/test_live_simulator.py -v
"""

import os
import sys
import random
import tempfile

import pytest
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import METERS, METER_IDS
from live_simulator import (
    COLUMN_ORDER,
    DEFAULT_INTERVAL_MINS,
    SCENARIO_WEIGHTS,
    compute_delta,
    generate_tick,
    load_last_readings,
    load_last_timestamps,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_rng():
    return random.Random(0)


@pytest.fixture
def base_readings():
    """Minimal last-readings state: all meters at 1000.0."""
    return {m["meter_id"]: 1000.0 for m in METERS}


@pytest.fixture
def base_timestamps():
    """Fixed reference timestamp for all meters."""
    ref = datetime(2024, 1, 31, 12, 0, 0)
    return {m["meter_id"]: ref for m in METERS}


@pytest.fixture
def one_tick_df(base_readings, base_timestamps, seeded_rng):
    return generate_tick(base_readings, base_timestamps, rng=seeded_rng)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_columns(self, one_tick_df):
        """Output must have exactly the required columns in the right order."""
        assert list(one_tick_df.columns) == COLUMN_ORDER

    def test_row_count(self, one_tick_df):
        """One tick must produce exactly 12 rows — one per meter."""
        assert len(one_tick_df) == 12

    def test_all_meters_present(self, one_tick_df):
        """Every configured meter_id must appear in a single tick."""
        assert set(one_tick_df["meter_id"]) == set(METER_IDS)

    def test_all_12_meters(self, one_tick_df):
        """Exactly 12 unique meter IDs."""
        assert one_tick_df["meter_id"].nunique() == 12

    def test_hostel_ids(self, one_tick_df):
        """All three hostel IDs must be represented."""
        assert set(one_tick_df["hostel_id"]) == {"H1", "H2", "H3"}

    def test_meter_reading_dtype(self, one_tick_df):
        """meter_reading must be numeric."""
        assert pd.api.types.is_numeric_dtype(one_tick_df["meter_reading"])

    def test_timestamp_parseable(self, one_tick_df):
        """timestamp column must be parseable as ISO 8601."""
        try:
            pd.to_datetime(one_tick_df["timestamp"])
        except Exception as exc:
            pytest.fail(f"timestamp not valid ISO 8601: {exc}")


# ---------------------------------------------------------------------------
# Cumulative / monotonic tests
# ---------------------------------------------------------------------------

class TestCumulative:
    def test_reading_increases_over_base(self, one_tick_df, base_readings):
        """Every new reading must be strictly greater than the base reading."""
        for _, row in one_tick_df.iterrows():
            assert row["meter_reading"] > base_readings[row["meter_id"]], (
                f"Meter {row['meter_id']} reading did not increase"
            )

    def test_multi_tick_monotonic(self, base_readings, base_timestamps, seeded_rng):
        """Across multiple ticks, cumulative readings must never decrease."""
        last_r = dict(base_readings)
        last_t = dict(base_timestamps)

        for _ in range(5):
            df = generate_tick(last_r, last_t, rng=seeded_rng)
            for _, row in df.iterrows():
                mid = row["meter_id"]
                assert row["meter_reading"] >= last_r[mid], (
                    f"Meter {mid} reading decreased at tick"
                )
                last_r[mid] = row["meter_reading"]
                last_t[mid] = datetime.fromisoformat(row["timestamp"])

    def test_timestamp_advances(self, one_tick_df, base_timestamps):
        """Every tick's timestamp must be exactly interval_mins ahead of base."""
        for _, row in one_tick_df.iterrows():
            mid      = row["meter_id"]
            expected = base_timestamps[mid] + timedelta(minutes=DEFAULT_INTERVAL_MINS)
            actual   = datetime.fromisoformat(row["timestamp"])
            assert actual == expected, (
                f"Meter {mid}: expected {expected}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

class TestScenarios:
    """Verify each scenario produces the expected consumption range."""

    # Spike must exceed normal by at least SPIKE_EXTRA_MIN
    def test_spike_exceeds_normal(self):
        rng        = random.Random(1)
        normal_val = compute_delta("normal", hour=10, rng=rng)
        rng2       = random.Random(1)
        spike_val  = compute_delta("spike",  hour=10, rng=rng2)
        # spike shares the same base, so it must be larger
        assert spike_val > normal_val

    def test_continuous_leak_exceeds_normal(self):
        rng      = random.Random(2)
        normal   = compute_delta("normal",          hour=14, rng=rng)
        rng2     = random.Random(2)
        leaking  = compute_delta("continuous_leak", hour=14, rng=rng2)
        assert leaking > normal

    def test_abnormal_night_exceeds_normal_night(self):
        """During night hours, abnormal_night must exceed the normal scenario."""
        night_hour = 2
        rng1 = random.Random(3)
        normal_night    = compute_delta("normal",         hour=night_hour, rng=rng1)
        rng2 = random.Random(3)
        abnormal_night  = compute_delta("abnormal_night", hour=night_hour, rng=rng2)
        assert abnormal_night > normal_night

    def test_normal_night_less_than_day(self):
        """Normal night consumption must be lower than normal daytime."""
        rng = random.Random(4)
        day_values   = [compute_delta("normal", hour=10, rng=rng) for _ in range(50)]
        night_values = [compute_delta("normal", hour=2,  rng=rng) for _ in range(50)]
        assert sum(night_values) < sum(day_values)

    def test_scenario_override(self, base_readings, base_timestamps):
        """scenario_overrides must force the specified scenario for that meter."""
        # Force the first meter to spike; all others normal
        first_meter = METER_IDS[0]
        overrides   = {first_meter: "spike"}
        rng = random.Random(5)
        df  = generate_tick(
            base_readings, base_timestamps,
            scenario_overrides=overrides,
            rng=rng,
        )
        # Spike reading must be > 1000 + SPIKE_EXTRA_MIN (200)
        spiked = df.loc[df["meter_id"] == first_meter, "meter_reading"].iloc[0]
        assert spiked > 1200.0, (
            f"Spike reading {spiked} not large enough for meter {first_meter}"
        )

    def test_unknown_scenario_raises(self):
        rng = random.Random(99)
        with pytest.raises(ValueError, match="Unknown scenario"):
            compute_delta("invalid_scenario", hour=10, rng=rng)


# ---------------------------------------------------------------------------
# Append-only / historical-data integrity tests
# ---------------------------------------------------------------------------

class TestAppendOnly:
    def test_appends_do_not_modify_history(self):
        """
        Existing rows in the CSV must be byte-identical after a simulation run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "readings.csv")

            # Seed the file with 10 known rows
            original_df = pd.DataFrame([
                {
                    "timestamp":     f"2024-01-31T{h:02d}:00:00",
                    "hostel_id":     m["hostel_id"],
                    "floor_id":      m["floor_id"],
                    "meter_id":      m["meter_id"],
                    "meter_reading": float(h * 60),
                }
                for m in METERS[:1]
                for h in range(10)
            ])
            original_df.to_csv(csv_path, index=False)

            snapshot_before = open(csv_path, "rb").read()

            # Run simulation — appends new rows
            run_simulation(ticks=1, csv_path=csv_path, rng=random.Random(7))

            full = open(csv_path, "rb").read()
            # The original content must be a prefix of the new file
            assert full.startswith(snapshot_before), (
                "Historical rows were modified or deleted by run_simulation()"
            )

    def test_new_rows_appended(self):
        """run_simulation must add exactly ticks × 12 rows to the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "readings.csv")

            new_df = run_simulation(ticks=3, csv_path=csv_path, rng=random.Random(8))
            assert len(new_df) == 36  # 3 ticks × 12 meters

            on_disk = pd.read_csv(csv_path)
            assert len(on_disk) == 36

    def test_second_run_appends_to_first(self):
        """Two consecutive run_simulation calls must accumulate rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "readings.csv")

            run_simulation(ticks=2, csv_path=csv_path, rng=random.Random(9))
            run_simulation(ticks=2, csv_path=csv_path, rng=random.Random(10))

            on_disk = pd.read_csv(csv_path)
            assert len(on_disk) == 48  # (2 + 2) ticks × 12 meters

    def test_readings_monotonic_after_append(self):
        """After two run_simulation calls, cumulative readings must be monotonic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "readings.csv")

            run_simulation(ticks=3, csv_path=csv_path, rng=random.Random(11))
            run_simulation(ticks=3, csv_path=csv_path, rng=random.Random(12))

            df = pd.read_csv(csv_path)
            for meter_id, group in df.groupby("meter_id"):
                vals = group["meter_reading"].values
                assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)), (
                    f"Meter {meter_id} readings not monotonic after append"
                )


# ---------------------------------------------------------------------------
# State loader tests
# ---------------------------------------------------------------------------

class TestStateLoaders:
    def test_load_last_readings_empty_file(self):
        """Missing CSV returns 0.0 for every meter."""
        last = load_last_readings("/nonexistent/path.csv")
        assert set(last.keys()) == set(METER_IDS)
        assert all(v == 0.0 for v in last.values())

    def test_load_last_readings_from_csv(self):
        """Returns the final meter_reading per meter from an existing CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "readings.csv")
            df = pd.DataFrame([
                {"meter_id": m["meter_id"], "meter_reading": float(i * 100)}
                for i, m in enumerate(METERS)
            ])
            df.to_csv(csv_path, index=False)

            last = load_last_readings(csv_path)
            for i, m in enumerate(METERS):
                assert last[m["meter_id"]] == float(i * 100)

    def test_load_last_timestamps_missing_file(self):
        """Missing CSV returns a recent datetime for every meter."""
        last = load_last_timestamps("/nonexistent/path.csv")
        assert set(last.keys()) == set(METER_IDS)
        for ts in last.values():
            assert isinstance(ts, datetime)

    def test_dry_run_does_not_write(self):
        """dry_run=True must not create or modify the CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "readings.csv")
            run_simulation(ticks=2, csv_path=csv_path, dry_run=True,
                           rng=random.Random(13))
            assert not os.path.exists(csv_path), (
                "dry_run=True must not write any file"
            )


# ---------------------------------------------------------------------------
# Interval configuration tests
# ---------------------------------------------------------------------------

class TestIntervalConfig:
    def test_custom_interval_advances_timestamp(self, base_readings, base_timestamps):
        """Tick timestamp gap must equal the requested interval_mins."""
        interval = 30
        rng = random.Random(20)
        df  = generate_tick(base_readings, base_timestamps,
                            interval_mins=interval, rng=rng)
        for _, row in df.iterrows():
            mid      = row["meter_id"]
            expected = base_timestamps[mid] + timedelta(minutes=interval)
            actual   = datetime.fromisoformat(row["timestamp"])
            assert actual == expected

    def test_15_minute_interval(self, base_readings, base_timestamps):
        interval = 15
        rng = random.Random(21)
        df  = generate_tick(base_readings, base_timestamps,
                            interval_mins=interval, rng=rng)
        assert len(df) == 12  # still 12 meters
