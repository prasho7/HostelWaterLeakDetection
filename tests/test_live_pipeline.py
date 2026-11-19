"""
test_live_pipeline.py
Pytest suite for src/live_pipeline.py — Phase 2 incremental processing.

All tests use isolated temporary directories so they never touch the
production data files or the real watermark.

Run from the project root:
    pytest tests/test_live_pipeline.py -v
"""

import json
import os
import sys
import random
import tempfile
from datetime import datetime, timedelta

import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import METERS, METER_IDS
from live_pipeline import (
    PROCESSED_COLS,
    READINGS_COLS,
    append_to_processed,
    enrich_new_rows,
    find_new_rows,
    load_last_processed_readings,
    load_watermark,
    run_pipeline,
    save_watermark,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal fixture CSVs
# ---------------------------------------------------------------------------

def _make_readings_csv(path: str, n_ticks: int = 5, start_reading: float = 0.0) -> pd.DataFrame:
    """
    Write a minimal readings.csv with all 12 meters for n_ticks ticks.
    Returns the DataFrame written.
    """
    base_ts = datetime(2024, 1, 1, 0, 0, 0)
    rows = []
    for tick in range(n_ticks):
        ts = base_ts + timedelta(hours=tick)
        for m in METERS:
            rows.append({
                "timestamp":     ts.isoformat(),
                "hostel_id":     m["hostel_id"],
                "floor_id":      m["floor_id"],
                "meter_id":      m["meter_id"],
                "meter_reading": round(start_reading + (tick + 1) * 60.0, 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _make_processed_csv(path: str, n_ticks: int = 5, start_reading: float = 0.0):
    """
    Write a minimal processed_readings.csv (simulates already-processed history).
    """
    base_ts = datetime(2024, 1, 1, 0, 0, 0)
    rows = []
    for tick in range(n_ticks):
        ts = base_ts + timedelta(hours=tick)
        for m in METERS:
            reading = start_reading + (tick + 1) * 60.0
            rows.append({
                "timestamp":     ts.isoformat(),
                "hostel_id":     m["hostel_id"],
                "floor_id":      m["floor_id"],
                "meter_id":      m["meter_id"],
                "meter_reading": round(reading, 4),
                "delta_litres":  60.0,
                "hour":          ts.hour,
                "day_of_week":   ts.weekday(),
            })
    df = pd.DataFrame(rows, columns=PROCESSED_COLS)
    df.to_csv(path, index=False)
    return df


def _watermark_at_tick(n_ticks: int) -> dict:
    """Return watermark dict reflecting all 12 meters processed up to tick n_ticks."""
    base_ts = datetime(2024, 1, 1, 0, 0, 0)
    last_ts = (base_ts + timedelta(hours=n_ticks - 1)).isoformat()
    return {m["meter_id"]: last_ts for m in METERS}


# ---------------------------------------------------------------------------
# Watermark tests
# ---------------------------------------------------------------------------

class TestWatermark:
    def test_missing_file_returns_epoch(self):
        marks = load_watermark("/nonexistent/path.json")
        assert set(marks.keys()) == set(METER_IDS)
        assert all(marks[mid] == "1970-01-01T00:00:00" for mid in METER_IDS)

    def test_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ts   = "2024-01-15T10:00:00"
            data = {mid: ts for mid in METER_IDS}
            save_watermark(data, path)
            loaded = load_watermark(path)
            assert loaded == data
        finally:
            os.unlink(path)

    def test_corrupted_file_returns_epoch(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("NOT VALID JSON {{{")
            path = f.name
        try:
            marks = load_watermark(path)
            assert all(marks[mid] == "1970-01-01T00:00:00" for mid in METER_IDS)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# find_new_rows tests
# ---------------------------------------------------------------------------

class TestFindNewRows:
    def test_all_rows_new_when_epoch_watermark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=3)
            epoch_wm = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, epoch_wm)
            # 3 ticks x 12 meters
            assert len(new) == 36

    def test_no_new_rows_when_watermark_at_last_tick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=3)
            wm = _watermark_at_tick(3)
            new = find_new_rows(csv, wm)
            assert len(new) == 0

    def test_only_ticks_after_watermark_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=5)
            # Mark first 3 ticks as already processed
            wm = _watermark_at_tick(3)
            new = find_new_rows(csv, wm)
            # Ticks 4 and 5 → 2 ticks x 12 meters
            assert len(new) == 24

    def test_missing_readings_csv_returns_empty(self):
        wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
        new = find_new_rows("/nonexistent/readings.csv", wm)
        assert new.empty

    def test_all_12_meters_in_new_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=2)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            assert set(new["meter_id"]) == set(METER_IDS)

    def test_rows_sorted_by_meter_then_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=3)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            # Verify meter_id grouping is sorted
            meter_ids = new["meter_id"].tolist()
            assert meter_ids == sorted(meter_ids, key=lambda x: (x, ))


# ---------------------------------------------------------------------------
# enrich_new_rows tests
# ---------------------------------------------------------------------------

class TestEnrichNewRows:
    def test_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=2)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            anchors = {mid: 0.0 for mid in METER_IDS}
            enriched = enrich_new_rows(new, anchors)
            assert list(enriched.columns) == PROCESSED_COLS

    def test_delta_litres_non_negative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=4)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            anchors = {mid: 0.0 for mid in METER_IDS}
            enriched = enrich_new_rows(new, anchors)
            assert (enriched["delta_litres"] >= 0).all()

    def test_first_delta_uses_anchor(self):
        """First new row's delta must equal (reading - anchor)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=1)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            # anchor at 100 for all meters; first tick reading is 60.0
            anchors  = {mid: 100.0 for mid in METER_IDS}
            enriched = enrich_new_rows(new, anchors)
            # reading=60 - anchor=100 → clipped to 0
            assert (enriched["delta_litres"] == 0.0).all()

    def test_hour_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=24)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            anchors  = {mid: 0.0 for mid in METER_IDS}
            enriched = enrich_new_rows(new, anchors)
            assert enriched["hour"].between(0, 23).all()

    def test_day_of_week_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=7)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            anchors  = {mid: 0.0 for mid in METER_IDS}
            enriched = enrich_new_rows(new, anchors)
            assert enriched["day_of_week"].between(0, 6).all()

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=READINGS_COLS)
        result = enrich_new_rows(empty, {})
        assert result.empty
        assert list(result.columns) == PROCESSED_COLS


# ---------------------------------------------------------------------------
# load_last_processed_readings tests
# ---------------------------------------------------------------------------

class TestLoadLastProcessedReadings:
    def test_missing_file_returns_zeros(self):
        anchors = load_last_processed_readings("/nonexistent/processed.csv")
        assert set(anchors.keys()) == set(METER_IDS)
        assert all(v == 0.0 for v in anchors.values())

    def test_loads_last_reading_per_meter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = os.path.join(tmpdir, "processed_readings.csv")
            _make_processed_csv(proc, n_ticks=3)
            anchors = load_last_processed_readings(proc)
            # tick 3: reading = 3 * 60 = 180
            for mid in METER_IDS:
                assert anchors[mid] == pytest.approx(180.0, abs=0.01)


# ---------------------------------------------------------------------------
# append_to_processed tests
# ---------------------------------------------------------------------------

class TestAppendToProcessed:
    def test_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = os.path.join(tmpdir, "processed_readings.csv")
            df   = pd.DataFrame([{
                "timestamp": "2024-01-01T00:00:00", "hostel_id": "H1",
                "floor_id": "H1-F1", "meter_id": "M-H1-F1",
                "meter_reading": 60.0, "delta_litres": 60.0,
                "hour": 0, "day_of_week": 0,
            }], columns=PROCESSED_COLS)
            append_to_processed(df, proc)
            on_disk = pd.read_csv(proc)
            assert list(on_disk.columns) == PROCESSED_COLS
            assert len(on_disk) == 1

    def test_appends_without_duplicate_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = os.path.join(tmpdir, "processed_readings.csv")
            _make_processed_csv(proc, n_ticks=2)
            original_len = len(pd.read_csv(proc))

            df = pd.DataFrame([{
                "timestamp": "2024-01-01T10:00:00", "hostel_id": "H1",
                "floor_id": "H1-F1", "meter_id": "M-H1-F1",
                "meter_reading": 1000.0, "delta_litres": 60.0,
                "hour": 10, "day_of_week": 0,
            }], columns=PROCESSED_COLS)
            append_to_processed(df, proc)

            on_disk = pd.read_csv(proc)
            assert len(on_disk) == original_len + 1
            # Ensure no row has 'timestamp' as a value (duplicate header row)
            assert not (on_disk["timestamp"] == "timestamp").any()


# ---------------------------------------------------------------------------
# run_pipeline integration tests  (isolated tmpdir, no detection engine)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """
    These tests run run_pipeline() with isolated CSV paths.
    Detection + alert sub-steps are skipped (they need the full historical data
    and are covered separately by test_leak_detector.py & test_alert_manager.py).
    We therefore only run the pipeline up to step 4 (append + watermark update)
    by patching the imports inline.
    """

    def _run_isolated(self, tmpdir, n_hist, n_new):
        """
        Helper: set up a tmpdir with n_hist processed ticks already done,
        then append n_new new ticks to readings.csv and run the pipeline.
        Returns (summary, processed_df, watermark).
        Stubs out the detection + alert sub-steps via sys.modules patching.
        """
        readings_path  = os.path.join(tmpdir, "readings.csv")
        processed_path = os.path.join(tmpdir, "processed_readings.csv")
        watermark_path = os.path.join(tmpdir, ".watermark.json")

        total_ticks = n_hist + n_new
        _make_readings_csv(readings_path, n_ticks=total_ticks)

        if n_hist > 0:
            _make_processed_csv(processed_path, n_ticks=n_hist)
            wm = _watermark_at_tick(n_hist)
            save_watermark(wm, watermark_path)

        import live_pipeline as lp
        import unittest.mock as mock

        # Stub lazy-imported sub-modules so detection engine doesn't run
        fake_ld = mock.MagicMock()
        fake_ld.run = mock.MagicMock()
        fake_am = mock.MagicMock()
        fake_am.main = mock.MagicMock()

        with mock.patch.dict(sys.modules, {
            "leak_detector": fake_ld,
            "alert_manager": fake_am,
        }):
            # Force re-evaluation of lazy imports inside run_pipeline
            with mock.patch.object(lp, "run_pipeline", wraps=lp.run_pipeline):
                summary = lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                    dry_run=False,
                )

        processed_df = pd.read_csv(processed_path) if os.path.exists(processed_path) else pd.DataFrame()
        wm_out       = load_watermark(watermark_path)
        return summary, processed_df, wm_out

    def test_no_new_rows_skips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readings_path  = os.path.join(tmpdir, "readings.csv")
            processed_path = os.path.join(tmpdir, "processed_readings.csv")
            watermark_path = os.path.join(tmpdir, ".watermark.json")

            _make_readings_csv(readings_path, n_ticks=3)
            _make_processed_csv(processed_path, n_ticks=3)
            wm = _watermark_at_tick(3)
            save_watermark(wm, watermark_path)

            import live_pipeline as lp
            import unittest.mock as mock
            with mock.patch.dict(sys.modules, {
                "leak_detector": mock.MagicMock(),
                "alert_manager": mock.MagicMock(),
            }):
                summary = lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                )

            assert summary["skipped"] is True
            assert summary["new_rows_found"] == 0

    def test_new_rows_found_and_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary, processed_df, wm_out = self._run_isolated(tmpdir, n_hist=3, n_new=2)

        assert summary["new_rows_found"] == 24  # 2 ticks x 12 meters
        assert summary["new_rows_written"] == 24
        assert summary["skipped"] is False

    def test_processed_csv_grows_by_new_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readings_path  = os.path.join(tmpdir, "readings.csv")
            processed_path = os.path.join(tmpdir, "processed_readings.csv")
            watermark_path = os.path.join(tmpdir, ".watermark.json")

            _make_readings_csv(readings_path, n_ticks=5)
            _make_processed_csv(processed_path, n_ticks=3)
            before = len(pd.read_csv(processed_path))
            wm = _watermark_at_tick(3)
            save_watermark(wm, watermark_path)

            import live_pipeline as lp
            import unittest.mock as mock
            with mock.patch.dict(sys.modules, {
                "leak_detector": mock.MagicMock(),
                "alert_manager": mock.MagicMock(),
            }):
                lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                )

            after = len(pd.read_csv(processed_path))
            assert after == before + 24  # 2 ticks x 12 meters added

    def test_second_run_is_idempotent(self):
        """Running the pipeline twice with no new data must not change the CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readings_path  = os.path.join(tmpdir, "readings.csv")
            processed_path = os.path.join(tmpdir, "processed_readings.csv")
            watermark_path = os.path.join(tmpdir, ".watermark.json")

            _make_readings_csv(readings_path, n_ticks=5)
            _make_processed_csv(processed_path, n_ticks=3)
            wm = _watermark_at_tick(3)
            save_watermark(wm, watermark_path)

            import live_pipeline as lp
            import unittest.mock as mock

            # First run: processes ticks 4 and 5
            with mock.patch.dict(sys.modules, {
                "leak_detector": mock.MagicMock(),
                "alert_manager": mock.MagicMock(),
            }):
                s1 = lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                )

            rows_after_first = len(pd.read_csv(processed_path))

            # Second run: nothing new
            with mock.patch.dict(sys.modules, {
                "leak_detector": mock.MagicMock(),
                "alert_manager": mock.MagicMock(),
            }):
                s2 = lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                )

            rows_after_second = len(pd.read_csv(processed_path))

            assert s1["new_rows_written"] == 24
            assert s2["skipped"] is True
            assert rows_after_second == rows_after_first

    def test_watermark_advances_after_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readings_path  = os.path.join(tmpdir, "readings.csv")
            processed_path = os.path.join(tmpdir, "processed_readings.csv")
            watermark_path = os.path.join(tmpdir, ".watermark.json")

            _make_readings_csv(readings_path, n_ticks=4)
            _make_processed_csv(processed_path, n_ticks=2)
            wm_before = _watermark_at_tick(2)
            save_watermark(wm_before, watermark_path)

            import live_pipeline as lp
            import unittest.mock as mock
            with mock.patch.dict(sys.modules, {
                "leak_detector": mock.MagicMock(),
                "alert_manager": mock.MagicMock(),
            }):
                lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                )

            wm_after = load_watermark(watermark_path)
            # Tick 4 = 2024-01-01T03:00:00
            expected_ts = datetime(2024, 1, 1, 3, 0, 0).isoformat()
            for mid in METER_IDS:
                assert wm_after[mid] == expected_ts

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readings_path  = os.path.join(tmpdir, "readings.csv")
            processed_path = os.path.join(tmpdir, "processed_readings.csv")
            watermark_path = os.path.join(tmpdir, ".watermark.json")

            _make_readings_csv(readings_path, n_ticks=3)
            epoch_wm = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            save_watermark(epoch_wm, watermark_path)

            import live_pipeline as lp
            lp.run_pipeline(
                readings_path=readings_path,
                processed_path=processed_path,
                watermark_path=watermark_path,
                dry_run=True,
            )

            # Neither processed CSV nor updated watermark should be written
            assert not os.path.exists(processed_path)
            wm_after = load_watermark(watermark_path)
            assert wm_after == epoch_wm

    def test_historical_rows_preserved(self):
        """Existing processed rows must be byte-identical after a pipeline run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readings_path  = os.path.join(tmpdir, "readings.csv")
            processed_path = os.path.join(tmpdir, "processed_readings.csv")
            watermark_path = os.path.join(tmpdir, ".watermark.json")

            _make_readings_csv(readings_path, n_ticks=5)
            _make_processed_csv(processed_path, n_ticks=3)
            snapshot = open(processed_path, "rb").read()
            wm = _watermark_at_tick(3)
            save_watermark(wm, watermark_path)

            import live_pipeline as lp
            import unittest.mock as mock
            with mock.patch.dict(sys.modules, {
                "leak_detector": mock.MagicMock(),
                "alert_manager": mock.MagicMock(),
            }):
                lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                )

            full = open(processed_path, "rb").read()
            assert full.startswith(snapshot), (
                "Historical processed rows were modified by run_pipeline()"
            )

    def test_delta_anchor_correct_across_batches(self):
        """
        Delta for the first new row must equal (new_reading - last_processed_reading).
        Fixture: 3 historical ticks (readings 60, 120, 180) already processed.
        New tick 4 reading = 240; anchor = 180 → expected delta = 60.
        Tick indices are 0-based: tick 0 → hour 0, tick 3 → hour 3.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            readings_path  = os.path.join(tmpdir, "readings.csv")
            processed_path = os.path.join(tmpdir, "processed_readings.csv")
            watermark_path = os.path.join(tmpdir, ".watermark.json")

            _make_readings_csv(readings_path, n_ticks=5)
            _make_processed_csv(processed_path, n_ticks=3)
            wm = _watermark_at_tick(3)
            save_watermark(wm, watermark_path)

            import live_pipeline as lp
            import unittest.mock as mock

            fake_ld = mock.MagicMock()
            fake_ld.run = mock.MagicMock()
            fake_am = mock.MagicMock()
            fake_am.main = mock.MagicMock()

            with mock.patch.dict(sys.modules, {
                "leak_detector": fake_ld,
                "alert_manager": fake_am,
            }):
                lp.run_pipeline(
                    readings_path=readings_path,
                    processed_path=processed_path,
                    watermark_path=watermark_path,
                )

            # Tick 4 (0-based) → hour 3 (base + 3h), reading = 4*60 = 240
            # anchor = 3*60 = 180 → delta = 60.0
            proc = pd.read_csv(processed_path)
            # Normalise timestamps to pd.Timestamp for format-agnostic comparison
            proc["timestamp"] = pd.to_datetime(proc["timestamp"], format="mixed")
            first_new_ts = pd.Timestamp("2024-01-01 03:00:00")
            first_new = proc[
                (proc["timestamp"] == first_new_ts) &
                (proc["meter_id"] == "M-H1-F1")
            ]
            assert len(first_new) == 1, (
                f"Expected 1 row for M-H1-F1 at {first_new_ts}; "
                f"available timestamps: {sorted(proc[proc['meter_id']=='M-H1-F1']['timestamp'].tolist())}"
            )
            assert first_new["delta_litres"].iloc[0] == pytest.approx(60.0, abs=0.01)


# ---------------------------------------------------------------------------
# Schema compatibility — enriched rows match data_processor.process() output
# ---------------------------------------------------------------------------

class TestSchemaCompatibility:
    def test_enriched_columns_match_processor_output(self):
        """Enriched new rows must have the same column set as data_processor.process()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv = os.path.join(tmpdir, "readings.csv")
            _make_readings_csv(csv, n_ticks=2)
            wm  = {mid: "1970-01-01T00:00:00" for mid in METER_IDS}
            new = find_new_rows(csv, wm)
            enriched = enrich_new_rows(new, {mid: 0.0 for mid in METER_IDS})
            assert list(enriched.columns) == PROCESSED_COLS

    def test_no_header_rows_in_appended_csv(self):
        """After multiple appends, no row should have 'timestamp' as a value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = os.path.join(tmpdir, "processed.csv")
            _make_processed_csv(proc, n_ticks=2)

            df_new = pd.DataFrame([{
                "timestamp": "2024-01-01T10:00:00", "hostel_id": "H1",
                "floor_id": "H1-F1", "meter_id": "M-H1-F1",
                "meter_reading": 999.0, "delta_litres": 60.0,
                "hour": 10, "day_of_week": 0,
            }], columns=PROCESSED_COLS)
            append_to_processed(df_new, proc)
            append_to_processed(df_new, proc)

            on_disk = pd.read_csv(proc)
            assert not (on_disk["timestamp"] == "timestamp").any()
