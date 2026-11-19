"""
test_annotation_store.py — Pytest suite for src/annotation_store.py (Phase 5)

Requirements tested
-------------------
R1  Add annotation to an alert.
R2  Add multiple annotations to the same alert.
R3  Verify notes remain attached to the correct alert (isolation).
R4  Resolve and reopen an alert; verify annotations remain.
R5  Verify persistence after save + reload.
R6  Validate empty notes and handle missing alert IDs safely.
R7  Persist annotations across application restarts (save + fresh load).

Additional tests
----------------
- Schema: required columns, dtypes.
- Ordering: annotations sorted by created_at ascending.
- Immutability: add_annotation does not mutate its input.
- Dashboard import: annotation_store importable alongside dashboard helpers.

Run from the project root:
    pytest tests/test_annotation_store.py -v
"""

import os
import sys
import tempfile
from datetime import datetime, timezone

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

import annotation_store as ann  # noqa: E402
from annotation_store import (  # noqa: E402
    ANNOTATION_COLUMNS,
    DEFAULT_AUTHOR,
    add_annotation,
    get_annotations_for_alert,
    load_annotations,
    save_annotations,
)
import alert_store  # noqa: E402  (needed for R4 — resolve/reopen then verify annotations)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ALERT_A = "alert-aaa-001"
ALERT_B = "alert-bbb-002"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def _make_csv(tmpdir: str, rows: list[dict]) -> str:
    path = os.path.join(tmpdir, "annotations.csv")
    df   = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS) if rows else _empty()
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_empty_df_has_all_columns(self):
        df = _empty()
        assert list(df.columns) == ANNOTATION_COLUMNS

    def test_load_missing_file_returns_empty(self):
        df = load_annotations("/nonexistent/path/annotations.csv")
        assert df.empty
        assert list(df.columns) == ANNOTATION_COLUMNS

    def test_add_returns_all_columns(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "test note")
        assert list(result.columns) == ANNOTATION_COLUMNS

    def test_annotation_id_is_non_empty_string(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "note")
        assert str(result.iloc[0]["annotation_id"]).strip() != ""

    def test_created_at_is_iso_string(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "note")
        ts_str = result.iloc[0]["created_at"]
        # Must parse as ISO datetime
        parsed = pd.to_datetime(ts_str)
        assert not pd.isna(parsed)

    def test_default_author(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "note")
        assert result.iloc[0]["author"] == DEFAULT_AUTHOR


# ---------------------------------------------------------------------------
# R1  Add annotation to an alert
# ---------------------------------------------------------------------------

class TestAddSingle:
    def test_adds_one_row(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "Checked valve on floor 1.")
        assert len(result) == 1

    def test_note_stored_correctly(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "Checked valve on floor 1.")
        assert result.iloc[0]["note"] == "Checked valve on floor 1."

    def test_alert_id_stored_correctly(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "some note")
        assert result.iloc[0]["alert_id"] == ALERT_A

    def test_custom_author_stored(self):
        df     = _empty()
        result = add_annotation(df, ALERT_A, "note", author="Engineer A")
        assert result.iloc[0]["author"] == "Engineer A"

    def test_custom_created_at_respected(self):
        ts = "2024-05-10T14:30:00"
        df     = _empty()
        result = add_annotation(df, ALERT_A, "note", created_at=ts)
        assert result.iloc[0]["created_at"] == ts


# ---------------------------------------------------------------------------
# R2  Add multiple annotations to the same alert
# ---------------------------------------------------------------------------

class TestAddMultiple:
    def test_two_notes_same_alert(self):
        df     = _empty()
        df     = add_annotation(df, ALERT_A, "First note")
        df     = add_annotation(df, ALERT_A, "Second note")
        assert len(df) == 2

    def test_three_notes_same_alert(self):
        df     = _empty()
        for i in range(3):
            df = add_annotation(df, ALERT_A, f"Note {i}")
        assert len(df) == 3
        assert (df["alert_id"] == ALERT_A).all()

    def test_all_notes_have_unique_annotation_ids(self):
        df     = _empty()
        for i in range(5):
            df = add_annotation(df, ALERT_A, f"Note {i}")
        assert df["annotation_id"].nunique() == 5


# ---------------------------------------------------------------------------
# R3  Notes remain attached to the correct alert (isolation)
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_note_not_returned_for_different_alert(self):
        df = _empty()
        df = add_annotation(df, ALERT_A, "Note for A")
        df = add_annotation(df, ALERT_B, "Note for B")

        notes_a = get_annotations_for_alert(df, ALERT_A)
        notes_b = get_annotations_for_alert(df, ALERT_B)

        assert len(notes_a) == 1
        assert notes_a.iloc[0]["note"] == "Note for A"
        assert len(notes_b) == 1
        assert notes_b.iloc[0]["note"] == "Note for B"

    def test_query_nonexistent_alert_returns_empty(self):
        df = _empty()
        df = add_annotation(df, ALERT_A, "Note")
        result = get_annotations_for_alert(df, "unknown-alert-id")
        assert result.empty

    def test_no_cross_contamination_multiple_notes(self):
        df = _empty()
        for i in range(3):
            df = add_annotation(df, ALERT_A, f"A note {i}")
        for i in range(2):
            df = add_annotation(df, ALERT_B, f"B note {i}")

        notes_a = get_annotations_for_alert(df, ALERT_A)
        notes_b = get_annotations_for_alert(df, ALERT_B)

        assert len(notes_a) == 3
        assert len(notes_b) == 2
        assert (notes_a["alert_id"] == ALERT_A).all()
        assert (notes_b["alert_id"] == ALERT_B).all()

    def test_annotations_sorted_by_created_at(self):
        df = _empty()
        df = add_annotation(df, ALERT_A, "oldest",  created_at="2024-01-01T10:00:00")
        df = add_annotation(df, ALERT_A, "newest",  created_at="2024-01-03T10:00:00")
        df = add_annotation(df, ALERT_A, "middle",  created_at="2024-01-02T10:00:00")

        notes = get_annotations_for_alert(df, ALERT_A)
        assert notes.iloc[0]["note"] == "oldest"
        assert notes.iloc[1]["note"] == "middle"
        assert notes.iloc[2]["note"] == "newest"


# ---------------------------------------------------------------------------
# R4  Resolve / reopen an alert — annotations must survive
# ---------------------------------------------------------------------------

SAMPLE_ALERTS_CSV = """\
alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
alert-001,H1,H1-F2,M-H1-F2,2024-01-01T03:00:00,2024-01-02T08:00:00,HIGH,34.98,30,OPEN,
"""


class TestAnnotationsSurviveResolutionCycle:
    def test_annotations_survive_resolve(self):
        """
        Adding a note to OPEN alert, then resolving the alert in alerts.csv,
        must not affect the annotations store.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            alerts_path = os.path.join(tmpdir, "alerts.csv")
            ann_path    = os.path.join(tmpdir, "annotations.csv")

            # Write alerts
            with open(alerts_path, "w") as f:
                f.write(SAMPLE_ALERTS_CSV)

            # Add annotation
            df_ann = load_annotations(ann_path)
            df_ann = add_annotation(df_ann, "alert-001", "Valve inspected.")
            save_annotations(df_ann, ann_path)

            # Resolve alert in alerts.csv (independent of annotations)
            df_alerts = alert_store.load_alerts(alerts_path)
            df_alerts = alert_store.resolve_alert(df_alerts, "alert-001")
            alert_store.save_alerts(df_alerts, alerts_path)

            # Verify alert is RESOLVED
            reloaded_alerts = alert_store.load_alerts(alerts_path)
            assert reloaded_alerts.iloc[0]["status"] == "RESOLVED"

            # Verify annotation still present and unchanged
            reloaded_ann = load_annotations(ann_path)
            assert len(reloaded_ann) == 1
            assert reloaded_ann.iloc[0]["note"] == "Valve inspected."
            assert reloaded_ann.iloc[0]["alert_id"] == "alert-001"

    def test_annotations_survive_reopen(self):
        """
        Annotations must still be present after the alert is reopened.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            alerts_path = os.path.join(tmpdir, "alerts.csv")
            ann_path    = os.path.join(tmpdir, "annotations.csv")

            with open(alerts_path, "w") as f:
                f.write(SAMPLE_ALERTS_CSV)

            # Add annotation then resolve
            df_ann = load_annotations(ann_path)
            df_ann = add_annotation(df_ann, "alert-001", "Checked pipes.")
            save_annotations(df_ann, ann_path)

            df_al = alert_store.load_alerts(alerts_path)
            df_al = alert_store.resolve_alert(df_al, "alert-001")
            alert_store.save_alerts(df_al, alerts_path)

            # Reopen
            df_al = alert_store.load_alerts(alerts_path)
            df_al = alert_store.reopen_alert(df_al, "alert-001")
            alert_store.save_alerts(df_al, alerts_path)

            reloaded_alerts = alert_store.load_alerts(alerts_path)
            assert reloaded_alerts.iloc[0]["status"] == "OPEN"

            reloaded_ann = load_annotations(ann_path)
            assert len(reloaded_ann) == 1
            assert reloaded_ann.iloc[0]["note"] == "Checked pipes."

    def test_multiple_annotations_survive_full_cycle(self):
        """Add 3 notes, resolve, reopen, re-resolve — all 3 notes persist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alerts_path = os.path.join(tmpdir, "alerts.csv")
            ann_path    = os.path.join(tmpdir, "annotations.csv")

            with open(alerts_path, "w") as f:
                f.write(SAMPLE_ALERTS_CSV)

            df_ann = load_annotations(ann_path)
            for i in range(3):
                df_ann = add_annotation(df_ann, "alert-001", f"Note {i}")
            save_annotations(df_ann, ann_path)

            # Resolve → reopen → resolve cycle
            df_al = alert_store.load_alerts(alerts_path)
            df_al = alert_store.resolve_alert(df_al, "alert-001")
            alert_store.save_alerts(df_al, alerts_path)

            df_al = alert_store.load_alerts(alerts_path)
            df_al = alert_store.reopen_alert(df_al, "alert-001")
            alert_store.save_alerts(df_al, alerts_path)

            df_al = alert_store.load_alerts(alerts_path)
            df_al = alert_store.resolve_alert(df_al, "alert-001")
            alert_store.save_alerts(df_al, alerts_path)

            # All 3 notes must still be there
            reloaded = load_annotations(ann_path)
            assert len(reloaded) == 3
            notes = set(reloaded["note"].tolist())
            assert notes == {"Note 0", "Note 1", "Note 2"}


# ---------------------------------------------------------------------------
# R5 / R7  Persistence — save + reload (across restarts)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "annotations.csv")
            df   = _empty()
            df   = add_annotation(df, ALERT_A, "Persistent note", author="Inspector")
            save_annotations(df, path)

            reloaded = load_annotations(path)
            assert len(reloaded) == 1
            assert reloaded.iloc[0]["note"]    == "Persistent note"
            assert reloaded.iloc[0]["author"]  == "Inspector"
            assert reloaded.iloc[0]["alert_id"] == ALERT_A

    def test_multiple_notes_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "annotations.csv")
            df   = _empty()
            df   = add_annotation(df, ALERT_A, "First")
            df   = add_annotation(df, ALERT_A, "Second")
            df   = add_annotation(df, ALERT_B, "Third — different alert")
            save_annotations(df, path)

            reloaded = load_annotations(path)
            assert len(reloaded) == 3
            a_notes = get_annotations_for_alert(reloaded, ALERT_A)
            b_notes = get_annotations_for_alert(reloaded, ALERT_B)
            assert len(a_notes) == 2
            assert len(b_notes) == 1

    def test_incremental_save_then_reload(self):
        """
        Simulate two separate sessions: first saves 1 note, second saves another.
        Both notes must be present after the second save.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "annotations.csv")

            # Session 1
            df1 = load_annotations(path)
            df1 = add_annotation(df1, ALERT_A, "Session 1 note")
            save_annotations(df1, path)

            # Session 2 (simulates app restart)
            df2 = load_annotations(path)
            df2 = add_annotation(df2, ALERT_A, "Session 2 note")
            save_annotations(df2, path)

            # Both notes present
            final = load_annotations(path)
            assert len(final) == 2
            notes = set(final["note"].tolist())
            assert "Session 1 note" in notes
            assert "Session 2 note" in notes

    def test_annotations_path_created_if_missing(self):
        """save_annotations creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "subdir", "annotations.csv")
            df     = _empty()
            df     = add_annotation(df, ALERT_A, "note")
            save_annotations(df, nested)
            assert os.path.exists(nested)


# ---------------------------------------------------------------------------
# R6  Validate empty notes and handle missing alert IDs safely
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_note_raises(self):
        df = _empty()
        with pytest.raises(ValueError, match="empty"):
            add_annotation(df, ALERT_A, "")

    def test_whitespace_only_note_raises(self):
        df = _empty()
        with pytest.raises(ValueError, match="empty"):
            add_annotation(df, ALERT_A, "   \t\n  ")

    def test_empty_alert_id_raises(self):
        df = _empty()
        with pytest.raises(ValueError, match="alert_id"):
            add_annotation(df, "", "some note")

    def test_whitespace_alert_id_raises(self):
        df = _empty()
        with pytest.raises(ValueError, match="alert_id"):
            add_annotation(df, "   ", "some note")

    def test_query_on_none_df_returns_empty(self):
        result = get_annotations_for_alert(None, ALERT_A)
        assert result.empty

    def test_query_on_empty_df_returns_empty(self):
        result = get_annotations_for_alert(_empty(), ALERT_A)
        assert result.empty

    def test_note_whitespace_stripped(self):
        """Leading/trailing whitespace in a non-empty note is stripped."""
        df     = _empty()
        result = add_annotation(df, ALERT_A, "  trim me  ")
        assert result.iloc[0]["note"] == "trim me"

    def test_load_annotations_handles_missing_columns(self):
        """Old CSV without all columns is loaded without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "annotations.csv")
            # Write CSV with only some columns
            with open(path, "w") as f:
                f.write("annotation_id,alert_id,note\n")
                f.write("id-001,alert-001,test\n")
            df = load_annotations(path)
            assert list(df.columns) == ANNOTATION_COLUMNS
            # Missing columns back-filled to ""
            assert df.iloc[0]["author"] == ""
            assert df.iloc[0]["created_at"] == ""


# ---------------------------------------------------------------------------
# Immutability — add_annotation must not mutate input
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_input_df_not_mutated(self):
        df       = _empty()
        original = df.copy()
        add_annotation(df, ALERT_A, "some note")
        pd.testing.assert_frame_equal(df, original)

    def test_input_df_with_rows_not_mutated(self):
        df  = _empty()
        df  = add_annotation(df, ALERT_A, "first note")
        snap = df.copy()
        add_annotation(df, ALERT_A, "second note")
        pd.testing.assert_frame_equal(df, snap)


# ---------------------------------------------------------------------------
# Dashboard integration — annotation_store importable in same context
# ---------------------------------------------------------------------------

class TestDashboardIntegration:
    def test_annotation_store_importable_alongside_dashboard_helpers(self):
        from dashboard import (  # noqa: PLC0415
            load_alerts as dash_load_alerts,
            filter_alerts,
        )
        from annotation_store import (  # noqa: PLC0415
            load_annotations as ann_load,
            add_annotation as ann_add,
        )
        # Both must be importable in the same test without collision
        assert callable(dash_load_alerts)
        assert callable(ann_load)
        assert callable(ann_add)
        assert callable(filter_alerts)
