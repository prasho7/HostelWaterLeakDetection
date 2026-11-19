"""
test_alert_resolution.py — Tests for Phase 1: Alert Resolution

Tests cover:
  1.  resolve_alert sets status=RESOLVED and records resolved_at timestamp.
  2.  resolve_alert preserves all other columns unchanged.
  3.  resolve_alert raises ValueError for an unknown alert_id.
  4.  resolve_alert raises ValueError when alert is already RESOLVED.
  5.  reopen_alert clears status back to OPEN and clears resolved_at.
  6.  reopen_alert raises ValueError for an unknown alert_id.
  7.  reopen_alert raises ValueError when alert is already OPEN.
  8.  Custom resolved_at timestamp is respected (not overwritten).
  9.  save_alerts + load_alerts round-trip: status change persists on disk.
  10. Persistence across multiple operations: resolve then reopen then re-resolve.
  11. load_alerts back-fills resolved_at="" for old CSV without that column.
  12. resolve_alert does not mutate the input DataFrame.
  13. reopen_alert does not mutate the input DataFrame.
"""

import os
import sys
import tempfile

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")

for path in (_PROJECT_ROOT, _SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import alert_store  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_ALERTS_CSV = """\
alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
alert-001,H1,H1-F2,M-H1-F2,2024-01-01T03:00:00,2024-01-02T08:00:00,HIGH,34.98,30,OPEN,
alert-002,H2,H2-F3,M-H2-F3,2024-01-04T06:00:00,2024-01-05T11:00:00,HIGH,31.20,30,OPEN,
alert-003,H3,H3-F4,M-H3-F4,2024-01-19T09:00:00,2024-01-20T13:00:00,HIGH,34.54,29,OPEN,
"""

# CSV that lacks the resolved_at column (simulates old format)
OLD_FORMAT_CSV = """\
alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status
alert-001,H1,H1-F2,M-H1-F2,2024-01-01T03:00:00,2024-01-02T08:00:00,HIGH,34.98,30,OPEN
alert-002,H2,H2-F3,M-H2-F3,2024-01-04T06:00:00,2024-01-05T11:00:00,HIGH,31.20,30,OPEN
"""


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a fresh in-memory DataFrame from the sample CSV content."""
    import io
    return alert_store.load_alerts.__wrapped__(io.StringIO(SAMPLE_ALERTS_CSV)) \
        if hasattr(alert_store.load_alerts, "__wrapped__") \
        else _load_from_string(SAMPLE_ALERTS_CSV)


def _load_from_string(csv_str: str) -> pd.DataFrame:
    """Helper: load alerts from a CSV string using alert_store logic."""
    import io
    df = pd.read_csv(io.StringIO(csv_str), dtype=str)
    if "resolved_at" not in df.columns:
        df["resolved_at"] = ""
    df["resolved_at"] = df["resolved_at"].fillna("")
    return df[alert_store.ALERT_COLUMNS].copy()


@pytest.fixture
def df() -> pd.DataFrame:
    return _load_from_string(SAMPLE_ALERTS_CSV)


@pytest.fixture
def tmp_csv(tmp_path) -> str:
    """Write sample CSV to a temp file; return its path."""
    path = str(tmp_path / "alerts.csv")
    with open(path, "w", newline="") as f:
        f.write(SAMPLE_ALERTS_CSV)
    return path


# ---------------------------------------------------------------------------
# Test 1 — resolve sets status=RESOLVED and records resolved_at
# ---------------------------------------------------------------------------
def test_resolve_sets_status_and_timestamp(df):
    result = alert_store.resolve_alert(df, "alert-001")
    row = result[result["alert_id"] == "alert-001"].iloc[0]
    assert row["status"] == "RESOLVED"
    assert row["resolved_at"] != ""


# ---------------------------------------------------------------------------
# Test 2 — resolve preserves all other columns
# ---------------------------------------------------------------------------
def test_resolve_preserves_other_columns(df):
    result = alert_store.resolve_alert(df, "alert-001")
    orig = df[df["alert_id"] == "alert-001"].iloc[0]
    new  = result[result["alert_id"] == "alert-001"].iloc[0]
    for col in ("hostel_id", "floor_id", "meter_id", "first_seen", "last_seen",
                "severity", "avg_z_score", "persistence_count"):
        assert new[col] == orig[col], f"Column {col!r} changed unexpectedly"


# ---------------------------------------------------------------------------
# Test 3 — resolve raises ValueError for unknown alert_id
# ---------------------------------------------------------------------------
def test_resolve_unknown_id_raises(df):
    with pytest.raises(ValueError, match="not found"):
        alert_store.resolve_alert(df, "DOES-NOT-EXIST")


# ---------------------------------------------------------------------------
# Test 4 — resolve raises ValueError if already RESOLVED
# ---------------------------------------------------------------------------
def test_resolve_already_resolved_raises(df):
    resolved = alert_store.resolve_alert(df, "alert-001")
    with pytest.raises(ValueError, match="already RESOLVED"):
        alert_store.resolve_alert(resolved, "alert-001")


# ---------------------------------------------------------------------------
# Test 5 — reopen clears status to OPEN and clears resolved_at
# ---------------------------------------------------------------------------
def test_reopen_clears_status_and_timestamp(df):
    resolved = alert_store.resolve_alert(df, "alert-002")
    reopened = alert_store.reopen_alert(resolved, "alert-002")
    row = reopened[reopened["alert_id"] == "alert-002"].iloc[0]
    assert row["status"] == "OPEN"
    assert row["resolved_at"] == ""


# ---------------------------------------------------------------------------
# Test 6 — reopen raises ValueError for unknown alert_id
# ---------------------------------------------------------------------------
def test_reopen_unknown_id_raises(df):
    with pytest.raises(ValueError, match="not found"):
        alert_store.reopen_alert(df, "DOES-NOT-EXIST")


# ---------------------------------------------------------------------------
# Test 7 — reopen raises ValueError if already OPEN
# ---------------------------------------------------------------------------
def test_reopen_already_open_raises(df):
    with pytest.raises(ValueError, match="already OPEN"):
        alert_store.reopen_alert(df, "alert-001")


# ---------------------------------------------------------------------------
# Test 8 — custom resolved_at timestamp is respected
# ---------------------------------------------------------------------------
def test_resolve_custom_timestamp(df):
    custom_ts = "2024-02-15T10:30:00"
    result = alert_store.resolve_alert(df, "alert-001", resolved_at=custom_ts)
    row = result[result["alert_id"] == "alert-001"].iloc[0]
    assert row["resolved_at"] == custom_ts


# ---------------------------------------------------------------------------
# Test 9 — save + load round-trip: status change persists on disk
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_persistence(tmp_csv, df):
    # Resolve alert-001 and save
    updated = alert_store.resolve_alert(df, "alert-001")
    alert_store.save_alerts(updated, tmp_csv)

    # Reload from disk
    reloaded = alert_store.load_alerts(tmp_csv)
    row = reloaded[reloaded["alert_id"] == "alert-001"].iloc[0]
    assert row["status"] == "RESOLVED"
    assert row["resolved_at"] != ""

    # Other alerts remain OPEN
    assert reloaded[reloaded["alert_id"] == "alert-002"].iloc[0]["status"] == "OPEN"


# ---------------------------------------------------------------------------
# Test 10 — multiple operations: resolve → reopen → re-resolve
# ---------------------------------------------------------------------------
def test_multiple_operations_persist(tmp_csv, df):
    # Step 1: resolve
    step1 = alert_store.resolve_alert(df, "alert-001")
    alert_store.save_alerts(step1, tmp_csv)

    # Step 2: reopen from disk
    loaded1 = alert_store.load_alerts(tmp_csv)
    step2 = alert_store.reopen_alert(loaded1, "alert-001")
    alert_store.save_alerts(step2, tmp_csv)

    loaded2 = alert_store.load_alerts(tmp_csv)
    assert loaded2[loaded2["alert_id"] == "alert-001"].iloc[0]["status"] == "OPEN"

    # Step 3: re-resolve from disk
    step3 = alert_store.resolve_alert(loaded2, "alert-001",
                                       resolved_at="2024-03-01T08:00:00")
    alert_store.save_alerts(step3, tmp_csv)

    loaded3 = alert_store.load_alerts(tmp_csv)
    row = loaded3[loaded3["alert_id"] == "alert-001"].iloc[0]
    assert row["status"] == "RESOLVED"
    assert row["resolved_at"] == "2024-03-01T08:00:00"


# ---------------------------------------------------------------------------
# Test 11 — load_alerts back-fills resolved_at for old CSV format
# ---------------------------------------------------------------------------
def test_backfill_resolved_at_for_old_csv(tmp_path):
    old_path = str(tmp_path / "old_alerts.csv")
    with open(old_path, "w", newline="") as f:
        f.write(OLD_FORMAT_CSV)

    df = alert_store.load_alerts(old_path)
    assert "resolved_at" in df.columns
    assert (df["resolved_at"] == "").all()


# ---------------------------------------------------------------------------
# Test 12 — resolve_alert does not mutate input DataFrame
# ---------------------------------------------------------------------------
def test_resolve_does_not_mutate_input(df):
    original_statuses = df["status"].tolist()
    alert_store.resolve_alert(df, "alert-001")
    assert df["status"].tolist() == original_statuses


# ---------------------------------------------------------------------------
# Test 13 — reopen_alert does not mutate input DataFrame
# ---------------------------------------------------------------------------
def test_reopen_does_not_mutate_input(df):
    resolved = alert_store.resolve_alert(df, "alert-001")
    resolved_statuses = resolved["status"].tolist()
    alert_store.reopen_alert(resolved, "alert-001")
    assert resolved["status"].tolist() == resolved_statuses


# ---------------------------------------------------------------------------
# Tests 14-17 — merge_with_existing (ID stability across pipeline re-runs)
# ---------------------------------------------------------------------------

import sys as _sys
_sys.path.insert(0, _SRC_DIR)
from alert_manager import merge_with_existing  # noqa: E402

EXISTING_ALERTS_CSV = """\
alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
stable-uuid-001,H1,H1-F2,M-H1-F2,2024-01-01T03:00:00,2024-01-02T08:00:00,HIGH,39.30,30,RESOLVED,2024-01-10T08:00:00
stable-uuid-002,H2,H2-F3,M-H2-F3,2024-01-04T06:00:00,2024-01-05T11:00:00,HIGH,31.20,30,OPEN,
"""

NEW_DETECTED_CSV = """\
alert_id,hostel_id,floor_id,meter_id,first_seen,last_seen,severity,avg_z_score,persistence_count,status,resolved_at
brand-new-uuid-A,H1,H1-F2,M-H1-F2,2024-01-01T03:00:00,2024-01-02T08:00:00,HIGH,39.30,30,OPEN,
brand-new-uuid-B,H2,H2-F3,M-H2-F3,2024-01-04T06:00:00,2024-01-05T11:00:00,HIGH,31.20,30,OPEN,
brand-new-uuid-C,H3,H3-F4,M-H3-F4,2024-01-19T09:00:00,2024-01-20T13:00:00,HIGH,34.54,29,OPEN,
"""


def _df_from_str(csv_str):
    import io
    df = pd.read_csv(io.StringIO(csv_str), dtype=str)
    if "resolved_at" not in df.columns:
        df["resolved_at"] = ""
    df["resolved_at"] = df["resolved_at"].fillna("")
    return df


def test_merge_preserves_existing_alert_id(tmp_path):
    """Test 14: merge keeps stable UUID for an alert that matched by key."""
    existing_path = str(tmp_path / "alerts.csv")
    with open(existing_path, "w") as f:
        f.write(EXISTING_ALERTS_CSV)

    new_df = _df_from_str(NEW_DETECTED_CSV)
    merged = merge_with_existing(new_df, existing_path)

    # Alert that existed before must keep its stable ID
    h1_row = merged[merged["meter_id"] == "M-H1-F2"].iloc[0]
    assert h1_row["alert_id"] == "stable-uuid-001", (
        f"Expected stable-uuid-001 but got {h1_row['alert_id']!r}"
    )


def test_merge_preserves_resolved_status(tmp_path):
    """Test 15: merge keeps RESOLVED status and resolved_at for matched alerts."""
    existing_path = str(tmp_path / "alerts.csv")
    with open(existing_path, "w") as f:
        f.write(EXISTING_ALERTS_CSV)

    new_df = _df_from_str(NEW_DETECTED_CSV)
    merged = merge_with_existing(new_df, existing_path)

    h1_row = merged[merged["meter_id"] == "M-H1-F2"].iloc[0]
    assert h1_row["status"] == "RESOLVED"
    assert h1_row["resolved_at"] == "2024-01-10T08:00:00"


def test_merge_new_alert_gets_fresh_uuid(tmp_path):
    """Test 16: a genuinely new alert (not in existing) keeps its fresh UUID."""
    existing_path = str(tmp_path / "alerts.csv")
    with open(existing_path, "w") as f:
        f.write(EXISTING_ALERTS_CSV)

    new_df = _df_from_str(NEW_DETECTED_CSV)
    merged = merge_with_existing(new_df, existing_path)

    # H3-F4 alert is brand-new — not in existing file
    h3_row = merged[merged["meter_id"] == "M-H3-F4"].iloc[0]
    assert h3_row["alert_id"] == "brand-new-uuid-C"
    assert h3_row["status"] == "OPEN"


def test_merge_no_existing_file_returns_new_df(tmp_path):
    """Test 17: when no existing file is present, new_df is returned unchanged."""
    missing_path = str(tmp_path / "does_not_exist.csv")
    new_df = _df_from_str(NEW_DETECTED_CSV)
    merged = merge_with_existing(new_df, missing_path)

    assert len(merged) == len(new_df)
    assert list(merged["alert_id"]) == list(new_df["alert_id"])


def test_pipeline_rerun_keeps_ids_stable(tmp_path):
    """
    Test 18: end-to-end simulation of pipeline re-run:
      - First run writes alerts with IDs X.
      - Warden resolves alert X[0].
      - Pipeline re-runs (new UUIDs generated) → merge preserves X[0] as RESOLVED.
    """
    alerts_path = str(tmp_path / "alerts.csv")

    # First run: write initial alerts
    first_run = _df_from_str(NEW_DETECTED_CSV)
    # Give them stable IDs as first run would
    first_run = first_run.copy()
    first_run.loc[first_run["meter_id"] == "M-H1-F2", "alert_id"] = "run1-id-H1F2"
    first_run.loc[first_run["meter_id"] == "M-H2-F3", "alert_id"] = "run1-id-H2F3"
    first_run.loc[first_run["meter_id"] == "M-H3-F4", "alert_id"] = "run1-id-H3F4"
    first_run.to_csv(alerts_path, index=False)

    # Warden resolves H1-F2 alert
    on_disk = alert_store.load_alerts(alerts_path)
    resolved = alert_store.resolve_alert(on_disk, "run1-id-H1F2",
                                          resolved_at="2024-02-01T10:00:00")
    alert_store.save_alerts(resolved, alerts_path)

    # Pipeline re-runs: build_alerts would generate new UUIDs
    second_run = _df_from_str(NEW_DETECTED_CSV)  # all brand-new UUIDs

    # merge_with_existing restores stable IDs and statuses
    merged = merge_with_existing(second_run, alerts_path)
    merged.to_csv(alerts_path, index=False)

    # Verify H1-F2 kept its ID and RESOLVED status
    final = alert_store.load_alerts(alerts_path)
    h1_row = final[final["meter_id"] == "M-H1-F2"].iloc[0]
    assert h1_row["alert_id"] == "run1-id-H1F2"
    assert h1_row["status"] == "RESOLVED"
    assert h1_row["resolved_at"] == "2024-02-01T10:00:00"


# ---------------------------------------------------------------------------
# Test 19 — find_matching_alert_id functionality
# ---------------------------------------------------------------------------
def test_find_matching_alert_id_direct_match(df):
    matched = alert_store.find_matching_alert_id(df, alert_id="alert-001")
    assert matched == "alert-001"


def test_find_matching_alert_id_fallback_timestamps(df):
    # Pass a non-existent alert_id but matching meter_id, first_seen, and last_seen
    matched = alert_store.find_matching_alert_id(
        df,
        alert_id="stale-old-uuid-999",
        meter_id="M-H1-F2",
        first_seen="2024-01-01T03:00:00",
        last_seen="2024-01-02T08:00:00",
    )
    assert matched == "alert-001"


def test_find_matching_alert_id_fallback_datetime_objects(df):
    matched = alert_store.find_matching_alert_id(
        df,
        alert_id="stale-old-uuid-999",
        meter_id="M-H1-F2",
        first_seen=pd.Timestamp("2024-01-01 03:00:00"),
        last_seen=pd.Timestamp("2024-01-02 08:00:00"),
    )
    assert matched == "alert-001"


def test_find_matching_alert_id_not_found(df):
    matched = alert_store.find_matching_alert_id(
        df,
        alert_id="unknown-uuid",
        meter_id="M-H9-F9",
        first_seen="2024-01-01T00:00:00",
        last_seen="2024-01-01T01:00:00",
    )
    assert matched is None


def test_find_matching_alert_id_empty_df():
    empty_df = pd.DataFrame(columns=alert_store.ALERT_COLUMNS)
    assert alert_store.find_matching_alert_id(empty_df, alert_id="foo") is None

