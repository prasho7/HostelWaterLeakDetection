"""
annotation_store.py — Phase 5: Per-Alert Annotation Persistence Layer

Provides pure functions for adding, loading, and querying timestamped
warden notes (annotations) attached to individual alerts.

Storage: data/annotations.csv
Schema:
    annotation_id : UUID string   — unique per note
    alert_id      : UUID string   — foreign key into alerts.csv
    author        : str           — who wrote the note (default "Warden")
    note          : str           — free-text note body
    created_at    : ISO 8601 str  — UTC timestamp of creation

Design decisions
----------------
- Completely independent of leak detection, pipeline, and alert resolution.
- Annotations are append-only: existing rows are never modified or deleted.
- Annotations survive alert resolve / reopen because they reference alert_id,
  which is stable across those operations.
- Empty or whitespace-only notes are rejected with a ValueError.
- An annotation for a non-existent alert_id is accepted (lookup-time guard is
  in the UI layer), but add_annotation validates that alert_id is non-empty.
"""

import os
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Project root — allows this module to be run standalone or imported
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Default paths (overridable in tests)
# ---------------------------------------------------------------------------
DEFAULT_ANNOTATIONS_PATH = os.path.join(_PROJECT_ROOT, "data", "annotations.csv")

# Column order for every saved/loaded CSV
ANNOTATION_COLUMNS = [
    "annotation_id",
    "alert_id",
    "author",
    "note",
    "created_at",
]

DEFAULT_AUTHOR = "Warden"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_annotations(path: str = DEFAULT_ANNOTATIONS_PATH) -> pd.DataFrame:
    """
    Load annotations.csv and return a DataFrame with all required columns.
    Returns an empty DataFrame (with correct columns) if the file is absent.
    All columns are read as strings; created_at is NOT auto-parsed so callers
    can choose their own datetime handling.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    df = pd.read_csv(path, dtype=str)

    # Back-fill any columns that might be missing (forward-compat guard)
    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")
    return df[ANNOTATION_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def get_annotations_for_alert(
    df: pd.DataFrame,
    alert_id: str,
) -> pd.DataFrame:
    """
    Return all annotations whose alert_id matches the given alert_id,
    sorted by created_at ascending (oldest first).

    Parameters
    ----------
    df       : annotations DataFrame (from load_annotations)
    alert_id : the alert UUID to filter on

    Returns
    -------
    Filtered DataFrame, possibly empty.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    result = df[df["alert_id"].astype(str).str.strip() == str(alert_id).strip()]
    return result.sort_values("created_at").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

def add_annotation(
    df: pd.DataFrame,
    alert_id: str,
    note: str,
    author: str = DEFAULT_AUTHOR,
    created_at: str | None = None,
) -> pd.DataFrame:
    """
    Append a new annotation row to the DataFrame and return the updated copy.

    The original DataFrame is not mutated.

    Parameters
    ----------
    df         : existing annotations DataFrame (from load_annotations)
    alert_id   : UUID string of the target alert
    note       : free-text annotation body (must be non-empty)
    author     : author name (defaults to "Warden")
    created_at : ISO 8601 timestamp string; defaults to current UTC time

    Returns
    -------
    New DataFrame with the annotation appended.

    Raises
    ------
    ValueError : if note is empty/whitespace or alert_id is empty.
    """
    if not str(alert_id).strip():
        raise ValueError("alert_id must not be empty.")

    if not str(note).strip():
        raise ValueError("Annotation note must not be empty.")

    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    new_row = pd.DataFrame(
        [{
            "annotation_id": str(uuid.uuid4()),
            "alert_id":      str(alert_id).strip(),
            "author":        str(author).strip() or DEFAULT_AUTHOR,
            "note":          str(note).strip(),
            "created_at":    created_at,
        }],
        columns=ANNOTATION_COLUMNS,
    )

    return pd.concat([df, new_row], ignore_index=True)[ANNOTATION_COLUMNS]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_annotations(
    df: pd.DataFrame,
    path: str = DEFAULT_ANNOTATIONS_PATH,
) -> None:
    """
    Persist the annotations DataFrame to CSV.
    Always writes the canonical ANNOTATION_COLUMNS column order.
    Creates any missing parent directories.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df[ANNOTATION_COLUMNS].to_csv(path, index=False)
