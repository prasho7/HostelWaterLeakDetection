"""
test_annotation_visibility.py — Unit tests for annotation visibility across OPEN and RESOLVED tabs.

Verifies:
1. Annotations attached to an OPEN alert appear only when rendered for the OPEN tab.
2. When the alert is resolved, annotations follow it to the RESOLVED tab and do NOT appear in OPEN.
3. When reopened, annotations follow it back to OPEN and do NOT appear in RESOLVED.
4. Multiple annotations remain intact across all transitions.
5. Dashboard auto-refresh and live status UI are removed from main().
"""

import os
import sys
import tempfile
import inspect
import pandas as pd
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.join(_TESTS_DIR, "..", "src")
for _p in (_TESTS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alert_store
import annotation_store
import dashboard


def test_dashboard_main_no_autorefresh_or_status_bar():
    """Verify Phase 4 auto-refresh fragment, sidebar refresh slider, and status bar are removed from main()."""
    source = inspect.getsource(dashboard.main)
    assert "_live_status_fragment" not in source, "main() must not define or call _live_status_fragment"
    assert "st.fragment" not in source, "main() must not use @st.fragment auto-refresh"
    assert "_render_status_bar" not in source, "main() must not call _render_status_bar"
    assert "refresh_secs" not in source, "main() must not reference refresh_secs slider"
    assert "st.sidebar" not in source, "main() must not render auto-refresh sidebar controls"


def test_annotation_visibility_lifecycle():
    """
    Test full lifecycle of annotations:
    1. Add notes to an OPEN alert.
    2. Verify annotations belong to OPEN and not RESOLVED.
    3. Resolve alert.
    4. Verify annotations belong to RESOLVED and not OPEN.
    5. Reopen alert.
    6. Verify annotations belong to OPEN and not RESOLVED.
    7. Verify all multiple notes remain intact.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        alerts_path = os.path.join(tmpdir, "alerts.csv")
        annotations_path = os.path.join(tmpdir, "annotations.csv")

        # Initial alert in OPEN state
        alert_id = "test-alert-uuid-1234"
        initial_alerts = pd.DataFrame([{
            "alert_id": alert_id,
            "hostel_id": "H1",
            "floor_id": "H1-F1",
            "meter_id": "M-H1-F1",
            "first_seen": "2024-01-01T03:00:00",
            "last_seen": "2024-01-02T08:00:00",
            "severity": "HIGH",
            "avg_z_score": 35.0,
            "persistence_count": 30,
            "status": "OPEN",
            "resolved_at": "",
        }])
        alert_store.save_alerts(initial_alerts, alerts_path)

        # 1. Add annotations to the alert
        ann_df = annotation_store.load_annotations(annotations_path)
        ann_df = annotation_store.add_annotation(ann_df, alert_id=alert_id, note="First inspection: dripping pipe", author="Warden 1")
        ann_df = annotation_store.add_annotation(ann_df, alert_id=alert_id, note="Second note: plumber assigned", author="Warden 2")
        annotation_store.save_annotations(ann_df, annotations_path)

        # Helper simulating dashboard filtering
        def get_tabs_data(a_path):
            current_alerts = alert_store.load_alerts(a_path)
            open_tab = current_alerts[current_alerts["status"] == "OPEN"]
            resolved_tab = current_alerts[current_alerts["status"] == "RESOLVED"]
            return open_tab, resolved_tab

        # Verify initial state: alert is in OPEN tab, NOT in RESOLVED tab
        open_tab, resolved_tab = get_tabs_data(alerts_path)
        assert alert_id in open_tab["alert_id"].values
        assert alert_id not in resolved_tab["alert_id"].values

        # Annotations are attached to this alert
        notes = annotation_store.get_annotations_for_alert(annotation_store.load_annotations(annotations_path), alert_id)
        assert len(notes) == 2
        assert list(notes["note"]) == ["First inspection: dripping pipe", "Second note: plumber assigned"]

        # 2. Resolve the alert
        raw_df = alert_store.load_alerts(alerts_path)
        raw_df = alert_store.resolve_alert(raw_df, alert_id)
        alert_store.save_alerts(raw_df, alerts_path)

        # Check tabs data after resolution
        open_tab, resolved_tab = get_tabs_data(alerts_path)
        assert alert_id not in open_tab["alert_id"].values, "Resolved alert must NOT appear in OPEN tab"
        assert alert_id in resolved_tab["alert_id"].values, "Resolved alert must appear in RESOLVED tab"

        # Verify notes follow to RESOLVED tab and remain intact
        notes_resolved = annotation_store.get_annotations_for_alert(annotation_store.load_annotations(annotations_path), alert_id)
        assert len(notes_resolved) == 2
        assert list(notes_resolved["note"]) == ["First inspection: dripping pipe", "Second note: plumber assigned"]

        # Add a 3rd note while RESOLVED
        ann_df = annotation_store.load_annotations(annotations_path)
        ann_df = annotation_store.add_annotation(ann_df, alert_id=alert_id, note="Resolved: valve replaced", author="Warden 1")
        annotation_store.save_annotations(ann_df, annotations_path)

        # 3. Reopen the alert
        raw_df = alert_store.load_alerts(alerts_path)
        raw_df = alert_store.reopen_alert(raw_df, alert_id)
        alert_store.save_alerts(raw_df, alerts_path)

        # Check tabs data after reopen
        open_tab, resolved_tab = get_tabs_data(alerts_path)
        assert alert_id in open_tab["alert_id"].values, "Reopened alert must appear in OPEN tab"
        assert alert_id not in resolved_tab["alert_id"].values, "Reopened alert must NOT appear in RESOLVED tab"

        # Verify all 3 notes are intact in OPEN tab
        notes_reopened = annotation_store.get_annotations_for_alert(annotation_store.load_annotations(annotations_path), alert_id)
        assert len(notes_reopened) == 3
        assert list(notes_reopened["note"]) == [
            "First inspection: dripping pipe",
            "Second note: plumber assigned",
            "Resolved: valve replaced"
        ]


def test_render_annotations_read_only_when_resolved():
    """Verify that _render_annotations only renders the input form when alert is OPEN."""
    source = inspect.getsource(dashboard._render_annotations)
    assert "if not is_resolved:" in source, "_render_annotations must guard form entry behind 'if not is_resolved:'"
    assert "with st.form(key=form_key):" in source, "Form must be inside the conditional block"

