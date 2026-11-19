"""
dashboard.py — Warden Dashboard for Hostel Water Leak Detector
Hierarchical navigation: Overview → Hostel → Floor

Production-grade UI with theme-adaptive styling, modern typography,
and high-contrast elements across light and dark modes.

Run from project root:  streamlit run src/dashboard.py
"""

import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st

# alert_store is used for resolve/reopen persistence
import importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alert_store       # noqa: E402
import annotation_store  # noqa: E402
import chart_builder     # noqa: E402
importlib.reload(alert_store)
importlib.reload(annotation_store)
importlib.reload(chart_builder)

# ---------------------------------------------------------------------------
# Data root: works whether launched from project root or from src/
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..", "data")

# Also make config importable
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import HOSTEL_FLOORS, FLOOR_ROOMS, LEAK_INJECT_METERS, METER_IDS  # noqa: E402

# ---------------------------------------------------------------------------
# Auto-refresh defaults (preserved for helper test compatibility)
# ---------------------------------------------------------------------------
DEFAULT_REFRESH_SECS = 5
STALE_MULTIPLIER     = 3


# ---------------------------------------------------------------------------
# Pure data-loading functions
# ---------------------------------------------------------------------------

def load_alerts(path: str) -> pd.DataFrame:
    """
    Load and parse alerts.csv via alert_store (handles resolved_at back-fill).
    Returns an empty DataFrame on missing file.
    Datetime columns first_seen and last_seen are parsed to datetime.
    """
    df = alert_store.load_alerts(path)
    if df.empty:
        return df
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df["last_seen"]  = pd.to_datetime(df["last_seen"])
    return df


def load_readings(path: str) -> pd.DataFrame:
    """Load and parse processed_readings.csv."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp", "hostel_id", "floor_id", "meter_id",
            "meter_reading", "delta_litres", "hour", "day_of_week",
        ])
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def load_anomalies(path: str) -> pd.DataFrame:
    """Load and parse anomalies.csv."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp", "hostel_id", "floor_id", "meter_id",
            "delta_litres", "median_baseline", "mad", "z_score_mad",
            "is_anomaly", "persistence_count",
        ])
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def filter_alerts(
    df: pd.DataFrame,
    hostel_id: str = None,
    floor_id: str = None,
    severity: str = None,
) -> pd.DataFrame:
    """
    Filter the alerts DataFrame by hostel, floor, and/or severity.
    None / 'All' means no filter on that dimension.
    Returns a copy.
    """
    result = df.copy()
    if hostel_id and hostel_id != "All":
        result = result[result["hostel_id"] == hostel_id]
    if floor_id and floor_id != "All":
        result = result[result["floor_id"] == floor_id]
    if severity and severity != "All":
        result = result[result["severity"] == severity]
    return result


# ---------------------------------------------------------------------------
# Severity colour helpers
# ---------------------------------------------------------------------------

SEVERITY_COLOR = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}
SEVERITY_BG    = {"HIGH": "#fef2f2", "MEDIUM": "#fffbeb", "LOW": "#f0fdf4"}


def _severity_badge(sev: str) -> str:
    color = SEVERITY_COLOR.get(sev, "#6b7280")
    return f'<span style="background:{color};color:#fff;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.04em;">{sev}</span>'


def get_hostel_status(hostel_alerts: pd.DataFrame) -> tuple[str, str, str]:
    """
    Returns (status_text, border_color, bg_color) based only on remaining OPEN alerts.
    """
    if hostel_alerts is None or hostel_alerts.empty:
        return "✅ Normal", "#16a34a", "#fafafa"

    open_alerts = hostel_alerts[hostel_alerts["status"] == "OPEN"]
    h_open = len(open_alerts)
    h_high = int((open_alerts["severity"] == "HIGH").sum()) if not open_alerts.empty else 0
    h_resolved = int((hostel_alerts["status"] == "RESOLVED").sum())

    if h_high > 0:
        return "⚠️ Leak Detected", "#dc2626", "#fafafa"
    elif h_open > 0:
        return "🔔 Alerts", "#d97706", "#fafafa"
    else:
        bg = "#f0fdf4" if h_resolved > 0 else "#fafafa"
        return "✅ Normal", "#16a34a", bg


def get_floor_status(floor_alerts: pd.DataFrame) -> tuple[str, str, str]:
    """
    Returns (status_text, border_color, bg_color) based only on remaining OPEN alerts.
    """
    if floor_alerts is None or floor_alerts.empty:
        return "✅ Normal", "#16a34a", "#fafafa"

    open_alerts = floor_alerts[floor_alerts["status"] == "OPEN"]
    f_open = len(open_alerts)
    f_high = int((open_alerts["severity"] == "HIGH").sum()) if not open_alerts.empty else 0
    f_resolved = int((floor_alerts["status"] == "RESOLVED").sum())

    if f_high > 0:
        return "⚠️ LEAK ALERT", "#dc2626", "#fafafa"
    elif f_open > 0:
        return "🔔 Alert", "#d97706", "#fafafa"
    else:
        bg = "#f0fdf4" if f_resolved > 0 else "#fafafa"
        return "✅ Normal", "#16a34a", bg


def get_alert_card_style(row: pd.Series | dict) -> dict:
    """
    Return styling dict for an alert card:
      - border_color: CSS color string
      - bg_color: CSS background string
      - badge_html: HTML string for the status/severity badge
      - is_resolved: bool
    """
    status = str(row.get("status", "OPEN")).strip().upper()
    severity = str(row.get("severity", "LOW")).strip().upper()
    is_resolved = (status == "RESOLVED")

    if is_resolved:
        return {
            "border_color": "#16a34a",
            "bg_color": "#f0fdf4",
            "badge_html": '<span style="background:#16a34a;color:#ffffff;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.04em;">✅ RESOLVED</span>',
            "is_resolved": True,
        }
    else:
        sev_color = SEVERITY_COLOR.get(severity, "#dc2626")
        sev_bg = SEVERITY_BG.get(severity, "#fef2f2")
        border = sev_color if severity == "HIGH" else ("#d97706" if severity == "MEDIUM" else "#16a34a")
        return {
            "border_color": border,
            "bg_color": sev_bg,
            "badge_html": f'<span style="background:{sev_color};color:#ffffff;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.04em;">{severity}</span>',
            "is_resolved": False,
        }


# ---------------------------------------------------------------------------
# Global Design System & CSS Injection
# ---------------------------------------------------------------------------

def _inject_custom_css():
    """Inject polished, modern, theme-adaptive CSS styles."""
    st.markdown(
        """
        <style>
        /* Seamless transparent header that perfectly matches light AND dark themes */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Remove the animated/glitching rainbow decoration line at the very top */
        [data-testid="stDecoration"] {
            display: none !important;
        }

        /* Hide the Deploy button to keep the header bar clean */
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }

        /* Ensure 3-dot main menu is fully accessible for Settings & Theme toggle */
        #MainMenu, [data-testid="stMainMenu"] {
            display: block !important;
            visibility: visible !important;
        }

        footer {
            visibility: hidden !important;
        }

        /* Container spacing: clean breathing room below top toolbar */
        .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }

        /* App Header */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.25rem;
            flex-wrap: wrap;
            gap: 8px;
        }
        .header-title-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-badge {
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: #ffffff !important;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 9999px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .app-subtitle {
            font-size: 0.9rem;
            opacity: 0.75;
            margin-bottom: 1.25rem;
            color: var(--text-color);
        }

        /* Streamlit Metrics Polish */
        [data-testid="stMetric"] {
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.08)) !important;
            border: 1px solid rgba(128, 128, 128, 0.18) !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stMetric"]:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: var(--text-color) !important;
            opacity: 0.85;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            color: var(--text-color) !important;
        }

        /* Breadcrumb Bar */
        .breadcrumb-bar {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: var(--text-color);
            opacity: 0.8;
            margin-bottom: 12px;
        }

        /* Hostel & Floor Cards */
        .h-card, .f-card {
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 8px;
            transition: all 0.2s ease-in-out;
            color: var(--text-color);
        }
        .h-card:hover, .f-card:hover {
            border-color: rgba(128, 128, 128, 0.35);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.07);
        }
        .card-header-line {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .card-title-text {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-color);
            margin: 0;
        }
        .card-desc-text {
            font-size: 13px;
            opacity: 0.75;
            margin: 2px 0 10px 0;
            color: var(--text-color);
        }
        .card-footer-metrics {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
            font-size: 13px;
            padding-top: 8px;
            border-top: 1px solid rgba(128, 128, 128, 0.12);
            color: var(--text-color);
        }

        /* Pill tags */
        .stat-pill {
            display: inline-flex;
            align-items: center;
            background: rgba(128, 128, 128, 0.12);
            color: var(--text-color);
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
        }
        .stat-pill-alert {
            background: rgba(220, 38, 38, 0.15) !important;
            color: #ef4444 !important;
            font-weight: 600;
        }
        .stat-pill-normal {
            background: rgba(22, 163, 74, 0.15) !important;
            color: #10b981 !important;
            font-weight: 600;
        }

        /* Alert Containers */
        .alert-item-card {
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 8px;
            color: var(--text-color);
        }
        .alert-item-card.is-open {
            border-left: 5px solid #dc2626 !important;
        }
        .alert-item-card.is-resolved {
            border-left: 5px solid #16a34a !important;
        }

        /* Note Items */
        .note-card-entry {
            border-left: 3px solid #0284c7;
            background: rgba(2, 132, 199, 0.08);
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 6px;
            color: var(--text-color);
        }
        .note-header-line {
            font-size: 11px;
            opacity: 0.7;
            font-weight: 600;
            color: var(--text-color);
        }
        .note-body-text {
            font-size: 13.5px;
            margin-top: 2px;
            color: var(--text-color);
        }

        /* Room Directory Grid */
        .room-box {
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 10px;
            padding: 14px 10px;
            text-align: center;
            color: var(--text-color);
            transition: transform 0.15s ease;
        }
        .room-box:hover {
            transform: translateY(-2px);
        }
        .room-num {
            font-weight: 700;
            font-size: 14px;
            margin-top: 4px;
            color: var(--text-color);
        }
        .room-sub {
            font-size: 11px;
            opacity: 0.65;
            color: var(--text-color);
        }

        /* Modern Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def _render_overview(alerts: pd.DataFrame, readings: pd.DataFrame):
    st.markdown(
        """
        <div class="app-header">
          <div class="header-title-row">
            <h1 style="margin: 0; padding: 0; font-size: 1.85rem; font-weight: 800;">🏢 Campus Water Leak Monitor</h1>
            <span class="header-badge">Warden Portal</span>
          </div>
        </div>
        <p class="app-subtitle">
          Real-time telemetry and anomaly detection across all campus student residential hostels.
        </p>
        """,
        unsafe_allow_html=True,
    )

    open_alerts_all = alerts[alerts["status"] == "OPEN"] if not alerts.empty else pd.DataFrame()
    total   = len(open_alerts_all)
    n_high  = int((open_alerts_all["severity"] == "HIGH").sum()) if not open_alerts_all.empty else 0
    n_med   = int((open_alerts_all["severity"] == "MEDIUM").sum()) if not open_alerts_all.empty else 0
    n_low   = int((open_alerts_all["severity"] == "LOW").sum()) if not open_alerts_all.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 Total Open Alerts", total)
    c2.metric("🔴 HIGH Severity",   n_high)
    c3.metric("🟡 MEDIUM Severity", n_med)
    c4.metric("🟢 LOW Severity",    n_low)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.subheader("Hostel Residential Blocks")
    st.caption("Select a hostel facility to inspect floor-by-floor meter telemetry and active alerts.")

    cols = st.columns(3)
    for i, (hostel_id, floors) in enumerate(HOSTEL_FLOORS.items()):
        hostel_alerts = alerts[alerts["hostel_id"] == hostel_id] if not alerts.empty else pd.DataFrame()
        h_open_alerts = hostel_alerts[hostel_alerts["status"] == "OPEN"] if not hostel_alerts.empty else pd.DataFrame()
        h_open   = len(h_open_alerts)
        h_resolved = int((hostel_alerts["status"] == "RESOLVED").sum()) if not hostel_alerts.empty else 0
        status, brd, bg = get_hostel_status(hostel_alerts)

        with cols[i]:
            alert_pill = (
                f'<span class="stat-pill stat-pill-alert">🚨 {h_open} Open Alert(s)</span>'
                if h_open > 0 else
                f'<span class="stat-pill stat-pill-normal">✅ Normal Flow</span>'
            )
            resolved_pill = (
                f'<span class="stat-pill">Resolved: <b>{h_resolved}</b></span>'
                if h_resolved > 0 else ""
            )

            st.markdown(
                f"""
                <div class="h-card" style="border-left: 5px solid {brd};">
                  <div class="card-header-line">
                    <h3 class="card-title-text">Hostel {hostel_id}</h3>
                    <span style="font-size: 13px; font-weight: 600;">{status}</span>
                  </div>
                  <div class="card-desc-text">
                    {len(floors)} Floors &nbsp;•&nbsp; {len(floors)*4} Rooms &nbsp;•&nbsp; {len(floors)} Flow Meters
                  </div>
                  <div class="card-footer-metrics">
                    {alert_pill}
                    {resolved_pill}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Inspect Hostel {hostel_id} →", key=f"btn_hostel_{hostel_id}", use_container_width=True):
                st.session_state["page"]    = "hostel"
                st.session_state["hostel"]  = hostel_id
                st.rerun()


def _render_hostel(hostel_id: str, alerts: pd.DataFrame, readings: pd.DataFrame, anomalies: pd.DataFrame = None):
    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("← All Hostels", key="btn_nav_overview", use_container_width=True):
            st.session_state["page"] = "overview"
            st.rerun()

    st.markdown(
        f"""
        <div class="breadcrumb-bar" style="margin-top: 4px;">
          <span>Hostel Overview</span> &nbsp;›&nbsp; <b style="color:var(--text-color);">{hostel_id}</b>
        </div>
        <div class="app-header">
          <div class="header-title-row">
            <h1 style="margin: 0; padding: 0; font-size: 1.75rem; font-weight: 800;">🏢 {hostel_id} — Floor Directory</h1>
            <span class="header-badge">Facility View</span>
          </div>
        </div>
        <p class="app-subtitle">
          4 Independent Floors &nbsp;•&nbsp; Dedicated Floor Water Meters &nbsp;•&nbsp; 4 Rooms per Floor
        </p>
        """,
        unsafe_allow_html=True,
    )

    hostel_alerts   = filter_alerts(alerts, hostel_id=hostel_id)
    hostel_readings = readings[readings["hostel_id"] == hostel_id] if not readings.empty else pd.DataFrame()

    floors = HOSTEL_FLOORS[hostel_id]
    cols   = st.columns(2)

    for i, floor_id in enumerate(floors):
        meter_id      = f"M-{floor_id}"
        floor_alerts  = hostel_alerts[hostel_alerts["floor_id"] == floor_id] if not hostel_alerts.empty else pd.DataFrame()
        floor_readings= hostel_readings[hostel_readings["floor_id"] == floor_id] if not hostel_readings.empty else pd.DataFrame()

        floor_open_alerts = floor_alerts[floor_alerts["status"] == "OPEN"] if not floor_alerts.empty else pd.DataFrame()
        f_open     = len(floor_open_alerts)
        f_resolved = int((floor_alerts["status"] == "RESOLVED").sum()) if not floor_alerts.empty else 0

        # Latest meter reading for this floor
        if not floor_readings.empty:
            latest = floor_readings.sort_values("timestamp").iloc[-1]
            last_reading = f"{latest['meter_reading']:,.1f} L"
            last_delta   = f"{latest['delta_litres']:.1f} L/hr"
        else:
            last_reading = "—"
            last_delta   = "—"

        status_text, brd, bg = get_floor_status(floor_alerts)

        with cols[i % 2]:
            alert_pill = (
                f'<span class="stat-pill stat-pill-alert">🚨 {f_open} Open Alert(s)</span>'
                if f_open > 0 else
                f'<span class="stat-pill stat-pill-normal">✅ Normal</span>'
            )
            resolved_pill = (
                f'<span class="stat-pill">Resolved: <b>{f_resolved}</b></span>'
                if f_resolved > 0 else ""
            )

            st.markdown(
                f"""
                <div class="f-card" style="border-left: 5px solid {brd};">
                  <div class="card-header-line">
                    <h3 class="card-title-text">🚰 Floor {floor_id}</h3>
                    <span style="font-size: 13px; font-weight: 600;">{status_text}</span>
                  </div>
                  <div class="card-desc-text">
                    Meter: <code>{meter_id}</code> &nbsp;•&nbsp; Rooms: {', '.join(FLOOR_ROOMS.get(floor_id, []))}
                  </div>
                  <div class="card-footer-metrics">
                    <div>Cumulative: <b>{last_reading}</b></div>
                    <div>Flow: <b>{last_delta}</b></div>
                    {alert_pill}
                    {resolved_pill}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open Floor Monitor ({floor_id}) →", key=f"btn_floor_{floor_id}", use_container_width=True):
                st.session_state["page"]   = "floor"
                st.session_state["hostel"] = hostel_id
                st.session_state["floor"]  = floor_id
                st.rerun()
            st.write("")

    # --- Interactive Hostel Comparative Telemetry Chart ---
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    st.subheader(f"📈 Hostel {hostel_id} — Multi-Floor Consumption Telemetry")
    st.caption("Interactive comparative flow across all 4 floor meters. Click floor labels in the legend to toggle/isolate lines.")

    if not hostel_readings.empty:
        col_h_r1, col_h_r2 = st.columns([3, 2])
        with col_h_r1:
            h_range = st.radio(
                "Hostel Time Range",
                options=["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"],
                horizontal=True,
                index=0,
                key=f"tr_hostel_{hostel_id}",
                label_visibility="collapsed",
            )
        h_custom_dates = None
        if h_range == "Custom Range":
            with col_h_r2:
                h_min = hostel_readings["timestamp"].min()
                h_max = hostel_readings["timestamp"].max()
                h_custom_dates = st.date_input(
                    "Select Date Range",
                    value=(h_min.date(), h_max.date()),
                    min_value=h_min.date(),
                    max_value=h_max.date(),
                    key=f"dr_hostel_{hostel_id}",
                    label_visibility="collapsed",
                )

        hostel_fig = chart_builder.build_hostel_telemetry_chart(
            hostel_readings=hostel_readings,
            hostel_id=hostel_id,
            time_range=h_range,
            custom_dates=h_custom_dates,
        )
        st.plotly_chart(
            hostel_fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
        )


# ---------------------------------------------------------------------------
# Phase 5 — Annotation renderer (called inside _render_floor per alert card)
# ---------------------------------------------------------------------------

def _render_annotations(
    alert_id: str,
    annotations_path: str,
    current_status: str = None,
    alerts_df: pd.DataFrame = None,
) -> None:
    """
    Render an expander beneath an alert card showing existing annotations
    and a form to add a new one.
    Guarantees annotations appear only inside their parent alert's current status tab.
    """
    # Guard: if current_status is provided, ensure the alert's status matches this tab
    if current_status:
        actual_status = None
        if alerts_df is not None and not alerts_df.empty:
            matching = alerts_df[alerts_df["alert_id"] == alert_id]
            if not matching.empty:
                actual_status = str(matching.iloc[0]["status"]).strip().upper()
        if not actual_status:
            alerts_path = os.path.join(_DATA_DIR, "alerts.csv")
            if os.path.exists(alerts_path):
                raw_df = alert_store.load_alerts(alerts_path)
                matching = raw_df[raw_df["alert_id"] == alert_id]
                if not matching.empty:
                    actual_status = str(matching.iloc[0]["status"]).strip().upper()

        if actual_status and actual_status != current_status.strip().upper():
            return

    all_annotations = annotation_store.load_annotations(annotations_path)
    alert_notes     = annotation_store.get_annotations_for_alert(all_annotations, alert_id)
    n               = len(alert_notes)
    is_resolved     = (current_status and current_status.strip().upper() == "RESOLVED")

    if is_resolved:
        label = f"📝 Inspection Notes ({n})"
    else:
        label = f"📝 Inspection Notes & Actions ({n})" if n else "📝 Add Inspection Note"

    form_key        = f"annotate_{alert_id}_{current_status}" if current_status else f"annotate_{alert_id}"

    with st.expander(label, expanded=False):
        # --- Display existing notes ---
        if not alert_notes.empty:
            for _, note_row in alert_notes.iterrows():
                ts     = str(note_row.get("created_at", ""))
                author = str(note_row.get("author", "Warden"))
                body   = str(note_row.get("note", ""))
                st.markdown(
                    f"""
                    <div class="note-card-entry">
                      <div class="note-header-line">👤 {author} &nbsp;•&nbsp; 🕒 {ts}</div>
                      <div class="note-body-text">{body}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif is_resolved:
            st.caption("No inspection notes recorded for this resolved incident.")

        # --- Add new note form (ONLY when alert is OPEN) ---
        if not is_resolved:
            if not alert_notes.empty:
                st.write("")
            with st.form(key=form_key):
                new_note = st.text_area(
                    "Inspection note",
                    placeholder="Log physical inspection details, maintenance findings, or plumbing repairs…",
                    label_visibility="collapsed",
                    height=75,
                )
                col_author, col_submit = st.columns([3, 1])
                with col_author:
                    author_input = st.text_input(
                        "Inspector / Author",
                        value=annotation_store.DEFAULT_AUTHOR,
                        label_visibility="collapsed",
                    )
                with col_submit:
                    submitted = st.form_submit_button("Post Note", use_container_width=True)

            if submitted:
                note_stripped = (new_note or "").strip()
                if not note_stripped:
                    st.warning("Note must not be empty.")
                else:
                    try:
                        updated = annotation_store.add_annotation(
                            all_annotations,
                            alert_id=alert_id,
                            note=note_stripped,
                            author=(author_input or annotation_store.DEFAULT_AUTHOR).strip(),
                        )
                        annotation_store.save_annotations(updated, annotations_path)
                        st.cache_data.clear()
                        st.rerun()
                    except ValueError as exc:
                        st.warning(f"Could not save note: {exc}")


def _render_floor(hostel_id: str, floor_id: str, alerts: pd.DataFrame, readings: pd.DataFrame, anomalies: pd.DataFrame = None):
    meter_id = f"M-{floor_id}"

    if anomalies is None:
        anom_path = os.path.join(_DATA_DIR, "anomalies.csv")
        anomalies = load_anomalies(anom_path)

    col_back1, col_back2 = st.columns([1, 4])
    with col_back1:
        if st.button("← Back to Hostel", key="btn_nav_hostel", use_container_width=True):
            st.session_state["page"] = "hostel"
            st.rerun()

    st.markdown(
        f"""
        <div class="breadcrumb-bar" style="margin-top: 4px;">
          <span>Hostel Overview</span> &nbsp;›&nbsp;
          <span>{hostel_id}</span> &nbsp;›&nbsp;
          <b style="color:var(--text-color);">{floor_id}</b>
        </div>
        <div class="app-header">
          <div class="header-title-row">
            <h1 style="margin: 0; padding: 0; font-size: 1.75rem; font-weight: 800;">🚰 Floor {floor_id} — Real-Time Monitor</h1>
            <span class="header-badge">Meter {meter_id}</span>
          </div>
        </div>
        <p class="app-subtitle">
          Floor-level IoT sub-meter telemetry and physical leak inspection tracking.
        </p>
        """,
        unsafe_allow_html=True,
    )

    floor_alerts   = filter_alerts(alerts, hostel_id=hostel_id, floor_id=floor_id)
    floor_readings = readings[
        (readings["hostel_id"] == hostel_id) & (readings["floor_id"] == floor_id)
    ].sort_values("timestamp") if not readings.empty else pd.DataFrame()

    # --- Summary metrics (based only on remaining OPEN alerts) ---
    floor_open = floor_alerts[floor_alerts["status"] == "OPEN"] if not floor_alerts.empty else pd.DataFrame()
    f_open = len(floor_open)
    f_high = int((floor_open["severity"] == "HIGH").sum()) if not floor_open.empty else 0
    f_med  = int((floor_open["severity"] == "MEDIUM").sum()) if not floor_open.empty else 0
    f_low  = int((floor_open["severity"] == "LOW").sum()) if not floor_open.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 Active Open Alerts", f_open)
    c2.metric("🔴 HIGH Severity",   f_high)
    c3.metric("🟡 MEDIUM Severity", f_med)
    c4.metric("🟢 LOW Severity",    f_low)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # --- Consumption chart ---
    st.subheader("📈 Hourly Consumption Telemetry")
    st.caption("Interactive IoT flow monitoring. Click legend layers to toggle, drag to zoom, double-click to reset.")

    if floor_readings.empty:
        st.info("No reading data available for this meter.")
    else:
        # Date range controls
        col_ctrl1, col_ctrl2 = st.columns([3, 2])
        with col_ctrl1:
            time_range = st.radio(
                "Filter Time Range",
                options=["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"],
                horizontal=True,
                index=0,
                key=f"tr_{meter_id}",
                label_visibility="collapsed",
            )
        custom_dates = None
        if time_range == "Custom Range":
            with col_ctrl2:
                min_ts = floor_readings["timestamp"].min()
                max_ts = floor_readings["timestamp"].max()
                custom_dates = st.date_input(
                    "Custom Date Range",
                    value=(min_ts.date(), max_ts.date()),
                    min_value=min_ts.date(),
                    max_value=max_ts.date(),
                    key=f"dr_{meter_id}",
                    label_visibility="collapsed",
                )

        fig = chart_builder.build_floor_telemetry_chart(
            readings_df=floor_readings,
            anomalies_df=anomalies,
            alerts_df=floor_alerts,
            meter_id=meter_id,
            time_range=time_range,
            custom_dates=custom_dates,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
        )

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # --- Alerts table with OPEN / RESOLVED tabs ---
    st.subheader("🚨 Floor Leak Alerts & Incident Management")
    st.caption("Floor meters detect anomalous consumption streaks. Exact leak sources inside rooms require physical inspection.")

    alerts_path = os.path.join(_DATA_DIR, "alerts.csv")
    annotations_path = os.path.join(_DATA_DIR, "annotations.csv")

    # Deduplicate floor alerts to guarantee no alert_id appears more than once
    if not floor_alerts.empty:
        floor_alerts = floor_alerts.drop_duplicates(subset=["alert_id"], keep="last")

    tab_open, tab_resolved = st.tabs(["🔴 OPEN INCIDENTS", "✅ RESOLVED HISTORY"])

    with tab_open:
        open_alerts = floor_alerts[floor_alerts["status"] == "OPEN"] if not floor_alerts.empty else pd.DataFrame()
        if open_alerts.empty:
            st.success("✅ No open alerts for this floor. Flow is within normal baseline parameters.")
        else:
            for _, row in open_alerts.iterrows():
                style = get_alert_card_style(row)
                with st.container():
                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        z_val = float(row['avg_z_score']) if pd.notna(row['avg_z_score']) else 0.0
                        hrs_val = int(row['persistence_count']) if pd.notna(row['persistence_count']) else 0
                        st.markdown(
                            f"""
                            <div class="alert-item-card is-open" style="border-left: 5px solid {style['border_color']};">
                              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <div>
                                  {style['badge_html']}
                                  &nbsp;&nbsp;<b style="font-size: 14px; color: var(--text-color);">Abnormal Flow Streak Detected</b>
                                </div>
                                <span class="stat-pill">Meter <code>{meter_id}</code></span>
                              </div>
                              <div style="display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--text-color); opacity: 0.9; margin-top: 4px;">
                                <div>📅 <b>{row["first_seen"].strftime("%d %b %Y, %H:%M")}</b> &nbsp;→&nbsp; <b>{row["last_seen"].strftime("%d %b %Y, %H:%M")}</b></div>
                                <div>⚡ Z-score: <b>{z_val:.1f}</b></div>
                                <div>⏱️ Persistence: <b>{hrs_val} hrs</b></div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_btn:
                        st.write("")
                        if st.button("Resolve ✓", key=f"resolve_{row['alert_id']}", type="primary", use_container_width=True):
                            raw_df = alert_store.load_alerts(alerts_path)
                            target_id = alert_store.find_matching_alert_id(
                                raw_df,
                                alert_id=row.get("alert_id"),
                                meter_id=row.get("meter_id"),
                                first_seen=row.get("first_seen"),
                                last_seen=row.get("last_seen"),
                            )
                            if not target_id:
                                st.warning("Alert could not be found. The data may have refreshed.")
                            else:
                                try:
                                    raw_df = alert_store.resolve_alert(raw_df, target_id)
                                    alert_store.save_alerts(raw_df, alerts_path)
                                except ValueError as e:
                                    if "already RESOLVED" in str(e):
                                        st.info("Alert is already resolved.")
                                    else:
                                        st.warning(f"Could not resolve alert: {e}")
                            st.cache_data.clear()
                            st.rerun()

                    _render_annotations(row["alert_id"], annotations_path, current_status="OPEN", alerts_df=floor_alerts)

    with tab_resolved:
        resolved_alerts = floor_alerts[floor_alerts["status"] == "RESOLVED"] if not floor_alerts.empty else pd.DataFrame()
        if not open_alerts.empty and not resolved_alerts.empty:
            resolved_alerts = resolved_alerts[~resolved_alerts["alert_id"].isin(open_alerts["alert_id"])]

        if resolved_alerts.empty:
            st.info("No resolved alerts on record for this floor.")
        else:
            for _, row in resolved_alerts.iterrows():
                style = get_alert_card_style(row)
                with st.container():
                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        resolved_ts = row.get("resolved_at", "")
                        resolved_str = f"Resolved: {resolved_ts}" if resolved_ts else "Resolved"
                        st.markdown(
                            f"""
                            <div class="alert-item-card is-resolved" style="border-left: 5px solid {style['border_color']};">
                              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <div>
                                  {style['badge_html']}
                                  &nbsp;&nbsp;<span class="stat-pill">{row["severity"]}</span>
                                </div>
                                <span class="stat-pill">Meter <code>{meter_id}</code></span>
                              </div>
                              <div style="display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--text-color); opacity: 0.9; margin-top: 4px;">
                                <div>📅 <b>{row["first_seen"].strftime("%d %b %Y, %H:%M")}</b> &nbsp;→&nbsp; <b>{row["last_seen"].strftime("%d %b %Y, %H:%M")}</b></div>
                                <div>✅ <b>{resolved_str}</b></div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_btn:
                        st.write("")
                        if st.button("Reopen ↩", key=f"reopen_{row['alert_id']}", use_container_width=True):
                            raw_df = alert_store.load_alerts(alerts_path)
                            target_id = alert_store.find_matching_alert_id(
                                raw_df,
                                alert_id=row.get("alert_id"),
                                meter_id=row.get("meter_id"),
                                first_seen=row.get("first_seen"),
                                last_seen=row.get("last_seen"),
                            )
                            if not target_id:
                                st.warning("Alert could not be found. The data may have refreshed.")
                            else:
                                try:
                                    raw_df = alert_store.reopen_alert(raw_df, target_id)
                                    alert_store.save_alerts(raw_df, alerts_path)
                                except ValueError as e:
                                    if "already OPEN" in str(e):
                                        st.info("Alert is already open.")
                                    else:
                                        st.warning(f"Could not reopen alert: {e}")
                            st.cache_data.clear()
                            st.rerun()

                    _render_annotations(row["alert_id"], annotations_path, current_status="RESOLVED", alerts_df=floor_alerts)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # --- Room directory ---
    st.subheader("🏠 Student Room Directory")
    st.caption(f"Inspection zones connected to meter {meter_id} on {floor_id}")
    rooms = FLOOR_ROOMS.get(floor_id, [])
    if rooms:
        rcols = st.columns(len(rooms))
        for i, room in enumerate(rooms):
            rcols[i].markdown(
                f"""
                <div class="room-box">
                  <div style="font-size: 22px;">🛏️</div>
                  <div class="room-num">Room {room}</div>
                  <div class="room-sub">{floor_id} Sector</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write("No room data available.")


# ---------------------------------------------------------------------------
# Phase 4 helpers — preserved for unit test suite compatibility
# ---------------------------------------------------------------------------

def get_latest_reading_time(readings: pd.DataFrame) -> datetime | None:
    """Return the most recent timestamp across all meters, or None if empty."""
    if readings.empty or "timestamp" not in readings.columns:
        return None
    ts = readings["timestamp"].max()
    return ts.to_pydatetime() if not pd.isna(ts) else None


def get_stale_meters(
    readings: pd.DataFrame,
    stale_threshold_seconds: float,
) -> list[str]:
    """
    Return meter IDs whose last reading is older than stale_threshold_seconds
    relative to the most-recent reading in the dataset.
    """
    if readings.empty:
        return list(METER_IDS)

    latest_ts = readings["timestamp"].max()

    stale = []
    for mid in METER_IDS:
        meter_rows = readings[readings["meter_id"] == mid]
        if meter_rows.empty:
            stale.append(mid)
        else:
            last_ts = meter_rows["timestamp"].max()
            gap_secs = (latest_ts - last_ts).total_seconds()
            if gap_secs > stale_threshold_seconds:
                stale.append(mid)
    return stale


def get_alert_counts(alerts: pd.DataFrame) -> dict:
    """
    Return a summary dict:
        open_total, open_high, open_medium, open_low, resolved_total
    """
    if alerts.empty:
        return {"open_total": 0, "open_high": 0,
                "open_medium": 0, "open_low": 0, "resolved_total": 0}

    open_df     = alerts[alerts["status"] == "OPEN"]
    resolved_df = alerts[alerts["status"] == "RESOLVED"]
    return {
        "open_total":    int(len(open_df)),
        "open_high":     int((open_df["severity"] == "HIGH").sum()),
        "open_medium":   int((open_df["severity"] == "MEDIUM").sum()),
        "open_low":      int((open_df["severity"] == "LOW").sum()),
        "resolved_total": int(len(resolved_df)),
    }


def _render_status_bar(
    readings: pd.DataFrame,
    alerts: pd.DataFrame,
    refresh_secs: int,
) -> None:
    """Status bar function preserved for backward test compatibility."""
    latest_ts   = get_latest_reading_time(readings)
    counts      = get_alert_counts(alerts)
    stale_meters = get_stale_meters(
        readings,
        stale_threshold_seconds=refresh_secs * STALE_MULTIPLIER,
    )

    if latest_ts is not None:
        ts_str = latest_ts.strftime("%d %b %Y %H:%M:%S")
        stale_note = ""
        if stale_meters:
            stale_note = (
                f" &nbsp;|&nbsp; "
                f'<span style="color:#dc2626;font-weight:600">'
                f"⚠ {len(stale_meters)} meter(s) stale: {', '.join(stale_meters)}"
                f"</span>"
            )
        st.markdown(
            f'<p style="font-size:13px;color:var(--text-color);margin:0">'
            f"🕐 Latest reading: <b>{ts_str}</b>{stale_note}"
            f"</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="font-size:13px;color:var(--text-color);opacity:0.6;margin:0">'
            "🕐 No readings available yet."
            "</p>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Alerts",    counts["open_total"])
    c2.metric("🔴 HIGH",        counts["open_high"])
    c3.metric("🟡 MEDIUM",      counts["open_medium"])
    c4.metric("🟢 LOW",         counts["open_low"])
    c5.metric("✅ Resolved",    counts["resolved_total"])


# ---------------------------------------------------------------------------
# Dashboard entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Campus Water Leak Detection — Warden Portal",
        page_icon="🚰",
        layout="wide",
    )

    _inject_custom_css()

    # Initialise navigation state
    if "page" not in st.session_state:
        st.session_state["page"] = "overview"
    if "hostel" not in st.session_state:
        st.session_state["hostel"] = None
    if "floor" not in st.session_state:
        st.session_state["floor"] = None

    # --- File paths ---
    alerts_path    = os.path.join(_DATA_DIR, "alerts.csv")
    readings_path  = os.path.join(_DATA_DIR, "processed_readings.csv")
    anomalies_path = os.path.join(_DATA_DIR, "anomalies.csv")

    # Direct manual dashboard loading
    alerts    = load_alerts(alerts_path)
    readings  = load_readings(readings_path)
    anomalies = load_anomalies(anomalies_path)

    # --- Route to correct page ---
    page = st.session_state["page"]

    if page == "overview":
        _render_overview(alerts, readings)

    elif page == "hostel":
        hostel_id = st.session_state.get("hostel")
        if hostel_id:
            _render_hostel(hostel_id, alerts, readings, anomalies=anomalies)
        else:
            st.session_state["page"] = "overview"
            st.rerun()

    elif page == "floor":
        hostel_id = st.session_state.get("hostel")
        floor_id  = st.session_state.get("floor")
        if hostel_id and floor_id:
            _render_floor(hostel_id, floor_id, alerts, readings, anomalies=anomalies)
        else:
            st.session_state["page"] = "overview"
            st.rerun()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
