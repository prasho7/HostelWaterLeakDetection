"""
chart_builder.py — Interactive Plotly telemetry chart generators for the Hostel Water Leak Detector.

Features:
- Hover inspections: Timestamp, consumption (L/hr), meter ID, MAD Z-score.
- Zoom, pan, autoscale and double-click reset.
- Dynamic date-range controls: 24 Hours, 7 Days, 30 Days, Custom Range, All Time.
- Separate distinct visual layers:
  1. Water Flow (L/hr)
  2. Baseline (Median Flow)
  3. Detection Threshold (MAD Z=3.5 Cutoff)
  4. Unconfirmed Anomalies (isolated single/double spikes < 3 hrs)
  5. Confirmed Leak Alert Readings (streaks >= 3 hrs)
  6. Shaded Alert Windows (OPEN in red, RESOLVED in green)
- Legend layer toggling (click to hide/show, double-click to isolate).
- High-contrast, theme-adaptive styling compatible with dark and light themes.
"""

from typing import Tuple, Optional, List
import os
import sys
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import MAD_Z_THRESHOLD, HOSTEL_FLOORS


# ---------------------------------------------------------------------------
# Theme Colors & Styling Constants
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "HIGH": "#dc2626",
    "MEDIUM": "#d97706",
    "LOW": "#16a34a",
    "RESOLVED": "#16a34a",
}

FLOOR_COLORS = [
    "#0284c7",  # Floor 1: Sky Blue
    "#8b5cf6",  # Floor 2: Violet/Purple
    "#10b981",  # Floor 3: Emerald Green
    "#f59e0b",  # Floor 4: Amber/Orange
]


def filter_by_date_range(
    df: pd.DataFrame,
    time_range: str = "All Time",
    custom_dates: Optional[Tuple] = None,
) -> pd.DataFrame:
    """Filter a DataFrame by timestamp according to the selected range."""
    if df.empty or "timestamp" not in df.columns:
        return df

    max_ts = df["timestamp"].max()
    if pd.isna(max_ts):
        return df

    if time_range == "Last 24 Hours":
        cutoff = max_ts - pd.Timedelta(hours=24)
        return df[df["timestamp"] >= cutoff].copy()
    elif time_range == "Last 7 Days":
        cutoff = max_ts - pd.Timedelta(days=7)
        return df[df["timestamp"] >= cutoff].copy()
    elif time_range == "Last 30 Days":
        cutoff = max_ts - pd.Timedelta(days=30)
        return df[df["timestamp"] >= cutoff].copy()
    elif time_range == "Custom Range" and custom_dates and len(custom_dates) == 2:
        try:
            start_ts = pd.to_datetime(custom_dates[0])
            end_ts = pd.to_datetime(custom_dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            return df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
        except Exception:
            return df
    return df.copy()


def build_floor_telemetry_chart(
    readings_df: pd.DataFrame,
    anomalies_df: Optional[pd.DataFrame],
    alerts_df: Optional[pd.DataFrame],
    meter_id: str,
    time_range: str = "All Time",
    custom_dates: Optional[Tuple] = None,
) -> go.Figure:
    """
    Generate an interactive Plotly chart for a single floor sub-meter.
    Includes baseline, threshold, unconfirmed anomalies, confirmed alerts,
    shaded alert spans, and date-range controls.
    """
    fig = go.Figure()

    if readings_df.empty:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text="No telemetry readings available for this meter.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#94a3b8")
            )]
        )
        return fig

    # Filter readings for this meter
    m_readings = readings_df[readings_df["meter_id"] == meter_id].sort_values("timestamp").copy()
    if m_readings.empty:
        m_readings = readings_df.sort_values("timestamp").copy()

    # Join with anomalies data if available
    if anomalies_df is not None and not anomalies_df.empty:
        m_anom = anomalies_df[anomalies_df["meter_id"] == meter_id].sort_values("timestamp").copy()
        if not m_anom.empty:
            merged = pd.merge(
                m_readings,
                m_anom[["timestamp", "median_baseline", "mad", "z_score_mad", "is_anomaly", "persistence_count"]],
                on="timestamp",
                how="left",
            )
        else:
            merged = m_readings.copy()
            merged["median_baseline"] = merged["delta_litres"].median()
            merged["mad"] = 1.0
            merged["z_score_mad"] = 0.0
            merged["is_anomaly"] = False
            merged["persistence_count"] = 0
    else:
        merged = m_readings.copy()
        merged["median_baseline"] = merged["delta_litres"].median()
        merged["mad"] = 1.0
        merged["z_score_mad"] = 0.0
        merged["is_anomaly"] = False
        merged["persistence_count"] = 0

    # Fill NaN values in anomaly columns
    if "median_baseline" not in merged.columns or merged["median_baseline"].isna().all():
        merged["median_baseline"] = merged["delta_litres"].median()
    merged["median_baseline"] = merged["median_baseline"].fillna(merged["delta_litres"].median())

    if "mad" not in merged.columns or merged["mad"].isna().all():
        merged["mad"] = 1.0
    merged["mad"] = merged["mad"].fillna(1.0)

    if "z_score_mad" not in merged.columns:
        merged["z_score_mad"] = 0.0
    merged["z_score_mad"] = merged["z_score_mad"].fillna(0.0)

    if "is_anomaly" not in merged.columns:
        merged["is_anomaly"] = False
    merged["is_anomaly"] = merged["is_anomaly"].fillna(False).astype(bool)

    # Compute hourly detection threshold flow: baseline + (3.5 * mad / 0.6745)
    merged["detection_threshold"] = merged["median_baseline"] + (MAD_Z_THRESHOLD * merged["mad"] / 0.6745)

    # Filter alerts for this meter
    m_alerts = alerts_df[alerts_df["meter_id"] == meter_id].copy() if (alerts_df is not None and not alerts_df.empty) else pd.DataFrame()

    # Identify readings that belong to confirmed leak alert windows
    confirmed_mask = pd.Series(False, index=merged.index)
    if not m_alerts.empty:
        for _, alt in m_alerts.iterrows():
            f_seen = pd.to_datetime(alt["first_seen"])
            l_seen = pd.to_datetime(alt["last_seen"])
            confirmed_mask |= (merged["timestamp"] >= f_seen) & (merged["timestamp"] <= l_seen)

    merged["is_confirmed_leak"] = confirmed_mask
    merged["is_unconfirmed_anomaly"] = merged["is_anomaly"] & (~confirmed_mask)

    # Apply date range filtering
    filtered = filter_by_date_range(merged, time_range=time_range, custom_dates=custom_dates)
    if filtered.empty:
        filtered = merged

    # 1. Base Water Flow line
    fig.add_trace(go.Scatter(
        x=filtered["timestamp"],
        y=filtered["delta_litres"],
        name="Water Flow (L/hr)",
        mode="lines",
        line=dict(color="#0284c7", width=2),
        customdata=list(zip(filtered["meter_id"], filtered["z_score_mad"])),
        hovertemplate=(
            "<b>💧 Water Flow:</b> %{y:.1f} L/hr<br>"
            "<b>🕒 Time:</b> %{x|%d %b %Y, %H:%M}<br>"
            "<b>📟 Meter:</b> %{customdata[0]}<br>"
            "<b>⚡ MAD Z-Score:</b> %{customdata[1]:.2f}<br>"
            "<extra></extra>"
        ),
    ))

    # 2. Baseline reference line (median)
    fig.add_trace(go.Scatter(
        x=filtered["timestamp"],
        y=filtered["median_baseline"],
        name="Baseline (Median)",
        mode="lines",
        line=dict(color="#94a3b8", width=1.4, dash="dot"),
        hovertemplate=(
            "<b>📊 Baseline Flow:</b> %{y:.1f} L/hr<br>"
            "<b>🕒 Time:</b> %{x|%d %b %Y, %H:%M}<br>"
            "<extra></extra>"
        ),
    ))

    # 3. Detection Threshold line (Z = 3.5 cutoff)
    fig.add_trace(go.Scatter(
        x=filtered["timestamp"],
        y=filtered["detection_threshold"],
        name="Threshold (Z=3.5)",
        mode="lines",
        line=dict(color="#f59e0b", width=1.2, dash="dash"),
        hovertemplate=(
            "<b>⚠️ Anomaly Threshold:</b> %{y:.1f} L/hr<br>"
            "<b>🕒 Time:</b> %{x|%d %b %Y, %H:%M}<br>"
            "<extra></extra>"
        ),
    ))

    # 4. Unconfirmed Anomalies (Spikes < 3 hrs)
    unconf = filtered[filtered["is_unconfirmed_anomaly"]]
    if not unconf.empty:
        fig.add_trace(go.Scatter(
            x=unconf["timestamp"],
            y=unconf["delta_litres"],
            name="Unconfirmed Anomalies (Spikes < 3h)",
            mode="markers",
            marker=dict(symbol="triangle-up", size=9, color="#f97316", line=dict(width=1.2, color="#7c2d12")),
            customdata=list(zip(unconf["meter_id"], unconf["z_score_mad"])),
            hovertemplate=(
                "<b>⚠️ Unconfirmed Anomaly</b><br>"
                "<b>💧 Flow:</b> %{y:.1f} L/hr<br>"
                "<b>🕒 Time:</b> %{x|%d %b %Y, %H:%M}<br>"
                "<b>📟 Meter:</b> %{customdata[0]}<br>"
                "<b>⚡ MAD Z-Score:</b> %{customdata[1]:.2f} (Transient Spike)<br>"
                "<extra></extra>"
            ),
        ))

    # 5. Confirmed Leak Alert Readings (Sustained streaks >= 3 hrs)
    conf = filtered[filtered["is_confirmed_leak"]]
    if not conf.empty:
        fig.add_trace(go.Scatter(
            x=conf["timestamp"],
            y=conf["delta_litres"],
            name="Confirmed Leak Alert Readings",
            mode="markers",
            marker=dict(symbol="diamond", size=9, color="#dc2626", line=dict(width=1.2, color="#7f1d1d")),
            customdata=list(zip(conf["meter_id"], conf["z_score_mad"])),
            hovertemplate=(
                "<b>🚨 Confirmed Leak Flow</b><br>"
                "<b>💧 Flow:</b> %{y:.1f} L/hr<br>"
                "<b>🕒 Time:</b> %{x|%d %b %Y, %H:%M}<br>"
                "<b>📟 Meter:</b> %{customdata[0]}<br>"
                "<b>⚡ MAD Z-Score:</b> %{customdata[1]:.2f} (Sustained Leak)<br>"
                "<extra></extra>"
            ),
        ))

    # 6. Shaded Alert Windows (vertical regions)
    if not m_alerts.empty:
        for _, alt in m_alerts.iterrows():
            f_seen = pd.to_datetime(alt["first_seen"])
            l_seen = pd.to_datetime(alt["last_seen"])
            # Only add if overlapping the current view
            if l_seen < filtered["timestamp"].min() or f_seen > filtered["timestamp"].max():
                continue

            is_res = (str(alt.get("status", "")).strip().upper() == "RESOLVED")
            fill_col = "rgba(22, 163, 74, 0.15)" if is_res else "rgba(220, 38, 38, 0.16)"
            line_col = "rgba(22, 163, 74, 0.6)" if is_res else "rgba(220, 38, 38, 0.6)"
            sev = alt.get("severity", "ALERT")
            annot = f"✅ Resolved ({sev})" if is_res else f"🚨 {sev} Leak"

            fig.add_vrect(
                x0=f_seen,
                x1=l_seen,
                fillcolor=fill_col,
                line=dict(color=line_col, width=1, dash="dot"),
                annotation_text=annot,
                annotation_position="top left",
                annotation_font=dict(size=10, color="#16a34a" if is_res else "#dc2626"),
            )

    # Polished layout
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=25, t=40, b=35),
        height=380,
        hovermode="closest",
        dragmode="zoom",
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.94)",
            font_size=12,
            font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            font_color="#f8fafc",
            bordercolor="rgba(148, 163, 184, 0.35)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            zeroline=False,
            tickfont=dict(size=10, color="#94a3b8"),
            linecolor="rgba(148, 163, 184, 0.3)",
            rangeselector=dict(
                buttons=list([
                    dict(count=24, label="24h", step="hour", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ]),
                bgcolor="rgba(128, 128, 128, 0.12)",
                activecolor="rgba(2, 132, 199, 0.35)",
                font=dict(size=10, color="#94a3b8"),
                x=0,
                y=1.03,
                yanchor="bottom",
            ),
        ),
        yaxis=dict(
            title=dict(text="Flow Rate (Litres / hr)", font=dict(size=11, color="#94a3b8")),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.25)",
            tickfont=dict(size=10, color="#94a3b8"),
            linecolor="rgba(148, 163, 184, 0.3)",
        ),
    )

    return fig


def build_hostel_telemetry_chart(
    hostel_readings: pd.DataFrame,
    hostel_id: str,
    time_range: str = "All Time",
    custom_dates: Optional[Tuple] = None,
) -> go.Figure:
    """
    Generate an interactive comparative Plotly chart showing all 4 floor meters in a hostel.
    Allows toggling floors via legend and hovering to inspect flow rates.
    """
    fig = go.Figure()

    if hostel_readings.empty:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text="No telemetry readings available for this hostel.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#94a3b8")
            )]
        )
        return fig

    filtered = filter_by_date_range(hostel_readings, time_range=time_range, custom_dates=custom_dates)
    if filtered.empty:
        filtered = hostel_readings

    floors = HOSTEL_FLOORS.get(hostel_id, [])
    for idx, floor_id in enumerate(floors):
        meter_id = f"M-{floor_id}"
        f_data = filtered[filtered["floor_id"] == floor_id].sort_values("timestamp")
        if f_data.empty:
            continue

        color = FLOOR_COLORS[idx % len(FLOOR_COLORS)]
        fig.add_trace(go.Scatter(
            x=f_data["timestamp"],
            y=f_data["delta_litres"],
            name=f"Floor {floor_id} ({meter_id})",
            mode="lines",
            line=dict(color=color, width=1.8),
            customdata=list(zip(f_data["floor_id"], f_data["meter_id"])),
            hovertemplate=(
                f"<b>🏢 Floor {floor_id}:</b> %{{y:.1f}} L/hr<br>"
                "<b>🕒 Time:</b> %{x|%d %b %Y, %H:%M}<br>"
                "<b>📟 Meter:</b> %{customdata[1]}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=25, t=40, b=35),
        height=350,
        hovermode="closest",
        dragmode="zoom",
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.94)",
            font_size=12,
            font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            font_color="#f8fafc",
            bordercolor="rgba(148, 163, 184, 0.35)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            zeroline=False,
            tickfont=dict(size=10, color="#94a3b8"),
            linecolor="rgba(148, 163, 184, 0.3)",
            rangeselector=dict(
                buttons=list([
                    dict(count=24, label="24h", step="hour", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ]),
                bgcolor="rgba(128, 128, 128, 0.12)",
                activecolor="rgba(2, 132, 199, 0.35)",
                font=dict(size=10, color="#94a3b8"),
                x=0,
                y=1.03,
                yanchor="bottom",
            ),
        ),
        yaxis=dict(
            title=dict(text="Flow Rate (Litres / hr)", font=dict(size=11, color="#94a3b8")),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.25)",
            tickfont=dict(size=10, color="#94a3b8"),
            linecolor="rgba(148, 163, 184, 0.3)",
        ),
    )

    return fig
