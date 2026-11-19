# config.py — Shared constants. Read-only for all modules. Owned by Architect.

# ---------------------------------------------------------------------------
# Hostel / Floor / Meter structure
# 3 hostels × 4 floors each = 12 independent floor-level water meters
# ---------------------------------------------------------------------------

# Each entry: hostel_id -> list of floor_ids
HOSTEL_FLOORS = {
    "H1": ["H1-F1", "H1-F2", "H1-F3", "H1-F4"],
    "H2": ["H2-F1", "H2-F2", "H2-F3", "H2-F4"],
    "H3": ["H3-F1", "H3-F2", "H3-F3", "H3-F4"],
}

# Meter ID convention: M-<floor_id>  (one meter per floor)
# Full flat list of all 12 meters
METERS = [
    {"hostel_id": "H1", "floor_id": "H1-F1", "meter_id": "M-H1-F1"},
    {"hostel_id": "H1", "floor_id": "H1-F2", "meter_id": "M-H1-F2"},
    {"hostel_id": "H1", "floor_id": "H1-F3", "meter_id": "M-H1-F3"},
    {"hostel_id": "H1", "floor_id": "H1-F4", "meter_id": "M-H1-F4"},
    {"hostel_id": "H2", "floor_id": "H2-F1", "meter_id": "M-H2-F1"},
    {"hostel_id": "H2", "floor_id": "H2-F2", "meter_id": "M-H2-F2"},
    {"hostel_id": "H2", "floor_id": "H2-F3", "meter_id": "M-H2-F3"},
    {"hostel_id": "H2", "floor_id": "H2-F4", "meter_id": "M-H2-F4"},
    {"hostel_id": "H3", "floor_id": "H3-F1", "meter_id": "M-H3-F1"},
    {"hostel_id": "H3", "floor_id": "H3-F2", "meter_id": "M-H3-F2"},
    {"hostel_id": "H3", "floor_id": "H3-F3", "meter_id": "M-H3-F3"},
    {"hostel_id": "H3", "floor_id": "H3-F4", "meter_id": "M-H3-F4"},
]

# Convenience: flat list of meter IDs
METER_IDS = [m["meter_id"] for m in METERS]

# ---------------------------------------------------------------------------
# Room directory (organisational only — rooms are NOT individually metered)
# Format: floor_id -> list of room names
# ---------------------------------------------------------------------------
FLOOR_ROOMS = {
    "H1-F1": ["101A", "101B", "101C", "101D"],
    "H1-F2": ["102A", "102B", "102C", "102D"],
    "H1-F3": ["103A", "103B", "103C", "103D"],
    "H1-F4": ["104A", "104B", "104C", "104D"],
    "H2-F1": ["201A", "201B", "201C", "201D"],
    "H2-F2": ["202A", "202B", "202C", "202D"],
    "H2-F3": ["203A", "203B", "203C", "203D"],
    "H2-F4": ["204A", "204B", "204C", "204D"],
    "H3-F1": ["301A", "301B", "301C", "301D"],
    "H3-F2": ["302A", "302B", "302C", "302D"],
    "H3-F3": ["303A", "303B", "303C", "303D"],
    "H3-F4": ["304A", "304B", "304C", "304D"],
}

# ---------------------------------------------------------------------------
# Simulation window
# ---------------------------------------------------------------------------
START_DATE            = "2024-01-01"   # ISO date
END_DATE              = "2024-01-31"   # ISO date (31 days)
READING_INTERVAL_MINS = 60             # one reading per hour per meter

# ---------------------------------------------------------------------------
# Leak injection — one meter per hostel, spread across different floors
# ---------------------------------------------------------------------------
LEAK_INJECT_METERS = ["M-H1-F2", "M-H2-F3", "M-H3-F4"]

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
MAD_Z_THRESHOLD = 3.5   # modified Z-score cutoff for anomaly
PERSISTENCE_MIN = 3     # consecutive anomalous readings needed to raise alert

# ---------------------------------------------------------------------------
# Phase 3 — Rolling window baseline parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS  = 7    # days of prior history used to build each baseline
ROLLING_MIN_READINGS = 10   # minimum prior readings before switching to rolling
                             # (falls back to global MAD baseline when below this)

# ---------------------------------------------------------------------------
# Alert severity bands (based on avg modified Z-score over the streak)
# ---------------------------------------------------------------------------
SEVERITY_LOW_MAX_Z    = 5.0    # z <= 5  → LOW
SEVERITY_MEDIUM_MAX_Z = 10.0   # z <= 10 → MEDIUM  (above → HIGH)
