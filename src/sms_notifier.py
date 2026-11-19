"""
sms_notifier.py — Twilio SMS Notification Module for Water Leak Alerts

Sends automated SMS notifications to the warden when newly generated
HIGH-severity leak alerts occur.

Features:
- Triggers only on new HIGH-severity alerts.
- Formats detailed alerts with ID, hostel, floor, meter, severity, timestamp, and description.
- Prevents duplicate SMS for the same alert via persistent notification tracking.
- Ignores LOW and MEDIUM severity alerts.
- Configurable via environment variables (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
  TWILIO_PHONE_NUMBER, WARDEN_PHONE_NUMBER).
- Never hardcodes or exposes credentials in logs or code.
- Handles API failures gracefully without interrupting pipeline processing.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Project path resolution
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DEFAULT_SENT_STORE_PATH = os.path.join(DEFAULT_DATA_DIR, ".sent_notifications.json")

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("sms_notifier")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------
def load_env(path: str | None = None) -> None:
    """Load key-value pairs from .env file into os.environ if not already present."""
    if path is None:
        path = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        logger.debug("Failed reading .env file: %s", exc)


# ---------------------------------------------------------------------------
# Security and logging helpers
# ---------------------------------------------------------------------------
def mask_phone_number(phone: str) -> str:
    """Mask phone number for safe logging (e.g. +12345678901 -> +123****8901)."""
    if not phone:
        return "N/A"
    p = str(phone).strip()
    if len(p) <= 6:
        return "****"
    return p[:4] + "*" * (len(p) - 8) + p[-4:] if len(p) >= 10 else p[:2] + "****" + p[-2:]


def sanitize_message(msg: str, secrets: list[str]) -> str:
    """Remove any credentials or tokens from error text before logging."""
    sanitized = str(msg)
    for s in secrets:
        if s and len(s) >= 4 and s in sanitized:
            sanitized = sanitized.replace(s, "[REDACTED]")
    return sanitized


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------
def format_alert_sms(alert: dict | pd.Series) -> str:
    """
    Format a concise, informative SMS message body for a leak alert.
    Includes alert ID, hostel, floor, meter, severity, timestamp, and description.
    """
    alert_id    = str(alert.get("alert_id") or "UNKNOWN").strip()
    hostel_id   = str(alert.get("hostel_id") or "N/A").strip()
    floor_id    = str(alert.get("floor_id") or "N/A").strip()
    meter_id    = str(alert.get("meter_id") or "N/A").strip()
    severity    = str(alert.get("severity") or "HIGH").strip().upper()
    timestamp   = str(alert.get("last_seen") or alert.get("first_seen") or "N/A").strip()
    persistence = alert.get("persistence_count")
    z_score     = alert.get("avg_z_score")

    details = []
    if persistence:
        details.append(f"{persistence} hrs persistence")
    if z_score is not None and str(z_score).strip() != "":
        try:
            details.append(f"avg Z-score: {float(z_score):.1f}")
        except (ValueError, TypeError):
            pass

    detail_str = f" ({', '.join(details)})" if details else ""

    message = (
        f"🚨 WATER LEAK ALERT: {severity} 🚨\n"
        f"Alert ID: {alert_id}\n"
        f"Hostel: {hostel_id} | Floor: {floor_id} | Meter: {meter_id}\n"
        f"Severity: {severity}\n"
        f"Timestamp: {timestamp}\n"
        f"Description: Sustained abnormal water flow detected{detail_str}. Immediate physical inspection required."
    )
    return message


# ---------------------------------------------------------------------------
# Sent notification persistence (Duplicate Prevention)
# ---------------------------------------------------------------------------
def load_sent_notifications(path: str = DEFAULT_SENT_STORE_PATH) -> set[str]:
    """Return the set of alert IDs that have already been notified."""
    if not os.path.isfile(path):
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return set(str(item) for item in data)
            elif isinstance(data, dict):
                return set(str(k) for k in data.keys())
    except Exception as exc:
        logger.warning("Could not read sent notifications file: %s", exc)

    return set()


def record_sent_notification(
    alert_id: str,
    path: str = DEFAULT_SENT_STORE_PATH,
    metadata: dict | None = None,
) -> None:
    """Record an alert ID in the persistent sent notifications store."""
    if not alert_id:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_records: dict[str, Any] = {}

    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_records = loaded
                elif isinstance(loaded, list):
                    existing_records = {str(item): {"sent_at": ""} for item in loaded}
        except Exception:
            existing_records = {}

    entry = {
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry.update(metadata)

    existing_records[str(alert_id)] = entry

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=2)
    except Exception as exc:
        logger.warning("Could not write to sent notifications file: %s", exc)


# ---------------------------------------------------------------------------
# SMS Notifier Class
# ---------------------------------------------------------------------------
class SMSNotifier:
    """Manages Twilio SMS notifications for leak alerts."""

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_phone: str | None = None,
        to_phone: str | None = None,
        sent_store_path: str | None = None,
        twilio_client: Any | None = None,
    ):
        load_env()
        self.account_sid = (account_sid or os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
        self.auth_token  = (auth_token or os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
        self.from_phone  = (from_phone or os.getenv("TWILIO_PHONE_NUMBER") or os.getenv("TWILIO_FROM_NUMBER") or "").strip()
        self.to_phone    = (to_phone or os.getenv("WARDEN_PHONE_NUMBER") or os.getenv("WARDEN_PHONE") or "").strip()

        self.sent_store_path = sent_store_path or DEFAULT_SENT_STORE_PATH
        self._client = twilio_client

    def is_configured(self) -> bool:
        """Return True if all required Twilio credentials and phone numbers are present."""
        return bool(self.account_sid and self.auth_token and self.from_phone and self.to_phone)

    @property
    def client(self) -> Any:
        """Lazy-initialize official Twilio Client when needed."""
        if self._client is None and self.is_configured():
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
            except Exception as exc:
                logger.error("Failed to initialize Twilio client: %s", exc)
                return None
        return self._client

    def is_alert_notified(self, alert_id: str) -> bool:
        """Check whether alert_id has already been notified."""
        sent_set = load_sent_notifications(self.sent_store_path)
        return str(alert_id) in sent_set

    def send_alert(
        self,
        alert: dict | pd.Series,
        force: bool = False,
    ) -> bool:
        """
        Send an SMS notification for an alert if it is HIGH severity and not previously notified.

        Parameters
        ----------
        alert : alert dict or pandas Series
        force : if True, bypass duplicate check (useful for manual re-sends)

        Returns
        -------
        bool : True if SMS was sent successfully, False otherwise.
        """
        if isinstance(alert, pd.Series):
            alert_dict = alert.to_dict()
        elif isinstance(alert, dict):
            alert_dict = alert
        else:
            logger.warning("Invalid alert data type: %s", type(alert))
            return False

        alert_id = str(alert_dict.get("alert_id") or "").strip()
        severity = str(alert_dict.get("severity") or "").strip().upper()

        # Rule 1 & 5: Only HIGH severity alerts trigger SMS
        if severity != "HIGH":
            logger.debug("Skipping SMS for alert %s (severity: %s; only HIGH triggers SMS)", alert_id, severity)
            return False

        # Rule 3 & 4: Prevent duplicate SMS for the same alert
        if not force and self.is_alert_notified(alert_id):
            logger.info("Skipping SMS for alert %s (already notified)", alert_id)
            return False

        # Rule 8: Verify configuration without throwing
        if not self.is_configured():
            logger.warning(
                "Twilio SMS not configured (missing environment variables). "
                "Skipping SMS notification for alert %s.", alert_id
            )
            return False

        client = self.client
        if client is None:
            logger.warning("Twilio client unavailable. Skipping SMS for alert %s.", alert_id)
            return False

        body = format_alert_sms(alert_dict)

        # Rule 9 & 10: Gracefully handle API failures and sanitize error logs
        try:
            message = client.messages.create(
                body=body,
                from_=self.from_phone,
                to=self.to_phone,
            )
            message_sid = getattr(message, "sid", "UNKNOWN_SID")
            logger.info(
                "SMS sent successfully for alert %s to %s (SID: %s)",
                alert_id, mask_phone_number(self.to_phone), message_sid
            )
            # Record notification in sent store
            record_sent_notification(
                alert_id,
                path=self.sent_store_path,
                metadata={
                    "message_sid": message_sid,
                    "to": mask_phone_number(self.to_phone),
                    "severity": severity,
                },
            )
            return True

        except Exception as exc:
            sanitized = sanitize_message(str(exc), [self.account_sid, self.auth_token])
            logger.error("Twilio API failed for alert %s: %s", alert_id, sanitized)
            return False

    def notify_new_alerts(
        self,
        alerts: pd.DataFrame | list[dict] | None,
    ) -> list[dict]:
        """
        Process a collection of newly created alerts and dispatch SMS for qualifying alerts.

        Returns a list of summary dicts with alert_id, status ('SENT', 'SKIPPED', 'FAILED'),
        and reason.
        """
        results = []
        if alerts is None:
            return results

        if isinstance(alerts, pd.DataFrame):
            if alerts.empty:
                return results
            items = alerts.to_dict("records")
        elif isinstance(alerts, list):
            items = alerts
        else:
            return results

        for item in items:
            alert_id = str(item.get("alert_id") or "").strip()
            sev = str(item.get("severity") or "").strip().upper()

            if sev != "HIGH":
                results.append({"alert_id": alert_id, "status": "SKIPPED", "reason": f"severity {sev}"})
                continue

            if self.is_alert_notified(alert_id):
                results.append({"alert_id": alert_id, "status": "SKIPPED", "reason": "already notified"})
                continue

            success = self.send_alert(item)
            if success:
                results.append({"alert_id": alert_id, "status": "SENT", "reason": "ok"})
            else:
                results.append({"alert_id": alert_id, "status": "FAILED", "reason": "api or config failure"})

        return results


# ---------------------------------------------------------------------------
# Default module-level convenience functions
# ---------------------------------------------------------------------------
_default_notifier: SMSNotifier | None = None


def get_default_notifier() -> SMSNotifier:
    """Return or initialize the singleton SMSNotifier instance."""
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = SMSNotifier()
    return _default_notifier


def send_alert_notification(
    alert: dict | pd.Series,
    notifier: SMSNotifier | None = None,
    force: bool = False,
) -> bool:
    """Convenience function to send SMS for a single alert."""
    inst = notifier or get_default_notifier()
    return inst.send_alert(alert, force=force)


def notify_new_alerts(
    alerts: pd.DataFrame | list[dict] | None,
    notifier: SMSNotifier | None = None,
) -> list[dict]:
    """Convenience function to send SMS notifications for newly created alerts."""
    inst = notifier or get_default_notifier()
    return inst.notify_new_alerts(alerts)
