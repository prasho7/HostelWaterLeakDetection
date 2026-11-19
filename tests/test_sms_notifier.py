"""
test_sms_notifier.py — Test suite for Twilio SMS Notification system.

Tests:
1. Twilio configuration checks (configured vs unconfigured).
2. Severity filtering: HIGH triggers SMS; MEDIUM and LOW are skipped.
3. Formatted message content: alert ID, hostel, floor, meter, severity, timestamp, description.
4. Duplicate prevention: repeated calls for the same alert do not send duplicate SMS.
5. Error handling: Twilio API failure handles gracefully without raising exceptions.
6. Credential security: sensitive tokens are never exposed in error logs.
7. Integration with alert_manager: newly created alerts detection.
8. Preservation of existing alert resolution and annotation state.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.join(_TESTS_DIR, "..", "src")
for _p in (_TESTS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alert_manager
import alert_store
import annotation_store
import sms_notifier
from sms_notifier import (
    SMSNotifier,
    format_alert_sms,
    load_sent_notifications,
    mask_phone_number,
    notify_new_alerts,
    record_sent_notification,
    sanitize_message,
    send_alert_notification,
)


@pytest.fixture
def mock_twilio_client():
    """Mock Twilio Client whose messages.create returns a mock message with a sid."""
    client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.sid = "SM_TEST_MOCK_SID_12345"
    client.messages.create.return_value = mock_msg
    return client


@pytest.fixture
def sample_high_alert():
    return {
        "alert_id": "alert-uuid-high-001",
        "hostel_id": "H1",
        "floor_id": "H1-F2",
        "meter_id": "M-H1-F2",
        "first_seen": "2024-01-01T03:00:00",
        "last_seen": "2024-01-02T08:00:00",
        "severity": "HIGH",
        "avg_z_score": 38.5,
        "persistence_count": 30,
        "status": "OPEN",
        "resolved_at": "",
    }


@pytest.fixture
def sample_medium_alert():
    return {
        "alert_id": "alert-uuid-med-002",
        "hostel_id": "H2",
        "floor_id": "H2-F1",
        "meter_id": "M-H2-F1",
        "first_seen": "2024-01-05T10:00:00",
        "last_seen": "2024-01-05T18:00:00",
        "severity": "MEDIUM",
        "avg_z_score": 8.5,
        "persistence_count": 8,
        "status": "OPEN",
        "resolved_at": "",
    }


@pytest.fixture
def sample_low_alert():
    return {
        "alert_id": "alert-uuid-low-003",
        "hostel_id": "H3",
        "floor_id": "H3-F1",
        "meter_id": "M-H3-F1",
        "first_seen": "2024-01-10T06:00:00",
        "last_seen": "2024-01-10T12:00:00",
        "severity": "LOW",
        "avg_z_score": 4.2,
        "persistence_count": 6,
        "status": "OPEN",
        "resolved_at": "",
    }


# ===========================================================================
# 1. Configuration & Security Tests
# ===========================================================================

def test_notifier_configuration():
    # Incomplete config
    notifier = SMSNotifier(
        account_sid="",
        auth_token="",
        from_phone="",
        to_phone="",
    )
    assert not notifier.is_configured()

    # Complete config
    configured = SMSNotifier(
        account_sid="AC1234567890",
        auth_token="secret_token_123",
        from_phone="+12345678901",
        to_phone="+19876543210",
    )
    assert configured.is_configured()


def test_mask_phone_number():
    assert mask_phone_number("+12345678901") == "+123****8901"
    assert mask_phone_number("+919876543210") == "+919*****3210"
    assert mask_phone_number("") == "N/A"
    assert mask_phone_number("123") == "****"


def test_sanitize_message():
    secret_token = "SUPER_SECRET_TWILIO_TOKEN_XYZ"
    secret_sid = "AC9999999999999999999"
    raw_error = f"Twilio returned error for {secret_sid} using token {secret_token}"
    sanitized = sanitize_message(raw_error, [secret_token, secret_sid])
    assert secret_token not in sanitized
    assert secret_sid not in sanitized
    assert "[REDACTED]" in sanitized


# ===========================================================================
# 2. Message Format Tests
# ===========================================================================

def test_format_alert_sms_includes_all_required_fields(sample_high_alert):
    msg = format_alert_sms(sample_high_alert)

    # 1. Alert ID
    assert "alert-uuid-high-001" in msg
    # 2. Hostel
    assert "H1" in msg
    # 3. Floor
    assert "H1-F2" in msg
    # 4. Meter
    assert "M-H1-F2" in msg
    # 5. Severity
    assert "HIGH" in msg
    # 6. Timestamp
    assert "2024-01-02T08:00:00" in msg
    # 7. Short description
    assert "Description:" in msg
    assert "physical inspection" in msg.lower() or "flow detected" in msg.lower()


# ===========================================================================
# 3. Severity Filtering Tests
# ===========================================================================

def test_high_severity_triggers_sms(mock_twilio_client, sample_high_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        sent_path = os.path.join(tmpdir, "sent.json")
        notifier = SMSNotifier(
            account_sid="AC12345",
            auth_token="auth_secret",
            from_phone="+12345678901",
            to_phone="+19876543210",
            sent_store_path=sent_path,
            twilio_client=mock_twilio_client,
        )

        result = notifier.send_alert(sample_high_alert)
        assert result is True
        mock_twilio_client.messages.create.assert_called_once()
        args, kwargs = mock_twilio_client.messages.create.call_args
        assert kwargs["to"] == "+19876543210"
        assert kwargs["from_"] == "+12345678901"
        assert sample_high_alert["alert_id"] in kwargs["body"]


def test_medium_and_low_severity_do_not_trigger_sms(mock_twilio_client, sample_medium_alert, sample_low_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        sent_path = os.path.join(tmpdir, "sent.json")
        notifier = SMSNotifier(
            account_sid="AC12345",
            auth_token="auth_secret",
            from_phone="+12345678901",
            to_phone="+19876543210",
            sent_store_path=sent_path,
            twilio_client=mock_twilio_client,
        )

        # MEDIUM severity
        assert notifier.send_alert(sample_medium_alert) is False
        mock_twilio_client.messages.create.assert_not_called()

        # LOW severity
        assert notifier.send_alert(sample_low_alert) is False
        mock_twilio_client.messages.create.assert_not_called()


def test_batch_notify_new_alerts_only_sends_high(mock_twilio_client, sample_high_alert, sample_medium_alert, sample_low_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        sent_path = os.path.join(tmpdir, "sent.json")
        notifier = SMSNotifier(
            account_sid="AC12345",
            auth_token="auth_secret",
            from_phone="+12345678901",
            to_phone="+19876543210",
            sent_store_path=sent_path,
            twilio_client=mock_twilio_client,
        )

        df = pd.DataFrame([sample_high_alert, sample_medium_alert, sample_low_alert])
        results = notifier.notify_new_alerts(df)

        assert len(results) == 3
        # HIGH should be SENT
        assert results[0]["alert_id"] == sample_high_alert["alert_id"]
        assert results[0]["status"] == "SENT"
        # MEDIUM & LOW should be SKIPPED
        assert results[1]["status"] == "SKIPPED"
        assert results[2]["status"] == "SKIPPED"

        # Twilio client should have been called exactly once
        assert mock_twilio_client.messages.create.call_count == 1


# ===========================================================================
# 4. Duplicate Prevention Tests
# ===========================================================================

def test_duplicate_prevention(mock_twilio_client, sample_high_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        sent_path = os.path.join(tmpdir, "sent.json")
        notifier = SMSNotifier(
            account_sid="AC12345",
            auth_token="auth_secret",
            from_phone="+12345678901",
            to_phone="+19876543210",
            sent_store_path=sent_path,
            twilio_client=mock_twilio_client,
        )

        # First call succeeds
        assert notifier.send_alert(sample_high_alert) is True
        assert mock_twilio_client.messages.create.call_count == 1

        # Second call with same alert ID is prevented
        assert notifier.send_alert(sample_high_alert) is False
        assert mock_twilio_client.messages.create.call_count == 1  # No second call

        # Check sent store persisted
        sent_ids = load_sent_notifications(sent_path)
        assert sample_high_alert["alert_id"] in sent_ids


def test_batch_duplicate_prevention(mock_twilio_client, sample_high_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        sent_path = os.path.join(tmpdir, "sent.json")
        notifier = SMSNotifier(
            account_sid="AC12345",
            auth_token="auth_secret",
            from_phone="+12345678901",
            to_phone="+19876543210",
            sent_store_path=sent_path,
            twilio_client=mock_twilio_client,
        )

        df = pd.DataFrame([sample_high_alert])

        # Run 1
        res1 = notifier.notify_new_alerts(df)
        assert res1[0]["status"] == "SENT"
        assert mock_twilio_client.messages.create.call_count == 1

        # Run 2 with same batch
        res2 = notifier.notify_new_alerts(df)
        assert res2[0]["status"] == "SKIPPED"
        assert "already notified" in res2[0]["reason"]
        assert mock_twilio_client.messages.create.call_count == 1  # Still 1


# ===========================================================================
# 5. API Failure Handling Tests
# ===========================================================================

def test_api_failure_handled_gracefully(sample_high_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        sent_path = os.path.join(tmpdir, "sent.json")

        failing_client = MagicMock()
        failing_client.messages.create.side_effect = Exception("Twilio 500 Internal Server Error")

        notifier = SMSNotifier(
            account_sid="AC12345",
            auth_token="auth_secret",
            from_phone="+12345678901",
            to_phone="+19876543210",
            sent_store_path=sent_path,
            twilio_client=failing_client,
        )

        # Must not raise an exception, returns False
        success = notifier.send_alert(sample_high_alert)
        assert success is False

        # Alert should NOT be recorded as sent
        sent_ids = load_sent_notifications(sent_path)
        assert sample_high_alert["alert_id"] not in sent_ids


def test_unconfigured_credentials_returns_false_cleanly(sample_high_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        sent_path = os.path.join(tmpdir, "sent.json")
        notifier = SMSNotifier(
            account_sid="",
            auth_token="",
            from_phone="",
            to_phone="",
            sent_store_path=sent_path,
        )
        assert notifier.send_alert(sample_high_alert) is False


# ===========================================================================
# 6. Pipeline Integration & State Preservation Tests
# ===========================================================================

def test_get_newly_created_alerts():
    with tempfile.TemporaryDirectory() as tmpdir:
        alerts_path = os.path.join(tmpdir, "alerts.csv")

        # Existing alerts on disk
        existing_df = pd.DataFrame([{
            "alert_id": "existing-001",
            "hostel_id": "H1",
            "floor_id": "H1-F1",
            "meter_id": "M-H1-F1",
            "first_seen": "2024-01-01T00:00:00",
            "last_seen": "2024-01-01T10:00:00",
            "severity": "HIGH",
            "avg_z_score": 35.0,
            "persistence_count": 10,
            "status": "OPEN",
            "resolved_at": "",
        }])
        existing_df.to_csv(alerts_path, index=False)

        # Incoming alerts: existing-001 plus brand-new alert new-002
        incoming_df = pd.DataFrame([
            {
                "alert_id": "existing-001",
                "hostel_id": "H1",
                "floor_id": "H1-F1",
                "meter_id": "M-H1-F1",
                "first_seen": "2024-01-01T00:00:00",
                "last_seen": "2024-01-01T10:00:00",
                "severity": "HIGH",
                "avg_z_score": 35.0,
                "persistence_count": 10,
                "status": "OPEN",
                "resolved_at": "",
            },
            {
                "alert_id": "new-002",
                "hostel_id": "H2",
                "floor_id": "H2-F3",
                "meter_id": "M-H2-F3",
                "first_seen": "2024-01-02T00:00:00",
                "last_seen": "2024-01-02T05:00:00",
                "severity": "HIGH",
                "avg_z_score": 30.0,
                "persistence_count": 5,
                "status": "OPEN",
                "resolved_at": "",
            },
        ])

        new_alerts = alert_manager.get_newly_created_alerts(incoming_df, existing_path=alerts_path)
        assert len(new_alerts) == 1
        assert new_alerts.iloc[0]["alert_id"] == "new-002"


def test_existing_alert_resolution_and_annotations_remain_intact():
    """Verify that alert resolution and annotations work normally alongside SMS notifier."""
    with tempfile.TemporaryDirectory() as tmpdir:
        alerts_path = os.path.join(tmpdir, "alerts.csv")
        ann_path    = os.path.join(tmpdir, "annotations.csv")

        alert_id = "test-alert-intact-999"
        df = pd.DataFrame([{
            "alert_id": alert_id,
            "hostel_id": "H1",
            "floor_id": "H1-F1",
            "meter_id": "M-H1-F1",
            "first_seen": "2024-01-01T00:00:00",
            "last_seen": "2024-01-01T10:00:00",
            "severity": "HIGH",
            "avg_z_score": 35.0,
            "persistence_count": 10,
            "status": "OPEN",
            "resolved_at": "",
        }])
        alert_store.save_alerts(df, alerts_path)

        # Add annotation
        ann_df = annotation_store.load_annotations(ann_path)
        ann_df = annotation_store.add_annotation(ann_df, alert_id, "Plumber assigned")
        annotation_store.save_annotations(ann_df, ann_path)

        # Resolve alert
        raw = alert_store.load_alerts(alerts_path)
        resolved = alert_store.resolve_alert(raw, alert_id)
        alert_store.save_alerts(resolved, alerts_path)

        # Verify alert is resolved
        reloaded = alert_store.load_alerts(alerts_path)
        assert reloaded.iloc[0]["status"] == "RESOLVED"
        assert reloaded.iloc[0]["resolved_at"] != ""

        # Verify annotation remains intact
        notes = annotation_store.get_annotations_for_alert(
            annotation_store.load_annotations(ann_path), alert_id
        )
        assert len(notes) == 1
        assert notes.iloc[0]["note"] == "Plumber assigned"
