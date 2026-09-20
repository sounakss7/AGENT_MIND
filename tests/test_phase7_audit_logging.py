"""
tests/test_phase7_audit_logging.py
----------------------------------
Unit and regression tests for Phase 7: Audit Logging.
Covers:
1. One-time initialization of collection and indexes (self._initialized).
2. Qdrant Cloud persistence filtering (only WARN and BLOCK persisted, INFO skipped).
3. Optimized get_stats using client.count() rather than full payload scrolling.
4. Graceful fallback from client.count() to scroll on exception or unsupported client.
5. 30-day retention cleanup (cleanup_old_events).
6. Security event labeling: AUTH_SUCCESS (INFO), AUTH_FAILED (WARN), AUTH_LOCKOUT (BLOCK).
"""

from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone, timedelta

from security_guard import AuditLogger, audit_logger


def test_ensure_collection_initializes_only_once():
    logger = AuditLogger()
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    # First call initializes
    logger._ensure_collection(mock_client)
    assert logger._initialized is True
    assert mock_client.get_collections.call_count == 1
    assert mock_client.create_collection.call_count == 1
    assert mock_client.create_payload_index.call_count == 4

    # Subsequent calls should be no-ops
    logger._ensure_collection(mock_client)
    logger._ensure_collection(mock_client)
    assert mock_client.get_collections.call_count == 1
    assert mock_client.create_collection.call_count == 1


def test_audit_logger_skips_info_events_from_qdrant_cloud():
    logger = AuditLogger()
    logger._initialized = True  # skip init for test
    mock_client = MagicMock()

    class DummyPointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    with patch.object(logger, "_get_client", return_value=mock_client), \
         patch("qdrant_client.models.PointStruct", DummyPointStruct):
        # 1. INFO events: INPUT_PASSED, OUTPUT_PASSED, AUTH_SUCCESS
        logger.log("sess_123", "INPUT_PASSED", detail="Safe input")
        logger.log("sess_123", "OUTPUT_PASSED", detail="Safe output")
        logger.log("sess_123", "AUTH_SUCCESS", detail="Login ok")

        # None of these should call upsert on Qdrant
        mock_client.upsert.assert_not_called()

        # 2. WARN events: INPUT_TOO_LONG, AUTH_FAILED, OUTPUT_REDACTED
        logger.log("sess_123", "INPUT_TOO_LONG", detail="Message too long")
        assert mock_client.upsert.call_count == 1
        call_payload = mock_client.upsert.call_args[1]["points"][0].payload
        assert call_payload["severity"] == "WARN"
        assert call_payload["event_type"] == "INPUT_TOO_LONG"

        # 3. BLOCK events: PROMPT_INJECTION, AUTH_LOCKOUT
        logger.log("sess_123", "PROMPT_INJECTION", detail="Malicious injection blocked")
        assert mock_client.upsert.call_count == 2
        call_payload_block = mock_client.upsert.call_args[1]["points"][0].payload
        assert call_payload_block["severity"] == "BLOCK"
        assert call_payload_block["event_type"] == "PROMPT_INJECTION"


def test_get_stats_uses_client_count_method():
    logger = AuditLogger()
    logger._initialized = True
    mock_client = MagicMock()

    # Setup count return mock
    count_response = MagicMock()
    count_response.count = 42
    mock_client.count.return_value = count_response

    with patch.object(logger, "_get_client", return_value=mock_client), \
         patch.object(logger, "get_events") as mock_get_events:
        stats = logger.get_stats(session_id="acc_user123")

        # Should invoke client.count and NOT scroll full payloads
        assert mock_client.count.called
        mock_get_events.assert_not_called()
        assert stats["total"] == 42
        assert stats["by_severity"]["WARN"] == 42
        assert stats["by_severity"]["BLOCK"] == 42


def test_get_stats_fallback_to_scroll_when_count_fails():
    logger = AuditLogger()
    logger._initialized = True
    mock_client = MagicMock()
    mock_client.count.side_effect = RuntimeError("count endpoint not supported")

    dummy_events = [
        {"severity": "WARN", "event_type": "INPUT_TOO_LONG", "timestamp": "2026-09-20T12:00:00"},
        {"severity": "BLOCK", "event_type": "PROMPT_INJECTION", "timestamp": "2026-09-20T13:00:00"},
        {"severity": "INFO", "event_type": "INPUT_PASSED", "timestamp": "2026-09-20T14:00:00"},
    ]

    with patch.object(logger, "_get_client", return_value=mock_client), \
         patch.object(logger, "get_events", return_value=dummy_events) as mock_get_events:
        stats = logger.get_stats(session_id="acc_user123")

        # Fallback to get_events was triggered
        mock_get_events.assert_called_once_with(session_id="acc_user123", limit=1000)
        assert stats["total"] == 3
        assert stats["by_severity"]["WARN"] == 1
        assert stats["by_severity"]["BLOCK"] == 1
        assert stats["by_severity"]["INFO"] == 1


def test_cleanup_old_events():
    logger = AuditLogger()
    mock_client = MagicMock()

    with patch.object(logger, "_get_client", return_value=mock_client):
        res = logger.cleanup_old_events(days=30)
        assert res == 1
        assert mock_client.delete.called
        delete_args = mock_client.delete.call_args[1]
        assert delete_args["collection_name"] == logger.COLLECTION
        selector = delete_args["points_selector"]
        assert selector is not None


def test_auth_event_severity_mapping():
    logger = AuditLogger()
    assert logger._SEVERITY_MAP["AUTH_SUCCESS"] == "INFO"
    assert logger._SEVERITY_MAP["AUTH_FAILED"] == "WARN"
    assert logger._SEVERITY_MAP["AUTH_LOCKOUT"] == "BLOCK"
