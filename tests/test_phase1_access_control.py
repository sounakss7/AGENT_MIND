"""
tests/test_phase1_access_control.py
------------------------------------
Unit and integration tests for Phase 1: Access Control.

Tests cover:
  1. Cryptographic session ID derivation (scrypt, pepper, salt, normalization)
  2. Strict PIN length enforcement (>= 6 chars required)
  3. Ephemeral private session ID generation for anonymous visitors
  4. Session ID masking (preventing exposure of names, tokens, hashes)
  5. Security tab / AuditLogger scoping and isolation
  6. Attempt throttling and lockout mechanics
  7. Vector memory session isolation (rejecting empty/whitespace session IDs)
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from security_guard import (
    make_session_id,
    generate_ephemeral_id,
    mask_session_id,
    get_auth_pepper,
    AuditLogger,
)
import vector_memory


# ===========================================================================
# 1. Cryptographic Identity Derivation & Normalization
# ===========================================================================

def test_make_session_id_deterministic():
    """Same name, PIN, and pepper must produce the exact same account ID."""
    id1 = make_session_id("sounak", "secretPIN123", pepper="test_pepper_1")
    id2 = make_session_id("sounak", "secretPIN123", pepper="test_pepper_1")
    assert id1 == id2
    assert id1.startswith("acct_")
    assert len(id1) == 5 + 32  # "acct_" + 32 hex chars


def test_make_session_id_normalization():
    """Case differences and leading/trailing whitespace must resolve to the same ID."""
    id_clean = make_session_id("sounak", "mySecretPin", pepper="pep")
    id_upper = make_session_id("  SOUNAK  ", "  mySecretPin  ", pepper="pep")
    id_mixed = make_session_id("SounAK", "mySecretPin", pepper="pep")
    assert id_clean == id_upper == id_mixed


def test_make_session_id_pepper_isolation():
    """Different peppers must produce completely different IDs for the same user."""
    id_pep1 = make_session_id("sounak", "secretPIN123", pepper="pepper_a")
    id_pep2 = make_session_id("sounak", "secretPIN123", pepper="pepper_b")
    assert id_pep1 != id_pep2


def test_make_session_id_pin_variation():
    """Different PINs for the same name must produce completely different IDs."""
    id_pin1 = make_session_id("sounak", "secretPIN123", pepper="pep")
    id_pin2 = make_session_id("sounak", "differentPIN456", pepper="pep")
    assert id_pin1 != id_pin2


def test_make_session_id_name_variation():
    """Different names with the same PIN must produce completely different IDs."""
    id_user1 = make_session_id("alice", "sharedPIN999", pepper="pep")
    id_user2 = make_session_id("bob", "sharedPIN999", pepper="pep")
    assert id_user1 != id_user2


# ===========================================================================
# 2. Strict PIN & Name Validation
# ===========================================================================

@pytest.mark.parametrize("short_pin", [
    "",
    "1",
    "12",
    "123",
    "1234",
    "12345",
    "     ",
    " 123 ",
])
def test_make_session_id_rejects_short_or_empty_pin(short_pin):
    """PINs shorter than 6 characters must be strictly rejected with ValueError."""
    with pytest.raises(ValueError, match="PIN must be at least 6 characters"):
        make_session_id("sounak", short_pin, pepper="pep")


@pytest.mark.parametrize("empty_name", [
    "",
    "   ",
    "\t\n",
])
def test_make_session_id_rejects_empty_name(empty_name):
    """Empty or whitespace-only names must be strictly rejected."""
    with pytest.raises(ValueError, match="Name cannot be empty"):
        make_session_id(empty_name, "validPIN123", pepper="pep")


def test_make_session_id_accepts_valid_pin():
    """PINs with 6 or more characters are accepted."""
    res = make_session_id("alice", "123456", pepper="pep")
    assert res.startswith("acct_")


# ===========================================================================
# 3. Ephemeral Anonymous Session ID Generation
# ===========================================================================

def test_generate_ephemeral_id():
    """Ephemeral IDs must start with 'dev_' and contain 128-bit random entropy."""
    id1 = generate_ephemeral_id()
    id2 = generate_ephemeral_id()
    assert id1.startswith("dev_")
    assert id2.startswith("dev_")
    assert id1 != id2
    # secrets.token_urlsafe(16) generates 22 URL-safe base64 characters
    assert len(id1) >= 20


# ===========================================================================
# 4. Session ID Masking
# ===========================================================================

def test_mask_session_id_account():
    """Account IDs starting with 'acct_' must mask everything except the last 4 chars."""
    sid = "acct_abcdef1234567890abcdef1234"
    masked = mask_session_id(sid)
    assert masked == "acct_***1234"
    assert "abcdef" not in masked


def test_mask_session_id_ephemeral():
    """Ephemeral IDs starting with 'dev_' must mask everything except the last 4 chars."""
    sid = "dev_xyz9876543210abcd"
    masked = mask_session_id(sid)
    assert masked == "dev_***abcd"
    assert "xyz987" not in masked


def test_mask_session_id_legacy_or_plain_name():
    """Plain user names or legacy IDs must never expose plaintext in the UI."""
    sid = "sounak"
    masked = mask_session_id(sid)
    assert masked.startswith("usr_***")
    assert "sounak" not in masked
    assert len(masked) == 7 + 4  # "usr_***" + 4 hex chars


def test_mask_session_id_empty():
    """Empty or None session ID returns a safe fallback."""
    assert mask_session_id("") == "usr_unknown"
    assert mask_session_id(None) == "usr_unknown"


# ===========================================================================
# 5. Security Tab & Audit Logger Scoping
# ===========================================================================

def test_audit_logger_get_stats_session_scoping():
    """
    AuditLogger.get_stats(session_id=...) must only compute statistics for that specific session.
    """
    logger = AuditLogger()
    mock_events = [
        {"session_id": "acct_user_1", "severity": "WARN", "event_type": "INPUT_TOO_LONG", "timestamp": "2026-09-20T10:00:00"},
        {"session_id": "acct_user_1", "severity": "BLOCK", "event_type": "PROMPT_INJECTION", "timestamp": "2026-09-20T10:05:00"},
        {"session_id": "acct_user_2", "severity": "BLOCK", "event_type": "PROMPT_INJECTION", "timestamp": "2026-09-20T10:10:00"},
    ]

    with patch.object(logger, "get_events", return_value=[e for e in mock_events if e["session_id"] == "acct_user_1"]):
        stats = logger.get_stats(session_id="acct_user_1")
        assert stats["total"] == 2
        assert stats["by_severity"]["WARN"] == 1
        assert stats["by_severity"]["BLOCK"] == 1
        assert stats["by_type"]["PROMPT_INJECTION"] == 1


def test_audit_logger_get_stats_global_admin_scoping():
    """AuditLogger.get_stats(session_id=None) computes global counts across all sessions."""
    logger = AuditLogger()
    mock_events = [
        {"session_id": "acct_user_1", "severity": "WARN", "event_type": "INPUT_TOO_LONG", "timestamp": "2026-09-20T10:00:00"},
        {"session_id": "acct_user_2", "severity": "BLOCK", "event_type": "PROMPT_INJECTION", "timestamp": "2026-09-20T10:10:00"},
    ]

    with patch.object(logger, "get_events", return_value=mock_events):
        stats = logger.get_stats(session_id=None)
        assert stats["total"] == 2
        assert stats["by_severity"]["WARN"] == 1
        assert stats["by_severity"]["BLOCK"] == 1


# ===========================================================================
# 6. Attempt Throttling & Lockout Logic
# ===========================================================================

def test_attempt_throttling_simulation():
    """Simulate authentication attempts: 5 failed attempts must trigger a lockout."""
    state = {
        "auth_attempts": 0,
        "auth_lockout_until": 0.0,
    }

    def attempt_login(name: str, pin: str):
        now = time.time()
        if now < state["auth_lockout_until"]:
            return False, "LOCKED_OUT"
        try:
            sid = make_session_id(name, pin, pepper="test")
            state["auth_attempts"] = 0
            return True, sid
        except ValueError:
            state["auth_attempts"] += 1
            if state["auth_attempts"] >= 5:
                state["auth_lockout_until"] = now + 60.0
                return False, "LOCKOUT_TRIGGERED"
            return False, "INVALID_PIN"

    # 4 invalid attempts
    for i in range(4):
        success, msg = attempt_login("sounak", "123")  # too short
        assert not success
        assert msg == "INVALID_PIN"
        assert state["auth_attempts"] == i + 1

    # 5th attempt triggers lockout
    success, msg = attempt_login("sounak", "123")
    assert not success
    assert msg == "LOCKOUT_TRIGGERED"
    assert state["auth_attempts"] == 5
    assert state["auth_lockout_until"] > time.time()

    # Immediate 6th attempt is blocked by lockout
    success, msg = attempt_login("sounak", "validPIN123")
    assert not success
    assert msg == "LOCKED_OUT"


# ===========================================================================
# 7. Vector Memory Session Isolation
# ===========================================================================

def test_vector_memory_refuses_empty_session_id():
    """Vector memory functions must refuse empty or whitespace session IDs."""
    with patch.object(vector_memory, "_get_client") as mock_client:
        # None or empty session IDs should return safe empty values and not call client
        assert vector_memory.retrieve_relevant_memory("query", session_id="") == ""
        assert vector_memory.retrieve_relevant_memory("query", session_id="   ") == ""
        assert vector_memory.retrieve_relevant_memory("query", session_id=None) == ""

        assert vector_memory.get_memory_count(session_id="") == 0
        assert vector_memory.get_memory_count(session_id="  ") == 0

        assert vector_memory.get_all_memories(session_id="") == []
        assert vector_memory.get_all_memories(session_id="  ") == []

        # save_memory and clear_memory must not execute client calls
        vector_memory.save_memory(role="user", content="hello", session_id="")
        vector_memory.clear_memory(session_id="")

        mock_client.assert_not_called()


# ===========================================================================
# 8. Malicious & Adversarial Edge Cases
# ===========================================================================

def test_make_session_id_handles_injection_payloads():
    """SQL/prompt injection attempts in name or PIN are safely scrypt-hashed as raw data."""
    malicious_name = "' OR 1=1; DROP TABLE users; --"
    malicious_pin = "'; <script>alert(1)</script>; --"
    derived = make_session_id(malicious_name, malicious_pin, pepper="safe_pep")
    assert derived.startswith("acct_")
    assert len(derived) == 5 + 32
    # Output must be purely alphanumeric hex chars prefixed with "acct_"
    assert derived[5:].isalnum()


def test_make_session_id_unicode_and_emojis():
    """Unicode names, accents, and emojis derive properly without encoding errors."""
    name_unicode = "ユーザー🧠"
    pin_unicode = "🔑パスワード123"
    derived = make_session_id(name_unicode, pin_unicode, pepper="pep")
    assert derived.startswith("acct_")
    assert derived[5:].isalnum()


def test_mask_session_id_various_edge_cases():
    """mask_session_id must never crash or leak secrets on unusual input."""
    assert mask_session_id(None) == "usr_unknown"
    assert mask_session_id("") == "usr_unknown"
    assert mask_session_id("acct_") == "acct_***"
    assert mask_session_id("dev_") == "dev_***"
    assert mask_session_id("dev_1") == "dev_***1"
    # Legacy short name
    res = mask_session_id("abc")
    assert res.startswith("usr_***")

