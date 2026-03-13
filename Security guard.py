"""
security_guard.py
-----------------
Native security layer for AGENT_MIND / Neuroplexa AI.
No external dependencies beyond what is already in requirements.txt.

Provides:
  • make_session_id() — SHA-256 hashed session IDs (name + PIN)
  • InputGuard        — validates / blocks user queries before LangGraph
  • OutputGuard       — sanitises LLM responses before display or storage
  • MemoryGuard       — sanitises content before saving to Qdrant
  • AuditLogger       — writes every security event to a Qdrant collection
"""

import re
import hashlib
import uuid
import logging
from datetime import datetime, date
from typing import Optional

# ---------------------------------------------------------------------------
# Qdrant constants
# Dummy zero-vectors are used for audit points (no semantic search needed).
# ---------------------------------------------------------------------------
AUDIT_COLLECTION = "security_audit"
VECTOR_DIM       = 384


# ═══════════════════════════════════════════════════════════════════════════
# 1. HASHED SESSION IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

def make_session_id(name: str, pin: str = "") -> str:
    """
    Produce a short, stable SHA-256 hash from (name + pin).

    Properties:
      • Same name + PIN on ANY device  →  identical session_id.
      • No PII stored anywhere; only the hash reaches Qdrant.
      • Without PIN: plain lowercase name (backward-compatible).
      • With PIN: SHA-256 makes brute-force practically infeasible.
    """
    raw = f"{name.strip().lower()}:{pin.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ═══════════════════════════════════════════════════════════════════════════
# 2. PROMPT INJECTION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS = [
    # Classic instruction overrides
    r"ignore\s+(all\s+)?(previous|prior|your|my|the)\s+(instructions?|prompts?|rules?|context)",
    r"disregard\s+(all\s+)?(previous|prior|your|my|the)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|your|all|the\s+above|prior)",
    r"override\s+(your\s+)?(instructions?|rules?|system|prompt)",

    # Role hijacking
    r"you\s+are\s+now\b",
    r"act\s+as\s+(if\s+)?(you('re|\s+are)|a\s+)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"your\s+(real|true|actual|hidden)\s+(self|purpose|goal|name|role)",
    r"switch\s+(to|into)\s+(developer|admin|god|jailbreak|unrestricted)\s+mode",
    r"developer\s+mode",
    r"jailbreak",
    r"\bDAN\b",
    r"do\s+anything\s+now",

    # System-level token injection
    r"\[system\]",
    r"\[inst\]",
    r"<\|system\|>",
    r"<\|im_start\|>",
    r"###\s*(instruction|system|override)",

    # Secret / credential extraction
    r"reveal\s+.{0,40}(key|token|secret|password|api|credential)",
    r"print\s+.{0,20}(key|token|secret|password)",
    r"what\s+(is|are)\s+(your|the)\s+(api\s+key|secret|token|password)",

    # Harmful content generation
    r"(give\s+me|tell\s+me|explain\s+how\s+to)\s+(make|build|create|synthesize)\s+(bomb|weapon|malware|virus|ransomware|exploit)",

    # Prompt leaking
    r"(repeat|print|output|show|tell\s+me)\s+(the\s+)?(system\s+prompt|instructions|above\s+text)",
]

_COMPILED_INJECTIONS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════════
# 3. PII / SECRET REDACTION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

_REDACT_PATTERNS = {
    "EMAIL":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE_IN":    re.compile(r"\b(\+91[\-\s]?)?[6-9]\d{9}\b"),
    "PHONE_INTL":  re.compile(r"\b\+?[1-9]\d{7,14}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ \-]?){13,16}\b"),
    "AADHAAR":     re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "PAN":         re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "GOOGLE_KEY":  re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    "GROQ_KEY":    re.compile(r"gsk_[A-Za-z0-9]{40,}"),
    "OPENAI_KEY":  re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "AWS_KEY":     re.compile(r"AKIA[0-9A-Z]{16}"),
    "GH_TOKEN":    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    "IP_ADDR":     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII/secrets with [REDACTED:<TYPE>]. Returns (clean_text, findings)."""
    findings = []
    for label, pattern in _REDACT_PATTERNS.items():
        if pattern.search(text):
            findings.append(label)
            text = pattern.sub(f"[REDACTED:{label}]", text)
    return text, findings


# ═══════════════════════════════════════════════════════════════════════════
# 4. GUARD RESULT
# ═══════════════════════════════════════════════════════════════════════════

class GuardResult:
    """
    Returned by every guard.validate() call.

    Attributes:
      passed     — True if the content is safe to use
      clean_text — sanitised / redacted version of the original text
      reason     — human-readable explanation (for logs, not shown to user)
      event_type — machine-readable code for AuditLogger
      findings   — list of specific findings (pattern names, PII types, etc.)
    """
    def __init__(self, passed: bool, clean_text: str,
                 reason: str = "", event_type: str = "", findings: list = None):
        self.passed     = passed
        self.clean_text = clean_text
        self.reason     = reason
        self.event_type = event_type
        self.findings   = findings or []

    def __bool__(self):
        return self.passed


# ═══════════════════════════════════════════════════════════════════════════
# 5. INPUT GUARD
# ═══════════════════════════════════════════════════════════════════════════

class InputGuard:
    """
    Validates every user query BEFORE it reaches the LangGraph router.

    Checks (in order):
      1. Length limit         — prevents context-flooding attacks
      2. Null-byte stripping  — silent sanitisation
      3. Gibberish detection  — blocks keyboard-spam / nonsense
      4. Prompt injection     — blocks jailbreak / override attempts
      5. Code block strip     — removes ``` blocks to prevent framing attacks
    """
    MAX_QUERY_LEN = 3000

    def validate(self, text: str) -> GuardResult:

        # 1. Length check
        if len(text) > self.MAX_QUERY_LEN:
            return GuardResult(
                passed     = False,
                clean_text = text,
                reason     = f"Query exceeds {self.MAX_QUERY_LEN} character limit.",
                event_type = "INPUT_TOO_LONG",
            )

        # 2. Strip control characters
        clean = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
        clean = clean.strip()

        if not clean:
            return GuardResult(
                passed     = False,
                clean_text = "",
                reason     = "Empty query after sanitisation.",
                event_type = "EMPTY_INPUT",
            )

        # 3. Gibberish detection
        words = clean.split()
        if len(clean) > 30:
            avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
            if avg_word_len > 18:
                return GuardResult(
                    passed     = False,
                    clean_text = clean,
                    reason     = "Input appears to be gibberish or keyboard spam.",
                    event_type = "GIBBERISH_INPUT",
                )

        # 4. Prompt injection detection
        for pattern in _COMPILED_INJECTIONS:
            if pattern.search(clean):
                return GuardResult(
                    passed     = False,
                    clean_text = clean,
                    reason     = "Prompt injection attempt detected.",
                    event_type = "PROMPT_INJECTION",
                    findings   = [pattern.pattern],
                )

        # 5. Strip inline code blocks
        clean = re.sub(r"```[\s\S]*?```", "[code block removed]", clean)

        return GuardResult(
            passed     = True,
            clean_text = clean,
            event_type = "INPUT_PASSED",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. OUTPUT GUARD
# ═══════════════════════════════════════════════════════════════════════════

class OutputGuard:
    """
    Validates every LLM response BEFORE it is shown or stored.

    Checks:
      1. PII / secret auto-redaction
      2. Toxic / harmful content blocking
    """

    _TOXIC_PATTERNS = [
        r"step[- ]by[- ]step\s+(guide|instructions?)\s+(to|for)\s+(make|build|create|synthesize)\s+(bomb|weapon|explosive)",
        r"how\s+to\s+(hack|crack|exploit|break\s+into)",
        r"(child|minor|underage).{0,40}(sexual|nude|naked|explicit)",
    ]
    _COMPILED_TOXIC = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]

    def validate(self, text: str) -> GuardResult:
        if not isinstance(text, str):
            return GuardResult(passed=True, clean_text=str(text), event_type="OUTPUT_PASSED")

        # 1. PII / secret redaction
        clean, findings = _redact_pii(text)
        was_redacted    = bool(findings)

        # 2. Toxic content check
        for pattern in self._COMPILED_TOXIC:
            if pattern.search(clean):
                return GuardResult(
                    passed     = False,
                    clean_text = "I can't provide that information.",
                    reason     = "Response blocked: potentially harmful content detected.",
                    event_type = "OUTPUT_BLOCKED",
                    findings   = [pattern.pattern],
                )

        if was_redacted:
            return GuardResult(
                passed     = True,
                clean_text = clean,
                reason     = "PII / secrets auto-redacted from response.",
                event_type = "OUTPUT_REDACTED",
                findings   = findings,
            )

        return GuardResult(
            passed     = True,
            clean_text = clean,
            event_type = "OUTPUT_PASSED",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 7. MEMORY GUARD
# ═══════════════════════════════════════════════════════════════════════════

class MemoryGuard:
    """
    Sanitises content BEFORE it is embedded and written to Qdrant.
    Ensures no PII or API keys are ever stored in the vector database.
    """

    def validate(self, content: str) -> GuardResult:
        clean, findings = _redact_pii(content)
        if findings:
            return GuardResult(
                passed     = True,        # still save — just redacted content
                clean_text = clean,
                reason     = "PII redacted before saving to Qdrant.",
                event_type = "MEMORY_REDACTED",
                findings   = findings,
            )
        return GuardResult(
            passed     = True,
            clean_text = clean,
            event_type = "MEMORY_PASSED",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. AUDIT LOGGER
# ═══════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Writes every security event to the 'security_audit' Qdrant collection.

    Each point stores:
      session_id, event_type, severity, detail, findings, timestamp

    Severity levels:
      INFO  — normal operation
      WARN  — suspicious but allowed (PII redacted, etc.)
      BLOCK — request or response was blocked
    """

    COLLECTION = AUDIT_COLLECTION

    _SEVERITY_MAP = {
        "INPUT_PASSED":    "INFO",
        "INPUT_TOO_LONG":  "WARN",
        "EMPTY_INPUT":     "WARN",
        "GIBBERISH_INPUT": "WARN",
        "PROMPT_INJECTION":"BLOCK",
        "OUTPUT_PASSED":   "INFO",
        "OUTPUT_REDACTED": "WARN",
        "OUTPUT_BLOCKED":  "BLOCK",
        "MEMORY_PASSED":   "INFO",
        "MEMORY_REDACTED": "WARN",
    }

    def _get_client(self):
        try:
            from vector_memory import _get_client
            return _get_client()
        except Exception:
            return None

    def _ensure_collection(self, client) -> None:
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
        try:
            existing = [c.name for c in client.get_collections().collections]
            if self.COLLECTION not in existing:
                client.create_collection(
                    collection_name = self.COLLECTION,
                    vectors_config  = VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                )
                print(f"[AuditLogger] Created collection '{self.COLLECTION}'.")
            for field in ["session_id", "severity", "event_type"]:
                try:
                    client.create_payload_index(
                        collection_name = self.COLLECTION,
                        field_name      = field,
                        field_schema    = PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"[AuditLogger] _ensure_collection error: {e}")

    def log(self, session_id: str, event_type: str,
            detail: str = "", findings: list = None) -> None:
        """Write one security event to Qdrant."""
        severity = self._SEVERITY_MAP.get(event_type, "INFO")
        print(f"[AUDIT] {severity} | {event_type} | {session_id[:16]} | {detail[:80]}")

        client = self._get_client()
        if client is None:
            return
        try:
            from qdrant_client.models import PointStruct
            self._ensure_collection(client)
            client.upsert(
                collection_name = self.COLLECTION,
                points = [PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = [0.0] * VECTOR_DIM,
                    payload = {
                        "session_id": session_id,
                        "event_type": event_type,
                        "severity":   severity,
                        "detail":     detail[:500],
                        "findings":   findings or [],
                        "timestamp":  datetime.utcnow().isoformat(),
                    },
                )],
            )
        except Exception as e:
            logging.warning(f"[AuditLogger] Could not write event: {e}")

    def get_events(self, session_id: str = None, limit: int = 500) -> list:
        """
        Fetch audit events from Qdrant.
        Pass session_id to filter to one user; omit for all events.
        Returns list sorted newest-first.
        """
        client = self._get_client()
        if client is None:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self._ensure_collection(client)

            filt = None
            if session_id:
                filt = Filter(must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                ])

            response, _ = client.scroll(
                collection_name = self.COLLECTION,
                scroll_filter   = filt,
                limit           = limit,
                with_payload    = True,
                with_vectors    = False,
            )
            events = [p.payload for p in response]
            events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return events

        except Exception as e:
            logging.warning(f"[AuditLogger] Could not fetch events: {e}")
            return []

    def get_stats(self) -> dict:
        """Aggregate counts used by the Security Dashboard."""
        events = self.get_events(limit=1000)
        today  = str(date.today())
        stats  = {
            "total":            len(events),
            "by_severity":      {"INFO": 0, "WARN": 0, "BLOCK": 0},
            "by_type":          {},
            "injections_today": 0,
        }
        for e in events:
            sev = e.get("severity", "INFO")
            typ = e.get("event_type", "UNKNOWN")
            stats["by_severity"][sev] = stats["by_severity"].get(sev, 0) + 1
            stats["by_type"][typ]     = stats["by_type"].get(typ, 0) + 1
            if typ == "PROMPT_INJECTION" and e.get("timestamp", "").startswith(today):
                stats["injections_today"] += 1
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# 9. MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════

input_guard  = InputGuard()
output_guard = OutputGuard()
memory_guard = MemoryGuard()
audit_logger = AuditLogger()