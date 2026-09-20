"""
vector_memory.py
----------------
Long-term semantic memory for AGENT_MIND using Qdrant Cloud +
sentence-transformers/all-MiniLM-L6-v2 embeddings.

Credentials are read from Streamlit secrets:
  QDRANT_URL     — your Qdrant Cloud cluster URL
  QDRANT_API_KEY — your Qdrant Cloud API key

SECURITY UPDATE:
  • save_memory() now runs content through MemoryGuard before storing.
  • PII (emails, phone numbers, Aadhaar, API keys, etc.) is auto-redacted.
  • A 'pii_redacted' flag is stored in payload for audit purposes.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    ScrollRequest,
)
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COLLECTION_NAME = "agent_mind_memory"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM      = 384
TOP_K           = 5
SCORE_THRESHOLD = float(os.environ.get("VECTOR_SCORE_THRESHOLD", "0.45"))

_BLOCKED_OR_ERROR_PREFIXES = (
    "[Response blocked",
    "⚠️",
    "Error:",
    "I can't provide that information",
    "Failed to generate image",
)

# ---------------------------------------------------------------------------
# SINGLETON HELPERS
# ---------------------------------------------------------------------------
_embedder: SentenceTransformer | None = None
_qdrant:   QdrantClient | None        = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[VectorMemory] Loading embedding model (first time only)...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        try:
            import streamlit as st
            qdrant_url     = st.secrets["QDRANT_URL"]
            qdrant_api_key = st.secrets["QDRANT_API_KEY"]
        except Exception:
            qdrant_url     = os.environ.get("QDRANT_URL", "")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in Streamlit secrets "
                "or environment variables."
            )

        print(f"[VectorMemory] Connecting to Qdrant Cloud: {qdrant_url}")
        _qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient) -> None:
    """Create collection and payload index if they don't exist."""
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"[VectorMemory] Created collection '{COLLECTION_NAME}' in Qdrant Cloud.")
    else:
        print(f"[VectorMemory] Collection '{COLLECTION_NAME}' already exists.")

    # IMPORTANT: Qdrant Cloud requires a payload index on any field used for filtering.
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="session_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("[VectorMemory] Payload index on 'session_id' created.")
    except Exception as e:
        print(f"[VectorMemory] Payload index already exists or note: {e}")


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def embed(text: str) -> List[float]:
    """Return a 384-dim embedding vector for the given text."""
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


def save_memory(role: str, content: str, session_id: str) -> None:
    """
    Persist a single conversation turn to Qdrant Cloud.
    ── SECURITY: content is passed through MemoryGuard before storing.
    PII is redacted and a flag is stored in the payload.
    Requires a non-empty, session-isolated session_id.
    Rejects errors and blocked response placeholders.
    """
    if not session_id or not session_id.strip():
        print("[VectorMemory] Refusing to save memory without a valid session_id.")
        return
    if not content or not content.strip():
        return

    # Filter out error or blocked response placeholders
    clean_strip = content.strip()
    if clean_strip.startswith(_BLOCKED_OR_ERROR_PREFIXES) or "ran into an exception" in clean_strip:
        print("[VectorMemory] Skipping storage of error/blocked response placeholder.")
        return

    try:
        # ── Security: sanitise before saving ──────────────────────
        from security_guard import memory_guard, audit_logger, mask_session_id
        guard_result = memory_guard.validate(content)
        safe_content = guard_result.clean_text

        if guard_result.event_type == "MEMORY_REDACTED":
            audit_logger.log(
                session_id  = session_id,
                event_type  = "MEMORY_REDACTED",
                detail      = f"PII redacted before saving. Types: {guard_result.findings}",
                findings    = guard_result.findings,
            )
            print(f"[VectorMemory] PII redacted before saving: {guard_result.findings}")

        client   = _get_client()
        vector   = embed(safe_content)
        point_id = str(uuid.uuid4())
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "role":         role,
                        "content":      safe_content,
                        "session_id":   session_id.strip(),
                        "timestamp":    datetime.utcnow().isoformat(),
                        "pii_redacted": guard_result.event_type == "MEMORY_REDACTED",
                    },
                )
            ],
        )
        print(f"[VectorMemory] Saved {role} message for session '{mask_session_id(session_id)}'.")
    except Exception as e:
        print(f"[VectorMemory] Warning — could not save memory: {e}")


def retrieve_relevant_memory(
    query: str,
    session_id: str,
    top_k: int = TOP_K,
    score_threshold: Optional[float] = None,
    max_chars: int = 2500,
) -> str:
    """
    Search Qdrant Cloud for semantically relevant past messages strictly for session_id.
    Caps the maximum returned memory context to max_chars.
    """
    if not session_id or not session_id.strip():
        return ""
    if score_threshold is None:
        score_threshold = SCORE_THRESHOLD

    try:
        client    = _get_client()
        query_vec = embed(query)
        search_filter = Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id.strip()))]
        )
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            query_filter=search_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        if not results:
            return ""
        hits  = sorted(results, key=lambda r: r.payload.get("timestamp", ""))
        lines = ["[Relevant past context retrieved from long-term memory]"]
        for hit in hits:
            p    = hit.payload
            role = p.get("role", "unknown").capitalize()
            ts   = p.get("timestamp", "")[:16].replace("T", " ")
            lines.append(f"{role} ({ts}): {p.get('content', '')}")

        full_context = "\n".join(lines)
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n[...additional memory truncated...]"
        return full_context
    except Exception as e:
        print(f"[VectorMemory] Warning — could not retrieve memory: {e}")
        return ""


def clear_memory(session_id: str) -> None:
    """Delete all memory entries strictly for a given session."""
    if not session_id or not session_id.strip():
        print("[VectorMemory] Refusing to clear memory without a valid session_id.")
        return
    try:
        from security_guard import mask_session_id
        client = _get_client()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id.strip()))]
            ),
        )
        print(f"[VectorMemory] Cleared cloud memory for session '{mask_session_id(session_id)}'.")
    except Exception as e:
        print(f"[VectorMemory] Warning — could not clear memory: {e}")


def get_memory_count(session_id: str) -> int:
    """Returns number of memories stored for a session."""
    if not session_id or not session_id.strip():
        return 0
    try:
        client = _get_client()
        count_result = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id.strip()))]
            ),
            exact=True,
        )
        return count_result.count
    except Exception as e:
        print(f"[VectorMemory] Warning — could not get count: {e}")
        return 0


def get_all_memories(session_id: str) -> List[Dict[str, Any]]:
    """
    Fetch ALL stored messages for a session from Qdrant, sorted by timestamp ascending.
    Returns a list of payload dicts: {role, content, session_id, timestamp}
    Uses scroll (not search) so no query vector needed — fetches everything.
    """
    if not session_id or not session_id.strip():
        return []
    try:
        from security_guard import mask_session_id
        client  = _get_client()
        results = []
        offset  = None

        session_filter = Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id.strip()))]
        )

        while True:
            response, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=session_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in response:
                results.append(point.payload)

            if next_offset is None:
                break
            offset = next_offset

        # Sort by timestamp ascending
        results.sort(key=lambda x: x.get("timestamp", ""))
        print(f"[VectorMemory] Fetched {len(results)} total messages for session '{session_id}'.")
        return results

    except Exception as e:
        print(f"[VectorMemory] Warning — could not fetch all memories: {e}")
        return []