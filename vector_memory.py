"""
vector_memory.py
----------------
Long-term semantic memory for AGENT_MIND using Qdrant Cloud +
sentence-transformers/all-MiniLM-L6-v2 embeddings.

Credentials are read from Streamlit secrets:
  QDRANT_URL     — your Qdrant Cloud cluster URL
  QDRANT_API_KEY — your Qdrant Cloud API key
"""

import os
import uuid
from typing import List, Dict, Any
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
SCORE_THRESHOLD = 0.35

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


def save_memory(role: str, content: str, session_id: str = "default") -> None:
    """Persist a single conversation turn to Qdrant Cloud."""
    if not content or not content.strip():
        return
    try:
        client   = _get_client()
        vector   = embed(content)
        point_id = str(uuid.uuid4())
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "role":       role,
                        "content":    content,
                        "session_id": session_id,
                        "timestamp":  datetime.utcnow().isoformat(),
                    },
                )
            ],
        )
        print(f"[VectorMemory] Saved {role} message for session '{session_id}'.")
    except Exception as e:
        print(f"[VectorMemory] Warning — could not save memory: {e}")


def retrieve_relevant_memory(
    query: str,
    session_id: str = "default",
    top_k: int = TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
) -> str:
    """Search Qdrant Cloud for semantically relevant past messages."""
    try:
        client    = _get_client()
        query_vec = embed(query)
        search_filter = Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
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
        return "\n".join(lines)
    except Exception as e:
        print(f"[VectorMemory] Warning — could not retrieve memory: {e}")
        return ""


def clear_memory(session_id: str = "default") -> None:
    """Delete all memory entries for a given session."""
    try:
        client = _get_client()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
        )
        print(f"[VectorMemory] Cleared cloud memory for session '{session_id}'.")
    except Exception as e:
        print(f"[VectorMemory] Warning — could not clear memory: {e}")


def get_memory_count(session_id: str = "default") -> int:
    """Returns number of memories stored for a session."""
    try:
        client = _get_client()
        count_result = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            exact=True,
        )
        return count_result.count
    except Exception as e:
        print(f"[VectorMemory] Warning — could not get count: {e}")
        return 0


def get_all_memories(session_id: str = "default") -> List[Dict[str, Any]]:
    """
    Fetch ALL stored messages for a session from Qdrant, sorted by timestamp ascending.
    Returns a list of payload dicts: {role, content, session_id, timestamp}
    Uses scroll (not search) so no query vector needed — fetches everything.
    """
    try:
        client  = _get_client()
        results = []
        offset  = None

        session_filter = Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
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