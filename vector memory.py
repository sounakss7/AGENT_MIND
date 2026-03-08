"""
vector_memory.py
----------------
Long-term semantic memory for AGENT_MIND using Qdrant (local) +
sentence-transformers/all-MiniLM-L6-v2 embeddings.

Drop this file in the same directory as agent.py and app.py.
"""

import uuid
import time
from typing import List
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
COLLECTION_NAME = "agent_mind_memory"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM       = 384          # all-MiniLM-L6-v2 output dimension
TOP_K            = 5            # how many past memories to retrieve
QDRANT_PATH      = "./qdrant_storage"   # local disk path; change to ":memory:" for testing


# ---------------------------------------------------------------------------
# SINGLETON HELPERS  (load once per process)
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
        _qdrant = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"[VectorMemory] Created collection '{COLLECTION_NAME}'.")


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def embed(text: str) -> List[float]:
    """Return a 384-dim embedding vector for the given text."""
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


def save_memory(role: str, content: str, session_id: str = "default") -> None:
    """
    Persist a single conversation turn to Qdrant.

    Args:
        role:       "user" or "assistant"
        content:    The message text
        session_id: Optional per-user/session tag for filtering
    """
    if not content or not content.strip():
        return

    client = _get_client()
    vector  = embed(content)
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


def retrieve_relevant_memory(
    query: str,
    session_id: str = "default",
    top_k: int = TOP_K,
    score_threshold: float = 0.35,
) -> str:
    """
    Search Qdrant for the most semantically relevant past messages.

    Returns a formatted string ready to inject into an LLM prompt, or
    an empty string if nothing relevant is found.
    """
    client      = _get_client()
    query_vec   = embed(query)

    # Optional: filter by session so different users don't bleed into each other
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

    # Sort by timestamp so the context reads chronologically
    hits = sorted(results, key=lambda r: r.payload.get("timestamp", ""))

    lines = ["[Relevant past context retrieved from long-term memory]"]
    for hit in hits:
        p    = hit.payload
        role = p.get("role", "unknown").capitalize()
        ts   = p.get("timestamp", "")[:16].replace("T", " ")
        lines.append(f"{role} ({ts}): {p.get('content', '')}")

    return "\n".join(lines)


def clear_memory(session_id: str = "default") -> None:
    """
    Delete all memory entries for a given session.
    Useful for the 'Clear Chat History' button.
    """
    client = _get_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
    )
    print(f"[VectorMemory] Cleared memory for session '{session_id}'.")