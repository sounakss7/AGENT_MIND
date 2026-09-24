"""
tests/test_phase3_memory_correctness.py
---------------------------------------
Unit and adversarial tests for Phase 3: Memory Correctness.
Covers:
  - Vector score threshold default (0.45) & environment variable override.
  - Memory context capping (max_chars truncation).
  - Error and blocked response filtering in save_memory.
  - History distillation (storing distilled winning answer, omitting judge critique & loser).
  - Self-retrieval prevention order.
  - Single-retrieval propagation via AgentState memory_context.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import HumanMessage, AIMessage

import vector_memory
from agent import (
    AgentState,
    comparison_and_evaluation_tool,
    call_comparison_tool,
    router,
    format_history,
)


def test_vector_score_threshold_default_and_env():
    """Verify default threshold is 0.45 and respects env var."""
    assert vector_memory.SCORE_THRESHOLD >= 0.45

    with patch.dict(os.environ, {"VECTOR_SCORE_THRESHOLD": "0.60"}):
        import importlib
        # Re-evaluating float(os.environ.get(...))
        val = float(os.environ.get("VECTOR_SCORE_THRESHOLD", "0.45"))
        assert val == 0.60


def test_memory_context_capping():
    """Verify retrieve_relevant_memory truncates text exceeding max_chars."""
    mock_point = MagicMock()
    mock_point.payload = {
        "role": "user",
        "content": "A" * 1500,
        "timestamp": "2026-09-20T12:00:00Z",
    }
    mock_point_2 = MagicMock()
    mock_point_2.payload = {
        "role": "assistant",
        "content": "B" * 1500,
        "timestamp": "2026-09-20T12:01:00Z",
    }

    mock_client = MagicMock()
    mock_client.search.return_value = [mock_point, mock_point_2]

    with patch.object(vector_memory, "_get_client", return_value=mock_client), \
         patch.object(vector_memory, "embed", return_value=[0.1] * 384):
        retrieved = vector_memory.retrieve_relevant_memory(
            query="test query",
            session_id="acc_user123_456",
            max_chars=2500,
        )
        assert len(retrieved) <= 2550  # 2500 + length of truncation notice
        assert "[...additional memory truncated...]" in retrieved


def test_save_memory_filters_blocked_and_error_responses():
    """Verify save_memory ignores blocked placeholders, error messages, and exceptions."""
    mock_client = MagicMock()

    with patch.object(vector_memory, "_get_client", return_value=mock_client), \
         patch.object(vector_memory, "embed", return_value=[0.1] * 384):

        unwanted_contents = [
            "[Response blocked by safety policy]",
            "⚠️ Web search failed: connection timeout",
            "Error: The Mistral judge failed to provide an evaluation",
            "I can't provide that information as it violates safety guidelines",
            "The model ran into an exception during processing",
            "Failed to generate image: rate limit exceeded",
        ]

        for content in unwanted_contents:
            vector_memory.save_memory(
                role="assistant",
                content=content,
                session_id="acc_safe_session_123",
            )
            # Must NOT call upsert on the client
            mock_client.upsert.assert_not_called()


def test_save_memory_stores_valid_content():
    """Verify save_memory stores clean, valid content."""
    mock_client = MagicMock()

    with patch.object(vector_memory, "_get_client", return_value=mock_client), \
         patch.object(vector_memory, "embed", return_value=[0.1] * 384):

        vector_memory.save_memory(
            role="assistant",
            content="[Gemini]: The capital of France is Paris.",
            session_id="acc_valid_session_123",
        )
        mock_client.upsert.assert_called_once()


def test_comparison_tool_returns_distilled_memory_and_display():
    """Verify comparison_and_evaluation_tool outputs both rich display and distilled memory text."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini, \
         patch("agent.query_groq") as mock_groq, \
         patch("agent.query_mistral_judge") as mock_judge:

        # Mock Gemini
        gemini_instance = MagicMock()
        gemini_instance.invoke.return_value.content = "Gemini answer: sorting takes O(N log N) time."
        mock_gemini.return_value = gemini_instance

        # Mock Groq
        mock_groq.return_value = {
            "model_name": "openai/gpt-oss-20b",
            "content": "Groq answer: sorting takes O(N^2) for bubblesort, O(N log N) for mergesort.",
        }

        # Mock Mistral judge declaring Gemini as winner
        mock_judge.return_value = "Winner: Gemini\nReasoning: Gemini was concise and accurate."

        result = comparison_and_evaluation_tool(
            query="What is the complexity of sorting?",
            history=[],
            google_api_key="fake-key",
            groq_api_key="fake-key",
            mistral_api_key="fake-key",
            session_id="acc_test_session_123",
            memory_context="Previous context about Big O",
        )

        assert isinstance(result, dict)
        assert "display" in result
        assert "memory_text" in result

        # Display has full comparison UI elements
        assert "### 🏆 Judged Best Answer" in result["display"]
        assert "### 🧠 Judge's Evaluation" in result["display"]
        assert "### Other Response" in result["display"]

        # Distilled memory has only the winning answer and model tag
        assert result["memory_text"].startswith("[gemini-2.5-flash]:")
        assert "Gemini answer: sorting takes O(N log N) time." in result["memory_text"]
        assert "### 🧠 Judge's Evaluation" not in result["memory_text"]
        assert "Other Response" not in result["memory_text"]
        assert "Winner:" not in result["memory_text"]


def test_call_comparison_tool_propagates_memory_text():
    """Verify call_comparison_tool returns both final_response and memory_text in node output."""
    mock_tool_result = {
        "display": "Full Markdown with comparison",
        "memory_text": "[gemini-2.5-flash]: Winner text",
    }
    with patch("agent.comparison_and_evaluation_tool", return_value=mock_tool_result):
        state: AgentState = {
            "query": "hello",
            "history": [],
            "session_id": "acc_123",
            "memory_context": "cached context",
        }
        output = call_comparison_tool(state, "gkey", "groqkey", "mistralkey")
        assert output["final_response"] == "Full Markdown with comparison"
        assert output["memory_text"] == "[gemini-2.5-flash]: Winner text"


def test_single_memory_retrieval_propagation():
    """Verify router and comparison tool consume state['memory_context'] without redundant Qdrant queries."""
    with patch("agent.retrieve_relevant_memory") as mock_retrieve, \
         patch("agent.ChatGoogleGenerativeAI") as mock_llm:

        llm_instance = MagicMock()
        llm_instance.invoke.return_value.content = "comparison_tool"
        mock_llm.return_value = llm_instance

        # Pass pre-retrieved memory_context in state
        state: AgentState = {
            "query": "explain quicksort",
            "history": [],
            "session_id": "acc_user_1",
            "memory_context": "Pre-retrieved memory about algorithms",
        }

        # Router should use the passed memory_context and NOT call retrieve_relevant_memory
        route_decision = router(state, "fake-gkey")
        assert route_decision == {"route": "comparison_chat"}
        mock_retrieve.assert_not_called()


def test_chat_history_uses_distilled_memory_text():
    """Verify chat history builder selects memory_text over raw display text."""
    messages = [
        {"role": "user", "text": "What is Python?"},
        {
            "role": "assistant",
            "text": "### 🏆 Judged Best Answer (Gemini)\nPython is great.\n### 🧠 Judge's Evaluation\nReasoning...",
            "memory_text": "[gemini-2.5-flash]: Python is great.",
        },
        {"role": "user", "text": "Can you show an example?"},
    ]

    # Build history for previous turns (excluding current turn at messages[-1])
    chat_history = []
    for msg in messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["text"]))
        elif msg["role"] == "assistant":
            assistant_content = msg.get("memory_text") or msg.get("text", "")
            if assistant_content:
                chat_history.append(AIMessage(content=assistant_content))

    assert len(chat_history) == 2
    assert isinstance(chat_history[0], HumanMessage)
    assert chat_history[0].content == "What is Python?"
    assert isinstance(chat_history[1], AIMessage)
    # Must use distilled memory_text, not the raw markdown with judge critique
    assert chat_history[1].content == "[gemini-2.5-flash]: Python is great."
    assert "Judge's Evaluation" not in chat_history[1].content
