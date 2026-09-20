"""
tests/test_phase5_reliability.py
---------------------------------
Unit and resilience tests for Phase 5: Reliability.
Covers:
  - Timeouts and retry with exponential backoff on model calls.
  - Router fallback chain: Gemini -> Groq -> keyword fallback.
  - Graceful handling when a model fails during comparison (auto-declare surviving model).
  - Randomized A/B order, neutral model labeling in judge prompt, and robust JSON parsing.
  - Web search tool source title and URL citation inclusion.
"""

import pytest
from unittest.mock import MagicMock, patch
import requests

from agent import (
    query_groq,
    query_mistral_judge,
    comparison_and_evaluation_tool,
    router,
    keyword_router_fallback,
    web_search_tool,
    AgentState,
)


# ===========================================================================
# 1. Retry and Backoff on Model Calls
# ===========================================================================

def test_query_groq_retries_on_rate_limit():
    """Verify query_groq retries on 429 status code and returns content on eventual success."""
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = "Rate limit exceeded"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "choices": [{"message": {"content": "Successfully recovered"}}]
    }

    with patch("agent.requests.post", side_effect=[mock_resp_429, mock_resp_200]) as mock_post, \
         patch("agent.time.sleep") as mock_sleep:
        res = query_groq("test prompt", "fake-key", max_retries=3, timeout=5)
        assert res["content"] == "Successfully recovered"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


def test_query_groq_returns_error_dict_on_persistent_failure():
    """Verify query_groq returns a structured error dict on unrecoverable failures."""
    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    mock_resp_503.text = "Service Unavailable"

    with patch("agent.requests.post", return_value=mock_resp_503), \
         patch("agent.time.sleep"):
        res = query_groq("test", "fake-key", max_retries=2, timeout=5)
        assert isinstance(res, dict)
        assert "error" in res
        assert "503" in res["error"]


def test_query_mistral_judge_retries_on_rate_limit():
    """Verify query_mistral_judge retries on 429 and returns judgment."""
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = "Rate limit"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "choices": [{"message": {"content": "Winner: A"}}]
    }

    with patch("agent.requests.post", side_effect=[mock_resp_429, mock_resp_200]) as mock_post, \
         patch("agent.time.sleep") as mock_sleep:
        res = query_mistral_judge("prompt", "fake-key", max_retries=3, timeout=5)
        assert res == "Winner: A"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


# ===========================================================================
# 2. Candidate Error Handling in Comparison Tool
# ===========================================================================

def test_comparison_tool_auto_declares_groq_when_gemini_fails():
    """If Gemini fails, Groq is declared winner automatically without calling judge."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq") as mock_groq, \
         patch("agent.query_mistral_judge") as mock_judge:

        # Gemini fails with exception
        gemini_mock = MagicMock()
        gemini_mock.invoke.side_effect = Exception("Gemini quota 429")
        mock_gemini_cls.return_value = gemini_mock

        # Groq succeeds
        mock_groq.return_value = {
            "model_name": "llama-3.1-8b-instant",
            "content": "Groq successful response.",
        }

        res = comparison_and_evaluation_tool("hi", [], "gkey", "groqkey", "mistralkey")
        assert "🏆 Judged Best Answer (Groq)" in res["display"]
        assert "Groq successful response." in res["display"]
        assert "Gemini (Failed)" in res["display"]
        assert res["memory_text"] == "[llama-3.1-8b-instant]: Groq successful response."
        # Mistral judge should NOT be called when only one model succeeded
        mock_judge.assert_not_called()


def test_comparison_tool_auto_declares_gemini_when_groq_fails():
    """If Groq fails, Gemini is declared winner automatically without calling judge."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq") as mock_groq, \
         patch("agent.query_mistral_judge") as mock_judge:

        # Gemini succeeds
        gemini_mock = MagicMock()
        gemini_mock.invoke.return_value.content = "Gemini successful response."
        mock_gemini_cls.return_value = gemini_mock

        # Groq returns structured error
        mock_groq.return_value = {
            "model_name": "llama-3.1-8b-instant",
            "error": "Groq API Error (500)",
        }

        res = comparison_and_evaluation_tool("hi", [], "gkey", "groqkey", "mistralkey")
        assert "🏆 Judged Best Answer (Gemini)" in res["display"]
        assert "Gemini successful response." in res["display"]
        assert "Groq (Failed)" in res["display"]
        assert res["memory_text"] == "[gemini-2.5-flash]: Gemini successful response."
        mock_judge.assert_not_called()


# ===========================================================================
# 3. Randomized Blind Evaluation and Structured JSON Parsing
# ===========================================================================

def test_comparison_tool_blind_evaluation_prompt_and_json_parsing():
    """Verify judge prompt contains neutral labels (Response A/B) and parses JSON output."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq") as mock_groq, \
         patch("agent.query_mistral_judge") as mock_judge, \
         patch("agent.random.choice", return_value=True):  # Gemini is A, Groq is B

        gemini_mock = MagicMock()
        gemini_mock.invoke.return_value.content = "Gemini response text."
        mock_gemini_cls.return_value = gemini_mock

        mock_groq.return_value = {
            "model_name": "llama-3.1-8b-instant",
            "content": "Groq response text.",
        }

        # Mock structured JSON judge response declaring B (Groq) as winner
        mock_judge.return_value = '{"winner": "B", "reasoning": "Response B is clearer and more concise."}'

        res = comparison_and_evaluation_tool("query", [], "gkey", "groqkey", "mistralkey")

        # Check the judge prompt sent to Mistral
        judge_call_prompt = mock_judge.call_args[0][0]
        assert "### Response A:" in judge_call_prompt
        assert "### Response B:" in judge_call_prompt
        # Neutral labels: must NOT contain 'Response A (Gemini)' or 'Response B (Groq)'
        assert "Response A (Gemini)" not in judge_call_prompt
        assert "Response B (Groq" not in judge_call_prompt

        # Verify Groq was mapped back correctly from "B"
        assert "🏆 Judged Best Answer (Groq)" in res["display"]
        assert res["memory_text"] == "[llama-3.1-8b-instant]: Groq response text."


def test_comparison_tool_falls_back_to_gemini_judge_when_mistral_errors():
    """Verify judge falls back to Gemini 2.5 Flash when Mistral returns HTTP 429/error."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq") as mock_groq, \
         patch("agent.query_mistral_judge") as mock_judge, \
         patch("agent.random.choice", return_value=True):  # Gemini is A, Groq is B

        gemini_candidate_mock = MagicMock()
        gemini_candidate_mock.invoke.return_value.content = "Gemini candidate response."

        gemini_judge_mock = MagicMock()
        gemini_judge_mock.invoke.return_value.content = '{"winner": "A", "reasoning": "Response A is more complete."}'

        # First call is candidate generation, second call is fallback judge
        mock_gemini_cls.side_effect = [gemini_candidate_mock, gemini_judge_mock]

        mock_groq.return_value = {
            "model_name": "llama-3.1-8b-instant",
            "content": "Groq candidate response.",
        }

        # Mistral judge returns HTTP 429 error
        mock_judge.return_value = "Error: The Mistral judge failed to provide an evaluation (HTTP 429)."

        res = comparison_and_evaluation_tool("test prompt", [], "fake_google_key", "fake_groq_key", "fake_mistral_key")

        # Judge source should indicate Gemini Fallback Judge
        assert "🧠 Judge's Evaluation (from Gemini (Fallback Judge))" in res["display"]
        assert "🏆 Judged Best Answer (Gemini)" in res["display"]
        assert res["memory_text"] == "[gemini-2.5-flash]: Gemini candidate response."



# ===========================================================================
# 4. Router Fallback Chain
# ===========================================================================

def test_router_fallback_gemini_to_groq():
    """When Gemini router fails, Groq router is invoked as fallback."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq") as mock_groq:

        # Gemini fails
        gemini_mock = MagicMock()
        gemini_mock.invoke.side_effect = Exception("Gemini down")
        mock_gemini_cls.return_value = gemini_mock

        # Groq succeeds
        mock_groq.return_value = {"content": "web_search_tool"}

        state: AgentState = {"query": "what is the news today", "history": []}
        route_decision = router(state, google_api_key="gkey", groq_api_key="groqkey")
        assert route_decision == {"route": "web_search"}


def test_router_fallback_all_llms_fail_to_keyword():
    """When both Gemini and Groq fail, keyword router takes over."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq", side_effect=Exception("Groq down")):

        # Gemini fails
        gemini_mock = MagicMock()
        gemini_mock.invoke.side_effect = Exception("Gemini down")
        mock_gemini_cls.return_value = gemini_mock

        # Image keyword
        state_img: AgentState = {"query": "generate image of a cybernetic tiger", "history": []}
        res_img = router(state_img, google_api_key="gkey", groq_api_key="groqkey")
        assert res_img == {"route": "image_generator"}

        # Search keyword
        state_search: AgentState = {"query": "what is today's weather in Tokyo?", "history": []}
        res_search = router(state_search, google_api_key="gkey", groq_api_key="groqkey")
        assert res_search == {"route": "web_search"}

        # Default fallback
        state_comp: AgentState = {"query": "Write a quicksort algorithm in Rust", "history": []}
        res_comp = router(state_comp, google_api_key="gkey", groq_api_key="groqkey")
        assert res_comp == {"route": "comparison_chat"}


# ===========================================================================
# 5. Web Search Source Title and URL Passing
# ===========================================================================

def test_web_search_passes_title_and_urls():
    """Verify web_search_tool includes source title and URL markdown links."""
    mock_tavily_cls = MagicMock()
    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search.return_value = {
        "results": [
            {
                "title": "Python 3.12 Release Notes",
                "url": "https://docs.python.org/3.12/",
                "content": "Python 3.12 introduces improved performance.",
            }
        ]
    }
    mock_tavily_cls.return_value = mock_tavily_instance

    with patch("agent.TavilyClient", mock_tavily_cls), \
         patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls:

        gemini_mock = MagicMock()
        gemini_mock.invoke.return_value.content = "According to Python docs, 3.12 is faster."
        mock_gemini_cls.return_value = gemini_mock

        web_search_tool("Python 3.12 features", "tavily_key", "google_key")

        # Verify prompt passed to LLM included Title and URL
        call_prompt = gemini_mock.invoke.call_args[0][0]
        assert "Source: [Python 3.12 Release Notes](https://docs.python.org/3.12/)" in call_prompt
        assert "Python 3.12 introduces improved performance." in call_prompt
