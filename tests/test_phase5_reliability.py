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
    SelfRouter,
    self_route_query,
    choose_groq_model,
    promote_candidate_as_winner,
)
from security_guard import evaluation_guard


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
            "model_name": "openai/gpt-oss-20b",
            "content": "Groq successful response.",
        }

        res = comparison_and_evaluation_tool("hi", [], "gkey", "groqkey", "mistralkey")
        assert "🏆 Judged Best Answer (Groq)" in res["display"]
        assert "Groq successful response." in res["display"]
        assert "Gemini (Failed)" in res["display"]
        assert res["memory_text"] == "[openai/gpt-oss-20b]: Groq successful response."
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
            "model_name": "openai/gpt-oss-20b",
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
            "model_name": "openai/gpt-oss-20b",
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
        assert res["memory_text"] == "[openai/gpt-oss-20b]: Groq response text."


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
            "model_name": "openai/gpt-oss-20b",
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


def test_format_judge_evaluation_json_and_latex():
    """Verify format_judge_evaluation extracts reasoning from JSON, dedents, and formats KaTeX math."""
    from agent import format_judge_evaluation

    raw_json = '''{
  "winner": "B",
  "reasoning": "
  **Correctness and Completeness:**
  Both responses correctly derive the equation \\( y = mx + c \\) from constant slope.
  **Winner:** B
  "
}'''

    cleaned = format_judge_evaluation(raw_json)
    # Must NOT have outer JSON braces or keys
    assert '{"winner"' not in cleaned
    assert '"reasoning"' not in cleaned
    # Must unwrap and convert LaTeX math delimiters to KaTeX $
    assert "$ y = mx + c $" in cleaned
    assert "\\( y = mx + c \\)" not in cleaned
    # Must preserve markdown bolding
    assert "**Correctness and Completeness:**" in cleaned
    # Must not have leading code block indentations
    assert not cleaned.startswith("  ")


# ===========================================================================
# 6. DeepSeek & Kimi Provider Integrations & Arena Matchups
# ===========================================================================

def test_query_deepseek_retries_and_success():
    """Verify query_deepseek retries on 429 and returns successful response."""
    from agent import query_deepseek

    mock_resp_429 = MagicMock(status_code=429, text="Rate limit")
    mock_resp_200 = MagicMock(status_code=200)
    mock_resp_200.json.return_value = {
        "choices": [{"message": {"content": "DeepSeek response"}}]
    }

    with patch("requests.post", side_effect=[mock_resp_429, mock_resp_200]) as mock_post, \
         patch("time.sleep"):
        res = query_deepseek("Hi", "fake_deepseek_key", model="deepseek-flash")
        assert mock_post.call_count == 2
        assert res["model_name"] == "deepseek-flash"
        assert res["content"] == "DeepSeek response"


def test_query_kimi_retries_and_error():
    """Verify query_kimi handles quota errors gracefully without raising unhandled exceptions."""
    from agent import query_kimi

    mock_resp_429 = MagicMock(status_code=429, text="Insufficient balance")

    with patch("requests.post", return_value=mock_resp_429) as mock_post, \
         patch("time.sleep"):
        res = query_kimi("Hi", "fake_kimi_key", model="kimi-k3", max_retries=2)
        assert res["model_name"] == "kimi-k3"
        assert "error" in res
        assert "Kimi API Error (429)" in res["error"]


def test_comparison_tool_with_deepseek_and_kimi_matchup():
    """Verify comparison tool runs DeepSeek vs Kimi arena matchup and parses winner correctly."""
    from agent import comparison_and_evaluation_tool

    with patch("agent.query_deepseek") as mock_ds, \
         patch("agent.query_kimi") as mock_km, \
         patch("agent.query_mistral_judge") as mock_judge, \
         patch("agent.random.choice", return_value=True):  # A is DeepSeek, B is Kimi

        mock_ds.return_value = {"model_name": "deepseek-flash", "content": "DeepSeek answer."}
        mock_km.return_value = {"model_name": "kimi-k3", "content": "Kimi answer."}
        mock_judge.return_value = '{"winner": "A", "reasoning": "DeepSeek was more concise."}'

        res = comparison_and_evaluation_tool(
            query="Explain gravity",
            history=[],
            google_api_key="fake_gkey",
            groq_api_key="fake_groqkey",
            mistral_api_key="fake_mistralkey",
            deepseek_api_key="fake_dskey",
            kimi_api_key="fake_kmkey",
            candidate_a_type="deepseek-flash",
            candidate_b_type="kimi-k3",
        )

        assert "🏆 Judged Best Answer (DeepSeek)" in res["display"]
        assert "deepseek-flash" in res["display"]
        assert "Other Response (Kimi)" in res["display"]
        assert res["memory_text"] == "[deepseek-flash]: DeepSeek answer."


def test_comparison_tool_auto_declares_when_deepseek_balance_error():
    """If DeepSeek has HTTP 402 Insufficient Balance, Gemini is declared winner automatically."""
    from agent import comparison_and_evaluation_tool

    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_deepseek") as mock_ds, \
         patch("agent.query_mistral_judge") as mock_judge:

        gemini_mock = MagicMock()
        gemini_mock.invoke.return_value.content = "Gemini answer."
        mock_gemini_cls.return_value = gemini_mock

        mock_ds.return_value = {
            "model_name": "deepseek-flash",
            "error": "DeepSeek API Error (402): Insufficient Balance",
        }

        res = comparison_and_evaluation_tool(
            query="Hello",
            history=[],
            google_api_key="fake_gkey",
            groq_api_key="fake_groqkey",
            mistral_api_key="fake_mistralkey",
            deepseek_api_key="fake_dskey",
            candidate_a_type="deepseek-flash",
            candidate_b_type="gemini",
        )

        # Gemini wins automatically because DeepSeek had an error
        assert "🏆 Judged Best Answer (Gemini)" in res["display"]
        assert "DeepSeek (Failed)" in res["display"]
        assert "Gemini selected automatically because DeepSeek encountered an error." in res["display"]
        mock_judge.assert_not_called()


# ===========================================================================
# 6. Groq Model Selection & Fallback
# ===========================================================================

def test_choose_groq_model_uses_valid_models():
    """Verify choose_groq_model returns valid production models (70B or 8B)."""
    complex_query = "Write a comprehensive Python script with asyncio to solve Dijkstra's algorithm."
    model_complex = choose_groq_model(complex_query)
    assert model_complex == "openai/gpt-oss-120b"

    simple_query = "Hello, what is your name?"
    model_simple = choose_groq_model(simple_query)
    assert model_simple == "openai/gpt-oss-20b"


# ===========================================================================
# 7. Arena Judge Selection & Human-in-the-Loop Override
# ===========================================================================

def test_human_judge_mode():
    """Verify judge_type='human' presents both candidates side-by-side without calling AI judges."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq") as mock_groq, \
         patch("agent.query_mistral_judge") as mock_judge:

        gemini_mock = MagicMock()
        gemini_mock.invoke.return_value.content = "Candidate A content from Gemini."
        mock_gemini_cls.return_value = gemini_mock

        mock_groq.return_value = {
            "model_name": "openai/gpt-oss-120b",
            "content": "Candidate B content from Groq.",
        }

        res = comparison_and_evaluation_tool(
            query="Explain recursion",
            history=[],
            google_api_key="fake_gkey",
            groq_api_key="fake_groqkey",
            mistral_api_key="fake_mistralkey",
            candidate_a_type="gemini",
            candidate_b_type="groq",
            judge_type="human",
        )

        assert res.get("is_human_judge") is True
        assert "🧑 Human Judge Arena: You Decide!" in res["display"]
        assert "Candidate A content from Gemini." in res["display"]
        assert "Candidate B content from Groq." in res["display"]
        mock_judge.assert_not_called()


def test_promote_candidate_as_winner():
    """Verify promote_candidate_as_winner formats selected winner correctly."""
    comp_data = {
        "winner_name": "Candidate A",
        "winner_model": "gemini-2.5-flash",
        "winner_answer": "Answer A text.",
        "loser_name": "Candidate B",
        "loser_model": "openai/gpt-oss-120b",
        "loser_answer": "Answer B text.",
    }

    # Promote Candidate B as winner
    swapped_text, new_mem = promote_candidate_as_winner("B", comp_data)
    assert "🏆 Human-Selected Best Answer (Candidate B)" in swapped_text
    assert "Answer B text." in swapped_text
    assert "Other Response (Candidate A)" in swapped_text
    assert new_mem == "[openai/gpt-oss-120b]: Answer B text."


# ===========================================================================
# 8. Multi-Dimensional Rubric Evaluation Guardrails
# ===========================================================================

def test_evaluation_guardrail_scoring():
    """Verify evaluation guardrail produces rubric scores and valid grades."""
    query = "Explain quicksort with code"
    response = """
    Quicksort is a divide-and-conquer algorithm.
    ```python
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    ```
    """
    scores = evaluation_guard.evaluate_response(query, response)
    assert "overall" in scores
    assert "clarity" in scores
    assert "completeness" in scores
    assert "adherence" in scores
    assert "rubric_grade" in scores
    assert 0 <= scores["overall"] <= 100
    assert any(g in scores["rubric_grade"] for g in ["A+", "A", "B+", "B", "C", "D"])


# ===========================================================================
# 9. Self-Routing Engine (Zero-API Intent Classification)
# ===========================================================================

def test_self_router_rules():
    """Verify SelfRouter correctly routes image, search, negative guards, and chat."""
    # Image commands & natural prompts
    assert SelfRouter.route("/image cybernetic tiger")["route"] == "image_generator"
    assert SelfRouter.route("generate image of a futuristic neon city")["route"] == "image_generator"
    assert SelfRouter.route("draw a cute kitten sitting on a mat")["route"] == "image_generator"
    assert SelfRouter.route("picture of a sunset over the ocean")["route"] == "image_generator"

    # Negative guard: coding/explanation asking about images
    assert SelfRouter.route("how to generate an image using python")["route"] == "comparison_chat"
    assert SelfRouter.route("explain how diffusion models generate images")["route"] == "comparison_chat"

    # Search commands & live queries
    assert SelfRouter.route("/search quantum computing breakthrough")["route"] == "web_search"
    assert SelfRouter.route("what is today's weather in Tokyo?")["route"] == "web_search"
    assert SelfRouter.route("what is the news today")["route"] == "web_search"
    assert SelfRouter.route("who won the 2026 super bowl")["route"] == "web_search"
    assert SelfRouter.route("bitcoin stock price right now")["route"] == "web_search"

    # Negative guard: coding asking about search
    assert SelfRouter.route("how to write binary search in python")["route"] == "comparison_chat"
    assert SelfRouter.route("write quicksort algorithm in Rust")["route"] == "comparison_chat"
    assert SelfRouter.route("derive y = mx + c")["route"] == "comparison_chat"


def test_router_self_mode_bypasses_all_llms():
    """Verify router with routing_mode='self' executes locally without calling Gemini or Groq."""
    with patch("agent.ChatGoogleGenerativeAI") as mock_gemini_cls, \
         patch("agent.query_groq") as mock_groq:

        state: AgentState = {
            "query": "generate image of a red race car",
            "history": [],
            "routing_mode": "self",
        }

        res = router(state, google_api_key="fake_key", groq_api_key="fake_key")
        assert res == {"route": "image_generator"}

        # Gemini and Groq should NEVER have been called
        mock_gemini_cls.assert_not_called()
        mock_groq.assert_not_called()



