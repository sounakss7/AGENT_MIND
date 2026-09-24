import os
import re
import time
import json
import random
import requests
from io import BytesIO
from PIL import Image
from typing import TypedDict, Optional, List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
import concurrent.futures
from functools import partial
from tavily import TavilyClient
from urllib.parse import quote_plus
import logging
import hashlib

# Document processing fallbacks
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

# =======================================================================================
# VECTOR MEMORY IMPORT
# =======================================================================================
from vector_memory import retrieve_relevant_memory

# =======================================================================================
# SECURITY LAYER IMPORT
# =======================================================================================
from security_guard import input_guard, output_guard, audit_logger, wrap_untrusted_data, evaluation_guard

# =======================================================================================
# MODEL BENCHMARKS MATRIX
# =======================================================================================
MODEL_BENCHMARKS = {
    "Gemini 2.5 Flash": {
        "provider": "Google",
        "mmlu": "82.0%",
        "math": "67.7%",
        "humaneval": "74.4%",
        "speed": "~0.8s latency",
        "throughput": "~120 t/s",
        "context": "1,000,000 tokens",
        "best_for": "Multimodal tasks, ultra-long context (1M), and rapid reasoning.",
    },
    "Groq GPT-OSS 120B": {
        "provider": "OpenAI / Groq",
        "mmlu": "87.5%",
        "math": "69.0%",
        "humaneval": "82.0%",
        "speed": "~250 tokens/sec",
        "throughput": "Ultra-fast LPU",
        "context": "128,000 tokens",
        "best_for": "High-complexity reasoning, coding, and high-speed large-scale generation.",
    },
    "Groq GPT-OSS 20B": {
        "provider": "OpenAI / Groq",
        "mmlu": "75.0%",
        "math": "54.0%",
        "humaneval": "65.0%",
        "speed": "~800 tokens/sec",
        "throughput": "Hyper-speed LPU",
        "context": "128,000 tokens",
        "best_for": "Real-time fast responses, low-latency chat, and concise answers.",
    },
    "DeepSeek V3 (Chat)": {
        "provider": "DeepSeek",
        "mmlu": "88.5%",
        "math": "75.9%",
        "humaneval": "82.6%",
        "speed": "~1.5s latency",
        "throughput": "~60 t/s",
        "context": "64,000 tokens",
        "best_for": "Advanced mathematical proofs, algorithmic coding, and analytical depth.",
    },
    "Kimi K3 (Moonshot)": {
        "provider": "Moonshot AI",
        "mmlu": "84.2%",
        "math": "62.5%",
        "humaneval": "76.0%",
        "speed": "~1.8s latency",
        "throughput": "~50 t/s",
        "context": "200,000 tokens",
        "best_for": "Long document synthesis, 200k context window, and conversational continuity.",
    },
    "Mistral Small (Judge)": {
        "provider": "Mistral AI",
        "mmlu": "81.2%",
        "math": "60.4%",
        "humaneval": "71.8%",
        "speed": "~1.0s latency",
        "throughput": "~90 t/s",
        "context": "32,000 tokens",
        "best_for": "Impartial evaluation, rubric scoring, and objective arbitration.",
    },
}

# =======================================================================================
# HELPER FUNCTIONS
# =======================================================================================

def format_history(history: List[BaseMessage]) -> str:
    """Formats the last few turns of history into a string for the LLM."""
    if not history:
        return "No previous context."
    recent_history = history[-4:]
    formatted = ""
    for msg in recent_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted += f"{role}: {msg.content}\n"
    return formatted


# Groq deprecated llama-3.3-70b-versatile and llama-3.1-8b-instant on Aug 16 2026.
# New recommended production models:
#   Large/Complex tasks: openai/gpt-oss-120b  (replaces llama-3.3-70b-versatile)
#   Fast/Simple tasks:   openai/gpt-oss-20b   (replaces llama-3.1-8b-instant)
GROQ_MODEL_LARGE = "openai/gpt-oss-120b"
GROQ_MODEL_FAST  = "openai/gpt-oss-20b"


def choose_groq_model(prompt: str) -> str:
    """Selects the best Groq model based on the complexity of the prompt."""
    p = prompt.lower()
    if any(x in p for x in ["python", "code", "algorithm", "bug", "function", "script",
                             "information", "analysis", "solution", "nlp", "essay",
                             "mathematics", "research", "reasoning", "benchmark", "derive",
                             "explain", "thermodynamics", "physics"]):
        return GROQ_MODEL_LARGE   # openai/gpt-oss-120b
    else:
        return GROQ_MODEL_FAST    # openai/gpt-oss-20b


def query_groq(prompt: str, groq_api_key: str, max_retries: int = 3, timeout: int = 30, preferred_model: Optional[str] = None):
    """
    Queries the Groq API with retries, exponential backoff, and automatic fallback.
    Falls back from openai/gpt-oss-120b -> openai/gpt-oss-20b on rate limits or errors.
    Returns a dict with 'model_name' and either 'content' or 'error'.
    """
    if not groq_api_key:
        return {"model_name": "Groq", "error": "No Groq API key provided."}

    primary_model = preferred_model or choose_groq_model(prompt)
    candidate_models = [primary_model]
    if primary_model != GROQ_MODEL_FAST:
        candidate_models.append(GROQ_MODEL_FAST)

    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    last_error = "Groq API exceeded max retries."

    for model in candidate_models:
        delay = 1.0
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}
        for attempt in range(max_retries):
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return {"model_name": model, "content": content}
                elif resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    logging.warning(f"Groq ({model}) HTTP {resp.status_code}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    last_error = f"Groq API Error ({resp.status_code}): {resp.text}"
                    break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Groq network error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                last_error = f"Groq Timeout/Connection Error: {e}"
                break
            except Exception as e:
                last_error = f"Groq Error: {e}"
                break

    return {"model_name": primary_model, "error": last_error}


def query_deepseek(
    prompt: str,
    deepseek_api_key: str,
    model: str = "deepseek-flash",
    max_retries: int = 3,
    timeout: int = 30,
) -> dict:
    """
    Queries DeepSeek API (OpenAI compatible) with retries and exponential backoff.
    Maps 'deepseek-flash' to production model 'deepseek-chat' (DeepSeek-V3).
    Endpoint: https://api.deepseek.com/chat/completions
    """
    if not deepseek_api_key:
        return {"model_name": model, "error": "No DeepSeek API key provided."}

    actual_model = "deepseek-chat" if model in ("deepseek-flash", "deepseek-chat") else model
    headers = {"Authorization": f"Bearer {deepseek_api_key}", "Content-Type": "application/json"}
    data = {"model": actual_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.post("https://api.deepseek.com/chat/completions", json=data, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return {"model_name": model, "content": content}
            elif resp.status_code == 402:
                return {"model_name": model, "error": "DeepSeek API Error (402): Insufficient Balance"}
            elif resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                logging.warning(f"DeepSeek API {resp.status_code} received. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            else:
                return {"model_name": model, "error": f"DeepSeek API Error ({resp.status_code}): {resp.text}"}
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                logging.warning(f"DeepSeek network error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            return {"model_name": model, "error": f"DeepSeek Timeout/Connection Error: {e}"}
        except Exception as e:
            return {"model_name": model, "error": f"DeepSeek Error: {e}"}

    return {"model_name": model, "error": "DeepSeek API exceeded max retries."}


def query_kimi(
    prompt: str,
    kimi_api_key: str,
    model: str = "kimi-k3",
    max_retries: int = 3,
    timeout: int = 30,
) -> dict:
    """
    Queries Kimi / Moonshot API (OpenAI compatible) with retries and exponential backoff.
    Maps 'kimi-k3' to production model 'moonshot-v1-8k'.
    Endpoint: https://api.moonshot.ai/v1/chat/completions (with cn fallback)
    """
    if not kimi_api_key:
        return {"model_name": model, "error": "No Kimi API key provided."}

    actual_model = "moonshot-v1-8k" if model in ("kimi-k3", "moonshot-v1-8k") else model
    headers = {"Authorization": f"Bearer {kimi_api_key}", "Content-Type": "application/json"}
    data = {"model": actual_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}

    endpoints = [
        "https://api.moonshot.ai/v1/chat/completions",
        "https://api.moonshot.cn/v1/chat/completions",
    ]

    last_err = "Kimi API exceeded max retries."
    for endpoint in endpoints:
        delay = 1.0
        for attempt in range(max_retries):
            try:
                resp = requests.post(endpoint, json=data, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return {"model_name": model, "content": content}
                elif resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    logging.warning(f"Kimi API {resp.status_code} received on {endpoint}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    last_err = f"Kimi API Error ({resp.status_code}): {resp.text}"
                    break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Kimi network error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                last_err = f"Kimi Timeout/Connection Error: {e}"
                break
            except Exception as e:
                return {"model_name": model, "error": f"Kimi Error: {e}"}

    return {"model_name": model, "error": last_err}


def query_mistral_judge(prompt: str, mistral_api_key: str, max_retries: int = 3, timeout: int = 30):
    """
    Queries Mistral judge with retries and exponential backoff on rate limits / server errors.
    Uses open-mistral-7b by default (free-tier compatible) with fallback to ministral-8b-latest.
    """
    if not mistral_api_key:
        return "Error: No Mistral API key provided."

    models_to_try = ["open-mistral-7b", "ministral-8b-latest"]
    headers = {"Authorization": f"Bearer {mistral_api_key}", "Content-Type": "application/json"}

    last_error = ""
    for model in models_to_try:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        }
        delay = 1.0
        for attempt in range(max_retries):
            try:
                resp = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                elif resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    logging.warning(f"Mistral Judge ({model}) {resp.status_code} received. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    logging.error(f"Mistral Judge ({model}) HTTP Error: {resp.status_code} - {resp.text}")
                    last_error = f"HTTP {resp.status_code}"
                    break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Mistral Judge ({model}) network error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                logging.error(f"Mistral Judge ({model}) Timeout/Connection Error: {e}")
                last_error = f"Timeout: {e}"
                break
            except Exception as e:
                logging.error(f"Mistral Judge ({model}) Exception: {e}")
                last_error = str(e)
                break

    return f"Error: The Mistral judge failed to provide an evaluation ({last_error})."


def format_judge_evaluation(judgment: str) -> str:
    """
    Extracts reasoning and formats evaluation from judge response (JSON or plain text).
    - Unwraps JSON {"winner": ..., "reasoning": ...} to extract clean markdown text.
    - Strips code fences (```json ... ```) if present.
    - Unindents lines to prevent accidental <pre><code> block formatting in markdown.
    - Converts LaTeX math delimiters \\( \\) -> $ and \\[ \\] -> $$ for Streamlit KaTeX rendering.
    """
    if not judgment or not isinstance(judgment, str):
        return ""

    reasoning = ""
    try:
        match = re.search(r"\{[\s\S]*?\}", judgment)
        if match:
            json_str = match.group(0)
            data = None
            try:
                data = json.loads(json_str)
            except Exception:
                fixed_json = re.sub(r'(?<!\\)\n', r'\\n', json_str)
                try:
                    data = json.loads(fixed_json)
                except Exception:
                    pass
            if data and isinstance(data, dict):
                reasoning = data.get("reasoning", "")
    except Exception:
        pass

    if not reasoning:
        reason_match = re.search(r'"reasoning"\s*:\s*"([\s\S]*?)"\s*\}', judgment)
        if reason_match:
            reasoning = reason_match.group(1)
        else:
            reasoning = re.sub(r"^```(?:json)?\s*", "", judgment.strip(), flags=re.MULTILINE)
            reasoning = re.sub(r"\s*```$", "", reasoning.strip(), flags=re.MULTILINE)

    if not reasoning.strip():
        reasoning = judgment

    # Dedent / trim lines so leading spaces don't trigger markdown code blocks
    lines = [line.strip() for line in reasoning.splitlines()]
    reasoning_clean = "\n".join(lines).strip()

    # Convert LaTeX delimiters \( \) -> $ and \[ \] -> $$ for Streamlit KaTeX math rendering
    reasoning_clean = re.sub(r'\\\((.*?)\\\)', r'$\1$', reasoning_clean)
    reasoning_clean = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', reasoning_clean, flags=re.DOTALL)

    return reasoning_clean


# =======================================================================================
# TOOL 1: COMPARISON & EVALUATION  (memory-aware + output guarded)
# =======================================================================================

def comparison_and_evaluation_tool(
    query: str,
    history: List[BaseMessage],
    google_api_key: str,
    groq_api_key: str,
    mistral_api_key: str,
    session_id: str = "default",
    memory_context: Optional[str] = None,
    deepseek_api_key: str = "",
    kimi_api_key: str = "",
    candidate_a_type: str = "gemini",
    candidate_b_type: str = "groq",
    judge_type: str = "mistral",
    enable_fallbacks: bool = True,
) -> dict:
    print(f"---TOOL: Executing Comparison ({candidate_a_type} vs {candidate_b_type} Judged by {judge_type})---")

    short_term_ctx = format_history(history)
    long_term_ctx  = memory_context if memory_context is not None else retrieve_relevant_memory(query, session_id=session_id)
    if long_term_ctx:
        print(f"[VectorMemory] Using memory context for comparison tool.")
        safe_memory_ctx = wrap_untrusted_data(long_term_ctx, "LONG_TERM_MEMORY", session_id=session_id)
    else:
        safe_memory_ctx = "None available."

    full_prompt_with_context = f"""
LONG-TERM MEMORY (semantically relevant past interactions):
{safe_memory_ctx}

SHORT-TERM CONTEXT (recent conversation turns):
{short_term_ctx}

CURRENT USER REQUEST:
{query}

Instructions: Respond to the CURRENT USER REQUEST.
Use the short-term context for immediate follow-ups ("rewrite that", "fix the bug").
Use the long-term memory only when the user refers to something discussed in a previous session.
"""

    fast_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key) if google_api_key else None
    gemini_model_name = "gemini-2.5-flash"

    def _safe_gemini():
        if not fast_llm:
            return {"model_name": gemini_model_name, "error": "Gemini API key not configured."}
        try:
            return fast_llm.invoke(full_prompt_with_context).content
        except Exception as e:
            logging.error(f"Gemini generation error: {e}")
            return {"error": f"Gemini Error: {e}"}

    def _execute_model(m_type: str, other_m_type: str = ""):
        mt = m_type.lower()
        res = None
        if "deepseek" in mt:
            res = query_deepseek(full_prompt_with_context, deepseek_api_key, model="deepseek-flash")
        elif "kimi" in mt or "moonshot" in mt:
            res = query_kimi(full_prompt_with_context, kimi_api_key, model="kimi-k3")
        elif "groq" in mt or "llama" in mt:
            res = query_groq(full_prompt_with_context, groq_api_key)
        else:
            res = _safe_gemini()

        # Automatic cross-model fallback if model returned an error
        if enable_fallbacks and isinstance(res, dict) and "error" in res:
            err_msg = str(res["error"])
            logging.warning(f"Contender {m_type} failed: {err_msg}. Attempting fallback...")
            # If DeepSeek or Kimi failed:
            if "deepseek" in mt or "kimi" in mt or "moonshot" in mt:
                # Fallback to Groq if key provided and Groq is not the other contender
                if groq_api_key and "groq" not in other_m_type.lower() and "llama" not in other_m_type.lower():
                    g_res = query_groq(full_prompt_with_context, groq_api_key)
                    if isinstance(g_res, dict) and "content" in g_res:
                        g_res["model_name"] = f"Groq (Fallback for {m_type})"
                        return g_res
                # Fallback to Gemini if key provided and Gemini is not the other contender
                if google_api_key and "gemini" not in other_m_type.lower():
                    gem_res = _safe_gemini()
                    if isinstance(gem_res, str):
                        return {"model_name": f"Gemini 2.5 Flash (Fallback for {m_type})", "content": gem_res}
            elif "groq" in mt or "llama" in mt:
                # If Groq failed, try Gemini if not the other contender
                if google_api_key and "gemini" not in other_m_type.lower():
                    gem_res = _safe_gemini()
                    if isinstance(gem_res, str):
                        return {"model_name": "Gemini 2.5 Flash (Fallback for Groq)", "content": gem_res}

        return res

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_a = executor.submit(_execute_model, candidate_a_type, candidate_b_type)
        future_b = executor.submit(_execute_model, candidate_b_type, candidate_a_type)
        result_a = future_a.result()
        result_b = future_b.result()

    # Determine errors on each side
    has_err_a = isinstance(result_a, dict) and "error" in result_a
    has_err_b = isinstance(result_b, dict) and "error" in result_b

    def _extract_model_and_label(m_type: str, res: any):
        mt = m_type.lower()
        if isinstance(res, dict) and "model_name" in res:
            m_name = res["model_name"]
            if "fallback" in m_name.lower():
                return m_name, m_name.split()[0]
        if "deepseek" in mt:
            m_name = res.get("model_name", "deepseek-flash") if isinstance(res, dict) else "deepseek-flash"
            return m_name, "DeepSeek"
        elif "kimi" in mt or "moonshot" in mt:
            m_name = res.get("model_name", "kimi-k3") if isinstance(res, dict) else "kimi-k3"
            return m_name, "Kimi"
        elif "groq" in mt or "llama" in mt:
            m_name = res.get("model_name", "Groq") if isinstance(res, dict) else "Groq"
            return m_name, "Groq"
        else:
            return gemini_model_name, "Gemini"

    model_name_a, label_a = _extract_model_and_label(candidate_a_type, result_a)
    model_name_b, label_b = _extract_model_and_label(candidate_b_type, result_b)

    text_a = result_a["error"] if has_err_a else (result_a.get("content", "") if isinstance(result_a, dict) else str(result_a or ""))
    text_b = result_b["error"] if has_err_b else (result_b.get("content", "") if isinstance(result_b, dict) else str(result_b or ""))
    judge_source = "Automated Selection"

    # Case 1: Both models failed
    if has_err_a and has_err_b:
        err_msg = f"### ⚠️ Both models failed to respond.\n\n- **{label_a}:** {text_a}\n- **{label_b}:** {text_b}"
        return {"display": err_msg, "memory_text": ""}

    # Case 2: Model A failed, Model B succeeded
    if has_err_a and not has_err_b:
        chosen_answer, chosen_model_name = text_b, model_name_b
        winner_name = label_b
        loser_response, loser_model_name, loser_name = text_a, model_name_a, f"{label_a} (Failed)"
        judgment_clean = f"{label_b} selected automatically because {label_a} encountered an error."
        judge_source = "Automated (Single Candidate)"

    # Case 3: Model B failed, Model A succeeded
    elif has_err_b and not has_err_a:
        chosen_answer, chosen_model_name = text_a, model_name_a
        winner_name = label_a
        loser_response, loser_model_name, loser_name = text_b, model_name_b, f"{label_b} (Failed)"
        judgment_clean = f"{label_a} selected automatically because {label_b} encountered an error."
        judge_source = "Automated (Single Candidate)"

    # Case 4: Both succeeded — call selected judge
    else:
        rubric_a = evaluation_guard.evaluate_response(query, text_a)
        rubric_b = evaluation_guard.evaluate_response(query, text_b)

        if judge_type == "human":
            human_text = (
                f"### 🧑 Human Judge Arena: You Decide!\n\n"
                f"Compare both candidate model responses below and decide which one should be promoted as the winner.\n\n"
                f"### Candidate A ({label_a})\n"
                f"#### Model: {model_name_a}\n\n{text_a}\n\n---\n\n"
                f"### Candidate B ({label_b})\n"
                f"#### Model: {model_name_b}\n\n{text_b}\n\n---\n\n"
                f"### 📊 Automated Multi-Dimensional Evals\n"
                f"- **{label_a} ({model_name_a}):** Overall Quality: **{rubric_a['overall']}%** (`{rubric_a['rubric_grade']}`) | Clarity: **{rubric_a['clarity']}%** | Depth: **{rubric_a['completeness']}%**\n"
                f"- **{label_b} ({model_name_b}):** Overall Quality: **{rubric_b['overall']}%** (`{rubric_b['rubric_grade']}`) | Clarity: **{rubric_b['clarity']}%** | Depth: **{rubric_b['completeness']}%**\n\n"
                f"*Choose a winning response using the evaluation buttons below.*"
            )
            return {
                "display": human_text,
                "memory_text": f"[{model_name_a}]: {text_a}",
                "winner_name": label_a,
                "winner_model": model_name_a,
                "winner_answer": text_a,
                "loser_name": label_b,
                "loser_model": model_name_b,
                "loser_answer": text_b,
                "judgment": "Awaiting human evaluation.",
                "judge_source": "🧑 Human Judge",
                "is_human_judge": True,
                "eval_scores": {"Candidate A": rubric_a, "Candidate B": rubric_b},
            }

        is_a_first = random.choice([True, False])
        if is_a_first:
            resp_a, resp_b = text_a, text_b
            model_a, model_b = model_name_a, model_name_b
            eval_label_a, eval_label_b = label_a, label_b
        else:
            resp_a, resp_b = text_b, text_a
            model_a, model_b = model_name_b, model_name_a
            eval_label_a, eval_label_b = label_b, label_a

        judge_prompt = f"""
You are an impartial AI evaluator. Compare two candidate responses to a user's query and declare a winner.
Evaluate purely on merit, correctness, helpfulness, and adherence to instructions.

### Long-Term Memory Context:
{safe_memory_ctx}

### Short-Term Conversation Context:
{short_term_ctx}

### Current User Query:
{query}

### Response A:
{resp_a}

### Response B:
{resp_b}

Instructions:
1. Determine which response is superior (A or B).
2. Output your response as a JSON object with:
   {{"winner": "A" or "B", "reasoning": "brief explanation"}}
   If JSON is not possible, begin your response with "Winner: A" or "Winner: B".
"""
        print(f"---JUDGE: Calling {judge_type} for evaluation---")
        if judge_type == "gemini":
            judge_source = "Gemini 2.5 Flash"
            if fast_llm:
                try:
                    judgment = fast_llm.invoke(judge_prompt).content
                except Exception as e:
                    judgment = f"Error: {e}"
            else:
                judgment = "Error: Gemini API key not configured."
        elif judge_type == "groq":
            judge_source = f"Groq {GROQ_MODEL_LARGE}"
            groq_judge_res = query_groq(judge_prompt, groq_api_key, preferred_model=GROQ_MODEL_LARGE)
            if isinstance(groq_judge_res, dict) and "content" in groq_judge_res:
                judgment = groq_judge_res["content"]
            else:
                judgment = f"Error: {groq_judge_res.get('error', 'Groq judge failed')}"
        else:
            judgment = query_mistral_judge(judge_prompt, mistral_api_key)
            judge_source = "Mistral"

            # Fallback to Gemini judge if Mistral is rate-limited or fails
            if judgment.startswith("Error:"):
                logging.warning(f"Mistral judge failed: {judgment}. Falling back to Gemini as secondary judge...")
                try:
                    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
                    gemini_judgment = judge_llm.invoke(judge_prompt).content
                    judgment = gemini_judgment
                    judge_source = "Gemini (Fallback Judge)"
                except Exception as e:
                    logging.error(f"Gemini fallback judge error: {e}")
                    judge_source = "Automated Fallback"

        # Robust winner parsing (JSON or regex fallback)
        chosen_ab = "A"
        try:
            json_match = re.search(r"\{[\s\S]*?\"winner\"\s*:\s*\"([AB])\"[\s\S]*?\}", judgment, re.IGNORECASE)
            if json_match:
                chosen_ab = json_match.group(1).upper()
            else:
                regex_match = re.search(r"winner\s*:\s*(?:response\s+)?([AB])\b", judgment, re.IGNORECASE)
                if regex_match:
                    chosen_ab = regex_match.group(1).upper()
                else:
                    if f"Winner: {eval_label_a}" in judgment or f"winner: {eval_label_a.lower()}" in judgment.lower():
                        chosen_ab = "A"
                    elif f"Winner: {eval_label_b}" in judgment or f"winner: {eval_label_b.lower()}" in judgment.lower():
                        chosen_ab = "B"
        except Exception:
            chosen_ab = "A"

        if chosen_ab == "A":
            winner_name = eval_label_a
            chosen_answer, chosen_model_name = resp_a, model_a
            loser_response, loser_model_name, loser_name = resp_b, model_b, eval_label_b
        else:
            winner_name = eval_label_b
            chosen_answer, chosen_model_name = resp_b, model_b
            loser_response, loser_model_name, loser_name = resp_a, model_a, eval_label_a

        # Format and clean judge evaluation (extract reasoning from JSON, convert LaTeX to KaTeX, strip code block indentation)
        formatted_judgment = format_judge_evaluation(judgment)
        judge_res = output_guard.validate(formatted_judgment)
        judgment_clean = judge_res.clean_text if judge_res.passed else "[Judge evaluation omitted due to content policy]"
        if judge_res.event_type in ("OUTPUT_REDACTED", "OUTPUT_BLOCKED"):
            audit_logger.log(
                session_id = session_id,
                event_type = judge_res.event_type,
                detail     = f"[Judge] {judge_res.reason}",
                findings   = judge_res.findings,
            )

    # ── OUTPUT GUARD: sanitise winner and loser separately ──
    chosen_result = output_guard.validate(chosen_answer)
    chosen_answer = chosen_result.clean_text
    if not chosen_result.passed:
        chosen_answer = "[Response blocked: winning answer violated content policy]"

    loser_result = output_guard.validate(loser_response)
    loser_clean = loser_result.clean_text if loser_result.passed else "[Alternative response omitted due to content policy]"

    for res, label in [(chosen_result, "Winner"), (loser_result, "Alternative")]:
        if res.event_type in ("OUTPUT_REDACTED", "OUTPUT_BLOCKED"):
            audit_logger.log(
                session_id = session_id,
                event_type = res.event_type,
                detail     = f"[{label}] {res.reason}",
                findings   = res.findings,
            )

    rubric_winner = evaluation_guard.evaluate_response(query, chosen_answer)
    rubric_loser  = evaluation_guard.evaluate_response(query, loser_clean)

    final_output  = f"### 🏆 Judged Best Answer ({winner_name})\n"
    final_output += f"#### Model: {chosen_model_name}\n\n{chosen_answer}\n\n"
    final_output += f"### 🧠 Judge's Evaluation (from {judge_source})\n{judgment_clean}\n\n"
    final_output += f"**📊 Evaluation Rubric:** Quality: **{rubric_winner['overall']}%** (`{rubric_winner['rubric_grade']}`) | Clarity: **{rubric_winner['clarity']}%** | Completeness: **{rubric_winner['completeness']}%**\n\n---\n\n"
    final_output += f"### Other Response ({loser_name})\n\n"
    final_output += f"#### Model: {loser_model_name}\n\n{loser_clean}"

    distilled_memory = f"[{chosen_model_name}]: {chosen_answer}"

    return {
        "display":       final_output,
        "memory_text":   distilled_memory,
        "winner_name":   winner_name,
        "winner_model":  chosen_model_name,
        "winner_answer": chosen_answer,
        "loser_name":    loser_name,
        "loser_model":   loser_model_name,
        "loser_answer":  loser_clean,
        "judgment":      judgment_clean,
        "judge_source":  judge_source,
        "eval_scores": {
            "winner": rubric_winner,
            "alternative": rubric_loser,
        },
    }


def swap_comparison_response(text: str, comp_data: Optional[dict] = None) -> tuple[str, str]:
    """
    Swaps winner and loser responses in a comparison output for Human-in-the-Loop overrides.
    Returns (swapped_markdown_text, new_memory_text).
    """
    if comp_data and isinstance(comp_data, dict) and comp_data.get("winner_answer"):
        winner_name   = comp_data.get("winner_name", "Candidate A")
        winner_model  = comp_data.get("winner_model") or comp_data.get("chosen_model", "")
        winner_answer = comp_data.get("winner_answer") or comp_data.get("chosen_answer", "")
        loser_name    = comp_data.get("loser_name", "Candidate B")
        loser_model   = comp_data.get("loser_model", "")
        loser_answer  = comp_data.get("loser_answer", "")
        judgment      = comp_data.get("judgment", "")
        judge_source  = comp_data.get("judge_source", "Mistral")

        rubric_promoted = evaluation_guard.evaluate_response("", loser_answer)

        swapped_text  = f"### 🏆 User-Promoted Best Answer ({loser_name}) *(Overridden by user)*\n"
        swapped_text += f"#### Model: {loser_model}\n\n{loser_answer}\n\n"
        swapped_text += f"### 🧠 Judge's Evaluation (from {judge_source})\n{judgment}\n\n"
        swapped_text += f"**📊 Evaluation Rubric:** Quality: **{rubric_promoted['overall']}%** (`{rubric_promoted['rubric_grade']}`) | Clarity: **{rubric_promoted['clarity']}%** | Completeness: **{rubric_promoted['completeness']}%**\n\n---\n\n"
        swapped_text += f"### Other Response ({winner_name}) *(Previously judged)*\n\n"
        swapped_text += f"#### Model: {winner_model}\n\n{winner_answer}"

        new_memory = f"[{loser_model}]: {loser_answer}"
        return swapped_text, new_memory

    # Regex fallback if comparison_data is absent
    pattern = (
        r"### 🏆 (?:Judged|User-Promoted|Human-Selected) Best Answer \((?P<winner_name>[^)]+)\)\s*\n+"
        r"#### Model: (?P<winner_model>[^\n]+)\s*\n+"
        r"(?P<winner_answer>[\s\S]*?)\n+"
        r"### 🧠 Judge's Evaluation [^\n]*\n+"
        r"(?P<judge_section>[\s\S]*?)\n+"
        r"---\s*\n+"
        r"### Other Response \((?P<loser_name>[^)]+)\)\s*\n+"
        r"#### Model: (?P<loser_model>[^\n]+)\s*\n+"
        r"(?P<loser_answer>[\s\S]*)$"
    )
    m = re.search(pattern, text.strip())
    if m:
        w_name = m.group("winner_name").replace(" *(Overridden by user)*", "").strip()
        w_model = m.group("winner_model").strip()
        w_ans = m.group("winner_answer").strip()
        j_sec = m.group("judge_section").strip()
        l_name = m.group("loser_name").replace(" *(Previously judged)*", "").strip()
        l_model = m.group("loser_model").strip()
        l_ans = m.group("loser_answer").strip()

        swapped_text = f"### 🏆 User-Promoted Best Answer ({l_name}) *(Overridden by user)*\n"
        swapped_text += f"#### Model: {l_model}\n\n{l_ans}\n\n"
        swapped_text += f"### 🧠 Judge's Evaluation (Overridden)\n{j_sec}\n\n---\n\n"
        swapped_text += f"### Other Response ({w_name}) *(Previously judged)*\n\n"
        swapped_text += f"#### Model: {w_model}\n\n{w_ans}"

        new_memory = f"[{l_model}]: {l_ans}"
        return swapped_text, new_memory

    return text, f"[User-Promoted]: {text}"


def promote_candidate_as_winner(winner_cand: str, comp_data: dict) -> tuple[str, str]:
    """
    Directly promotes Candidate A or B as winner from Human Judge mode.
    Returns (markdown_text, new_memory).
    """
    if winner_cand.upper() == "B":
        w_name = comp_data.get("loser_name", "Candidate B")
        w_model = comp_data.get("loser_model", "Model B")
        w_ans = comp_data.get("loser_answer", "")
        l_name = comp_data.get("winner_name", "Candidate A")
        l_model = comp_data.get("winner_model", "Model A")
        l_ans = comp_data.get("winner_answer", "")
    else:
        w_name = comp_data.get("winner_name", "Candidate A")
        w_model = comp_data.get("winner_model", "Model A")
        w_ans = comp_data.get("winner_answer", "")
        l_name = comp_data.get("loser_name", "Candidate B")
        l_model = comp_data.get("loser_model", "Model B")
        l_ans = comp_data.get("loser_answer", "")

    rubric_winner = evaluation_guard.evaluate_response("", w_ans)

    swapped_text  = f"### 🏆 Human-Selected Best Answer ({w_name})\n"
    swapped_text += f"#### Model: {w_model}\n\n{w_ans}\n\n"
    swapped_text += f"### 🧑 Human Judge Evaluation\nYou directly evaluated both model candidates and declared {w_name} ({w_model}) the winner.\n\n"
    swapped_text += f"**📊 Evaluation Rubric:** Quality: **{rubric_winner['overall']}%** (`{rubric_winner['rubric_grade']}`) | Clarity: **{rubric_winner['clarity']}%** | Completeness: **{rubric_winner['completeness']}%**\n\n---\n\n"
    swapped_text += f"### Other Response ({l_name})\n\n"
    swapped_text += f"#### Model: {l_model}\n\n{l_ans}"

    new_memory = f"[{w_model}]: {w_ans}"
    return swapped_text, new_memory


# ===================================================================
# TOOL 2: IMAGE GENERATION
# ===================================================================
def image_generation_tool(prompt: str, google_api_key: str, pollinations_token: str) -> dict:
    logging.info(f"---TOOL: Generating Image for prompt: '{prompt}'---")
    try:
        enhancer_llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        enhancer_prompt = f"""
You are a "Top Class" prompt engineer. Rewrite the user's simple prompt into a hyper-detailed,
vibrant image generation description (Subject, Style, Lighting, Technicals).

User's prompt: "{prompt}"
"""
        final_prompt   = enhancer_llm.invoke(enhancer_prompt).content.strip()
        encoded_prompt = quote_plus(final_prompt)
        url            = f"https://gen.pollinations.ai/image/{encoded_prompt}?model=gptimage"
        headers        = {"Authorization": f"Bearer {pollinations_token}"}

        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        return {"image": img, "caption": f"Your prompt: '{prompt}'"}

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error: {http_err}")
        return {"error": f"The image generation service returned an error: {http_err}"}
    except requests.exceptions.ReadTimeout as timeout_err:
        logging.error(f"Timeout: {timeout_err}")
        return {"error": "The image generation service timed out. Please try again."}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": f"Failed to generate image: {e}"}


def extract_file_text(
    file_bytes: bytes,
    file_type: str = "",
    file_name: str = "",
    max_ocr_pages: int = 10,
    warn_callback = None,
    info_callback = None,
) -> str:
    """
    Extracts text from uploaded file bytes (PDF or plain text/code).
    For PDFs without an embedded text layer, performs OCR via PyMuPDF (fitz)
    at 200 DPI, capped at max_ocr_pages with a warning banner.
    """
    is_pdf = "pdf" in (file_type or "").lower() or (file_name or "").lower().endswith(".pdf")
    file_text = ""

    if is_pdf:
        if PdfReader is not None:
            try:
                reader = PdfReader(BytesIO(file_bytes))
                for page in reader.pages:
                    file_text += (page.extract_text() or "")
            except Exception as e:
                logging.warning(f"PdfReader extraction failed: {e}")

        if not file_text.strip():
            if info_callback:
                info_callback("No text layer found. Performing OCR...")
            if fitz is not None and pytesseract is not None:
                try:
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    total_pages = len(doc)
                    pages_to_ocr = min(total_pages, max_ocr_pages)
                    if total_pages > max_ocr_pages:
                        msg = f"⚠️ Document has {total_pages} pages. OCR is capped at the first {max_ocr_pages} pages to prevent timeouts."
                        if warn_callback:
                            warn_callback(msg)
                        logging.warning(msg)

                    for i in range(pages_to_ocr):
                        page = doc[i]
                        # 200 DPI ensures crisp resolution for reliable OCR recognition
                        pix = page.get_pixmap(dpi=200)
                        img = Image.open(BytesIO(pix.tobytes("png")))
                        page_text = pytesseract.image_to_string(img)
                        if page_text:
                            file_text += page_text + "\n"
                except Exception as e:
                    logging.error(f"OCR processing failed: {e}")
                    if warn_callback:
                        warn_callback(f"⚠️ OCR processing failed: {e}")
            else:
                msg = "⚠️ OCR dependencies (fitz / pytesseract) not available to process image-only PDF."
                if warn_callback:
                    warn_callback(msg)
                logging.warning(msg)
    else:
        file_text = file_bytes.decode("utf-8", errors="ignore")

    return file_text


# ===================================================================
# TOOL 3: FILE ANALYSIS
# ===================================================================
def file_analysis_tool(
    question: str,
    file_content_as_text: str,
    google_api_key: str,
    history: Optional[List[BaseMessage]] = None,
):
    print("---TOOL: Executing Empowered File Analysis---")
    streaming_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, streaming=True)

    safe_file_content = wrap_untrusted_data(file_content_as_text[:40000], "UPLOADED_FILE")

    history_section = ""
    if history:
        history_text = format_history(history)
        if history_text and history_text != "No previous context.":
            history_section = f"\n**Recent Conversation Context:**\n{history_text}\n"

    prompt = f"""
**Your Persona:** You are a highly intelligent AI assistant and a multi-disciplinary expert.

**The Task:** A user has uploaded a file and asked a question. Use the file content as the
primary source of truth, but enrich your answer with your own expertise.
{history_section}
**User's Question:**
{question}

**Provided File Content:**
{safe_file_content}

**Your Comprehensive Analysis:**
"""
    return streaming_llm.stream([HumanMessage(content=prompt)])


# ===================================================================
# TOOL 4: WEB SEARCH
# ===================================================================
def web_search_tool(query: str, tavily_api_key: str, google_api_key: str) -> str:
    print("---TOOL: Executing Web Search and Analysis---")
    try:
        tavily         = TavilyClient(api_key=tavily_api_key)
        search_results = tavily.search(query=query, search_depth="advanced", max_results=5)
        
        formatted_results = []
        for r in search_results.get("results", []):
            title = r.get("title", "Source")
            url   = r.get("url", "")
            snippet = r.get("content", "")
            formatted_results.append(f"Source: [{title}]({url})\nContent: {snippet}\n")

        search_content = "\n".join(formatted_results)
        safe_search_content = wrap_untrusted_data(search_content, "WEB_SEARCH")

        analyzer_llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        analysis_prompt = f"""
You are an expert research analyst. Answer the query based ONLY on the search results provided.
Cite the source titles and markdown links where relevant.

### User Query:
{query}

### Web Search Results:
{safe_search_content}

Your Answer:
"""
        result = analyzer_llm.invoke(analysis_prompt).content

        # ── OUTPUT GUARD on web search result ─────────────────
        guard_result = output_guard.validate(result)
        return guard_result.clean_text

    except Exception as e:
        return f"⚠️ Web search failed: {e}"


# ===================================================================
# AGENT STATE, ROUTER, GRAPH
# ===================================================================

class AgentState(TypedDict, total=False):
    query:            str
    history:          List[BaseMessage]
    route:            str
    final_response:   Optional[any]
    session_id:       str
    memory_context:   Optional[str]
    memory_text:      Optional[str]
    candidate_a_type: Optional[str]
    candidate_b_type: Optional[str]
    judge_type:       Optional[str]
    comparison_data:  Optional[dict]
    routing_mode:     Optional[str]


# --- NODE WRAPPERS ---

def call_comparison_tool(
    state: AgentState,
    google_api_key: str,
    groq_api_key: str,
    mistral_api_key: str,
    deepseek_api_key: str = "",
    kimi_api_key: str = "",
    candidate_a_type: str = "gemini",
    candidate_b_type: str = "groq",
    judge_type: str = "mistral",
):
    cand_a = state.get("candidate_a_type") or candidate_a_type
    cand_b = state.get("candidate_b_type") or candidate_b_type
    j_type = state.get("judge_type") or judge_type
    response = comparison_and_evaluation_tool(
        state["query"],
        state.get("history", []),
        google_api_key,
        groq_api_key,
        mistral_api_key,
        session_id=state.get("session_id", "default"),
        memory_context=state.get("memory_context"),
        deepseek_api_key=deepseek_api_key,
        kimi_api_key=kimi_api_key,
        candidate_a_type=cand_a,
        candidate_b_type=cand_b,
        judge_type=j_type,
    )
    if isinstance(response, dict):
        return {
            "final_response":  response["display"],
            "memory_text":     response.get("memory_text", response["display"]),
            "comparison_data": response,
        }
    return {"final_response": response, "memory_text": response}


def call_image_tool(state: AgentState, google_api_key: str, pollinations_token: str):
    res = image_generation_tool(state["query"], google_api_key, pollinations_token)
    mem_text = f"Image generated for prompt: {state['query']}" if isinstance(res, dict) and "image" in res else None
    return {"final_response": res, "memory_text": mem_text}


def call_web_search_tool(state: AgentState, tavily_api_key: str, google_api_key: str):
    res = web_search_tool(state["query"], tavily_api_key, google_api_key)
    return {"final_response": res, "memory_text": res}


# --- ROUTER & FALLBACKS ---

# ===================================================================
# SELF-ROUTING ENGINE & INTENT CLASSIFIER
# ===================================================================

class SelfRouter:
    """
    High-performance, deterministic Self-Routing Engine.
    Executes intent classification in pure Python (<0.1ms), eliminating external
    LLM network latency (1-2s saved) and preventing Gemini API quota exhaustion.
    """

    # Direct slash commands
    CMD_IMAGE = re.compile(r"^/(image|draw|img|render|sketch|paint)\b", re.IGNORECASE)
    CMD_SEARCH = re.compile(r"^/(search|find|web|browse|lookup|google)\b", re.IGNORECASE)

    # Negative guards: queries containing image/search words but asking for code, explanations, or definitions
    IMAGE_NEGATIVE_PATTERNS = [
        re.compile(r"\bhow\s+(to|do|can)\s+(i|we|you)?\s*(generate|create|draw|make|render)\s+(an?\s+)?images?\b", re.IGNORECASE),
        re.compile(r"\b(code|script|python|library|api|algorithm|model)\s+to\s+(generate|create|draw)\b", re.IGNORECASE),
        re.compile(r"\b(explain|what\s+is|difference\s+between)\b.*\b(diffusion|gan|image|dall-e|midjourney|stable\s+diffusion)\b", re.IGNORECASE),
        re.compile(r"\b(write|show)\s+(me\s+)?(a\s+)?(function|code|script|program)\b", re.IGNORECASE),
    ]

    SEARCH_NEGATIVE_PATTERNS = [
        re.compile(r"\b(binary|linear|depth\s+first|breadth\s+first|tree|graph|string|regex|pattern)\s+search\b", re.IGNORECASE),
        re.compile(r"\b(write|implement|code|algorithm|function)\b.*\bsearch\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(is|was|were)\s+(the\s+)?(meaning|definition|origin|history|cause)\s+of\b", re.IGNORECASE),
        re.compile(r"\bderive\b|\bcalculate\b|\bsolve\b", re.IGNORECASE),
    ]

    # Positive Image Generation patterns
    IMAGE_PATTERNS = [
        # Explicit generate/draw/sketch/paint verbs + image nouns
        re.compile(r"\b(generate|create|make|draw|sketch|render|paint|produce)\b\s*(an?|the|me\s+an?|some)?\s*(hyperrealistic|photorealistic|cinematic|digital|artistic|cute|detailed|3d|vibrant|oil|pencil)?\s*(image|picture|photo|photograph|wallpaper|illustration|drawing|artwork|portrait|sketch|graphic|avatar|render)\s*(of|with|featuring|showing)?\b", re.IGNORECASE),
        re.compile(r"\b(draw|sketch|paint)\s+(me\s+)?(an?|the)?\s*([a-zA-Z0-9\s]+?)\s*(with|in|on|at|against|under)?\b", re.IGNORECASE),
        re.compile(r"^(image|picture|photo|illustration|drawing|sketch|painting)\s+of\b", re.IGNORECASE),
        re.compile(r"\b(generate\s+image|create\s+image|make\s+picture|draw\s+a\s+picture|sketch\s+of)\b", re.IGNORECASE),
    ]

    # Positive Web Search / Live Info patterns
    SEARCH_PATTERNS = [
        # Temporal + Informational keywords (forward and reverse order)
        re.compile(r"\b(today|yesterday|tomorrow|tonight|this\s+week|this\s+month|right\s+now|currently|current|latest|breaking|recent)\b.*\b(news|weather|temperature|forecast|price|prices|stock|crypto|bitcoin|inflation|score|match|game|election|updates?)\b", re.IGNORECASE),
        re.compile(r"\b(news|weather|temperature|forecast|price|prices|stock|crypto|bitcoin|score|match|election|updates?)\b.*\b(today|yesterday|tomorrow|tonight|this\s+week|this\s+month|right\s+now|currently|current|latest|breaking|recent)\b", re.IGNORECASE),
        # Weather / Forecast specifically
        re.compile(r"\b(weather|temperature|forecast|rain|snow|humidity)\s+(in|at|for)\s+[a-zA-Z\s]+", re.IGNORECASE),
        # Real-time entities & events
        re.compile(r"\b(who\s+won|score\s+of|winner\s+of|result\s+of)\s+the\s+[a-zA-Z0-9\s]+(match|game|cup|series|tournament|super\s+bowl|election|award|finals?)\b", re.IGNORECASE),
        re.compile(r"\b(stock\s+price|share\s+price|market\s+cap|crypto\s+price|bitcoin\s+price)\b", re.IGNORECASE),
        re.compile(r"\b(current|latest|newest)\s+(version|release|features|status|specs|updates?)\s+of\b", re.IGNORECASE),
        re.compile(r"\bwho\s+is\s+(the\s+)?(current|present)\s+(president|prime\s+minister|ceo|governor|chancellor|leader)\b", re.IGNORECASE),
        # Direct search requests
        re.compile(r"\b(search\s+(the\s+web|online|google|internet|for)|browse\s+(the\s+web|for)|look\s+up\s+online|find\s+(articles?|news|info)\s+(on|about))\b", re.IGNORECASE),
        # URL detection
        re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE),
    ]

    @classmethod
    def route(cls, query: str, history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Classifies query intent deterministically into:
        - 'image_generator'
        - 'web_search'
        - 'comparison_chat' (default for coding, math, reasoning, conversation)
        """
        start_t = time.perf_counter()
        q = (query or "").strip()

        # 1. Direct slash commands
        if cls.CMD_IMAGE.search(q):
            return {
                "route": "image_generator",
                "engine": "self_router",
                "matched_rule": "Command: /image prefix",
                "confidence": 1.0,
                "latency_ms": round((time.perf_counter() - start_t) * 1000, 3),
            }
        if cls.CMD_SEARCH.search(q):
            return {
                "route": "web_search",
                "engine": "self_router",
                "matched_rule": "Command: /search prefix",
                "confidence": 1.0,
                "latency_ms": round((time.perf_counter() - start_t) * 1000, 3),
            }

        # 2. Check Image Generation Intent (with negative guards)
        is_img_neg = any(p.search(q) for p in cls.IMAGE_NEGATIVE_PATTERNS)
        if not is_img_neg:
            for p in cls.IMAGE_PATTERNS:
                if p.search(q):
                    return {
                        "route": "image_generator",
                        "engine": "self_router",
                        "matched_rule": f"Regex: Image Pattern",
                        "confidence": 0.98,
                        "latency_ms": round((time.perf_counter() - start_t) * 1000, 3),
                    }

        # 3. Check Web Search Intent (with negative guards)
        is_search_neg = any(p.search(q) for p in cls.SEARCH_NEGATIVE_PATTERNS)
        if not is_search_neg:
            for p in cls.SEARCH_PATTERNS:
                if p.search(q):
                    return {
                        "route": "web_search",
                        "engine": "self_router",
                        "matched_rule": f"Regex: Search Pattern",
                        "confidence": 0.95,
                        "latency_ms": round((time.perf_counter() - start_t) * 1000, 3),
                    }

        # 4. Default: Mixture of Agents Arena (Reasoning, Coding, Analysis, Chat)
        return {
            "route": "comparison_chat",
            "engine": "self_router",
            "matched_rule": "Default: Mixture of Agents Competitive Arena",
            "confidence": 0.99,
            "latency_ms": round((time.perf_counter() - start_t) * 1000, 3),
        }


def self_route_query(query: str, history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
    """Helper to run the Self-Routing Engine."""
    return SelfRouter.route(query, history=history)


def keyword_router_fallback(query: str) -> str:
    """Deterministic rule-based routing fallback powered by SelfRouter."""
    return SelfRouter.route(query)["route"]


def router(
    state: AgentState,
    google_api_key: str,
    groq_api_key: Optional[str] = None,
    default_routing_mode: Optional[str] = None,
):
    print("---AGENT: Routing query---")
    query        = state["query"]
    history      = state.get("history", [])
    session_id   = state.get("session_id", "default")
    routing_mode = state.get("routing_mode") or default_routing_mode

    # Case A: If self-routing is requested, execute our zero-API Self-Routing Engine directly
    if routing_mode == "self":
        self_res = SelfRouter.route(query, history=history)
        print(f"---AGENT: Self-Routing selected -> {self_res['route']} ({self_res['matched_rule']} in {self_res['latency_ms']}ms)---")
        return {"route": self_res["route"]}

    # Case B: If LLM routing is requested or legacy fallback is needed
    short_term_ctx = format_history(history)
    long_term_ctx  = state.get("memory_context")
    if long_term_ctx is None:
        long_term_ctx = retrieve_relevant_memory(query, session_id=session_id, top_k=3)
    safe_memory_ctx = wrap_untrusted_data(long_term_ctx, "LONG_TERM_MEMORY", session_id=session_id) if long_term_ctx else "None."

    router_prompt = f"""
You are a master routing agent. Determine the user's primary intent.

Long-Term Memory Context (past relevant exchanges):
{safe_memory_ctx}

Short-Term Context (recent turns):
{short_term_ctx}

Current User Query: "{query}"

Choices:
1. `comparison_tool`: complex questions, coding, analysis, general chat, follow-ups.
2. `image_generation_tool`: ONLY if the user explicitly asks to create/draw/generate an image.
3. `web_search_tool`: real-time information, news, weather, current events.

Return ONLY the tool name.
"""
    decision = None

    # Try Groq if explicitly requested as primary router
    if routing_mode == "groq" and groq_api_key:
        try:
            groq_res = query_groq(router_prompt, groq_api_key, max_retries=1, timeout=10)
            if isinstance(groq_res, dict) and "content" in groq_res:
                resp_text = groq_res["content"].strip()
                if "web_search_tool" in resp_text:
                    decision = "web_search"
                elif "image_generation_tool" in resp_text:
                    decision = "image_generator"
                elif "comparison_tool" in resp_text:
                    decision = "comparison_chat"
        except Exception as e:
            logging.warning(f"Groq router error: {e}. Attempting Gemini fallback...")

    # Step 1: Try primary router (Gemini)
    if decision is None and google_api_key:
        try:
            router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
            response = router_llm.invoke(router_prompt).content.strip()
            if "web_search_tool" in response:
                decision = "web_search"
            elif "image_generation_tool" in response:
                decision = "image_generator"
            elif "comparison_tool" in response:
                decision = "comparison_chat"
        except Exception as e:
            logging.warning(f"Primary Gemini router error: {e}. Attempting Groq fallback...")

    # Step 2: Try secondary router (Groq)
    if decision is None and groq_api_key:
        try:
            groq_res = query_groq(router_prompt, groq_api_key, max_retries=1, timeout=10)
            if isinstance(groq_res, dict) and "content" in groq_res:
                resp_text = groq_res["content"].strip()
                if "web_search_tool" in resp_text:
                    decision = "web_search"
                elif "image_generation_tool" in resp_text:
                    decision = "image_generator"
                elif "comparison_tool" in resp_text:
                    decision = "comparison_chat"
        except Exception as e:
            logging.warning(f"Secondary Groq router error: {e}. Attempting self-routing fallback...")

    # Step 3: Tertiary deterministic self-routing fallback
    if decision is None:
        decision = keyword_router_fallback(query)
        print(f"---AGENT: Fallback router selected -> {decision}---")

    return {"route": decision}


# --- BUILD AGENT ---

def build_agent(
    google_api_key: str,
    groq_api_key: str,
    pollinations_token: str,
    tavily_api_key: str,
    mistral_api_key: str,
    deepseek_api_key: str = "",
    kimi_api_key: str = "",
    candidate_a_type: str = "gemini",
    candidate_b_type: str = "groq",
    judge_type: str = "mistral",
    routing_mode: str = "self",
):
    workflow = StateGraph(AgentState)

    router_with_keys  = partial(
        router,
        google_api_key=google_api_key,
        groq_api_key=groq_api_key,
        default_routing_mode=routing_mode,
    )
    comparison_node   = partial(
        call_comparison_tool,
        google_api_key=google_api_key,
        groq_api_key=groq_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        kimi_api_key=kimi_api_key,
        candidate_a_type=candidate_a_type,
        candidate_b_type=candidate_b_type,
        judge_type=judge_type,
    )
    image_node        = partial(call_image_tool, google_api_key=google_api_key,
                                pollinations_token=pollinations_token)
    web_search_node   = partial(call_web_search_tool, tavily_api_key=tavily_api_key,
                                google_api_key=google_api_key)

    workflow.add_node("router",          router_with_keys)
    workflow.add_node("comparison_chat", comparison_node)
    workflow.add_node("image_generator", image_node)
    workflow.add_node("web_search",      web_search_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "comparison_chat": "comparison_chat",
            "image_generator": "image_generator",
            "web_search":      "web_search",
        },
    )

    workflow.add_edge("comparison_chat", END)
    workflow.add_edge("image_generator", END)
    workflow.add_edge("web_search",      END)

    return workflow.compile()