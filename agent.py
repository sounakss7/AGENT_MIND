import os
import re
import time
import json
import random
import requests
from io import BytesIO
from PIL import Image
from typing import TypedDict, Optional, List
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
from security_guard import input_guard, output_guard, audit_logger, wrap_untrusted_data

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


def choose_groq_model(prompt: str):
    """Selects the best Groq model based on the complexity of the prompt."""
    p = prompt.lower()
    if any(x in p for x in ["python", "code", "algorithm", "bug", "function", "script",
                             "information", "analysis", "solution", "nlp", "essay",
                             "mathematics", "research", "reasoning"]):
        return "openai/gpt-oss-120b"
    else:
        return "llama-3.1-8b-instant"


def query_groq(prompt: str, groq_api_key: str, max_retries: int = 3, timeout: int = 30):
    """
    Queries the Groq API with retries and exponential backoff on transient/rate errors.
    Returns a dict with 'model_name' and either 'content' or 'error'.
    """
    model = choose_groq_model(prompt)
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return {"model_name": model, "content": content}
            elif resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                logging.warning(f"Groq API {resp.status_code} received. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            else:
                return {"model_name": model, "error": f"Groq API Error ({resp.status_code}): {resp.text}"}
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                logging.warning(f"Groq network error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            return {"model_name": model, "error": f"Groq Timeout/Connection Error: {e}"}
        except Exception as e:
            return {"model_name": model, "error": f"Groq Error: {e}"}

    return {"model_name": model, "error": "Groq API exceeded max retries."}


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
) -> dict:
    print("---TOOL: Executing Comparison (Judged by Mistral with Memory)---")

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

    fast_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    gemini_model_name = "gemini-2.5-flash"

    def _safe_gemini():
        try:
            return fast_llm.invoke(full_prompt_with_context).content
        except Exception as e:
            logging.error(f"Gemini generation error: {e}")
            return {"error": f"Gemini Error: {e}"}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_gemini = executor.submit(_safe_gemini)
        future_groq   = executor.submit(query_groq, full_prompt_with_context, groq_api_key)
        gemini_result = future_gemini.result()
        groq_result   = future_groq.result()

    # Determine errors on each side
    gemini_has_err = isinstance(gemini_result, dict) and "error" in gemini_result
    groq_has_err   = isinstance(groq_result, dict) and "error" in groq_result

    gemini_text = gemini_result["error"] if gemini_has_err else (gemini_result or "")
    groq_text   = groq_result["error"] if groq_has_err else (groq_result.get("content", "") if isinstance(groq_result, dict) else str(groq_result))
    groq_model_name = groq_result.get("model_name", "Groq") if isinstance(groq_result, dict) else "Groq"
    judge_source = "Automated Selection"

    # Case 1: Both models failed
    if gemini_has_err and groq_has_err:
        err_msg = f"### ⚠️ Both models failed to respond.\n\n- **Gemini:** {gemini_text}\n- **Groq:** {groq_text}"
        return {"display": err_msg, "memory_text": ""}

    # Case 2: Gemini failed, Groq succeeded
    if gemini_has_err and not groq_has_err:
        chosen_answer, chosen_model_name = groq_text, groq_model_name
        winner_name = "Groq"
        loser_response, loser_model_name, loser_name = gemini_text, gemini_model_name, "Gemini (Failed)"
        judgment_clean = "Groq selected automatically because Gemini encountered an error."
        judge_source = "Automated (Single Candidate)"

    # Case 3: Groq failed, Gemini succeeded
    elif groq_has_err and not gemini_has_err:
        chosen_answer, chosen_model_name = gemini_text, gemini_model_name
        winner_name = "Gemini"
        loser_response, loser_model_name, loser_name = groq_text, groq_model_name, "Groq (Failed)"
        judgment_clean = "Gemini selected automatically because Groq encountered an error."
        judge_source = "Automated (Single Candidate)"

    # Case 4: Both succeeded — call Mistral judge with randomized A/B order and neutral labels
    else:
        is_gemini_a = random.choice([True, False])
        if is_gemini_a:
            resp_a, resp_b = gemini_text, groq_text
            model_a, model_b = gemini_model_name, groq_model_name
            label_a, label_b = "Gemini", "Groq"
        else:
            resp_a, resp_b = groq_text, gemini_text
            model_a, model_b = groq_model_name, gemini_model_name
            label_a, label_b = "Groq", "Gemini"

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
        print("---JUDGE: Calling Mistral for evaluation---")
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
                    if "Winner: Gemini" in judgment or "winner: gemini" in judgment.lower():
                        chosen_ab = "A" if is_gemini_a else "B"
                    elif "Winner: Groq" in judgment or "winner: groq" in judgment.lower():
                        chosen_ab = "B" if is_gemini_a else "A"
        except Exception:
            chosen_ab = "A"

        if chosen_ab == "A":
            winner_name = label_a
            chosen_answer, chosen_model_name = resp_a, model_a
            loser_response, loser_model_name, loser_name = resp_b, model_b, label_b
        else:
            winner_name = label_b
            chosen_answer, chosen_model_name = resp_b, model_b
            loser_response, loser_model_name, loser_name = resp_a, model_a, label_a

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

    final_output  = f"### 🏆 Judged Best Answer ({winner_name})\n"
    final_output += f"#### Model: {chosen_model_name}\n\n{chosen_answer}\n\n"
    final_output += f"### 🧠 Judge's Evaluation (from {judge_source})\n{judgment_clean}\n\n---\n\n"
    final_output += f"### Other Response ({loser_name})\n\n"
    final_output += f"#### Model: {loser_model_name}\n\n{loser_clean}"

    distilled_memory = f"[{chosen_model_name}]: {chosen_answer}"

    return {
        "display":     final_output,
        "memory_text": distilled_memory,
    }


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
    query:          str
    history:        List[BaseMessage]
    route:          str
    final_response: Optional[any]
    session_id:     str
    memory_context: Optional[str]
    memory_text:    Optional[str]


# --- NODE WRAPPERS ---

def call_comparison_tool(state: AgentState, google_api_key: str, groq_api_key: str, mistral_api_key: str):
    response = comparison_and_evaluation_tool(
        state["query"],
        state.get("history", []),
        google_api_key,
        groq_api_key,
        mistral_api_key,
        session_id=state.get("session_id", "default"),
        memory_context=state.get("memory_context"),
    )
    if isinstance(response, dict):
        return {
            "final_response": response["display"],
            "memory_text":    response.get("memory_text", response["display"]),
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

def keyword_router_fallback(query: str) -> str:
    """Deterministic rule-based routing fallback if LLMs fail."""
    q = query.lower()
    if any(k in q for k in ["generate image", "create an image", "draw", "sketch", "picture of"]):
        return "image_generator"
    if any(k in q for k in ["search", "weather", "latest news", "today", "current price", "who won", "stock price", "browse"]):
        return "web_search"
    return "comparison_chat"


def router(state: AgentState, google_api_key: str, groq_api_key: Optional[str] = None):
    print("---AGENT: Routing query---")
    query      = state["query"]
    history    = state.get("history", [])
    session_id = state.get("session_id", "default")

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

    # Step 1: Try primary router (Gemini)
    if google_api_key:
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
            logging.warning(f"Secondary Groq router error: {e}. Attempting keyword fallback...")

    # Step 3: Tertiary deterministic keyword fallback
    if decision is None:
        decision = keyword_router_fallback(query)
        print(f"---AGENT: Fallback router selected -> {decision}---")

    return {"route": decision}


# --- BUILD AGENT ---

def build_agent(google_api_key: str, groq_api_key: str, pollinations_token: str,
                tavily_api_key: str, mistral_api_key: str):
    workflow = StateGraph(AgentState)

    router_with_keys  = partial(router, google_api_key=google_api_key, groq_api_key=groq_api_key)
    comparison_node   = partial(call_comparison_tool, google_api_key=google_api_key,
                                groq_api_key=groq_api_key, mistral_api_key=mistral_api_key)
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