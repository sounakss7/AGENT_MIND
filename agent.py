import os
import re
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

# =======================================================================================
# VECTOR MEMORY IMPORT
# =======================================================================================
from vector_memory import retrieve_relevant_memory   # <-- NEW

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


def query_groq(prompt: str, groq_api_key: str):
    model = choose_groq_model(prompt)
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return {"model_name": model, "content": content}
        return f"❌ Groq API Error: {resp.text}"
    except Exception as e:
        return f"⚠️ Groq Error: {e}"


def query_mistral_judge(prompt: str, mistral_api_key: str):
    model = "mistral-small-latest"
    headers = {"Authorization": f"Bearer {mistral_api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024}
    try:
        resp = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"Mistral Judge HTTP Error: {http_err} - {resp.text}")
        return "Error: The Mistral judge failed to provide an evaluation (HTTP error)."
    except Exception as e:
        logging.error(f"Mistral Judge Exception: {e}")
        return f"Error: The Mistral judge ran into an exception: {e}"


# =======================================================================================
# TOOL 1: COMPARISON & EVALUATION  (now memory-aware)
# =======================================================================================

def comparison_and_evaluation_tool(
    query: str,
    history: List[BaseMessage],
    google_api_key: str,
    groq_api_key: str,
    mistral_api_key: str,
    session_id: str = "default",          # <-- NEW
) -> str:
    print("---TOOL: Executing Comparison (Judged by Mistral with Memory)---")

    # 1. Short-term context (last 4 turns from in-session history)
    short_term_ctx = format_history(history)

    # 2. Long-term semantic context from Qdrant  <-- NEW
    long_term_ctx = retrieve_relevant_memory(query, session_id=session_id)
    if long_term_ctx:
        print(f"[VectorMemory] Retrieved {long_term_ctx.count(chr(10))} memory hits for comparison tool.")

    full_prompt_with_context = f"""
LONG-TERM MEMORY (semantically relevant past interactions):
{long_term_ctx if long_term_ctx else "None available."}

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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_gemini = executor.submit(lambda: fast_llm.invoke(full_prompt_with_context).content)
        future_groq   = executor.submit(query_groq, full_prompt_with_context, groq_api_key)

        gemini_response = future_gemini.result()
        groq_result     = future_groq.result()

    groq_response    = ""
    groq_model_name  = "Groq (Error)"

    if isinstance(groq_result, dict):
        groq_response   = groq_result["content"]
        groq_model_name = groq_result["model_name"]
    else:
        groq_response = groq_result

    judge_prompt = f"""
You are an impartial AI evaluator. Compare two responses to a user's query and declare a winner.

### Long-Term Memory Context:
{long_term_ctx if long_term_ctx else "None."}

### Short-Term Conversation Context:
{short_term_ctx}

### Current User Query:
{query}

### Response A (Gemini):
{gemini_response}

### Response B (Groq - model: {groq_model_name}):
{groq_response}

Instructions:
1. Begin with "Winner: Gemini" or "Winner: Groq".
2. Explain your reasoning. Did the models use the memory context correctly?
3. Evaluate purely on merit.
"""

    print("---JUDGE: Calling Mistral for evaluation---")
    judgment = query_mistral_judge(judge_prompt, mistral_api_key)

    match       = re.search(r"winner\s*:\s*(gemini|groq)", judgment, re.IGNORECASE)
    winner_name = match.group(1).capitalize() if match else "Evaluation"

    if winner_name == "Gemini":
        chosen_answer, chosen_model_name = gemini_response, gemini_model_name
        loser_response, loser_model_name, loser_name = groq_response, groq_model_name, "Groq"
    elif winner_name == "Groq":
        chosen_answer, chosen_model_name = groq_response, groq_model_name
        loser_response, loser_model_name, loser_name = gemini_response, gemini_model_name, "Gemini"
    else:
        chosen_answer, chosen_model_name = gemini_response, gemini_model_name
        loser_response, loser_model_name, loser_name = groq_response, groq_model_name, "Groq"

    final_output  = f"### 🏆 Judged Best Answer ({winner_name})\n"
    final_output += f"#### Model: {chosen_model_name}\n\n{chosen_answer}\n\n"
    final_output += f"### 🧠 Judge's Evaluation (from Mistral)\n{judgment}\n\n---\n\n"
    final_output += f"### Other Response ({loser_name})\n\n"
    final_output += f"#### Model: {loser_model_name}\n\n{loser_response}"

    return final_output


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


# ===================================================================
# TOOL 3: FILE ANALYSIS
# ===================================================================
def file_analysis_tool(question: str, file_content_as_text: str, google_api_key: str):
    print("---TOOL: Executing Empowered File Analysis---")
    streaming_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, streaming=True)

    prompt = f"""
**Your Persona:** You are a highly intelligent AI assistant and a multi-disciplinary expert.

**The Task:** A user has uploaded a file and asked a question. Use the file content as the
primary source of truth, but enrich your answer with your own expertise.

**User's Question:**
{question}

**Provided File Content:**
---
{file_content_as_text[:40000]}
---

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
        search_content = "\n".join([r["content"] for r in search_results["results"]])

        analyzer_llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        analysis_prompt = f"""
You are an expert research analyst. Answer the query based ONLY on the search results provided.

### User Query:
{query}

### Web Search Results:
---
{search_content}
---

Your Answer:
"""
        return analyzer_llm.invoke(analysis_prompt).content

    except Exception as e:
        return f"⚠️ Web search failed: {e}"


# ===================================================================
# AGENT STATE, ROUTER, GRAPH
# ===================================================================

class AgentState(TypedDict):
    query:          str
    history:        List[BaseMessage]
    route:          str
    final_response: Optional[any]
    session_id:     str                    # <-- NEW


# --- NODE WRAPPERS ---

def call_comparison_tool(state: AgentState, google_api_key: str, groq_api_key: str, mistral_api_key: str):
    response = comparison_and_evaluation_tool(
        state["query"],
        state.get("history", []),
        google_api_key,
        groq_api_key,
        mistral_api_key,
        session_id=state.get("session_id", "default"),   # <-- NEW
    )
    return {"final_response": response}


def call_image_tool(state: AgentState, google_api_key: str, pollinations_token: str):
    return {"final_response": image_generation_tool(state["query"], google_api_key, pollinations_token)}


def call_web_search_tool(state: AgentState, tavily_api_key: str, google_api_key: str):
    return {"final_response": web_search_tool(state["query"], tavily_api_key, google_api_key)}


# --- ROUTER ---

def router(state: AgentState, google_api_key: str):
    print("---AGENT: Routing query---")
    router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    query      = state["query"]
    history    = state.get("history", [])
    session_id = state.get("session_id", "default")

    short_term_ctx = format_history(history)

    # Pull semantic memory to help routing too  <-- NEW
    long_term_ctx = retrieve_relevant_memory(query, session_id=session_id, top_k=3)

    router_prompt = f"""
You are a master routing agent. Determine the user's primary intent.

Long-Term Memory Context (past relevant exchanges):
{long_term_ctx if long_term_ctx else "None."}

Short-Term Context (recent turns):
{short_term_ctx}

Current User Query: "{query}"

Choices:
1. `comparison_tool`: complex questions, coding, analysis, general chat, follow-ups.
2. `image_generation_tool`: ONLY if the user explicitly asks to create/draw/generate an image.
3. `web_search_tool`: real-time information, news, weather, current events.

Return ONLY the tool name.
"""
    response = router_llm.invoke(router_prompt).content.strip()

    if "web_search_tool" in response:
        print("---AGENT: Decision -> Web Search Tool---")
        return {"route": "web_search"}
    elif "image_generation_tool" in response:
        print("---AGENT: Decision -> Image Generation Tool---")
        return {"route": "image_generator"}
    else:
        print("---AGENT: Decision -> Comparison & Evaluation Tool---")
        return {"route": "comparison_chat"}


# --- BUILD AGENT ---

def build_agent(google_api_key: str, groq_api_key: str, pollinations_token: str,
                tavily_api_key: str, mistral_api_key: str):
    workflow = StateGraph(AgentState)

    router_with_keys  = partial(router, google_api_key=google_api_key)
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