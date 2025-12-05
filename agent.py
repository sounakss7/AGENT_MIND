import os
import re
import requests
from io import BytesIO
from PIL import Image
from typing import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.schema import BaseMessage
from langgraph.graph import StateGraph, END
import concurrent.futures
from functools import partial
from tavily import TavilyClient
from urllib.parse import quote_plus
import logging

# =======================================================================================
# TOOL 1: THE COMPARISON & EVALUATION WORKFLOW (Judged by Mistral)
# =======================================================================================

def choose_groq_model(prompt: str):
    p = prompt.lower()
    if any(x in p for x in ["python", "code", "algorithm", "bug", "function", "script" ,"information" , "analysis" , "solution" , "NLP" ,"essay" , "mathematics" ,"research" ,"reasoning"]):
        return "openai/gpt-oss-120b"
    else:
        return "llama-3.1-8b-instant"

def query_groq(prompt: str, groq_api_key: str):
    """Queries Groq and returns a dict with model_name and content, or an error string."""
    model = choose_groq_model(prompt)
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            # ---MODIFIED: Return a dict to include the model name ---
            return {"model_name": model, "content": content} 
        return f"❌ Groq API Error: {resp.text}"
    except Exception as e:
        return f"⚠️ Groq Error: {e}"

# --- NEW: Dedicated function to call Mistral for judging ---
def query_mistral_judge(prompt: str, mistral_api_key: str):
    """A dedicated function to call Mistral for judging."""
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

# --- MODIFIED: The evaluation tool now uses Mistral as the judge ---
def comparison_and_evaluation_tool(query: str, google_api_key: str, groq_api_key: str, mistral_api_key: str) -> str:
    """
    Runs a query through Gemini and Groq, has a MISTRAL AI judge evaluate the best response,
    and formats everything into a comprehensive answer.
    """
    print("---TOOL: Executing Comparison (Judged by Mistral)---")
    
    fast_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    gemini_model_name = "gemini-2.5-flash"
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_gemini = executor.submit(lambda: fast_llm.invoke(query).content)
        future_groq = executor.submit(query_groq, query, groq_api_key)
        
        gemini_response = future_gemini.result()
        groq_result = future_groq.result()

    # --- MODIFIED: Handle the dict or error string from query_groq ---
    groq_response = ""
    groq_model_name = "Groq (Error)"

    if isinstance(groq_result, dict):
        groq_response = groq_result["content"]
        groq_model_name = groq_result["model_name"]
    else:
        # It's an error string
        groq_response = groq_result

    judge_prompt = f"""
    You are an impartial AI evaluator. Compare two responses to a user's query and declare a winner.

    ### User Query:
    {query}
    ### Response A (Gemini):
    {gemini_response}
    ### Response B (Groq - model: {groq_model_name}):
    {groq_response}

    Instructions:
    1. Begin with "Winner: Gemini" or "Winner: Groq".
    2. Explain your reasoning clearly, comparing accuracy, clarity, and completeness.
    3. **Evaluate the responses purely on their merit for the given query. Do not show bias towards any model provider. Your judgment must be neutral and unbiased.**
    """
    
    print("---JUDGE: Calling Mistral for evaluation---")
    judgment = query_mistral_judge(judge_prompt, mistral_api_key)
    
    match = re.search(r"winner\s*:\s*(gemini|groq)", judgment, re.IGNORECASE)
    winner_name = match.group(1).capitalize() if match else "Evaluation"
    
    # --- THIS IS THE NEW LOGIC YOU REQUESTED ---
    
    chosen_answer = ""
    chosen_model_name = ""
    loser_response = ""
    loser_model_name = ""
    loser_name = ""

    if winner_name == "Gemini":
        chosen_answer = gemini_response
        chosen_model_name = gemini_model_name
        loser_response = groq_response
        loser_model_name = groq_model_name
        loser_name = "Groq"
    elif winner_name == "Groq":
        chosen_answer = groq_response
        chosen_model_name = groq_model_name
        loser_response = gemini_response
        loser_model_name = gemini_model_name
        loser_name = "Gemini"
    else:
        # Fallback if regex fails to find a clear winner
        chosen_answer = gemini_response # Default to Gemini
        chosen_model_name = gemini_model_name
        loser_response = groq_response
        loser_model_name = groq_model_name
        loser_name = "Groq"


    # 1. The Winner's Response
    final_output = f"### 🏆 Judged Best Answer ({winner_name})\n"
    final_output += f"#### Model: {chosen_model_name}\n\n{chosen_answer}\n\n"
    
    # 2. The Judge's Evaluation
    final_output += f"### 🧠 Judge's Evaluation (from Mistral)\n{judgment}\n\n---\n\n"

    # 3. The Loser's Response
    final_output += f"### Other Response ({loser_name})\n\n"
    final_output += f"#### Model: {loser_model_name}\n\n{loser_response}"

    # --- END OF MODIFIED LOGIC ---
    
    return final_output

# ===================================================================
# TOOL 2: IMAGE GENERATION TOOL (Unchanged)
# ===================================================================
def image_generation_tool(prompt: str, google_api_key: str, pollinations_token: str) -> dict:
    """Use this tool when the user asks to create, draw, or generate an image."""
    logging.info(f"---TOOL: Generating Image for prompt: '{prompt}'---")
    try:
        enhancer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        
        # --- MODIFIED: This is the new "Top Class" prompt enhancer ---
        enhancer_prompt = f"""
You are a "Top Class" prompt engineer, a master of visual language. Your job is to rewrite a user's simple prompt into a hyper-detailed, vibrant, and masterful image generation description. The output must be optimized for a model like Pollinations.ai (which uses Stable Diffusion).

The user's prompt is: "{prompt}"

---
**The Formula for a 'Top Class' Prompt (Follow this structure):**
1.  **Core Subject:** Start with a hyper-detailed description of the main subject (e.g., "A majestic dragon with shimmering emerald scales and glowing blue eyes...").
2.  **Style & Medium:** This is critical. `digital painting`, `photograph`, `oil on canvas`, `3D render`, `anime key visual`, `watercolor`.
3.  **Scene & Environment:** Describe the background. `in a forgotten temple`, `on a neon-lit cyberpunk street`, `in a sun-drenched meadow`.
4.  **Lighting & Atmosphere:** Set the mood. `cinematic lighting`, `volumetric god rays`, `eerie, foggy night`, `warm afternoon sun`, `dramatic shadows`.
5.  **Artist & Platform Keywords:** This adds huge power. `trending on ArtStation`, `Unreal Engine 5`, `V-Ray render`, `by Greg Rutkowski and Artgerm`.
6.  **Technical Details:** The "magic keywords." `8k`, `UHD`, `hyper-detailed`, `intricate details`, `sharp focus`, `depth of field (bokeh)`, `f/1.8`.

---
**Example Transformation:**
* **User Prompt:** "a cat in a hat"
* **Your 'Top Class' Prompt:** "A hyper-detailed, photorealistic 8k close-up portrait of a fluffy ginger cat wearing a tiny, dapper felt fedora, cinematic lighting with a shallow depth of field, sharp focus on the cat's vibrant green eyes, trending on ArtStation, shot on a Sony A7R IV."

Now, transform the user's prompt into a 'Top Class' masterpiece.
"""
        # --- END OF MODIFIED SECTION ---
        
        final_prompt = enhancer_llm.invoke(enhancer_prompt).content.strip()
        
        encoded_prompt = quote_plus(final_prompt)
        
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?token={pollinations_token}"
        
        response = requests.get(url, timeout=120)
        
        response.raise_for_status() 
        
        img_bytes = response.content
        img = Image.open(BytesIO(img_bytes))
        
        return {"image": img, "caption": f"Your prompt: '{prompt}'"}

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return {"error": f"The image generation service returned an error: {http_err}"}
    
    except requests.exceptions.ReadTimeout as timeout_err:
        logging.error(f"Image generation timed out: {timeout_err}")
        return {"error": "The image generation service timed out (took longer than 120s). It might be very busy. Please try again in a moment."}
    
    except Exception as e:
        logging.error(f"An unexpected error occurred in image generation: {e}")
        return {"error": f"Failed to generate image: {e}"}
# ===================================================================
# TOOL 3: FILE ANALYSIS TOOL (Unchanged)
# ===================================================================

def file_analysis_tool(question: str, file_content_as_text: str, google_api_key: str):
    """
    Use this tool when the user has uploaded a file and is asking a question about it.
    This tool is now empowered to use its own expertise to provide a comprehensive analysis.
    """
    print("---TOOL: Executing Empowered File Analysis---")
    streaming_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, streaming=True) 
    
    prompt = f"""
    **Your Persona:** You are a highly intelligent AI assistant and a multi-disciplinary expert. Your goal is to provide the most helpful and insightful analysis possible, combining the provided file content with your own vast knowledge.

    **The Task:** A user has uploaded a file and asked a question. Use the file content as the primary source of truth and context, but you are encouraged to enrich your answer with your own expertise, especially when asked for an evaluation, opinion, or a subjective score.

    **How to Behave Based on File Content:**
    * **If the file appears to be CODE (e.g., Python, JavaScript, etc.):** Act as a senior software engineer. Analyze its structure, logic, efficiency, and style. If the user asks for a score or review, provide a thoughtful evaluation with justifications and suggestions for improvement.
    * **If the file appears to be TEXT (e.g., an article, report, essay):** Act as a research analyst. Summarize key points, extract specific information, and answer the user's questions. You can add relevant context from your own knowledge if it enhances the answer (e.g., providing historical context for an article).
    * **For all other file types:** Do your best to interpret the text content and provide a helpful, intelligent response to the user's question.

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
# TOOL 4: WEB SEARCH & REAL-TIME DATA ANALYSIS (Unchanged)
# ===================================================================
def web_search_tool(query: str, tavily_api_key: str, google_api_key: str) -> str:
    """
    Use this tool to get real-time information, answer questions about current events,
    or for any query that requires up-to-date knowledge from the internet.
    """
    print("---TOOL: Executing Web Search and Analysis---")
    try:
        tavily = TavilyClient(api_key=tavily_api_key)
        search_results = tavily.search(query=query, search_depth="advanced", max_results=5)
        
        search_content = "\n".join([result["content"] for result in search_results["results"]])
        
        analyzer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        analysis_prompt = f"""
        You are an expert research analyst. You have been given a user's query and the results from a web search.
        Your task is to provide a clear, concise, and comprehensive answer to the user's query based *only* on the provided search results.
        Cite your sources using the information available in the search results if possible.

        ### User Query:
        {query}

        ### Web Search Results:
        ---
        {search_content}
        ---

        Your Answer:
        """
        final_answer = analyzer_llm.invoke(analysis_prompt).content
        return final_answer
        
    except Exception as e:
        return f"⚠️ Web search failed: {e}"
    
# ===================================================================
# THE AGENT: A "WORKSHOP MANAGER" THAT CHOOSES THE RIGHT TOOL
# ===================================================================

class AgentState(TypedDict):
    query: str
    route: str  # Add this key to store the router's decision
    history: list[BaseMessage]
    final_response: Optional[any]

# --- MODIFIED: Wrapper function now needs the mistral_api_key ---
def call_comparison_tool(state: AgentState, google_api_key: str, groq_api_key: str, mistral_api_key: str):
    response = comparison_and_evaluation_tool(
        state['query'], 
        google_api_key, 
        groq_api_key,
        mistral_api_key  # Pass the key through
    )
    return {"final_response": response}

def call_image_tool(state: AgentState, google_api_key: str, pollinations_token: str):
    response = image_generation_tool(state['query'], google_api_key, pollinations_token)
    return {"final_response": response}

def call_web_search_tool(state: AgentState, tavily_api_key: str, google_api_key: str):
    response = web_search_tool(state['query'], tavily_api_key, google_api_key)
    return {"final_response": response}

# --- Router is UNCHANGED (still routes to 'comparison_tool') ---
def router(state: AgentState, google_api_key: str):
    """The brain of the agent. Decides which tool to use and updates the 'route' state key."""
    print("---AGENT: Routing query---")
    router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    query = state['query']
    
    router_prompt = f"""
    You are a master routing agent. Determine the user's primary intent and select the appropriate tool. You have three choices:
    1.  `comparison_tool`: Use for complex questions, coding problems, analysis, or any text-based query needing a detailed, evaluated answer.
    2.  `image_generation_tool`: Use ONLY if the user explicitly asks to create, draw, or generate an image.
    3.  `web_search_tool`: Use this for any query that requires real-time, up-to-date information. This includes questions about current events, news, weather, recent scientific discoveries, or topics created after 2023.

    User Query: "{query}"
    Return ONLY the tool name (`comparison_tool`, `image_generation_tool`, or `web_search_tool`).
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

# --- MODIFIED: Define the Agentic Graph, adding the mistral_api_key ---
def build_agent(google_api_key: str, groq_api_key: str, pollinations_token: str, tavily_api_key: str, mistral_api_key: str):
    """
    Builds the agent graph.
    Requires an additional 'mistral_api_key' for the judge.
    """
    workflow = StateGraph(AgentState)

    router_with_keys = partial(router, google_api_key=google_api_key)
    
    # Update the partial function for the comparison node to include the new key
    comparison_node = partial(
        call_comparison_tool, 
        google_api_key=google_api_key, 
        groq_api_key=groq_api_key,
        mistral_api_key=mistral_api_key # Pass key to the node
    )
    image_node = partial(call_image_tool, google_api_key=google_api_key, pollinations_token=pollinations_token)
    web_search_node = partial(call_web_search_tool, tavily_api_key=tavily_api_key, google_api_key=google_api_key)

    workflow.add_node("router", router_with_keys)
    workflow.add_node("comparison_chat", comparison_node)
    workflow.add_node("image_generator", image_node)
    workflow.add_node("web_search", web_search_node)

    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"comparison_chat": "comparison_chat", "image_generator": "image_generator", "web_search": "web_search"}
    )
    
    workflow.add_edge("comparison_chat", END)
    workflow.add_edge("image_generator", END)
    workflow.add_edge("web_search", END)
    
    return workflow.compile() 