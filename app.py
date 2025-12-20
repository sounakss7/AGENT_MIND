import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pytesseract
from datetime import datetime
import re
import time
import pandas as pd
import random
from urllib.parse import quote_plus
import asyncio
import json

# --- TTS Library ---
from gtts import gTTS

# --- LangChain Schema for Memory ---
from langchain.schema import HumanMessage, AIMessage

# --- Import the Agent Logic ---
# Ensure agent.py is in the same directory
from agent import build_agent, file_analysis_tool

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def generate_audio_from_text(text: str) -> bytes | None:
    """Generates MP3 audio from text and returns it as bytes."""
    # Clean text to remove markdown for better speech
    text = re.sub(r'(\*\*|##|###|####|`|```)', '', text)
    if not text or not text.strip():
        return None
    try:
        audio_fp = BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.getvalue()
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return None

def create_copy_button(text_to_copy: str, button_key: str):
    """Creates a button to copy text to clipboard using HTML/JS."""
    button_id = f"copy_btn_{button_key}"
    text_id = f"text_{button_key}"
    
    # Escape quotes in text to prevent JS errors
    safe_text = text_to_copy.replace('"', '&quot;').replace("'", "&apos;").replace('\n', '\\n')

    html_code = f"""
        <textarea id="{text_id}" style="position: absolute; left: -9999px;">{safe_text}</textarea>
        <button id="{button_id}" style="
            background-color: transparent; 
            border: 1px solid #4CAF50; 
            color: #4CAF50; 
            padding: 5px 10px; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 12px;">Copy Text</button>
    """
    js_code = f"""
        <script>
            document.getElementById("{button_id}").addEventListener("click", function() {{
                var text = document.getElementById("{text_id}").value;
                navigator.clipboard.writeText(text).then(function() {{
                    var btn = document.getElementById("{button_id}");
                    var originalText = btn.innerHTML;
                    btn.innerHTML = 'Copied!';
                    btn.style.borderColor = "#ffffff";
                    btn.style.color = "#ffffff";
                    setTimeout(function() {{
                        btn.innerHTML = originalText;
                        btn.style.borderColor = "#4CAF50";
                        btn.style.color = "#4CAF50";
                    }}, 2000);
                }}, function(err) {{
                    console.error('Could not copy text: ', err);
                }});
            }});
        </script>
    """
    st.components.v1.html(html_code + js_code, height=40)

def set_animated_fluid_background():
    """Sets the Deep Blue & Purple animated gradient background."""
    st.markdown(
         """
        <style>
        @keyframes fluidMove {
            0% { background-position: 0% 50%; }
            25% { background-position: 100% 50%; }
            50% { background-position: 100% 100%; }
            75% { background-position: 0% 100%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            background: linear-gradient(45deg, #0a0c27, #001f5a, #4a0d6a, #0052D4);
            background-size: 300% 300%;
            animation: fluidMove 20s ease infinite;
            color: #ffffff;
        }
        
        /* Component Styling Overrides */
        [data-testid="stSidebar"] > div:first-child {
            background-color: rgba(10, 12, 39, 0.95);
        }
        .st-emotion-cache-16txtl3 {
            background-color: rgba(10, 12, 39, 0.8);
        }
        [data-testid="chat-message-container"] {
            background-color: rgba(0, 31, 90, 0.7);
            border-radius: 10px;
            padding: 15px !important;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        </style>
        """,
         unsafe_allow_html=True
     )

# =================================================================================
# PAGE SETUP
# =================================================================================
st.set_page_config(page_title="🤖 Neuroplexa AI", page_icon="🧠", layout="wide")
set_animated_fluid_background()

# --- Load Secrets ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    pollinations_token = st.secrets["POLLINATIONS_TOKEN"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    mistral_api_key = st.secrets["MISTRAL_API_KEY"]
    tavily_api_key = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Please add it to your Streamlit Secrets.")
    st.stop()

# =================================================================================
# SESSION STATE INITIALIZATION
# =================================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_requests": 0,
        "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
        "total_latency": 0.0,
        "average_latency": 0.0,
        "accuracy_feedback": {"👍": 0, "👎": 0},
        "last_query_details": {}
    }

# =================================================================================
# SIDEBAR
# =================================================================================
with st.sidebar:
    st.title("🧠 Neuroplexa AI")
    st.write("Multi-Model Agent with Memory")
    
    st.header("🔍 Google Search")
    search_query = st.text_input("Search the web directly...", key="google_search")
    if st.button("Search"):
        if search_query:
            encoded_query = quote_plus(search_query)
            search_url = f"[https://www.google.com/search?q=](https://www.google.com/search?q=){encoded_query}"
            st.link_button("Open Google search results", url=search_url) 
        else:
            st.warning("Please enter a search query.")

    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader("Upload a file to ask questions about it", type=["pdf", "txt", "py", "js", "html", "css"])
    
    st.header("🧭 Utilities")
    if st.button("Clear Chat History & Reset Metrics"):
        st.session_state.messages = []
        st.session_state.trajectory = []
        st.session_state.metrics = {
            "total_requests": 0, "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
            "total_latency": 0.0, "average_latency": 0.0, "accuracy_feedback": {"👍": 0, "👎": 0}, "last_query_details": {}
        }
        st.rerun()

    st.markdown("### 💡 AI Tip")
    st.info(random.choice([
        "I now have memory! Ask me to 'rewrite that' or 'explain more'.",
        "The Comparison tool uses Gemini & Llama 3 simultaneously.",
        "Use 'Web Search' for news or 'Draw' for images."
    ]))
    
    st.header("📊 Live Stats")
    metrics = st.session_state.metrics
    col1, col2 = st.columns(2)
    col1.metric("Requests", metrics["total_requests"])
    col2.metric("Avg Latency", f"{metrics['average_latency']:.2f} s")
    
    if metrics["total_requests"] > 0:
        tool_df = pd.DataFrame(list(metrics["tool_usage"].items()), columns=['Tool', 'Count'])
        st.bar_chart(tool_df.set_index('Tool'), height=150)
        
    st.subheader("📈 Live App Accuracy")
    total_feedback = metrics["accuracy_feedback"]["👍"] + metrics["accuracy_feedback"]["👎"]
    if total_feedback > 0:
        positive_rate = (metrics["accuracy_feedback"]["👍"] / total_feedback) * 100
        st.metric("Positive Feedback Rate", f"{positive_rate:.1f}%")
    else:
        st.info("No feedback yet.")

    with st.expander("🕵️ See Last Query Details"):
        st.json(metrics["last_query_details"])

# =================================================================================
# CHAT DISPLAY LOGIC
# =================================================================================
st.title("🧠 Neuroplexa AI Workspace")

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # --- Text Content ---
        if "text" in message:
            st.markdown(message["text"])
            
            # --- Audio & Copy Utilities (Only for Assistant) ---
            if message["role"] == "assistant":
                c1, c2 = st.columns([1, 4])
                
                with c1:
                    audio_bytes = message.get("audio_bytes")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        if st.button("🎧 Listen", key=f"listen_btn_{i}"):
                            with st.spinner("Generating audio..."):
                                new_audio_bytes = generate_audio_from_text(message["text"])
                                if new_audio_bytes:
                                    st.session_state.messages[i]["audio_bytes"] = new_audio_bytes
                                    st.rerun()
                                else:
                                    st.error("Audio failed.")
                
                with c2:
                    create_copy_button(message["text"], button_key=f"text_copy_{i}")

        # --- Image Content ---
        if "image_bytes" in message:
            img = Image.open(BytesIO(message["image_bytes"]))
            st.image(img, caption=message.get("caption"))
            
            st.download_button(
                label="⬇️ Download Image",
                data=message["image_bytes"],
                file_name=f"generated_image_{i}.png",
                mime="image/png",
                key=f"download_btn_{i}"
            )

# =================================================================================
# AGENT DEBUGGER LOGIC
# =================================================================================
async def run_agent_and_capture_trajectory(agent, inputs):
    """
    Runs the agent using astream_events and captures a detailed trace.
    Accepts 'inputs' dict containing query and history.
    """
    trace_steps = []
    current_step = {}
    final_response = None
    tool_used = "N/A"

    # Pass the 'inputs' dict directly to astream_events
    async for event in agent.astream_events(inputs, version="v1"):
        kind = event["event"]
        
        if kind == "on_chain_start":
            if event["name"] != "LangGraph":
                current_step = {
                    "name": event["name"],
                    "input": event["data"].get("input"),
                    "output": None
                }
                if event['name'] == "comparison_chat": tool_used = "Comparison"
                elif event['name'] == "image_generator": tool_used = "Image Gen"
                elif event['name'] == "web_search": tool_used = "Web Search"

        if kind == "on_chain_end":
            if event["name"] != "LangGraph":
                output = event["data"].get("output")
                if current_step.get("name") == event["name"]:
                    current_step["output"] = output
                    trace_steps.append(current_step)
                    current_step = {}
                
                if isinstance(output, dict) and 'final_response' in output:
                    final_response = output['final_response']

    return final_response, trace_steps, tool_used

def pretty_print_dict(d):
    def safe_converter(o):
        if isinstance(o, (Image.Image, bytes)):
            return f"<{type(o).__name__} object>"
        return str(o)
    if not isinstance(d, dict): return f"```\n{str(d)}\n```"
    return "```json\n" + json.dumps(d, indent=2, default=safe_converter) + "\n```"

# =================================================================================
# MAIN CHAT INPUT & EXECUTION LOGIC
# =================================================================================
if prompt := st.chat_input("Ask a question, request an image, or upload a file..."):
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "text": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            tool_used_key = ""

            # --- PATH 1: File Analysis (if file uploaded) ---
            if uploaded_file:
                tool_used_key = "File Analysis"
                file_bytes = uploaded_file.read()
                file_text = ""
                
                # PDF Text Extraction
                if "pdf" in uploaded_file.type:
                    reader = PdfReader(BytesIO(file_bytes))
                    for page in reader.pages: file_text += page.extract_text() or ""
                    # Fallback to OCR if empty
                    if not file_text.strip():
                        st.info("No text layer found. Performing OCR (this may take a moment)...")
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        for page in doc:
                            pix = page.get_pixmap()
                            img = Image.open(BytesIO(pix.tobytes("png")))
                            file_text += pytesseract.image_to_string(img)
                else:
                    # Plain Text
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                
                # Stream Response
                response_stream = file_analysis_tool(prompt, file_text, google_api_key)
                full_response = st.write_stream(response_stream)
                
                st.session_state.messages.append({"role": "assistant", "text": full_response, "audio_bytes": None})
            
            # --- PATH 2: Agent Execution (with Memory) ---
            else:
                agent = build_agent(google_api_key, groq_api_key, pollinations_token, tavily_api_key, mistral_api_key)
                
                # 1. Convert Visual History to LangChain History
                chat_history = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["text"]))
                    elif msg["role"] == "assistant" and "text" in msg:
                        # Exclude errors or purely image messages from context if desired, 
                        # but generally we want text responses.
                        chat_history.append(AIMessage(content=msg["text"]))
                
                # 2. Construct Inputs Dictionary
                inputs = {
                    "query": prompt,
                    "history": chat_history
                }
                
                # 3. Run Agent
                final_response, trace_steps, tool_used_key = asyncio.run(run_agent_and_capture_trajectory(agent, inputs))
                st.session_state.trajectory.append({"prompt": prompt, "steps": trace_steps})

                # 4. Handle Response Types
                if isinstance(final_response, str):
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "text": final_response, "audio_bytes": None})
                
                elif isinstance(final_response, dict) and "image" in final_response:
                    img_data = final_response["image"]
                    buf = BytesIO()
                    img_data.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.image(byte_im, caption=final_response.get("caption", prompt))
                    
                    # Store image in history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "image_bytes": byte_im, 
                        "text": f"Image generated for: *{prompt}*",
                        "caption": final_response.get("caption", prompt)
                    })
                
                else:
                    error_msg = final_response.get("error", "An unknown error occurred.")
                    st.markdown(f"**Error:** {error_msg}")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_msg}", "audio_bytes": None})
            
            # --- Metrics Recording ---
            end_time = time.time()
            latency = end_time - start_time
            metrics = st.session_state.metrics
            metrics["total_requests"] += 1
            if tool_used_key and tool_used_key in metrics["tool_usage"]:
                metrics["tool_usage"][tool_used_key] += 1
            metrics["total_latency"] += latency
            metrics["average_latency"] = metrics["total_latency"] / metrics["total_requests"]
            metrics["last_query_details"] = {
                "timestamp": datetime.now().isoformat(), "prompt": prompt,
                "tool_used": tool_used_key, "latency_seconds": round(latency, 2)
            }
            
            # Rerun to update the UI (show audio button, update charts)
            st.rerun()

# =================================================================================
# FOOTER / DEBUG VIEW
# =================================================================================
if st.session_state.trajectory:
    with st.expander("🕵️ Agent Trajectory (Debug View)", expanded=False):
        for run in reversed(st.session_state.trajectory):
            st.markdown(f"#### Prompt: *'{run.get('prompt', 'N/A')}'*")
            steps = run.get('steps', [])
            for step in steps:
                st.markdown(f"##### 🎬 Step: `{step.get('name', 'Unknown')}`")
                
                c_in, c_out = st.columns(2)
                with c_in:
                    st.markdown("**Input:**")
                    st.markdown(pretty_print_dict(step.get('input', {})), unsafe_allow_html=True)
                with c_out:
                    st.markdown("**Output:**")
                    st.markdown(pretty_print_dict(step.get('output', {})), unsafe_allow_html=True)
            st.markdown("---")

# Feedback Logic
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_idx = len(st.session_state.messages) - 1
    if f"feedback_{last_idx}" not in st.session_state:
        st.write("Rate this answer:")
        c1, c2, c3 = st.columns([1, 1, 10])
        if c1.button("👍", key=f"good_{last_idx}"):
            st.session_state.metrics["accuracy_feedback"]["👍"] += 1
            st.session_state[f"feedback_{last_idx}"] = "given"
            st.toast("Feedback recorded!")
            st.rerun()
        if c2.button("👎", key=f"bad_{last_idx}"):
            st.session_state.metrics["accuracy_feedback"]["👎"] += 1
            st.session_state[f"feedback_{last_idx}"] = "given"
            st.toast("Feedback recorded!")
            st.rerun()