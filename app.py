import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pytesseract
from datetime import datetime, timezone
import re
import time
import pandas as pd
import random
from urllib.parse import quote_plus
import asyncio
import json
from collections import defaultdict

# --- TTS Library ---
from gtts import gTTS

# --- LangChain Schema for Memory ---
from langchain.schema import HumanMessage, AIMessage

# --- Import the Agent Logic ---
from agent import build_agent, file_analysis_tool

# --- Vector Memory ---
from vector_memory import save_memory, clear_memory, get_memory_count, get_all_memories

# --- Security Layer ---
from security_guard import (
    input_guard,
    output_guard,
    memory_guard,
    audit_logger,
    make_session_id,
)

# =================================================================================
# PAGE SETUP
# =================================================================================
st.set_page_config(page_title="🤖 Neuroplexa AI", page_icon="🧠", layout="wide")

# =================================================================================
# SESSION ID — CROSS-DEVICE PERSISTENT
# =================================================================================
components.html(
    """
    <script>
        let userId = localStorage.getItem("neuroplexa_user_id");
        if (!userId) {
            userId = "user_" + Math.random().toString(36).substr(2, 12);
            localStorage.setItem("neuroplexa_user_id", userId);
        }
        const url = new URL(window.parent.location.href);
        if (url.searchParams.get("uid") !== userId) {
            url.searchParams.set("uid", userId);
            window.parent.location.href = url.toString();
        }
    </script>
    """,
    height=0,
)

query_params = st.query_params
auto_uid     = query_params.get("uid", "")

if "session_id" not in st.session_state:
    st.session_state.session_id = auto_uid if auto_uid else "default_user"
elif auto_uid and auto_uid != "default_user" and st.session_state.session_id == "default_user":
    st.session_state.session_id = auto_uid

if "manual_name_set" not in st.session_state:
    st.session_state.manual_name_set = False

SESSION_ID = st.session_state.session_id


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def generate_audio_from_text(text: str) -> bytes | None:
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
    button_id = f"copy_btn_{button_key}"
    text_id   = f"text_{button_key}"
    safe_text = text_to_copy.replace('"', '&quot;').replace("'", "&apos;").replace('\n', '\\n')
    html_code = f"""
        <textarea id="{text_id}" style="position: absolute; left: -9999px;">{safe_text}</textarea>
        <button id="{button_id}" style="
            background-color: transparent; border: 1px solid #4CAF50; color: #4CAF50;
            padding: 5px 10px; border-radius: 5px; cursor: pointer; font-size: 12px;">
            Copy Text
        </button>"""
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
                }}, function(err) {{ console.error('Could not copy text: ', err); }});
            }});
        </script>"""
    st.components.v1.html(html_code + js_code, height=40)


def set_animated_fluid_background():
    st.markdown(
        """
        <style>
        @keyframes fluidMove {
            0%   { background-position: 0%   50%; }
            25%  { background-position: 100% 50%; }
            50%  { background-position: 100% 100%; }
            75%  { background-position: 0%   100%; }
            100% { background-position: 0%   50%; }
        }
        .stApp {
            background: linear-gradient(45deg, #0a0c27, #001f5a, #4a0d6a, #0052D4);
            background-size: 300% 300%;
            animation: fluidMove 20s ease infinite;
            color: #ffffff;
        }
        [data-testid="stSidebar"] > div:first-child { background-color: rgba(10, 12, 39, 0.95); }
        .st-emotion-cache-16txtl3                   { background-color: rgba(10, 12, 39, 0.8); }
        [data-testid="chat-message-container"] {
            background-color: rgba(0, 31, 90, 0.7);
            border-radius: 10px; padding: 15px !important;
            margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.1);
        }
        .history-card {
            background: rgba(0, 31, 90, 0.6);
            border: 1px solid rgba(79, 142, 247, 0.3);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 8px;
        }
        .date-header {
            color: #4f8ef7;
            font-size: 1.1rem;
            font-weight: bold;
            margin: 16px 0 8px 0;
            border-bottom: 1px solid rgba(79,142,247,0.3);
            padding-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_animated_fluid_background()


# =================================================================================
# LOAD SECRETS
# =================================================================================
try:
    google_api_key     = st.secrets["GOOGLE_API_KEY"]
    pollinations_token = st.secrets["POLLINATIONS_TOKEN"]
    groq_api_key       = st.secrets["GROQ_API_KEY"]
    mistral_api_key    = st.secrets["MISTRAL_API_KEY"]
    tavily_api_key     = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Please add it to your Streamlit Secrets.")
    st.stop()


# =================================================================================
# SESSION STATE
# =================================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_requests":  0,
        "tool_usage":      {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
        "total_latency":   0.0,
        "average_latency": 0.0,
        "accuracy_feedback": {"👍": 0, "👎": 0},
        "last_query_details": {},
    }
if "security_events" not in st.session_state:
    st.session_state.security_events = []   # in-session event log for sidebar badge


# =================================================================================
# HISTORY BROWSER HELPERS
# =================================================================================

def label_date(ts_str: str) -> str:
    try:
        dt    = datetime.fromisoformat(ts_str)
        today = datetime.utcnow().date()
        diff  = (today - dt.date()).days
        if diff == 0:   return "📅 Today"
        elif diff == 1: return "📅 Yesterday"
        elif diff < 7:  return f"📅 {dt.strftime('%A')}"
        else:           return f"📅 {dt.strftime('%B %d, %Y')}"
    except Exception:
        return "📅 Unknown Date"


def format_time(ts_str: str) -> str:
    try:
        return datetime.fromisoformat(ts_str).strftime("%I:%M %p")
    except Exception:
        return ""


def group_messages_into_conversations(messages: list) -> list:
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "user":
            asst_msg = messages[i + 1] if (i + 1 < len(messages) and
                       messages[i + 1].get("role") == "assistant") else None
            turns.append({
                "date_label":   label_date(msg.get("timestamp", "")),
                "time":         format_time(msg.get("timestamp", "")),
                "timestamp":    msg.get("timestamp", ""),
                "user_content": msg.get("content", ""),
                "asst_content": asst_msg.get("content", "") if asst_msg else "",
            })
            i += 2 if asst_msg else 1
        else:
            i += 1

    grouped = defaultdict(list)
    for turn in turns:
        grouped[turn["date_label"]].append(turn)

    date_order    = ["📅 Today", "📅 Yesterday"]
    sorted_groups = []
    for label in date_order:
        if label in grouped:
            sorted_groups.append((label, grouped[label]))
    for label, turns in grouped.items():
        if label not in date_order:
            sorted_groups.append((label, turns))
    return sorted_groups


def detect_tool(content: str) -> tuple[str, str]:
    if "🏆 Judged Best Answer" in content:
        return "⚖️ Comparison", "#4f8ef7"
    elif "Image generated" in content or "Your prompt:" in content:
        return "🎨 Image Gen", "#8e44ad"
    elif "web search" in content.lower() or "search results" in content.lower():
        return "🌐 Web Search", "#1abc9c"
    elif "file" in content.lower() or "document" in content.lower():
        return "📂 File Analysis", "#e67e22"
    else:
        return "⚖️ Comparison", "#4f8ef7"


def render_history_tab():
    SESSION_ID = st.session_state.session_id
    st.markdown("## 📜 Chat History Browser")
    st.caption(f"Showing all past conversations for memory ID: `{SESSION_ID}`")

    col_r, col_s, _ = st.columns([1, 1, 6])
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            if "history_cache" in st.session_state:
                del st.session_state["history_cache"]
            st.rerun()
    with col_s:
        search_term = st.text_input(
            "🔍 Search memories...", placeholder="e.g. Python, cricket",
            label_visibility="collapsed", key="history_search",
        )
    st.markdown("---")

    if "history_cache" not in st.session_state:
        with st.spinner("Loading your conversation history from Qdrant..."):
            st.session_state.history_cache = get_all_memories(SESSION_ID)

    all_messages = st.session_state.history_cache
    if search_term:
        all_messages = [m for m in all_messages
                        if search_term.lower() in m.get("content", "").lower()]

    if not all_messages:
        st.info("🧠 No memories found yet. Start chatting and your conversations will appear here!")
        return

    grouped     = group_messages_into_conversations(all_messages)
    total_turns = sum(len(t) for _, t in grouped)

    m1, m2, m3 = st.columns(3)
    m1.metric("💬 Total Conversations", total_turns)
    m2.metric("🧠 Total Messages", len(all_messages))
    m3.metric("📅 Days Active", len(grouped))
    st.markdown("---")

    for date_label, turns in grouped:
        st.markdown(
            f"<div class='date-header'>{date_label} — "
            f"{len(turns)} conversation{'s' if len(turns) > 1 else ''}</div>",
            unsafe_allow_html=True,
        )
        for idx, turn in enumerate(reversed(turns)):
            user_preview   = (turn["user_content"][:80] + "…"
                              if len(turn["user_content"]) > 80
                              else turn["user_content"])
            tool_label, tool_color = detect_tool(turn["asst_content"])
            expander_title = f"{turn['time']}  |  {tool_label}  |  \"{user_preview}\""

            with st.expander(expander_title, expanded=False):
                badge_col, ts_col = st.columns([1, 3])
                with badge_col:
                    st.markdown(
                        f'<span style="background:{tool_color};color:white;padding:3px 10px;'
                        f'border-radius:12px;font-size:0.8rem;font-weight:bold">{tool_label}</span>',
                        unsafe_allow_html=True,
                    )
                with ts_col:
                    st.caption(f"🕐 {turn['timestamp'][:19].replace('T', ' ')} UTC")
                st.markdown("---")

                st.markdown("**🧑 You asked:**")
                st.markdown(
                    f'<div style="background:rgba(79,142,247,0.15);border-left:3px solid #4f8ef7;'
                    f'padding:10px 14px;border-radius:6px;margin-bottom:8px">'
                    f'{turn["user_content"]}</div>',
                    unsafe_allow_html=True,
                )

                if turn["asst_content"]:
                    st.markdown("**🤖 Neuroplexa replied:**")
                    asst_text = turn["asst_content"]
                    if len(asst_text) > 1500:
                        st.markdown(
                            f'<div style="background:rgba(26,188,156,0.1);border-left:3px solid #1abc9c;'
                            f'padding:10px 14px;border-radius:6px">{asst_text[:1500]}…</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander("Show full response"):
                            st.markdown(asst_text)
                    else:
                        st.markdown(
                            f'<div style="background:rgba(26,188,156,0.1);border-left:3px solid #1abc9c;'
                            f'padding:10px 14px;border-radius:6px">{asst_text}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("")
                if st.button("↩️ Reload this conversation into chat",
                             key=f"reload_{date_label}_{idx}", use_container_width=False):
                    st.session_state.messages = [
                        {"role": "user",      "text": turn["user_content"]},
                        {"role": "assistant", "text": turn["asst_content"], "audio_bytes": None},
                    ]
                    st.success("✅ Loaded! Switch to the 💬 Chat tab to continue.")


# =================================================================================
# SECURITY DASHBOARD
# =================================================================================

def render_security_tab():
    st.markdown("## 🔒 Security Dashboard")
    st.caption("Real-time audit log of all security events for your session.")

    SESSION_ID = st.session_state.session_id

    col_r, _ = st.columns([1, 7])
    with col_r:
        if st.button("🔄 Refresh Events", use_container_width=True):
            if "audit_cache" in st.session_state:
                del st.session_state["audit_cache"]
            st.rerun()

    # Load audit events (cached per session)
    if "audit_cache" not in st.session_state:
        with st.spinner("Loading security events from Qdrant..."):
            st.session_state.audit_cache = audit_logger.get_events(limit=500)

    all_events = st.session_state.audit_cache
    my_events  = [e for e in all_events if e.get("session_id") == SESSION_ID]

    # Summary metrics
    stats = audit_logger.get_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔍 Total Events",       stats["total"])
    c2.metric("✅ INFO",               stats["by_severity"].get("INFO",  0))
    c3.metric("⚠️ WARN",              stats["by_severity"].get("WARN",  0), delta_color="inverse")
    c4.metric("🚫 BLOCKED",           stats["by_severity"].get("BLOCK", 0), delta_color="inverse")
    c5.metric("💉 Injections Today",  stats.get("injections_today", 0),    delta_color="inverse")

    st.markdown("---")

    # Event type bar chart
    if stats["by_type"]:
        st.markdown("#### 📊 Event Breakdown")
        type_df = pd.DataFrame(
            list(stats["by_type"].items()), columns=["Event Type", "Count"]
        ).sort_values("Count", ascending=False)
        st.bar_chart(type_df.set_index("Event Type"), height=200)

    st.markdown("---")

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        scope = st.radio(
            "Show events for:", ["My session only", "All sessions"],
            horizontal=True, key="audit_scope",
        )
    with col_f2:
        sev_filter = st.multiselect(
            "Filter by severity:", ["INFO", "WARN", "BLOCK"],
            default=["WARN", "BLOCK"], key="audit_sev",
        )

    events_to_show = my_events if scope == "My session only" else all_events
    if sev_filter:
        events_to_show = [e for e in events_to_show if e.get("severity") in sev_filter]

    st.markdown(f"**Showing {len(events_to_show)} event(s)**")
    st.markdown("---")

    if not events_to_show:
        st.success("✅ No security events matching your filter. All clear!")
        return

    SEVERITY_ICONS = {"INFO": "🟢", "WARN": "🟡", "BLOCK": "🔴"}

    for evt in events_to_show[:100]:
        sev    = evt.get("severity",   "INFO")
        etype  = evt.get("event_type", "UNKNOWN")
        ts     = evt.get("timestamp",  "")[:19].replace("T", " ")
        sid    = evt.get("session_id", "")[:16]
        icon   = SEVERITY_ICONS.get(sev, "⚪")
        detail = evt.get("detail",   "")
        finds  = evt.get("findings", [])

        with st.expander(f"{icon} `{etype}`  |  {ts}  |  session: `{sid}`", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Severity:** `{sev}`")
                st.markdown(f"**Event:** `{etype}`")
            with col_b:
                st.markdown(f"**Time:** {ts} UTC")
                st.markdown(f"**Session:** `{sid}`")
            if detail:
                st.markdown(f"**Detail:** {detail}")
            if finds:
                st.markdown(f"**Findings:** `{', '.join(str(f) for f in finds)}`")


# =================================================================================
# SIDEBAR
# =================================================================================
with st.sidebar:
    st.title("🧠 Neuroplexa AI")
    st.write("Multi-Model Agent with Long-Term Memory")

    # ── Identity ──────────────────────────────────────────────────
    st.markdown("### 👤 Your Memory Identity")
    st.caption("Same name + PIN = same memories on ANY device.")

    name_input = st.text_input(
        "Enter your name:",
        placeholder="e.g. sounak",
        value="" if not st.session_state.manual_name_set else st.session_state.session_id,
        key="name_input_field",
    )
    pin_input = st.text_input(
        "PIN (optional):",
        placeholder="e.g. 1234",
        type="password",
        key="pin_input_field",
        max_chars=8,
        help="A PIN makes your memory ID impossible to guess even if someone knows your name.",
    )

    col_set, col_reset = st.columns(2)
    with col_set:
        if st.button("✅ Set Name", use_container_width=True):
            if name_input.strip():
                if pin_input.strip():
                    clean = make_session_id(name_input.strip(), pin_input.strip())
                    st.success("🔐 Secure hashed ID set.")
                else:
                    clean = name_input.strip().lower().replace(" ", "_")
                    st.success(f"Memory ID: `{clean}`")

                st.session_state.session_id      = clean
                st.session_state.manual_name_set = True
                for cache_key in ["history_cache", "audit_cache"]:
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]
                SESSION_ID = clean
                audit_logger.log(SESSION_ID, "INPUT_PASSED", detail="Session ID set by user.")
                st.rerun()
            else:
                st.warning("Please enter a name first.")

    with col_reset:
        if st.button("🔄 Reset ID", use_container_width=True):
            st.session_state.manual_name_set = False
            st.session_state.session_id      = auto_uid if auto_uid else "default_user"
            for cache_key in ["history_cache", "audit_cache"]:
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
            SESSION_ID = st.session_state.session_id
            st.rerun()

    mem_count = get_memory_count(SESSION_ID)
    st.metric("🧠 Memories stored", mem_count)
    st.caption(f"🔑 Memory ID: `{SESSION_ID[:20]}{'…' if len(SESSION_ID) > 20 else ''}`")

    # ── Security status ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛡️ Security Status")
    recent_blocks = [e for e in st.session_state.security_events
                     if e.get("severity") == "BLOCK"]
    if recent_blocks:
        st.warning(f"⚠️ {len(recent_blocks)} blocked event(s) this session.")
    else:
        st.success("✅ No threats detected this session.")

    st.markdown("---")

    # ── Google Search ─────────────────────────────────────────────
    st.header("🔍 Google Search")
    search_query = st.text_input("Search the web directly...", key="google_search")
    if st.button("Search"):
        if search_query:
            encoded_query = quote_plus(search_query)
            st.link_button("Open Google search results",
                           url=f"https://www.google.com/search?q={encoded_query}")
        else:
            st.warning("Please enter a search query.")

    # ── File Upload ───────────────────────────────────────────────
    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader(
        "Upload a file to ask questions about it",
        type=["pdf", "txt", "py", "js", "html", "css"],
    )

    # ── Utilities ─────────────────────────────────────────────────
    st.header("🧭 Utilities")
    if st.button("🗑️ Clear Chat History & Reset Metrics"):
        clear_memory(session_id=SESSION_ID)
        for cache_key in ["history_cache", "audit_cache"]:
            if cache_key in st.session_state:
                del st.session_state[cache_key]
        st.session_state.messages        = []
        st.session_state.trajectory      = []
        st.session_state.security_events = []
        st.session_state.metrics = {
            "total_requests": 0,
            "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
            "total_latency": 0.0, "average_latency": 0.0,
            "accuracy_feedback": {"👍": 0, "👎": 0}, "last_query_details": {},
        }
        st.rerun()

    st.markdown("### 💡 AI Tip")
    st.info(random.choice([
        "Check the 📜 History tab to browse all your past conversations!",
        "Use 🔍 Search in History to find any past message instantly.",
        "Click ↩️ Reload in History to continue any past conversation.",
        "Type your name + PIN for maximum memory security!",
        "Same name + PIN on phone + laptop = same AI memory everywhere.",
        "Check 🔒 Security tab to see your threat audit log.",
    ]))

    # ── Live stats ────────────────────────────────────────────────
    st.header("📊 Live Stats")
    metrics  = st.session_state.metrics
    col1, col2 = st.columns(2)
    col1.metric("Requests",    metrics["total_requests"])
    col2.metric("Avg Latency", f"{metrics['average_latency']:.2f} s")

    if metrics["total_requests"] > 0:
        tool_df = pd.DataFrame(
            list(metrics["tool_usage"].items()), columns=["Tool", "Count"]
        )
        st.bar_chart(tool_df.set_index("Tool"), height=150)

    st.subheader("📈 Live App Accuracy")
    total_feedback = (metrics["accuracy_feedback"]["👍"]
                      + metrics["accuracy_feedback"]["👎"])
    if total_feedback > 0:
        positive_rate = (metrics["accuracy_feedback"]["👍"] / total_feedback) * 100
        st.metric("Positive Feedback Rate", f"{positive_rate:.1f}%")
    else:
        st.info("No feedback yet.")

    with st.expander("🕵️ See Last Query Details"):
        st.json(metrics["last_query_details"])


# =================================================================================
# MAIN TABS
# =================================================================================
chat_tab, history_tab, security_tab = st.tabs(["💬 Chat", "📜 History", "🔒 Security"])

with history_tab:
    render_history_tab()

with security_tab:
    render_security_tab()


# =================================================================================
# CHAT TAB
# =================================================================================
with chat_tab:
    st.title("🧠 Neuroplexa AI Workspace")

    # Render existing messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if "text" in message:
                st.markdown(message["text"])
                if message["role"] == "assistant":
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        audio_bytes = message.get("audio_bytes")
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                        else:
                            if st.button("🎧 Listen", key=f"listen_btn_{i}"):
                                with st.spinner("Generating audio..."):
                                    new_audio = generate_audio_from_text(message["text"])
                                    if new_audio:
                                        st.session_state.messages[i]["audio_bytes"] = new_audio
                                        st.rerun()
                                    else:
                                        st.error("Audio failed.")
                    with c2:
                        create_copy_button(message["text"], button_key=f"text_copy_{i}")

            if "image_bytes" in message:
                st.image(Image.open(BytesIO(message["image_bytes"])),
                         caption=message.get("caption"))
                st.download_button(
                    label="⬇️ Download Image", data=message["image_bytes"],
                    file_name=f"generated_image_{i}.png", mime="image/png",
                    key=f"download_btn_{i}",
                )

    # ── Agent async runner ────────────────────────────────────────
    async def run_agent_and_capture_trajectory(agent, inputs):
        trace_steps    = []
        current_step   = {}
        final_response = None
        tool_used      = "N/A"

        async for event in agent.astream_events(inputs, version="v1"):
            kind = event["event"]
            if kind == "on_chain_start" and event["name"] != "LangGraph":
                current_step = {
                    "name":  event["name"],
                    "input": event["data"].get("input"),
                    "output": None,
                }
                if   event["name"] == "comparison_chat":   tool_used = "Comparison"
                elif event["name"] == "image_generator":   tool_used = "Image Gen"
                elif event["name"] == "web_search":        tool_used = "Web Search"

            if kind == "on_chain_end" and event["name"] != "LangGraph":
                output = event["data"].get("output")
                if current_step.get("name") == event["name"]:
                    current_step["output"] = output
                    trace_steps.append(current_step)
                    current_step = {}
                if isinstance(output, dict) and "final_response" in output:
                    final_response = output["final_response"]

        return final_response, trace_steps, tool_used

    def pretty_print_dict(d):
        def safe_converter(o):
            if isinstance(o, (Image.Image, bytes)):
                return f"<{type(o).__name__} object>"
            return str(o)
        if not isinstance(d, dict):
            return f"```\n{str(d)}\n```"
        return "```json\n" + json.dumps(d, indent=2, default=safe_converter) + "\n```"

    # ── Chat input ────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question, request an image, or upload a file..."):
        SESSION_ID = st.session_state.session_id

        # ════════════════════════════════════════════════════════
        # SECURITY GATE — INPUT GUARD
        # ════════════════════════════════════════════════════════
        input_result = input_guard.validate(prompt)

        audit_logger.log(
            session_id = SESSION_ID,
            event_type = input_result.event_type,
            detail     = input_result.reason or "Input accepted.",
            findings   = input_result.findings,
        )
        st.session_state.security_events.append({
            "event_type": input_result.event_type,
            "severity":   "BLOCK" if not input_result.passed else "INFO",
            "detail":     input_result.reason,
        })

        if not input_result.passed:
            friendly = {
                "PROMPT_INJECTION": "🚫 That request looks like a policy violation. Please rephrase.",
                "INPUT_TOO_LONG":   f"✂️ Your message is too long (max {input_guard.MAX_QUERY_LEN} chars). Please shorten it.",
                "GIBBERISH_INPUT":  "🤔 That doesn't look like a valid question. Could you rephrase?",
                "EMPTY_INPUT":      "💬 Please enter a message.",
            }
            st.warning(friendly.get(input_result.event_type,
                                    "⚠️ Request could not be processed. Please try again."))
            st.stop()

        # Use the sanitised query from here on
        clean_prompt = input_result.clean_text

        # ════════════════════════════════════════════════════════
        # PROCEED WITH CLEAN PROMPT
        # ════════════════════════════════════════════════════════
        st.session_state.messages.append({"role": "user", "text": clean_prompt})
        save_memory(role="user", content=clean_prompt, session_id=SESSION_ID)

        for cache_key in ["history_cache", "audit_cache"]:
            if cache_key in st.session_state:
                del st.session_state[cache_key]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time    = time.time()
                tool_used_key = ""

                # ── PATH 1: File Analysis ─────────────────────
                if uploaded_file:
                    tool_used_key = "File Analysis"
                    file_bytes    = uploaded_file.read()
                    file_text     = ""

                    if "pdf" in uploaded_file.type:
                        reader = PdfReader(BytesIO(file_bytes))
                        for page in reader.pages:
                            file_text += page.extract_text() or ""
                        if not file_text.strip():
                            st.info("No text layer found. Performing OCR...")
                            doc = fitz.open(stream=file_bytes, filetype="pdf")
                            for page in doc:
                                pix = page.get_pixmap()
                                img = Image.open(BytesIO(pix.tobytes("png")))
                                file_text += pytesseract.image_to_string(img)
                    else:
                        file_text = file_bytes.decode("utf-8", errors="ignore")

                    response_stream = file_analysis_tool(clean_prompt, file_text, google_api_key)
                    full_response   = st.write_stream(response_stream)

                    # Output guard
                    out_result    = output_guard.validate(full_response)
                    full_response = out_result.clean_text
                    if out_result.event_type not in ("OUTPUT_PASSED",):
                        audit_logger.log(SESSION_ID, out_result.event_type,
                                         detail=out_result.reason,
                                         findings=out_result.findings)
                    if out_result.event_type == "OUTPUT_REDACTED":
                        st.caption("🛡️ *Some sensitive content was automatically redacted.*")

                    save_memory(role="assistant", content=full_response, session_id=SESSION_ID)
                    st.session_state.messages.append(
                        {"role": "assistant", "text": full_response, "audio_bytes": None}
                    )

                # ── PATH 2: Agent Execution ───────────────────
                else:
                    agent = build_agent(
                        google_api_key, groq_api_key, pollinations_token,
                        tavily_api_key, mistral_api_key,
                    )

                    chat_history = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["text"]))
                        elif msg["role"] == "assistant" and "text" in msg:
                            chat_history.append(AIMessage(content=msg["text"]))

                    inputs = {
                        "query":      clean_prompt,
                        "history":    chat_history,
                        "session_id": SESSION_ID,
                    }

                    final_response, trace_steps, tool_used_key = asyncio.run(
                        run_agent_and_capture_trajectory(agent, inputs)
                    )
                    st.session_state.trajectory.append(
                        {"prompt": clean_prompt, "steps": trace_steps}
                    )

                    if isinstance(final_response, str):
                        # Output guard
                        out_result     = output_guard.validate(final_response)
                        final_response = out_result.clean_text

                        if out_result.event_type not in ("OUTPUT_PASSED",):
                            audit_logger.log(SESSION_ID, out_result.event_type,
                                             detail=out_result.reason,
                                             findings=out_result.findings)
                            st.session_state.security_events.append({
                                "event_type": out_result.event_type,
                                "severity":   "WARN" if out_result.passed else "BLOCK",
                                "detail":     out_result.reason,
                            })

                        if not out_result.passed:
                            st.error("🚫 Response blocked by content policy.")
                        else:
                            st.markdown(final_response)
                            if out_result.event_type == "OUTPUT_REDACTED":
                                st.caption("🛡️ *Some sensitive content was automatically redacted.*")

                        save_memory(role="assistant", content=final_response, session_id=SESSION_ID)
                        st.session_state.messages.append(
                            {"role": "assistant", "text": final_response, "audio_bytes": None}
                        )

                    elif isinstance(final_response, dict) and "image" in final_response:
                        img_data = final_response["image"]
                        buf      = BytesIO()
                        img_data.save(buf, format="PNG")
                        byte_im  = buf.getvalue()
                        st.image(byte_im, caption=final_response.get("caption", clean_prompt))
                        st.session_state.messages.append({
                            "role":        "assistant",
                            "image_bytes": byte_im,
                            "text":        f"Image generated for: *{clean_prompt}*",
                            "caption":     final_response.get("caption", clean_prompt),
                        })

                    else:
                        error_msg = (
                            final_response.get("error", "An unknown error occurred.")
                            if isinstance(final_response, dict) else str(final_response)
                        )
                        st.markdown(f"**Error:** {error_msg}")
                        st.session_state.messages.append(
                            {"role": "assistant", "text": f"Error: {error_msg}", "audio_bytes": None}
                        )

                # Metrics update
                end_time = time.time()
                latency  = end_time - start_time
                metrics  = st.session_state.metrics
                metrics["total_requests"] += 1
                if tool_used_key and tool_used_key in metrics["tool_usage"]:
                    metrics["tool_usage"][tool_used_key] += 1
                metrics["total_latency"]      += latency
                metrics["average_latency"]     = (metrics["total_latency"]
                                                  / metrics["total_requests"])
                metrics["last_query_details"]  = {
                    "timestamp":       datetime.now().isoformat(),
                    "prompt":          clean_prompt,
                    "tool_used":       tool_used_key,
                    "latency_seconds": round(latency, 2),
                    "security_status": input_result.event_type,
                }
                st.rerun()

    # ── Trajectory Debug ──────────────────────────────────────────
    if st.session_state.trajectory:
        with st.expander("🕵️ Agent Trajectory (Debug View)", expanded=False):
            for run in reversed(st.session_state.trajectory):
                st.markdown(f"#### Prompt: *'{run.get('prompt', 'N/A')}'*")
                for step in run.get("steps", []):
                    st.markdown(f"##### 🎬 Step: `{step.get('name', 'Unknown')}`")
                    c_in, c_out = st.columns(2)
                    with c_in:
                        st.markdown("**Input:**")
                        st.markdown(pretty_print_dict(step.get("input", {})),
                                    unsafe_allow_html=True)
                    with c_out:
                        st.markdown("**Output:**")
                        st.markdown(pretty_print_dict(step.get("output", {})),
                                    unsafe_allow_html=True)
                st.markdown("---")

    # ── Feedback ──────────────────────────────────────────────────
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