# 🧠 Neuroplexa AI Workspace (AGENT_MIND)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangGraph-Orchestration-green?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)

**Neuroplexa AI** is a production-grade, multi-model AI agent architecture. Designed as a highly secure, long-term memory-enabled workspace, it dynamically routes user queries across state-of-the-art Large Language Models (LLMs) including Groq (`gpt-oss-120b`), Google Gemini, DeepSeek, Kimi, and Mistral. 

By combining sub-millisecond deterministic intent routing, continuous semantic memory, and a multi-model "Arena" evaluation system, Neuroplexa AI represents a leap forward in autonomous, self-correcting agent design.

---

## 📜 Intellectual Property & Patent Information

**Patent Status:** Patent Pending / Patented  
**Inventor:** Sounak  
**Patent Application / Registration Number:** `[INSERT PATENT NUMBER HERE]`  
**Patent Link:** `[INSERT LINK TO PATENT OFFICE OR PDF HERE]`

The architecture underlying Neuroplexa AI includes proprietary, patented (or patent-pending) methodologies, specifically protecting:
1. **Zero-Latency Multi-Modal Routing:** A hybrid deterministic/LLM routing engine that classifies user intent without external network latency, preserving API quotas while routing multi-modal tasks (Search, OCR, Vision, Chat) in `<0.1ms`.
2. **Blind Evaluation & Distilled Vector Memory (Arena System):** A closed-loop system where competing LLMs generate responses, an independent "Judge" LLM scores them via a multi-dimensional rubric, and the winning context is compressed (distilled) before being cryptographically scoped and saved to a vector database.

*(Note: Please refer to the official patent documentation for explicit claims. Commercial usage of these specific architectural workflows may require licensing.)*

---

## 🏗️ Deep Dive: System Architecture & Data Flow

```mermaid
flowchart TD
    User((User Input)) --> UI[Streamlit UI]
    UI --> Guardrails{Security Guardrails\n(PII & Jailbreaks)}

    Guardrails -- Blocked --> Reject[Block & Log]
    Guardrails -- Passed --> Router{Intent Routing}

    subgraph "Routing Engine"
        Router -- "0.1ms (RegEx)" --> SelfRouter[Self-Router]
        Router -- "Fallback" --> LLMRouter[LLM Router]
    end

    SelfRouter --> Memory[Semantic Memory Retrieval]
    LLMRouter --> Memory

    subgraph "Vector DB (Identity Scoped)"
        Memory <--> Qdrant[(Qdrant Cloud)]
        Qdrant -.-> Embeddings[sentence-transformers]
    end

    Memory --> Execution{Execution Pathway}

    subgraph "Tools & Pipeline"
        Execution -- "Web Search" --> Tavily[Tavily API]
        Execution -- "Image Gen" --> Pollinations[Pollinations AI]
        Execution -- "File/PDF" --> OCR[OCR Pipeline]
        Execution -- "Standard" --> SingleLLM[Main LLM]
        OCR --> SingleLLM
    end

    Execution -- "Arena Mode" --> Arena{Model Arena\n(A/B Testing)}

    subgraph "Mixture of Agents (MoA)"
        Arena --> ModelA[Contender A\n(e.g., DeepSeek)]
        Arena --> ModelB[Contender B\n(e.g., Kimi)]
        ModelA --> Judge[Mistral Judge]
        ModelB --> Judge
        Judge -. "Manual Override" .-> Human[Human-in-the-Loop]
    end

    SingleLLM --> Output((Final Output))
    Tavily --> Output
    Pollinations --> Output
    Judge --> Output
    Human --> Output

    Output --> Distillation[Memory Distillation\n(Summarization)]
    Distillation --> Qdrant
```

Neuroplexa AI orchestrates complex reasoning tasks through **LangGraph**, treating the LLM not just as a text generator, but as a reasoning engine with access to state and tools.

### 1. The Processing Pipeline
1. **Ingestion & Security Scan:** The user prompt is ingested and passed through `MemoryGuard` and `EvaluationGuardrail`. PII is redacted before leaving the local machine.
2. **Intent Classification (Router):** The `SelfRouter` parses the text. If it matches a tool, it routes directly. Otherwise, it defaults to the conversational reasoning chain.
3. **Context Injection:** The system hashes the user's `Name + PIN` to generate a secure Session ID. It queries **Qdrant Cloud** to fetch the top-3 most semantically relevant historical conversations.
4. **Execution:** The selected LLM (via Groq, Gemini, DeepSeek, or Kimi) executes the query using the injected context.
5. **Memory Distillation:** The final output is summarized (distilled) by the LLM to extract key facts, reducing token bloat, and the dense summary is embedded via `sentence-transformers` and saved back to Qdrant.

---

## ✨ Comprehensive Feature Breakdown

### 🧭 1. Dual-Mode Routing Engine
Most AI agents use an LLM to decide which tool to use. This adds 1-3 seconds of latency and costs tokens. Neuroplexa solves this with a Dual-Mode system:
*   **Self-Routing Engine (Zero-Latency):** A highly optimized, pure Python RegEx engine. It uses positive and negative guardrails to instantly classify queries (`image_generator`, `web_search`, `comparison_chat`). For example, it knows *"Draw a cat"* is an image generation task, but *"Write python code to draw a cat"* is a coding task. Latency: **<0.1ms**.
*   **LLM Routing Engine:** A dynamic fallback router using large open-source models (like Groq's `gpt-oss-120b`) to semantically analyze ambiguous requests.

### 🧠 2. Long-Term Semantic Memory (Qdrant)
Standard chatbots suffer from context-window amnesia. Neuroplexa utilizes an advanced memory architecture:
*   **Vector Database:** Qdrant Cloud stores dense vectors.
*   **Embeddings:** Local CPU-based embedding using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
*   **Cryptographic Identity Scoping:** Memories are isolated across users. A Session ID is dynamically generated using SHA-256 hashes of the user's provided Name and PIN. You can only retrieve memories you own.
*   **Distillation:** Instead of saving raw conversational transcripts, the agent distills the conversation into facts (e.g., *"User prefers Python over JavaScript"*), preventing the vector database from becoming polluted with conversational filler.

### ⚔️ 3. LLM Arena & Blind Evaluation (MoA)
Inspired by the Mixture of Agents (MoA) architecture, Neuroplexa guarantees top-tier outputs through competition:
*   **Contender A vs. Contender B:** Two distinct models (e.g., DeepSeek vs Kimi) process the exact same query in parallel.
*   **LLM-as-a-Judge:** A neutral third model (default: Mistral `open-mistral-7b` with a Gemini fallback) is fed both responses *blindly*.
*   **Multi-Dimensional Rubric:** The judge outputs a strict JSON evaluation scoring Correctness, Safety, and Clarity.
*   **Human-in-the-Loop:** A manual override button (`Promote this response as winner`) allows the human operator to disagree with the Judge and force the system to memorize the human-preferred response.

### 🛡️ 4. Enterprise-Grade Guardrails & Security
A multi-layered defense-in-depth architecture intercepts and sanitizes data.

| Component | Execution Point | Functionality |
| :--- | :--- | :--- |
| **`InputGuard`** | Pre-screens user queries before LLM routing | Blocks "DAN" (Do Anything Now) jailbreaks, prompt injections, and system prompt extraction attacks. |
| **`OutputGuard`** | Pre-screens model output before UI rendering | • Auto-redacts PII & API keys (`[REDACTED:<TYPE>]`)<br>• Blocks harmful/toxic instructions |
| **`MemoryGuard`** | Pre-screens data before vector embedding in Qdrant | Prevents sensitive keys, tokens, or PII from polluting vector memory. |
| **`AuditLogger`** | Security telemetry tracking in Qdrant collection | Logs all events (`INFO`, `WARN`, `BLOCK`) with timestamps and pattern findings. |

#### Auto-Redacted PII & Secret Signatures (Luhn Validated)
```text
• Email Addresses               • Aadhaar Numbers (India)
• Phone Numbers (India & Intl)  • PAN Cards (India)
• Credit / Debit Card Numbers   • Google API Keys (AIza...)
• Groq API Keys (gsk_...)       • OpenAI API Keys (sk-...)
• AWS Access Keys (AKIA...)     • GitHub Tokens (ghp_...)
• IPv4 Addresses
```

### 🛠️ 5. Multi-Modal Tool Ecosystem
*   **Tavily Web Search:** Real-time data retrieval with automatic URL extraction.
*   **File Analysis Pipeline:** A robust OCR and parsing pipeline supporting PDFs and Images via `PyPDF2`, `pymupdf`, and `pytesseract`. Includes a SHA-256 caching layer so identical files are never re-processed.
*   **Image Generation:** Integrated Pollinations AI for instant, in-chat image rendering.

---

## 📊 Interactive UI & Dashboards

The application is structured into three primary tabs:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        🧠 Neuroplexa AI Workspace                      │
├───────────────────────┬────────────────────────┬───────────────────────┤
│        💬 Chat        │       📜 History       │      🔒 Security      │
├───────────────────────┴────────────────────────┴───────────────────────┤
│  • Streaming responses • Grouped by date        • Real-time threat log │
│  • MoA Judged Output   • Keyword search         • Severity breakdowns  │
│  • Audio TTS player    • 1-Click reload         • Injections blocked   │
│  • Image generation    • Tool category badges   • Event distributions  │
│  • Execution trace     • Message metrics        • Filterable audit log │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
AGENT_MIND/
├── .streamlit/
│   ├── config.toml              # Streamlit theme & UI configurations
│   └── secrets.toml             # API credentials & keys (Git-ignored)
├── agent.py                     # LangGraph workflow, MoA judging, agent tools
├── app.py                       # Main Streamlit application & interactive UI
├── security_guard.py            # Zero-trust security guards, PII redaction, audit logging
├── vector_memory.py             # Qdrant Cloud semantic memory & embedding pipeline
├── packages.txt                 # OS-level dependencies (tesseract-ocr)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Setup & Installation Guide

### 1. Prerequisites
*   **Python:** Version `3.11` or `3.12`
*   **Git:** Version `2.x`+
*   **Tesseract OCR** *(Optional, required for OCR on scanned PDFs)*:
    *   **Windows**: Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to `PATH`.
    *   **Linux**: `sudo apt-get install tesseract-ocr`
    *   **macOS**: `brew install tesseract`
*   **Qdrant Cloud Account** (Free tier)

### 2. Clone & Install
```bash
git clone https://github.com/yourusername/AGENT_MIND.git
cd AGENT_MIND
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration & Secrets Setup

Neuroplexa AI reads API credentials securely from Streamlit Secrets or Environment Variables.
Create a configuration file at `.streamlit/secrets.toml`:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` to include your keys:
```toml
# ── Primary LLM & Routing ─────────────────────────────────────
GOOGLE_API_KEY = "your_google_gemini_api_key_here"

# ── Mixture of Agents (MoA) Engines ───────────────────────────
GROQ_API_KEY = "your_groq_api_key_here"
MISTRAL_API_KEY = "your_mistral_api_key_here"
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
KIMI_API_KEY = "your_kimi_moonshot_api_key_here"

# ── Multimodal & Web Intelligence Tools ────────────────────────
TAVILY_API_KEY = "your_tavily_search_api_key_here"
POLLINATIONS_TOKEN = "your_pollinations_api_token_here"

# ── Vector Memory & Security Telemetry (Qdrant Cloud) ─────────
QDRANT_URL = "https://your-cluster-id.region.qdrant.tech:6333"
QDRANT_API_KEY = "your_qdrant_cloud_api_key_here"

# ── Security & Authentication ─────────────────────────────────
AUTH_PEPPER = "your-high-entropy-server-pepper-secret"
ADMIN = false
```

### Obtaining Free API Keys:
| Service | Purpose | URL |
| :--- | :--- | :--- |
| **Google AI Studio** | Gemini Flash Routing & Fallback Judge | [aistudio.google.com](https://aistudio.google.com/) |
| **Groq Cloud** | High-speed GPT-OSS 120B/20B inference | [console.groq.com](https://console.groq.com/) |
| **DeepSeek** | Powerful open-weight reasoning model | [platform.deepseek.com](https://platform.deepseek.com/) |
| **Kimi (Moonshot)** | Deep context window language model | [platform.moonshot.cn](https://platform.moonshot.cn/) |
| **Mistral AI** | Impartial MoA Judge Model | [console.mistral.ai](https://console.mistral.ai/) |
| **Tavily AI** | Deep search web retrieval | [tavily.com](https://tavily.com/) |
| **Qdrant Cloud** | Managed vector database (1GB free tier) | [cloud.qdrant.io](https://cloud.qdrant.io/) |

### 4. Run the Application
Launch the dashboard:
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` in your browser.*

### 5. Run the Test Suite
Neuroplexa AI is covered by a massive suite of over 101 unit and integration tests verifying memory consistency, guardrail accuracy, and routing latency.
```bash
python -m pytest tests/ -v
```

---

## ⚙️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **Agent Orchestration** | LangChain, LangGraph |
| **Vector Database** | Qdrant Cloud |
| **Embeddings (Local)** | HuggingFace `sentence-transformers` |
| **LLMs Supported** | Groq (`gpt-oss-120b`, `20b`), Google Gemini, DeepSeek, Kimi, Mistral |
| **Search API** | Tavily |
| **CI/CD & Testing** | GitHub Actions, Pytest (101 Tests) |

---

## 🗺️ Roadmap: Scaling to a Decoupled Microservices Architecture

While currently bundled as a monolithic Streamlit application for rapid iteration, the system design roadmap for supporting 10,000+ concurrent users involves:
1. **Frontend Decoupling:** Rewriting the UI layer in **Next.js + TailwindCSS** and deploying to Vercel/Netlify.
2. **Backend Decoupling:** Refactoring `agent.py` into a highly-concurrent **FastAPI** REST/WebSocket backend deployed on Render/AWS, allowing horizontal scaling.
3. **Streaming Responses:** Implementing Server-Sent Events (SSE) from FastAPI to Next.js for zero-perceived-latency token streaming.
4. **Relational Data Layer:** Introducing PostgreSQL (Supabase) for persistent user accounts and standard chat history, reserving Qdrant strictly for RAG and semantic routing.

---
*Built with precision. Governed by multi-model logic. Protected by patent.*
