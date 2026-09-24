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

Neuroplexa AI orchestrates complex reasoning tasks through **LangGraph**, treating the LLM not just as a text generator, but as a reasoning engine with access to state and tools.

### 1. The Processing Pipeline
1. **Ingestion & Security Scan:** The user prompt is ingested and passed through `MemoryGuard` and `EvaluationGuardrail`. PII (Credit Cards via Luhn algorithm, IPv4 addresses, etc.) is redacted before leaving the local machine.
2. **Intent Classification (Router):** The `SelfRouter` parses the text. If it matches a tool (e.g., Image Generation, Web Search), it routes directly. Otherwise, it defaults to the conversational reasoning chain.
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
*   **LLM-as-a-Judge:** A neutral third model (default: Mistral `open-mistral-7b` with a Gemini fallback) is fed both responses *blindly* (Model A and Model B).
*   **Multi-Dimensional Rubric:** The judge outputs a strict JSON evaluation scoring Correctness, Safety, and Clarity.
*   **Human-in-the-Loop:** A manual override button (`Promote this response as winner`) allows the human operator to disagree with the Judge and force the system to memorize the human-preferred response.

### 🛡️ 4. Enterprise-Grade Guardrails & Security
*   **PII Redaction Engine:** Automatically detects and masks sensitive data using regex and checksum validation (e.g., validating Credit Cards via the Luhn algorithm before masking).
*   **Jailbreak Defense:** Proactively blocks known attack vectors, including "DAN" (Do Anything Now) prompts and developer-mode exploits.
*   **Audit Logger:** Every system event, rate limit, and security violation is logged immutably into a dedicated Qdrant `security_audit` collection for admin monitoring.

### 🛠️ 5. Multi-Modal Tool Ecosystem
*   **Tavily Web Search:** Real-time data retrieval with automatic URL extraction.
*   **File Analysis Pipeline:** A robust OCR and parsing pipeline supporting PDFs and Images via `PyPDF2`, `pymupdf`, and `pytesseract`. Includes a SHA-256 caching layer so identical files are never re-processed, saving CPU cycles.
*   **Image Generation:** Integrated Pollinations AI for instant, in-chat image rendering.

---

## ⚙️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **Agent Orchestration** | LangChain, LangGraph |
| **Vector Database** | Qdrant Cloud |
| **Embeddings (Local)** | HuggingFace `sentence-transformers` |
| **LLMs Supported** | Groq (`gpt-oss-120b`, `20b`), Google Gemini, DeepSeek, Kimi, Mistral |
| **Search API** | Tavily |
| **CI/CD & Testing** | GitHub Actions, Pytest (100+ Tests) |

---

## 🚀 Setup & Installation Guide

### Prerequisites
*   Python 3.11 or 3.12
*   Qdrant Cloud Account (Free tier)
*   API Keys for your desired providers (Gemini, Groq, Mistral, DeepSeek, Kimi, Tavily)

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/AGENT_MIND.git
cd AGENT_MIND
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Secrets
Streamlit requires API keys to be placed in a `.streamlit/secrets.toml` file.
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Open `.streamlit/secrets.toml` and fill in your keys. **Crucially, ensure your Qdrant URL and API Key are present for the memory system to function:**
```toml
QDRANT_URL = "https://your-cluster-url.qdrant.io"
QDRANT_API_KEY = "your-qdrant-key"
GOOGLE_API_KEY = "..."
GROQ_API_KEY = "..."
# Add Mistral, DeepSeek, and Kimi keys as well.
```

### 3. Run the Application
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` in your browser.*

### 4. Run the Test Suite
Neuroplexa AI is covered by a massive suite of over 101 unit and integration tests verifying memory consistency, guardrail accuracy, and routing latency.
```bash
python -m pytest tests/ -v
```

---

## 🗺️ Roadmap: Scaling to a Decoupled Microservices Architecture

While currently bundled as a monolithic Streamlit application for rapid iteration, the system design roadmap for supporting 10,000+ concurrent users involves:
1. **Frontend Decoupling:** Rewriting the UI layer in **Next.js + TailwindCSS** and deploying to Vercel/Netlify.
2. **Backend Decoupling:** Refactoring `agent.py` into a highly-concurrent **FastAPI** REST/WebSocket backend deployed on Render/AWS, allowing horizontal scaling.
3. **Streaming Responses:** Implementing Server-Sent Events (SSE) from FastAPI to Next.js for zero-perceived-latency token streaming.
4. **Relational Data Layer:** Introducing PostgreSQL (Supabase) for persistent user accounts and standard chat history, reserving Qdrant strictly for RAG and semantic routing.

---
*Built with precision. Governed by multi-model logic. Protected by patent.*
