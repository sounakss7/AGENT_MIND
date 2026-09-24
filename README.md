# 🧠 Neuroplexa AI Workspace (AGENT_MIND)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangGraph-Orchestration-green?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple?style=for-the-badge)

**Neuroplexa AI** is a production-grade, multi-model AI agent architecture built with Streamlit, LangGraph, and Qdrant. It is designed to act as a secure, long-term memory-enabled workspace that dynamically routes queries across multiple state-of-the-art LLMs (Groq, Gemini, DeepSeek, Kimi, and Mistral).

---

## 🏗️ System Architecture & Core Flow

The system processes user inputs through a robust pipeline designed for speed, security, and accuracy:
1. **Input & Guardrails:** User query is sanitized for PII, prompt injections, and jailbreaks.
2. **Intent Routing:** The query is routed (either via ultra-fast RegEx or an LLM Router) to the correct tool or chat pipeline.
3. **Context Retrieval:** Relevant past conversations are fetched from Qdrant Vector DB using semantic search.
4. **Execution & Tools:** The agent executes the request (using Web Search, File OCR, or Image Generation if needed).
5. **Arena / LLM-as-a-Judge (Optional):** Complex queries are sent to two competing models. A third "Judge" model evaluates them blind and declares a winner.
6. **Memory Distillation:** The final output is distilled and saved back to the Vector DB for future context.

---

## ✨ Key Features & Technical Areas

### 🧭 1. High-Performance Query Routing
The system features a dual-mode routing engine to direct user intent:
*   **Self-Routing Engine (Zero-Latency):** A pure Python, deterministic RegEx-based router that classifies intent (`image_generator`, `web_search`, or `chat`) in **<0.1ms**. It bypasses external APIs entirely, saving 1-3 seconds of latency and preserving API quota.
*   **LLM Routing Engine:** A dynamic fallback router that uses massive models (e.g., Groq `gpt-oss-120b`) to semantically analyze complex intents when strict rules fall short.

### 🧠 2. Long-Term Semantic Memory (Qdrant)
Unlike standard chatbots that forget everything on refresh, Neuroplexa has a persistent identity-based memory system.
*   **Vector Database:** Powered by Qdrant Cloud.
*   **Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` to vectorize conversations.
*   **Identity Scoping:** Memories are securely isolated. A unique Session ID is generated using a hashed combination of the user's Name + PIN, ensuring you only retrieve your own past context.

### ⚔️ 3. Multi-Model Arena & Blind Evaluation
To guarantee the highest quality answers, the system employs an "Arena Mode":
*   **Blind A/B Testing:** Two distinct models (e.g., DeepSeek vs. Kimi, or Groq vs. Gemini) generate answers simultaneously.
*   **LLM-as-a-Judge:** A dedicated judge model (default: Mistral `open-mistral-7b` with Gemini fallback) evaluates both responses on Correctness, Clarity, and Safety, outputting a structured JSON verdict.
*   **Human-in-the-Loop:** Users can disagree with the Judge and click **"Promote this response as winner"** to manually override the AI's decision.

### 🛡️ 4. Security & Guardrails
Enterprise-grade safety checks are enforced on both inputs and outputs:
*   **PII Redaction:** Automatically detects and masks Credit Card numbers (via Luhn algorithm checks), IPv4 addresses, and sensitive tokens before they hit external APIs.
*   **Jailbreak Defense:** Blocks "DAN" prompts, developer mode exploits, and toxic content.
*   **Audit Logging:** Security events (e.g., rate-limit triggers, unauthorized access attempts) are logged securely in Qdrant for administrative review.

### 🛠️ 5. Multi-Modal Tool Ecosystem
The agent is equipped with several real-world tools:
*   **Tavily Web Search:** Fetches real-time internet data, news, and weather.
*   **File Pipeline (OCR):** Extracts text from PDFs and images using `PyPDF2`, `pymupdf`, and `pytesseract`. Caches extractions via SHA-256 to prevent redundant processing.
*   **Pollinations Image Gen:** Generates high-quality images directly in the chat based on user prompts.

---

## ⚙️ Technology Stack

*   **Frontend:** Streamlit (Python)
*   **Agent Orchestration:** LangChain, LangGraph
*   **Vector Database:** Qdrant Cloud
*   **Embeddings:** HuggingFace `sentence-transformers`
*   **Supported LLMs:** Groq (`gpt-oss-120b`, `gpt-oss-20b`), Google Gemini (1.5 Flash/Pro), DeepSeek (`deepseek-chat`), Kimi (`moonshot-v1-8k`), Mistral AI.
*   **CI/CD:** GitHub Actions (Pytest suite with 100+ unit/integration tests).

---

## 🚀 Local Setup & Installation

### Prerequisites
*   Python 3.11 or 3.12
*   Qdrant Cloud Account (Free tier)
*   API Keys (Google Gemini, Groq, Mistral, DeepSeek, Kimi, Tavily)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AGENT_MIND.git
   cd AGENT_MIND
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets:**
   Copy the example secrets file and add your API keys:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   *Edit `.streamlit/secrets.toml` to include your QDRANT_URL, QDRANT_API_KEY, and model API keys.*

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Run the Test Suite:**
   ```bash
   python -m pytest tests/ -v
   ```

---

## 🗺️ Future Roadmap (Scaling for Production)
The next evolution of this project involves shifting from a stateful Streamlit monolith to a highly scalable, decoupled architecture to support thousands of concurrent users:
*   **Frontend:** Rewrite UI in Next.js + TailwindCSS (Deployed on Vercel).
*   **Backend:** Wrap LangGraph logic in a stateless FastAPI server with Server-Sent Events (SSE) for streaming (Deployed on Render).
*   **Database:** Integrate PostgreSQL (Supabase) for persistent, relational chat history.
