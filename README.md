# 🧠 Neuroplexa AI Workspace (AGENT_MIND)

<div align="center">

[![Patent Filed](https://img.shields.io/badge/Patent-Filed_%23202631059925-orange?style=for-the-badge&logo=probot)](https://github.com/sounakss7)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Orchestration-green?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple?style=for-the-badge&logo=qdrant)](https://qdrant.tech)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Test_Suite-101%2F101_Passing-brightgreen?style=for-the-badge)](tests/)

**A High-Performance, Multi-Modal Agentic AI Workspace with Zero-Latency Deterministic Routing, Identity-Scoped Vector Memory, and Blind LLM-as-a-Judge Arena Architecture.**

[Architecture](#-system-architecture--flow) • [Patent Specifications](#-intellectual-property--patent-information) • [Core Subsystems](#-core-subsystems-deep-dive) • [Security & Guardrails](#-enterprise-grade-zero-trust-guardrails) • [Setup & Deployment](#-setup--installation-guide) • [Scaling Roadmap](#-system-design-scaling-roadmap)

</div>

---

## 📜 Intellectual Property & Patent Information

| Field | Detail |
| :--- | :--- |
| **Patent Status** | **Patent Filed / Patent Pending** |
| **Application Number** | **`#202631059925`** |
| **Jurisdiction** | Indian Patent Office (IPO) |
| **Inventor** | **Sounak Sarkar** ([@sounakss7](https://github.com/sounakss7)) |
| **Inventor Role** | AI/ML Engineer • Agentic AI Systems Architect |
| **Institution** | Dr. Sudhir Chandra Sur Institute of Technology (MAKAUT) |

### Patented Architectural Innovations
The architecture underlying **Neuroplexa AI (AGENT_MIND)** is protected under Patent Application **`#202631059925`**, covering three novel technical claims:

1. **Deterministic Zero-Latency Multi-Modal Intent Routing with Negative Boundary Constraints:**
   - A sub-millisecond ($<0.1\text{ ms}$) non-neural pattern classification algorithm (`SelfRouter`) that resolves query intent across multi-modal modalities (Text, Image Synthesis, Web Search, OCR Parsing) without making external LLM inference calls, avoiding token exhaustion and 429 rate-limiting.
2. **Closed-Loop Mixture of Agents (MoA) Arena with Distilled Semantic Vector Memory Injection:**
   - A multi-agent evaluation protocol wherein concurrent competing LLMs are judged blind by an independent arbitrator model using a multi-dimensional JSON scoring matrix, with dynamic human-in-the-loop override capability and automatic factual knowledge distillation into an encrypted vector database.
3. **Cryptographically Scoped Identity-Partitioned Vector Storage:**
   - A client-side deterministic identity generation system utilizing salted double-hashing ($\text{HMAC-SHA256}$) across user identifiers to guarantee cross-tenant isolation and zero data leakage within shared cloud vector collections.

---

## 🏗️ System Architecture & Flow

The following sequence details how Neuroplexa AI processes user queries from raw input to verified memory distillation:

```mermaid
flowchart TD
    User((User Ingestion)) --> UI[Streamlit Workspace]
    UI --> Guardrails{Zero-Trust Guardrails\nInputGuard & PII}

    Guardrails -- Violates Policy --> SecLog[Block & Emit Security Audit Log]
    SecLog --> AuditDB[(Qdrant: security_audit)]
    Guardrails -- Sanitized Clean --> Router{Dual-Mode Router}

    subgraph "Routing Engine (Patented)"
        Router -- "0.1ms Deterministic" --> SelfRouter[SelfRouter Automata]
        Router -- "Ambiguous Semantic" --> LLMRouter[LLM Router Engine]
    end

    SelfRouter --> MemoryEngine[Identity-Scoped Context Retrieval]
    LLMRouter --> MemoryEngine

    subgraph "Vector Engine (Qdrant Cloud)"
        MemoryEngine <--> KeyHasher[HMAC-SHA256 Identity Scoper]
        KeyHasher <--> Qdrant[(Qdrant Cloud: agent_mind_memory)]
        Qdrant -.-> EmbeddingModel[sentence-transformers/all-MiniLM-L6-v2]
    end

    MemoryEngine --> ExecutionPath{Execution Routing}

    subgraph "Multi-Modal Execution & Tool Ecosystem"
        ExecutionPath -- "Search Intent" --> Tavily[Tavily Search API]
        ExecutionPath -- "Image Intent" --> Pollinations[Pollinations AI Synthesis]
        ExecutionPath -- "Document/PDF" --> OCR[PyMuPDF + Tesseract OCR]
        ExecutionPath -- "Standard Prompt" --> PrimaryLLM[Groq / Gemini / DeepSeek / Kimi]
        OCR --> PrimaryLLM
    end

    ExecutionPath -- "Arena A/B Mode" --> ArenaSplit{Model Arena Dispatcher}

    subgraph "Mixture of Agents (MoA) Arena"
        ArenaSplit --> ContenderA[Contender A\n(e.g., DeepSeek-V3)]
        ArenaSplit --> ContenderB[Contender B\n(e.g., Kimi / Groq)]
        ContenderA --> BlindJudge[Mistral Impartial Judge]
        ContenderB --> BlindJudge
        BlindJudge --> JudgeVerdict[JSON Evaluation Verdict]
        JudgeVerdict -. "Human Disagreement" .-> HITL[Human-in-the-Loop Override]
    end

    PrimaryLLM --> OutputGate((Output Synthesis))
    Tavily --> OutputGate
    Pollinations --> OutputGate
    JudgeVerdict --> OutputGate
    HITL --> OutputGate

    OutputGate --> OutputGuard{OutputGuard & Sanitizer}
    OutputGuard --> UIResponse[Render Response in UI]

    OutputGuard --> DistillAgent[Memory Distillation Worker]
    DistillAgent -->|Extract Core Knowledge| Qdrant
```

---

## 🧮 Mathematical & Algorithmic Formulation

### 1. Identity-Scoped Hash Isolation
To prevent cross-tenant data leakage in multi-user environments without requiring heavyweight relational auth databases, user memory partitions are deterministically scoped via:

$$\text{SessionID} = \text{HMAC-SHA256}\Big(\text{Name} \parallel \text{PIN}, \mathcal{K}_{\text{pepper}}\Big)$$

Where:
- $\text{Name}$ and $\text{PIN}$ are client-supplied credentials.
- $\mathcal{K}_{\text{pepper}}$ is a server-side high-entropy secret stored in `.streamlit/secrets.toml`.
- Any query against Qdrant Cloud injects a payload filter: `{"must": [{"key": "session_id", "match": {"value": \text{SessionID}}}]}`.

### 2. Semantic Memory Retrieval Score
Retrieval from Qdrant Cloud computes the cosine similarity between the embedded user prompt $\mathbf{e}(q) \in \mathbb{R}^{384}$ and stored context vectors $\mathbf{e}(d_i) \in \mathbb{R}^{384}$:

$$\text{Sim}(q, d_i) = \frac{\mathbf{e}(q) \cdot \mathbf{e}(d_i)}{\|\mathbf{e}(q)\|_2 \|\mathbf{e}(d_i)\|_2}$$

Context is injected into the model prompt if and only if:

$$\text{Sim}(q, d_i) \ge \tau_{\text{threshold}} \quad (\tau = 0.55)$$

### 3. Credit Card Validation via Luhn Algorithm
Before redacting suspected payment card numbers in `security_guard.py`, the token sequence $C = [c_1, c_2, \dots, c_n]$ is validated mathematically:

$$\sum_{i=1}^{n} f(c_{n-i+1}, i) \equiv 0 \pmod{10}$$

$$\text{where } f(d, i) = \begin{cases} d & \text{if } i \text{ is odd} \\ 2d - 9 & \text{if } i \text{ is even and } 2d > 9 \\ 2d & \text{if } i \text{ is even and } 2d \le 9 \end{cases}$$

---

## 🔬 Core Subsystems Deep Dive

### 1. 🧭 Zero-Latency Self-Routing Subsystem (`agent.py`)
Traditional agentic systems (e.g., standard LangChain agents) call an LLM to decide which tool to execute. This incurs:
- **1,500ms – 3,000ms network latency**
- **Token consumption on every prompt**
- **High vulnerability to HTTP 429 quota exhaustion**

Neuroplexa AI features a patented **`SelfRouter`** engine executing in **$<0.1\text{ ms}$** on CPU using compiled regular expression automata with strict positive signals and negative guardrail exclusions:

```python
# SelfRouter Logic Demonstration
is_image = bool(IMAGE_POSITIVE_PATTERN.search(query)) and not bool(IMAGE_NEGATIVE_GUARD.search(query))
is_search = bool(SEARCH_POSITIVE_PATTERN.search(query)) and not bool(SEARCH_NEGATIVE_GUARD.search(query))
```
- **Positive Match:** `image: "draw a cyberpunk skyline"`, `search: "what is the latest stock price of NVDA"`
- **Negative Boundary Guard:** *"Write Python code to generate an image with PIL"* $\rightarrow$ Correctly recognized as a **coding/chat request**, NOT sent to the image generator tool.

### 2. 🧠 Vector Memory & Distillation Engine (`vector_memory.py`)
- **Storage:** Managed Qdrant Cloud cluster with HNSW vector index.
- **Local Embedding Generation:** CPU-optimized `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense representations without external API calls.
- **Distillation Protocol:** Instead of storing entire multi-turn chat dialogues (which saturate context windows with conversational filler), a background distillation node prompts the model to summarize key user facts, preferences, and project context into a compact factual JSON record.

### 3. ⚔️ Mixture of Agents (MoA) Arena with Human-in-the-Loop
When high-assurance answers are required, Neuroplexa runs an impartial blind evaluation:
1. **Parallel Dispatch:** The prompt is dispatched simultaneously to two distinct frontier LLMs (e.g., DeepSeek-V3 vs Moonshot Kimi, or Groq vs Gemini).
2. **Blind Judge Evaluation:** An independent judge model (Mistral `open-mistral-7b` with Google Gemini fallback) inspects both answers anonymously:
   ```json
   {
     "reasoning": "Model A provided comprehensive code with error handling, while Model B missed edge cases.",
     "winner": "A",
     "scores": {
       "correctness": 9.5,
       "clarity": 9.0,
       "safety": 10.0
     }
   }
   ```
3. **Human-in-the-Loop (HITL) Override:** If the user disagrees with the Judge, a single click on **"Promote this response as winner"** in the UI overrides the automated verdict and stores the user-preferred output into vector memory.

### 4. 📄 Multi-Modal File & OCR Pipeline
- **Formats Supported:** Scanned PDFs, digital PDFs, PNG, JPG, TIFF, WEBP.
- **Dual-Layer Extraction:** Fast text-layer extraction via `PyMuPDF` (`fitz`) and `PyPDF2`, falling back to `pytesseract` OCR for rasterized pages.
- **Deduplication Cache:** Every uploaded file is hashed via **SHA-256**. Identical files submitted across sessions are resolved from cache in **0.001s**, saving CPU cycles.

---

## 🛡️ Enterprise-Grade Zero-Trust Guardrails

The `security_guard.py` module enforces strict boundary controls across all inputs, outputs, and storage pipelines:

| Guard Component | Execution Point | Primary Functionality |
| :--- | :--- | :--- |
| **`InputGuard`** | Pre-routing ingress | Neutralizes prompt injection, "DAN" exploits, jailbreaks, system prompt exfiltration, and toxic inputs. |
| **`OutputGuard`** | Post-inference egress | Intercepts model generation; redacts leaks of internal system prompts and auto-redacts PII/API credentials. |
| **`MemoryGuard`** | Pre-vectorization | Pre-screens text before Qdrant embedding to ensure no secrets or sensitive identity data enter long-term storage. |
| **`AuditLogger`** | Continuous | Asynchronously records threat telemetry (`INFO`, `WARN`, `BLOCK`) directly to Qdrant collection `security_audit`. |

### Auto-Redacted PII & Credential Signatures

```text
• Email Addresses               • Aadhaar Numbers (India - 12 Digits)
• Phone Numbers (Intl & India)  • PAN Cards (India - 10 Chars)
• Credit & Debit Cards (Luhn)   • Google API Keys (AIza...)
• Groq API Keys (gsk_...)       • OpenAI API Keys (sk-...)
• AWS Access Key IDs (AKIA...)  • GitHub Personal Tokens (ghp_...)
• IPv4 Network Addresses        • Bearer / JWT Tokens
```

---

## 📊 Interactive UI & Dashboards

The application is structured into three dedicated workspaces:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              🧠 Neuroplexa AI Workspace                                │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│         💬 Chat          │         📜 History          │          🔒 Security          │
├──────────────────────────┴─────────────────────────────┴───────────────────────────────┤
│  • Token Streaming       │  • Chronological Threading  │  • Real-Time Threat Log       │
│  • MoA Judged Output     │  • Full-Text Keyword Search │  • Severity Breakdown Charts  │
│  • Audio TTS Synthesis   │  • 1-Click Session Reload   │  • Injection Attack Telemetry │
│  • Inline Pollinations   │  • Filter by Modality/Tool  │  • PII Redaction Audit Feed   │
│  • Human Override Button │  • Token Usage Analytics    │  • Qdrant Cluster Health      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
AGENT_MIND/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI pipeline running pytest suite
├── .streamlit/
│   ├── config.toml              # UI styling, custom palette & typography
│   └── secrets.toml.example     # Secrets template for local development
├── tests/
│   ├── conftest.py              # Mock fixtures for CI execution without live API keys
│   ├── test_agent.py            # Unit tests for routing, execution & MoA
│   ├── test_security.py         # Test cases for PII redaction & jailbreak blocks
│   ├── test_memory.py           # Verification of Qdrant vector operations
│   └── test_phase8_hygiene.py   # Hygiene, security, and dependency verification
├── agent.py                     # LangGraph graph compilation, SelfRouter & tool dispatch
├── app.py                       # Streamlit application UI, state manager & tabs
├── security_guard.py            # Zero-trust guardrails, PII regex, Luhn check & audit logger
├── vector_memory.py             # Qdrant client wrapper, embeddings & identity hashing
├── packages.txt                 # Linux system dependencies (tesseract-ocr)
├── requirements.txt             # Pinned Python package dependencies
└── README.md                    # Core project documentation & patent claims
```

---

## 🚀 Setup & Installation Guide

### 1. Prerequisites
*   **Python:** Version `3.11` or `3.12`
*   **Git:** Version `2.x+`
*   **Tesseract OCR** *(Required for OCR capabilities on scanned documents)*:
    *   **Windows**: Download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to `PATH`.
    *   **Linux (Ubuntu/Debian)**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
    *   **macOS**: `brew install tesseract`

### 2. Clone & Environment Initialization
```bash
git clone https://github.com/sounakss7/AGENT_MIND.git
cd AGENT_MIND

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create `.streamlit/secrets.toml` from the template:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Populate the secrets file with your credentials:
```toml
# ── Core LLMs & Routing ───────────────────────────────────────
GOOGLE_API_KEY = "your_gemini_api_key"
GROQ_API_KEY = "your_groq_api_key"
MISTRAL_API_KEY = "your_mistral_api_key"
DEEPSEEK_API_KEY = "your_deepseek_api_key"
KIMI_API_KEY = "your_kimi_api_key"

# ── Multi-Modal Intelligence ──────────────────────────────────
TAVILY_API_KEY = "your_tavily_search_api_key"
POLLINATIONS_TOKEN = "your_pollinations_api_token"

# ── Vector Memory & Telemetry (Qdrant Cloud) ──────────────────
QDRANT_URL = "https://your-cluster-id.region.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "your_qdrant_api_key"

# ── Security & Authentication ─────────────────────────────────
AUTH_PEPPER = "random_high_entropy_salt_string_here"
ADMIN = false
```

### Free Tier API Key Resources:
| Provider | Modality | Sign-Up Link |
| :--- | :--- | :--- |
| **Google AI Studio** | Gemini 1.5 Flash / Pro | [aistudio.google.com](https://aistudio.google.com/) |
| **Groq Cloud** | GPT-OSS 120B / 20B Inference | [console.groq.com](https://console.groq.com/) |
| **DeepSeek Platform**| DeepSeek-V3 / R1 Reasoning | [platform.deepseek.com](https://platform.deepseek.com/) |
| **Moonshot Kimi** | 128k Long-Context Language Model | [platform.moonshot.cn](https://platform.moonshot.cn/) |
| **Mistral AI** | Mistral Small / Large (Judge) | [console.mistral.ai](https://console.mistral.ai/) |
| **Tavily AI** | Real-Time Agentic Search | [tavily.com](https://tavily.com/) |
| **Qdrant Cloud** | 1GB Free Managed Vector Cluster | [cloud.qdrant.io](https://cloud.qdrant.io/) |

### 4. Running the Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 5. Running the Test Suite
The codebase is validated by **101 unit and integration tests** testing guardrails, vector memory, and routing speed:
```bash
pytest tests/ -v
```

---

## 🛠️ Technology Stack Reference

| Layer | Component | Specification |
| :--- | :--- | :--- |
| **UI Framework** | Streamlit | Python 3.11, Custom CSS, Audio TTS player, reactive states |
| **Agent Orchestration** | LangGraph / LangChain | Directed Acyclic Graph (DAG) state graph with conditional edges |
| **Vector Database** | Qdrant Cloud | HNSW indexing, Euclidean/Cosine distance, dynamic payload filtering |
| **Embeddings** | HuggingFace | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings) |
| **Frontier LLMs** | Multi-Provider | Groq (`gpt-oss-120b`, `gpt-oss-20b`), Gemini, DeepSeek, Kimi, Mistral |
| **Web Crawling** | Tavily SDK | Structured real-time query extraction |
| **OCR & Docs** | PyMuPDF + Tesseract | 300 DPI rendering, SHA-256 cached document extraction |
| **Quality & CI/CD** | GitHub Actions | Automated linting, pytest matrix validation |

---

## 🗺️ System Design: Scaling Roadmap

To transition Neuroplexa AI from a single-node Streamlit workspace to an enterprise multi-tenant SaaS serving 50,000+ concurrent users, the following decoupled architecture is planned:

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       Next.js 15 Frontend       │       │       FastAPI Microservice      │
│     (Vercel Edge Network)       │ ----> │         (Render / AWS)          │
│ • TailwindCSS + Shadcn UI       │ <SSE- │ • Stateless LangGraph Workers   │
│ • Token-by-Token Streaming (SSE)│       │ • Async Tool Dispatch Pool      │
└─────────────────────────────────┘       └─────────────────────────────────┘
                 │                                         │
                 ▼                                         ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Supabase / Neon DB          │       │          Qdrant Cloud           │
│         (PostgreSQL)            │       │      (Distributed Cluster)      │
│ • User Authentication (OAuth)   │       │ • Semantic Vector Memory        │
│ • Relational Chat Threads       │       │ • Security Audit Telemetry Logs │
│ • Arena Vote Metrics & Billing  │       │ • Cross-Tenant Isolated Namesp. │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

1. **Frontend Decoupling:** Next.js (React 19) deployed to **Vercel** with edge caching and streaming markdown renderers.
2. **Stateless Backend:** Refactor `agent.py` into a high-throughput **FastAPI** server on **Render/Railway** utilizing Server-Sent Events (SSE).
3. **Relational Database:** Migrate conversation histories and user profiles to **PostgreSQL (Supabase)**, reserving Qdrant exclusively for semantic similarity retrieval.
4. **Caching & Rate Limiting:** Introduce **Upstash Redis** to protect LLM endpoints with sliding-window rate limiting.

---

## 👨‍💻 Inventor & Maintainer

**Sounak Sarkar**  
*AI/ML Engineer • Agentic AI Systems Architect*  
- **GitHub:** [@sounakss7](https://github.com/sounakss7)  
- **Patent Application:** `#202631059925` (IPO)  
- **Academic Affiliation:** B.Tech CSE (AI & ML) '26, Dr. Sudhir Chandra Sur Institute of Technology (MAKAUT)  
- **Location:** Kolkata, West Bengal, India

---

## 📄 License & Attribution

This project is open-source under the **MIT License**. See the [LICENSE](LICENSE) file for complete terms.

```bibtex
@misc{sarkar2026neuroplexa,
  author = {Sarkar, Sounak},
  title = {Neuroplexa AI Workspace: Deterministic Intent-Routing and Mixture-of-Agents Architecture with Distilled Vector Memory},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sounakss7/AGENT_MIND}},
  note = {Indian Patent Application No. 202631059925}
}
```

<div align="center">
<b>Built with precision. Governed by multi-model logic. Protected by Patent #202631059925.</b>
</div>
