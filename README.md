# 🧠 AGENT_MIND

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Built by Sounak**

An advanced, production-ready multi-tool AI agent built with **Streamlit** and **LangGraph**. This application serves as a robust agentic framework that can search the web, generate images, perform comparative analysis between LLMs (with MoA judging), and analyze user-uploaded files, all backed by a high-security execution layer and long-term vector memory.

---

## ✨ Features

-   **🤖 Agentic Router:** A central LangGraph agent that intelligently routes user queries to the most appropriate tool using both short-term conversational context and long-term semantic memory.
-   **⚖️ Dual-Model Comparison & Evaluation (MoA):** Runs the same query concurrently on both **Google's Gemini 2.5 Flash** and a model via **Groq** (GPT-OSS or Llama-3.1). A "Judge" LLM (**Mistral**) evaluating both responses autonomously to declare a winner.
-   **🛡️ Comprehensive Security Guard:** Features a built-in security layer (`security_guard.py`) that includes:
    - **Input Guard:** Blocks prompt injections, jailbreaks, and Gibberish.
    - **Output & Memory Guard:** Auto-redacts PII (Emails, Phone numbers, Keys, Aadhaar, PAN) using sophisticated pattern matching before displaying outputs or saving to memory.
    - **Audit Logger:** Writes all security actions/blocks to a Qdrant audit collection.
-   **🧠 Long-Term Vector Memory:** Powered by **Qdrant Cloud** and **SentenceTransformers**. Retains context across sessions securely, meaning the agent remembers past interactions using semantic search.
-   **🌐 Real-Time Web Search:** Utilizes the **Tavily API** to fetch up-to-date real-time intelligence off the internet.
-   **🎨 AI Image Generation:** Integrates with the **Pollinations AI** framework to generate high-quality images based on dynamically enhanced (Gemini) user prompts.
-   **📂 Advanced File Analysis:**
    - Supports deep-dive analysis on `.pdf`, `.txt`, `.py`, `.js`, etc., acting as an Expert Persona.
-   **🎧 On-Demand Audio (TTS):** Converts agent text responses into speech playback using `gTTS`.

## ⚙️ Architecture & Workflow

The architecture is built upon a LangGraph state machine bridging core capabilities. 
When a query enters the system:
1. Passes through the **InputGuard** to drop malicious scripts or jailbreaks.
2. Short-term and Long-term context are retrieved from memory.
3. The **Router** LLM selects an optimal tool (`File Analysis`, `Web Search`, `Image Gen`, or `Comparison Chat`).
4. Outputs are passed through the **OutputGuard** to redact PII and toxic content.
5. Clean data is sent back to the User and embedded into **Qdrant Vector Memory**.

![Agent Workflow Diagram](workflow_architecture.png)

## 🛠️ Tech Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **AI Orchestration:** [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
-   **Vector Database:** [Qdrant Cloud](https://qdrant.tech/)
-   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
-   **LLMs / APIs:** 
    - Google Gemini
    - Groq API (GPT-OSS, Llama 3)
    - Mistral AI 
    - Tavily Search API
    - Pollinations AI

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

-   Python 3.9+
-   Git

### 2. Clone the Repository

```bash
git clone https://github.com/sounakss7/AGENT_MIND.git
cd AGENT_MIND
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install the necessary pip packages. 

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys (Streamlit Secrets)

The application uses Streamlit Secrets. Create the `.streamlit` directory and add your keys:

1.  Create a folder named `.streamlit` in the root directory.
2.  Inside this folder, create a file named `secrets.toml`.
3.  Add your API credentials as follows:

```toml
# .streamlit/secrets.toml

# LLM Providers
GOOGLE_API_KEY = "YOUR_GOOGLE_KEY"
GROQ_API_KEY = "YOUR_GROQ_KEY"
MISTRAL_API_KEY = "YOUR_MISTRAL_KEY"

# Tools
TAVILY_API_KEY = "YOUR_TAVILY_KEY"
POLLINATIONS_TOKEN = "YOUR_POLLINATIONS_TOKEN" # Optional

# Qdrant Vector Memory (Long-term Context & Auditing)
QDRANT_URL = "YOUR_QDRANT_CLUSTER_URL"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"
```

## ▶️ How to Run the Application

Once your virtual environment is active and keys are set, run the app using Streamlit:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.
