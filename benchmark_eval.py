# benchmark_eval.py
"""
Benchmark on GSM8K:
- Gemini 2.5 Flash alone
- Groq alone
- Your multi-agent orchestrator (Gemini 2.5 + Groq + Mistral judge)

Run from project root:
    python benchmark_eval.py
"""

import os
import re

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agent import (
    comparison_and_evaluation_tool,
    query_groq,
)

# ================= 1. Load .env =================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not GOOGLE_API_KEY or not GROQ_API_KEY or not MISTRAL_API_KEY:
    raise RuntimeError(
        "❌ Missing API keys. Please set GOOGLE_API_KEY, GROQ_API_KEY, "
        "and MISTRAL_API_KEY in your .env file."
    )


# ================= 2. Model wrappers =================

def answer_with_gemini(question: str) -> str:
    """
    Baseline: Gemini 2.5 Flash alone.
    Uses the SAME model you use in your app: gemini-2.5-flash.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    # We allow it to explain if it wants; we'll just extract the final number.
    prompt = f"""
You are solving a math word problem.

- You may show your reasoning.
- But end your answer with the final numeric result on the last line.

Question:
{question}

Answer:
"""
    return llm.invoke(prompt).content.strip()


def answer_with_groq(question: str) -> str:
    """
    Baseline: Groq (your choose_groq_model + query_groq logic).
    """
    result = query_groq(question, GROQ_API_KEY)
    if isinstance(result, dict):
        return result["content"].strip()
    # if query_groq returned an error string
    return ""


def answer_with_agent(question: str) -> str:
    """
    Your full multi-agent pipeline:
    Gemini 2.5 + Groq + Mistral judge.

    Uses comparison_and_evaluation_tool from agent.py and
    extracts ONLY the 'Judged Best Answer' section.
    """
    full_output = comparison_and_evaluation_tool(
        query=question,
        google_api_key=GOOGLE_API_KEY,
        groq_api_key=GROQ_API_KEY,
        mistral_api_key=MISTRAL_API_KEY,
    )

    # Your function returns markdown like:
    # ### 🏆 Judged Best Answer (...)
    # #### Model: ...
    #
    # <ANSWER>
    #
    # ### 🧠 Judge's Evaluation ...
    match = re.search(
        r"### 🏆 Judged Best Answer.*?Model:.*?\n\n(.*?)(?:\n\n### 🧠|$)",
        full_output,
        re.S,
    )
    if match:
        best_answer = match.group(1).strip()
        return best_answer

    # Fallback: give everything back if parsing fails
    return full_output.strip()


# ================= 3. GSM8K helpers =================

def extract_gsm8k_answer(raw: str) -> str:
    """
    GSM8K gold answers often end with '#### 42'.
    This extracts the '42' part.
    """
    match = re.search(r"####\s*([^\n]+)", raw)
    if match:
        return match.group(1).strip()
    nums = re.findall(r"-?\d+\.?\d*", raw)
    return nums[-1] if nums else raw.strip()


def normalize(ans: str) -> str:
    """
    Normalise answers by:
    - taking the last number if present
    - otherwise lowercasing and stripping
    """
    ans = ans.strip()
    nums = re.findall(r"-?\d+\.?\d*", ans)
    if nums:
        return nums[-1]
    return ans.lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


# ================= 4. Main evaluation loop =================

def evaluate_gsm8k(num_examples: int = 20):
    """
    Evaluate Gemini 2.5, Groq, and your agent on first N GSM8K test problems.
    """
    print(f"\n📘 Loading GSM8K test split (first {num_examples} examples)...")
    ds = load_dataset("gsm8k", "main")["test"].select(range(num_examples))

    gemini_ok = groq_ok = agent_ok = 0

    for i, item in enumerate(ds):
        question = item["question"]
        gold_raw = item["answer"]
        gold = extract_gsm8k_answer(gold_raw)

        print("\n" + "=" * 70)
        print(f"[Q{i+1}] {question}")
        print(f"Gold answer: {gold}")

        # ---- Gemini 2.5 ----
        try:
            g_ans = answer_with_gemini(question)
            g_ok = is_correct(g_ans, gold)
            gemini_ok += int(g_ok)
            print(f"Gemini 2.5: {g_ans}  -> {'✅' if g_ok else '❌'}")
        except Exception as e:
            print(f"Gemini error: {e}")

        # ---- Groq ----
        try:
            q_ans = answer_with_groq(question)
            q_ok = is_correct(q_ans, gold)
            groq_ok += int(q_ok)
            print(f"Groq:       {q_ans}  -> {'✅' if q_ok else '❌'}")
        except Exception as e:
            print(f"Groq error: {e}")

        # ---- Your Agent ----
        try:
            a_ans = answer_with_agent(question)
            a_ok = is_correct(a_ans, gold)
            agent_ok += int(a_ok)
            print(f"Agent:      {a_ans}  -> {'✅' if a_ok else '❌'}")
        except Exception as e:
            print(f"Agent error: {e}")

    print("\n" + "=" * 70)
    print("📊 FINAL GSM8K RESULTS")
    print(f"Examples:        {num_examples}")
    print(f"Gemini 2.5 acc.: {gemini_ok / num_examples:.3f}")
    print(f"Groq acc.:       {groq_ok / num_examples:.3f}")
    print(f"Agent acc.:      {agent_ok / num_examples:.3f}")


if __name__ == "__main__":
    evaluate_gsm8k(num_examples=20)   # you can increase later
