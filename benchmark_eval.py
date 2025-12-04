# benchmark_eval.py
"""
Benchmark on GSM8K:
- Gemini 2.5 Flash alone
- Groq (openai/gpt-oss-120b via your choose_groq_model) alone
- Your multi-agent orchestrator (Gemini 2.5 + Groq 120B + Mistral judge)

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
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = f"""
You are solving a mathematics word problem.

You may show reasoning, but make sure the **final numeric answer**
appears clearly on the last line.

Question:
{question}

Answer:
"""
    return llm.invoke(prompt).content.strip()


def answer_with_groq_120b(question: str) -> str:
    """
    Baseline: Groq, but routed so choose_groq_model() selects openai/gpt-oss-120b.

    We prepend a short hint ("This is a mathematics word problem...")
    so your choose_groq_model() sees words like 'mathematics' and
    picks the 120B model.
    """
    routed_question = (
        "This is a mathematics word problem. Solve it carefully step by step.\n\n"
        + question
    )
    result = query_groq(routed_question, GROQ_API_KEY)
    if isinstance(result, dict):
        return result["content"].strip()
    return ""  # if query_groq returned an error string


def answer_with_agent(question: str) -> str:
    """
    Your full multi-agent pipeline: Gemini 2.5 + Groq (120B) + Mistral judge.

    We call comparison_and_evaluation_tool() with the same 'routed' question
    so internally choose_groq_model() also selects openai/gpt-oss-120b.
    """
    routed_question = (
        "This is a mathematics word problem. Solve it carefully step by step.\n\n"
        + question
    )

    full_output = comparison_and_evaluation_tool(
        query=routed_question,
        google_api_key=GOOGLE_API_KEY,
        groq_api_key=GROQ_API_KEY,
        mistral_api_key=MISTRAL_API_KEY,
    )

    # Your function returns markdown like:
    # "### 🏆 Judged Best Answer (...)\n#### Model: ...\n\n<ANSWER>\n\n### 🧠 Judge's Evaluation ..."
    match = re.search(
        r"### 🏆 Judged Best Answer.*?Model:.*?\n\n(.*?)(?:\n\n### 🧠|$)",
        full_output,
        re.S,
    )
    if match:
        return match.group(1).strip()

    return full_output.strip()


# ================= 3. GSM8K helpers =================

def extract_gsm8k_answer(raw: str) -> str:
    """
    GSM8K gold answers often end with '#### 42'.
    This extracts the '42' part (but we still normalize further below).
    """
    match = re.search(r"####\s*([^\n]+)", raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


def extract_final_number(text: str) -> str | None:
    """
    Extract the final numeric answer from any model output by:
    - Handling LaTeX \boxed{7}
    - Removing commas in numbers (57,500 -> 57500)
    - Stripping markdown symbols (*, $, `)
    - Grabbing the LAST number mentioned (models usually end with the final answer)
    """
    if text is None:
        return None

    # Remove LaTeX boxed
    clean = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)

    # Remove commas inside numbers and common wrappers
    clean = clean.replace(",", "")
    clean = re.sub(r"[\*\$`]", "", clean)

    # Find all integer / float numbers
    nums = re.findall(r"-?\d+\.?\d*", clean)
    if not nums:
        return None

    return nums[-1]


def normalize(ans: str) -> str:
    """
    Normalize an answer for comparison:
    - Prefer the final numeric value if present
    - Else, lowercase trimmed text
    """
    if ans is None:
        return ""
    num = extract_final_number(ans)
    if num is not None:
        return num
    return ans.strip().lower()


def is_correct(pred: str, gold: str) -> bool:
    """
    Compare predicted vs gold answer after normalization.
    """
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)
    return str(pred_norm) == str(gold_norm)


# ================= 4. Main evaluation loop =================

def evaluate_gsm8k(num_examples: int = 20):
    """
    Evaluate Gemini 2.5, Groq 120B, and your agent on first N GSM8K test problems.
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
        print(f"Gold answer (raw): {gold}  -> normalized: {normalize(gold)}")

        # ---- Gemini 2.5 ----
        try:
            g_ans = answer_with_gemini(question)
            g_ok = is_correct(g_ans, gold)
            gemini_ok += int(g_ok)
            print(f"\nGemini 2.5 answer: {g_ans}")
            print(f"Gemini 2.5 -> {'✅' if g_ok else '❌'}  (normalized: {normalize(g_ans)})")
        except Exception as e:
            print(f"Gemini error: {e}")

        # ---- Groq 120B ----
        try:
            q_ans = answer_with_groq_120b(question)
            q_ok = is_correct(q_ans, gold)
            groq_ok += int(q_ok)
            print(f"\nGroq 120B answer: {q_ans}")
            print(f"Groq 120B   -> {'✅' if q_ok else '❌'}  (normalized: {normalize(q_ans)})")
        except Exception as e:
            print(f"Groq error: {e}")

        # ---- Your Agent (Gemini + Groq + Mistral) ----
        try:
            a_ans = answer_with_agent(question)
            a_ok = is_correct(a_ans, gold)
            agent_ok += int(a_ok)
            print(f"\nAgent answer: {a_ans}")
            print(f"Agent       -> {'✅' if a_ok else '❌'}  (normalized: {normalize(a_ans)})")
        except Exception as e:
            print(f"Agent error: {e}")

    print("\n" + "=" * 70)
    print("📊 FINAL GSM8K RESULTS")
    print(f"Examples:        {num_examples}")
    print(f"Gemini 2.5 acc.: {gemini_ok / num_examples:.3f}")
    print(f"Groq 120B acc.:  {groq_ok / num_examples:.3f}")
    print(f"Agent acc.:      {agent_ok / num_examples:.3f}")


if __name__ == "__main__":
    evaluate_gsm8k(num_examples=20)   # increase later if you want
