import os
import re
import math
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# NOTE: Assuming agent.py contains comparison_and_evaluation_tool and query_groq
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


# ================= 2. Model wrappers (No Change) =================

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

    # Use the robust extraction pattern for the Judge's final block
    match = re.search(
        r"### Chosen Answer\n(.*?)(?=\n\n### 🧠|$)",
        full_output,
        re.S,
    )
    if match:
        return match.group(1).strip()

    # Fallback in case the markdown structure is completely different
    return full_output.strip()


# ================= 3. GSM8K helpers (Updated Robust Logic) =================

def extract_gsm8k_answer(raw: str) -> str:
    """
    GSM8K gold answers often end with '#### 42'.
    This extracts the '42' part.
    """
    match = re.search(r"####\s*([^\n]+)", raw)
    if match:
        return match.group(1).strip()
    return raw.strip()

def normalize_text(text: str) -> str:
    """Removes non-essential symbols and units for cleaner extraction."""
    # Remove commas inside numbers and common wrappers/units
    clean = re.sub(r'[$,]', '', text)
    # Remove common units following a number
    clean = re.sub(r'(?<=\d)\s*(years|lemons|dollars|cents|kg|L|hours|minutes|seconds|percent|%)', '', clean, flags=re.IGNORECASE)
    # Remove markdown/latex symbols that can surround a number
    clean = re.sub(r'[\*\$`#\\]', '', clean)
    return clean.strip()

def extract_final_number(text: str) -> Optional[str]:
    """
    [FIXED/ROBUST]: Extracts the final numerical answer from a complex LLM output string,
    prioritizing boxed/explicit formats and falling back to the last number.
    Handles sentences, full stops, and multiple numbers.
    """
    if not text:
        return None
        
    normalized_text = normalize_text(text)
    
    # --- Priority 1: LaTeX Boxed Format (\boxed{}) ---
    # We strip the box first in normalize_text, so this checks the original box content
    match_boxed = re.search(r"\\boxed\{(-?\d+(\.\d+)?)\}", text)
    if match_boxed:
        return normalize_text(match_boxed.group(1))

    # --- Priority 2: Last Number in the Text ---
    # Finds all numbers (integers or floats, potentially negative)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", normalized_text)

    # Return the last number found (most reliable fallback for math)
    if numbers:
        return numbers[-1]
    
    # --- Priority 3: Number on the very last line (Fallback only if no number was found earlier) ---
    # Finds number at the end of the very last line
    last_line = normalized_text.split('\n')[-1].strip()
    match_last_line = re.search(r"(-?\d+(?:\.\d+)?)\s*$", last_line)
    if match_last_line:
        return match_last_line.group(1)

    return None

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
        # If a number is extracted, return it as the normalized answer
        return num.strip()
        
    # Fallback for non-numeric answers (rare in GSM8K)
    return ans.strip().lower()

def is_correct(pred: str, gold: str) -> bool:
    """
    Compare predicted vs gold answer using numerical tolerance for floats.
    """
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)

    # Check for simple string match (mostly for integers)
    if pred_norm == gold_norm:
        return True

    # Check for numerical closeness (float comparison)
    try:
        pred_float = float(pred_norm)
        gold_float = float(gold_norm)
        # Use math.isclose for robust float comparison
        return math.isclose(pred_float, gold_float, rel_tol=1e-4)
    except ValueError:
        # If conversion to float fails (e.g., non-numeric answer), rely on exact string match
        return False


# ================= 4. Main evaluation loop (No Change) =================

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
        print(f"Gold answer (raw): {gold}   -> normalized: {normalize(gold)}")

        # ---- Gemini 2.5 ----
        try:
            g_ans = answer_with_gemini(question)
            g_ok = is_correct(g_ans, gold)
            gemini_ok += int(g_ok)
            print(f"\nGemini 2.5 answer: {g_ans}")
            print(f"Gemini 2.5 -> {'✅' if g_ok else '❌'}     (normalized: {normalize(g_ans)})")
        except Exception as e:
            print(f"Gemini error: {e}")

        # ---- Groq 120B ----
        try:
            q_ans = answer_with_groq_120b(question)
            q_ok = is_correct(q_ans, gold)
            groq_ok += int(q_ok)
            print(f"\nGroq 120B answer: {q_ans}")
            print(f"Groq 120B    -> {'✅' if q_ok else '❌'}  (normalized: {normalize(q_ans)})")
        except Exception as e:
            print(f"Groq error: {e}")

        # ---- Your Agent (Gemini + Groq + Mistral) ----
        try:
            a_ans = answer_with_agent(question)
            a_ok = is_correct(a_ans, gold)
            agent_ok += int(a_ok)
            print(f"\nAgent answer: {a_ans}")
            print(f"Agent        -> {'✅' if a_ok else '❌'}  (normalized: {normalize(a_ans)})")
        except Exception as e:
            print(f"Agent error: {e}")

    print("\n" + "=" * 70)
    print("📊 FINAL GSM8K RESULTS")
    print(f"Examples:       {num_examples}")
    print(f"Gemini 2.5 acc.: {gemini_ok / num_examples:.3f}")
    print(f"Groq 120B acc.:     {groq_ok / num_examples:.3f}")
    print(f"Agent acc.:         {agent_ok / num_examples:.3f}")


if __name__ == "__main__":
    evaluate_gsm8k(num_examples=50)