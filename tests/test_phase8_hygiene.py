"""
tests/test_phase8_hygiene.py
----------------------------
Unit and hygiene tests for Phase 8.
Covers:
1. requirements.txt cleanliness, absence of duplicates, and exact pinned dependencies.
2. Graceful secrets loading without unhandled KeyError or startup failure.
3. MIT LICENSE file presence and validation.
4. .streamlit/secrets.toml.example template coverage.
5. GitHub Actions CI workflow file structure.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_txt_clean_and_no_duplicates():
    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt must exist"

    with open(req_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    # Normalize package names for case-insensitive duplicate check
    normalized = []
    for line in lines:
        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
        normalized.append(pkg_name)

    # Check for duplicate package declarations
    duplicates = [pkg for pkg in set(normalized) if normalized.count(pkg) > 1]
    assert duplicates == [], f"Found duplicate packages in requirements.txt: {duplicates}"

    # Verify pinned framework stack is strictly preserved
    raw_text = req_path.read_text(encoding="utf-8")
    assert "langchain==0.2.16" in raw_text
    assert "langchain-core==0.2.38" in raw_text
    assert "langchain-google-genai==1.0.10" in raw_text
    assert "langchain-openai==0.1.25" in raw_text
    assert "langgraph==0.2.28" in raw_text
    assert "qdrant-client==1.9.1" in raw_text

    # Verify unused bloat packages were removed
    assert "datasets" not in raw_text
    assert "feedparser" not in raw_text
    assert "pdf2image" not in raw_text
    assert "concurrent-log-handler" not in raw_text
    assert "fal-client" not in raw_text


def test_get_secret_graceful_fallback():
    from security_guard import get_secret

    # 1. When present in st.secrets
    with patch("streamlit.secrets", {"MY_TEST_KEY": "secret_val"}):
        assert get_secret("MY_TEST_KEY") == "secret_val"

    # 2. When missing in st.secrets but in os.environ
    with patch("streamlit.secrets", {}), patch.dict(os.environ, {"ENV_TEST_KEY": "env_val"}):
        assert get_secret("ENV_TEST_KEY") == "env_val"

    # 3. When missing in both, returns default without raising KeyError
    with patch("streamlit.secrets", {}), patch.dict(os.environ, {}, clear=True):
        assert get_secret("NON_EXISTENT_KEY", default="default_fallback") == "default_fallback"
        assert get_secret("ANOTHER_MISSING_KEY") == ""


def test_license_file_exists_and_valid():
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists(), "LICENSE file must exist"

    content = license_path.read_text(encoding="utf-8")
    assert "MIT License" in content
    assert "Copyright (c) 2026 Sounak" in content


def test_secrets_toml_example_exists_and_complete():
    example_path = REPO_ROOT / ".streamlit" / "secrets.toml.example"
    assert example_path.exists(), ".streamlit/secrets.toml.example must exist"

    content = example_path.read_text(encoding="utf-8")
    required_keys = [
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "POLLINATIONS_TOKEN",
        "TAVILY_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "AUTH_PEPPER",
        "ADMIN",
    ]
    for key in required_keys:
        assert key in content, f"Key {key} missing from secrets.toml.example"


def test_github_ci_workflow_exists_and_valid():
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), ".github/workflows/ci.yml must exist"

    content = ci_path.read_text(encoding="utf-8")
    assert "actions/checkout" in content
    assert "actions/setup-python" in content
    assert "pytest" in content
