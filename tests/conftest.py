"""
tests/conftest.py
-----------------
Pytest configuration for AGENT_MIND.
Provides mocks for optional / heavy cloud and ML dependencies when running tests
in local environments.
"""

import sys
import types
from unittest.mock import MagicMock

# Mock heavy/cloud libraries if not installed in local environment
for mod_name in [
    "qdrant_client",
    "qdrant_client.models",
    "sentence_transformers",
    "streamlit",
    "streamlit.components.v1",
    "tavily",
    "requests",
    "requests.exceptions",
    "PIL",
    "PIL.Image",
    "langchain_google_genai",
    "langgraph",
    "langgraph.graph",
    "fitz",
    "pytesseract",
    "pypdf",
    "PyPDF2",
]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

# Mock langchain and langchain.schema with real message classes if not installed
if "langchain" not in sys.modules:
    try:
        __import__("langchain")
    except ImportError:
        langchain_mod = types.ModuleType("langchain")
        schema_mod = types.ModuleType("langchain.schema")

        class BaseMessage:
            def __init__(self, content=""):
                self.content = content
            def __repr__(self):
                return f"{self.__class__.__name__}(content={self.content!r})"

        class HumanMessage(BaseMessage):
            pass

        class AIMessage(BaseMessage):
            pass

        schema_mod.BaseMessage = BaseMessage
        schema_mod.HumanMessage = HumanMessage
        schema_mod.AIMessage = AIMessage

        sys.modules["langchain"] = langchain_mod
        sys.modules["langchain.schema"] = schema_mod
        langchain_mod.schema = schema_mod
