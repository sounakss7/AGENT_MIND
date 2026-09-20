"""
tests/conftest.py
-----------------
Pytest configuration for AGENT_MIND.
Provides mocks for optional / heavy cloud and ML dependencies (qdrant_client,
sentence_transformers, streamlit) when running tests in local environments.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy/cloud libraries if not installed in local environment
for mod_name in [
    "qdrant_client",
    "qdrant_client.models",
    "sentence_transformers",
    "streamlit",
    "streamlit.components.v1",
]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()
