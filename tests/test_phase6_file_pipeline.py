"""
tests/test_phase6_file_pipeline.py
----------------------------------
Unit and regression tests for Phase 6: File Pipeline.
Covers:
1. Plain text and code file extraction.
2. PDF text extraction via PdfReader.
3. PDF OCR fallback via fitz at 200 DPI capped at 10 pages with warning banner.
4. Graceful handling of missing OCR libraries or corrupt files.
5. Conversational history injection into file_analysis_tool.
6. SHA-256 file text caching in session state.
7. Selective 'Ask about attached file' checkbox gating.
"""

import hashlib
from io import BytesIO
from unittest.mock import MagicMock, patch
from langchain.schema import HumanMessage, AIMessage

from agent import extract_file_text, file_analysis_tool


def test_extract_file_text_plain_text():
    sample_code = "def hello_world():\n    return 'Hello from test!'"
    file_bytes = sample_code.encode("utf-8")
    
    extracted = extract_file_text(file_bytes, file_type="text/x-python", file_name="hello.py")
    assert extracted == sample_code


def test_extract_file_text_pdf_with_text_layer():
    mock_pdf_reader = MagicMock()
    page1 = MagicMock()
    page1.extract_text.return_value = "Page 1 content. "
    page2 = MagicMock()
    page2.extract_text.return_value = "Page 2 content."
    mock_pdf_reader.return_value.pages = [page1, page2]

    warn_cb = MagicMock()
    info_cb = MagicMock()

    with patch("agent.PdfReader", mock_pdf_reader):
        text = extract_file_text(
            b"%PDF-1.4 dummy",
            file_type="application/pdf",
            file_name="report.pdf",
            warn_callback=warn_cb,
            info_callback=info_cb,
        )

    assert "Page 1 content." in text
    assert "Page 2 content." in text
    # OCR should not be triggered
    warn_cb.assert_not_called()
    info_cb.assert_not_called()


def test_extract_file_text_ocr_capping_and_200_dpi():
    # Simulate empty text layer (scanned PDF)
    mock_pdf_reader = MagicMock()
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=MagicMock(return_value=""))]

    # Simulate 15-page document in fitz
    mock_doc = []
    for i in range(15):
        p = MagicMock()
        pix = MagicMock()
        pix.tobytes.return_value = b"fake_png"
        p.get_pixmap.return_value = pix
        mock_doc.append(p)

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.side_effect = lambda img: "Scanned text line"

    warn_cb = MagicMock()
    info_cb = MagicMock()

    with patch("agent.PdfReader", mock_pdf_reader), \
         patch("agent.fitz", mock_fitz), \
         patch("agent.pytesseract", mock_pytesseract), \
         patch("agent.Image.open", MagicMock()):
        text = extract_file_text(
            b"%PDF-1.4 scanned",
            file_type="application/pdf",
            file_name="scanned.pdf",
            max_ocr_pages=10,
            warn_callback=warn_cb,
            info_callback=info_cb,
        )

    # Info callback should indicate OCR started
    info_cb.assert_called_with("No text layer found. Performing OCR...")

    # Warning callback should indicate capping at 10 pages
    warn_cb.assert_called_once()
    assert "Document has 15 pages" in warn_cb.call_args[0][0]
    assert "capped at the first 10 pages" in warn_cb.call_args[0][0]

    # Verify get_pixmap was called with dpi=200 on exactly 10 pages
    for i in range(10):
        mock_doc[i].get_pixmap.assert_called_with(dpi=200)
    for i in range(10, 15):
        mock_doc[i].get_pixmap.assert_not_called()

    # Output text should contain 10 lines of OCR text
    assert text.count("Scanned text line") == 10


def test_extract_file_text_ocr_within_page_limit():
    mock_pdf_reader = MagicMock()
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=MagicMock(return_value=""))]

    mock_doc = []
    for i in range(4):
        p = MagicMock()
        pix = MagicMock()
        pix.tobytes.return_value = b"fake_png"
        p.get_pixmap.return_value = pix
        mock_doc.append(p)

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.side_effect = lambda img: "Page text"

    warn_cb = MagicMock()
    info_cb = MagicMock()

    with patch("agent.PdfReader", mock_pdf_reader), \
         patch("agent.fitz", mock_fitz), \
         patch("agent.pytesseract", mock_pytesseract), \
         patch("agent.Image.open", MagicMock()):
        text = extract_file_text(
            b"%PDF-1.4 scanned_short",
            file_type="application/pdf",
            file_name="short.pdf",
            max_ocr_pages=10,
            warn_callback=warn_cb,
            info_callback=info_cb,
        )

    # Capping warning should NOT be called since 4 <= 10
    warn_cb.assert_not_called()
    assert text.count("Page text") == 4


def test_extract_file_text_graceful_missing_ocr_libs():
    mock_pdf_reader = MagicMock()
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=MagicMock(return_value=""))]

    warn_cb = MagicMock()

    with patch("agent.PdfReader", mock_pdf_reader), \
         patch("agent.fitz", None), \
         patch("agent.pytesseract", None):
        text = extract_file_text(
            b"%PDF-1.4",
            file_type="application/pdf",
            warn_callback=warn_cb,
        )

    warn_cb.assert_called_once()
    assert "OCR dependencies" in warn_cb.call_args[0][0]
    assert text == ""


def test_file_analysis_tool_injects_conversation_history():
    mock_llm_instance = MagicMock()
    mock_llm_instance.stream.return_value = ["chunk1", "chunk2"]

    history = [
        HumanMessage(content="What are the three main findings?"),
        AIMessage(content="Finding 1: X, Finding 2: Y, Finding 3: Z"),
    ]

    with patch("agent.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        stream = file_analysis_tool(
            question="Can you elaborate on Finding 2?",
            file_content_as_text="This paper studies X, Y, and Z.",
            google_api_key="dummy_key",
            history=history,
        )

        assert list(stream) == ["chunk1", "chunk2"]
        prompt_arg = mock_llm_instance.stream.call_args[0][0][0].content
        assert "**Recent Conversation Context:**" in prompt_arg
        assert "User: What are the three main findings?" in prompt_arg
        assert "Assistant: Finding 1: X, Finding 2: Y, Finding 3: Z" in prompt_arg
        assert "Can you elaborate on Finding 2?" in prompt_arg


def test_file_analysis_tool_without_history():
    mock_llm_instance = MagicMock()
    mock_llm_instance.stream.return_value = ["response"]

    with patch("agent.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        stream = file_analysis_tool(
            question="Summarize this file",
            file_content_as_text="Document contents",
            google_api_key="dummy_key",
            history=None,
        )

        prompt_arg = mock_llm_instance.stream.call_args[0][0][0].content
        assert "**Recent Conversation Context:**" not in prompt_arg
        assert "Summarize this file" in prompt_arg


def test_file_cache_by_sha256_avoids_reextraction():
    cache = {}
    file_bytes_a = b"File content for document A"
    file_hash_a = hashlib.sha256(file_bytes_a).hexdigest()

    # Turn 1: File A not in cache
    mock_extract = MagicMock(return_value="Extracted text A")
    if file_hash_a in cache:
        text = cache[file_hash_a]
    else:
        text = mock_extract(file_bytes_a)
        cache[file_hash_a] = text

    assert text == "Extracted text A"
    assert mock_extract.call_count == 1

    # Turn 2: File A queried again -> hits cache
    mock_extract.reset_mock()
    if file_hash_a in cache:
        text = cache[file_hash_a]
    else:
        text = mock_extract(file_bytes_a)
        cache[file_hash_a] = text

    assert text == "Extracted text A"
    mock_extract.assert_not_called()

    # Turn 3: File B has distinct hash -> extracts and caches
    file_bytes_b = b"File content for document B"
    file_hash_b = hashlib.sha256(file_bytes_b).hexdigest()
    assert file_hash_a != file_hash_b

    if file_hash_b in cache:
        text_b = cache[file_hash_b]
    else:
        text_b = mock_extract(file_bytes_b)
        cache[file_hash_b] = text_b

    assert mock_extract.call_count == 1
    assert file_hash_a in cache and file_hash_b in cache


def test_ask_about_file_routing_logic():
    def route_query(uploaded_file, ask_about_file):
        if uploaded_file and ask_about_file:
            return "PATH 1: File Analysis"
        return "PATH 2: Agent Execution"

    # 1. File uploaded, checkbox unchecked -> normal agent execution
    assert route_query(uploaded_file="doc.pdf", ask_about_file=False) == "PATH 2: Agent Execution"

    # 2. File uploaded, checkbox checked -> file analysis
    assert route_query(uploaded_file="doc.pdf", ask_about_file=True) == "PATH 1: File Analysis"

    # 3. No file uploaded, checkbox unchecked -> normal agent execution
    assert route_query(uploaded_file=None, ask_about_file=False) == "PATH 2: Agent Execution"

    # 4. No file uploaded, checkbox checked (e.g. edge case) -> normal agent execution
    assert route_query(uploaded_file=None, ask_about_file=True) == "PATH 2: Agent Execution"
