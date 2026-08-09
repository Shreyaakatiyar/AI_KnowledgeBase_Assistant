import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ingestion import recursive_split


def test_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = recursive_split(text, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splits_into_multiple_chunks():
    text = "Sentence one. " * 200  # ~2800 characters
    chunks = recursive_split(text, chunk_size=800, overlap=150)
    assert len(chunks) > 1


def test_consecutive_chunks_share_overlap():
    text = "AAAA. BBBB. CCCC. DDDD. " * 50
    chunks = recursive_split(text, chunk_size=100, overlap=20)
    if len(chunks) > 1:
        overlap_region = chunks[0][-20:]
        assert overlap_region in chunks[1]


def test_empty_text_returns_no_chunks():
    chunks = recursive_split("", chunk_size=800, overlap=150)
    assert chunks == []


def test_whitespace_only_text_returns_no_chunks():
    chunks = recursive_split("   \n\n   ", chunk_size=800, overlap=150)
    assert chunks == []