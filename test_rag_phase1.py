import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Source Code"))

from rag import chunk_pages


def test_chunk_pages_preserves_pages_and_overlap():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_pages([(1, " ".join(words))], chunk_size=200, overlap=50)
    assert len(chunks) == 2
    assert all(page == 1 for page, _ in chunks)
    assert chunks[0][1].split()[-1] == "w199"
    assert chunks[1][1].split()[0] == "w150"


def test_empty_page_is_skipped():
    assert chunk_pages([(1, "   "), (2, "hello world")], chunk_size=200) == [(2, "hello world")]
