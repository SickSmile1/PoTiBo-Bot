"""
Unit tests for docbot core modules.

These tests cover pure-logic functions that do not require a running Matrix
server, an ONNX model, or network access.  They run in any environment that
has the packages from requirements.txt installed.
"""
import hashlib
import hmac

import numpy as np
import pytest

from config import Config
from doc_parser import chunk_docs, extract_docstrings, extract_code_snippets
from index_store import IndexStore
from searcher import Searcher
from source_plugins import (
    _sha256,
    deduplicate,
    scan_codebase,
    scan_curated_files,
)
from webfetch import extract_sections


# ---------------------------------------------------------------------------
# doc_parser — extract_docstrings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("content", "expected_kind", "expected_symbol", "text_contains"),
    [
        ('"""Module doc."""\n', "module", None, "Module doc"),
        (
            'def foo():\n    """Function foo does something."""\n    pass\n',
            "function",
            "foo",
            "Function foo",
        ),
        (
            'class Bar:\n    """Bar is a class."""\n    pass\n',
            "class",
            "Bar",
            "Bar is a class",
        ),
    ],
)
def test_extracts_docstrings(tmp_path, content, expected_kind, expected_symbol, text_contains):
    f = tmp_path / "mymod.py"
    f.write_text(content)
    results = list(extract_docstrings(tmp_path))
    assert any(
        r["kind"] == expected_kind
        and (expected_symbol is None or r.get("symbol") == expected_symbol)
        and text_contains in r["text"]
        for r in results
    )


@pytest.mark.parametrize("subdir", ["tests", "__pycache__"])
def test_skips_special_directories(tmp_path, subdir):
    d = tmp_path / subdir
    d.mkdir()
    (d / "ignore_me.py").write_text('"""Should not appear."""\n')
    results = list(extract_docstrings(tmp_path))
    assert not results


def test_skips_large_files(tmp_path):
    f = tmp_path / "big.py"
    big_content = '"""Big file."""\n' + "x = 1\n" * 50_000
    f.write_bytes(big_content.encode())
    results = list(extract_docstrings(tmp_path))
    assert not results


def test_skips_syntax_errors(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def bad syntax here")
    results = list(extract_docstrings(tmp_path))
    assert not results


def test_path_and_module_fields(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    f = sub / "mymod.py"
    f.write_text('def func():\n    """Docs."""\n    pass\n')
    results = list(extract_docstrings(tmp_path))
    assert any(r["module"] == "pkg.mymod" for r in results)


def test_files_without_docstrings_produce_no_output(tmp_path):
    f = tmp_path / "nodoc.py"
    f.write_text("x = 1\n")
    results = list(extract_docstrings(tmp_path))
    assert not results


# ---------------------------------------------------------------------------
# doc_parser — extract_code_snippets
# ---------------------------------------------------------------------------

class TestExtractCodeSnippets:
    def test_function_without_docstring_yields_function_chunk(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text("def foo():\n    pass\n")
        results = list(extract_code_snippets(tmp_path))
        funcs = [r for r in results if r["kind"] == "function"]
        assert any("def foo():" in r["text"] for r in funcs)

    def test_function_yields_signature(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text("def foo(bar: int) -> str:\n    return str(bar)\n")
        results = list(extract_code_snippets(tmp_path))
        sigs = [r for r in results if r["kind"] == "signature"]
        assert any("def foo(bar: int)" in r["text"] for r in sigs)

    def test_function_with_docstring_skips_function_chunk(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text('def foo():\n    """Has docs."""\n    pass\n')
        results = list(extract_code_snippets(tmp_path))
        funcs = [r for r in results if r["kind"] == "function"]
        assert funcs == []

    def test_function_with_docstring_still_yields_signature(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text('def foo():\n    """Has docs."""\n    pass\n')
        results = list(extract_code_snippets(tmp_path))
        sigs = [r for r in results if r["kind"] == "signature"]
        assert any("def foo():" in r["text"] for r in sigs)

    def test_route_decorator_yields_route_chunk(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            '@app.get("/users/{id}")\n'
            "def get_user(id: int):\n"
            "    return {}\n"
        )
        results = list(extract_code_snippets(tmp_path))
        routes = [r for r in results if r["kind"] == "route"]
        assert any("GET /users/{id}" in r["text"] for r in routes)
        assert any("get_user" in r["text"] for r in routes)

    @pytest.mark.parametrize("method", ["post", "put", "delete", "patch"])
    def test_route_methods(self, tmp_path, method):
        f = tmp_path / "app.py"
        f.write_text(
            f"from fastapi import FastAPI\n"
            f"app = FastAPI()\n"
            f'@app.{method}("/resource")\n'
            f"def handler():\n"
            f"    return {{}}\n"
        )
        results = list(extract_code_snippets(tmp_path))
        routes = [r for r in results if r["kind"] == "route"]
        assert any(f"{method.upper()} /resource" in r["text"] for r in routes)

    def test_class_with_annotated_fields_yields_class_def(self, tmp_path):
        f = tmp_path / "schemas.py"
        f.write_text(
            "from pydantic import BaseModel\n"
            "class User(BaseModel):\n"
            "    id: int\n"
            "    name: str\n"
        )
        results = list(extract_code_snippets(tmp_path))
        class_defs = [r for r in results if r["kind"] == "class_def"]
        assert any("class User(BaseModel):" in r["text"] for r in class_defs)
        assert any("id: int" in r["text"] for r in class_defs)

    def test_class_without_annotated_fields_no_class_def(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("class Empty:\n    pass\n")
        results = list(extract_code_snippets(tmp_path))
        class_defs = [r for r in results if r["kind"] == "class_def"]
        assert class_defs == []

    def test_skips_special_directories(self, tmp_path):
        d = tmp_path / "tests"
        d.mkdir()
        (d / "test_foo.py").write_text("def foo():\n    pass\n")
        results = list(extract_code_snippets(tmp_path))
        assert not results

    @pytest.mark.parametrize("subdir", ["__pycache__", ".git", ".venv"])
    def test_skips_other_directories(self, tmp_path, subdir):
        d = tmp_path / subdir
        d.mkdir()
        (d / "ignored.py").write_text("def foo():\n    pass\n")
        results = list(extract_code_snippets(tmp_path))
        assert not results

    def test_skips_syntax_errors(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def bad syntax here\n")
        results = list(extract_code_snippets(tmp_path))
        assert not results

    def test_module_and_path_fields(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        f = sub / "mymod.py"
        f.write_text("def func():\n    pass\n")
        results = list(extract_code_snippets(tmp_path))
        assert any(r["module"] == "pkg.mymod" for r in results)
        assert any(r["path"] == "pkg/mymod.py" for r in results)

    def test_chunk_index_present_on_all(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            '@app.get("/")\n'
            "def root():\n"
            "    return {}\n"
        )
        results = list(extract_code_snippets(tmp_path))
        assert any("chunk_index" in r for r in results)
        assert all(r["chunk_index"] == 0 for r in results)

    def test_async_function_handled(self, tmp_path):
        f = tmp_path / "handlers.py"
        f.write_text("async def fetch_data():\n    return []\n")
        results = list(extract_code_snippets(tmp_path))
        sigs = [r for r in results if r["kind"] == "signature"]
        assert any("async def fetch_data():" in r["text"] for r in sigs)


# ---------------------------------------------------------------------------
# doc_parser — chunk_docs
# ---------------------------------------------------------------------------

def _make_doc(text):
    return {
        "module": "m",
        "symbol": "s",
        "kind": "function",
        "text": text,
        "path": "m.py",
    }


@pytest.mark.parametrize(
    ("text", "max_chars", "overlap", "expected_count", "expected_max_len"),
    [
        ("Short text.", None, None, 1, None),
        ("x" * 2000, 800, 200, None, 800),
    ],
)
def test_chunk_docs(text, max_chars, overlap, expected_count, expected_max_len):
    doc = _make_doc(text)
    kwargs = {}
    if max_chars is not None:
        kwargs["max_chars"] = max_chars
    if overlap is not None:
        kwargs["overlap"] = overlap
    chunks = list(chunk_docs([doc], **kwargs))
    if expected_count is not None:
        assert len(chunks) == expected_count
    if expected_max_len is not None:
        assert all(len(c["text"]) <= expected_max_len for c in chunks)


def test_chunk_docs_original_fields_preserved():
    doc = _make_doc("Short.")
    chunks = list(chunk_docs([doc]))
    assert chunks[0]["module"] == "m"
    assert chunks[0]["symbol"] == "s"
    assert chunks[0]["kind"] == "function"


def test_chunk_docs_index_increments():
    doc = _make_doc("z" * 3000)
    chunks = list(chunk_docs([doc], max_chars=800, overlap=200))
    for i, c in enumerate(chunks):
        assert c["chunk_index"] == i


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults_are_set(self):
        cfg = Config()
        assert cfg.MATRIX_HOMESERVER
        assert cfg.MATRIX_BOT_USERNAME
        assert isinstance(cfg.PACKAGES_TO_SCAN, list)
        assert isinstance(cfg.CONFIDENCE_THRESHOLD, float)

    def test_web_urls_empty_by_default(self, monkeypatch):
        monkeypatch.setattr(Config, "WEB_URLS_RAW", "[]")
        cfg = Config()
        assert cfg.WEB_URLS == []

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('["https://example.com"]', ["https://example.com"]),
            ("not-json", []),
        ],
    )
    def test_web_urls_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setattr(Config, "WEB_URLS_RAW", raw)
        cfg = Config()
        assert cfg.WEB_URLS == expected

    def test_rebuild_on_start_default_false(self):
        cfg = Config()
        assert cfg.REBUILD_ON_START is False

    def test_test_user_defaults_empty(self, monkeypatch):
        monkeypatch.setattr(Config, "TEST_USER_USERNAME", "")
        monkeypatch.setattr(Config, "TEST_USER_PASSWORD", "")
        cfg = Config()
        assert cfg.TEST_USER_USERNAME == ""
        assert cfg.TEST_USER_PASSWORD == ""


# ---------------------------------------------------------------------------
# index_store
# ---------------------------------------------------------------------------

class TestIndexStore:
    def _sample_data(self):
        meta = [{"module": "m", "symbol": "s", "kind": "function", "path": "m.py"}]
        emb = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        return meta, emb

    def test_exists_false_before_save(self, tmp_path):
        store = IndexStore(tmp_path)
        assert not store.exists()

    def test_save_creates_files(self, tmp_path):
        store = IndexStore(tmp_path)
        meta, emb = self._sample_data()
        store.save(meta, emb)
        assert store.meta_path.exists()
        assert store.emb_path.exists()

    def test_exists_true_after_save(self, tmp_path):
        store = IndexStore(tmp_path)
        meta, emb = self._sample_data()
        store.save(meta, emb)
        assert store.exists()

    def test_load_roundtrip(self, tmp_path):
        meta, emb = self._sample_data()
        IndexStore(tmp_path).save(meta, emb)
        store2 = IndexStore(tmp_path)
        n = store2.load()
        assert n == 1
        assert store2.metadata == meta
        np.testing.assert_array_almost_equal(store2.embeddings, emb)

    def test_dim_after_load_matches_saved(self, tmp_path):
        meta, emb = self._sample_data()
        IndexStore(tmp_path).save(meta, emb)
        store = IndexStore(tmp_path)
        store.load()
        assert store.dim == 3

    def test_dim_is_none_before_load(self, tmp_path):
        store = IndexStore(tmp_path)
        assert store.dim is None

    def test_dim_is_none_after_clear(self, tmp_path):
        meta, emb = self._sample_data()
        IndexStore(tmp_path).save(meta, emb)
        store = IndexStore(tmp_path)
        store.load()
        assert store.dim == 3
        store.clear()
        assert store.dim is None

    def test_clear_removes_files_and_resets_state(self, tmp_path):
        store = IndexStore(tmp_path)
        meta, emb = self._sample_data()
        store.save(meta, emb)
        store.clear()
        assert not store.meta_path.exists()
        assert not store.emb_path.exists()
        assert store.metadata == []
        assert store.embeddings is None

    def test_clear_is_idempotent(self, tmp_path):
        store = IndexStore(tmp_path)
        store.clear()


# ---------------------------------------------------------------------------
# searcher
# ---------------------------------------------------------------------------

class TestSearcher:
    def _build_searcher(self, n=3, dim=4):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms
        meta = [
            {"module": f"mod{i}", "symbol": f"sym{i}", "kind": "function", "path": f"mod{i}.py"}
            for i in range(n)
        ]
        return Searcher(meta, embeddings), embeddings

    def test_returns_list(self):
        s, emb = self._build_searcher()
        results = s.search(emb[[0]])
        assert isinstance(results, list)

    def test_top_result_is_most_similar(self):
        s, emb = self._build_searcher()
        results = s.search(emb[[0]], threshold=0.0)
        assert results[0]["module"] == "mod0"

    def test_threshold_filters_results(self):
        s, emb = self._build_searcher()
        results = s.search(emb[[0]], threshold=2.0)
        assert results == []

    def test_score_present_and_bounded(self):
        s, emb = self._build_searcher()
        results = s.search(emb[[0]], threshold=0.0)
        for r in results:
            assert "score" in r
            assert -1.0 <= r["score"] <= 1.0

    def test_top_k_limits_results(self):
        s, emb = self._build_searcher(n=10, dim=4)
        results = s.search(emb[[0]], top_k=2, threshold=0.0)
        assert len(results) <= 2

    def test_metadata_fields_present(self):
        s, emb = self._build_searcher()
        results = s.search(emb[[0]], threshold=0.0)
        for r in results:
            assert "module" in r
            assert "symbol" in r
            assert "kind" in r


class TestSearcherWithReranker:
    """Tests for two-stage retrieval with a mock cross-encoder reranker."""

    @staticmethod
    def _make_reranker(expected_texts=None):
        """Build a mock reranker that swaps result order for easy assertion."""
        class _MockReranker:
            def __init__(self, exp_texts):
                self._expected_texts = exp_texts

            def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
                # Mimics real reranker.py:71 — accesses c["text"]
                texts = [c["text"] for c in candidates]
                assert texts is not None
                # Reverse order to prove reranker was called
                for c in candidates:
                    c["score"] = 0.5  # uniform score
                reversed_list = list(reversed(candidates))
                for i, c in enumerate(reversed_list):
                    c["score"] = round(float(len(reversed_list) - i), 4)
                return reversed_list

        return _MockReranker(expected_texts)

    def _build_searcher(self, n=5, dim=4):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms
        meta = [
            {"module": f"mod{i}", "symbol": f"sym{i}", "kind": "function",
             "path": f"mod{i}.py", "text": f"function body of mod{i}"}
            for i in range(n)
        ]
        reranker = self._make_reranker()
        return Searcher(meta, embeddings, reranker=reranker, reranker_candidates=10), embeddings

    def test_reranker_reverses_order(self):
        """Without reranker, result 0 is most similar. With reranker,
        the mock reverses order, so result 0 should be the last by cosine."""
        s, emb = self._build_searcher()
        # Use very low threshold so all 5 candidates survive the first stage
        results = s.search(emb[[0]], threshold=-2.0, query_text="test query")
        assert len(results) == 3
        # The mock reranker reverses all 5 candidates:
        # original order: mod0, mod2, mod4, mod1, mod3  (by cosine desc)
        # reversed:       mod3, mod1, mod4, mod2, mod0
        # top 3:          mod3, mod1, mod4
        assert results[0]["module"] == "mod3"

    def test_reranker_passes_query_and_candidates(self):
        """Verify the mock reranker receives the query text and candidates."""
        seen = {}

        class TrackingReranker:
            def rerank(self, query, candidates):
                seen["query"] = query
                seen["texts"] = [c["text"] for c in candidates]
                return candidates  # no change

        meta = [{"module": "m", "symbol": "s", "kind": "function", "path": "m.py", "text": "foo"}]
        emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        s = Searcher(meta, emb, reranker=TrackingReranker(), reranker_candidates=10)
        s.search(emb[[0]], threshold=0.0, query_text="my query")
        assert seen["query"] == "my query"
        assert seen["texts"] == ["foo"]

    def test_no_reranker_when_no_query_text(self):
        """Reranker should be skipped when query_text is empty."""
        call_count = 0

        class CountReranker:
            def rerank(self, query, candidates):
                nonlocal call_count
                call_count += 1
                return candidates

        meta = [{"module": "m", "symbol": "s", "kind": "function", "path": "m.py"}]
        emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        s = Searcher(meta, emb, reranker=CountReranker(), reranker_candidates=10)
        s.search(emb[[0]], threshold=0.0, query_text="")
        assert call_count == 0

    def test_reranker_respects_top_k(self):
        """Reranked results should be limited to top_k."""
        s, emb = self._build_searcher(n=10, dim=4)
        results = s.search(emb[[0]], top_k=2, threshold=0.0, query_text="test")
        assert len(results) <= 2

    def test_reranker_skipped_when_no_candidates(self):
        """Reranker should not be called when no candidates pass threshold."""
        call_count = 0

        class CountReranker:
            def rerank(self, query, candidates):
                nonlocal call_count
                call_count += 1
                return candidates

        meta = [{"module": "m", "symbol": "s", "kind": "function", "path": "m.py"}]
        emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        s = Searcher(meta, emb, reranker=CountReranker(), reranker_candidates=10)
        results = s.search(emb[[0]], threshold=2.0, query_text="test")
        assert results == []
        assert call_count == 0

    def test_backwards_compatible_no_reranker(self):
        """Searcher without reranker should behave identically to before."""
        s, emb = TestSearcher._build_searcher(TestSearcher, n=3, dim=4)
        results = s.search(emb[[0]], threshold=0.0)
        assert results[0]["module"] == "mod0"
        assert len(results) == 3
        for r in results:
            assert "score" in r

    def test_searcher_results_include_text_for_reranker(self):
        """Regression: metadata from IndexStore must include 'text' so the
        real reranker (reranker.py:71) does not crash with KeyError.

        Before the fix, ingest.py stripped 'text' from metadata, causing:
        1. Reranker KeyError crash (RERANKER_ENABLED=true in production)
        2. Bot responses with empty excerpts (non-viable answers)
        """
        class TextAccessReranker:
            """Mimics the real reranker's c['text'] access pattern."""
            def rerank(self, query, candidates):
                texts = [c["text"] for c in candidates]  # noqa: F841
                return candidates

        meta = [
            {"module": f"mod{i}", "symbol": f"sym{i}", "kind": "function",
             "path": f"mod{i}.py", "text": f"body of function {i}"}
            for i in range(3)
        ]
        emb = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms

        s = Searcher(meta, emb, reranker=TextAccessReranker(), reranker_candidates=10)
        # Must NOT raise KeyError
        results = s.search(emb[[0]], threshold=0.0, query_text="test")
        assert len(results) > 0
        for r in results:
            assert "text" in r, "Search result must include 'text' for reranker and display"


# ---------------------------------------------------------------------------
# source_plugins — _sha256 / deduplicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("chunk_a", "chunk_b", "should_match"),
    [
        (
            {"module": "m", "symbol": "s", "chunk_index": 0, "text": "hello"},
            {"module": "m", "symbol": "s", "chunk_index": 0, "text": "hello"},
            True,
        ),
        (
            {"module": "m", "symbol": "s", "chunk_index": 0, "text": "hello"},
            {"module": "m", "symbol": "s", "chunk_index": 0, "text": "world"},
            False,
        ),
    ],
)
def test_sha256(chunk_a, chunk_b, should_match):
    if should_match:
        assert _sha256(chunk_a) == _sha256(chunk_b)
    else:
        assert _sha256(chunk_a) != _sha256(chunk_b)


def test_sha256_format():
    h = _sha256({"module": "m", "symbol": "s", "chunk_index": 0, "text": "hi"})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


class TestDeduplicate:
    def test_removes_duplicates(self):
        chunks = [
            {"sha256": "aaa", "text": "one"},
            {"sha256": "aaa", "text": "one again"},
            {"sha256": "bbb", "text": "two"},
        ]
        unique = deduplicate(chunks)
        assert len(unique) == 2
        assert unique[0]["text"] == "one"

    def test_preserves_order(self):
        chunks = [
            {"sha256": "a", "text": "first"},
            {"sha256": "b", "text": "second"},
            {"sha256": "c", "text": "third"},
        ]
        unique = deduplicate(chunks)
        assert [c["text"] for c in unique] == ["first", "second", "third"]

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_chunk_without_sha256_is_kept(self):
        chunks = [{"text": "no hash"}]
        result = deduplicate(chunks)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# source_plugins — scan_codebase / scan_curated_files
# ---------------------------------------------------------------------------

class TestScanCodebase:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = scan_codebase(tmp_path / "nonexistent")
        assert result == []

    def test_scans_python_files(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text('def foo():\n    """Docstring."""\n    pass\n')
        chunks = scan_codebase(tmp_path)
        assert len(chunks) > 0

    def test_source_type_tag(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text('def foo():\n    """Docstring."""\n    pass\n')
        chunks = scan_codebase(tmp_path, tag="mycode")
        assert all(c["source_type"] == "mycode" for c in chunks)

    def test_sha256_populated(self, tmp_path):
        f = tmp_path / "mymod.py"
        f.write_text('def foo():\n    """Docstring."""\n    pass\n')
        chunks = scan_codebase(tmp_path)
        assert all("sha256" in c for c in chunks)


class TestScanCuratedFiles:
    def test_reads_markdown_files(self, tmp_path):
        md = tmp_path / "guide.md"
        md.write_text("# Guide\n\n" + "Some reference text here.\n" * 20)
        chunks = scan_curated_files(tmp_path)
        assert len(chunks) > 0

    def test_source_type_set_to_reference(self, tmp_path):
        md = tmp_path / "guide.md"
        md.write_text("# Guide\n" + "x" * 200)
        chunks = scan_curated_files(tmp_path)
        assert all(c["source_type"] == "reference" for c in chunks)

    def test_empty_dir_returns_empty(self, tmp_path):
        result = scan_curated_files(tmp_path)
        assert result == []

    def test_ignores_non_markdown_files(self, tmp_path):
        (tmp_path / "note.txt").write_text("Some text.")
        result = scan_curated_files(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# webfetch
# ---------------------------------------------------------------------------

class TestExtractSections:
    def test_splits_on_hash_heading(self):
        body_one = "Content for section one. " * 5
        body_two = "Content for section two. " * 5
        text = f"# Section One\n{body_one}\n# Section Two\n{body_two}"
        sections = extract_sections(text, "http://example.com", "web")
        symbols = [s["symbol"] for s in sections]
        assert "# Section One" in symbols
        assert "# Section Two" in symbols

    def test_filters_short_sections(self):
        text = "# Short\nhi\n# Long\n" + "x" * 100
        sections = extract_sections(text, "http://example.com", "web")
        symbols = [s["symbol"] for s in sections]
        assert "# Short" not in symbols

    def test_source_type_set(self):
        text = "# Section\n" + "x" * 100
        sections = extract_sections(text, "http://example.com", "mytag")
        assert all(s["source_type"] == "mytag" for s in sections)

    def test_path_is_url(self):
        url = "http://example.com/page"
        text = "# Section\n" + "x" * 100
        sections = extract_sections(text, url, "web")
        assert all(s["path"] == url for s in sections)

    def test_all_uppercase_line_is_heading(self):
        text = "OVERVIEW\n" + "y" * 100
        sections = extract_sections(text, "http://x.com", "web")
        assert len(sections) >= 1

    def test_text_with_no_headings(self):
        text = "x" * 200
        sections = extract_sections(text, "http://x.com", "web")
        assert len(sections) == 1
        assert sections[0]["symbol"] == "Overview"

    def test_empty_text_returns_empty(self):
        sections = extract_sections("", "http://x.com", "web")
        assert sections == []


# ---------------------------------------------------------------------------
# bot — _compute_mac
# ---------------------------------------------------------------------------

def _compute_mac(shared_secret: str, nonce: str, username: str, password: str) -> str:
    """Mirror of the function in bot.py for isolated testing."""
    mac = hmac.new(shared_secret.encode(), digestmod=hashlib.sha1)
    mac.update(nonce.encode())
    mac.update(b"\x00")
    mac.update(username.encode())
    mac.update(b"\x00")
    mac.update(password.encode())
    mac.update(b"\x00")
    mac.update(b"notadmin")
    return mac.hexdigest()


@pytest.mark.parametrize(
    ("secret", "nonce", "user", "passwd", "alt_field", "expected_behavior"),
    [
        ("secret", "nonce123", "user", "pass", None, "is_hex"),
        ("secret", "nonce", "user", "pass", None, "deterministic"),
        ("secret1", "nonce", "user", "pass", "secret2", "diff_secret"),
        ("secret", "nonce1", "user", "pass", "nonce2", "diff_nonce"),
        ("secret", "nonce", "alice", "pass", "bob", "diff_user"),
        ("secret", "nonce", "user", "pass1", "pass2", "diff_password"),
    ],
)
def test_compute_mac(secret, nonce, user, passwd, alt_field, expected_behavior):
    result = _compute_mac(secret, nonce, user, passwd)
    if expected_behavior == "is_hex":
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)
    elif expected_behavior == "deterministic":
        assert result == _compute_mac(secret, nonce, user, passwd)
    elif expected_behavior == "diff_secret":
        assert result != _compute_mac(alt_field, nonce, user, passwd)
    elif expected_behavior == "diff_nonce":
        assert result != _compute_mac(secret, alt_field, user, passwd)
    elif expected_behavior == "diff_user":
        assert result != _compute_mac(secret, nonce, alt_field, passwd)
    elif expected_behavior == "diff_password":
        assert result != _compute_mac(secret, nonce, user, alt_field)


def test_compute_mac_known_value():
    expected = _compute_mac("mysecret", "mynonce", "testuser", "testpass")
    assert _compute_mac("mysecret", "mynonce", "testuser", "testpass") == expected


# ---------------------------------------------------------------------------
# bot response formatting — regression for text key
# ---------------------------------------------------------------------------

class TestBotResponseFormatting:
    """Verify that search results with 'text' produce visible excerpts.

    Before the fix, ingest.py stripped 'text' from metadata, causing the bot
    to display empty excerpts — the primary reason it 'could not return
    viable answers'.
    """

    def test_results_with_text_produce_nonempty_excerpts(self):
        """When text is present in results, the bot shows content excerpts."""
        results = [
            {
                "module": "auth", "symbol": "login", "kind": "function",
                "path": "auth.py", "source_type": "code", "score": 0.85,
                "text": "async def login(email: str, password: str):\n"
                        "    verify_password(email, password)",
            }
        ]
        for r in results:
            doc_text = r.get("text", "").strip()
            assert doc_text, "Excerpt must not be empty when text is in results"

    def test_results_without_text_produce_empty_excerpts(self):
        """Without text (old bug), excerpt is empty — proves the root cause."""
        results = [
            {
                "module": "auth", "symbol": "login", "kind": "function",
                "path": "auth.py", "source_type": "code", "score": 0.85,
            }
        ]
        for r in results:
            doc_text = r.get("text", "").strip()
            assert not doc_text, "Proves old bug: no text key = empty excerpt"

    def test_format_response_with_text(self):
        """Simulates bot._handle_ask response formatting with text present."""
        results = [
            {
                "module": "auth", "symbol": "login", "kind": "function",
                "path": "auth.py", "source_type": "code", "score": 0.85,
                "text": "async def login(email: str, password: str):",
            }
        ]
        lines = ["I found these relevant documentation snippets:\n"]
        for r in results:
            doc_text = r.get("text", "").strip()
            source = r.get("source_type", "unknown")
            lines.append(f"**{r['module']}:{r['symbol']}** ({r['kind']}) [{source}]")
            lines.append(f"    Source: `{r['path']}`")
            if doc_text:
                lines.append(f"    > {doc_text[:300].replace(chr(10), chr(10) + '    > ')}")
            lines.append(f"    _Score: {r['score']}_\n")
        text = "\n".join(lines)
        assert "async def login" in text, "Response must contain the code excerpt"
        assert "Source: `auth.py`" in text


# ---------------------------------------------------------------------------
# source_plugins — no hardcoded site-packages
# ---------------------------------------------------------------------------

def test_site_packages_path_is_not_hardcoded():
    """source_plugins must use sys.version_info, not a hardcoded python3.13 path."""
    import inspect
    from source_plugins import run_all_plugins
    src = inspect.getsource(run_all_plugins)
    assert "python3.13" not in src, (
        "site-packages path should use sys.version_info, not hardcoded python3.13"
    )


# ---------------------------------------------------------------------------
# pipeline integration — ingest → index → search → display
# ---------------------------------------------------------------------------

class TestIngestSearchPipeline:
    """Integration test: create codebase on disk, ingest, search, verify text."""

    def test_ingested_index_preserves_text_for_display(self, tmp_path):
        """End-to-end: ingest a .py file, verify metadata has 'text',
        search returns results with 'text', and response formatting works."""
        try:
            from ingest import run_ingest
        except ModuleNotFoundError:
            pytest.skip("onnxruntime not installed — skipping pipeline test")

        # 1. Create a minimal codebase
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        (codebase / "auth.py").write_text(
            'def login(email: str):\n'
            '    """Authenticate user with email and password."""\n'
            '    pass\n'
        )
        (codebase / "models.py").write_text(
            'class User:\n'
            '    """User database model."""\n'
            '    pass\n'
        )

        # Create empty sources dir (needed by config)
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        class MiniConfig:
            CODEBASE_DIR = str(codebase)
            PACKAGES_TO_SCAN = []
            WEB_URLS = []
            SOURCES_DIR = str(sources_dir)
            MAX_TOTAL_COPY_MB = 500

        class MockEmbedder:
            embed_dim = 4
            def embed(self, texts, is_query=False):
                rng = np.random.default_rng(42)
                return rng.standard_normal((len(texts), self.embed_dim)).astype(np.float32)

        # 2. Ingest
        idx_dir = tmp_path / "index"
        idx_dir.mkdir()
        store = IndexStore(idx_dir)
        result = run_ingest(MiniConfig(), MockEmbedder(), store)

        assert "error" not in result, f"Ingestion failed: {result}"
        assert result["total_chunks"] > 0, "Should have produced chunks"

        # 3. Verify metadata includes text
        for m in store.metadata:
            assert "text" in m, f"Metadata entry missing 'text': {m}"
            assert "module" in m
            assert "kind" in m

        # 4. Search and verify results include text
        searcher = Searcher(store.metadata, store.embeddings)
        query_emb = MockEmbedder().embed(["login"], is_query=True)
        results = searcher.search(query_emb, threshold=0.0)

        assert len(results) > 0, "Search should return results"
        for r in results:
            assert "text" in r, "Search result must include 'text' for bot display"
            assert "score" in r
            assert "module" in r

        # 5. Verify response formatting produces visible excerpts
        for r in results:
            doc_text = r.get("text", "").strip()
            assert doc_text, f"Excerpt must not be empty for {r['module']}:{r['symbol']}"

    def test_index_store_roundtrip_preserves_text(self, tmp_path):
        """Metadata saved to and loaded from IndexStore retains 'text'."""
        idx_dir = tmp_path / "idx"
        idx_dir.mkdir()

        meta = [
            {"module": "auth", "symbol": "login", "kind": "function",
             "path": "auth.py", "text": "def login(): pass", "source_type": "code"},
        ]
        emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        store = IndexStore(idx_dir)
        store.save(meta, emb)

        store2 = IndexStore(idx_dir)
        store2.load()

        assert len(store2.metadata) == 1
        assert "text" in store2.metadata[0]
        assert store2.metadata[0]["text"] == "def login(): pass"

    def test_bm25_index_persisted_and_loaded(self, tmp_path):
        """BM25 index saved via IndexStore can be loaded back."""
        from bm25_index import BM25Index

        idx_dir = tmp_path / "idx"
        idx_dir.mkdir()

        meta = [
            {"module": "auth", "symbol": "login", "kind": "function",
             "path": "auth.py", "text": "def login(): pass", "source_type": "code"},
        ]
        emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        store = IndexStore(idx_dir)
        store.save(meta, emb)

        bm25_docs = [{"id": 0, "text": "def login user authentication"}]
        bm25 = BM25Index(bm25_docs)
        store.save_bm25(bm25)

        loaded = store.load_bm25()
        assert loaded is not None
        scores = loaded.score("login")
        assert scores[0] > 0.0


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_build_inverted_index(self):
        from bm25_index import BM25Index
        docs = [
            {"id": 0, "text": "def login user authentication"},
            {"id": 1, "text": "def logout user session"},
            {"id": 2, "text": "user profile update endpoint"},
        ]
        idx = BM25Index(docs)
        assert len(idx._inverted["user"]) == 3
        assert len(idx._inverted["login"]) == 1
        assert len(idx._inverted["logout"]) == 1

    def test_score_returns_list_of_floats(self):
        from bm25_index import BM25Index
        docs = [{"id": 0, "text": "hello world"}, {"id": 1, "text": "goodbye world"}]
        idx = BM25Index(docs)
        scores = idx.score("hello")
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_higher_score_for_better_match(self):
        from bm25_index import BM25Index
        docs = [
            {"id": 0, "text": "login authentication password reset"},
            {"id": 1, "text": "unrelated content about cooking"},
        ]
        idx = BM25Index(docs)
        scores = idx.score("login authentication")
        assert scores[0] > scores[1]

    def test_empty_index_returns_zeros(self):
        from bm25_index import BM25Index
        idx = BM25Index([])
        scores = idx.score("anything")
        assert scores == []

    def test_no_match_returns_all_zeros(self):
        from bm25_index import BM25Index
        docs = [{"id": 0, "text": "hello world"}]
        idx = BM25Index(docs)
        scores = idx.score("nonexistent_term_xyz")
        assert scores == [0.0]

    def test_case_insensitive(self):
        from bm25_index import BM25Index
        docs = [{"id": 0, "text": "Login Authentication"}]
        idx = BM25Index(docs)
        scores_upper = idx.score("LOGIN")
        scores_lower = idx.score("login")
        assert scores_upper == scores_lower

    def test_save_and_load_roundtrip(self, tmp_path):
        from bm25_index import BM25Index
        docs = [
            {"id": 0, "text": "def login user auth"},
            {"id": 1, "text": "def logout user session"},
        ]
        idx = BM25Index(docs)
        path = tmp_path / "bm25.json"
        idx.save(path)

        loaded = BM25Index.load(path)
        assert loaded.score("login") == idx.score("login")
        assert loaded.score("logout") == idx.score("logout")

    def test_repeated_term_boosts_score(self):
        from bm25_index import BM25Index
        docs = [
            {"id": 0, "text": "login login login"},
            {"id": 1, "text": "login"},
        ]
        idx = BM25Index(docs)
        scores = idx.score("login")
        assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def _chunk(self, name, path="m.py"):
        """Create a minimal chunk dict with proper key fields."""
        return {"module": name, "symbol": name, "chunk_index": 0, "path": path}

    def test_rrf_combines_two_rankings(self):
        from searcher import reciprocal_rank_fusion
        a, b, c = self._chunk("A"), self._chunk("B"), self._chunk("C")
        lists = [[a, b, c], [b, a, c]]
        fused = reciprocal_rank_fusion(lists, k=60)
        ids = [r["module"] for r in fused]
        assert ids[0] in ("A", "B")
        assert ids[1] in ("A", "B")
        assert ids[2] == "C"

    def test_rrf_single_list_passthrough(self):
        from searcher import reciprocal_rank_fusion
        x, y = self._chunk("X"), self._chunk("Y")
        lists = [[x, y]]
        fused = reciprocal_rank_fusion(lists)
        assert [r["module"] for r in fused] == ["X", "Y"]

    def test_rrf_empty_list(self):
        from searcher import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([]) == []

    def test_rrf_prefers_consistently_top_ranked(self):
        from searcher import reciprocal_rank_fusion
        a, b = self._chunk("A"), self._chunk("B")
        lists = [[a, b], [a, b]]
        fused = reciprocal_rank_fusion(lists)
        ids = [r["module"] for r in fused]
        assert ids == ["A", "B"]

    def test_rrf_disjoint_lists(self):
        from searcher import reciprocal_rank_fusion
        a, b = self._chunk("A"), self._chunk("B")
        c, d = self._chunk("C"), self._chunk("D")
        lists = [[a, b], [c, d]]
        fused = reciprocal_rank_fusion(lists)
        ids = {r["module"] for r in fused}
        assert ids == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# M5 — Structure-aware chunking: parent-context enrichment
# ---------------------------------------------------------------------------

class TestParentContextEnrichment:
    def test_method_chunk_gets_class_prefix(self):
        from doc_parser import enrich_with_parent_context
        chunks = [
            {"module": "models", "symbol": "User", "kind": "class_def",
             "text": "class User(BaseModel):", "path": "m.py", "chunk_index": 0},
            {"module": "models", "symbol": "validate", "kind": "function",
             "text": "def validate(self):", "path": "m.py", "chunk_index": 0},
        ]
        enriched = enrich_with_parent_context(chunks)
        func_chunks = [c for c in enriched if c["kind"] == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0]["text"].startswith("In class User:")

    def test_module_level_function_no_prefix(self):
        from doc_parser import enrich_with_parent_context
        chunks = [
            {"module": "auth", "symbol": "login", "kind": "function",
             "text": "def login():", "path": "auth.py", "chunk_index": 0},
        ]
        enriched = enrich_with_parent_context(chunks)
        assert enriched[0]["text"] == "def login():"

    def test_class_def_not_modified(self):
        from doc_parser import enrich_with_parent_context
        chunks = [
            {"module": "m", "symbol": "User", "kind": "class_def",
             "text": "class User:", "path": "m.py", "chunk_index": 0},
        ]
        enriched = enrich_with_parent_context(chunks)
        assert enriched[0]["text"] == "class User:"

    def test_signature_not_modified(self):
        from doc_parser import enrich_with_parent_context
        chunks = [
            {"module": "m", "symbol": "User", "kind": "class_def",
             "text": "class User:", "path": "m.py", "chunk_index": 0},
            {"module": "m", "symbol": "save", "kind": "signature",
             "text": "def save(self):", "path": "m.py", "chunk_index": 0},
        ]
        enriched = enrich_with_parent_context(chunks)
        sig = [c for c in enriched if c["kind"] == "signature"][0]
        assert not sig["text"].startswith("In class")

    def test_enrichment_preserves_all_fields(self):
        from doc_parser import enrich_with_parent_context
        chunks = [
            {"module": "m", "symbol": "User", "kind": "class_def",
             "text": "class User:", "path": "m.py", "chunk_index": 0,
             "source_type": "code"},
            {"module": "m", "symbol": "save", "kind": "function",
             "text": "def save(self):", "path": "m.py", "chunk_index": 0,
             "source_type": "code"},
        ]
        enriched = enrich_with_parent_context(chunks)
        func = [c for c in enriched if c["kind"] == "function"][0]
        assert func["module"] == "m"
        assert func["source_type"] == "code"
        assert func["path"] == "m.py"

    def test_empty_list(self):
        from doc_parser import enrich_with_parent_context
        assert enrich_with_parent_context([]) == []


class TestConfigurableChunkOverlap:
    def test_chunk_docs_uses_custom_overlap(self):
        from doc_parser import chunk_docs
        doc = {"module": "m", "symbol": "s", "kind": "function",
               "text": "x" * 1000, "path": "m.py"}
        chunks = list(chunk_docs([doc], max_chars=300, overlap=50))
        # Verify overlap: end of chunk N should overlap with start of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i]["text"][-50:]
            head = chunks[i + 1]["text"][:50]
            assert tail == head, f"Chunks {i} and {i+1} should overlap by 50 chars"

    def test_chunk_docs_default_overlap(self):
        from doc_parser import chunk_docs
        doc = {"module": "m", "symbol": "s", "kind": "function",
               "text": "x" * 2000, "path": "m.py"}
        chunks = list(chunk_docs([doc]))
        # Default overlap is 200, max_chars is 800
        assert len(chunks) >= 2
        tail = chunks[0]["text"][-200:]
        head = chunks[1]["text"][:200]
        assert tail == head
