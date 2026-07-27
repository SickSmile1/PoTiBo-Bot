import ast
from pathlib import Path

MAX_FILE_SIZE_BYTES = 200 * 1024  # skip files larger than 200 KB


def extract_docstrings(codebase_dir: Path):
    """Walk a directory tree and yield Python docstrings found via AST.

    Skips ``test/``, ``__pycache__/``, ``node_modules/``, ``.git/``, and
    ``.venv/`` directories.  Files with ``SyntaxError`` are silently skipped.

    Args:
        codebase_dir: Root directory to recursively search for ``.py`` files.

    Yields:
        dict with keys ``module`` (dotted module path), ``symbol``
        (module/class/function name), ``kind`` ("module"/"class"/"function"),
        ``text`` (the docstring body), and ``path`` (relative file path).
    """
    for py_file in codebase_dir.rglob("*.py"):
        rel = py_file.relative_to(codebase_dir)
        parts = rel.parts
        if any(p in ("test", "tests", "__pycache__", "node_modules", ".git", ".venv") for p in parts):
            continue
        if py_file.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except SyntaxError:
            continue

        module = ".".join(rel.with_suffix("").parts)

        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            yield {
                "module": module,
                "symbol": module,
                "kind": "module",
                "text": mod_doc,
                "path": str(rel),
            }

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    yield {
                        "module": module,
                        "symbol": node.name,
                        "kind": kind,
                        "text": doc,
                        "path": str(rel),
                    }


MAX_CODE_CHARS = 400  # code is denser than prose, truncate earlier

_ROUTE_METHODS = frozenset({"get", "post", "put", "delete", "patch", "route", "api_route"})


def _route_text(node: ast.expr, handler: str) -> str | None:
    """Extract HTTP method + path from a route decorator.

    Returns something like ``"GET /users/{id} → get_user"``,
    or ``None`` if the node is not a route decorator.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr not in _ROUTE_METHODS:
        return None

    method = {"api_route": "ANY", "route": "ANY"}.get(
        node.func.attr, node.func.attr.upper()
    )
    path = ""
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        path = node.args[0].value
    for kw in node.keywords:
        if kw.arg == "path" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            path = kw.value.value
    if not path:
        return None
    return f"{method} {path} → {handler}"


def extract_code_snippets(codebase_dir: Path):
    """Walk ``.py`` files and yield structured code chunks for indexing.

    Unlike ``extract_docstrings`` which only yields docstring text, this
    function yields chunks about the **code itself**:

    * ``signature`` — the ``def`` line of every function/async-function.
    * ``function`` — the full source of functions **without** a docstring
      (so they are still findable).
    * ``route`` — HTTP method + path + handler name for every FastAPI-like
      route decorator found.
    * ``class_def`` — class header + annotated field definitions for classes
      that have typed fields (e.g. Pydantic / SQLModel models).

    Skips the same directories as ``extract_docstrings``.

    Args:
        codebase_dir: Root directory to recursively search for ``.py`` files.

    Yields:
        dict with keys ``module``, ``symbol``, ``kind``, ``text``, ``path``,
        and ``chunk_index`` (always 0).
    """
    for py_file in codebase_dir.rglob("*.py"):
        rel = py_file.relative_to(codebase_dir)
        parts = rel.parts
        if any(p in ("test", "tests", "__pycache__", "node_modules", ".git", ".venv") for p in parts):
            continue
        if py_file.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except SyntaxError:
            continue

        module = ".".join(rel.with_suffix("").parts)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_doc = bool(ast.get_docstring(node))

                # Route decorators
                for dec in node.decorator_list:
                    info = _route_text(dec, node.name)
                    if info:
                        yield {
                            "module": module,
                            "symbol": node.name,
                            "kind": "route",
                            "text": info,
                            "path": str(rel),
                            "chunk_index": 0,
                        }

                seg = ast.get_source_segment(text, node)
                if not seg:
                    continue

                seg_lines = seg.split("\n")
                def_line = None
                for line in seg_lines:
                    s = line.strip()
                    if s.startswith("def ") or s.startswith("async def "):
                        def_line = s[:200]
                        break

                if def_line:
                    yield {
                        "module": module,
                        "symbol": node.name,
                        "kind": "signature",
                        "text": def_line,
                        "path": str(rel),
                        "chunk_index": 0,
                    }

                if not has_doc:
                    body = seg
                    for i, line in enumerate(seg_lines):
                        if line.strip().startswith("def ") or line.strip().startswith("async def "):
                            body = "\n".join(seg_lines[i:])
                            break
                    yield {
                        "module": module,
                        "symbol": node.name,
                        "kind": "function",
                        "text": body[:MAX_CODE_CHARS],
                        "path": str(rel),
                        "chunk_index": 0,
                    }

            elif isinstance(node, ast.ClassDef):
                seg = ast.get_source_segment(text, node)
                if not seg:
                    continue

                class_header = None
                for line in seg.split("\n"):
                    if line.strip().startswith("class "):
                        class_header = line.strip()
                        break

                fields = []
                source_lines = text.split("\n")
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        idx = item.lineno - 1
                        if 0 <= idx < len(source_lines):
                            fields.append(source_lines[idx].rstrip())

                if class_header and fields:
                    yield {
                        "module": module,
                        "symbol": node.name,
                        "kind": "class_def",
                        "text": class_header + "\n" + "\n".join(fields),
                        "path": str(rel),
                        "chunk_index": 0,
                    }


def chunk_docs(docs, max_chars=800, overlap=200):
    """Split long docstrings into overlapping chunks.

    Documents shorter than ``max_chars`` are passed through unchanged.
    Longer documents are split with ``overlap`` characters of context
    carried between chunks.

    Args:
        docs: Iterable of dicts (as yielded by ``extract_docstrings``).
        max_chars: Maximum characters per chunk (default 800).
        overlap: Overlap in characters between consecutive chunks (default 200).

    Yields:
        dict: Same as input but with ``text`` truncated to ``max_chars``
              (or fewer for the final chunk) and a ``chunk_index`` added.
    """
    for doc in docs:
        text = doc["text"]
        if len(text) <= max_chars:
            yield {**doc, "chunk_index": 0}
        else:
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + max_chars, len(text))
                yield {**doc, "text": text[start:end], "chunk_index": chunk_idx}
                chunk_idx += 1
                start += max_chars - overlap


def enrich_with_parent_context(chunks: list[dict]) -> list[dict]:
    """Prepend parent class context to function chunks.

    When a ``function`` chunk comes from the same file as a ``class_def``
    chunk, its text is prefixed with ``In class <ClassName>:`` so that
    search queries like "User validate" can find class methods even when
    the method text itself doesn't mention the class name.

    Only ``function`` kind chunks are enriched.  ``signature``, ``class_def``,
    ``route``, ``module``, ``curated``, and ``web-section`` kinds are
    passed through unchanged.

    Args:
        chunks: List of chunk dicts (as produced by ``extract_docstrings``,
                ``extract_code_snippets``, or ``chunk_docs``).

    Returns:
        New list with enriched text for applicable function chunks.
    """
    # Build a map of path → class symbol for class_def chunks
    class_by_path: dict[str, str] = {}
    for c in chunks:
        if c.get("kind") == "class_def":
            class_by_path[c["path"]] = c["symbol"]

    enriched = []
    for c in chunks:
        if c.get("kind") == "function" and c["path"] in class_by_path:
            parent = class_by_path[c["path"]]
            enriched.append({**c, "text": f"In class {parent}: {c['text']}"})
        else:
            enriched.append(c)
    return enriched
