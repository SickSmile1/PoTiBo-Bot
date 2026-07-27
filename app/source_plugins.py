import hashlib
import sys
from pathlib import Path

from doc_parser import extract_docstrings as scan_py_docstrings
from doc_parser import extract_code_snippets as scan_py_code
from doc_parser import chunk_docs, enrich_with_parent_context
from webfetch import fetch_url_text, extract_sections


def _sha256(chunk: dict) -> str:
    """Compute a short content-addressable hash for a chunk dict.

    The hash is taken over ``module:symbol:chunk_index:text`` so that
    identical content from different sources is still considered unique.

    Args:
        chunk: Chunk dict containing at least ``module``, ``symbol``,
               ``chunk_index``, and ``text`` keys.

    Returns:
        str: First 16 hex characters of the SHA-256 digest.
    """
    raw = f"{chunk['module']}:{chunk['symbol']}:{chunk['chunk_index']}:{chunk['text']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def scan_codebase(codebase_dir: Path, tag: str = "code") -> list[dict]:
    """Extract Python docstrings from a project codebase directory.

    Walks all ``.py`` files under ``codebase_dir``, parses them with the
    ``ast`` module, and yields docstrings for modules, classes, and functions.

    Args:
        codebase_dir: Path to the root of the project source tree.
        tag: Value to set on the ``source_type`` field of each chunk.

    Returns:
        list[dict]: Chunks with keys module, symbol, kind, text, path,
                    source_type, chunk_index, sha256.
    """
    if not codebase_dir.exists():
        print(f"  Codebase dir {codebase_dir} not found, skipping")
        return []
    raw = list(scan_py_docstrings(codebase_dir))
    chunks = list(chunk_docs(raw))
    code_chunks = list(scan_py_code(codebase_dir))
    chunks.extend(code_chunks)
    chunks = enrich_with_parent_context(chunks)
    for c in chunks:
        c["source_type"] = tag
        c["sha256"] = _sha256(c)
    print(f"  Codebase ({codebase_dir}): {len(raw)} docstrings + {len(code_chunks)} code snippets → {len(chunks)} total chunks")
    return chunks


def scan_site_packages(site_packages_dir: Path, packages: list[str], tag: str = "library") -> list[dict]:
    """Extract Python docstrings from specific installed packages.

    Args:
        site_packages_dir: Path to Python's site-packages directory.
        packages: List of package names (directory names) to scan.
        tag: Value to set on the ``source_type`` field.

    Returns:
        list[dict]: Chunks tagged with the given source_type.
    """
    chunks = []
    for pkg in packages:
        pkg_dir = site_packages_dir / pkg
        if not pkg_dir.exists():
            print(f"  Package {pkg} not found at {pkg_dir}, skipping")
            continue
        raw = list(scan_py_docstrings(pkg_dir))
        pkg_chunks = list(chunk_docs(raw))
        for c in pkg_chunks:
            c["source_type"] = tag
            c["sha256"] = _sha256(c)
        chunks.extend(pkg_chunks)
        print(f"  Package {pkg}: {len(raw)} docstrings → {len(pkg_chunks)} chunks")
    return chunks


def scan_curated_files(sources_dir: Path, tag: str = "reference") -> list[dict]:
    """Read curated markdown documents and chunk them for indexing.

    Useful for shipping reference material such as HTTP status codes or
    security cheatsheets inside the container image.

    Args:
        sources_dir: Directory containing ``.md`` files to index.
        tag: Value to set on the ``source_type`` field.

    Returns:
        list[dict]: Chunks tagged with reference source_type.
    """
    chunks = []
    for md_file in sources_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        chunk = {
            "module": md_file.stem,
            "symbol": md_file.stem,
            "kind": "curated",
            "text": text,
            "path": str(md_file),
            "source_type": tag,
        }
        sub_chunks = list(chunk_docs([chunk]))
        for c in sub_chunks:
            c["sha256"] = _sha256(c)
        chunks.extend(sub_chunks)
        print(f"  Curated {md_file.name}: {len(sub_chunks)} chunks")
    return chunks


def scan_web_urls(urls: list[str], tag: str = "web") -> list[dict]:
    """Fetch web pages, extract text by section, and chunk for indexing.

    Each URL is fetched with ``httpx``, stripped of HTML tags with
    BeautifulSoup, and split by headings.

    Args:
        urls: List of URLs to fetch and index.
        tag: Value to set on the ``source_type`` field.

    Returns:
        list[dict]: Chunks tagged with web source_type.
    """
    chunks = []
    for url in urls:
        print(f"  Fetching {url} ...")
        text = fetch_url_text(url)
        if not text:
            continue
        sections = extract_sections(text, url, tag)
        sub_chunks = []
        for s in sections:
            sub = list(chunk_docs([s]))
            sub_chunks.extend(sub)
        for c in sub_chunks:
            c["sha256"] = _sha256(c)
        chunks.extend(sub_chunks)
        print(f"    → {len(sub_chunks)} chunks")
    return chunks


def deduplicate(all_chunks: list[dict]) -> list[dict]:
    """Remove chunks whose sha256 hash already appears in the list.

    Preserves the first occurrence of each hash.

    Args:
        all_chunks: List of chunk dicts (each must have a ``sha256`` key).

    Returns:
        list[dict]: Deduplicated list.
    """
    seen = set()
    unique = []
    for c in all_chunks:
        h = c.get("sha256")
        if not h:
            unique.append(c)
            continue
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique


def run_all_plugins(config) -> list[dict]:
    """Execute every registered source plugin and return deduplicated chunks.

    Order of execution:
    1. Project codebase docstrings (``code``).
    2. Installed site-packages for chosen libraries (``library``).
    3. Curated reference files shipped with the image (``reference``).
    4. Web URLs fetched at indexing time (``web``).

    Results are deduplicated by SHA-256 hash before returning.

    Args:
        config: Application configuration with paths, package list, and URLs.

    Returns:
        list[dict]: All unique chunks from all plugins.
    """
    all_chunks = []

    all_chunks.extend(
        scan_codebase(Path(config.CODEBASE_DIR), tag="code")
    )

    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    all_chunks.extend(
        scan_site_packages(site_packages, config.PACKAGES_TO_SCAN, tag="library")
    )

    all_chunks.extend(
        scan_curated_files(Path(config.SOURCES_DIR), tag="reference")
    )

    if config.WEB_URLS:
        all_chunks.extend(
            scan_web_urls(config.WEB_URLS, tag="web")
        )

    unique = deduplicate(all_chunks)
    if len(unique) < len(all_chunks):
        print(f"  Dedup removed {len(all_chunks) - len(unique)} duplicate chunks")

    return unique
