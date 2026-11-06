import httpx
from bs4 import BeautifulSoup


def fetch_url_text(url: str, timeout: int = 30) -> str | None:
    """Download a URL, strip HTML tags, and return visible text.

    Removes ``script``, ``style``, ``nav``, ``footer``, ``header``, and
    ``aside`` elements before extracting text.  Lines shorter than 30
    characters are filtered out to reduce noise.

    Args:
        url: The fully qualified URL to fetch.
        timeout: HTTP request timeout in seconds (default 30).

    Returns:
        str | None: The extracted text content, or None on failure.
    """
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.split("\n") if len(line.strip()) > 30]
        return "\n".join(lines)
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def extract_sections(text: str, url: str, tag: str) -> list[dict]:
    """Split plain text into sections delimited by headings.

    A heading is any line that starts with ``#`` or is fully uppercase.
    Each section becomes a chunk dict ready for further processing.

    Args:
        text: Plain text content (already stripped of HTML).
        url: The source URL (stored in the ``module`` and ``path`` fields).
        tag: Value for the ``source_type`` field.

    Returns:
        list[dict]: Section chunks with keys module, symbol, kind, text,
                    path, and source_type.  Sections shorter than 50
                    characters are discarded.
    """
    chunks = []
    lines = text.split("\n")
    current_heading = "Overview"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.isupper():
            if current_lines:
                body = "\n".join(current_lines).strip()
                if len(body) > 50:
                    chunks.append({
                        "module": url,
                        "symbol": current_heading,
                        "kind": "web-section",
                        "text": body,
                        "path": url,
                        "source_type": tag,
                    })
            current_heading = stripped[:80]
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if len(body) > 50:
            chunks.append({
                "module": url,
                "symbol": current_heading,
                "kind": "web-section",
                "text": body,
                "path": url,
                "source_type": tag,
            })

    return chunks
