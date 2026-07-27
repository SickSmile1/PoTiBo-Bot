# DocBot Improvement Plan

## Problem

DocBot currently indexes **only docstrings**. On this codebase, 83% of functions
and 94% of classes have no docstring, so they are invisible to the search
index. 80 FastAPI routes are also invisible.

Additionally, the embedding model (BGE-small-384) and flat cosine-similarity
retrieval leave quality on the table.

## Resource constraints

- 2 vCPUs, 4-8 GB total RAM
- Synapse uses ~4 GB → ~2-4 GB free for docbot
- No GPU
- No external API calls (fully self-contained)

## Goal

Maximise retrieval quality using **only local embed + rerank models** — no LLM,
no external APIs.

---

## Milestones

### M1 — Index actual source code (not just docstrings)

**Problem:** 83% of functions, 94% of classes, 100% of route endpoints produce
zero chunks. Queries like "how does login work" find nothing if the handler has
no docstring.

**Fix:** Add AST-based code extraction alongside docstrings. Each `.py` file
yields structured chunks for:

| Chunk kind | What it contains | Helps with |
|-----------|-----------------|------------|
| `function` | Function signature + body (no docstring) | "how does login handler work" |
| `signature` | `def foo(bar: int) -> str:` (always) | "what params does foo take" |
| `route` | `GET /api/users/{id} → get_user` | "what endpoints exist for users" |
| `class_def` | `class User(BaseModel):` + field types | "what fields does User model have" |

**Estimated index growth:** ~80 chunks → ~680 chunks (~0.7 MB at 384-dim).

**Files changed:**
- `app/doc_parser.py` — add `extract_code_snippets()` + helpers
- `app/source_plugins.py` — call it from `scan_codebase()`
- `app/tests/test_unit.py` — add tests

---

### M2 — Cross-encoder reranker

**Problem:** Bi-encoder cosine similarity is a frozen approximation. A
cross-encoder scores (query, chunk) pairs directly, which is much more
accurate.

**Fix:** Two-stage retrieval: bi-encoder top-20 → cross-encoder rerank top-3.

- Model: `Xenova/bge-reranker-base` ONNX (~279 MB quantized)
- Only runs at query time on 20 pairs (~300 ms on CPU)
- ~500 MB RAM during query

**Files changed:**
- `app/reranker.py` (new) — ONNX cross-encoder, download + inference
- `app/searcher.py` — add reranking step
- `app/config.py` — add `RERANKER_ENABLED` toggle

---

### M3 — Embedding model upgrade

**Problem:** BGE-small-384 (33 MB) is the smallest BGE variant.

**Fix:** Parameterise model repo/dim, switch to `Xenova/bge-base-en-v1.5`
(768-dim, ~110 MB).

- +70 MB RAM
- Auto-detect output dimension from ONNX metadata
- Existing index auto-rebuilt on dimension mismatch

**Files changed:**
- `app/embedder.py` — parameterise model repo, auto-detect dim
- `app/config.py` — add model config keys

---

### M4 — Hybrid search (BM25 + semantic fusion)

**Problem:** Dense retrieval misses exact keyword matches (function names,
error codes, HTTP methods).

**Fix:** Build BM25 inverted index at ingest. Fuse BM25 + cosine scores with
Reciprocal Rank Fusion (RRF).

- ~5 KB index overhead
- ~0.5 ms per query

**Files changed:**
- `app/index_store.py` — persist BM25 index
- `app/searcher.py` — BM25 scorer + RRF
- `app/config.py` — `HYBRID_SEARCH_ENABLED` toggle

---

### M5 — Structure-aware chunking

**Problem:** Fixed 800-char chunking splits at arbitrary positions.

**Fix:**
- Keep docstrings intact (already per-function from AST)
- Parent-context enrichment: "In class X: " prefix for method chunks
- Configurable chunk overlap (currently hardcoded 200)

**Files changed:**
- `app/doc_parser.py` — context enrichment
- `app/config.py` — overlap config key

---

## Implementation order

```
M1 (code indexing)  ── biggest gap, pure logic, no model download
    │
    ▼
M3 (embed upgrade)  ── trivial parameterisation
    │
    ▼
M2 (reranker)       ── biggest quality per compute, benefits from richer M1 chunks
    │
    ▼
M5 (better chunks)  ── polish, independent
    │
    ▼
M4 (BM25 hybrid)    ── cherry on top
```

## Resource impact summary

| Milestone   | RAM added | CPU per query | Chunks |
|-------------|-----------|---------------|--------|
| M1          | 0 MB      | 0%            | ~80 → ~680 |
| M2          | ~500 MB   | +300 ms       | —      |
| M3          | +70 MB    | 0%            | —      |
| M4          | 0 MB      | +0.5 ms       | —      |
| M5          | 0 MB      | 0%            | —      |
| **Total**   | **~570 MB** | **+~300 ms** | **8.5× more indexable content** |
