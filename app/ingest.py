import time

from embedder import Embedder
from index_store import IndexStore
from source_plugins import run_all_plugins


def run_ingest(config, embedder: Embedder, store: IndexStore) -> dict:
    """Run all source plugins, embed the collected chunks, and persist to disk.

    Args:
        config: Application configuration (codebase paths, package list, URLs).
        embedder: Embedder instance for encoding chunk text into vectors.
        store: IndexStore instance for writing metadata and embeddings.

    Returns:
        dict with keys ``total_chunks`` (int) and ``embedding_shape`` (list[int]),
        or ``{"error": "No content found from any source"}`` if no chunks.
    """
    print("Running all source plugins ...")
    t0 = time.time()
    chunks = run_all_plugins(config)
    print(f"Total chunks from all sources: {len(chunks)}")

    if not chunks:
        return {"error": "No content found from any source"}

    texts = [c["text"] for c in chunks]
    metadata = [{k: c[k] for k in c} for c in chunks]

    batch_size = 32
    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"  Embedding {total} texts in {n_batches} batches of {batch_size} ...")
    t1 = time.time()
    import numpy as np
    parts = []
    for i in range(0, total, batch_size):
        batch_num = i // batch_size + 1
        print(f"    Batch {batch_num}/{n_batches} ({i}-{min(i + batch_size, total)}) ...")
        parts.append(embedder.embed(texts[i: i + batch_size]))
    embeddings = np.concatenate(parts, axis=0)
    print(f"  Embedded in {time.time() - t1:.1f}s ({embeddings.shape})")

    store.save(metadata, embeddings)

    if getattr(config, "HYBRID_SEARCH_ENABLED", False):
        from bm25_index import BM25Index
        bm25_docs = [{"id": i, "text": texts[i]} for i in range(len(texts))]
        bm25_index = BM25Index(bm25_docs)
        store.save_bm25(bm25_index)
        print(f"  Saved BM25 index ({len(bm25_docs)} docs)")

    print(f"  Saved to {store.data_dir}")
    print(f"Total time: {time.time() - t0:.1f}s")

    return {
        "total_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape),
    }
