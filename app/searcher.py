import numpy as np


def _chunk_key(item: dict) -> str:
    """Derive a content-based dedup key for a chunk dict.

    Uses module, symbol, chunk_index, and path to uniquely identify a chunk
    across cosine and BM25 result lists.
    """
    return f"{item.get('module', '')}:{item.get('symbol', '')}:{item.get('chunk_index', 0)}:{item.get('path', '')}"


def reciprocal_rank_fusion(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each item is scored as ``sum(1 / (k + rank_i + 1))`` across all lists
    where it appears.  Items with the same content key are merged (the
    higher-ranked version is kept, with the fused score).

    Args:
        lists: List of ranked result lists.  Each item must be a dict.
        k: RRF constant (default 60).  Higher values reduce the impact
            of rank differences.

    Returns:
        Fused list sorted by RRF score descending, with an added
        ``rrf_score`` key on each item.
    """
    if not lists:
        return []

    scores: dict[str, float] = {}
    key_to_item: dict[str, dict] = {}

    for ranked_list in lists:
        for rank, item in enumerate(ranked_list):
            key = _chunk_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in key_to_item:
                key_to_item[key] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**key_to_item[key], "rrf_score": s} for key, s in ranked]


class Searcher:
    def __init__(self, metadata: list[dict], embeddings: np.ndarray,
                 reranker=None, reranker_candidates: int = 20,
                 bm25_index=None):
        """Store the index metadata and embedding matrix for search.

        Args:
            metadata: List of chunk dicts loaded from index.meta.json.
            embeddings: Float32 numpy array of shape ``(n_chunks, dim)``,
                        each row L2-normalised.
            reranker: Optional ``Reranker`` instance for cross-encoder
                      re-scoring of candidate results.
            reranker_candidates: How many initial candidates to retrieve
                                 from the bi-encoder before reranking.
                                 Only used when ``reranker`` is set.
            bm25_index: Optional ``BM25Index`` instance for hybrid search.
                        When provided, cosine and BM25 results are fused
                        via Reciprocal Rank Fusion.
        """
        self.metadata = metadata
        self.embeddings = embeddings
        self.reranker = reranker
        self.reranker_candidates = reranker_candidates
        self.bm25_index = bm25_index

    def search(self, query_emb: np.ndarray, top_k: int = 3,
               threshold: float = 0.65, query_text: str = "") -> list[dict]:
        """Retrieve top-k results, optionally refined by a cross-encoder
        and/or fused with BM25 keyword scoring.

        Pipeline stages (each optional):

        1. **Cosine similarity** — fast bi-encoder retrieval.
        2. **BM25 scoring** — keyword-based retrieval (when ``bm25_index``
           is set and ``query_text`` is provided).
        3. **RRF fusion** — combine cosine and BM25 rankings (when both
           are available).
        4. **Cross-encoder reranking** — re-score top candidates (when
           ``reranker`` is set).

        Args:
            query_emb: Query embedding of shape ``(1, dim)`` (L2-normalised).
            top_k: Maximum number of final results to return.
            threshold: Minimum cosine similarity score for the first-stage
                       filter.  Results below this are discarded before
                       reranking.
            query_text: Raw query string (required for reranking and BM25;
                        ignored when neither is configured).

        Returns:
            list[dict]: Up to ``top_k`` results, each containing a ``score``
                        key plus all metadata fields from the matching chunk.
                        Empty list if no result meets the threshold.
        """
        candidate_count = (
            max(top_k, self.reranker_candidates)
            if self.reranker else top_k
        )

        # ── Stage 1: Cosine similarity ───────────────────────────────
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:candidate_count]

        cosine_results = []
        for idx in top_indices:
            score = float(round(scores[idx], 4))
            if score < threshold:
                continue
            cosine_results.append({"score": score, **self.metadata[idx]})

        # ── Stage 2+3: BM25 + RRF fusion (optional) ─────────────────
        results = cosine_results
        if self.bm25_index and query_text and cosine_results:
            bm25_scores = self.bm25_index.score(query_text)
            if bm25_scores:
                bm25_ranked = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )
                bm25_results = []
                for idx in bm25_ranked:
                    if bm25_scores[idx] <= 0:
                        break
                    bm25_results.append({"score": bm25_scores[idx], **self.metadata[idx]})

                if bm25_results:
                    fused = reciprocal_rank_fusion(
                        [cosine_results, bm25_results], k=60
                    )
                    results = fused

        # ── Stage 4: Cross-encoder reranking (optional) ──────────────
        if self.reranker and results and query_text:
            results = self.reranker.rerank(query_text, results)

        return results[:top_k]
