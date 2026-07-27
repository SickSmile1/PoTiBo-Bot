"""Okapi BM25 inverted index for hybrid search.

Provides keyword-based scoring to complement dense vector similarity.
Used together with cosine search via Reciprocal Rank Fusion (RRF).
"""
import json
import math
from collections import defaultdict
from pathlib import Path


class BM25Index:
    """Okapi BM25 inverted index over a list of document dicts.

    Args:
        docs: List of dicts, each with at least an ``id`` and ``text`` key.
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalisation parameter (default 0.75).
    """

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self._inverted: dict[str, list[int]] = defaultdict(list)
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._n = len(docs)

        total_len = 0
        for i, doc in enumerate(docs):
            tokens = doc["text"].lower().split()
            self._doc_lens.append(len(tokens))
            total_len += len(tokens)
            for tok in set(tokens):
                self._inverted[tok].append(i)

        self._avgdl = total_len / max(self._n, 1)

    def score(self, query: str) -> list[float]:
        """Score all documents against the query.

        Args:
            query: Space-separated query string.

        Returns:
            List of BM25 scores, one per document.  Zero for documents
            with no matching terms.
        """
        if not self.docs:
            return []

        tokens = query.lower().split()
        scores = [0.0] * self._n

        for tok in tokens:
            if tok not in self._inverted:
                continue
            df = len(self._inverted[tok])
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id in self._inverted[tok]:
                doc_tokens = self.docs[doc_id]["text"].lower().split()
                tf = doc_tokens.count(tok)
                dl = self._doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1.0))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom

        return scores

    def save(self, path: Path):
        """Persist index to a JSON file.

        Args:
            path: Destination file path.
        """
        data = {
            "docs": self.docs,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load a persisted index from a JSON file.

        Args:
            path: Path to the JSON file written by ``save()``.

        Returns:
            A new ``BM25Index`` instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(data["docs"], k1=data["k1"], b=data["b"])
