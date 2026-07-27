import json
import numpy as np
from pathlib import Path


class IndexStore:
    def __init__(self, data_dir: Path):
        """Prepare paths for metadata JSON, embeddings NPY, and BM25 index.

        Args:
            data_dir: Directory where index files are stored and loaded from.
        """
        self.data_dir = data_dir
        self.meta_path = data_dir / "index.meta.json"
        self.emb_path = data_dir / "index.embeddings.npy"
        self.bm25_path = data_dir / "index.bm25.json"
        self.metadata: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def exists(self) -> bool:
        """Check whether both metadata and embedding files exist on disk.

        Returns:
            bool: True if both files exist, False otherwise.
        """
        return self.meta_path.exists() and self.emb_path.exists()

    def save(self, metadata: list[dict], embeddings: np.ndarray):
        """Persist metadata and embeddings to disk and update in-memory state.

        Args:
            metadata: List of chunk dicts (including the ``text`` key for
                      reranker scoring and bot response display).
            embeddings: Float32 numpy array of shape ``(n_chunks, embed_dim)``.
        """
        self.metadata = metadata
        self.embeddings = embeddings
        with open(self.meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        np.save(str(self.emb_path), embeddings)

    def save_bm25(self, bm25_index):
        """Persist the BM25 index to disk.

        Args:
            bm25_index: A ``BM25Index`` instance (from ``bm25_index`` module).
        """
        bm25_index.save(self.bm25_path)

    def load_bm25(self):
        """Load the BM25 index from disk.

        Returns:
            A ``BM25Index`` instance, or None if no BM25 index file exists.
        """
        if not self.bm25_path.exists():
            return None
        from bm25_index import BM25Index
        return BM25Index.load(self.bm25_path)

    def clear(self):
        """Delete index files from disk and reset in-memory state."""
        self.meta_path.unlink(missing_ok=True)
        self.emb_path.unlink(missing_ok=True)
        self.bm25_path.unlink(missing_ok=True)
        self.metadata = []
        self.embeddings = None

    def load(self) -> int:
        """Load metadata and embeddings from disk into memory.

        Returns:
            int: Number of chunks loaded.
        """
        with open(self.meta_path) as f:
            self.metadata = json.load(f)
        self.embeddings = np.load(str(self.emb_path))
        return len(self.metadata)

    @property
    def dim(self) -> int | None:
        """Embedding dimension of the currently loaded index, or None."""
        if self.embeddings is not None:
            return self.embeddings.shape[1]
        return None
