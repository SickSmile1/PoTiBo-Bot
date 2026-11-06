"""Cross-encoder reranker for refining bi-encoder search results.

The reranker scores (query, candidate) pairs directly with a BGE cross-encoder
model, producing more accurate relevance scores than cosine similarity on
frozen embeddings.
"""
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL_REPO = "https://huggingface.co/Xenova/bge-reranker-base/resolve/main/onnx"
TOKENIZER_REPO = "https://huggingface.co/Xenova/bge-reranker-base/resolve/main"
MAX_SEQ_LEN = 512  # BGE-reranker-base supports up to 512 tokens


def _download(url: str, dest: Path):
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"Blocked non-HTTP(S) URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


class Reranker:
    def __init__(self, model_dir: Path):
        """Download the ONNX cross-encoder and tokenizer if missing, then load.

        Args:
            model_dir: Directory where ``reranker.onnx`` and ``tokenizer.json``
                       are stored or will be downloaded to.
        """
        model_path = model_dir / "reranker.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        if not model_path.exists():
            print(f"Downloading reranker model from {MODEL_REPO} ...")
            _download(f"{MODEL_REPO}/model_quantized.onnx", model_path)
        if not tokenizer_path.exists():
            _download(f"{TOKENIZER_REPO}/tokenizer.json", tokenizer_path)

        try:
            self.session = ort.InferenceSession(str(model_path))
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception:
            print("Corrupt reranker files detected, re-downloading ...")
            model_path.unlink(missing_ok=True)
            tokenizer_path.unlink(missing_ok=True)
            _download(f"{MODEL_REPO}/model_quantized.onnx", model_path)
            _download(f"{TOKENIZER_REPO}/tokenizer.json", tokenizer_path)
            self.session = ort.InferenceSession(str(model_path))
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score & re-sort candidates by cross-encoder relevance.

        Args:
            query: The raw query string.
            candidates: List of chunk dicts, each with a ``text`` key.

        Returns:
            Same list of dicts with updated ``score`` values, sorted
            descending by score.
        """
        if not candidates:
            return candidates

        texts = [c["text"] for c in candidates]
        scores = self._score_pairs(query, texts)

        for c, s in zip(candidates, scores):
            c["score"] = round(float(s), 4)

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def _score_pairs(self, query: str, texts: list[str]) -> np.ndarray:
        """Score (query, document) pairs through the cross-encoder.

        The tokenizer handles ``[CLS] query [SEP] document [SEP]`` formatting
        automatically when passed a list of ``(text, pair)`` tuples.

        Returns:
            np.ndarray of shape ``(len(texts),)`` with raw logit scores.
        """
        pairs = [(query, t) for t in texts]
        encoded = self.tokenizer.encode_batch(pairs)

        max_len = min(max(len(e.ids) for e in encoded), MAX_SEQ_LEN)

        input_ids = np.zeros((len(pairs), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(pairs), max_len), dtype=np.int64)

        for i, e in enumerate(encoded):
            ids = e.ids[:max_len]
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        required = {inp.name for inp in self.session.get_inputs()}
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in required:
            token_type_ids = np.zeros((len(pairs), max_len), dtype=np.int64)
            feed["token_type_ids"] = token_type_ids

        outputs = self.session.run(None, feed)
        return outputs[0].flatten()
