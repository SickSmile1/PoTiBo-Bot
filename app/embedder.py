import urllib.parse
import urllib.request
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer


_HF = "https://huggingface.co/{model_repo}/resolve/main"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _download(url: str, dest: Path):
    """Download a file from ``url`` and write it to ``dest``.

    Creates parent directories as needed.

    Args:
        url: Source URL to download from.
        dest: Destination path on disk.
    """
    if not urllib.parse.urlparse(url).scheme in ("http", "https"):
        raise ValueError(f"Blocked non-HTTP(S) URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


class Embedder:
    def __init__(self, model_dir: Path, model_repo: str = "Xenova/bge-small-en-v1.5"):
        """Download the ONNX model and tokenizer if missing, then load both.

        The model is identified by its HuggingFace repo path (e.g.
        ``Xenova/bge-small-en-v1.5``).  On first use the model files are
        fetched from HuggingFace and cached in ``model_dir`` for subsequent
        runs.

        Args:
            model_dir: Directory where ``model.onnx`` and ``tokenizer.json``
                       are stored or will be downloaded to.
            model_repo: HuggingFace repo path, e.g.
                        ``"Xenova/bge-small-en-v1.5"`` (384-dim) or
                        ``"Xenova/bge-base-en-v1.5"`` (768-dim).
        """
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        sentinel_path = model_dir / ".model_repo"
        hf_url = _HF.format(model_repo=model_repo)

        # If the cached model was downloaded for a different repo, clear it.
        if sentinel_path.exists():
            cached_repo = sentinel_path.read_text().strip()
            if cached_repo != model_repo:
                print(f"Model repo changed ({cached_repo} → {model_repo}), re-downloading")
                for p in (model_path, tokenizer_path, sentinel_path):
                    p.unlink(missing_ok=True)

        if not model_path.exists():
            print(f"Downloading model from {model_repo} ...")
            _download(f"{hf_url}/onnx/model.onnx", model_path)
        if not tokenizer_path.exists():
            _download(f"{hf_url}/tokenizer.json", tokenizer_path)

        sentinel_path.write_text(model_repo)

        self.session = ort.InferenceSession(str(model_path))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Auto-detect embedding dimension from ONNX output metadata.
        # Output shape: (batch, seq_len, hidden_dim) — the last axis is
        # the hidden size and differs per model variant.
        output_shape = self.session.get_outputs()[0].shape
        self.embed_dim = output_shape[-1]

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode a list of texts into L2-normalized embedding vectors.

        If ``is_query`` is True each text is prefixed with the BGE
        instruction for retrieval queries.

        The pipeline is: tokenize → ONNX forward → mean pooling → normalize.

        Args:
            texts: One or more text strings to embed.
            is_query: If True prepend the BGE query instruction prefix.

        Returns:
            np.ndarray: Float32 array of shape ``(len(texts), embed_dim)``,
                        each row L2-normalised.
        """
        if is_query:
            texts = [f"{QUERY_PREFIX}{t}" for t in texts]

        encoded = self.tokenizer.encode_batch(texts)
        max_len = min(max(len(e.ids) for e in encoded), 256)

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, e in enumerate(encoded):
            ids = e.ids[:max_len]
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1

        # Determine required inputs dynamically so the same code works with
        # model variants that do or don't need token_type_ids.
        required = {inp.name for inp in self.session.get_inputs()}
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in required:
            feed["token_type_ids"] = token_type_ids

        outputs = self.session.run(None, feed)

        token_emb = outputs[0]
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        embeddings = np.sum(token_emb * mask, axis=1) / np.sum(mask, axis=1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / norms).astype(np.float32)
