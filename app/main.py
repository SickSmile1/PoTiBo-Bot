import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

GIT_CMD = shutil.which("git") or "/usr/bin/git"

from config import Config
from embedder import Embedder
from index_store import IndexStore
from searcher import Searcher
from ingest import run_ingest
from bot import DocBot, register_matrix_user

try:
    from reranker import Reranker as _Reranker
except ImportError:
    _Reranker = None

_COPY_IGNORE = shutil.ignore_patterns(
    "node_modules", ".venv", "venv", "__pycache__", ".git",
    ".idea", ".vscode", "*.pyc", "*.egg-info", "*.pyo",
    "dist", "build", ".next", ".nuxt",
)
_CLONE_DEST = "/tmp/codebase-clone"
# Only these subdirectories of CODEBASE_SRC_DIR are copied into the temp clone.
_INCLUDE_SUBDIRS = ("app", "frontend")


def _total_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def prepare_codebase(config: Config) -> str | None:
    """Return a local path with the codebase to index, or None on failure.

    Priority:
    1. Copy from CODEBASE_SRC_DIR (local mount, preferred — no credentials needed).
    2. Git-clone from GIT_REPO_URL (fallback; embed credentials in the URL via .env).

    The returned path is always /tmp/codebase-clone so the caller can safely
    delete it after ingestion without touching the original source.
    """
    dest = Path(_CLONE_DEST)
    if dest.exists():
        shutil.rmtree(dest)

    src_dir = config.CODEBASE_SRC_DIR
    if src_dir and Path(src_dir).exists():
        src = Path(src_dir)
        dest.mkdir(parents=True, exist_ok=True)
        total_mb = 0.0
        for subdir in _INCLUDE_SUBDIRS:
            sub_src = src / subdir
            if not sub_src.exists():
                print(f"  Subdir '{subdir}' not found in {src}, skipping")
                continue
            sub_mb = _total_size_mb(sub_src)
            total_mb += sub_mb
            if total_mb > config.MAX_TOTAL_COPY_MB:
                print(
                    f"Source exceeds MAX_TOTAL_COPY_MB={config.MAX_TOTAL_COPY_MB} MB. Aborting."
                )
                return None
            print(f"  Copying {subdir}/ ({sub_mb:.0f} MB) ...")
            shutil.copytree(sub_src, dest / subdir, ignore=_COPY_IGNORE)
        print(f"Copy complete: {total_mb:.0f} MB → {dest}")
        return str(dest)

    if config.GIT_REPO_URL:
        print(f"CODEBASE_SRC_DIR not set, cloning from {config.GIT_REPO_URL} ...")
        dest.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [GIT_CMD, "clone", config.GIT_REPO_URL, str(dest)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"Git clone failed: {result.stderr.strip()}")
            return None
        print(f"Cloned into {dest}")
        return str(dest)

    print("No CODEBASE_SRC_DIR or GIT_REPO_URL configured.")
    return None


async def main():
    """Load or build the index, then start the Matrix bot.

    Startup sequence:
    1. Load the ONNX embedder (downloads model on first run).
    2. If REBUILD_ON_START or no index on disk: copy/clone codebase, run all
       source plugins, embed chunks, save index, then delete the temp copy.
    3. If an index already exists and REBUILD_ON_START is false: load from disk.
    4. Create a Searcher and DocBot, then enter the Matrix sync loop.
    """
    config = Config()

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_dir = data_dir / "model"
    model_dir.mkdir(exist_ok=True)

    print(f"Loading embedder ({config.EMBED_MODEL_REPO}) ...")
    embedder = Embedder(model_dir, model_repo=config.EMBED_MODEL_REPO)

    store = IndexStore(data_dir)

    if config.REBUILD_ON_START and store.exists():
        print("REBUILD_ON_START=true — clearing existing index")
        store.clear()

    chunk_count = 0
    if store.exists():
        chunk_count = store.load()
        if store.dim != embedder.embed_dim:
            print(
                f"Embedding dim mismatch (index: {store.dim}, model: "
                f"{embedder.embed_dim}), clearing and rebuilding"
            )
            store.clear()
            chunk_count = 0

    if not store.exists():
        print("No existing index found, building ...")
        codebase_path = prepare_codebase(config)
        if not codebase_path:
            print("No codebase available, exiting")
            sys.exit(1)

        # Temporarily override CODEBASE_DIR so source plugins point at the clone
        config.CODEBASE_DIR = codebase_path

        try:
            result = run_ingest(config, embedder, store)
        finally:
            if Path(_CLONE_DEST).exists():
                shutil.rmtree(_CLONE_DEST)
                print(f"Deleted temp codebase clone: {_CLONE_DEST}")

        if "error" in result:
            print(f"Ingestion failed: {result['error']}")
            sys.exit(1)
        chunk_count = result.get("total_chunks", 0)
        print(f"Ingestion complete: {result}")
    else:
        chunk_count = store.load()
        print(f"Loaded existing index: {chunk_count} chunks")

    if config.TEST_USER_USERNAME and config.TEST_USER_PASSWORD:
        print(f"Registering smoke-test user @{config.TEST_USER_USERNAME} ...")
        await register_matrix_user(
            config.MATRIX_HOMESERVER,
            config.MATRIX_REGISTRATION_SHARED_SECRET,
            config.TEST_USER_USERNAME,
            config.TEST_USER_PASSWORD,
        )

    reranker = None
    if config.RERANKER_ENABLED:
        if _Reranker is None:
            print("Reranker module not available, disabling")
        else:
            reranker_dir = data_dir / "reranker"
            reranker_dir.mkdir(exist_ok=True)
            print("Loading reranker ...")
            reranker = _Reranker(reranker_dir)

    searcher = Searcher(
        store.metadata,
        store.embeddings,
        reranker=reranker,
        reranker_candidates=config.RERANKER_CANDIDATES,
        bm25_index=store.load_bm25() if config.HYBRID_SEARCH_ENABLED else None,
    )
    bot = DocBot(config, embedder, searcher, chunks=chunk_count)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
