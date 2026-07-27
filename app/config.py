import json
import os


class Config:
    MATRIX_HOMESERVER = os.getenv("MATRIX_HOMESERVER", "http://synapse:8008")
    MATRIX_SERVER_NAME = os.getenv("MATRIX_SERVER_NAME", "localhost")
    MATRIX_BOT_USERNAME = os.getenv("MATRIX_BOT_USERNAME", "doc_bot")
    DOCBOT_PASSWORD = os.getenv("DOCBOT_PASSWORD", "")
    DOCBOT_HEARTBEAT_TOKEN = os.getenv("DOCBOT_HEARTBEAT_TOKEN", "")
    DATA_DIR = os.getenv("DATA_DIR", "/data")
    CODEBASE_DIR = os.getenv("CODEBASE_DIR", "/codebase")
    CODEBASE_SRC_DIR = os.getenv("CODEBASE_SRC_DIR", "")
    MATRIX_REGISTRATION_SHARED_SECRET = os.getenv("MATRIX_REGISTRATION_SHARED_SECRET", "")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    GIT_REPO_URL = os.getenv("GIT_REPO_URL", "https://github.com/SickSmile1/fastapi_playground.git")
    REBUILD_ON_START = os.getenv("REBUILD_ON_START", "false").lower() in ("1", "true", "yes")
    MAX_TOTAL_COPY_MB = int(os.getenv("MAX_TOTAL_COPY_MB", "500"))

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    SOURCES_DIR = os.getenv("SOURCES_DIR", os.path.join(APP_DIR, "sources"))

    PACKAGES_TO_SCAN = os.getenv("PACKAGES_TO_SCAN", "fastapi,starlette,pydantic").split(",")

    WEB_URLS_RAW = os.getenv("WEB_URLS", "[]")
    @property
    def WEB_URLS(self) -> list[str]:
        """Parse the WEB_URLS env var as a JSON array of URL strings.

        Returns:
            list[str]: The configured URLs, or an empty list on parse failure.
        """
        try:
            return json.loads(self.WEB_URLS_RAW)
        except json.JSONDecodeError:
            return []

    EMBED_MODEL_REPO = os.getenv("EMBED_MODEL_REPO", "Xenova/bge-small-en-v1.5")

    RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() in ("1", "true", "yes")
    RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "20"))

    HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() in ("1", "true", "yes")

    CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    VERBOSE = os.getenv("DOCBOT_VERBOSE", "0") == "1"

    # Test user created at startup so smoke tests can log in without admin access.
    TEST_USER_USERNAME = os.getenv("TEST_USER_USERNAME", "")
    TEST_USER_PASSWORD = os.getenv("TEST_USER_PASSWORD", "")

    # Reaction routing: JSON object mapping emoji → room name.
    # e.g. '{"⭐": "Starred", "📌": "Action Items"}'
    REACT_ROUTES_RAW = os.getenv("REACT_ROUTES", "{}")

    @property
    def REACT_ROUTES(self) -> dict[str, str]:
        """Parse REACT_ROUTES env var as a JSON object mapping emoji to room names."""
        try:
            return json.loads(self.REACT_ROUTES_RAW)
        except json.JSONDecodeError:
            return {}
