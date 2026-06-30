"""
config.py — Centralized configuration for Multi-Document Comparator.

Loads all settings from environment variables via python-dotenv.
Import this module anywhere in the project to access config constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── OpenRouter / LLM ────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

# ── Ollama / Embeddings ──────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "mxbai-embed-large:latest")

# ── ChromaDB ────────────────────────────────────────────────────────────────
CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME: str = "multi_doc_comparator"

# ── Retrieval Parameters ─────────────────────────────────────────────────────
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
TOP_P: float = 0.9
TOP_K: int = int(os.getenv("TOP_K", "8"))
FETCH_K: int = int(os.getenv("FETCH_K", "20"))
SIMILARITY_METRIC: str = os.getenv("SIMILARITY_METRIC", "cosine")
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS: int = 300
CHUNK_OVERLAP_TOKENS: int = 50
TIKTOKEN_ENCODING: str = "o200k_base"  # encoding for token counting (maps to GPT-4o)

# ── File Upload Limits ───────────────────────────────────────────────────────
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES: int = int(os.getenv("MAX_FILES", "10"))

# ── Paths ────────────────────────────────────────────────────────────────────
UPLOAD_DIR: Path = BASE_DIR / "uploaded_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)

# ── OCR ──────────────────────────────────────────────────────────────────────
# Pages with fewer characters than this threshold are treated as scanned
SCANNED_PAGE_CHAR_THRESHOLD: int = 100

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = BASE_DIR / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
