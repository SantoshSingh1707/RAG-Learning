"""Central configuration for the RAG application.

All filesystem paths are resolved relative to the project root so the CLI and
Streamlit application behave consistently regardless of the current working
directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Deliberately no logging.basicConfig here: this module is imported by libraries
# and tests, and configuring global logging as an import side effect would
# override whatever the entry point (app.py, ingest_data.py,
# migrate_legacy_index.py) chose. Warnings still reach stderr via logging's
# last-resort handler.
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
# Environment variables keep precedence over the file so that CI, containers,
# and shell sessions can override local defaults. That precedence is silent by
# design, so shadowed_credentials below reports any credential a process
# variable is overriding: an edited .env that never takes effect is otherwise
# very hard to diagnose. shadowed_env_names below reports every such setting.
load_dotenv(ENV_FILE, override=False)


# Every environment variable this module reads. load_dotenv keeps process
# variables authoritative, so if one of these is set in the shell it silently
# wins over the .env file and an edit there appears to do nothing. Reporting is
# driven by this list rather than by a "looks like a secret" heuristic: a
# shadowed OLLAMA_MODEL is just as confusing as a shadowed API key, while an
# unrelated override such as PYTHONPATH is noise we should not report.
APP_SETTING_NAMES = frozenset(
    {
        "EMBEDDING_MODEL",
        "EMBEDDING_MODEL_REVISION",
        "MISTRAL_MAX_RETRIES",
        "MISTRAL_MODEL",
        "MISTRAL_TIMEOUT_SECONDS",
        "OLLAMA_BASE_URL",
        "OLLAMA_KEEP_ALIVE",
        "OLLAMA_MODEL",
        "OLLAMA_NUM_CTX",
        "OLLAMA_NUM_PREDICT",
        "RAG_CHUNK_OVERLAP",
        "RAG_CHUNK_SIZE",
        "RAG_COLLECTION_NAME",
        "RAG_EMBEDDING_BATCH_SIZE",
        "RAG_LLM_PROVIDER",
        "RAG_MAX_CONTEXT_CHARS",
        "RAG_MAX_HISTORY_MESSAGES",
        "RAG_MAX_UPLOAD_BYTES",
        "RAG_MIN_SCORE",
        "RAG_OCR_DPI",
        "RAG_TOP_K",
    }
)

# Secrets are reported even when they are not part of APP_SETTING_NAMES, since
# any provider may read a credential this module does not know about.
_SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _detect_shadowed_env_names() -> tuple[str, ...]:
    """Return settings whose .env value is overridden by this process.

    Both credential and non-credential settings are reported: a shell variable
    that quietly overrides .env is the failure mode this exists to make
    visible, and it is equally invisible for OLLAMA_MODEL as for an API key.
    """
    if not ENV_FILE.is_file():
        return ()
    try:
        file_values = dotenv_values(ENV_FILE)
    except Exception:
        logger.debug("Unable to read %s for override diagnostics", ENV_FILE, exc_info=True)
        return ()

    shadowed: list[str] = []
    for name, file_value in file_values.items():
        if not file_value:
            continue
        process_value = os.environ.get(name)
        if process_value is None or process_value == file_value:
            continue
        is_relevant = name in APP_SETTING_NAMES or any(
            marker in name.upper() for marker in _SECRET_NAME_MARKERS
        )
        if is_relevant:
            shadowed.append(name)
            logger.warning(
                "Environment variable %s is set in this process and overrides the value in %s. "
                "The .env value is being ignored. Run 'Remove-Item Env:\\%s' to use the file value.",
                name,
                ENV_FILE.name,
                name,
            )
    return tuple(shadowed)


shadowed_env_names = _detect_shadowed_env_names()

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
TEXT_DIR = DATA_DIR / "textfiles"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# v2 uses an explicit cosine index. The old collection used Chroma's default
# squared-L2 index and must not be modified in place.
VECTOR_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "rag_documents_v2")
LEGACY_VECTOR_COLLECTION_NAME = "pdf_documents"

DEFAULT_EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
PINNED_EMBEDDING_MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
# Pin the default model revision. Custom models are left unpinned unless the
# caller explicitly supplies a revision.
EMBEDDING_MODEL_REVISION = os.getenv("EMBEDDING_MODEL_REVISION") or (
    PINNED_EMBEDDING_MODEL_REVISION if EMBEDDING_MODEL == DEFAULT_EMBEDDING_MODEL else None
)
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-2506")
MISTRAL_TIMEOUT_SECONDS = int(os.getenv("MISTRAL_TIMEOUT_SECONDS", "45"))
MISTRAL_MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "2"))

# Chat provider. "ollama" runs fully locally through the Ollama daemon; "mistral"
# uses the hosted API and requires MISTRAL_API_KEY. Set RAG_LLM_PROVIDER in .env
# to switch, so an existing Mistral setup keeps working unchanged.
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "mistral").strip().lower()
SUPPORTED_LLM_PROVIDERS = ("mistral", "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
# The embedding model and the chat model share the same GPU here, so the context
# window is budgeted against what is left after the embeddings are resident.
# A 6 GB card running a 7B Q4 model alongside MiniLM exhausts VRAM above roughly
# 4k tokens, so keep this conservative and raise it only on a larger GPU.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
# Ollama keeps the model resident for this long after the last request, which
# removes the ~40 s cold start on every Streamlit rerun.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Keep retrieved context inside the local context window. At roughly 3.5 chars
# per token this leaves room for the system prompt, the question, and the reply.
MAX_CONTEXT_CHARS = int(
    os.getenv("RAG_MAX_CONTEXT_CHARS", "12000" if LLM_PROVIDER == "ollama" else "120000")
)

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
# One home for the embedding batch size: bulk ingestion and the app's upload
# path must stage the same number of chunks per embedding call, otherwise the
# two paths would drift into different memory footprints on the same GPU.
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "128"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
DEFAULT_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))
MAX_HISTORY_MESSAGES = int(os.getenv("RAG_MAX_HISTORY_MESSAGES", "8"))
MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
OCR_DPI = int(os.getenv("RAG_OCR_DPI", "150"))
