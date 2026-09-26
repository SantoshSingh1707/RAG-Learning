"""Streamlit entry point for the RAG question-and-answer application.

The interface is split in two on purpose: static application chrome (CSS and
layout markup) is defined in this module, while every user- or source-derived
value is rendered with native Streamlit widgets so untrusted content is never
interpolated into HTML.
"""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
    LLM_PROVIDER,
    MAX_HISTORY_MESSAGES,
    MAX_UPLOAD_BYTES,
    VECTOR_COLLECTION_NAME,
    VECTOR_STORE_DIR,
    shadowed_env_names,
)
from src.data_loader import (
    OCRUnavailableError,
    process_single_pdf,
    process_single_txt,
    split_document,
)
from src.embedding import EmbeddingManager
from src.search import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    RAGRetrieval,
    RetrievalError,
    build_chat_model,
    describe_chat_model,
    rag_enhanced,
)
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SESSION_MESSAGES = max(4, MAX_HISTORY_MESSAGES * 2)
# Which credential each provider actually reads, so an irrelevant shadowed
# variable never produces a warning about a key the app does not use.
PROVIDER_CREDENTIALS = {
    "mistral": ("MISTRAL_API_KEY",),
    "ollama": (),
}
SUGGESTED_PROMPTS = (
    (
        "Summarize the collection",
        "Give me a concise overview of the main themes in these documents.",
    ),
    (
        "Find key evidence",
        "What are the most important facts or findings in the indexed documents?",
    ),
    (
        "Compare the sources",
        "What agreements, differences, or open questions appear across the sources?",
    ),
)

st.set_page_config(
    page_title="RAG Knowledge Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# The markup below is static application chrome, not user content, and it is
# rendered through st.html (DOMPurify-sanitized) rather than
# unsafe_allow_html. All user and source data is rendered with native
# Streamlit widgets, so no untrusted value ever reaches the HTML layer.
# Keeping the CSS in one place makes the Streamlit shell feel intentional
# without adding a second frontend framework or a network request.
_APP_CSS = """
<style>
:root {
    --rag-ink: #f4f7fb;
    --rag-muted: #a9b8cc;
    --rag-subtle: #71839b;
    --rag-bg: #08111f;
    --rag-panel: #101d31;
    --rag-border: rgba(148, 177, 214, 0.16);
    --rag-cyan: #7dd3fc;
    --rag-violet: #c4b5fd;
    --rag-green: #86efac;
    --rag-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    --rag-shadow: 0 24px 70px rgba(0, 0, 0, 0.22);
}

.stApp {
    background:
        radial-gradient(circle at 78% -8%, rgba(91, 137, 255, 0.16), transparent 32rem),
        radial-gradient(circle at 10% 28%, rgba(45, 212, 191, 0.07), transparent 28rem),
        var(--rag-bg);
    color: var(--rag-ink);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

[data-testid="stHeader"] {
    background: rgba(8, 17, 31, 0.78);
    border-bottom: 1px solid rgba(148, 177, 214, 0.08);
}

.block-container {
    max-width: 1440px;
    padding: 1.35rem 2.75rem 4rem;
}

h1, h2, h3, h4 {
    letter-spacing: -0.035em;
}

p, label, .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--rag-muted);
}

.rag-hero {
    position: relative;
    isolation: isolate;
    min-height: 285px;
    overflow: hidden;
    margin: 0.15rem 0 1.35rem;
    padding: 2.4rem 2.7rem;
    border: 1px solid rgba(148, 177, 214, 0.18);
    border-radius: 28px;
    background:
        linear-gradient(125deg, rgba(20, 39, 68, 0.94), rgba(11, 24, 43, 0.9)),
        var(--rag-panel);
    box-shadow: var(--rag-shadow);
}

.rag-hero__layer {
    position: absolute;
    inset: 0;
    pointer-events: none;
}

.rag-hero__grid {
    opacity: 0.28;
    background-image:
        linear-gradient(rgba(125, 211, 252, 0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(125, 211, 252, 0.08) 1px, transparent 1px);
    background-size: 34px 34px;
    mask-image: linear-gradient(110deg, transparent 5%, black 65%, transparent 100%);
}

.rag-hero__glow {
    inset: -35% 38% 15% 38%;
    border-radius: 50%;
    background: rgba(125, 211, 252, 0.17);
    filter: blur(52px);
    animation: rag-breathe 12s ease-in-out infinite alternate;
}

.rag-hero__ring {
    width: 390px;
    height: 390px;
    right: -118px;
    top: -168px;
    border: 1px solid rgba(196, 181, 253, 0.24);
    border-radius: 50%;
    box-shadow:
        0 0 0 24px rgba(196, 181, 253, 0.035),
        0 0 0 48px rgba(125, 211, 252, 0.025);
    animation: rag-orbit 18s linear infinite;
}

.rag-hero__content {
    position: relative;
    z-index: 4;
    max-width: 730px;
}

.rag-eyebrow,
.rag-section-label,
.rag-sidebar-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--rag-cyan);
    font-size: 0.7rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    line-height: 1.2;
    text-transform: uppercase;
}

.rag-status-dot {
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 50%;
    background: var(--rag-green);
    box-shadow: 0 0 0 5px rgba(134, 239, 172, 0.1), 0 0 18px rgba(134, 239, 172, 0.65);
}

.rag-status-label {
    margin-left: 0.25rem;
    padding: 0.26rem 0.55rem;
    border: 1px solid rgba(134, 239, 172, 0.24);
    border-radius: 999px;
    color: var(--rag-green);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
}

.rag-hero h1 {
    max-width: 680px;
    margin: 1.2rem 0 0.85rem;
    color: var(--rag-ink);
    font-size: clamp(2.3rem, 5vw, 4.3rem);
    font-weight: 720;
    line-height: 0.98;
}

.rag-hero h1 span {
    color: var(--rag-cyan);
}

.rag-hero__copy {
    max-width: 610px;
    margin: 0;
    color: #c5d2e2;
    font-size: 1.02rem;
    line-height: 1.65;
}

.rag-hero__chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.45rem;
}

.rag-hero__chip {
    padding: 0.38rem 0.68rem;
    border: 1px solid rgba(148, 177, 214, 0.2);
    border-radius: 999px;
    color: #c7d7e9;
    background: rgba(125, 211, 252, 0.06);
    font-size: 0.74rem;
    font-weight: 600;
}

.rag-section-label {
    margin: 0.35rem 0 0.65rem;
    color: var(--rag-subtle);
}

[data-testid="stMetric"] {
    min-height: 106px;
    padding: 1rem 1.1rem 0.9rem;
    border: 1px solid var(--rag-border);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(21, 38, 64, 0.82), rgba(16, 29, 49, 0.82));
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    transition: transform 180ms var(--rag-ease-out), border-color 180ms ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(125, 211, 252, 0.34);
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: var(--rag-subtle);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: var(--rag-ink);
    font-size: 1.55rem;
    font-weight: 700;
}

.rag-welcome {
    position: relative;
    overflow: hidden;
    margin-top: 0.35rem;
    padding: 1.65rem 1.75rem 1.45rem;
    border: 1px solid var(--rag-border);
    border-radius: 22px;
    background: linear-gradient(140deg, rgba(21, 38, 64, 0.88), rgba(14, 28, 47, 0.9));
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.12);
}

.rag-welcome::after {
    position: absolute;
    right: -80px;
    bottom: -120px;
    width: 260px;
    height: 260px;
    border: 1px solid rgba(125, 211, 252, 0.14);
    border-radius: 50%;
    box-shadow: 0 0 0 28px rgba(125, 211, 252, 0.025), 0 0 0 56px rgba(196, 181, 253, 0.02);
    content: "";
    pointer-events: none;
}

.rag-welcome h2 {
    position: relative;
    z-index: 1;
    margin: 0.55rem 0 0.45rem;
    color: var(--rag-ink);
    font-size: 1.65rem;
}

.rag-welcome p {
    position: relative;
    z-index: 1;
    max-width: 650px;
    margin: 0;
    line-height: 1.6;
}

.rag-steps {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin-top: 1.35rem;
}

.rag-step {
    min-height: 112px;
    padding: 0.9rem;
    border: 1px solid rgba(148, 177, 214, 0.13);
    border-radius: 14px;
    background: rgba(8, 17, 31, 0.28);
}

.rag-step__number {
    display: block;
    margin-bottom: 0.55rem;
    color: var(--rag-violet);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.1em;
}

.rag-step strong {
    display: block;
    color: var(--rag-ink);
    font-size: 0.86rem;
}

.rag-step small {
    display: block;
    margin-top: 0.35rem;
    color: var(--rag-subtle);
    font-size: 0.74rem;
    line-height: 1.45;
}

.rag-conversation-header {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.55rem 0 0.8rem;
}

.rag-conversation-header h2 {
    margin: 0;
    font-size: 1.35rem;
}

.rag-conversation-header p {
    margin: 0.3rem 0 0;
    font-size: 0.82rem;
}

[data-testid="stChatMessage"] {
    padding: 0.8rem 1rem;
    border: 1px solid rgba(148, 177, 214, 0.1);
    border-radius: 18px;
    background: rgba(16, 29, 49, 0.54);
}

[data-testid="stChatMessage"]:has([data-testid="stAvatar"] img[alt*="assistant"]) {
    border-color: rgba(125, 211, 252, 0.18);
    background: linear-gradient(135deg, rgba(21, 38, 64, 0.8), rgba(16, 29, 49, 0.64));
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #dce7f3;
    line-height: 1.65;
}

[data-testid="stChatInput"] {
    border: 1px solid rgba(125, 211, 252, 0.24) !important;
    border-radius: 18px !important;
    background: rgba(16, 29, 49, 0.92) !important;
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.16);
}

[data-testid="stChatInput"] textarea {
    color: var(--rag-ink) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--rag-subtle) !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 177, 214, 0.12);
    background: #0b1728;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.25rem 1rem 2rem;
}

.rag-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0.15rem 0 1.35rem;
}

.rag-brand__mark {
    display: grid;
    width: 2.2rem;
    height: 2.2rem;
    place-items: center;
    border: 1px solid rgba(125, 211, 252, 0.35);
    border-radius: 12px;
    color: var(--rag-bg);
    background: linear-gradient(145deg, var(--rag-cyan), var(--rag-violet));
    font-size: 1.1rem;
    font-weight: 850;
    box-shadow: 0 8px 24px rgba(125, 211, 252, 0.18);
}

.rag-brand__name {
    color: var(--rag-ink);
    font-size: 0.92rem;
    font-weight: 750;
    letter-spacing: -0.01em;
}

.rag-brand__sub {
    margin-top: 0.1rem;
    color: var(--rag-subtle);
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.rag-sidebar-label {
    margin: 0 0 0.75rem;
    color: var(--rag-subtle);
    font-size: 0.64rem;
}

.rag-sidebar-divider {
    height: 1px;
    margin: 1.15rem 0;
    background: rgba(148, 177, 214, 0.12);
}

.rag-index-status {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.7rem 0.8rem;
    border: 1px solid rgba(134, 239, 172, 0.18);
    border-radius: 13px;
    color: #c5f6d3;
    background: rgba(134, 239, 172, 0.06);
    font-size: 0.76rem;
}

.rag-index-status__dot {
    width: 0.45rem;
    height: 0.45rem;
    flex: 0 0 auto;
    border-radius: 50%;
    background: var(--rag-green);
    box-shadow: 0 0 12px rgba(134, 239, 172, 0.6);
}

.rag-footer-note {
    margin-top: 1.4rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(148, 177, 214, 0.1);
    color: var(--rag-subtle);
    font-size: 0.68rem;
    line-height: 1.5;
}

button {
    transition: transform 160ms var(--rag-ease-out), background-color 160ms ease,
        border-color 160ms ease, color 160ms ease;
}

@media (hover: hover) and (pointer: fine) {
    button:hover {
        transform: translateY(-1px);
    }

    button:not(:focus-visible):active {
        transform: scale(0.98);
    }
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
[role="button"]:focus-visible {
    outline: 3px solid var(--rag-cyan);
    outline-offset: 2px;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(148, 177, 214, 0.12);
    border-radius: 15px;
    background: rgba(16, 29, 49, 0.45);
}

[data-testid="stExpander"] summary {
    color: #d6e2ef;
    font-weight: 650;
}

.rag-how-it-works {
    margin: 1.25rem 0 0.8rem;
    padding: 1rem 1.1rem;
    border: 1px dashed rgba(148, 177, 214, 0.18);
    border-radius: 16px;
    color: var(--rag-muted);
    background: rgba(16, 29, 49, 0.32);
    font-size: 0.8rem;
    line-height: 1.6;
}

.rag-how-it-works strong {
    color: var(--rag-ink);
}

@keyframes rag-breathe {
    from { transform: translate3d(-2%, 0, 0) scale(0.98); opacity: 0.55; }
    to { transform: translate3d(3%, 2%, 0) scale(1.04); opacity: 0.85; }
}

@keyframes rag-orbit {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@media (max-width: 900px) {
    .block-container {
        padding: 1rem 1.25rem 3rem;
    }

    .rag-hero {
        min-height: 255px;
        padding: 1.8rem 1.5rem;
        border-radius: 22px;
    }

    .rag-hero h1 {
        font-size: clamp(2.15rem, 9vw, 3.3rem);
    }

    .rag-steps {
        grid-template-columns: 1fr;
    }

    .rag-step {
        min-height: 0;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }

    .rag-hero__glow,
    .rag-hero__ring {
        animation: none !important;
        transform: none !important;
    }
}
</style>
"""


def _inject_styles() -> None:
    """Apply the static visual system for the Streamlit shell."""
    st.html(_APP_CSS)


def _safe_display_name(name: str) -> str:
    """Keep only a filename component for temporary-file and UI use."""
    candidate = Path(name).name
    candidate = re.sub(r"[^A-Za-z0-9._() -]+", "_", candidate).strip()
    return candidate or "uploaded-document"


def _format_count(value: Any) -> str:
    """Format an index count without assuming it is already an integer."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _build_download_text(answer: str, sources: list[dict[str, Any]]) -> str:
    lines = "\n".join(
        f"- {source.get('source_file', 'unknown')} (page {source.get('page', 'unknown')})"
        for source in sources
    )
    return f"{answer}\n\nSources:\n{lines}" if lines else answer


def _render_source_item(source: dict[str, Any], index: int) -> None:
    """Render one evidence excerpt as a compact, readable card."""
    try:
        score = float(source.get("similarity_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if not np.isfinite(score):
        score = 0.0
    score = max(0.0, min(1.0, score))

    source_name = str(source.get("source_file") or "Unknown source")
    source_name = source_name[:180]
    page = source.get("page")
    location = f"Page {page}" if page not in (None, "", "unknown") else "Location unavailable"
    # Already bounded by _enhanced_sources when the source dict was built, so
    # this needs no further truncation.
    content = str(source.get("content") or "")

    with st.container(border=True):
        st.caption(f"SOURCE {index:02d}")
        meta_col, score_col = st.columns([3.2, 1], gap="small")
        with meta_col:
            st.text(source_name)
            st.caption(location)
        with score_col:
            st.caption("MATCH")
            st.write(f"**{score:.1%}**")
        st.progress(score)
        if content:
            st.text(content)
        else:
            st.caption("No preview text was returned for this match.")


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Evidence · {len(sources)} sources", expanded=False):
        st.caption("The answer is grounded in the excerpts below. Open this panel to audit it.")
        for index, source in enumerate(sources, start=1):
            _render_source_item(source, index)


def _render_context(context: str) -> None:
    if not context:
        return
    with st.expander("Retrieved context", expanded=False):
        st.caption("Untrusted source material used to ground this response.")
        st.text(context)


def _render_assistant_payload(
    answer: str,
    sources: list[dict[str, Any]],
    context: str,
    message_id: str,
    *,
    show_context: bool,
) -> None:
    """Render an assistant response and its audit/download affordances."""
    st.markdown(answer)
    if sources:
        _render_sources(sources)
        st.download_button(
            "Download answer",
            data=_build_download_text(answer, sources),
            file_name=f"rag_answer_{message_id}.txt",
            mime="text/plain",
            key=f"download_{message_id}",
            use_container_width=True,
        )
    if show_context:
        _render_context(context)
    if not sources:
        st.caption("No source excerpts cleared the current relevance gate.")


def _render_welcome() -> None:
    """Show a useful first-run state with concrete starting points."""
    st.html(
        """
        <section class="rag-welcome" aria-label="Getting started with your knowledge workspace">
          <div class="rag-section-label">START HERE</div>
          <h2>Ask like a researcher.</h2>
          <p>Turn a collection of documents into a focused answer. Ask a broad question, then narrow in with source filters and evidence.</p>
          <div class="rag-steps">
            <div class="rag-step"><span class="rag-step__number">01 / ASK</span><strong>Pose a precise question</strong><small>Use a topic, claim, or decision you need to understand.</small></div>
            <div class="rag-step"><span class="rag-step__number">02 / TRACE</span><strong>Inspect the evidence</strong><small>Every answer opens with the source excerpts behind it.</small></div>
            <div class="rag-step"><span class="rag-step__number">03 / ITERATE</span><strong>Refine the signal</strong><small>Filter by source or raise the similarity gate to reduce noise.</small></div>
          </div>
        </section>
        """
    )
    st.html('<div class="rag-section-label">SUGGESTED PROMPTS</div>')
    prompt_columns = st.columns(3)
    for column, (label, prompt) in zip(prompt_columns, SUGGESTED_PROMPTS, strict=False):
        if column.button(label, key=f"suggestion_{label}", use_container_width=True):
            st.session_state["pending_query"] = prompt
            st.rerun()


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def load_rag_components() -> tuple[Any, Any, VectorStore, EmbeddingManager] | None:
    """Load expensive resources once per Streamlit process."""
    try:
        embedding_manager = EmbeddingManager()
        vectorstore = VectorStore(
            collection_name=VECTOR_COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        )
        retriever = RAGRetrieval(vectorstore, embedding_manager)
        llm = build_chat_model()
        return retriever, llm, vectorstore, embedding_manager
    except Exception:
        logger.exception("Unable to initialize RAG components")
        return None


def _process_upload(
    uploaded_file: Any, embedding_manager: EmbeddingManager, vectorstore: VectorStore
) -> int:
    """Process an uploaded file in a temporary directory and return chunk count."""
    raw_name = str(getattr(uploaded_file, "name", "uploaded-document"))
    display_name = _safe_display_name(raw_name)
    file_bytes = uploaded_file.getbuffer()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"The upload is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    with tempfile.TemporaryDirectory(prefix="rag-upload-") as temporary_dir:
        temporary_path = Path(temporary_dir) / display_name
        temporary_path.write_bytes(bytes(file_bytes))
        suffix = temporary_path.suffix.lower()
        if suffix == ".pdf":
            documents = process_single_pdf(str(temporary_path), display_name=display_name)
        elif suffix == ".txt":
            documents = process_single_txt(str(temporary_path), display_name=display_name)
        else:
            raise ValueError("Only PDF and TXT files are supported.")

    chunks = split_document(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    if not chunks:
        return 0

    source_ids = {str(chunk.metadata.get("source_id", "")) for chunk in chunks}
    if len(source_ids) != 1 or "" in source_ids:
        raise ValueError("Uploaded documents must resolve to one stable source.")
    source_id = source_ids.pop()
    with tempfile.TemporaryDirectory(prefix="rag-upload-embedding-") as staging_directory:
        staging_root = Path(staging_directory)
        staged_batches = []
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
            embeddings = embedding_manager.generate_embeddings(
                [chunk.page_content for chunk in batch],
                is_query=False,
                show_progress_bar=True,
            )
            embedding_path = staging_root / f"batch-{start:012d}.npy"
            np.save(embedding_path, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
            staged_batches.append((batch, embedding_path))

        # Delete the previous version only after all new embeddings exist.
        vectorstore.delete_sources([source_id])
        for batch, embedding_path in staged_batches:
            vectorstore.add_documents(batch, np.load(embedding_path, allow_pickle=False))
    return len(chunks)


def _read_source_catalog(vectorstore: VectorStore) -> tuple[list[dict[str, Any]], bool]:
    try:
        return vectorstore.get_source_catalog(), True
    except Exception:
        logger.exception("Unable to read source catalog")
        return [], False


def _render_sidebar(
    vectorstore: VectorStore,
    embedding_manager: EmbeddingManager,
    source_catalog: list[dict[str, Any]],
    source_catalog_available: bool,
) -> tuple[int, float, bool, list[str]]:
    """Render the control room and return the active retrieval settings."""
    source_labels = {
        item["source_id"]: item.get("source_file", item["source_id"]) for item in source_catalog
    }

    with st.sidebar:
        st.html(
            """
            <div class="rag-brand">
              <div class="rag-brand__mark" aria-hidden="true">◈</div>
              <div><div class="rag-brand__name">RAG workspace</div><div class="rag-brand__sub">grounded answers</div></div>
            </div>
            """
        )

        # Report every setting this shell is overriding. A shadowed OLLAMA_MODEL
        # or RAG_TOP_K is as invisible as a shadowed API key and just as likely
        # to make an .env edit look like it did nothing.
        if shadowed_env_names:
            shown = ", ".join(shadowed_env_names)
            verb = "is" if len(shadowed_env_names) == 1 else "are"
            st.warning(
                f"{shown} {verb} set in this shell and {verb} overriding the .env file, "
                f"so edits there have no effect. Clear with: "
                f"Remove-Item Env:{shadowed_env_names[0]}"
            )

        if source_catalog_available:
            st.html(
                '<div class="rag-index-status"><span class="rag-index-status__dot"></span>Cosine index online</div>'
            )
        else:
            st.error("Source catalog unavailable")

        provider_label = describe_chat_model()
        scope = "Local model" if LLM_PROVIDER == "ollama" else "Hosted model"
        st.html(
            f'<div class="rag-index-status"><span class="rag-index-status__dot"></span>'
            f"{scope}: {provider_label}</div>"
        )

        st.html('<div class="rag-sidebar-divider"></div>')
        st.html('<div class="rag-sidebar-label">RETRIEVAL CONTROLS</div>')
        top_k = st.slider(
            "Documents to retrieve",
            1,
            10,
            DEFAULT_TOP_K,
            help="How many evidence chunks to consider before the answer is generated.",
        )
        score_threshold = st.slider(
            "Similarity gate",
            0.0,
            1.0,
            DEFAULT_MIN_SCORE,
            0.05,
            help="Hide matches below this cosine similarity score.",
        )
        return_context = st.checkbox(
            "Show retrieved context",
            value=True,
            help="Display the untrusted source text used for the answer.",
        )

        st.html('<div class="rag-sidebar-divider"></div>')
        with st.expander("＋  Add a document", expanded=False):
            st.caption("PDF and TXT files are chunked, embedded, and added to the active index.")
            uploaded_file = st.file_uploader(
                "PDF or TXT file",
                type=["pdf", "txt"],
                help="Maximum upload size is configured by RAG_MAX_UPLOAD_BYTES.",
            )
            if uploaded_file and st.button(
                "Process and add",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.status("Processing document…", expanded=True):
                        chunk_count = _process_upload(uploaded_file, embedding_manager, vectorstore)
                    if chunk_count:
                        display_name = _safe_display_name(str(uploaded_file.name))
                        st.session_state["flash"] = (
                            f"Added {display_name} · {_format_count(chunk_count)} chunks"
                        )
                        st.rerun()
                    else:
                        st.error("No extractable text was found.")
                except OCRUnavailableError:
                    logger.exception("OCR is unavailable for uploaded document")
                    st.error("OCR is not available. Run `uv sync` and retry.")
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    logger.exception("Unable to process uploaded document")
                    st.error("The document could not be processed. Check the server log.")

        st.html('<div class="rag-sidebar-divider"></div>')
        st.html('<div class="rag-sidebar-label">SEARCH SCOPE</div>')
        selected_sources = st.multiselect(
            "Filter by source",
            options=list(source_labels),
            default=[],
            format_func=lambda source_id: source_labels[source_id],
            help="Leave empty to search the full index. Select one or more sources to narrow the search.",
        )

        if source_labels:
            with st.expander("Manage indexed sources", expanded=False):
                doc_to_remove = st.selectbox(
                    "Select a source",
                    ["", *source_labels],
                    format_func=lambda source_id: source_labels.get(source_id, source_id),
                )
                if doc_to_remove and st.button(
                    "Remove selected source",
                    use_container_width=True,
                ):
                    st.session_state["confirm_remove"] = doc_to_remove

                confirmation = st.session_state.get("confirm_remove")
                if confirmation == doc_to_remove:
                    st.warning("This removes the source and its chunks from the active index.")
                    confirm_col, cancel_col = st.columns(2)
                    if confirm_col.button("Confirm", type="primary", use_container_width=True):
                        try:
                            with st.status("Removing source…"):
                                removed = vectorstore.remove_source(doc_to_remove)
                            if removed:
                                st.session_state["flash"] = "Source removed from the active index."
                                st.session_state.pop("confirm_remove", None)
                                st.rerun()
                            else:
                                st.error("No matching source data was found.")
                        except Exception:
                            logger.exception("Unable to remove source")
                            st.error("The source could not be removed.")
                    if cancel_col.button("Cancel", use_container_width=True):
                        st.session_state.pop("confirm_remove", None)
                        st.rerun()

        st.html('<div class="rag-sidebar-divider"></div>')
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state.pop("pending_query", None)
            st.rerun()
        st.html(
            '<div class="rag-footer-note">Answers are generated from the active cosine index. Source text is treated as untrusted reference data, never as instructions.</div>'
        )

    return top_k, score_threshold, return_context, selected_sources


def _render_hero() -> None:
    """Render the static product hero; dynamic metrics stay native Streamlit widgets."""
    st.html(
        """
        <section class="rag-hero" role="region" aria-label="RAG knowledge workspace">
          <div class="rag-hero__layer rag-hero__grid" aria-hidden="true"></div>
          <div class="rag-hero__layer rag-hero__glow" aria-hidden="true"></div>
          <div class="rag-hero__layer rag-hero__ring" aria-hidden="true"></div>
          <div class="rag-hero__content">
            <div class="rag-eyebrow"><span class="rag-status-dot"></span> Knowledge workspace <span class="rag-status-label">Ready</span></div>
            <h1>Find the signal<br><span>in your documents.</span></h1>
            <p class="rag-hero__copy">A focused space to ask better questions, trace the evidence, and move from a broad idea to a grounded answer.</p>
            <div class="rag-hero__chips"><span class="rag-hero__chip">Cosine retrieval</span><span class="rag-hero__chip">Source-aware answers</span><span class="rag-hero__chip">Private by design</span></div>
          </div>
        </section>
        """
    )


def _render_index_snapshot(
    vectorstore: VectorStore,
    source_catalog: list[dict[str, Any]],
    top_k: int,
    score_threshold: float,
) -> None:
    """Render lightweight index and retrieval health metrics."""
    try:
        indexed_chunks: Any = vectorstore.collection.count()
    except Exception:
        logger.exception("Unable to read index count")
        indexed_chunks = None

    st.html('<div class="rag-section-label">INDEX SNAPSHOT</div>')
    metric_columns = st.columns(4)
    with metric_columns[0]:
        st.metric("Indexed chunks", _format_count(indexed_chunks))
    with metric_columns[1]:
        st.metric("Sources", _format_count(len(source_catalog)))
    with metric_columns[2]:
        st.metric("Evidence budget", f"{top_k} docs")
    with metric_columns[3]:
        st.metric("Relevance gate", f"{score_threshold:.0%}")
    st.caption(
        "Active collection: cosine similarity index · Context is bounded before it reaches the model."
    )


def _handle_query(
    query: str,
    *,
    retriever: RAGRetrieval,
    llm: Any,
    top_k: int,
    score_threshold: float,
    return_context: bool,
    selected_sources: list[str],
) -> None:
    """Run one grounded question and append its bounded conversation turn."""
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-MAX_HISTORY_MESSAGES * 2 :]
    ]
    st.session_state.messages.append({"role": "user", "content": query, "id": str(uuid.uuid4())})
    with st.chat_message("user", avatar="user"):
        st.text(query)

    with st.chat_message("assistant", avatar="assistant"):
        try:
            with st.spinner("Searching the index and grounding the answer…"):
                result = rag_enhanced(
                    query=query,
                    retriever=retriever,
                    llm=llm,
                    top_k=top_k,
                    min_score=score_threshold,
                    return_context=return_context,
                    source_filter=selected_sources or None,
                    history=history,
                )
            answer = str(result.get("answer") or "No response was generated.")
            sources = list(result.get("sources") or [])
            context = str(result.get("context") or "")
            message_id = str(uuid.uuid4())
            _render_assistant_payload(
                answer,
                sources,
                context,
                message_id,
                show_context=return_context,
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "context": context,
                    "id": message_id,
                }
            )
            st.session_state.messages = st.session_state.messages[-MAX_SESSION_MESSAGES:]
        except RetrievalError:
            logger.exception("Document retrieval failed")
            st.error("Document retrieval failed. Check the vector-store logs.")
        except ProviderUnavailableError as error:
            logger.error("Chat provider unavailable: %s", error)
            st.error(str(error))
        except ProviderAuthError as error:
            logger.exception("Chat provider rejected the credentials")
            credential = next(iter(PROVIDER_CREDENTIALS.get(LLM_PROVIDER, ())), None)
            if credential:
                st.error(
                    f"The chat provider rejected {credential}. Update it in .env "
                    "and restart the app, then ask again."
                )
            else:
                # A local provider has no credential to fix, so the server-side
                # message is the only actionable detail available.
                st.error(f"{error} Check the local provider logs and restart the app.")
        except ProviderRateLimitError as error:
            logger.warning("Chat provider rate limit reached: %s", error)
            st.warning("The chat provider is rate limiting requests. Wait a moment and retry.")
        except ConnectionError as error:
            logger.warning("Chat provider unreachable: %s", error)
            st.warning(
                "Could not reach the chat provider. Check your network connection and retry."
            )
        except Exception:
            logger.exception("Query processing failed")
            st.error("The answer could not be generated. Check the server log.")


def main() -> None:
    """Run the Streamlit workspace."""
    _inject_styles()
    components = load_rag_components()
    if components is None:
        st.error(
            "RAG components could not be initialized. Check the API key, model revision, "
            "and vector-store configuration in the setup documentation."
        )
        st.stop()

    retriever, llm, vectorstore, embedding_manager = components
    if "messages" not in st.session_state:
        st.session_state.messages = []

    source_catalog, source_catalog_available = _read_source_catalog(vectorstore)
    top_k, score_threshold, return_context, selected_sources = _render_sidebar(
        vectorstore,
        embedding_manager,
        source_catalog,
        source_catalog_available,
    )

    if flash := st.session_state.pop("flash", None):
        st.sidebar.success(flash)

    _render_hero()
    _render_index_snapshot(vectorstore, source_catalog, top_k, score_threshold)

    st.html(
        """
        <div class="rag-conversation-header">
          <div><div class="rag-section-label">LIVE CONVERSATION</div><h2>Ask the collection</h2><p>Every response keeps its evidence close.</p></div>
        </div>
        """
    )

    chat_query = st.chat_input("Ask a question about your documents…")
    pending_query = st.session_state.get("pending_query")
    has_submitted_query = bool(chat_query or pending_query)

    if not st.session_state.messages and not has_submitted_query:
        _render_welcome()
    else:
        for message in st.session_state.messages:
            with st.chat_message(
                message["role"],
                avatar="user" if message["role"] == "user" else "assistant",
            ):
                if message["role"] == "user":
                    st.text(message["content"])
                else:
                    _render_assistant_payload(
                        message["content"],
                        list(message.get("sources") or []),
                        str(message.get("context") or ""),
                        str(message.get("id") or uuid.uuid4()),
                        show_context=return_context,
                    )

        st.html(
            '<div class="rag-how-it-works"><strong>How it works:</strong> your question is embedded, matched against the cosine index, filtered by the selected sources, and answered with the strongest excerpts. Retrieved text is reference material—not executable instructions.</div>'
        )

    query = chat_query or st.session_state.pop("pending_query", None)
    if query:
        _handle_query(
            str(query),
            retriever=retriever,
            llm=llm,
            top_k=top_k,
            score_threshold=score_threshold,
            return_context=return_context,
            selected_sources=selected_sources,
        )


if __name__ == "__main__":
    main()
