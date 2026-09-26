"""Behavioural coverage for the Streamlit entry point.

app.py is the one module with no direct tests, and it is where the upload
path, the conversation loop, and every provider error message actually live.
That code is reachable from a test, but only once the expensive
collaborators are replaced: the real EmbeddingManager pulls a model, the
real VectorStore opens a multi-gigabyte index, and build_chat_model reaches
out to Ollama. None of that belongs in CI.

Seam note. AppTest executes app.py into a fresh ``__main__``, so patching the
``app`` module would not reach the running script. Patching the modules
app.py imports *from* does work, because the exec-time import re-reads the
attribute out of ``sys.modules``. These tests patch ``src.embedding``,
``src.vector_store``, ``src.search``, and ``src.config``, which leaves
app.py's own ``load_rag_components`` and ``main()`` running for real. That
exercises more production code than stubbing the loader out would.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app as app_module
import src.config as config_module
import src.embedding as embedding_module
import src.search as search_module
import src.vector_store as vector_store_module

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
EMBEDDING_DIMENSIONS = 4
CHAT_PROMPT = "Ask a question about your documents…"

DEFAULT_SOURCES = [
    {"source_id": "bulk:pdf:alpha.pdf", "file_type": "pdf", "source_file": "alpha.pdf"},
    {"source_id": "bulk:txt:notes.txt", "file_type": "txt", "source_file": "notes.txt"},
]

DEFAULT_RAG_RESULT = {
    "answer": "The transformer paper introduces multi-head attention.",
    "sources": [
        {
            "source_id": "bulk:pdf:alpha.pdf",
            "source_file": "alpha.pdf",
            "page": "3",
            "similarity_score": 0.82,
            "content": "We propose a new simple network architecture.",
        }
    ],
    "context": "CONTEXT-BODY: multi-head attention allows the model to attend to "
    "different positions simultaneously.",
}


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeCollection:
    """Stands in for the Chroma collection handle."""

    def __init__(self, chunk_count: int) -> None:
        self._chunk_count = chunk_count

    def count(self) -> int:
        return self._chunk_count


class FakeVectorStore:
    """Records the mutating calls app.py makes, and nothing else."""

    def __init__(
        self,
        sources: list[dict[str, Any]] | None = None,
        chunk_count: int = 0,
        *,
        catalog_error: bool = False,
    ) -> None:
        self.collection = FakeCollection(chunk_count)
        self._sources = list(DEFAULT_SOURCES if sources is None else sources)
        self._catalog_error = catalog_error
        self.removed: list[str] = []
        self.deleted: list[str] = []
        self.added: list[tuple[int, int]] = []

    def get_source_catalog(self) -> list[dict[str, Any]]:
        if self._catalog_error:
            raise RuntimeError("catalog unavailable")
        return list(self._sources)

    def remove_source(self, source_id: str) -> int:
        self.removed.append(source_id)
        return 3

    def delete_sources(self, source_ids: list[str]) -> None:
        self.deleted.extend(source_ids)

    def add_documents(self, documents: list[Any], embeddings: Any) -> int:
        self.added.append((len(documents), len(embeddings)))
        return len(documents)


class FakeEmbeddingManager:
    """Returns fixed-width vectors and records every call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_embeddings(
        self,
        texts: list[str],
        is_query: bool = False,
        show_progress_bar: bool | None = None,
    ) -> list[list[float]]:
        self.calls.append({"count": len(texts), "is_query": is_query})
        return [[0.1] * EMBEDDING_DIMENSIONS for _ in texts]


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(str(prompt))
        return SimpleNamespace(content="Answer from the fake provider.")


class FakeUploadedFile:
    """The slice of the Streamlit UploadedFile surface that _process_upload uses."""

    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._buffer = bytearray(payload)

    def getbuffer(self) -> memoryview:
        return memoryview(self._buffer)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class Harness(SimpleNamespace):
    """The fakes plus the AppTest, so a test can assert on both."""


def build_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sources: list[dict[str, Any]] | None = None,
    chunk_count: int = 4321,
    catalog_error: bool = False,
    rag_result: dict[str, Any] | None = None,
    rag_error: BaseException | None = None,
    shadowed: tuple[str, ...] = (),
    load_error: bool = False,
) -> Harness:
    """Patch every external collaborator, run the app once, and return both."""
    vectorstore = FakeVectorStore(
        sources=sources, chunk_count=chunk_count, catalog_error=catalog_error
    )
    embedding_manager = FakeEmbeddingManager()
    llm = FakeLlm()
    retriever = SimpleNamespace(name="fake-retriever")
    queries: list[dict[str, Any]] = []

    monkeypatch.setattr(vector_store_module, "VectorStore", lambda *a, **k: vectorstore)
    monkeypatch.setattr(
        embedding_module,
        "EmbeddingManager",
        (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model unavailable")))
        if load_error
        else (lambda *a, **k: embedding_manager),
    )
    monkeypatch.setattr(search_module, "RAGRetrieval", lambda *a, **k: retriever)
    monkeypatch.setattr(search_module, "build_chat_model", lambda *a, **k: llm)
    # shadowed_env_names is a module-level value, not a function, so it has to be
    # replaced with a sequence. A callable here is truthy and then gets joined.
    monkeypatch.setattr(config_module, "shadowed_env_names", list(shadowed))

    # load_rag_components is @st.cache_resource. Streamlit keys that cache on
    # __module__ + __qualname__, and every AppTest run execs app.py into a fresh
    # __main__, so all runs in this process share one cache entry. Without this,
    # the first test's fakes would be returned to every later test.
    monkeypatch.setattr(st, "cache_resource", lambda **kwargs: lambda function: function)

    if rag_error is not None:

        def _raise(**kwargs: Any) -> dict[str, Any]:
            raise rag_error

        monkeypatch.setattr(search_module, "rag_enhanced", _raise)
    else:
        payload = DEFAULT_RAG_RESULT if rag_result is None else rag_result

        def _succeed(**kwargs: Any) -> dict[str, Any]:
            queries.append(kwargs)
            return payload

        monkeypatch.setattr(search_module, "rag_enhanced", _succeed)

    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    return Harness(
        at=at,
        vectorstore=vectorstore,
        embedding_manager=embedding_manager,
        llm=llm,
        queries=queries,
    )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    return build_app(monkeypatch)


def metric_map(at: AppTest) -> dict[str, str]:
    return {metric.label: str(metric.value) for metric in at.metric}


def error_texts(at: AppTest) -> list[str]:
    return [str(element.value) for element in at.error]


def warning_texts(scope: Any) -> list[str]:
    return [str(element.value) for element in scope.warning]


def sidebar_button(harness: Harness, label: str) -> Any:
    """Click a sidebar button by its visible label."""
    matches = [button for button in harness.at.sidebar.button if button.label == label]
    assert matches, f"sidebar has no {label!r} button"
    return matches[0]


def surfaced_texts(harness: Harness) -> str:
    """Every user-facing error and warning, main body and sidebar alike."""
    at = harness.at
    return " ".join(
        [
            *(str(element.value) for element in at.error),
            *(str(element.value) for element in at.warning),
            *(str(element.value) for element in at.sidebar.error),
            *(str(element.value) for element in at.sidebar.warning),
        ]
    )


def chat_text(harness: Harness) -> str:
    """All rendered text inside the chat transcript."""
    return " ".join(
        str(block.value) for message in harness.at.chat_message for block in message.markdown
    )


# --------------------------------------------------------------------------
# Boot and chrome
# --------------------------------------------------------------------------


def test_app_boots_without_exceptions(harness: Harness) -> None:
    assert harness.at.exception == []


def test_index_snapshot_reports_live_index_size(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = build_app(monkeypatch, chunk_count=126872, sources=DEFAULT_SOURCES * 3)
    metrics = metric_map(harness.at)
    assert metrics["Indexed chunks"] == "126,872"
    assert metrics["Sources"] == "6"


def test_index_snapshot_survives_a_failing_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken collection must not take the whole page down."""

    class ExplodingCollection:
        def count(self) -> int:
            raise RuntimeError("collection is corrupt")

    harness = build_app(monkeypatch)
    harness.vectorstore.collection = ExplodingCollection()
    harness.at.run()
    assert harness.at.exception == []
    # The metric must fall back to the placeholder rather than raising or lying.
    assert metric_map(harness.at)["Indexed chunks"] == app_module._format_count(None)
    assert metric_map(harness.at)["Sources"] == "2"


def test_unreadable_source_catalog_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_app(monkeypatch, catalog_error=True)
    assert harness.at.exception == []
    assert metric_map(harness.at)["Sources"] == "0"


def test_load_failure_shows_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = build_app(monkeypatch, load_error=True)
    assert harness.at.exception == []
    assert any("could not be initialized" in text for text in error_texts(harness.at))
    # The app must stop rather than render a chat UI with no backend.
    assert metric_map(harness.at) == {}


def test_shadowed_settings_are_surfaced_in_the_sidebar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_app(monkeypatch, shadowed=("OLLAMA_MODEL",))
    warnings = warning_texts(harness.at.sidebar)
    assert any("OLLAMA_MODEL" in text for text in warnings)
    assert any("Remove-Item Env:" in text for text in warnings)


def test_no_shadowing_warning_when_nothing_is_shadowed(harness: Harness) -> None:
    assert warning_texts(harness.at.sidebar) == []


# --------------------------------------------------------------------------
# Conversation
# --------------------------------------------------------------------------


def ask(harness: Harness, question: str) -> Harness:
    harness.at.chat_input[0].set_value(question)
    harness.at.run()
    return harness


def test_answer_is_appended_to_session_state(harness: Harness) -> None:
    ask(harness, "What is multi-head attention?")
    messages = harness.at.session_state["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What is multi-head attention?"
    assert messages[1]["content"] == DEFAULT_RAG_RESULT["answer"]
    assert harness.at.exception == []


def test_retrieval_settings_are_forwarded_to_the_search_call(
    harness: Harness,
) -> None:
    ask(harness, "What is multi-head attention?")
    assert len(harness.queries) == 1
    call = harness.queries[0]
    assert call["query"] == "What is multi-head attention?"
    assert call["top_k"] == app_module.DEFAULT_TOP_K
    assert call["min_score"] == app_module.DEFAULT_MIN_SCORE
    assert call["return_context"] is True
    assert call["source_filter"] is None


def test_slider_changes_reach_the_search_call(harness: Harness) -> None:
    harness.at.sidebar.slider[0].set_value(9)
    harness.at.sidebar.slider[1].set_value(0.75)
    harness.at.run()
    ask(harness, "How many heads?")
    call = harness.queries[0]
    assert call["top_k"] == 9
    assert call["min_score"] == pytest.approx(0.75)


def test_source_filter_narrows_the_search(harness: Harness) -> None:
    harness.at.sidebar.multiselect[0].set_value(["bulk:pdf:alpha.pdf"])
    harness.at.run()
    ask(harness, "Anything about alpha?")
    assert harness.queries[0]["source_filter"] == ["bulk:pdf:alpha.pdf"]


def test_turn_history_is_passed_to_the_next_question(harness: Harness) -> None:
    ask(harness, "First question?")
    ask(harness, "Second question?")
    assert harness.queries[1]["history"] == [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": DEFAULT_RAG_RESULT["answer"]},
    ]


def test_conversation_is_bounded(harness: Harness) -> None:
    turns = app_module.MAX_SESSION_MESSAGES // 2 + 3
    for index in range(turns):
        ask(harness, f"Question number {index}?")
    messages = harness.at.session_state["messages"]
    assert len(messages) == app_module.MAX_SESSION_MESSAGES
    # The newest turn must survive the trim.
    assert messages[-1]["content"] == DEFAULT_RAG_RESULT["answer"]


def test_clear_conversation_empties_the_history(harness: Harness) -> None:
    ask(harness, "Something worth keeping?")
    assert harness.at.session_state["messages"]
    sidebar_button(harness, "Clear conversation").click()
    harness.at.run()
    assert harness.at.session_state["messages"] == []


def test_answer_is_rendered_with_its_evidence(harness: Harness) -> None:
    ask(harness, "What is multi-head attention?")
    assert "multi-head attention" in chat_text(harness).lower()


def test_context_is_hidden_when_the_toggle_is_off(harness: Harness) -> None:
    harness.at.sidebar.checkbox[0].set_value(False)
    harness.at.run()
    ask(harness, "What is multi-head attention?")
    assert harness.queries[0]["return_context"] is False
    assert "Retrieved context" not in [expander.label for expander in harness.at.expander]


def test_context_is_shown_when_the_toggle_is_on(harness: Harness) -> None:
    ask(harness, "What is multi-head attention?")
    assert harness.queries[0]["return_context"] is True
    assert "Retrieved context" in [expander.label for expander in harness.at.expander]


# --------------------------------------------------------------------------
# Provider failure translation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (search_module.ProviderAuthError("rejected"), "local provider logs"),
        (search_module.ProviderRateLimitError("slow down"), "rate limiting requests"),
        (search_module.ProviderUnavailableError("no model"), "no model"),
        (ConnectionError("refused"), "Could not reach the chat provider"),
        (search_module.RetrievalError("bad index"), "Document retrieval failed"),
    ],
)
def test_provider_failures_become_readable_ui_messages(
    monkeypatch: pytest.MonkeyPatch, raised: BaseException, expected: str
) -> None:
    harness = build_app(monkeypatch, rag_error=raised)
    ask(harness, "Anything?")
    assert harness.at.exception == []
    assert expected in surfaced_texts(harness)
    # A failed turn must not be stored as if it had an answer.
    roles = [message["role"] for message in harness.at.session_state["messages"]]
    assert roles == ["user"]


def test_unexpected_error_does_not_leak_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = build_app(monkeypatch, rag_error=ValueError("secret internal detail"))
    ask(harness, "Anything?")
    surfaced = surfaced_texts(harness)
    assert "The answer could not be generated" in surfaced
    assert "secret internal detail" not in surfaced


def test_auth_error_names_the_credential_of_the_active_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "LLM_PROVIDER", "mistral")
    harness = build_app(monkeypatch, rag_error=search_module.ProviderAuthError("nope"))
    ask(harness, "Anything?")
    assert "MISTRAL_API_KEY" in surfaced_texts(harness)


def test_auth_error_under_a_local_provider_points_at_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the message used to blame MISTRAL_API_KEY even under Ollama."""
    monkeypatch.setattr(config_module, "LLM_PROVIDER", "ollama")
    harness = build_app(monkeypatch, rag_error=search_module.ProviderAuthError("nope"))
    ask(harness, "Anything?")
    surfaced = surfaced_texts(harness)
    assert "MISTRAL_API_KEY" not in surfaced
    assert "local provider logs" in surfaced


# --------------------------------------------------------------------------
# Source management
# --------------------------------------------------------------------------


def test_removing_a_source_is_confirmed_before_it_happens(harness: Harness) -> None:
    harness.at.sidebar.selectbox[0].set_value("bulk:pdf:alpha.pdf")
    harness.at.run()
    [b for b in harness.at.sidebar.button if b.label == "Remove selected source"][0].click()
    harness.at.run()
    assert harness.vectorstore.removed == []
    assert any("removes the source" in text for text in warning_texts(harness.at.sidebar))


def test_confirming_removal_deletes_the_source(harness: Harness) -> None:
    harness.at.sidebar.selectbox[0].set_value("bulk:pdf:alpha.pdf")
    harness.at.run()
    [b for b in harness.at.sidebar.button if b.label == "Remove selected source"][0].click()
    harness.at.run()
    [b for b in harness.at.sidebar.button if b.label == "Confirm"][0].click()
    harness.at.run()
    assert harness.vectorstore.removed == ["bulk:pdf:alpha.pdf"]


def test_cancelling_removal_leaves_the_index_untouched(harness: Harness) -> None:
    harness.at.sidebar.selectbox[0].set_value("bulk:pdf:alpha.pdf")
    harness.at.run()
    [b for b in harness.at.sidebar.button if b.label == "Remove selected source"][0].click()
    harness.at.run()
    [b for b in harness.at.sidebar.button if b.label == "Cancel"][0].click()
    harness.at.run()
    assert harness.vectorstore.removed == []


# --------------------------------------------------------------------------
# Upload path
# --------------------------------------------------------------------------


def test_process_upload_replaces_a_source_with_fresh_chunks(harness: Harness) -> None:
    payload = b"Retrieval augmented generation grounds answers in retrieved text. " * 40
    added = app_module._process_upload(
        FakeUploadedFile("notes.txt", payload), harness.embedding_manager, harness.vectorstore
    )
    assert added > 0
    # Staging must delete the previous source before writing its replacement.
    assert harness.vectorstore.deleted == ["upload:txt:notes.txt"]
    assert sum(count for count, _ in harness.vectorstore.added) == added
    assert all(
        vector_count == EMBEDDING_DIMENSIONS for _, vector_count in harness.vectorstore.added
    )
    assert harness.embedding_manager.calls[0]["is_query"] is False


def test_process_upload_splits_a_large_document_into_batches(harness: Harness) -> None:
    # Each sentence must differ: repeated text is deduplicated at ingestion, so a
    # repeated payload would collapse to a handful of chunks and never batch.
    payload = b"".join(
        f"Unique observation number {index} for batch coverage. ".encode() for index in range(6000)
    )
    added = app_module._process_upload(
        FakeUploadedFile("long.txt", payload), harness.embedding_manager, harness.vectorstore
    )
    assert added > app_module.EMBEDDING_BATCH_SIZE
    assert len(harness.vectorstore.added) > 1
    assert sum(count for count, _ in harness.vectorstore.added) == added
    assert all(count <= app_module.EMBEDDING_BATCH_SIZE for count, _ in harness.vectorstore.added)


def test_process_upload_rejects_an_unsupported_extension(harness: Harness) -> None:
    with pytest.raises(ValueError, match="PDF and TXT"):
        app_module._process_upload(
            FakeUploadedFile("notes.docx", b"content"),
            harness.embedding_manager,
            harness.vectorstore,
        )


def test_process_upload_enforces_the_size_limit(harness: Harness, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 8)
    with pytest.raises(ValueError, match="larger than"):
        app_module._process_upload(
            FakeUploadedFile("notes.txt", b"x" * 64),
            harness.embedding_manager,
            harness.vectorstore,
        )


def test_process_upload_returns_zero_for_an_empty_document(harness: Harness) -> None:
    added = app_module._process_upload(
        FakeUploadedFile("empty.txt", b"   \n  \n"),
        harness.embedding_manager,
        harness.vectorstore,
    )
    assert added == 0
    assert harness.vectorstore.deleted == []


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32", "system32"),
        ("/absolute/path/file.txt", "file.txt"),
        ("", "uploaded-document"),
    ],
)
def test_safe_display_name_strips_directories(raw: str, expected: str) -> None:
    assert app_module._safe_display_name(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0"), (999, "999"), (1000, "1,000"), (126872, "126,872"), (None, "—")],
)
def test_format_count(value: Any, expected: str) -> None:
    assert app_module._format_count(value) == expected


def test_build_download_text_includes_the_answer_and_its_sources() -> None:
    text = app_module._build_download_text("The answer.", list(DEFAULT_RAG_RESULT["sources"]))
    assert "The answer." in text
    assert "alpha.pdf" in text
