from __future__ import annotations

import re
from pathlib import Path

import src.config as config


def _write_env(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_shadowed_env_reports_process_override(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "MISTRAL_API_KEY=file-value\nHF_TOKEN=file-token\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("MISTRAL_API_KEY", "stale-process-value")

    assert config._detect_shadowed_env_names() == ("MISTRAL_API_KEY",)


def test_shadowed_env_is_empty_when_values_match(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "MISTRAL_API_KEY=same-value\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("MISTRAL_API_KEY", "same-value")

    assert config._detect_shadowed_env_names() == ()


def test_shadowed_env_reports_non_credential_settings(tmp_path, monkeypatch) -> None:
    # A shell variable that overrides OLLAMA_MODEL is just as invisible as one
    # that overrides an API key, and just as likely to make .env look broken.
    env_file = tmp_path / ".env"
    _write_env(env_file, "OLLAMA_MODEL=llama3.1:8b\nRAG_TOP_K=5\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:latest")
    monkeypatch.setenv("RAG_TOP_K", "9")

    assert config._detect_shadowed_env_names() == ("OLLAMA_MODEL", "RAG_TOP_K")


def test_shadowed_env_ignores_variables_the_app_does_not_read(tmp_path, monkeypatch) -> None:
    # Unrelated overrides such as a stale PYTHONPATH are noise, not diagnostics.
    env_file = tmp_path / ".env"
    _write_env(env_file, "PYTHONPATH=/some/other/checkout\nSOME_UNRELATED_FLAG=1\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("PYTHONPATH", "/another/checkout")
    monkeypatch.setenv("SOME_UNRELATED_FLAG", "0")

    assert config._detect_shadowed_env_names() == ()


def test_shadowed_env_handles_missing_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent.env")

    assert config._detect_shadowed_env_names() == ()


def test_shadowed_env_skips_blank_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "MISTRAL_API_KEY=\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("MISTRAL_API_KEY", "from-process")

    assert config._detect_shadowed_env_names() == ()


def test_every_setting_read_by_config_is_reported_when_shadowed() -> None:
    # Guards against a new setting being added to config.py but not to the
    # reporting list, which would silently reintroduce the override failure.
    source = Path(config.__file__).read_text(encoding="utf-8")
    read_names = set(re.findall(r'os\.getenv\(\s*"([A-Z0-9_]+)"', source))

    assert read_names, "expected config.py to read settings via os.getenv"
    assert read_names <= config.APP_SETTING_NAMES, (
        f"not covered by shadowing reports: {sorted(read_names - config.APP_SETTING_NAMES)}"
    )
