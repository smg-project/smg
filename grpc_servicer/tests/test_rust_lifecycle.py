"""Unit tests for the lifecycle helpers the Rust servicer launchers share."""

from __future__ import annotations

import sys
import types

import pytest
from smg_grpc_servicer import rust_lifecycle


def test_a_local_directory_is_the_tokenizer_directory(tmp_path):
    assert rust_lifecycle.resolve_tokenizer_dir(str(tmp_path)) == str(tmp_path)


def test_a_cached_snapshot_counts_only_once_it_holds_a_tokenizer(tmp_path, monkeypatch):
    """The engine downloads the same repo in parallel; a snapshot it has
    started (config present, tokenizer not yet) must not satisfy the local
    lookup, the Hub attempt that follows fetches the tokenizer files."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    calls: list[bool] = []

    def fake_snapshot_download(repo, revision=None, allow_patterns=None, local_files_only=False):
        calls.append(local_files_only)
        assert "*.json" in allow_patterns and "*.safetensors" not in str(allow_patterns)
        if not local_files_only:
            (snapshot / "tokenizer.json").write_text("{}")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    assert rust_lifecycle.resolve_tokenizer_dir("org/model") == str(snapshot)
    assert calls == [True, False]

    # Once the tokenizer is cached the local lookup is enough.
    calls.clear()
    assert rust_lifecycle.resolve_tokenizer_dir("org/model") == str(snapshot)
    assert calls == [True]


def test_an_unresolvable_tokenizer_is_none(tmp_path, monkeypatch):
    def failing(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=failing)
    )
    assert rust_lifecycle.resolve_tokenizer_dir("org/missing") is None


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer.model", "x.tiktoken", "vocab.json"])
def test_holds_tokenizer_recognises_each_loader_input(tmp_path, name):
    assert not rust_lifecycle.holds_tokenizer(str(tmp_path))
    (tmp_path / name).write_text("")
    assert rust_lifecycle.holds_tokenizer(str(tmp_path))
