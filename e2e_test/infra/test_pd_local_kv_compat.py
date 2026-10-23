"""Offline source and explicit opt-in controls; no engine is imported."""

import hashlib
import sys
from types import SimpleNamespace as NS

import pytest
from infra import pd_local_kv_compat as local
from infra import pd_scheduler_compat as compat


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_explicit_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SMG_PD_TRACE_LOCAL_KV", raising=False)
    else:
        monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", value)
    assert local.enabled() is expected


@pytest.mark.parametrize("value", ["", "true", "2"])
def test_invalid_flag_rejected(monkeypatch, value):
    monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", value)
    with pytest.raises(ValueError, match="opt-in"):
        local.enabled()


def test_opt_in_checks_additional_source_before_backend_import(monkeypatch, tmp_path):
    native = tmp_path / "native.py"
    native.write_bytes(b"native CPU fixture")
    allocator = tmp_path / "allocator.py"
    allocator.write_bytes(b"allocator CPU fixture")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(compat, "SOURCE_SHA256", {native.name: digest(native)})
    monkeypatch.setattr(local, "SOURCE_SHA256", {allocator.name: digest(allocator)})
    monkeypatch.setattr(
        compat, "find_spec", lambda _: NS(submodule_search_locations=[str(tmp_path)])
    )
    monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", "1")
    assert compat.require_installed_contract()["local_kv_sources"] == local.SOURCE_SHA256
    allocator.write_bytes(b"modified allocator")
    with pytest.raises(RuntimeError, match="source contract"):
        compat.require_installed_contract()
    monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", "0")
    assert "local_kv_sources" not in compat.require_installed_contract()
    monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", "1")
    allocator.unlink()
    with pytest.raises(RuntimeError, match="source contract"):
        compat.require_installed_contract()


def test_additional_loaded_module_origin_is_required(monkeypatch):
    module = NS(__file__="/foreign/allocator.py", __spec__=NS(origin="/foreign/allocator.py"))
    monkeypatch.setitem(sys.modules, "vllm.allocator", module)
    with pytest.raises(RuntimeError, match="loaded observer source"):
        compat.require_loaded_contract(
            {"root": "/pinned", "local_kv_sources": {"allocator.py": "hash"}}, None, None
        )
