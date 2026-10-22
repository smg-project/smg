"""Observer controls; synthetic paths test collection, not native compatibility."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import runtime_identity  # noqa: E402


def test_maps_selects_only_relevant_native_paths():
    maps = "\n".join(
        [
            "a-b r-xp 0 0 0 /venv/tvm_ffi/lib/libtvm_ffi.so",
            "a-b r-xp 0 0 0 /opt/tilelang/build/libtilelang.so",
            "a-b r-xp 0 0 0 /secret/private.so",
            "a-b r-xp 0 0 0 /venv/tvm_ffi/registry.py",
            "a-b r-xp 0 0 0 [heap]",
        ]
    )
    assert runtime_identity.libraries(maps) == [
        "/opt/tilelang/build/libtilelang.so",
        "/venv/tvm_ffi/lib/libtvm_ffi.so",
    ]


def test_file_identity_tracks_real_bytes_and_missing(tmp_path):
    observer = runtime_identity.RuntimeIdentity(tmp_path)
    path = tmp_path / "public.py"
    assert not observer.file(path)["exists"]
    path.write_bytes(b"first")
    first = observer.file(path)
    path.write_bytes(b"second")
    second = observer.file(path)
    assert first["sha256"] != second["sha256"] and second["bytes"] == 6


@pytest.mark.parametrize("missing", ["ffi", "tilelang", "tvm"])
def test_success_rejects_missing_loaded_identity(tmp_path, monkeypatch, missing):
    names = {
        "ffi": "/venv/tvm_ffi/lib/libtvm_ffi.so",
        "tilelang": "/opt/tilelang/libtilelang.so",
        "tvm": "/opt/tilelang/libtvm.so",
    }
    observer = runtime_identity.RuntimeIdentity(tmp_path)
    monkeypatch.setattr(
        observer,
        "capture",
        lambda phase: {
            "loaded_libraries": [
                {"path": p, "exists": True} for n, p in names.items() if n != missing
            ]
        },
    )
    with pytest.raises(RuntimeError, match="missing actual loaded"):
        observer.pytest_sessionfinish(SimpleNamespace(), 0)


def test_failed_native_keeps_exception_source_and_finally_snapshot(tmp_path, monkeypatch):
    observer = runtime_identity.RuntimeIdentity(tmp_path)
    phases = []
    monkeypatch.setattr(observer, "capture", lambda phase: phases.append(phase))
    call = SimpleNamespace(
        excinfo=SimpleNamespace(
            traceback=[
                SimpleNamespace(path=Path("/opt/tilelang/tvm/ir/attrs.py")),
                SimpleNamespace(path=Path("/secret/private.py")),
            ]
        )
    )
    observer.pytest_exception_interact(None, call, SimpleNamespace(nodeid="native"))
    observer.pytest_sessionfinish(SimpleNamespace(), 1)
    assert observer.failure_sources == {"/opt/tilelang/tvm/ir/attrs.py"}
    assert phases == ["exception:native", "session-finish:1"]


def test_runtime_pins_preserved_and_final_environment_recorded(tmp_path, monkeypatch):
    import json

    import environment

    monkeypatch.setattr(environment, "OUT", tmp_path)
    monkeypatch.setenv("PIP_CONSTRAINT", "/public/image.txt /public/runtime-pins.txt")
    path = tmp_path / "runtime-pins.txt"
    path.write_text("apache-tvm-ffi==0.1.12+amdreadonlydict2\n")
    environment.pin_state("before")
    environment.pin_state("final")
    state = json.loads((tmp_path / "runtime-pins-state-final.json").read_text())
    assert state["text"] == path.read_text() and "image.txt" in state["PIP_CONSTRAINT"]
    path.write_text("apache-tvm-ffi==0.1.12\n")
    with pytest.raises(ValueError, match="runtime pins changed"):
        environment.pin_state("final")


def test_success_rejects_wrong_actual_ffi_native_hash(tmp_path, monkeypatch):
    observer = runtime_identity.RuntimeIdentity(tmp_path)
    paths = [
        "/venv/tvm_ffi/lib/libtvm_ffi.so",
        "/venv/tvm_ffi/core.abi3.so",
        "/opt/tilelang/libtilelang.so",
        "/opt/tilelang/libtvm.so",
    ]
    monkeypatch.setattr(
        observer,
        "capture",
        lambda phase: {
            "loaded_libraries": [{"path": p, "exists": True, "sha256": "wrong"} for p in paths]
        },
    )
    with pytest.raises(RuntimeError, match="loaded FFI native identity differs"):
        observer.pytest_sessionfinish(SimpleNamespace(), 0)
