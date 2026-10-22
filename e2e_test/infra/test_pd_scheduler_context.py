"""CPU checks: source fixtures and raw admission records, never backend execution."""

import hashlib
import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
from infra import pd_scheduler_compat as compat
from infra.pd_request_identity import RequestIdentity


def test_source_guard_rejects_missing_or_changed_files(monkeypatch, tmp_path):
    expected = hashlib.sha256(b"pinned CPU source fixture").hexdigest()
    monkeypatch.setattr(compat, "SOURCE_SHA256", {"scheduler.py": expected})
    with pytest.raises(RuntimeError, match="unknown observer source contract"):
        compat.require_source_contract(tmp_path)
    path = tmp_path / "scheduler.py"
    path.write_bytes(b"pinned CPU source fixture")
    assert compat.require_source_contract(tmp_path) == {"scheduler.py": expected}
    path.write_bytes(b"same version, changed behavior")
    with pytest.raises(RuntimeError, match="unknown observer source contract"):
        compat.require_source_contract(tmp_path)


@pytest.mark.parametrize("locations", [None, [], ["a", "b"]])
def test_package_location_must_be_unambiguous(monkeypatch, locations):
    monkeypatch.setattr(
        compat, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=locations)
    )
    with pytest.raises(RuntimeError, match="unknown vLLM package location"):
        compat.require_installed_contract()


def test_entry_guard_failure_prevents_backend_import(monkeypatch):
    def unknown():
        raise RuntimeError("unknown source fixture")

    monkeypatch.setattr(compat, "require_installed_contract", unknown)
    sys.modules.pop("infra.pd_vllm_scheduler", None)
    before = {name for name in sys.modules if name.startswith(("vllm", "torch"))}
    with pytest.raises(RuntimeError, match="unknown source fixture"):
        importlib.import_module("infra.pd_vllm_scheduler")
    assert {name for name in sys.modules if name.startswith(("vllm", "torch"))} == before


@pytest.mark.parametrize("record", [None, {}, {"request": None}])
def test_malformed_admission_cannot_supply_an_id(record):
    with pytest.raises(ValueError, match="unknown request admission record"):
        correlate(RequestIdentity.new("run", "2"), record)


@pytest.mark.parametrize("foreign", ["file", "spec", "class"])
def test_checked_root_cannot_attest_foreign_loaded_scheduler(monkeypatch, tmp_path, foreign):
    classes = []
    for suffix, cls_name in (("scheduler", "Scheduler"), ("async_scheduler", "AsyncScheduler")):
        name = f"vllm.v1.core.sched.{suffix}"
        path = str(tmp_path / f"v1/core/sched/{suffix}.py")
        module = ModuleType(name)
        module.__file__ = path
        module.__spec__ = ModuleSpec(name, None, origin=path)
        cls = type(cls_name, (), {"__module__": name})
        classes.append(cls)
        monkeypatch.setitem(sys.modules, name, module)
    contract = {"root": str(tmp_path)}
    compat.require_loaded_contract(contract, *classes)
    module = sys.modules["vllm.v1.core.sched.scheduler"]
    if foreign == "file":
        module.__file__ = "/unrelated/scheduler.py"
    elif foreign == "spec":
        module.__spec__.origin = "/unrelated/scheduler.py"
    else:
        classes[0].__module__ = "foreign_scheduler"
    with pytest.raises(
        RuntimeError, match="unknown (loaded observer source|native scheduler class origin)"
    ):
        compat.require_loaded_contract(contract, *classes)


def admission(identity, worker="p-0", role="prefill", suffix="0123abcd"):
    return {
        "run": identity.run,
        "attempt": identity.attempt,
        "worker": worker,
        "role": role,
        "instance": f"{worker}-instance",
        "schema": 1,
        "event": "add_request_return",
        "registered": True,
        "request": {"request_id": f"chatcmpl-{identity.request_id}-{suffix}"},
    }


def correlate(identity, record, worker="p-0", role="prefill"):
    return identity.admitted_id(record, worker=worker, role=role, instance=f"{worker}-instance")


def test_explicit_header_correlates_distinct_p_d_and_retry_admissions():
    identity = RequestIdentity.new("run", "attempt-2")
    assert identity.headers == {"x-request-id": identity.request_id}
    assert identity.request_id != RequestIdentity.new("run", "attempt-2").request_id
    p = correlate(identity, admission(identity))
    d = correlate(identity, admission(identity, "d-0", "decode", "abcdef01"), "d-0", "decode")
    retry = correlate(identity, admission(identity, suffix="11223344"))
    assert len({p, d, retry}) == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("run", "other"),
        ("attempt", "old"),
        ("worker", "p-1"),
        ("role", "decode"),
        ("event", "finish_requests_return"),
        ("registered", False),
        ("registered", 1),
        ("instance", "previous-process"),
        ("schema", True),
    ],
)
def test_wrong_context_is_unknown(key, value):
    identity = RequestIdentity.new("run", "2")
    record = admission(identity)
    record[key] = value
    with pytest.raises(ValueError, match="unknown request admission context"):
        correlate(identity, record)


@pytest.mark.parametrize(
    "internal",
    [
        "chatcmpl-client",
        "chatcmpl-client_0-0123abcd",
        "chatcmpl-client-0123abcg",
        "chatcmpl-client-0123abcd-extra",
        "chatcmpl-other-0123abcd",
    ],
)
def test_unproven_id_shapes_are_unknown(internal):
    identity = RequestIdentity("run", "2", "client")
    record = admission(identity)
    record["request"]["request_id"] = internal
    with pytest.raises(ValueError, match="unknown internal request ID"):
        correlate(identity, record)


def test_client_metacharacters_are_literal_and_header_contract_is_strict():
    identity = RequestIdentity("run", "2", "literal.+[]")
    record = admission(identity)
    assert correlate(identity, record) == record["request"]["request_id"]
    for bad in ("", "space inside", "x" * 129, "中文", "\t"):
        with pytest.raises(ValueError):
            RequestIdentity("run", "2", bad)
