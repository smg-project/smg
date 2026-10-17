"""Put this repo's ``grpc_servicer/`` at the front of ``sys.path`` so
``smg_grpc_servicer`` imports resolve in-repo when tests run from the repo root
(as CI does), without an editable install.

``SMG_GRPC_PROTO_PATH`` names a directory holding locally generated
``smg_grpc_proto`` stubs (``scripts/gen_proto_stubs.py``), so the tests see
this checkout's proto instead of the released package.

Every test runs with no servicer implementation selected by the environment,
and its teardown puts back what the shell exported, if anything
(:func:`no_servicer_impl_selected`).
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_GRPC_SERVICER_ROOT = Path(__file__).resolve().parent.parent
if sys.path[:1] != [str(_GRPC_SERVICER_ROOT)]:
    sys.path.insert(0, str(_GRPC_SERVICER_ROOT))

_PROTO_PATH = os.environ.get("SMG_GRPC_PROTO_PATH")
if _PROTO_PATH and _PROTO_PATH not in sys.path:
    sys.path.insert(0, _PROTO_PATH)


@pytest.fixture(autouse=True)
def no_servicer_impl_selected():
    """No servicer implementation is selected by the environment while a test
    runs; at its end the shell's own selection, if any, is put back.

    The ``SMG_<ENGINE>_SERVICER_IMPL`` variables select the implementation a
    launcher serves, and the switch writes a flag's decision back to them
    (``rust_lifecycle.servicer_switch``) so the engine processes a launcher
    starts, and the Python servicer's guard, agree with the flag. A test that
    drives the switch with a flag against the process environment, as a
    launcher does, would otherwise hand its selection to every test after it:
    the tests that construct the Python vLLM servicer, whose constructor
    refuses a Rust selection, then fail wherever vLLM is importable. The
    shell's own export is set aside for the same reason: a unit test is not
    the launcher the guard protects, and a test that wants a selection makes
    it itself (``monkeypatch.setenv``). The restore is by hand because
    ``monkeypatch.delenv`` of a name that is absent records nothing, and a
    write-back after it would survive the test.
    """
    from smg_grpc_servicer import sglang, tokenspeed, vllm

    names = (vllm.SERVICER_IMPL_ENV, sglang.SERVICER_IMPL_ENV, tokenspeed.SERVICER_IMPL_ENV)
    before = {name: os.environ.pop(name, None) for name in names}
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


_TOKENSPEED_SERVICER = _GRPC_SERVICER_ROOT / "smg_grpc_servicer" / "tokenspeed" / "servicer.py"


@pytest.fixture
def tokenspeed_servicer_mod(monkeypatch):
    """The TokenSpeed servicer module, loaded with the engine's imports stubbed.

    Importing the engine (and torch) loads the runtime's dependencies; these
    stand-ins cover the servicer module's imports and nothing of the method a
    test drives. The module is loaded afresh for every test, under its own
    name, so a test's monkeypatching never leaks into the next.
    """
    symbols = {
        "torch": "",
        "tokenspeed.runtime.multimodal.inputs": "Modality MultimodalDataItem MultimodalInputs",
        "tokenspeed.runtime.multimodal.shm_transport": "ShmTensorHandle",
        "tokenspeed.runtime.pd.kv_events": "KVEventBatch",
    }
    for name, attrs in symbols.items():
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent not in sys.modules:
                module = ModuleType(parent)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, parent, module)
        module = ModuleType(name)
        for attr in attrs.split():
            setattr(module, attr, object)
        monkeypatch.setitem(sys.modules, name, module)
    name = "smg_grpc_servicer.tokenspeed.servicer"
    spec = importlib.util.spec_from_file_location(name, _TOKENSPEED_SERVICER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module
