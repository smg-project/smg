"""Put this repo's ``grpc_servicer/`` at the front of ``sys.path`` so
``smg_grpc_servicer`` imports resolve in-repo when tests run from the repo root
(as CI does), without an editable install.

``SMG_GRPC_PROTO_PATH`` names a directory holding locally generated
``smg_grpc_proto`` stubs (``scripts/gen_proto_stubs.py``), so the tests see
this checkout's proto instead of the released package.

Every test runs with nothing of the launchers in the environment (no servicer
implementation selected, no handshake address or spawn method of a launch),
and its teardown puts back what the shell exported, if anything
(:func:`nothing_of_the_launchers_in_the_environment`).
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
def nothing_of_the_launchers_in_the_environment():
    """The variables the launchers write into the process environment are
    unset while a test runs; at its end the shell's own values, if any, are
    put back.

    The ``SMG_<ENGINE>_SERVICER_IMPL`` variables select the implementation a
    launcher serves, and the switch writes a flag's decision back to them
    (``rust_lifecycle.servicer_switch``) so the engine processes a launcher
    starts, and the Python servicer's guard, agree with the flag. A test that
    drives the switch with a flag against the process environment, as a
    launcher does, would otherwise hand its selection to every test after it:
    the tests that construct the Python vLLM servicer, whose constructor
    refuses a Rust selection, then fail wherever vLLM is importable. A launch
    writes more, straight into ``os.environ`` for the engine processes to
    inherit: the SGLang rank launch its ZMQ handshake address and engine
    index, the vLLM engine-core launch the worker spawn method (the TokenSpeed
    rank launch writes nothing). A test that drives a launch would hand those
    to every test after it the same way, whatever it stubbed. The shell's own
    exports are set aside for the same reason: a unit test is not the launcher
    the guard protects, and a test that wants a value makes it itself
    (``monkeypatch.setenv``); the planned warm-up's knob of each launcher is
    among them, so a shell that turns the warm-up off for its own servers
    does not turn it off under a test that expects a plan. The restore is by
    hand because
    ``monkeypatch.delenv`` of a name that is absent records nothing, and a
    write-back after it would survive the test.
    """
    from smg_grpc_servicer import sglang, tokenspeed, vllm
    from smg_grpc_servicer.sglang import warmup as sglang_warmup
    from smg_grpc_servicer.tokenspeed import warmup as tokenspeed_warmup
    from smg_grpc_servicer.vllm import warmup as vllm_warmup

    names = (
        vllm.SERVICER_IMPL_ENV,
        sglang.SERVICER_IMPL_ENV,
        tokenspeed.SERVICER_IMPL_ENV,
        sglang.HANDSHAKE_ENV,
        sglang.ENGINE_INDEX_ENV,
        vllm.MULTIPROC_METHOD_ENV,
        vllm_warmup.WARMUP_ENV,
        sglang_warmup.WARMUP_ENV,
        tokenspeed_warmup.WARMUP_ENV,
    )
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
_VLLM_ERRORS = _GRPC_SERVICER_ROOT / "smg_grpc_servicer" / "vllm" / "errors.py"


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


@pytest.fixture
def vllm_errors_mod(monkeypatch):
    """The vLLM servicer's error module, loaded with the engine's exception
    hierarchy stubbed (the current one: the client/server split and the
    admission-control ``GracefulHTTPError`` that carries an HTTP status).
    """

    class VLLMError(Exception):
        pass

    class VLLMClientError(VLLMError):
        pass

    class VLLMNotFoundError(VLLMClientError):
        pass

    class GracefulHTTPError(VLLMError):
        def __init__(self, message, http_status):
            super().__init__(message)
            self.http_status = http_status

    class EngineGenerateError(VLLMError):
        pass

    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    exceptions = ModuleType("vllm.exceptions")
    exceptions.VLLMClientError = VLLMClientError
    exceptions.VLLMNotFoundError = VLLMNotFoundError
    exceptions.GracefulHTTPError = GracefulHTTPError
    monkeypatch.setitem(sys.modules, "vllm.exceptions", exceptions)
    engine_exceptions = ModuleType("vllm.v1.engine.exceptions")
    engine_exceptions.EngineGenerateError = EngineGenerateError
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.exceptions", engine_exceptions)
    name = "smg_grpc_servicer.vllm.errors"
    spec = importlib.util.spec_from_file_location(name, _VLLM_ERRORS)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module
