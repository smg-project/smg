"""Put this repo's ``grpc_servicer/`` at the front of ``sys.path`` so
``smg_grpc_servicer`` imports resolve in-repo when tests run from the repo root
(as CI does), without an editable install.

``SMG_GRPC_PROTO_PATH`` names a directory holding locally generated
``smg_grpc_proto`` stubs (``scripts/gen_proto_stubs.py``), so the tests see
this checkout's proto instead of the released package.
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
