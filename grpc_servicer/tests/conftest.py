"""Put this repo's ``grpc_servicer/`` at the front of ``sys.path`` so
``smg_grpc_servicer`` imports resolve in-repo when tests run from the repo root
(as CI does), without an editable install.

``SMG_GRPC_PROTO_PATH`` names a directory holding locally generated
``smg_grpc_proto`` stubs (``scripts/gen_proto_stubs.py``), so the tests see
this checkout's proto instead of the released package.
"""

import os
import sys
from pathlib import Path

_GRPC_SERVICER_ROOT = Path(__file__).resolve().parent.parent
if sys.path[:1] != [str(_GRPC_SERVICER_ROOT)]:
    sys.path.insert(0, str(_GRPC_SERVICER_ROOT))

_PROTO_PATH = os.environ.get("SMG_GRPC_PROTO_PATH")
if _PROTO_PATH and _PROTO_PATH not in sys.path:
    sys.path.insert(0, _PROTO_PATH)
