"""Python entrypoint for the Rust WorkerControl server.

The callback boundary is control-plane-only: inference requests and token
streaming stay in Rust and the engine-native transports. ``init_tracing``
installs the Rust tracing subscriber once; later calls are no-ops.
"""

from smg.smg_rs import WorkerControlServer, init_tracing

__all__ = ["WorkerControlServer", "init_tracing"]
