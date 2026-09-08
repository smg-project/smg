"""Python engine adapter entrypoint for the Rust WorkerControl server.

The callback boundary is intentionally control-plane-only. Inference requests
and token streaming remain entirely in Rust/engine-native transports.
"""

from smg.smg_rs import WorkerControlServer

__all__ = ["WorkerControlServer"]
