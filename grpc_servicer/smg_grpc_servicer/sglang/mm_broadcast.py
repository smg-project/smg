"""Optional single-node shared-memory transport for Rust ingress TP broadcasts.

The ZMQ decoder already views large tensor frames without a copy. The next
scheduler broadcast, however, pickles ordinary tensors into the collective.
Reuse the engine's SHM lifecycle so that only handles cross the TP group.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
ENV = "SMG_SGLANG_MM_SHM_BROADCAST"
MIN_BYTES = 32 * 1024


def enabled() -> bool:
    """Resolve the opt-out for the explicitly selected Rust media worker."""
    override = os.environ.get(ENV)
    if override is not None:
        return override == "1"
    # This hook is installed only by the Rust ZMQ scheduler plugin. Other
    # transports keep their existing defaults; capability checks run per rank.
    return os.environ.get("SMG_SGLANG_MM_PROCESSOR", "").strip().lower() == "smg"


def broadcast_with_shm(original, receiver, recv_reqs, *args, **kwargs):
    """Use engine-owned SHM cleanup only for compatible local TP broadcasts."""
    # A handle names memory on this host, never across nodes. Engines without
    # the collective materialization/error path retain the inline transport.
    args_view = getattr(receiver, "server_args", None)
    if (
        getattr(args_view, "nnodes", None) == 1
        and getattr(args_view, "tp_size", 1) > 1
        and hasattr(receiver, "_finalize_shm_features")
        and recv_reqs
    ):
        try:
            from sglang.srt.managers.mm_utils import ShmPointerMMData
        except ImportError:
            return original(receiver, recv_reqs, *args, **kwargs)

        for req in recv_reqs:
            mm = getattr(req, "mm_inputs", None)
            for item in getattr(mm, "mm_items", ()):
                feature = getattr(item, "feature", None)
                if (
                    feature is None
                    or not getattr(feature, "is_cpu", False)
                    or feature.numel() * feature.element_size() < MIN_BYTES
                ):
                    continue
                try:
                    pointer = ShmPointerMMData(feature)
                except OSError:
                    # The engine reserves pages before copying; ENOSPC is a
                    # recoverable allocation failure, not a fatal SIGBUS.
                    logger.warning("SMG MM SHM allocation failed; using inline TP transport")
                    continue
                # broadcast_pyobj returns the source object unchanged. Open
                # the source rank's handle too, just as unpickling opens it on
                # peers, before the engine's all-rank barrier and unlink.
                pointer.__setstate__(pointer.__getstate__())
                item.feature = pointer
    return original(receiver, recv_reqs, *args, **kwargs)
