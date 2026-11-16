"""Scheduler subprocess launcher for the TokenSpeed gRPC server.

Delegates to TokenSpeed's ``_launch_subprocesses`` and returns the
``AsyncLLM`` + scheduler-info dict the gRPC server cares about. Both belong
to the engine's in-process Python frontend, which the engine no longer
ships: they are imported when the Python path is taken, so the worker
entrypoint imports on every engine and the Rust path never reaches them.
"""

from __future__ import annotations

import logging
from typing import Any

from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def launch_engine(
    server_args: ServerArgs,
    port_args: PortArgs | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Launch the scheduler subprocess(es) and return the live ``AsyncLLM``.

    Raises ``RuntimeError`` on non-rank-0 nodes (which return ``None`` and
    block forever on the dummy health server — they never serve gRPC), and
    on an engine without the in-process frontend, naming the path that
    serves it.
    """
    try:
        from tokenspeed.runtime.entrypoints.engine import _launch_subprocesses
    except ImportError as error:
        raise RuntimeError(
            "This TokenSpeed ships no in-process Python frontend (AsyncLLM), so the Python "
            "servicer path cannot serve it; drop --servicer-impl python / "
            f"SMG_TOKENSPEED_SERVICER_IMPL=python and run the default, the Rust servicer ({error})"
        ) from error

    async_llm, _template_manager, scheduler_info = _launch_subprocesses(
        server_args=server_args,
        port_args=port_args,
    )

    if async_llm is None:
        raise RuntimeError(
            "launch_engine() returned no AsyncLLM — only rank 0 may serve gRPC traffic."
        )

    logger.info(
        "TokenSpeed engine ready: max_total_num_tokens=%s max_req_input_len=%s",
        scheduler_info.get("max_total_num_tokens"),
        scheduler_info.get("max_req_input_len"),
    )
    return async_llm, scheduler_info
