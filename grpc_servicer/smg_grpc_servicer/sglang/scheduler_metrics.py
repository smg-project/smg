"""SGLang's scheduler metrics on the Rust path, where its HTTP sidecar served them.

Under ``--enable-metrics`` SGLang's scheduler ranks record their series
(running and queued requests, KV cache usage, generation throughput, queue
time, cache hit rate) into Prometheus's multiprocess directory,
``PROMETHEUS_MULTIPROC_DIR``. SGLang's HTTP sidecar serves that directory, but
only beside the Python servicer's request manager. On the Rust path the
launcher serves it instead, at the sidecar's address, so a scrape set up for
the sidecar keeps the scheduler's series.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from smg_grpc_servicer.hostport import host_port

logger = logging.getLogger(__name__)


class SchedulerMetricsEndpoint:
    """Prometheus's own HTTP server on a daemon thread, over a registry that
    reads the directory at each scrape."""

    def __init__(self, server: Any, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread

    def close(self) -> None:
        """Stop serving and release the port."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


def serve_scheduler_metrics(server_args: Any) -> SchedulerMetricsEndpoint | None:
    """Serve the scheduler ranks' metrics directory on the sidecar's address
    when the launch asked for metrics. A failure to serve it is logged and
    returns ``None``: as with SGLang's sidecar, the gRPC servicer runs on."""
    if not getattr(server_args, "enable_metrics", False):
        return None
    # Not at the top: prometheus_client picks memory or file storage when first
    # imported, and this module loads before SGLang sets PROMETHEUS_MULTIPROC_DIR.
    # An early import keeps this process's own metrics out of the directory.
    from prometheus_client import CollectorRegistry, start_http_server
    from prometheus_client.multiprocess import MultiProcessCollector

    host = getattr(server_args, "host", None) or "0.0.0.0"
    # SGLang's own default for the sidecar; the field is gone from releases
    # that dropped the sidecar.
    port = getattr(server_args, "smg_http_sidecar_port", None) or int(server_args.port) + 1
    address = host_port(host, port)
    try:
        # On Linux, bind() raises OverflowError (not an OSError) for a port past
        # 65535; --port 65535 derives one.
        if not 0 < port < 65536:
            raise ValueError(f"port {port} is outside the TCP range 1-65535")
        registry = CollectorRegistry()
        # Raises ValueError when this process has no metrics directory.
        MultiProcessCollector(registry)
        server, thread = start_http_server(port, addr=host, registry=registry)
    except (OSError, ValueError) as error:
        logger.warning(
            "SGLang's scheduler metrics (--enable-metrics) are not served on %s: %s",
            address,
            error,
        )
        return None
    logger.info("SGLang's scheduler metrics on http://%s/metrics", address)
    return SchedulerMetricsEndpoint(server, thread)
