"""The encode dispatch of a language-side worker, for the Rust servicer.

Under encode/prefill disaggregation a ``--language-only`` worker runs no
vision tower: the frontend hands each request's image references to the
encoders and tells the scheduler to wait for their embeddings. SGLang's own
frontend (its tokenizer manager) does that with two objects of its own in the
tokenizer-manager process: an ``EncoderBootstrapServer`` (where encoders
register, on ``--encoder-bootstrap-port``, pre-filled with ``--encoder-urls``)
and an ``MMReceiver`` (``create_mm_receiver``: the assignment across the
encoders, the encode request to each over HTTP or over SGLang's encoder gRPC
service, the dispatch thread, the failure notice back to the scheduler).

:class:`EncodeDispatchBridge` hosts those same two objects in the servicer
process, where the Python servicer and the prefill side's bootstrap server
already live, and drives them per request for the Rust server: it builds
the ``GenerateReqInput`` the receiver reads (``rid`` and ``image_data``),
calls ``send_encode_request`` and hands Rust what the receiver set on it
(the wait flag, the per-encoder assignment, the encoder snapshot) plus the
release the receiver's thread waits for before it may report a failure. The
protocol is the binding's (``smg.servicer.SglangGrpcServer(encode_dispatcher=...)``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: The encoder transfer backends on which the scheduler ranks receive the
#: embeddings themselves (their own receiver), so the frontend's part is the
#: dispatch alone. ``zmq_to_tokenizer`` makes the frontend the receiver (it
#: assembles the embeddings and processes the prompt itself), which neither
#: servicer runs.
SCHEDULER_RECEIVE_BACKENDS = ("zmq_to_scheduler", "mooncake")


def resolve_encoder_transfer_backend(server_args: Any) -> str:
    """The transfer backend a ``--language-only`` launch runs, with SGLang's
    own resolution of ``auto`` (model- and TP-aware); an engine release
    without the resolver takes ``auto`` as ``zmq_to_scheduler``, which is what
    the resolver picks for every model but one."""
    backend = str(getattr(server_args, "encoder_transfer_backend", "auto") or "auto")
    if backend != "auto":
        return backend
    try:
        from sglang.srt.server_args import resolve_encoder_transfer_backend as resolve
    except ImportError:
        return "zmq_to_scheduler"
    model_arch = ""
    try:
        from smg_grpc_servicer.sglang.rust import _model_config

        architectures = getattr(_model_config(server_args).hf_config, "architectures", None) or []
        model_arch = str(architectures[0]) if architectures else ""
    except Exception:  # the resolver only needs the one architecture it special-cases
        model_arch = ""
    return str(resolve(backend, model_arch, int(getattr(server_args, "tp_size", 1) or 1)))


def dispatches_from_the_frontend(server_args: Any) -> bool:
    """Whether a launch is a language-side worker whose frontend dispatches
    to the encoders (``--language-only`` on a backend the scheduler ranks
    receive on)."""
    return bool(getattr(server_args, "language_only", False)) and (
        resolve_encoder_transfer_backend(server_args) in SCHEDULER_RECEIVE_BACKENDS
    )


class EncodeDispatchBridge:
    """SGLang's own language-side objects, driven per request for the Rust
    server. Built before the server (it is handed in as a keyword argument),
    started once the runtime context is published (the receiver reads the
    disaggregation and serving config from it), stopped with the ranks."""

    def __init__(self, server_args: Any):
        self.server_args = server_args
        self.encoder_urls: list[str] = list(getattr(server_args, "encoder_urls", None) or [])
        self.bootstrap_server: Any | None = None
        self.mm_receiver: Any | None = None

    @property
    def started(self) -> bool:
        return self.mm_receiver is not None

    def start(self) -> None:
        """Start the bootstrap server and the receiver, as SGLang's own
        frontend does: the shared URL list is pre-filled with ``--encoder-urls``
        and grows as encoders register. Idempotent."""
        if self.started:
            return
        from sglang.srt.disaggregation.encoder.receiver import (
            EncoderBootstrapServer,
            create_mm_receiver,
        )

        self._ensure_runtime_context()
        host = getattr(self.server_args, "host", None) or "0.0.0.0"
        port = int(getattr(self.server_args, "encoder_bootstrap_port", 8997) or 8997)
        self.bootstrap_server = EncoderBootstrapServer(host=host, port=port, urls=self.encoder_urls)
        self.mm_receiver = create_mm_receiver(self.server_args, encode_urls=self.encoder_urls)
        logger.info(
            "Encode dispatch started for the language-only worker: bootstrap server on %s:%s, "
            "%d encoder URL(s) configured",
            host,
            port,
            len(self.encoder_urls),
        )

    def _ensure_runtime_context(self) -> None:
        """The receiver reads SGLang's process-wide config; the in-process rank
        launch publishes it before this runs, the spawned (off-main-thread)
        launch does not, so publish it here when nothing has."""
        try:
            from sglang.srt.runtime_context import publish, publish_role
        except ImportError:
            return
        if publish_role() is None:
            publish(self.server_args, role="tokenizer")

    def dispatch(
        self, rid: str, image_urls: list[str], on_error: Callable[[str, int], None]
    ) -> tuple[bool, dict[int, list[int]] | None, list[str] | None, Callable[[], None] | None]:
        """One request's dispatch, from the Rust server's blocking thread:
        what the receiver set on the request (``need_wait_for_mm_inputs``,
        ``num_items_assigned`` keyed by the modality's enum value,
        ``encoder_urls``) and the release the receiver's thread waits for
        before it reports a failure through ``on_error`` (``error_msg``,
        ``error_code``). With no encoder registered the receiver sets the wait
        flag off and dispatches nothing (its own warning says so); the Rust
        side refuses the request then."""
        if self.mm_receiver is None:
            raise RuntimeError("the encode dispatch was not started")
        from sglang.srt.managers.io_struct import GenerateReqInput

        request = GenerateReqInput(rid=rid, image_data=list(image_urls))
        dispatch_ready = self.mm_receiver.send_encode_request(
            request,
            on_dispatch_error=lambda error: on_error(
                str(getattr(error, "error_msg", error)), int(getattr(error, "error_code", 502))
            ),
        )
        try:
            assigned = getattr(request, "num_items_assigned", None)
            num_items_assigned = (
                {
                    int(getattr(modality, "value", modality)): [int(count) for count in counts]
                    for modality, counts in assigned.items()
                }
                if assigned
                else None
            )
            encoder_urls = getattr(request, "encoder_urls", None)
            return (
                bool(getattr(request, "need_wait_for_mm_inputs", False)),
                num_items_assigned,
                list(encoder_urls) if encoder_urls is not None else None,
                dispatch_ready.set if dispatch_ready is not None else None,
            )
        except BaseException:
            # The receiver's thread is parked on the event from the send on; an
            # answer that never reaches Rust releases it here, so the thread
            # reports instead of waiting forever.
            if dispatch_ready is not None:
                dispatch_ready.set()
            raise

    def shutdown(self) -> None:
        """Stop the bootstrap server; the receiver has nothing to stop (its
        dispatch threads are daemon threads, as behind SGLang's own frontend)."""
        server, self.bootstrap_server = self.bootstrap_server, None
        if server is not None:
            close = getattr(server, "close", None)
            if callable(close):
                close()
        self.mm_receiver = None
