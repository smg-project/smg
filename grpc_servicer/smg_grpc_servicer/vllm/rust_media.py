"""Worker-side media processing for the Rust servicer.

The Rust request path serves ``media_refs`` through the processors the Python
servicer runs (``--mm-processor inprocess``: vLLM's MediaConnector and the
engine's renderer; ``redis``: the sidecar), so the media is fetched and
processed by vLLM's own code on either servicer. :class:`RustMediaBridge`
hosts one of them next to a standalone vLLM ``InputProcessor`` (what an
AsyncLLM would run over the same engine input), runs each request on the
launcher's asyncio loop, and hands Rust what the engine needs: the prompt ids
with their placeholders expanded, and ``mm_features`` as vLLM's own msgpack
encoder writes them (primary buffer plus zero-copy tensor frames), which Rust
relays to the engine untouched.

Rust drives it through callbacks rather than awaiting Python: ``submit`` and
``probe`` schedule the work and call ``done`` when it finishes. The protocol
is the binding's (``smg.servicer.VllmGrpcServer(media_processor=...)``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any

import grpc

from smg_grpc_servicer.vllm.media_refs import MediaRefItem, validate_schemes
from smg_grpc_servicer.vllm.mm_processor import (
    MODE_INPROCESS,
    MODE_OFF,
    MmProcessorUnavailable,
    MmSettings,
    build_mm_processor,
)

logger = logging.getLogger(__name__)

# What `done` is told when the request failed; Rust maps these to statuses.
KIND_OK = "ok"
KIND_INVALID = "invalid"
KIND_UNAVAILABLE = "unavailable"
KIND_INTERNAL = "internal"

_RETRYABLE_CODES = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.RESOURCE_EXHAUSTED)


def lend_buffer(buffer: Any) -> Any:
    """A tensor-storage buffer (the encoder's zero-copy aux frame, a uint8
    memoryview) lent to Rust without a copy: a read-only, C-contiguous numpy
    view that keeps the storage alive; Rust reads the range off the view. The
    storage is the request's own (vLLM's processor built it for this request
    and nothing else refers to it), which is what makes lending it sound. The
    binding builds against Python's limited API, which has no buffer protocol,
    hence the view rather than the buffer itself."""
    import numpy as np

    view = np.ascontiguousarray(np.frombuffer(buffer, dtype=np.uint8))
    view.setflags(write=False)
    return view


# The Python servicer's exception-to-status mapping, resolved once: it needs
# vLLM's exception types, and a vLLM that moved them fails here, at launch,
# as it fails the Python servicer. Only an absent vLLM (the launcher's
# engine-free tests) falls back: a `ValueError` is the caller's, anything
# else is internal.
try:
    from smg_grpc_servicer.vllm.errors import grpc_code_for as _grpc_code_for
except ModuleNotFoundError as _missing:
    if _missing.name != "vllm":
        raise

    def _grpc_code_for(exc: BaseException) -> grpc.StatusCode:
        return (
            grpc.StatusCode.INVALID_ARGUMENT
            if isinstance(exc, ValueError)
            else grpc.StatusCode.INTERNAL
        )


class _EngineView:
    """What the processors read off an AsyncLLM: the config and its renderer."""

    def __init__(self, vllm_config: Any, renderer: Any) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.renderer = renderer


class RustMediaBridge:
    """One media processor, driven from Rust through the asyncio loop."""

    def __init__(
        self,
        processor: Any,
        input_processor: Any,
        encoder: Any,
        *,
        loop: asyncio.AbstractEventLoop,
        source: str,
        sampling_params: Callable[[], Any],
        supported_tasks: tuple[str, ...] = ("generate",),
        renderer: Any = None,
    ) -> None:
        self._processor = processor
        self._input_processor = input_processor
        self._encoder = encoder
        self._renderer = renderer
        self._loop = loop
        self._sampling_params = sampling_params
        self._supported_tasks = supported_tasks
        self.name: str = processor.name
        self.source: str = source
        self.max_inflight: int = int(processor.max_inflight)

    @property
    def schemes(self) -> str:
        """The schemes the processor accepts now; a sidecar announces its own
        set when probed, so this is read after each probe."""
        return self._processor.schemes

    @classmethod
    def build(
        cls, vllm_config: Any, settings: MmSettings, loop: asyncio.AbstractEventLoop
    ) -> RustMediaBridge | None:
        """The bridge for this engine's config and the launcher's ``--mm-*``
        settings, or ``None`` when worker-side processing is off (or the model
        takes no media). Raises ``ValueError`` on a setting vLLM cannot serve,
        as the Python servicer's constructor does."""
        resolved = settings.resolve()
        if resolved.processor == MODE_OFF:
            return None
        model_config = vllm_config.model_config
        if not getattr(model_config, "is_multimodal_model", False):
            logger.warning(
                "mm_processor=%s ignored: the served model is not multimodal", resolved.processor
            )
            return None
        from vllm import SamplingParams
        from vllm.renderers import renderer_from_config
        from vllm.v1.engine.input_processor import InputProcessor
        from vllm.v1.serial_utils import MsgpackEncoder

        # One renderer for both: the processor renders the media through it,
        # and the input processor's cache bookkeeping is its renderer's.
        renderer = renderer_from_config(vllm_config)
        processor = build_mm_processor(_EngineView(vllm_config, renderer), settings=resolved)
        if processor is None:
            return None
        return cls(
            processor,
            InputProcessor(vllm_config, renderer),
            MsgpackEncoder(),
            loop=loop,
            source=resolved.source,
            sampling_params=SamplingParams,
            renderer=renderer,
        )

    def start_warmup(self) -> None:
        """Run vLLM's multimodal processor warmup in the background, as its
        own frontend does once the engine process exists: the first requests
        otherwise pay the processor's JIT and cache priming (seconds of CPU)
        on the serving path. Queued on the renderer's single-worker executor,
        so it never overlaps a request."""
        if self.name != MODE_INPROCESS:
            return  # the sidecar runs the processor; nothing here would use the warmup
        start = getattr(self._renderer, "start_mm_warmup_in_background", None)
        if start is None:
            logger.debug("renderer has no start_mm_warmup_in_background; skipping the warmup")
            return
        try:
            start()
        except Exception as e:  # noqa: BLE001 - warmup is an optimisation
            logger.warning("Multimodal processor warmup not started: %s", e)

    # -- the protocol Rust drives ------------------------------------------

    def probe(self, done: Callable[[bool, str], None]) -> None:
        """Ask the processor whether it serves; ``done(serving, schemes)``
        answers with the schemes it accepts as of this probe."""

        def finish(future: Future) -> None:
            try:
                serving = bool(future.result())
            except Exception as e:  # noqa: BLE001 - a failing probe means "not serving"
                logger.warning("media processor probe failed: %s", e)
                serving = False
            done(serving, self.schemes)

        self._schedule(self._processor.probe(), finish)

    def submit(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        prompt_text: str | None,
        items: Sequence[tuple[str, str]],
        arrival_time: float,
        want_identity: bool,
        done: Callable[[str, Any], None],
    ) -> None:
        """Process one request's references; ``done(kind, payload)`` answers
        with ``("ok", (prompt_token_ids, mm_features, aux_frames, cache_salt,
        media_identity))`` or ``(kind, message)`` for a failure."""
        work = self._process(
            request_id, list(prompt_token_ids), prompt_text, items, arrival_time, want_identity
        )
        self._schedule(work, lambda future: self._finish(request_id, future, done))

    # -- internals ----------------------------------------------------------

    def _schedule(self, work: Any, finish: Callable[[Future], None]) -> None:
        future = asyncio.run_coroutine_threadsafe(work, self._loop)
        future.add_done_callback(finish)

    async def _process(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        prompt_text: str | None,
        items: Sequence[tuple[str, str]],
        arrival_time: float,
        want_identity: bool,
    ) -> tuple[list[int], bytes | None, list[Any], str | None, bytes | None]:
        refs = [MediaRefItem(modality=modality, url=url) for modality, url in items]
        validate_schemes(refs, self._processor.accepted_schemes)
        engine_input = await self._processor.process(
            prompt_token_ids, prompt_text, refs, arrival_time, request_id=request_id
        )
        identity = self._identity(request_id, engine_input) if want_identity else None
        # Inline on the loop, as AsyncLLM runs it for a rendered engine input:
        # no blocking work is left at this point, and a thread hop here costs
        # two GIL handoffs per request behind the busy processor thread.
        core = self._input_processor.process_inputs(
            request_id,
            engine_input,
            self._sampling_params(),
            self._supported_tasks,
            arrival_time=arrival_time,
        )
        mm_features: bytes | None = None
        aux_frames: list[Any] = []
        if core.mm_features:
            buffers = self._encoder.encode(core.mm_features)
            mm_features = bytes(buffers[0])
            aux_frames = [lend_buffer(buffer) for buffer in buffers[1:]]
        return (
            list(core.prompt_token_ids),
            mm_features,
            aux_frames,
            getattr(core, "cache_salt", None),
            identity,
        )

    @staticmethod
    def _identity(request_id: str, engine_input: Any) -> bytes | None:
        """The PD prefill leg's media identity, serialized; an optimisation
        with a fallback (the decode leg reprocesses), so a shape it cannot
        read must not fail a served request."""
        # Imported here: the identity module needs torch, and this module
        # must import on the launcher's engine-free path.
        from smg_grpc_servicer.vllm.media_identity import (
            build_media_identity,
            media_identity_supported,
        )

        if not media_identity_supported():
            logger.warning(
                "Request %s: the installed smg-grpc-proto has no media_identity; "
                "the decode leg will reprocess the media",
                request_id,
            )
            return None
        try:
            identity = build_media_identity(engine_input)
        except Exception as e:  # noqa: BLE001 - any failure falls back
            logger.warning(
                "Request %s: media identity not built (%s); the decode leg will reprocess "
                "the media",
                request_id,
                e,
            )
            return None
        return identity.SerializeToString() if identity is not None else None

    @staticmethod
    def _finish(request_id: str, future: Future, done: Callable[[str, Any], None]) -> None:
        """Classify the outcome the way the Python servicer maps it to a
        status: retryable failures UNAVAILABLE, the caller's INVALID_ARGUMENT,
        anything else INTERNAL."""
        try:
            result = future.result()
        except MmProcessorUnavailable as e:
            logger.warning("Media processing unavailable for request %s: %s", request_id, e)
            done(KIND_UNAVAILABLE, str(e))
        except Exception as e:  # noqa: BLE001 - every failure must reach the caller
            code = _grpc_code_for(e)
            if code is grpc.StatusCode.INTERNAL:
                logger.exception("Media processing failed for request %s", request_id)
                done(KIND_INTERNAL, str(e))
            elif code in _RETRYABLE_CODES:
                logger.warning("Media processing unavailable for request %s: %s", request_id, e)
                done(KIND_UNAVAILABLE, str(e))
            else:
                logger.warning("Media of request %s rejected (%s): %s", request_id, code.name, e)
                done(KIND_INVALID, str(e))
        else:
            done(KIND_OK, result)
