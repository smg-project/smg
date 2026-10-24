# mypy: ignore-errors
"""
vLLM gRPC Servicer

Implements the VllmEngine gRPC service on top of vLLM's EngineClient.
"""

import asyncio
import itertools
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

import grpc
import torch
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_proto.generated import common_pb2
from transformers import BatchFeature
from vllm import PoolingParams, SamplingParams, TokensPrompt
from vllm.engine.protocol import EngineClient
from vllm.inputs.engine import MultiModalInput as VllmMultiModalInput
from vllm.inputs.engine import mm_input, tokens_input
from vllm.logger import init_logger
from vllm.logprobs import PromptLogprobs, SampleLogprobs
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams

from smg_grpc_servicer.kv_relay import Engine, relay
from smg_grpc_servicer.tokenizer_bundle import tokenizer_bundle, tokenizer_chunks
from smg_grpc_servicer.trace_context import trace_headers
from smg_grpc_servicer.vllm import attach_vllm_logging, settle_logging
from smg_grpc_servicer.vllm.admin import flush_cache, start_profile, stop_profile
from smg_grpc_servicer.vllm.config_view import EngineConfigView, uses_mrope
from smg_grpc_servicer.vllm.errors import GracefulHTTPError, VLLMClientError, grpc_code_for
from smg_grpc_servicer.vllm.kv_events import (
    rank_sources_for,
    resolve_kv_events_config,
)
from smg_grpc_servicer.vllm.kv_transfer import (
    params_from_request,
    params_to_response_fields,
)

# The launcher imports this module before it defines serve_grpc: the moment
# the servicer switch has to be in place (see launcher_switch).
from smg_grpc_servicer.vllm.launcher_switch import install_launcher_switch
from smg_grpc_servicer.vllm.loads import LoadTracker, load_record, scheduler_load_fields
from smg_grpc_servicer.vllm.media_identity import build_media_identity, media_identity_supported
from smg_grpc_servicer.vllm.media_refs import parse_media_refs, validate_schemes
from smg_grpc_servicer.vllm.mm_processor import (
    ENV_PROCESSOR,
    PROCESSOR_FLAG,
    MmProcessorUnavailable,
    MmSettings,
    build_mm_processor,
    launcher_settings,
)
from smg_grpc_servicer.vllm.mm_salt import (
    engine_accepts_mm_inputs,
    has_preprocessed_mm_payload,
    mm_identity_cache_salt,
)
from smg_grpc_servicer.vllm.mm_tensors import tensor_from_proto
from smg_grpc_servicer.vllm.model_info import (
    constrained_decoding_start,
    engine_reasoning_parser,
    mm_device_do_normalize,
    mm_encoder_budget,
    mm_item_limits,
    model_facts,
    running_window,
    server_facts,
)
from smg_grpc_servicer.vllm.warmup import WARMUP_ENV, start_engine_warmup

from .mm_keys import (
    batches_missing_pixels,
    describes_media_twice,
    mm_batches,
    mm_identity_hashes,
    modality_key,
    modality_name,
    primary_encoder_key,
)

install_launcher_switch()

logger = init_logger(__name__)
attach_vllm_logging()
try:
    from vllm.version import __version__ as VLLM_VERSION
except Exception:  # pragma: no cover - version lookup is best-effort
    VLLM_VERSION = ""


def _prompt_length(prompt) -> int:
    """The token count of an engine prompt (0 when it is text the engine tokenizes)."""
    if isinstance(prompt, dict):
        ids = prompt.get("prompt_token_ids")
        return len(ids) if ids is not None else 0
    return 0


def _kv_capacity_tokens(engine) -> int:
    """The KV cache's token capacity, when the engine config exposes it."""
    cache = EngineConfigView(getattr(engine, "vllm_config", None)).cache
    if cache.num_gpu_blocks is not None and cache.block_size is not None:
        return cache.num_gpu_blocks * cache.block_size
    return 0


def _max_running_requests(engine) -> int:
    """The scheduler's running window (``max_num_seqs``), when exposed."""
    return running_window(getattr(engine, "vllm_config", None))


def _managed_ranks(engine) -> list[int]:
    """The data-parallel ranks this servicer serves, in order: the engine
    client's managed ranks (a slice of all the ranks under external or
    hybrid load balancing, from ``data_parallel_rank`` on), else that start
    rank from the config alone, else rank 0. ``GetLoads`` reports each of
    them, as the Rust servicer's does."""
    managed = getattr(getattr(engine, "engine_core", None), "engine_ranks_managed", None)
    if isinstance(managed, (list, tuple, range)):
        ranks = sorted({rank for rank in managed if isinstance(rank, int)})
        if ranks:
            return ranks
    return [EngineConfigView(getattr(engine, "vllm_config", None)).parallel.data_parallel_rank]


def _first_rank(engine) -> int:
    """The lowest data-parallel rank this servicer serves (the first of
    :func:`_managed_ranks`): the rank a request that names none is counted
    on, and the one whose load entry is zero-filled before its first
    snapshot rather than left out."""
    return _managed_ranks(engine)[0]


def _rank_in_flight(engine, request_id: str, choice: int = 0) -> int | None:
    """The data-parallel rank vLLM's load-balancing engine client routed
    ``request_id`` to, while the request is in flight: for an ``n > 1``
    request the rank of its ``choice``, each choice being a request of its
    own to the engine. The client keeps the engine of every request it
    routed, to route its abort (``reqs_in_flight``, keyed by the engine's
    internal request id, valued with the engine's two-byte identity, which
    is its rank), and the output processor keeps the internal ids under the
    id the caller gave (``external_req_ids``; a choice's id starts with its
    index). ``None`` when the client keeps no such table (one engine), the
    request is not routed yet, or it has finished."""
    table = getattr(getattr(engine, "engine_core", None), "reqs_in_flight", None)
    names = getattr(getattr(engine, "output_processor", None), "external_req_ids", None)
    if not isinstance(table, dict) or not isinstance(names, dict):
        return None
    internal_ids = list(names.get(request_id) or ())
    own = [internal_id for internal_id in internal_ids if internal_id.startswith(f"{choice}_")]
    for internal_id in own + [
        internal_id for internal_id in internal_ids if internal_id not in own
    ]:
        identity = table.get(internal_id)
        if isinstance(identity, (bytes, bytearray)) and len(identity) == 2:
            return int.from_bytes(identity, "little")
    return None


def _routed_rank(engine, request_id: str, choice: int = 0) -> int | None:
    """The rank an unpinned request (its ``choice``) went to, for its queued
    prompt: the one the engine's load-balancing client routed it to (see
    :func:`_rank_in_flight`), this servicer's first rank when the client
    keeps no routing table (one engine: every request is its), ``None``
    while the request is not routed yet."""
    if not isinstance(getattr(getattr(engine, "engine_core", None), "reqs_in_flight", None), dict):
        return _first_rank(engine)
    return _rank_in_flight(engine, request_id, choice)


def _request_rank(
    engine,
    request_id: str,
    pinned: int | None,
    choice: int = 0,
    remembered: int | None = None,
) -> int:
    """The rank serving ``request_id`` (its ``choice``) from its first output
    on, for the load tracker: the one the engine's client routed it to while
    it is in flight (a pinned request is in that table too); else the rank
    the tracker learned for it while it was queued (``remembered``: a
    request the engine has already retired, whose only output was its last,
    is in neither table any more); else the rank the request named; else
    this servicer's first rank."""
    rank = _routed_rank(engine, request_id, choice)
    if rank is not None:
        return rank
    if remembered is not None:
        return remembered
    if pinned is not None:
        return int(pinned)
    return _first_rank(engine)


def _latest_scheduler_stats(engine, engine_idx: int = 0):
    """Best-effort read of the most recent ``SchedulerStats`` snapshot.

    vLLM has no synchronous "current stats" accessor on ``AsyncLLM``/``EngineClient``;
    ``SchedulerStats`` arrive asynchronously and are cached on the stat loggers. This
    reaches into ``engine.logger_manager.stat_loggers`` and returns the freshest
    snapshot, handling the logger-shape variants:

    - ``LoggingStatLogger``                  -> ``.last_scheduler_stats`` (its own
      engine's, ``engine_index``; another index reads nothing from it)
    - ``AggregatedLoggingStatLogger`` (DP)   -> ``.last_scheduler_stats_dict[idx]``
    - ``PerEngineStatLoggerAdapter``         -> ``.per_engine_stat_loggers[idx]``
    - ``PrometheusStatLogger``               -> skipped (no cached snapshot)

    Returns ``None`` when stats logging is disabled (``--disable-log-stats``) or no
    engine step has produced outputs yet.
    """
    logger_manager = getattr(engine, "logger_manager", None)
    if logger_manager is None:
        return None
    for sl in getattr(logger_manager, "stat_loggers", None) or []:
        per = getattr(sl, "last_scheduler_stats_dict", None)
        if isinstance(per, dict) and engine_idx in per:
            return per[engine_idx]
        stats = getattr(sl, "last_scheduler_stats", None)
        if stats is not None and getattr(sl, "engine_index", 0) == engine_idx:
            return stats
        per_engine = getattr(sl, "per_engine_stat_loggers", None)
        if isinstance(per_engine, dict) and engine_idx in per_engine:
            nested = getattr(per_engine[engine_idx], "last_scheduler_stats", None)
            if nested is not None:
                return nested
    return None


class VllmEngineServicer(vllm_engine_pb2_grpc.VllmEngineServicer):
    """
    gRPC servicer implementing the VllmEngine service.

    Handles 8 RPCs:
    - Generate: Streaming text generation
    - Embed: Embeddings
    - HealthCheck: Health probe
    - Abort: Cancel requests out-of-band
    - GetModelInfo: Model metadata
    - GetServerInfo: Server state
    - GetLoads: Scheduler load metrics
    - GetTokenizer: Stream tokenizer artifacts
    """

    def __init__(
        self,
        async_llm: EngineClient,
        start_time: float,
        mm_settings: MmSettings | None = None,
    ):
        """
        Initialize the servicer.

        Args:
            async_llm: The EngineClient instance (e.g. AsyncLLM)
            start_time: The server start time, in seconds since epoch
            mm_settings: The launcher's `--mm-*` flags; None takes the
                settings the plugin kept from this process's launcher parse
                (upstream's launcher builds the servicer without its
                namespace), else an older launcher resolves everything from
                the environment
        """
        # The Rust path takes the process before this class exists; reaching
        # here with the flag set means the launcher never consulted it.
        from smg_grpc_servicer.vllm.rust import require_python_impl

        require_python_impl()
        # vLLM 0.31+ configures logging after this module is imported: attach
        # again, and end the hold of what was logged before (the launcher
        # configured logging by now, if it was going to).
        settle_logging()
        self.engine = async_llm
        self.start_time = start_time
        # Resolve KV-event publishing config from the engine. Non-None only when
        # vLLM was started with --kv-events-config enabling the ZMQ publisher.
        self._kv_events_config = resolve_kv_events_config(async_llm)
        # Queued token-work, generation throughput and hit rate for GetLoads,
        # per rank, from the requests this servicer forwards (vLLM's stats
        # carry none); an unpinned request's queued prompt counts on the rank
        # the engine routes it to, read once it has.
        self._loads = LoadTracker(rank_of=lambda request_id: _routed_rank(async_llm, request_id))
        # Flag > env > default, resolved once so each value names its source.
        self._mm_settings = (mm_settings or launcher_settings() or MmSettings()).resolve()
        # The engine's encoder budget, derived once (vLLM's helper builds the
        # largest item the model's processor can produce) and advertised so
        # the Router's media pipeline refuses a larger item before the engine.
        self._mm_encoder_budget = mm_encoder_budget(getattr(async_llm, "vllm_config", None))
        # Worker-side media processing (media_refs); None keeps refs rejected.
        self._mm_processor = build_mm_processor(async_llm, settings=self._mm_settings)
        # One cap over all the multimodal work this servicer runs off the event
        # loop, whether it fetches the media itself or converts tensors the
        # router already prepared. Both are sized by the same setting, so a
        # worker's memory ceiling does not depend on which path a request takes.
        self._mm_limit = (
            self._mm_processor.max_inflight
            if self._mm_processor is not None
            else self._mm_settings.max_inflight
        )
        self._mm_inflight = asyncio.Semaphore(self._mm_limit)
        self._mm_waiting = 0
        self._unhealthy_logged = False
        logger.info(
            "VllmEngineServicer initialized (mm_processor=%s, source=%s)",
            self._mm_processor.name if self._mm_processor is not None else "off",
            self._mm_settings.source,
        )
        # The planned warm-up (:mod:`smg_grpc_servicer.vllm.warmup`): in a
        # launcher's process (the plugin kept its parse, the serving coroutine
        # runs) the plan runs over the engine on the event loop and the health
        # service holds NOT_SERVING until it is over; a servicer built
        # elsewhere (a test, a harness) warms nothing.
        self._warmup = None
        if launcher_settings() is not None:
            self._warmup = start_engine_warmup(async_llm, setting_text=os.environ.get(WARMUP_ENV))

    async def _acquire_mm_slot(self) -> None:
        """Take one multimodal slot, shedding load instead of queueing forever.

        Once as many requests are waiting as the worker can run at once, the
        ones behind them will not be reached before their callers give up. A
        retryable refusal now sends them to a worker that can take them, rather
        than a deadline later that tells the caller nothing about where to go.
        """
        if self._mm_waiting >= self._mm_limit:
            raise MmProcessorUnavailable(
                f"worker is saturated: {self._mm_limit} multimodal requests in flight "
                f"and as many waiting"
            )
        self._mm_waiting += 1
        try:
            await self._mm_inflight.acquire()
        finally:
            self._mm_waiting -= 1

    async def _off_the_event_loop(self, work, *args):
        """Run one piece of blocking multimodal work under the in-flight cap.

        A worker thread cannot be interrupted, so the slot is handed back when
        the thread itself finishes rather than when the caller stops waiting.
        A caller that goes away while the work is still running therefore does
        not let the next one start against memory that is still held.
        """
        await self._acquire_mm_slot()
        running = asyncio.ensure_future(asyncio.to_thread(work, *args))
        # Kept so the callback below can tell whether a caller is still waiting.
        shielded = asyncio.shield(running)

        def finished(done: asyncio.Future) -> None:
            self._mm_inflight.release()
            if done.cancelled():
                return
            # Nobody may be left to read a failure; take it here so the
            # loop does not report it as never retrieved.
            error = done.exception()
            if error is not None and shielded.cancelled():
                # A cancelled caller has no RPC left to fail, so the log is the
                # only place this failure can still be seen.
                logger.error(
                    "Multimodal work %s failed after its caller went away",
                    getattr(work, "__name__", work),
                    exc_info=error,
                )

        running.add_done_callback(finished)
        return await shielded

    async def Generate(
        self,
        request: vllm_engine_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[vllm_engine_pb2.GenerateResponse, None]:
        """
        Handle streaming generation requests.

        Supports n>1 by sending separate chunk/complete messages for each output index.
        When streaming with n>1, chunks for different indices are interleaved.

        Args:
            request: The GenerateRequest protobuf
            context: gRPC context

        Yields:
            GenerateResponse protobuf messages (streaming)
        """
        request_id = request.request_id
        input_type = request.WhichOneof("input")
        # One batch per modality: `mm_inputs`, then `extra_mm_inputs` for a
        # request that mixes image and video. A pixel-less mm payload with
        # grid tensors is the PD decode side's form: enough to rebuild mm
        # features (positions + block hashing).
        preprocessed_batches = [
            batch for batch in mm_batches(request) if has_preprocessed_mm_payload(batch)
        ]
        has_preprocessed_mm = bool(preprocessed_batches)
        media_ref_count = len(request.media_refs.items) if request.HasField("media_refs") else 0
        logger.info(
            "Generate request %s: input_type=%s, stream=%s, preprocessed_mm=%s, "
            "media_refs=%d, dp_rank=%s",
            request_id,
            input_type,
            request.stream,
            has_preprocessed_mm,
            media_ref_count,
            request.data_parallel_rank if request.HasField("data_parallel_rank") else None,
        )

        kv_transfer_params: dict | None = None
        # What a PD prefill side learned about its media, for the decode side.
        media_identity = None
        engine_started = False
        handed_to_engine = False
        try:
            arrival_time = time.time()
            kv_transfer_params = params_from_request(request)

            if request.HasField("media_refs"):
                # Media references from the router: the worker fetches and runs
                # vLLM's own processor over the unexpanded placeholder anchors.
                if input_type != "tokenized" or describes_media_twice(request):
                    raise ValueError(
                        "media_refs requires tokenized input and cannot be combined with "
                        "preprocessed multimodal inputs"
                    )
                if self._mm_processor is None:
                    raise ValueError(
                        f"media_refs sent but {PROCESSOR_FLAG} ({ENV_PROCESSOR}) is off on this "
                        "worker; check the router's --mm-processing and this worker's "
                        "mm_processor label"
                    )
                items = parse_media_refs(request.media_refs)
                validate_schemes(items, self._mm_processor.accepted_schemes)
                await self._acquire_mm_slot()
                try:
                    prompt = await self._mm_processor.process(
                        list(request.tokenized.input_ids),
                        request.tokenized.original_text or None,
                        items,
                        arrival_time,
                        request_id=request_id,
                    )
                finally:
                    self._mm_inflight.release()
                # A PD prefill side answers with the identity so decode is
                # served without pixels or references. The identity is an
                # optimisation with a fallback (decode reprocesses), so a
                # shape it cannot read must not fail a served request.
                if kv_transfer_params is not None:
                    if media_identity_supported():
                        try:
                            media_identity = build_media_identity(prompt)
                        except Exception as e:  # noqa: BLE001 - any failure falls back
                            logger.warning(
                                "Request %s: media identity not built (%s); the decode side "
                                "will reprocess the media",
                                request_id,
                                e,
                            )
                            media_identity = None
                    else:
                        logger.warning(
                            "Request %s: the installed smg-grpc-proto has no media_identity; "
                            "the decode side will reprocess the media",
                            request_id,
                        )
            elif has_preprocessed_mm and input_type == "tokenized":
                # A pixel-less payload (PD decode side) is only decodable with
                # remote KV: a local recompute would schedule the vision
                # encoder with no pixels and crash the engine.
                # Every batch needs its own, not just one of them: a batch left
                # out contributes no encoder tensor, and the engine would run
                # that modality's encoder with nothing to encode.
                if batches_missing_pixels(preprocessed_batches) and kv_transfer_params is None:
                    logger.warning(
                        "Request %s: pixel-less multimodal payload with no kv_transfer_params; "
                        "rejecting (prefill worker did not hand off KV?)",
                        request_id,
                    )
                    raise ValueError(
                        "multimodal payload carries grid tensors but no pixel_values and "
                        "no kv_transfer_params; a pixel-less leg requires remote KV"
                    )
                # Preprocessed multimodal from Rust router.
                # Token IDs already have expanded placeholders; tensors are
                # ready for the model. Bypass the renderer entirely. The tensor
                # copies and dtype casts scale with the payload (hundreds of
                # megabytes for a many-image request), so they run off the
                # event loop and health checks keep being answered meanwhile.
                prompt = await self._off_the_event_loop(
                    self._build_preprocessed_mm_inputs, request.tokenized, preprocessed_batches
                )
                prompt["arrival_time"] = arrival_time
            elif input_type == "tokenized":
                prompt: TokensPrompt = {"prompt_token_ids": list(request.tokenized.input_ids)}
                if request.tokenized.original_text:
                    prompt["prompt"] = request.tokenized.original_text
                # Tensor-less mm payload (grid-less PD decode side): fold the
                # kept mm hashes into cache_salt so different images cannot
                # alias. Grid-carrying sides took the preprocessed path above.
                batches = mm_batches(request)
                if batches:
                    cache_salt = mm_identity_cache_salt(mm_identity_hashes(batches))
                    if cache_salt is not None:
                        prompt["cache_salt"] = cache_salt
                    if uses_mrope(getattr(self.engine, "model_config", None)):
                        logger.warning(
                            "Request %s carries mm identity but no grid tensors on an "
                            "M-RoPE model; decode-side positions will be text-only",
                            request_id,
                        )
                prompt = self.engine.renderer.process_for_engine(prompt, arrival_time=arrival_time)
            else:
                prompt = request.text

            # Build sampling params with detokenize=False
            sampling_params = self._sampling_params_from_proto(
                request.sampling_params,
                stream=request.stream,
                kv_transfer_params=kv_transfer_params,
            )
            tokenization_kwargs = self._tokenization_kwargs_from_proto(request.sampling_params)

            # Extract logprobs configuration
            num_logprobs = sampling_params.logprobs
            num_prompt_logprobs = sampling_params.prompt_logprobs

            # Track which indices have sent their first chunk
            seen_indices: set[int] = set()

            pinned_rank = (
                request.data_parallel_rank if request.HasField("data_parallel_rank") else None
            )
            # The rank the request goes to: the pin, else the engine's choice,
            # which the tracker reads once the engine has routed it (at every
            # load read while the request is queued) and this loop reads when
            # a choice first shows up (each choice of an n > 1 request is
            # routed on its own); a non-streamed request's only output comes
            # once the engine has retired it, so the tracker's reading is
            # what names its rank then.
            self._loads.submitted(request_id, _prompt_length(prompt), rank=pinned_rank)
            ranks: dict[int, int] = {}
            remembered: int | None = None

            def rank_of(choice: int) -> int:
                nonlocal remembered
                if choice not in ranks:
                    if not ranks:
                        # Read once, at the first choice: the first output
                        # takes the request out of the tracker's queue, and
                        # the other choices of a retired request need it too.
                        remembered = self._loads.pending_rank(request_id)
                    ranks[choice] = _request_rank(
                        self.engine, request_id, pinned_rank, choice, remembered=remembered
                    )
                return ranks[choice]

            # From here the engine owns the request: a caller that gives up
            # is answered by the engine's own abort, and the pre-admission
            # notice below must not be sent for an id the engine may still
            # hold (see _notify_kv_transfer_rejected). The one exception is
            # a refusal generate() raises before the engine core holds the
            # request, told apart in the except branch below.
            handed_to_engine = True
            async for output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers(context.invocation_metadata()),
                data_parallel_rank=pinned_rank,
            ):
                if not engine_started:
                    # The engine holds the request from its first output on
                    # (a refusal below never gets here). The prompt's cached
                    # count is the reporting choice's (the first output of an
                    # n > 1 request is whichever choice stepped first).
                    engine_started = True
                    self._loads.first_output(
                        request_id,
                        len(output.prompt_token_ids or ()),
                        getattr(output, "num_cached_tokens", 0) or 0,
                        rank=rank_of(output.outputs[0].index if output.outputs else 0),
                    )
                for completion in output.outputs:
                    self._loads.generated(len(completion.token_ids), rank=rank_of(completion.index))
                # For streaming, send chunks for EACH completion output (n outputs)
                if request.stream:
                    for completion in output.outputs:
                        idx = completion.index
                        is_first = idx not in seen_indices
                        seen_indices.add(idx)

                        # Send chunk with delta data (Rust accumulates for vLLM)
                        yield self._chunk_response(
                            output,
                            completion=completion,
                            num_logprobs=num_logprobs,
                            num_prompt_logprobs=num_prompt_logprobs,
                            is_first_chunk=is_first,
                        )

                        # Send Complete when sequence finishes (n>1 support)
                        if completion.finish_reason:
                            yield self._complete_response(
                                output,
                                completion=completion,
                                num_logprobs=num_logprobs,
                                num_prompt_logprobs=num_prompt_logprobs,
                                media_identity=media_identity,
                            )

                # For non-streaming, send complete response when finished
                if output.finished and not request.stream:
                    for completion in output.outputs:
                        yield self._complete_response(
                            output,
                            completion=completion,
                            num_logprobs=num_logprobs,
                            num_prompt_logprobs=num_prompt_logprobs,
                            media_identity=media_identity,
                        )

        except asyncio.CancelledError:
            # A caller that gives up while media is still being fetched leaves
            # the same blocks pinned as any other pre-admission failure, and the
            # cleanup has to outlive a second cancellation to land.
            await asyncio.shield(
                self._notify_kv_transfer_rejected(request_id, kv_transfer_params, handed_to_engine)
            )
            raise
        except MmProcessorUnavailable as e:
            # Retryable: the router re-selects a worker on UNAVAILABLE.
            logger.warning("Media processing unavailable for request %s: %s", request_id, e)
            await self._notify_kv_transfer_rejected(
                request_id, kv_transfer_params, handed_to_engine
            )
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except Exception as e:
            code = grpc_code_for(e)
            if code is grpc.StatusCode.INTERNAL:
                logger.exception("Error in Generate for request %s", request_id)
            else:
                logger.warning("Generate request %s rejected (%s): %s", request_id, code.name, e)
            # A refusal before the first output is generate()'s rejection of
            # the request's inputs (the prompt plus max_tokens over the model
            # length, bad sampling params: a VLLMClientError, the ValueError
            # of releases before 0.27) or of its admission (GracefulHTTPError):
            # both are raised before the engine core holds the request and
            # generate() re-raises them with no abort, so no live id is left
            # behind and the notice is still owed, as the Rust wire sends it
            # when its generate call returns an error. Every other failure
            # after the hand-over reaches here through the engine's own abort,
            # which answers it; an EngineGenerateError that wraps a ValueError
            # could come from either side and stays without the notice.
            refused = (
                isinstance(e, (VLLMClientError, GracefulHTTPError, ValueError))
                and not engine_started
            )
            await self._notify_kv_transfer_rejected(
                request_id, kv_transfer_params, handed_to_engine and not refused
            )
            await context.abort(code, str(e))
        finally:
            self._loads.finished(request_id)

    async def _notify_kv_transfer_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict | None,
        handed_to_engine: bool,
    ) -> None:
        """Free remote prefill blocks early when a decode request dies pre-admission.

        Without this, the prefill engine keeps the blocks pinned until the NIXL
        lease expires (30s default).

        Only a request the engine never received is notified (a refusal
        ``generate()`` raises before the engine core holds the request counts
        as never received; the caller tells it apart). The notice is a
        second engine request under the same id, submitted pre-aborted so the
        connector's request_finished hook runs, and once ``engine.generate``
        has been entered it kills the engine either way: while the request's
        KV is still being pulled the engine keeps the aborted request tracked
        until the pull completes and refuses the duplicate id; once the pull is
        done the notice's own request is freed in the step that admits it, and
        a connector that reports its empty pull (the prefill-block release) as
        a finished transfer names an id the scheduler no longer tracks. The
        engine's own abort answers a caller that gives up after admission; the
        prefill side's lease covers the window in which the engine never got
        the request.
        """
        if handed_to_engine or not kv_transfer_params:
            return
        if not kv_transfer_params.get("do_remote_prefill"):
            return
        # Older vLLM releases lack this hook; the lease expiry covers them
        notify = getattr(self.engine, "notify_kv_transfer_request_rejected", None)
        if notify is None:
            return
        try:
            await notify(request_id, kv_transfer_params)
        except Exception:
            logger.warning(
                "Failed to notify KV connector about rejected request %s",
                request_id,
                exc_info=True,
            )

    async def Embed(
        self,
        request: vllm_engine_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.EmbedResponse:
        """
        Handle embedding requests.

        Calls vLLM's encode() API with PoolingParams and returns the embedding vector.

        Args:
            request: The EmbedRequest protobuf
            context: gRPC context

        Returns:
            EmbedResponse protobuf
        """
        request_id = request.request_id
        logger.info("Embed request %s", request_id)

        try:
            if not request.HasField("tokenized"):
                raise ValueError("EmbedRequest requires tokenized input")

            prompt = tokens_input(
                prompt_token_ids=list(request.tokenized.input_ids),
                prompt=request.tokenized.original_text or None,
            )

            pooling_params = PoolingParams(task="embed")

            # encode() is an async generator; collect the final result
            final_output = None
            async for output in self.engine.encode(
                prompt=prompt,
                pooling_params=pooling_params,
                request_id=request_id,
            ):
                final_output = output

            if final_output is None or not final_output.finished:
                msg = f"Embed request {request_id} did not produce a result"
                logger.warning(msg)
                await context.abort(grpc.StatusCode.INTERNAL, msg)

            embedding = final_output.outputs.data.tolist()

            return vllm_engine_pb2.EmbedResponse(
                embedding=embedding,
                prompt_tokens=len(final_output.prompt_token_ids),
                embedding_dim=len(embedding),
            )

        except grpc.aio.AbortError:
            raise
        except Exception as e:
            code = grpc_code_for(e)
            if code is grpc.StatusCode.INTERNAL:
                logger.exception("Embed failed for request %s", request_id)
            else:
                logger.warning("Embed request %s rejected (%s): %s", request_id, code.name, e)
            await context.abort(code, str(e))

    async def HealthCheck(
        self,
        request: vllm_engine_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.HealthCheckResponse:
        """
        Handle health check requests.

        Args:
            request: The HealthCheckRequest protobuf
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf
        """
        is_healthy = not self.engine.errored
        message = "Health" if is_healthy else "Engine is not alive"

        # Probes arrive on a fixed interval and the answer is engine state the
        # caller already receives, so only the turn for the worse is worth a
        # line, and only the first one: the engine does not come back.
        if not is_healthy and not self._unhealthy_logged:
            self._unhealthy_logged = True
            logger.error("HealthCheck is now reporting unhealthy: %s", message)

        return vllm_engine_pb2.HealthCheckResponse(healthy=is_healthy, message=message)

    async def FlushCache(
        self,
        request: common_pb2.FlushCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.FlushCacheResponse:
        return await flush_cache(self.engine, request, context)

    async def StartProfile(
        self,
        request: common_pb2.StartProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.ProfileResponse:
        return await start_profile(self.engine, request, context)

    async def StopProfile(
        self,
        request: common_pb2.StopProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.ProfileResponse:
        return await stop_profile(self.engine, context)

    async def Abort(
        self,
        request: vllm_engine_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.AbortResponse:
        """
        Out-of-band abort requests.

        Args:
            request: The AbortRequest protobuf
            context: gRPC context

        Returns:
            AbortResponse protobuf
        """
        request_ids = request.request_ids
        logger.info("Abort requests: %s", request_ids)

        try:
            await self.engine.abort(request_ids)
        except Exception as e:
            code = grpc_code_for(e)
            if code is grpc.StatusCode.INTERNAL:
                logger.exception("Abort failed for requests %s", request_ids)
            else:
                logger.warning("Abort rejected (%s) for requests %s: %s", code.name, request_ids, e)
            await context.abort(code, str(e))
        return vllm_engine_pb2.AbortResponse()

    async def GetModelInfo(
        self,
        request: vllm_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetModelInfoResponse:
        """
        Handle model info requests.

        Args:
            request: The GetModelInfoRequest protobuf
            context: gRPC context

        Returns:
            GetModelInfoResponse protobuf
        """
        facts = model_facts(self.engine.model_config)
        return vllm_engine_pb2.GetModelInfoResponse(
            max_req_input_len=facts["max_context_length"], **facts
        )

    async def GetServerInfo(
        self,
        request: vllm_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetServerInfoResponse:
        """
        Handle server info requests.

        Args:
            request: The GetServerInfoRequest protobuf
            context: gRPC context

        Returns:
            GetServerInfoResponse protobuf
        """
        facts = server_facts(self.engine.vllm_config)
        # Older proto packages omit newer metadata fields; set them below
        # only when their descriptors support them.
        max_num_seqs = facts.pop("max_num_seqs", 0)
        encoder_dtype = facts.pop("multimodal_encoder_dtype", "")
        mm_processor = ""
        mm_media_ref_schemes = ""
        # A --language-model-only engine accepts no multimodal inputs, so it
        # must not advertise worker-side media processing either: the router
        # would send media references this worker cannot expand.
        if (
            self._mm_processor is not None
            and engine_accepts_mm_inputs(self.engine.model_config)
            and await self._mm_processor.probe()
        ):
            mm_processor = self._mm_processor.name
            mm_media_ref_schemes = self._mm_processor.schemes

        info = vllm_engine_pb2.GetServerInfoResponse(
            mm_processor=mm_processor,
            mm_media_ref_schemes=mm_media_ref_schemes,
            **facts,
        )
        # Where the processor mode came from, for the gateway's /workers; a
        # proto package predating the field simply leaves it out.
        if mm_processor and "mm_processor_source" in info.DESCRIPTOR.fields_by_name:
            info.mm_processor_source = self._mm_settings.source
        # Whether pixels are normalized on device, so the Router sends this
        # engine raw pixels; likewise absent from an older proto package.
        if "mm_device_do_normalize" in info.DESCRIPTOR.fields_by_name:
            info.mm_device_do_normalize = mm_device_do_normalize(self.engine.vllm_config)
        if "multimodal_encoder_dtype" in info.DESCRIPTOR.fields_by_name:
            info.multimodal_encoder_dtype = encoder_dtype
        if max_num_seqs and "max_num_seqs" in info.DESCRIPTOR.fields_by_name:
            info.max_num_seqs = max_num_seqs
        # The engine's own per-prompt media limits, for the Router's media
        # pipeline to hold requests to; likewise absent from an older package.
        if "mm_item_limits" in info.DESCRIPTOR.fields_by_name:
            info.mm_item_limits = mm_item_limits(self.engine.vllm_config)
        # The engine's encoder budget, for the Router's media pipeline to
        # refuse an item the engine could never place; likewise absent from
        # an older package.
        if self._mm_encoder_budget and "mm_encoder_budget" in info.DESCRIPTOR.fields_by_name:
            info.mm_encoder_budget = self._mm_encoder_budget
        # The engine's reasoning parser and where it starts a grammar on a
        # thinking prompt, for the Router to shape a forced tool call's
        # grammar by; likewise absent from an older package.
        if "constrained_decoding_start" in info.DESCRIPTOR.fields_by_name:
            info.engine_reasoning_parser = engine_reasoning_parser(self.engine.vllm_config)
            info.constrained_decoding_start = constrained_decoding_start(self.engine.vllm_config)
        return info

    async def GetLoads(
        self,
        request: vllm_engine_pb2.GetLoadsRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetLoadsResponse:
        """
        Handle load-metric requests.

        One SchedulerLoad per data-parallel rank this servicer serves (see
        ``_managed_ranks``), from the latest SchedulerStats snapshot cached
        on the engine's stat loggers for that rank: ``token_usage`` carries
        KV-cache utilization ([0,1)) and ``num_running_reqs`` /
        ``num_waiting_reqs`` report queue depth. ``num_waiting_uncached_tokens``,
        ``gen_throughput`` and ``cache_hit_rate`` come from this servicer's own
        bookkeeping of the requests it forwards, per rank
        (``smg_grpc_servicer.vllm.loads``), since vLLM's stats do not carry them.
        The entries are the figures the ranks' pushed load records carry, so
        a poll refreshes every rank, those whose event stream is quiet too.

        The first rank's entry is always present (zero-filled when no snapshot
        is available yet, e.g. with --disable-log-stats or before the first
        engine step) so callers treat the worker as responsive rather than
        dropping the poll; another rank without a readable snapshot gets no
        entry rather than another rank's figures (see ``_rank_counts``).

        Note: ``request.dp_rank`` and ``request.include`` are accepted but not
        yet applied — every served rank is reported with only core metrics, so
        there is nothing to filter. They are reserved for future
        sectioned-metrics support.

        Args:
            request: The GetLoadsRequest protobuf
            context: gRPC context

        Returns:
            GetLoadsResponse protobuf
        """
        loads = self._scheduler_loads()
        return vllm_engine_pb2.GetLoadsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=VLLM_VERSION,
            dp_rank_count=len(loads),
            loads=loads,
        )

    def _scheduler_counts(self, engine_idx: int = 0) -> tuple[int, int, float] | None:
        """Running, waiting and KV usage from the latest ``SchedulerStats``
        snapshot of the engine at ``engine_idx`` (the data-parallel rank),
        what ``GetLoads`` and the pushed load record both report; ``None``
        when there is none to read (no engine step yet, stats logging off,
        or a rank the stat loggers do not keep apart)."""
        stats = _latest_scheduler_stats(self.engine, engine_idx)
        if stats is None:
            return None
        return (
            int(getattr(stats, "num_running_reqs", 0) or 0),
            int(getattr(stats, "num_waiting_reqs", 0) or 0),
            float(getattr(stats, "kv_cache_usage", 0.0) or 0.0),
        )

    def _rank_counts(self, rank: int) -> tuple[int, int, float] | None:
        """``rank``'s running, waiting and KV usage for a load report
        (``GetLoads`` and the pushed record alike): its own snapshot's;
        zeros for the servicer's first rank before its first snapshot, so a
        poll always has an entry to read; ``None`` for another rank whose
        snapshot the servicer cannot read, which gets no entry rather than
        another rank's figures or zeros that read as idle."""
        counts = self._scheduler_counts(rank)
        if counts is None and rank == _first_rank(self.engine):
            return (0, 0, 0.0)
        return counts

    def _scheduler_loads(self) -> list[vllm_engine_pb2.SchedulerLoad]:
        """The ``GetLoads`` entries: one per rank this servicer serves (see
        ``_managed_ranks``) with counts to report (see ``_rank_counts``),
        each with that rank's own counts and this servicer's own estimate
        for it, the figures the rank's pushed record carries
        (``_load_record``)."""
        max_total_num_tokens = _kv_capacity_tokens(self.engine)
        max_running_requests = _max_running_requests(self.engine)
        loads = []
        for rank in _managed_ranks(self.engine):
            counts = self._rank_counts(rank)
            if counts is None:
                continue
            num_running, num_waiting, kv_usage = counts
            loads.append(
                vllm_engine_pb2.SchedulerLoad(
                    **scheduler_load_fields(
                        num_running,
                        num_waiting,
                        kv_usage,
                        self._loads.estimate(num_waiting, rank=rank),
                        dp_rank=rank,
                        max_total_num_tokens=max_total_num_tokens,
                        max_running_requests=max_running_requests,
                    )
                )
            )
        return loads

    def _load_record(self, dp_rank: int | None) -> common_pb2.EngineLoad | None:
        """The load record the KV-event stream carries for the batch's rank
        (the servicer's first rank when it names none, see ``_first_rank``):
        that rank's own scheduler counts and this servicer's own queued-work,
        throughput and hit-rate figures for that rank, as the Rust servicer
        stamps its batches. A rank whose snapshot the servicer cannot read
        gets no record rather than another rank's figures; the first rank
        reports zeros before its first snapshot, as ``GetLoads`` does (see
        ``_rank_counts``). A read that raises is the recorder's to report."""
        rank = _first_rank(self.engine) if dp_rank is None else dp_rank
        counts = self._rank_counts(rank)
        if counts is None:
            return None
        num_running, num_waiting, kv_usage = counts
        return load_record(
            num_running,
            num_waiting,
            kv_usage,
            self._loads.estimate(num_waiting, rank=rank),
            dp_rank=rank,
            max_total_num_tokens=_kv_capacity_tokens(self.engine),
            max_running_requests=_max_running_requests(self.engine),
        )

    async def GetTokenizer(
        self,
        request: common_pb2.GetTokenizerRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.GetTokenizerChunk]:
        """Stream tokenizer artifacts as a ZIP bundle.

        Resolves the tokenizer directory from model_config, zips all relevant
        tokenizer files, and streams them as GetTokenizerChunk messages.
        The final chunk carries the SHA-256 fingerprint of the full archive.
        """
        logger.info("Receive GetTokenizer request")

        tokenizer_path = self.engine.model_config.tokenizer
        if not tokenizer_path:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Tokenizer path is not configured on this server.",
            )
        # TODO: model_config.tokenizer may be an HF model ID (e.g. "meta-llama/...")
        # rather than a local path. vLLM does not resolve it on the config object.
        # For now, GetTokenizer only works when vLLM is started with a local path.
        tokenizer_dir = Path(tokenizer_path)

        # Reading and compressing the tokenizer directory is file I/O and CPU
        # measured in hundreds of milliseconds: the shared bundle builds it in
        # a worker thread, once, and the engine keeps answering everything else.
        try:
            zip_data, sha256 = await tokenizer_bundle(tokenizer_dir)
        except Exception as e:
            logger.exception("Failed to build tokenizer ZIP")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        logger.info(
            "Streaming tokenizer bundle: %d bytes, sha256=%s",
            len(zip_data),
            sha256,
        )

        for chunk in tokenizer_chunks(zip_data, sha256):
            yield chunk

    # ========== Helper methods ==========

    def _build_preprocessed_mm_inputs(
        self,
        tokenized: vllm_engine_pb2.TokenizedInput,
        mm_protos: Sequence[vllm_engine_pb2.MultimodalInputs],
    ) -> VllmMultiModalInput:
        """Build vLLM MultiModalInput from preprocessed proto data.

        Bypasses HF processor entirely — the tensors were already computed by
        the Rust router. Pixel values are optional: the PD decode side carries
        only the grid tensors (positions + block hashing need no pixels).
        Field layouts (batched / flat / shared) are also determined by the
        router via ``batched_keys`` and ``flat_keys`` proto fields.

        Each batch is one modality; a request mixing image and video brings
        one batch each, and their tensors, hashes and placeholders are keyed
        by modality the way vLLM's own processors lay them out.
        """
        prompt_token_ids = list(tokenized.input_ids)
        prompt_ids_tensor: torch.Tensor | None = None
        # Cast floating-point tensors to model dtype (e.g. bfloat16).
        # This mirrors _postprocess_output in multimodal/processing/context.py
        # which is skipped when bypassing the HF processor.
        model_dtype = self.engine.model_config.dtype

        hf_dict: dict[str, torch.Tensor] = {}
        fields_config: dict[str, MultiModalFieldConfig] = {}
        mm_hashes: dict[str, list[str]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}

        for mm_proto in mm_protos:
            # Image vs video: vLLM routes each modality to a different encoder and
            # expects the pixel tensor under a modality-specific key. The router
            # sends the generic ``pixel_values`` field; rename it to
            # ``pixel_values_videos`` for video (grid/size tensors already carry
            # video-specific keys).
            mm_modality = modality_name(mm_proto)
            is_video = mm_modality == "video"

            def mm_key(key: str, is_video: bool = is_video) -> str:
                return modality_key(key, is_video)

            num_items = len(mm_proto.mm_placeholders)

            # Deserialize all tensors from proto. The PD decode side carries no
            # pixel_values (KV arrives via the P/D transfer), only grid tensors.
            # The primary tensor is registered under the model's forward kwarg
            # (``encoder_input_key``; DeepSeek-V4.1 takes ``patches``), the same
            # name the router uses in ``batched_keys`` / ``flat_keys``.
            batch: dict[str, torch.Tensor] = {}
            if mm_proto.HasField("pixel_values"):
                primary_key = mm_key(primary_encoder_key(mm_proto))
                batch[primary_key] = tensor_from_proto(mm_proto.pixel_values)
            for key, td in mm_proto.model_specific_tensors.items():
                batch[mm_key(key)] = tensor_from_proto(td)
            for key, tensor in batch.items():
                if tensor.is_floating_point():
                    batch[key] = tensor.to(dtype=model_dtype)
            shared_keys = batch.keys() & hf_dict.keys()
            if shared_keys:
                raise ValueError(
                    "multimodal batches carry the same tensor keys "
                    f"{sorted(shared_keys)}; one batch per modality is expected"
                )
            hf_dict.update(batch)

            cpu_keys = {mm_key(k) for k in mm_proto.keep_on_cpu_keys}

            # Field configs are fully determined by the Rust router.
            batched = {mm_key(k) for k in mm_proto.batched_keys}
            flat = {mm_key(k): mm_key(v) for k, v in mm_proto.flat_keys.items()}
            flat_sizes_cache: dict[str, torch.Tensor] = {}
            for key in batch:
                on_cpu = key in cpu_keys
                if key in batched:
                    fields_config[key] = MultiModalFieldConfig.batched(
                        mm_modality, keep_on_cpu=on_cpu
                    )
                elif key in flat:
                    sizes_key = flat[key]
                    if sizes_key not in flat_sizes_cache:
                        flat_sizes_cache[sizes_key] = batch[sizes_key].flatten().to(torch.int64)
                    fields_config[key] = MultiModalFieldConfig.flat_from_sizes(
                        mm_modality, flat_sizes_cache[sizes_key], keep_on_cpu=on_cpu
                    )
                else:
                    fields_config[key] = MultiModalFieldConfig.shared(mm_modality, num_items)

            if mm_proto.mm_hashes:
                mm_hashes.setdefault(mm_modality, []).extend(mm_proto.mm_hashes)

            # When structural tokens (e.g. <|image_start|>, separators) are
            # present in the placeholder range, we must set is_embed so vLLM
            # only scatters encoder embeddings into patch-token positions
            # (im_token_id).
            if mm_proto.mm_placeholders:
                im_token_id = mm_proto.im_token_id if mm_proto.HasField("im_token_id") else None
                if im_token_id is not None and prompt_ids_tensor is None:
                    # Pre-convert to tensor for vectorized mask building
                    prompt_ids_tensor = torch.tensor(prompt_token_ids, dtype=torch.int64)
                placeholders = mm_placeholders.setdefault(mm_modality, [])
                for p in mm_proto.mm_placeholders:
                    is_embed = None
                    if im_token_id is not None:
                        mask = prompt_ids_tensor[p.offset : p.offset + p.length] == im_token_id
                        # Only set is_embed when there are non-embed positions,
                        # otherwise None means "all positions are embeds" which
                        # is both correct and avoids unnecessary overhead.
                        if not mask.all():
                            is_embed = mask
                    placeholders.append(
                        PlaceholderRange(offset=p.offset, length=p.length, is_embed=is_embed)
                    )

        batch_feature = BatchFeature(hf_dict, tensor_type="pt")
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(batch_feature, fields_config)

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            prompt=tokenized.original_text or None,
        )

    @staticmethod
    def _sampling_params_from_proto(
        params: vllm_engine_pb2.SamplingParams,
        stream: bool = True,
        kv_transfer_params: dict | None = None,
    ) -> SamplingParams:
        """
        Convert protobuf SamplingParams to vLLM SamplingParams.

        Args:
            params: Protobuf SamplingParams message
            stream: Whether streaming is enabled
            kv_transfer_params: Connector KV-transfer params dict (PD disaggregation)

        Returns:
            vLLM SamplingParams with detokenize=False and structured_outputs
        """
        # Build stop sequences
        stop = list(params.stop) if params.stop else None
        stop_token_ids = list(params.stop_token_ids) if params.stop_token_ids else None

        # Handle structured outputs constraints
        structured_outputs = None
        constraint_field = params.WhichOneof("constraint")
        if constraint_field:
            if constraint_field == "json_schema":
                structured_outputs = StructuredOutputsParams(json=params.json_schema)
            elif constraint_field == "regex":
                structured_outputs = StructuredOutputsParams(regex=params.regex)
            elif constraint_field == "grammar":
                structured_outputs = StructuredOutputsParams(grammar=params.grammar)
            elif constraint_field == "structural_tag":
                structured_outputs = StructuredOutputsParams(structural_tag=params.structural_tag)
            elif constraint_field == "json_object":
                structured_outputs = StructuredOutputsParams(json_object=params.json_object)
            elif constraint_field == "choice":
                structured_outputs = StructuredOutputsParams(choice=list(params.choice.choices))

        # Opaque connector params, passed to the engine verbatim (NIXL/Mooncake)
        extra_args = {"kv_transfer_params": kv_transfer_params} if kv_transfer_params else None

        # Create SamplingParams
        # output_kind=DELTA: Return only new tokens in each chunk (for streaming)
        return SamplingParams(
            temperature=params.temperature if params.HasField("temperature") else 1.0,
            top_p=params.top_p if params.top_p != 0.0 else 1.0,
            top_k=params.top_k,
            min_p=params.min_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            repetition_penalty=params.repetition_penalty
            if params.repetition_penalty != 0.0
            else 1.0,
            max_tokens=params.max_tokens if params.HasField("max_tokens") else None,
            min_tokens=params.min_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
            ignore_eos=params.ignore_eos,
            n=params.n if params.n > 0 else 1,
            logprobs=params.logprobs if params.HasField("logprobs") else None,
            prompt_logprobs=params.prompt_logprobs if params.HasField("prompt_logprobs") else None,
            seed=params.seed if params.HasField("seed") else None,
            include_stop_str_in_output=params.include_stop_str_in_output,
            logit_bias=dict(params.logit_bias) if params.logit_bias else None,
            structured_outputs=structured_outputs,
            extra_args=extra_args,
            # detokenize must be True if stop strings are used
            detokenize=bool(stop),
            output_kind=RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY,
        )

    @staticmethod
    def _build_top_logprobs(
        logprob_entry: dict,
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.TopLogProbs:
        """Build TopLogProbs proto from a logprob entry dict."""
        top = vllm_engine_pb2.TopLogProbs()
        if num_top_logprobs and num_top_logprobs > 0 and logprob_entry:
            for tid, lp in itertools.islice(logprob_entry.items(), num_top_logprobs):
                top.token_ids.append(tid)
                top.values.append(lp.logprob)
        return top

    @staticmethod
    def _build_output_logprobs(
        logprobs: SampleLogprobs | None,
        token_ids: list[int],
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.OutputLogProbs | None:
        """
        Convert vLLM SampleLogprobs to proto OutputLogProbs.

        Args:
            logprobs: vLLM logprobs (list of dict[int, Logprob])
            token_ids: Token IDs for each position
            num_top_logprobs: Number of top logprobs to include

        Returns:
            OutputLogProbs proto or None
        """
        if not logprobs:
            return None

        proto = vllm_engine_pb2.OutputLogProbs()

        for token_id, logprob_entry in zip(token_ids, logprobs):
            if logprob := logprob_entry.get(token_id):
                proto.token_logprobs.append(logprob.logprob)
                proto.token_ids.append(token_id)

                if num_top_logprobs:
                    proto.top_logprobs.append(
                        VllmEngineServicer._build_top_logprobs(logprob_entry, num_top_logprobs)
                    )

        return proto if proto.token_ids else None

    @staticmethod
    def _build_input_logprobs(
        prompt_logprobs: PromptLogprobs | None,
        prompt_token_ids: list[int],
        num_top_logprobs: int | None,
    ) -> vllm_engine_pb2.InputLogProbs | None:
        """
        Convert vLLM PromptLogprobs to proto InputLogProbs.

        Args:
            prompt_logprobs: vLLM prompt logprobs (list of dict[int, Logprob] | None)
            prompt_token_ids: Prompt token IDs
            num_top_logprobs: Number of top logprobs to include

        Returns:
            InputLogProbs proto or None
        """
        if not prompt_logprobs:
            return None

        proto = vllm_engine_pb2.InputLogProbs()

        for token_id, logprob_entry in zip(prompt_token_ids, prompt_logprobs):
            token_logprob = vllm_engine_pb2.InputTokenLogProb()

            # First token has no logprob (None)
            if logprob_entry is not None and token_id in logprob_entry:
                token_logprob.value = logprob_entry[token_id].logprob

            proto.token_logprobs.append(token_logprob)
            proto.token_ids.append(token_id)
            if num_top_logprobs:
                proto.top_logprobs.append(
                    VllmEngineServicer._build_top_logprobs(logprob_entry, num_top_logprobs)
                )

        return proto if proto.token_ids else None

    @staticmethod
    def _tokenization_kwargs_from_proto(
        params: vllm_engine_pb2.SamplingParams,
    ) -> dict[str, int] | None:
        if params.HasField("truncate_prompt_tokens"):
            return {"truncate_prompt_tokens": params.truncate_prompt_tokens}
        return None

    @staticmethod
    def _chunk_response(
        output: RequestOutput,
        completion: "CompletionOutput | None" = None,
        num_logprobs: int | None = None,
        num_prompt_logprobs: int | None = None,
        is_first_chunk: bool = False,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a streaming chunk response from vLLM output.
        When output_kind=DELTA, vLLM returns only new tokens automatically.

        Note: This sends DELTA logprobs (only for new tokens in this chunk).
        The Rust side is responsible for accumulating if needed.

        Args:
            output: vLLM RequestOutput (with delta tokens when output_kind=DELTA)
            completion: Specific CompletionOutput to use (for n>1 support).
                       If None, uses output.outputs[0] for backwards compatibility.
            num_logprobs: Number of top logprobs for output tokens
            num_prompt_logprobs: Number of top logprobs for prompt tokens
            is_first_chunk: Whether this is the first chunk for this index
                           (include input_logprobs only on first chunk)

        Returns:
            GenerateResponse with chunk field set
        """
        # Use provided completion or fall back to first output
        if completion is None:
            completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty chunk
            return vllm_engine_pb2.GenerateResponse(
                chunk=vllm_engine_pb2.GenerateStreamChunk(
                    token_ids=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                    index=0,
                ),
            )

        # Build output logprobs for this chunk's tokens (delta, not cumulative)
        output_logprobs = VllmEngineServicer._build_output_logprobs(
            completion.logprobs, completion.token_ids, num_logprobs
        )

        # Build input logprobs only on first chunk for this index
        input_logprobs = None
        if is_first_chunk:
            input_logprobs = VllmEngineServicer._build_input_logprobs(
                output.prompt_logprobs,
                output.prompt_token_ids,
                num_prompt_logprobs,
            )

        # When output_kind=DELTA, completion.token_ids contains only new tokens
        # vLLM handles the delta logic internally
        # completion_tokens = delta count (client will accumulate)
        return vllm_engine_pb2.GenerateResponse(
            chunk=vllm_engine_pb2.GenerateStreamChunk(
                token_ids=completion.token_ids,
                prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                completion_tokens=len(completion.token_ids),  # Delta count
                cached_tokens=output.num_cached_tokens,
                output_logprobs=output_logprobs,
                input_logprobs=input_logprobs,
                index=completion.index,
            ),
        )

    @staticmethod
    def _complete_response(
        output: RequestOutput,
        completion: "CompletionOutput | None" = None,
        num_logprobs: int | None = None,
        num_prompt_logprobs: int | None = None,
        media_identity: "vllm_engine_pb2.MediaIdentity | None" = None,
    ) -> vllm_engine_pb2.GenerateResponse:
        """
        Build a final completion response from vLLM output.

        For non-streaming (FINAL_ONLY): completion has all tokens and logprobs.
        For streaming (DELTA): completion has last delta; Rust accumulates.

        Args:
            output: vLLM RequestOutput (finished=True)
            completion: Specific CompletionOutput to use (for n>1 support).
                       If None, uses output.outputs[0] for backwards compatibility.
            num_logprobs: Number of top logprobs for output tokens
            num_prompt_logprobs: Number of top logprobs for prompt tokens
            media_identity: A PD prefill side's processed media, for the decode side

        Returns:
            GenerateResponse with complete field set
        """
        # Use provided completion or fall back to first output
        if completion is None:
            completion = output.outputs[0] if output.outputs else None

        if completion is None:
            # Empty completion
            return vllm_engine_pb2.GenerateResponse(
                complete=vllm_engine_pb2.GenerateComplete(
                    output_ids=[],
                    finish_reason="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=0,
                    index=0,
                ),
            )

        # Build output logprobs from completion's data
        # For non-streaming: this has all logprobs
        # For streaming: this has only last delta (Rust accumulates from chunks)
        output_logprobs = VllmEngineServicer._build_output_logprobs(
            completion.logprobs, completion.token_ids, num_logprobs
        )

        # Build input logprobs
        input_logprobs = VllmEngineServicer._build_input_logprobs(
            output.prompt_logprobs,
            output.prompt_token_ids,
            num_prompt_logprobs,
        )

        # Connector KV-transfer params returned by the engine (PD prefill side)
        kv_transfer_params, kv_transfer_params_json = params_to_response_fields(
            output.kv_transfer_params
        )

        # Build matched_stop kwargs from stop_reason (int token ID or str stop sequence)
        stop_kwargs = {}
        # A proto package predating the field cannot carry the identity.
        if (
            media_identity is not None
            and "media_identity" in vllm_engine_pb2.GenerateComplete.DESCRIPTOR.fields_by_name
        ):
            stop_kwargs["media_identity"] = media_identity
        if completion.stop_reason is not None:
            if isinstance(completion.stop_reason, int):
                stop_kwargs["matched_token_id"] = completion.stop_reason
            else:
                stop_kwargs["matched_stop_str"] = str(completion.stop_reason)

        # Per-request speculative counts; needs vLLM's --per-request-spec-decode-metrics.
        spec = getattr(completion, "spec_decode_metrics", None)
        spec_accepted = sum(j * n for j, n in enumerate(spec.histogram)) if spec else 0

        # Build complete response
        # When streaming (DELTA mode): completion.token_ids will be empty/last delta
        # When non-streaming (FINAL_ONLY mode): completion.token_ids has all tokens
        # Client will accumulate token counts for streaming
        return vllm_engine_pb2.GenerateResponse(
            complete=vllm_engine_pb2.GenerateComplete(
                output_ids=completion.token_ids,
                finish_reason=completion.finish_reason or "stop",
                spec_accepted_tokens=spec_accepted,
                spec_draft_tokens=spec.num_draft_tokens if spec else 0,
                prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                completion_tokens=len(completion.token_ids),
                cached_tokens=output.num_cached_tokens,
                output_logprobs=output_logprobs,
                input_logprobs=input_logprobs,
                index=completion.index,
                kv_transfer_params=kv_transfer_params,
                kv_transfer_params_json=kv_transfer_params_json,
                **stop_kwargs,
            ),
        )

    async def SubscribeKvEvents(
        self,
        request: common_pb2.SubscribeKvEventsRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.KvEventBatch]:
        """Relay vLLM's ZMQ KV cache events, every DP rank's publisher, as one
        gRPC stream (see ``smg_grpc_servicer.kv_relay``)."""
        if self._kv_events_config is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "KV cache events not enabled. Start vLLM with "
                "--kv-events-config "
                '\'{"enable_kv_cache_events": true, "publisher": "zmq"}\'',
            )

        config = self._kv_events_config
        async for proto_batch in relay(
            rank_sources_for(config, self.engine),
            Engine.VLLM,
            request.start_sequence_number,
            context,
            topic=str(getattr(config, "topic", "") or ""),
            hwm=getattr(config, "hwm", None),
            load_source=self._load_record,
        ):
            yield proto_batch
