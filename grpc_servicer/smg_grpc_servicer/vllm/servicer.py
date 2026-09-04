# mypy: ignore-errors
"""
vLLM gRPC Servicer

Implements the VllmEngine gRPC service on top of vLLM's EngineClient.
"""

import asyncio
import hashlib
import itertools
import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import grpc
import msgspec
import torch
import zmq
import zmq.asyncio
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_proto.generated import common_pb2
from transformers import BatchFeature
from vllm import PoolingParams, SamplingParams, TokensPrompt
from vllm.distributed.kv_events import KVEventBatch
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

from smg_grpc_servicer import mm_shm
from smg_grpc_servicer.tokenizer_bundle import CHUNK_SIZE, build_tokenizer_zip
from smg_grpc_servicer.vllm.kv_events import (
    endpoint_for_rank,
    resolve_kv_events_config,
    stream_kv_events,
)
from smg_grpc_servicer.vllm.kv_transfer import params_from_request, params_to_response_fields
from smg_grpc_servicer.vllm.media_refs import parse_media_refs, validate_schemes
from smg_grpc_servicer.vllm.mm_processor import (
    DEFAULT_MAX_INFLIGHT,
    ENV_MAX_INFLIGHT,
    ENV_PROCESSOR,
    MmProcessorUnavailable,
    build_mm_processor,
    env_int,
)
from smg_grpc_servicer.vllm.mm_salt import has_preprocessed_mm_payload, mm_identity_cache_salt

logger = init_logger(__name__)
SAMPLING_DEFAULT_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
)


def _filtered_sampling_defaults(params: dict | None) -> dict:
    if not params:
        return {}
    return {
        key: params[key]
        for key in SAMPLING_DEFAULT_KEYS
        if key in params and params[key] is not None
    }


# Proto dtype string → torch dtype
_PROTO_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "int64": torch.int64,
    "uint32": torch.uint32,
}


def _tensor_from_proto(td: vllm_engine_pb2.TensorData) -> torch.Tensor:
    """Deserialize a TensorData proto message into a torch.Tensor."""
    torch_dtype = _PROTO_DTYPE_MAP.get(td.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported proto tensor dtype: {td.dtype!r}")
    payload = mm_shm.tensor_payload_bytes(td)
    return torch.frombuffer(bytearray(payload), dtype=torch_dtype).reshape(*td.shape)


try:
    from vllm.version import __version__ as VLLM_VERSION
except Exception:  # pragma: no cover - version lookup is best-effort
    VLLM_VERSION = ""


def _latest_scheduler_stats(engine, engine_idx: int = 0):
    """Best-effort read of the most recent ``SchedulerStats`` snapshot.

    vLLM has no synchronous "current stats" accessor on ``AsyncLLM``/``EngineClient``;
    ``SchedulerStats`` arrive asynchronously and are cached on the stat loggers. This
    reaches into ``engine.logger_manager.stat_loggers`` and returns the freshest
    snapshot, handling the logger-shape variants:

    - ``LoggingStatLogger``                  -> ``.last_scheduler_stats``
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
        if stats is not None:
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

    def __init__(self, async_llm: EngineClient, start_time: float):
        """
        Initialize the servicer.

        Args:
            async_llm: The EngineClient instance (e.g. AsyncLLM)
            start_time: The server start time, in seconds since epoch
        """
        self.engine = async_llm
        self.start_time = start_time
        # Resolve KV-event publishing config from the engine. Non-None only when
        # vLLM was started with --kv-events-config enabling the ZMQ publisher.
        self._kv_events_config = resolve_kv_events_config(async_llm)
        # Worker-side media processing (media_refs); None keeps refs rejected.
        self._mm_processor = build_mm_processor(async_llm)
        self._mm_inflight = asyncio.Semaphore(
            env_int(os.environ, ENV_MAX_INFLIGHT, DEFAULT_MAX_INFLIGHT)
        )
        logger.info(
            "VllmEngineServicer initialized (mm_processor=%s)",
            self._mm_processor.name if self._mm_processor is not None else "off",
        )

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
        # A pixel-less mm payload with grid tensors is the PD decode leg's
        # form: enough to rebuild mm features (positions + block hashing).
        has_preprocessed_mm = request.HasField("mm_inputs") and has_preprocessed_mm_payload(
            request.mm_inputs
        )
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
        engine_started = False
        try:
            arrival_time = time.time()
            kv_transfer_params = params_from_request(request)

            if request.HasField("media_refs"):
                # Media references from the router: the worker fetches and runs
                # vLLM's own processor over the unexpanded placeholder anchors.
                if input_type != "tokenized" or request.HasField("mm_inputs"):
                    raise ValueError(
                        "media_refs requires tokenized input and is mutually exclusive "
                        "with mm_inputs"
                    )
                if self._mm_processor is None:
                    raise ValueError(
                        f"media_refs sent but {ENV_PROCESSOR} is off on this worker; check the "
                        "router's SMG_MM_PROCESSING and this worker's mm_processor label"
                    )
                items = parse_media_refs(request.media_refs)
                validate_schemes(items, self._mm_processor.accepted_schemes)
                async with self._mm_inflight:
                    try:
                        prompt = await self._mm_processor.process(
                            list(request.tokenized.input_ids),
                            request.tokenized.original_text or None,
                            items,
                            arrival_time,
                        )
                    except RuntimeError as e:
                        # vLLM's placeholder validation: anchor/count mismatch is
                        # a terminal client error, not something to retry.
                        raise ValueError(f"multimodal placeholder validation failed: {e}") from e
            elif has_preprocessed_mm and input_type == "tokenized":
                # A pixel-less payload (PD decode leg) is only decodable with
                # remote KV: a local recompute would schedule the vision
                # encoder with no pixels and crash the engine.
                if not request.mm_inputs.HasField("pixel_values") and kv_transfer_params is None:
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
                # ready for the model. Bypass the renderer entirely.
                prompt = self._build_preprocessed_mm_inputs(request.tokenized, request.mm_inputs)
                prompt["arrival_time"] = arrival_time
            elif input_type == "tokenized":
                prompt: TokensPrompt = {"prompt_token_ids": list(request.tokenized.input_ids)}
                if request.tokenized.original_text:
                    prompt["prompt"] = request.tokenized.original_text
                # Tensor-less mm payload (grid-less PD decode leg): fold the
                # kept mm hashes into cache_salt so different images cannot
                # alias. Grid-carrying legs took the preprocessed path above.
                if request.HasField("mm_inputs"):
                    cache_salt = mm_identity_cache_salt(request.mm_inputs.mm_hashes)
                    if cache_salt is not None:
                        prompt["cache_salt"] = cache_salt
                    model_config = getattr(self.engine, "model_config", None)
                    if model_config is not None and getattr(model_config, "uses_mrope", False):
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

            async for output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                tokenization_kwargs=tokenization_kwargs,
                data_parallel_rank=(
                    request.data_parallel_rank if request.HasField("data_parallel_rank") else None
                ),
            ):
                engine_started = True
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
                            )

                # For non-streaming, send complete response when finished
                if output.finished and not request.stream:
                    for completion in output.outputs:
                        yield self._complete_response(
                            output,
                            completion=completion,
                            num_logprobs=num_logprobs,
                            num_prompt_logprobs=num_prompt_logprobs,
                        )

        except ValueError as e:
            # Invalid request error (equiv to 400).
            await self._notify_kv_transfer_rejected(request_id, kv_transfer_params, engine_started)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except MmProcessorUnavailable as e:
            # Retryable: the router re-selects a worker on UNAVAILABLE.
            logger.warning("Media processing unavailable for request %s: %s", request_id, e)
            await self._notify_kv_transfer_rejected(request_id, kv_transfer_params, engine_started)
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except Exception as e:
            logger.exception("Error in Generate for request %s", request_id)
            await self._notify_kv_transfer_rejected(request_id, kv_transfer_params, engine_started)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def _notify_kv_transfer_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict | None,
        engine_started: bool,
    ) -> None:
        """Free remote prefill blocks early when a decode request dies pre-admission.

        Without this, the prefill engine keeps the blocks pinned until the NIXL
        lease expires (30s default).
        """
        if engine_started or not kv_transfer_params:
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
        except ValueError as e:
            logger.warning("Embed invalid request %s: %s", request_id, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Embed failed for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

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

        logger.info("HealthCheck request: healthy=%s, message=%s", is_healthy, message)

        return vllm_engine_pb2.HealthCheckResponse(healthy=is_healthy, message=message)

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

        await self.engine.abort(request_ids)
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
        model_config = self.engine.model_config
        hf_config = model_config.hf_config

        # eos_token_id can be int or list[int]
        eos = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos, int):
            eos_token_ids = [eos]
        elif isinstance(eos, list):
            eos_token_ids = eos
        else:
            eos_token_ids = []

        sampling_defaults = _filtered_sampling_defaults(
            model_config.get_diff_sampling_param() or {}
        )

        return vllm_engine_pb2.GetModelInfoResponse(
            model_path=model_config.model,
            is_generation=model_config.runner_type == "generate",
            max_context_length=model_config.max_model_len,
            vocab_size=model_config.get_vocab_size(),
            supports_vision=model_config.is_multimodal_model,
            served_model_name=model_config.served_model_name or model_config.model,
            tokenizer_path=model_config.tokenizer or "",
            model_type=getattr(hf_config, "model_type", "") or "",
            architectures=model_config.architectures or [],
            eos_token_ids=eos_token_ids,
            pad_token_id=getattr(hf_config, "pad_token_id", None) or 0,
            bos_token_id=getattr(hf_config, "bos_token_id", None) or 0,
            max_req_input_len=model_config.max_model_len,
            default_sampling_params_json=(
                json.dumps(sampling_defaults, separators=(",", ":")) if sampling_defaults else ""
            ),
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
        kv_connector = ""
        kv_role = ""
        kv_engine_id = ""
        parallel = self.engine.vllm_config.parallel_config
        kv_transfer_config = self.engine.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            kv_connector = kv_transfer_config.kv_connector or ""
            kv_role = kv_transfer_config.kv_role or ""
            # Base engine_id; with DP the engine cores serve `{id}_dp{rank}` and
            # the router derives the suffix from the rank it pins per request
            kv_engine_id = getattr(kv_transfer_config, "engine_id", "") or ""

        mm_processor = ""
        mm_media_ref_schemes = ""
        if (
            self._mm_processor is not None
            and self.engine.model_config.is_multimodal_model
            and await self._mm_processor.probe()
        ):
            mm_processor = self._mm_processor.name
            mm_media_ref_schemes = self._mm_processor.schemes

        return vllm_engine_pb2.GetServerInfoResponse(
            kv_connector=kv_connector,
            kv_role=kv_role,
            kv_engine_id=kv_engine_id,
            data_parallel_size=parallel.data_parallel_size,
            shm_namespace_id=mm_shm.shm_namespace_id(),
            mm_processor=mm_processor,
            mm_media_ref_schemes=mm_media_ref_schemes,
        )

    async def GetLoads(
        self,
        request: vllm_engine_pb2.GetLoadsRequest,
        context: grpc.aio.ServicerContext,
    ) -> vllm_engine_pb2.GetLoadsResponse:
        """
        Handle load-metric requests.

        Reads the latest SchedulerStats snapshot cached on the engine's stat
        loggers and maps it onto a single-DP-rank SchedulerLoad: ``token_usage``
        carries KV-cache utilization ([0,1)) and ``num_running_reqs`` /
        ``num_waiting_reqs`` report queue depth.

        Always returns exactly one SchedulerLoad entry (zero-filled when no
        snapshot is available yet, e.g. with --disable-log-stats or before the
        first engine step) so callers treat the worker as responsive rather than
        dropping the poll.

        Note: ``request.dp_rank`` and ``request.include`` are accepted but not yet
        applied — vLLM reports a single DP rank (``dp_rank_count=1``) with only
        core metrics, so there is nothing to filter. They are reserved for future
        multi-DP / sectioned-metrics support.

        Args:
            request: The GetLoadsRequest protobuf
            context: gRPC context

        Returns:
            GetLoadsResponse protobuf
        """
        stats = _latest_scheduler_stats(self.engine)
        if stats is not None:
            num_running = int(getattr(stats, "num_running_reqs", 0) or 0)
            num_waiting = int(getattr(stats, "num_waiting_reqs", 0) or 0)
            kv_usage = float(getattr(stats, "kv_cache_usage", 0.0) or 0.0)
        else:
            num_running = 0
            num_waiting = 0
            kv_usage = 0.0

        load = vllm_engine_pb2.SchedulerLoad(
            dp_rank=0,
            num_running_reqs=num_running,
            num_waiting_reqs=num_waiting,
            num_total_reqs=num_running + num_waiting,
            token_usage=max(0.0, kv_usage),
        )

        return vllm_engine_pb2.GetLoadsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=VLLM_VERSION,
            dp_rank_count=1,
            loads=[load],
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

        # Build ZIP archive in memory
        try:
            zip_buffer = build_tokenizer_zip(tokenizer_dir)
        except Exception as e:
            logger.exception("Failed to build tokenizer ZIP")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        zip_data = zip_buffer.getbuffer()
        sha256 = hashlib.sha256(zip_data).hexdigest()

        logger.info(
            "Streaming tokenizer bundle: %d bytes, sha256=%s",
            len(zip_data),
            sha256,
        )

        # Stream chunks; SHA-256 only on the final chunk
        offset = 0
        total = len(zip_data)
        while offset < total:
            end = min(offset + CHUNK_SIZE, total)
            is_last = end == total
            yield common_pb2.GetTokenizerChunk(
                data=bytes(zip_data[offset:end]),
                sha256=sha256 if is_last else "",
            )
            offset = end

    # ========== Helper methods ==========

    def _build_preprocessed_mm_inputs(
        self,
        tokenized: vllm_engine_pb2.TokenizedInput,
        mm_proto: vllm_engine_pb2.MultimodalInputs,
    ) -> VllmMultiModalInput:
        """Build vLLM MultiModalInput from preprocessed proto data.

        Bypasses HF processor entirely — the tensors were already computed by
        the Rust router. Pixel values are optional: the PD decode leg carries
        only the grid tensors (positions + block hashing need no pixels).
        Field layouts (batched / flat / shared) are also determined by the
        router via ``batched_keys`` and ``flat_keys`` proto fields.
        """
        prompt_token_ids = list(tokenized.input_ids)
        num_items = len(mm_proto.mm_placeholders)

        # Image vs video: vLLM routes each modality to a different encoder and
        # expects the pixel tensor under a modality-specific key. The router sends
        # the generic ``pixel_values`` field; rename it to ``pixel_values_videos``
        # for the video path (grid/size tensors already carry video-specific keys).
        is_video = mm_proto.modality == common_pb2.VIDEO
        mm_modality = "video" if is_video else "image"

        def mm_key(key: str) -> str:
            if is_video and key == "pixel_values":
                return "pixel_values_videos"
            return key

        # Deserialize all tensors from proto. The PD decode leg carries no
        # pixel_values (KV arrives via the P/D transfer), only grid tensors.
        hf_dict: dict[str, torch.Tensor] = {}
        if mm_proto.HasField("pixel_values"):
            hf_dict[mm_key("pixel_values")] = _tensor_from_proto(mm_proto.pixel_values)
        for key, td in mm_proto.model_specific_tensors.items():
            hf_dict[mm_key(key)] = _tensor_from_proto(td)

        # Cast floating-point tensors to model dtype (e.g. bfloat16).
        # This mirrors _postprocess_output in multimodal/processing/context.py
        # which is skipped when bypassing the HF processor.
        model_dtype = self.engine.model_config.dtype
        for key in hf_dict:
            if hf_dict[key].is_floating_point():
                hf_dict[key] = hf_dict[key].to(dtype=model_dtype)

        cpu_keys = {mm_key(k) for k in mm_proto.keep_on_cpu_keys}

        # Field configs are fully determined by the Rust router.
        batched = {mm_key(k) for k in mm_proto.batched_keys}
        flat = {mm_key(k): mm_key(v) for k, v in mm_proto.flat_keys.items()}
        fields_config: dict[str, MultiModalFieldConfig] = {}
        flat_sizes_cache: dict[str, torch.Tensor] = {}
        for key in hf_dict:
            on_cpu = key in cpu_keys
            if key in batched:
                fields_config[key] = MultiModalFieldConfig.batched(mm_modality, keep_on_cpu=on_cpu)
            elif key in flat:
                sizes_key = flat[key]
                if sizes_key not in flat_sizes_cache:
                    flat_sizes_cache[sizes_key] = hf_dict[sizes_key].flatten().to(torch.int64)
                fields_config[key] = MultiModalFieldConfig.flat_from_sizes(
                    mm_modality, flat_sizes_cache[sizes_key], keep_on_cpu=on_cpu
                )
            else:
                fields_config[key] = MultiModalFieldConfig.shared(mm_modality, num_items)

        batch_feature = BatchFeature(hf_dict, tensor_type="pt")
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(batch_feature, fields_config)

        # Build mm_hashes: dict[str, list[str]]
        mm_hashes: dict[str, list[str]] = {}
        if mm_proto.mm_hashes:
            mm_hashes[mm_modality] = list(mm_proto.mm_hashes)

        # Build mm_placeholders: dict[str, list[PlaceholderRange]]
        # When structural tokens (e.g. <|image_start|>, separators) are present
        # in the placeholder range, we must set is_embed so vLLM only scatters
        # encoder embeddings into patch-token positions (im_token_id).
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if mm_proto.mm_placeholders:
            im_token_id = mm_proto.im_token_id if mm_proto.HasField("im_token_id") else None
            # Pre-convert to tensor for vectorized mask building
            prompt_ids_tensor = (
                torch.tensor(prompt_token_ids, dtype=torch.int64)
                if im_token_id is not None
                else None
            )
            placeholders = []
            for p in mm_proto.mm_placeholders:
                is_embed = None
                if prompt_ids_tensor is not None:
                    mask = prompt_ids_tensor[p.offset : p.offset + p.length] == im_token_id
                    # Only set is_embed when there are non-embed positions,
                    # otherwise None means "all positions are embeds" which is
                    # both correct and avoids unnecessary overhead.
                    if not mask.all():
                        is_embed = mask
                placeholders.append(
                    PlaceholderRange(offset=p.offset, length=p.length, is_embed=is_embed)
                )
            mm_placeholders[mm_modality] = placeholders

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

        # Connector KV-transfer params returned by the engine (PD prefill leg)
        kv_transfer_params, kv_transfer_params_json = params_to_response_fields(
            output.kv_transfer_params
        )

        # Build matched_stop kwargs from stop_reason (int token ID or str stop sequence)
        stop_kwargs = {}
        if completion.stop_reason is not None:
            if isinstance(completion.stop_reason, int):
                stop_kwargs["matched_token_id"] = completion.stop_reason
            else:
                stop_kwargs["matched_stop_str"] = str(completion.stop_reason)

        # Build complete response
        # When streaming (DELTA mode): completion.token_ids will be empty/last delta
        # When non-streaming (FINAL_ONLY mode): completion.token_ids has all tokens
        # Client will accumulate token counts for streaming
        return vllm_engine_pb2.GenerateResponse(
            complete=vllm_engine_pb2.GenerateComplete(
                output_ids=completion.token_ids,
                finish_reason=completion.finish_reason or "stop",
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
        """Bridge vLLM's in-process ZMQ KV cache events to a gRPC stream.

        The ZMQ publisher's sequence numbers are used directly as the gRPC
        batch sequence numbers.
        """
        if self._kv_events_config is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "KV cache events not enabled. Start vLLM with "
                "--kv-events-config "
                '\'{"enable_kv_cache_events": true, "publisher": "zmq"}\'',
            )

        config = self._kv_events_config

        # For DP attention each rank publishes on port + rank with independent
        # sequence counters; subscribing to several on one socket interleaves
        # them and breaks gap detection. Subscribe to rank 0 only for now.
        # TODO(phase3): per-rank virtual workers or merged renumbering.
        pub_endpoint = endpoint_for_rank(config.endpoint, 0)

        zmq_ctx = zmq.asyncio.Context.instance()
        sub_socket = zmq_ctx.socket(zmq.SUB)
        sub_socket.subscribe(config.topic.encode("utf-8"))
        sub_socket.connect(pub_endpoint)
        logger.info("SubscribeKvEvents: connected to ZMQ endpoint %s", pub_endpoint)

        decoder = msgspec.msgpack.Decoder(KVEventBatch)

        try:
            async for proto_batch in stream_kv_events(
                sub_socket,
                decoder.decode,
                lambda: context.send_initial_metadata(()),
                context.cancelled,
            ):
                yield proto_batch
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("SubscribeKvEvents failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        finally:
            sub_socket.close(linger=0)
            logger.info("SubscribeKvEvents: stream closed")
