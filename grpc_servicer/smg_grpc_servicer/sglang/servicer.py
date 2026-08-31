"""
SGLang gRPC Servicer

Implements the SglangScheduler gRPC service using GrpcRequestManager
for orchestration without tokenization.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import grpc
import msgspec
import numpy as np
import sglang
import torch
import zmq
import zmq.asyncio
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    KVEventsConfig,
    ZmqEventPublisher,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.managers.io_struct import (
    FlushCacheReqInput,
    ProfileReq,
    ProfileReqType,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.sampling.sampling_params import SamplingParams as SGLSamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.utils import get_exception_traceback
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from smg_grpc_proto.generated import common_pb2

from smg_grpc_servicer.sglang.health_servicer import SGLangHealthServicer
from smg_grpc_servicer.sglang.request_manager import GrpcRequestManager
from smg_grpc_servicer.sglang.utils import abort_code_from_output, to_token_id_array
from smg_grpc_servicer.tokenizer_bundle import CHUNK_SIZE, build_tokenizer_zip

logger = logging.getLogger(__name__)
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))
# Profile round-trips include trace serialization, which can take minutes.
PROFILE_COMM_TIMEOUT = 600.0


def _aggregate_communicator_results(results: list, ok_message: str) -> tuple[bool, str]:
    """Aggregate per-DP-rank communicator outputs into (success, message)."""
    if not results:
        return False, "No response from scheduler"
    failures = [r for r in results if not r.success]
    if failures:
        return False, " | ".join(r.message or "failed" for r in failures)
    return True, ok_message


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


def _convert_loads_to_protobuf(
    result: LoadSnapshot,
) -> sglang_scheduler_pb2.SchedulerLoad:
    """Convert a LoadSnapshot to a protobuf SchedulerLoad message."""
    scheduler_load = sglang_scheduler_pb2.SchedulerLoad(
        dp_rank=result.dp_rank,
        num_running_reqs=result.num_running_reqs,
        num_waiting_reqs=result.num_waiting_reqs,
        num_total_reqs=result.num_running_reqs + result.num_waiting_reqs,
        num_used_tokens=result.num_used_tokens,
        max_total_num_tokens=result.max_total_num_tokens,
        token_usage=result.token_usage,
        gen_throughput=result.gen_throughput,
        cache_hit_rate=result.cache_hit_rate,
        utilization=result.utilization,
        max_running_requests=result.max_running_requests,
        # Queued token-work: waiting-queue tokens not served from cache.
        num_waiting_uncached_tokens=result.num_waiting_uncached_tokens,
    )

    # Add optional sections using CopyFrom for proper protobuf assignment
    if result.memory:
        scheduler_load.memory.CopyFrom(
            sglang_scheduler_pb2.MemoryMetrics(
                weight_gb=result.memory.weight_gb,
                kv_cache_gb=result.memory.kv_cache_gb,
                graph_gb=result.memory.graph_gb,
                token_capacity=result.memory.token_capacity,
            )
        )

    if result.speculative:
        scheduler_load.speculative.CopyFrom(
            sglang_scheduler_pb2.SpeculativeMetrics(
                accept_length=result.speculative.accept_length,
                accept_rate=result.speculative.accept_rate,
            )
        )

    if result.lora:
        scheduler_load.lora.CopyFrom(
            sglang_scheduler_pb2.LoRAMetrics(
                slots_used=result.lora.slots_used,
                slots_total=result.lora.slots_total,
                utilization=result.lora.utilization,
            )
        )

    if result.disaggregation:
        scheduler_load.disaggregation.CopyFrom(
            sglang_scheduler_pb2.DisaggregationMetrics(
                mode=result.disaggregation.mode,
                prefill_prealloc_queue_reqs=result.disaggregation.prefill_prealloc_queue_reqs,
                prefill_inflight_queue_reqs=result.disaggregation.prefill_inflight_queue_reqs,
                decode_prealloc_queue_reqs=result.disaggregation.decode_prealloc_queue_reqs,
                decode_transfer_queue_reqs=result.disaggregation.decode_transfer_queue_reqs,
                decode_retracted_queue_reqs=result.disaggregation.decode_retracted_queue_reqs,
                kv_transfer_speed_gb_s=result.disaggregation.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=result.disaggregation.kv_transfer_latency_ms,
            )
        )

    if result.queues:
        scheduler_load.queues.CopyFrom(
            sglang_scheduler_pb2.QueueMetrics(
                waiting=result.queues.waiting,
                grammar=result.queues.grammar,
                paused=result.queues.paused,
                retracted=result.queues.retracted,
            )
        )

    return scheduler_load


def _compute_aggregate_protobuf(
    loads: list,
) -> sglang_scheduler_pb2.AggregateMetrics:
    """Compute aggregate metrics from list of SchedulerLoad protobuf messages."""
    if not loads:
        return sglang_scheduler_pb2.AggregateMetrics()

    n = len(loads)
    total_running = sum(load.num_running_reqs for load in loads)
    total_waiting = sum(load.num_waiting_reqs for load in loads)

    return sglang_scheduler_pb2.AggregateMetrics(
        total_running_reqs=total_running,
        total_waiting_reqs=total_waiting,
        total_reqs=total_running + total_waiting,
        avg_token_usage=round(sum(load.token_usage for load in loads) / n, 4),
        avg_throughput=round(sum(load.gen_throughput for load in loads) / n, 2),
        avg_utilization=round(sum(load.utilization for load in loads) / n, 4),
    )


class SGLangSchedulerServicer(sglang_scheduler_pb2_grpc.SglangSchedulerServicer):
    """
    Standalone gRPC service implementation using GrpcRequestManager.
    Fully separated from HTTP server with its own process and no shared globals.
    """

    def __init__(
        self,
        request_manager: GrpcRequestManager,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_info: dict,
        scheduler_info: dict,
        health_servicer: SGLangHealthServicer | None = None,
    ):
        """Initialize the standalone gRPC service."""
        self.request_manager = request_manager
        self.server_args = server_args
        self.model_config = model_config
        self.model_info = model_info
        self.scheduler_info = scheduler_info
        self.start_time = time.time()
        self.health_servicer = health_servicer
        # SamplingParams.normalize requires a tokenizer whenever string stops
        # are used; the scheduler matches them with its own tokenizer.
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )
        self.mm_receiver = None
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            from sglang.srt.disaggregation import encode_receiver as mm_receiver

            self.mm_receiver = mm_receiver.create_mm_receiver(self.server_args)

        # Parse KV events config for SubscribeKvEvents support
        self._kv_events_config: KVEventsConfig | None = None
        self._kv_event_id_counter = 0
        if server_args.kv_events_config:
            try:
                self._kv_events_config = KVEventsConfig.from_cli(server_args.kv_events_config)
                if self._kv_events_config.publisher != "zmq":
                    logger.info(
                        "KV events publisher is '%s', SubscribeKvEvents disabled",
                        self._kv_events_config.publisher,
                    )
                    self._kv_events_config = None
                else:
                    logger.info(
                        "KV events enabled: endpoint=%s",
                        self._kv_events_config.endpoint,
                    )
            except Exception as e:
                logger.warning("Failed to parse kv_events_config: %s", e)

        # Start the request manager's event loop using auto_create_handle_loop
        self.request_manager.auto_create_handle_loop()

        logger.info("gRPC scheduler servicer initialized")

    async def Generate(
        self,
        request: sglang_scheduler_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[sglang_scheduler_pb2.GenerateResponse]:
        """Handle generation requests with streaming responses."""
        logger.info(f"Receive generation request: {request.request_id}")

        try:
            # Convert gRPC request to internal format
            tokenized_req = self._convert_generate_request(request)
            self._handle_epd_disaggregation_encode_request(request, tokenized_req)

            # Submit to request manager (automatically handles n>1)
            response_generator = self.request_manager.generate_request(
                obj=tokenized_req,
                request_id=request.request_id,
                grpc_context=context,
            )

            async for output in response_generator:
                # Handle batch responses (for n>1 non-streaming)
                if isinstance(output, list):
                    for batch_output in output:
                        if "error" in batch_output:
                            await context.abort(
                                abort_code_from_output(batch_output),
                                batch_output["error"],
                            )
                        else:
                            # All non-error batch outputs are final responses
                            yield self._create_completion_response(request.request_id, batch_output)
                else:
                    # Handle single response (for streaming or n=1 non-streaming)
                    if "error" in output:
                        await context.abort(
                            abort_code_from_output(output),
                            output["error"],
                        )
                    elif request.stream:
                        yield self._create_chunk_response(request.request_id, output)
                        if output.get("finished", False):
                            yield self._create_completion_response(request.request_id, output)
                    else:
                        # Non-streaming n=1: single completion response
                        yield self._create_completion_response(request.request_id, output)

        except grpc.aio.AbortError:
            raise
        except ValueError as e:
            logger.warning(f"Generate invalid request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.error(
                f"Generate failed for request {request.request_id}: {e}\n"
                f"{get_exception_traceback()}"
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Embed(
        self,
        request: sglang_scheduler_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.EmbedResponse:
        """Handle embedding requests."""
        logger.info(f"Receive embedding request: {request.request_id}")

        try:
            tokenized_req = self._convert_embed_request(request)

            future = await self.request_manager.embedding_request(
                obj=tokenized_req,
                request_id=request.request_id,
            )

            result = await future

            if "error" in result:
                code = abort_code_from_output(result)
                await context.abort(code, result["error"])
                return

            embedding = result["embedding"]
            return sglang_scheduler_pb2.EmbedResponse(
                embedding=embedding,
                prompt_tokens=result.get("prompt_tokens", 0),
                embedding_dim=len(embedding),
            )

        except grpc.aio.AbortError:
            raise
        except ValueError as e:
            logger.warning(f"Embed invalid request {request.request_id}: {e}")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.error(
                f"Embed failed for request {request.request_id}: {e}\n{get_exception_traceback()}"
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def HealthCheck(
        self,
        request: sglang_scheduler_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.HealthCheckResponse:
        """
        Check the health of the inference server by sending a special request to generate one token.
        Similar to HTTP server's /health endpoint.
        """
        rid = f"HEALTH_CHECK_{time.time()}"
        logger.info(f"Receive health check request: {rid}")

        if self.request_manager.gracefully_exit:
            logger.info("Health check request received during shutdown. Returning unhealthy.")
            return sglang_scheduler_pb2.HealthCheckResponse(
                healthy=False, message="Server is shutting down"
            )

        # Create a special health check request
        sampling_params = SGLSamplingParams(max_new_tokens=1, temperature=0.0)
        sampling_params.normalize(tokenizer=self.tokenizer)

        # Create health check request
        is_generation = self.scheduler_info.get("is_generation")
        if is_generation is None:
            is_generation = not self.server_args.is_embedding

        if is_generation:
            health_req = TokenizedGenerateReqInput(
                rid=rid,
                input_text="",
                input_ids=to_token_id_array([0]),
                input_embeds=None,
                mm_inputs=None,
                token_type_ids=None,
                sampling_params=sampling_params,
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                stream=False,
                token_ids_logprob=None,
                require_reasoning=False,
            )
            # Set disaggregation params if needed
            if self.server_args.disaggregation_mode != DisaggregationMode.NULL.value:
                health_req.bootstrap_host = FAKE_BOOTSTRAP_HOST
                health_req.bootstrap_room = 0
        else:
            sampling_params.max_new_tokens = 0
            health_req = TokenizedEmbeddingReqInput(
                rid=rid,
                input_text="",
                input_ids=to_token_id_array([0]),
                mm_inputs=None,
                token_type_ids=[0],
                sampling_params=sampling_params,
            )

        # Submit health check request
        async def run_health_check():
            try:
                async for _ in self.request_manager.generate_request(
                    obj=health_req,
                    request_id=rid,
                ):
                    # Got at least one response, server is healthy
                    return True
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                return False
            return False

        task = asyncio.create_task(run_health_check())

        # Wait for response with timeout
        tic = time.time()
        while time.time() < tic + HEALTH_CHECK_TIMEOUT:
            await asyncio.sleep(1)
            # Check if we got a response from scheduler
            if self.request_manager.last_receive_tstamp > tic:
                task.cancel()
                # Clean up health check state
                self.request_manager._cleanup_request_state(rid)
                return sglang_scheduler_pb2.HealthCheckResponse(
                    healthy=True, message="Health check passed"
                )

        # Timeout - server not responding
        task.cancel()
        self.request_manager._cleanup_request_state(rid)
        logger.warning(f"Health check timeout after {HEALTH_CHECK_TIMEOUT}s")
        return sglang_scheduler_pb2.HealthCheckResponse(
            healthy=False, message=f"Health check timeout after {HEALTH_CHECK_TIMEOUT}s"
        )

    async def Abort(
        self,
        request: sglang_scheduler_pb2.AbortRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.AbortResponse:
        """Abort an ongoing request."""
        logger.info(f"Receive abort request: {request.request_id}")

        try:
            success = await self.request_manager.abort_request(request.request_id)

            return sglang_scheduler_pb2.AbortResponse(
                success=success,
                message=f"Request {request.request_id} {'aborted' if success else 'not found'}",
            )
        except Exception as e:
            logger.error(
                f"Abort failed for request {request.request_id}: {e}\n{get_exception_traceback()}"
            )
            return sglang_scheduler_pb2.AbortResponse(
                success=False,
                message=str(e),
            )

    def _default_sampling_params_json(self) -> str:
        defaults = dict(self.model_config.get_default_sampling_params() or {})
        preferred = self.server_args.preferred_sampling_params
        if preferred:
            if isinstance(preferred, str):
                try:
                    preferred = json.loads(preferred)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse preferred_sampling_params JSON")
                    preferred = {}
            if isinstance(preferred, dict):
                defaults.update(preferred)
            else:
                logger.warning(
                    "Ignoring preferred_sampling_params with unsupported type: %s",
                    type(preferred).__name__,
                )

        defaults = _filtered_sampling_defaults(defaults)
        return json.dumps(defaults, separators=(",", ":")) if defaults else ""

    def _preferred_sampling_params_json(self) -> str:
        preferred = self.server_args.preferred_sampling_params
        if isinstance(preferred, dict):
            return json.dumps(preferred, separators=(",", ":"))
        if isinstance(preferred, str):
            return preferred
        return ""

    async def GetModelInfo(
        self,
        _request: sglang_scheduler_pb2.GetModelInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetModelInfoResponse:
        """Get model information."""
        logger.debug("Receive model info request")

        is_generation = self.scheduler_info.get("is_generation")
        if is_generation is None:
            is_generation = not self.server_args.is_embedding

        return sglang_scheduler_pb2.GetModelInfoResponse(
            model_path=self.server_args.model_path,
            tokenizer_path=self.server_args.tokenizer_path or "",
            is_generation=is_generation,
            preferred_sampling_params=self._preferred_sampling_params_json(),
            default_sampling_params_json=self._default_sampling_params_json(),
            weight_version=self.server_args.weight_version or "",
            served_model_name=self.server_args.served_model_name,
            max_context_length=self.model_info["max_context_length"],
            vocab_size=self.model_info["vocab_size"],
            supports_vision=self.model_info["supports_vision"],
            model_type=self.model_info.get("model_type") or "",
            architectures=self.model_info.get("architectures") or [],
            eos_token_ids=self.model_info["eos_token_ids"],
            pad_token_id=self.model_info["pad_token_id"],
            bos_token_id=self.model_info["bos_token_id"],
            max_req_input_len=self.model_info["max_req_input_len"],
            # Classification model support
            id2label_json=self.model_info.get("id2label_json") or "",
            num_labels=self.model_info.get("num_labels") or 0,
        )

    async def GetServerInfo(
        self,
        _request: sglang_scheduler_pb2.GetServerInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetServerInfoResponse:
        """Get server information."""
        logger.debug("Receive server info request")

        server_args_dict = dataclasses.asdict(self.server_args)
        server_args_struct = Struct()

        def make_serializable(obj):
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple, set)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return str(obj)

        serializable_args = make_serializable(server_args_dict)
        # Decode constraints (json_schema/regex/ebnf) are enforced by the
        # grammar backend from the first output token; `require_reasoning`
        # does not delay grammar activation in gRPC mode. Declared so the
        # router knows constrained completions cannot contain reasoning.
        serializable_args["constrained_decoding_mode"] = "from_first_token"
        server_args_struct.update(serializable_args)

        # Convert scheduler_info to Struct
        scheduler_info_struct = Struct()
        scheduler_info_struct.update(self.scheduler_info)

        # Get runtime state from request manager
        manager_state = self.request_manager.get_server_info()

        # Calculate uptime
        uptime = time.time() - self.start_time

        # Create timestamp
        start_timestamp = Timestamp()
        start_timestamp.FromSeconds(int(self.start_time))

        return sglang_scheduler_pb2.GetServerInfoResponse(
            server_args=server_args_struct,
            scheduler_info=scheduler_info_struct,
            active_requests=manager_state["active_requests"],
            is_paused=manager_state["paused"],
            last_receive_timestamp=manager_state["last_receive_time"],
            uptime_seconds=uptime,
            sglang_version=sglang.__version__,
            server_type="grpc",
            start_time=start_timestamp,
            max_total_num_tokens=self.scheduler_info.get("max_total_num_tokens", 0),
        )

    async def GetLoads(
        self,
        request: sglang_scheduler_pb2.GetLoadsRequest,
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetLoadsResponse:
        """
        Get comprehensive load metrics for all DP ranks.

        Reads the schedulers' published LoadSnapshots, providing full
        parity with the HTTP /v1/loads endpoint.
        """
        logger.debug("Receive get loads request")

        include = list(request.include) if request.include else ["all"]
        dp_rank = request.dp_rank if request.HasField("dp_rank") else None

        try:
            results = await self.request_manager.get_loads(include=include, dp_rank=dp_rank)
        except ValueError as e:
            # Validation error (e.g., invalid include sections)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return sglang_scheduler_pb2.GetLoadsResponse()
        except TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Timeout waiting for scheduler response")
            return sglang_scheduler_pb2.GetLoadsResponse()
        except Exception as e:
            logger.error(f"GetLoads failed: {e}\n{get_exception_traceback()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get load metrics: {e}")
            return sglang_scheduler_pb2.GetLoadsResponse()

        loads = [_convert_loads_to_protobuf(r) for r in results]

        return sglang_scheduler_pb2.GetLoadsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=sglang.__version__,
            dp_rank_count=len(loads),
            loads=loads,
            aggregate=_compute_aggregate_protobuf(loads),
        )

    async def FlushCache(
        self,
        request: common_pb2.FlushCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.FlushCacheResponse:
        """Flush the KV cache on all scheduler processes."""
        logger.debug("Receive flush cache request (timeout_s=%.1f)", request.timeout_s)
        timeout_s = request.timeout_s
        comm_timeout = max(30.0, timeout_s + 10.0)
        try:
            results = await self.request_manager.send_communicator_req(
                FlushCacheReqInput(timeout_s=timeout_s),
                "flush_cache_communicator",
                timeout=comm_timeout,
            )
        except TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(
                f"Flush cache timed out after {comm_timeout}s. "
                "The scheduler may be unresponsive or under heavy load."
            )
            return common_pb2.FlushCacheResponse(
                success=False, message=f"Flush cache timed out after {comm_timeout}s"
            )
        except Exception as e:
            logger.error(f"FlushCache failed: {e}\n{get_exception_traceback()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Flush cache failed: {e}")
            return common_pb2.FlushCacheResponse(success=False, message=f"Flush cache failed: {e}")

        success, message = _aggregate_communicator_results(results, "Cache flushed successfully")
        if not results:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
        return common_pb2.FlushCacheResponse(success=success, message=message)

    async def StartProfile(
        self,
        request: common_pb2.StartProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.ProfileResponse:
        """Start the profiler on all scheduler processes."""
        logger.debug("Receive start profile request")

        # Env-var overrides mirror sglang's TokenizerCommunicatorMixin defaults.
        with_stack = request.with_stack if request.HasField("with_stack") else None
        with_stack = (with_stack is not False) and get_bool_env_var(
            "SGLANG_PROFILE_WITH_STACK", "true"
        )
        record_shapes = request.record_shapes if request.HasField("record_shapes") else None
        record_shapes = (record_shapes is not False) and get_bool_env_var(
            "SGLANG_PROFILE_RECORD_SHAPES", "true"
        )

        req = ProfileReq(
            req_type=ProfileReqType.START_PROFILE,
            output_dir=request.output_dir if request.HasField("output_dir") else None,
            start_step=request.start_step if request.HasField("start_step") else None,
            num_steps=request.num_steps if request.HasField("num_steps") else None,
            activities=list(request.activities) if request.activities else None,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_by_stage=request.profile_by_stage,
            profile_id=str(time.time()),
        )
        return await self._run_profile_req(req, "Start profiling", context)

    async def StopProfile(
        self,
        request: common_pb2.StopProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.ProfileResponse:
        """Stop the profiler and export traces."""
        logger.debug("Receive stop profile request")
        req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
        return await self._run_profile_req(req, "Stop profiling", context)

    async def _run_profile_req(
        self,
        req: ProfileReq,
        op_name: str,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.ProfileResponse:
        """Send a ProfileReq to the scheduler and aggregate per-rank results."""
        try:
            results = await self.request_manager.send_communicator_req(
                req, "profile_communicator", timeout=PROFILE_COMM_TIMEOUT
            )
        except TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(f"{op_name} timed out after {PROFILE_COMM_TIMEOUT}s")
            return common_pb2.ProfileResponse(
                success=False, message=f"{op_name} timed out after {PROFILE_COMM_TIMEOUT}s"
            )
        except Exception as e:
            logger.error(f"{op_name} failed: {e}\n{get_exception_traceback()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{op_name} failed: {e}")
            return common_pb2.ProfileResponse(success=False, message=f"{op_name} failed: {e}")

        success, message = _aggregate_communicator_results(results, f"{op_name} succeeded")
        if not results:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
        return common_pb2.ProfileResponse(success=success, message=message)

    async def GetTokenizer(
        self,
        request: common_pb2.GetTokenizerRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.GetTokenizerChunk]:
        """Stream tokenizer artifacts as a ZIP bundle.

        Resolves the tokenizer directory from server_args, zips all relevant
        tokenizer files, and streams them as GetTokenizerChunk messages.
        The final chunk carries the SHA-256 fingerprint of the full archive.
        """
        logger.info("Receive GetTokenizer request")

        tokenizer_path = self.server_args.tokenizer_path
        if not tokenizer_path:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Tokenizer path is not configured on this server.",
            )
        tokenizer_dir = Path(tokenizer_path)

        # Build ZIP archive in memory
        try:
            zip_buffer = build_tokenizer_zip(tokenizer_dir)
        except Exception as e:
            logger.error(f"Failed to build tokenizer ZIP: {e}\n{get_exception_traceback()}")
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

    async def SubscribeKvEvents(
        self,
        request: common_pb2.SubscribeKvEventsRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.KvEventBatch]:
        """Bridge internal ZMQ KV cache events to gRPC server-streaming.

        Uses the ZMQ publisher's native sequence numbers as gRPC sequence
        numbers directly.
        """
        if self._kv_events_config is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "KV cache events not enabled. Start SGLang with "
                '--kv-events-config \'{"publisher": "zmq"}\'',
            )
            return

        config = self._kv_events_config

        # Resolve the PUB endpoint to a connectable address.
        # The publisher binds to e.g. "tcp://*:5557"; we connect to localhost.
        pub_endpoint = config.endpoint.replace("*", "127.0.0.1")

        # For DP attention, each rank publishes on port + rank with
        # independent sequence counters. Subscribing to multiple ranks
        # on one socket interleaves independent counters, breaking gap
        # detection. For now, subscribe to rank 0 only.
        # TODO(phase2): per-rank virtual workers or merged renumbering.
        pub_endpoint = ZmqEventPublisher.offset_endpoint_port(pub_endpoint, 0)

        zmq_ctx = zmq.asyncio.Context.instance()
        sub_socket = zmq_ctx.socket(zmq.SUB)
        sub_socket.subscribe(config.topic.encode("utf-8"))
        sub_socket.connect(pub_endpoint)

        logger.info("SubscribeKvEvents: connected to ZMQ endpoint %s", pub_endpoint)

        # Send response headers immediately so the tonic client's
        # subscribe_kv_events().await resolves without waiting for the first
        # yielded event (grpc.aio defers headers until first yield otherwise).
        await context.send_initial_metadata(())

        decoder = msgspec.msgpack.Decoder(KVEventBatch)

        # Stream live events using the ZMQ publisher's native seq numbers.
        try:
            while not context.cancelled():
                try:
                    frames = await asyncio.wait_for(sub_socket.recv_multipart(), timeout=1.0)
                except TimeoutError:
                    continue

                # ZMQ multipart: [topic, seq_bytes, payload]
                if len(frames) < 3:
                    continue

                zmq_seq = int.from_bytes(frames[1], "big")
                payload = frames[2]

                try:
                    raw_batch = decoder.decode(payload)
                except Exception as e:
                    logger.warning("Failed to decode KV event batch: %s", e)
                    continue

                yield self._convert_kv_event_batch(raw_batch, zmq_seq)
        except asyncio.CancelledError:
            pass
        finally:
            sub_socket.close(linger=0)
            logger.info("SubscribeKvEvents: stream closed")

    def _convert_kv_event_batch(
        self, raw_batch: KVEventBatch, seq_num: int
    ) -> common_pb2.KvEventBatch:
        """Convert a ZMQ KVEventBatch to proto KvEventBatch."""
        proto_batch = common_pb2.KvEventBatch(
            sequence_number=seq_num,
            timestamp=raw_batch.ts,
        )
        if raw_batch.attn_dp_rank is not None:
            proto_batch.dp_rank = raw_batch.attn_dp_rank

        for event in raw_batch.events:
            proto_event = self._convert_kv_event(event)
            if proto_event is not None:
                proto_batch.events.append(proto_event)

        return proto_batch

    def _convert_kv_event(self, event) -> common_pb2.KvCacheEvent | None:
        """Convert a single raw KV event to proto KvCacheEvent."""
        self._kv_event_id_counter += 1
        event_id = self._kv_event_id_counter

        if isinstance(event, BlockStored):
            # SGLang emits one BlockStored per page with block_hashes=[single_hash]
            # and token_ids containing only that page's tokens.
            blocks = []
            for i, bh in enumerate(event.block_hashes):
                start = i * event.block_size
                end = start + event.block_size
                block = common_pb2.KvBlock(
                    block_hash=bh,
                    token_ids=event.token_ids[start:end],
                    block_size=event.block_size,
                )
                if event.lora_id is not None:
                    block.lora_id = event.lora_id
                blocks.append(block)

            stored = common_pb2.KvBlocksStored(blocks=blocks)
            if event.parent_block_hash is not None:
                stored.parent_block_hash = event.parent_block_hash

            return common_pb2.KvCacheEvent(event_id=event_id, stored=stored)

        elif isinstance(event, BlockRemoved):
            return common_pb2.KvCacheEvent(
                event_id=event_id,
                removed=common_pb2.KvBlocksRemoved(block_hashes=event.block_hashes),
            )

        elif isinstance(event, AllBlocksCleared):
            return common_pb2.KvCacheEvent(event_id=event_id, cleared=common_pb2.KvCacheCleared())

        return None

    def _handle_epd_disaggregation_encode_request(
        self,
        grpc_req: sglang_scheduler_pb2.GenerateRequest,
        tokenized_req: TokenizedGenerateReqInput,
    ) -> None:
        if not self.mm_receiver:
            return

        image_urls = list(grpc_req.mm_inputs.image_urls)
        if not image_urls:
            return

        encode_req = self.mm_receiver.build_and_send_encode_request(
            image_urls=image_urls,
            rid=grpc_req.request_id,
        )
        tokenized_req.need_wait_for_mm_inputs = bool(encode_req.need_wait_for_mm_inputs)
        tokenized_req.num_items_assigned = encode_req.num_items_assigned

    # Helper methods for request/response conversion

    def _convert_generate_request(
        self, grpc_req: sglang_scheduler_pb2.GenerateRequest
    ) -> TokenizedGenerateReqInput:
        """Convert gRPC GenerateRequest to internal format."""

        # Extract tokenized input
        if not grpc_req.HasField("tokenized"):
            raise ValueError("Tokenized input must be provided")

        input_text = grpc_req.tokenized.original_text
        input_ids = to_token_id_array(grpc_req.tokenized.input_ids)

        # Convert sampling params
        sampling_params = self._convert_sampling_params(grpc_req.sampling_params)
        sampling_params.normalize(tokenizer=self.tokenizer)

        # Extract disaggregated params if present
        bootstrap_host = None
        bootstrap_port = None
        bootstrap_room = None
        if grpc_req.HasField("disaggregated_params"):
            # Don't use 'or None' as it treats 0 as falsy
            bootstrap_host = (
                grpc_req.disaggregated_params.bootstrap_host
                if grpc_req.disaggregated_params.bootstrap_host
                else None
            )
            bootstrap_port = (
                grpc_req.disaggregated_params.bootstrap_port
                if grpc_req.disaggregated_params.bootstrap_port
                else None
            )
            bootstrap_room = (
                grpc_req.disaggregated_params.bootstrap_room
            )  # Can be 0, don't use 'or None'

        # Parse multimodal inputs if present
        mm_inputs = None
        if grpc_req.HasField("mm_inputs") and grpc_req.mm_inputs.HasField("pixel_values"):
            mm_inputs = self._parse_mm_inputs(grpc_req.mm_inputs)

        # Create request
        require_reasoning = bool(grpc_req.require_reasoning)

        return TokenizedGenerateReqInput(
            rid=grpc_req.request_id,
            input_text=input_text,
            input_ids=input_ids,
            input_embeds=None,
            mm_inputs=mm_inputs,
            token_type_ids=None,
            sampling_params=sampling_params,
            return_logprob=grpc_req.return_logprob,
            logprob_start_len=(
                grpc_req.logprob_start_len if grpc_req.logprob_start_len is not None else -1
            ),
            top_logprobs_num=grpc_req.top_logprobs_num or 0,
            stream=grpc_req.stream or False,
            lora_id=grpc_req.lora_id if grpc_req.lora_id else None,
            token_ids_logprob=(
                list(grpc_req.token_ids_logprob) if grpc_req.token_ids_logprob else None
            ),
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            require_reasoning=require_reasoning,
        )

    @staticmethod
    def _decode_tensor_data(tensor_data):
        """Decode a proto TensorData message into a torch.Tensor."""
        dtype_map = {"float32": np.float32, "int64": np.int64}
        np_dtype = dtype_map.get(tensor_data.dtype, np.float32)
        shape = list(tensor_data.shape)
        arr = np.frombuffer(tensor_data.data, dtype=np_dtype).reshape(shape)
        return torch.from_numpy(arr)

    def _parse_mm_inputs(self, mm_proto) -> MultimodalProcessorOutput:
        """Parse proto MultimodalInputs into a MultimodalProcessorOutput for the scheduler."""
        # Decode pixel_values from typed TensorData field
        pixel_values = self._decode_tensor_data(mm_proto.pixel_values)

        # Decode model-specific tensors
        model_specific_data = {}
        for key, tensor_data in mm_proto.model_specific_tensors.items():
            model_specific_data[key] = self._decode_tensor_data(tensor_data)

        # Convert placeholder ranges to offsets: list of (start, end_inclusive)
        offsets = [(p.offset, p.offset + p.length - 1) for p in mm_proto.mm_placeholders]
        if not offsets:
            logger.warning(
                "No mm_placeholders from Rust gateway — token expansion may have "
                "failed to find the placeholder token in input_ids. "
                "Check that placeholder_token_id matches the tokenized image token."
            )
            offsets = None

        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=pixel_values,
            model_specific_data=model_specific_data,
            offsets=offsets,
        )

        im_token_id = mm_proto.im_token_id if mm_proto.HasField("im_token_id") else None

        return MultimodalProcessorOutput(
            mm_items=[mm_item],
            im_token_id=im_token_id,
        )

    def _convert_embed_request(
        self, grpc_req: sglang_scheduler_pb2.EmbedRequest
    ) -> TokenizedEmbeddingReqInput:
        """Convert gRPC EmbedRequest to internal format."""

        # Extract tokenized input
        if not grpc_req.HasField("tokenized"):
            raise ValueError("Tokenized input must be provided")

        input_text = grpc_req.tokenized.original_text
        input_ids = to_token_id_array(grpc_req.tokenized.input_ids)

        # Convert sampling params
        sampling_params = self._convert_sampling_params(grpc_req.sampling_params)

        # For embedding requests, max_new_tokens should be 0.
        # The scheduler logic expects an integer, not None.
        sampling_params.max_new_tokens = 0

        sampling_params.normalize(tokenizer=self.tokenizer)

        return TokenizedEmbeddingReqInput(
            rid=grpc_req.request_id,
            input_text=input_text,
            input_ids=input_ids,
            mm_inputs=None,
            token_type_ids=list(grpc_req.token_type_ids),
            sampling_params=sampling_params,
        )

    def _convert_sampling_params(
        self, grpc_params: sglang_scheduler_pb2.SamplingParams
    ) -> SGLSamplingParams:
        """Convert gRPC SamplingParams to internal format."""

        # Handle constraint types
        regex = None
        json_schema = None
        ebnf_grammar = None
        structural_tag = None

        if grpc_params.HasField("regex"):
            regex = grpc_params.regex
        elif grpc_params.HasField("json_schema"):
            json_schema = grpc_params.json_schema
        elif grpc_params.HasField("ebnf_grammar"):
            ebnf_grammar = grpc_params.ebnf_grammar
        elif grpc_params.HasField("structural_tag"):
            structural_tag = grpc_params.structural_tag

        # Handle optional parameters conversion
        custom_params = (
            MessageToDict(grpc_params.custom_params)
            if grpc_params.HasField("custom_params")
            else None
        )
        max_new_tokens = (
            grpc_params.max_new_tokens if grpc_params.HasField("max_new_tokens") else None
        )
        stream_interval = (
            grpc_params.stream_interval if grpc_params.HasField("stream_interval") else None
        )
        logit_bias = dict(grpc_params.logit_bias) if grpc_params.logit_bias else None
        stop = list(grpc_params.stop) if grpc_params.stop else None
        stop_token_ids = list(grpc_params.stop_token_ids) if grpc_params.stop_token_ids else None

        return SGLSamplingParams(
            temperature=grpc_params.temperature,
            top_p=grpc_params.top_p,
            top_k=grpc_params.top_k,
            min_p=grpc_params.min_p,
            frequency_penalty=grpc_params.frequency_penalty,
            presence_penalty=grpc_params.presence_penalty,
            repetition_penalty=grpc_params.repetition_penalty,
            max_new_tokens=max_new_tokens,
            min_new_tokens=grpc_params.min_new_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=grpc_params.skip_special_tokens,
            spaces_between_special_tokens=grpc_params.spaces_between_special_tokens,
            no_stop_trim=grpc_params.no_stop_trim,
            regex=regex,
            json_schema=json_schema,
            ebnf=ebnf_grammar,
            structural_tag=structural_tag,
            n=grpc_params.n,
            ignore_eos=grpc_params.ignore_eos,
            stream_interval=stream_interval,
            logit_bias=logit_bias,
            custom_params=custom_params,
        )

    def _convert_output_logprobs_to_proto(
        self, logprobs_data: dict
    ) -> sglang_scheduler_pb2.OutputLogProbs | None:
        """Convert output logprobs dict to proto (no None values, plain floats)."""
        if not logprobs_data:
            return None

        token_logprobs_val = logprobs_data.get("token_logprobs_val", [])
        token_logprobs_idx = logprobs_data.get("token_logprobs_idx", [])
        top_logprobs_val = logprobs_data.get("top_logprobs_val", [])
        top_logprobs_idx = logprobs_data.get("top_logprobs_idx", [])

        # Build TopLogProbs entries
        top_logprobs_proto = []
        if top_logprobs_val and top_logprobs_idx:
            for val_list, idx_list in zip(top_logprobs_val, top_logprobs_idx):
                top_logprobs_proto.append(
                    sglang_scheduler_pb2.TopLogProbs(
                        values=val_list,
                        token_ids=idx_list,
                    )
                )

        return sglang_scheduler_pb2.OutputLogProbs(
            token_logprobs=token_logprobs_val,  # Plain float array
            token_ids=token_logprobs_idx,
            top_logprobs=top_logprobs_proto,
        )

    def _convert_input_logprobs_to_proto(
        self, logprobs_data: dict
    ) -> sglang_scheduler_pb2.InputLogProbs | None:
        """Convert input logprobs dict to proto (first token is None, wrapped in InputTokenLogProb)."""
        if not logprobs_data:
            return None

        token_logprobs_val = logprobs_data.get("token_logprobs_val", [])
        token_logprobs_idx = logprobs_data.get("token_logprobs_idx", [])
        top_logprobs_val = logprobs_data.get("top_logprobs_val", [])
        top_logprobs_idx = logprobs_data.get("top_logprobs_idx", [])

        # Wrap values in InputTokenLogProb (None for first token, value for others)
        token_logprobs_wrapped = [
            (
                sglang_scheduler_pb2.InputTokenLogProb()
                if x is None
                else sglang_scheduler_pb2.InputTokenLogProb(value=x)
            )
            for x in token_logprobs_val
        ]

        # Build TopLogProbs entries
        top_logprobs_proto = []
        if top_logprobs_val and top_logprobs_idx:
            for val_list, idx_list in zip(top_logprobs_val, top_logprobs_idx):
                top_logprobs_proto.append(
                    sglang_scheduler_pb2.TopLogProbs(
                        values=val_list,
                        token_ids=idx_list,
                    )
                )

        return sglang_scheduler_pb2.InputLogProbs(
            token_logprobs=token_logprobs_wrapped,
            token_ids=token_logprobs_idx,
            top_logprobs=top_logprobs_proto,
        )

    def _create_chunk_response(
        self, request_id: str, output: dict
    ) -> sglang_scheduler_pb2.GenerateResponse:
        """Create a streaming chunk response."""
        meta_info = output.get("meta_info", {})

        # Convert output logprobs if present
        output_logprobs_proto = self._convert_output_logprobs_to_proto(
            output.get("output_logprobs")
        )

        # Convert input logprobs if present (only in first chunk)
        input_logprobs_proto = self._convert_input_logprobs_to_proto(output.get("input_logprobs"))

        return sglang_scheduler_pb2.GenerateResponse(
            request_id=request_id,
            chunk=sglang_scheduler_pb2.GenerateStreamChunk(
                token_ids=output.get("token_ids", []),
                prompt_tokens=meta_info.get("prompt_tokens", 0),
                completion_tokens=meta_info.get("completion_tokens", 0),
                cached_tokens=meta_info.get("cached_tokens", 0),
                reasoning_tokens=meta_info.get("reasoning_tokens", 0),
                output_logprobs=output_logprobs_proto,
                input_logprobs=input_logprobs_proto,
                index=output.get("index", 0),
            ),
        )

    def _create_completion_response(
        self, request_id: str, output: dict
    ) -> sglang_scheduler_pb2.GenerateResponse:
        """Create a completion response."""

        # Extract meta info and finish reason details
        meta_info = output.get("meta_info", {})
        finish_reason_data = meta_info.get("finish_reason")

        # Determine finish reason, default is stop
        finish_reason = "stop"
        if finish_reason_data:
            if isinstance(finish_reason_data, dict):
                finish_reason_type = finish_reason_data.get("type")
            else:
                # Handle legacy string format
                finish_reason_type = finish_reason_data

            if finish_reason_type == "length":
                finish_reason = "length"
            elif finish_reason_type == "abort":
                finish_reason = "abort"

        # Extract matched_stop information
        matched_stop_kwargs = {}
        if isinstance(finish_reason_data, dict) and "matched" in finish_reason_data:
            matched = finish_reason_data["matched"]
            if isinstance(matched, int):
                matched_stop_kwargs["matched_token_id"] = matched
            elif isinstance(matched, str):
                matched_stop_kwargs["matched_stop_str"] = matched

        # Convert output logprobs if present
        output_logprobs_proto = self._convert_output_logprobs_to_proto(
            output.get("output_logprobs")
        )

        # Convert input logprobs if present
        input_logprobs_proto = self._convert_input_logprobs_to_proto(output.get("input_logprobs"))

        return sglang_scheduler_pb2.GenerateResponse(
            request_id=request_id,
            complete=sglang_scheduler_pb2.GenerateComplete(
                output_ids=output.get("token_ids", []),
                finish_reason=finish_reason,
                prompt_tokens=meta_info.get("prompt_tokens", 0),
                completion_tokens=meta_info.get(
                    "completion_tokens", len(output.get("token_ids", []))
                ),
                cached_tokens=meta_info.get("cached_tokens", 0),
                reasoning_tokens=meta_info.get("reasoning_tokens", 0),
                output_logprobs=output_logprobs_proto,
                input_logprobs=input_logprobs_proto,
                index=output.get("index", 0),
                **matched_stop_kwargs,
            ),
        )

    async def shutdown(self):
        """Shutdown the service."""
        logger.info("Shutting down gRPC service")

        # Mark health service as NOT_SERVING before shutdown
        if self.health_servicer:
            self.health_servicer.set_not_serving()

        # Shutdown request manager (handles its own tasks)
        await self.request_manager.shutdown()
