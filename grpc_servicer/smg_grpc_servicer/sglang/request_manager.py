"""
gRPC Request Manager - Orchestrates request lifecycle without tokenization.
Mimics TokenizerManager's state management and ZMQ communication patterns.
"""

import asyncio
import copy
import dataclasses
import logging
import os
import signal
import sys
import threading
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import grpc
import zmq
import zmq.asyncio
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    FlushCacheReqOutput,
    GetInternalStateReqOutput,
    HealthCheckOutput,
    ProfileReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot, create_load_snapshot_reader
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    calibrate_time_diff,
    real_time,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_or_create_event_loop, kill_process_tree
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class _GrpcCommunicator:
    """
    Communicator for request/response patterns with scheduler.

    Thread-safe and handles the async request/response cycle with proper
    timeout handling to prevent hangs if the scheduler becomes unresponsive.
    """

    DEFAULT_TIMEOUT = 30.0  # seconds

    def __init__(self, sender: zmq.Socket, fan_out: int = 1):
        self._sender = sender
        self._fan_out = fan_out
        self._result_event: asyncio.Event | None = None
        self._result_values: list[Any] | None = None
        self._lock = asyncio.Lock()

    async def __call__(self, obj, timeout: float = DEFAULT_TIMEOUT) -> list[Any]:
        """
        Send request and wait for response(s).

        Args:
            obj: Request object to send to scheduler
            timeout: Maximum time to wait for response (seconds)

        Returns:
            List of response objects from scheduler(s)

        Raises:
            asyncio.TimeoutError: If no response within timeout
        """
        async with self._lock:
            # Initialize state BEFORE sending to avoid race condition
            self._result_event = asyncio.Event()
            self._result_values = []

            # Send request to scheduler
            if obj:
                self._sender.send_pyobj(obj)

            try:
                # Wait for response(s) with timeout
                await asyncio.wait_for(self._result_event.wait(), timeout=timeout)
                return self._result_values
            finally:
                # Always clean up state
                self._result_event = None
                self._result_values = None

    def handle_recv(self, recv_obj: Any):
        """
        Handle received response from scheduler.

        Called by handle_loop when a matching response type is received.
        Safe to call even if no request is pending (will be ignored).
        """
        if self._result_values is not None and self._result_event is not None:
            self._result_values.append(recv_obj)
            if len(self._result_values) >= self._fan_out:
                self._result_event.set()


class GrpcSignalHandler:
    """Minimal signal handler for gRPC server - delegates real crash handling to scheduler."""

    def __init__(self, grpc_manager):
        self.grpc_manager = grpc_manager

    def sigterm_handler(self, signum=None, frame=None):
        """Handle SIGTERM by gracefully shutting down gRPC server."""
        logger.warning(f"SIGTERM received. {signum=} {frame=}. Shutting down gRPC server...")
        self.grpc_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        """Handle SIGQUIT from failed scheduler process."""
        logger.error(
            "Received SIGQUIT from scheduler process. Scheduler failed, shutting down gRPC server."
        )
        logger.info("Note: Crash dumps are handled by the scheduler process, not the gRPC server.")
        # Just exit cleanly - the scheduler handles crash dumps
        kill_process_tree(os.getpid(), include_parent=True)


@dataclasses.dataclass
class GrpcReqState:
    """State tracking for a gRPC request."""

    # Request identification
    request_id: str
    grpc_context: grpc.aio.ServicerContext | None

    # Communication
    out_queue: asyncio.Queue
    finished: bool
    event: asyncio.Event
    obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput

    # Metrics (same as TokenizerManager's ReqState)
    time_stats: APIServerReqTimeStats
    last_completion_tokens: int = 1

    # Streaming state
    stream_finished: bool = False
    input_logprobs_sent: bool = False  # Track if input logprobs were sent in streaming

    # Token accumulation (for non-streaming)
    output_ids: list[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: list[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: list[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: list[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: list[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: list[list[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: list[list[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: list[list[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: list[list[int]] = dataclasses.field(default_factory=list)

    # Session state
    session_id: str | None = None
    is_session_request: bool = False


class GrpcRequestManager:
    """
    Manages gRPC request lifecycle, mimicking TokenizerManager's orchestration
    behaviors without tokenization.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        bootstrap_server=None,
    ):
        """Initialize the gRPC request manager."""
        self.server_args = server_args
        self.port_args = port_args

        # ZMQ Communication Setup (same pattern as TokenizerManager)
        self.context = zmq.asyncio.Context(2)

        # Socket for receiving outputs from scheduler
        # If skip_tokenizer_init mode, scheduler sends outputs to tokenizer_ipc_name.
        self.recv_ipc_name = (
            port_args.tokenizer_ipc_name
            if server_args.skip_tokenizer_init
            else port_args.detokenizer_ipc_name
        )
        self.recv_from_scheduler = get_zmq_socket(
            self.context, zmq.PULL, self.recv_ipc_name, bind=True
        )

        # Communicator responses (ProfileReqOutput, FlushCacheReqOutput, ...)
        # always arrive on tokenizer_ipc_name via the scheduler's
        # send_to_tokenizer path. In skip_tokenizer_init mode
        # recv_from_scheduler already binds that endpoint, so generation
        # outputs and communicator responses share one socket; otherwise a
        # second socket is needed for communicator responses.
        if server_args.skip_tokenizer_init:
            self.recv_from_tokenizer = None
        else:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.tokenizer_ipc_name, bind=True
            )

        # Socket for sending requests to scheduler
        self.send_to_scheduler = get_zmq_socket(
            self.context, zmq.PUSH, port_args.scheduler_input_ipc_name, bind=True
        )

        # State Management (from TokenizerManager)
        self.rid_to_state: dict[str, GrpcReqState] = {}
        self.asyncio_tasks: set = set()
        self.gracefully_exit = False
        self.no_create_loop = False
        self.event_loop = None

        # Pause/Resume Control
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()

        # Metrics
        self.last_receive_tstamp = real_time()

        # Crash dump for debugging
        self.crash_dump_request_list = []
        self.crash_dump_performed = False

        # disaggregation mode
        self.disaggregation_mode = DisaggregationMode(server_args.disaggregation_mode)

        # Bootstrap server (passed from serve_grpc, not started here)
        self.bootstrap_server = bootstrap_server

        # Schedulers publish per-dp-rank LoadSnapshots (SHM; zmq for
        # multi-node DP). This manager fills sglang's TokenizerManager role.
        # sglang 0.5.18 dropped the server_args parameter; the reader now
        # derives dp_size from the process-global parallel config itself.
        self.load_snapshot_reader = create_load_snapshot_reader(
            port_args, caller="TokenizerManager"
        )

        # Communicators for request/response patterns with scheduler
        # Note: These must be initialized after send_to_scheduler socket is created
        self.profile_communicator = _GrpcCommunicator(
            self.send_to_scheduler, fan_out=server_args.dp_size
        )
        self.flush_cache_communicator = _GrpcCommunicator(
            self.send_to_scheduler, fan_out=server_args.dp_size
        )
        self.get_internal_state_communicator = _GrpcCommunicator(
            self.send_to_scheduler, fan_out=server_args.dp_size
        )

        logger.info(
            f"GrpcRequestManager initialized with ZMQ IPC: "
            f"recv={self.recv_ipc_name}, "
            f"send={port_args.scheduler_input_ipc_name}"
        )
        if self.bootstrap_server:
            logger.info(
                f"Bootstrap server initialized for disaggregation mode: "
                f"{server_args.disaggregation_mode}"
            )

    async def generate_request(
        self,
        obj: TokenizedGenerateReqInput,
        request_id: str | None = None,
        grpc_context: grpc.aio.ServicerContext | None = None,
    ) -> AsyncGenerator[dict | list[dict], None]:
        """
        Submit a generation request to the scheduler with n>1 parallel sampling support.

        This method implements the same two-phase approach as tokenizer_manager.py:
        1. Phase 1: Send prefix caching request (max_new_tokens=0)
        2. Phase 2: Send n generation requests that reuse the cached prefix

        Yields individual responses for streaming, or aggregated responses for non-streaming.
        """
        n = getattr(obj.sampling_params, "n", 1)

        if n <= 1:
            async for response in self._handle_single_request(obj, request_id, grpc_context):
                yield response
            return

        # N>1 handling - two-phase approach
        logger.debug(f"Multiple sampling request (n={n}), using two-phase approach")

        # Generate base request ID if not provided
        if request_id is None:
            base_request_id = f"grpc-{uuid.uuid4().hex}"
        else:
            base_request_id = request_id

        # Phase 1: Cache the common prefix
        logger.debug(f"Phase 1: Caching prefix for request {base_request_id}")
        prefix_obj = copy.copy(obj)
        prefix_obj.sampling_params = copy.copy(obj.sampling_params)
        prefix_obj.sampling_params.max_new_tokens = 0  # Prefill-only
        prefix_obj.sampling_params.n = 1  # Don't replicate prefix request

        # Send prefix caching request and consume response
        async for _ in self._handle_single_request(
            prefix_obj, f"{base_request_id}-prefix", grpc_context
        ):
            # Consume prefix response (usually just one chunk with finish_reason)
            pass

        logger.debug(f"Phase 1 completed: Prefix cached for {base_request_id}")

        # Phase 2: Generate n parallel requests
        logger.debug(f"Phase 2: Generating {n} parallel requests")
        generators = []
        request_ids = []

        for i in range(n):
            # Create individual generation request
            gen_obj = copy.copy(obj)
            gen_obj.sampling_params = copy.copy(obj.sampling_params)
            gen_obj.sampling_params.n = 1  # Each request generates 1 response

            gen_request_id = f"{base_request_id}-{i}"
            request_ids.append(gen_request_id)

            # Start generation request
            generators.append(self._handle_single_request(gen_obj, gen_request_id, grpc_context))

        # Handle response aggregation
        is_stream = getattr(obj, "stream", False)

        if not is_stream:
            # Non-streaming: collect all responses and return as batch
            logger.debug(f"Non-streaming mode: collecting {n} responses")
            responses = []
            for generator in generators:
                async for response in generator:
                    responses.append(response)
            yield responses  # Return all responses as a batch
        else:
            # Streaming mode: multiplex responses with index for ordering
            logger.debug(f"Streaming mode: multiplexing {n} streams")
            rid_to_index = {rid: i for i, rid in enumerate(request_ids)}

            # Create async tasks for all generators
            task_map = {}
            for generator in generators:
                task = asyncio.create_task(generator.__anext__())
                task_map[task] = generator

            # Process responses as they arrive
            while task_map:
                done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    generator = task_map.pop(task)
                    try:
                        response = await task

                        # Add index for client-side ordering
                        if isinstance(response, dict):
                            response_rid = response.get("request_id", "")
                            if response_rid in rid_to_index:
                                response["index"] = rid_to_index[response_rid]

                        yield response

                        # Create next task for this generator
                        next_task = asyncio.create_task(generator.__anext__())
                        task_map[next_task] = generator

                    except StopAsyncIteration:
                        # This generator is finished
                        pass

    async def _handle_single_request(
        self,
        obj: TokenizedGenerateReqInput,
        request_id: str | None = None,
        grpc_context: grpc.aio.ServicerContext | None = None,
    ):
        """Handle a single request - core implementation without n>1 logic."""
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"grpc-{uuid.uuid4().hex}"

        obj.rid = request_id

        self._req_stats_init(obj, grpc_context)
        state = self.rid_to_state[request_id]
        self.record_request_for_crash_dump(obj)

        try:
            # Send to scheduler - let exceptions bubble up to grpc_server.py
            state.time_stats.set_api_server_dispatch_time()
            await self._send_to_scheduler(obj)
            state.time_stats.set_api_server_dispatch_finish_time()

            is_stream = getattr(obj, "stream", False)

            while True:
                try:
                    response = await state.out_queue.get()

                    if is_stream:
                        yield response

                    # Non-streaming: yield final response with accumulated tokens from state
                    if isinstance(response, dict) and response.get("finished", False):
                        if not is_stream:
                            final_response = response.copy()
                            final_response["token_ids"] = state.output_ids
                            yield final_response
                        break

                except asyncio.CancelledError:
                    # Task was cancelled by gRPC framework when client disconnected
                    logger.info(f"Request {request_id} cancelled by client")
                    await self.abort_request(request_id)
                    raise  # Re-raise to let gRPC server handle cleanup

        finally:
            # Always clean up request state when exiting
            self._cleanup_request_state(request_id)

    def _cleanup_request_state(self, request_id: str):
        """Clean up local request state (does not notify scheduler)."""
        if request_id in self.rid_to_state:
            del self.rid_to_state[request_id]

    async def embedding_request(
        self,
        obj: TokenizedEmbeddingReqInput,
        request_id: str | None = None,
    ) -> asyncio.Future:
        """
        Submit an embedding request to the scheduler.
        Returns a future that will contain the embedding result.
        """
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"grpc-embed-{uuid.uuid4().hex}"

        obj.rid = request_id

        self._req_stats_init(obj)
        state = self.rid_to_state[request_id]

        # Create future for result
        future = asyncio.Future()

        # Send to scheduler
        try:
            state.time_stats.set_api_server_dispatch_time()
            await self._send_to_scheduler(obj)
            state.time_stats.set_api_server_dispatch_finish_time()
        except Exception as e:
            del self.rid_to_state[request_id]
            future.set_exception(e)
            return future

        # Wait for result in background
        async def wait_for_result():
            try:
                await state.event.wait()
                result = await state.out_queue.get()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                # Clean up
                if request_id in self.rid_to_state:
                    del self.rid_to_state[request_id]

        asyncio.create_task(wait_for_result())
        return future

    async def abort_request(self, request_id: str) -> bool:
        """Abort a running request.

        Sends abort request to scheduler and marks local state as finished
        to stop processing any further outputs from the scheduler.
        """
        # Skip aborting health check requests (they clean themselves up)
        if request_id.startswith("HEALTH_CHECK"):
            return False

        # Mark state as finished immediately to stop processing scheduler outputs
        state = self.rid_to_state.get(request_id)
        if state:
            state.finished = True
            state.stream_finished = True
            logger.debug(f"Marked request {request_id} as aborted locally")

        # Send abort to scheduler - the scheduler will send AbortReq back
        # which will be handled by _handle_abort_req
        abort_req = AbortReq(rid=request_id)
        try:
            await self._send_to_scheduler(abort_req)
            logger.debug(f"Sent abort to scheduler for request {request_id}")
        except Exception as e:
            logger.error(f"Failed to send abort request to scheduler: {e}")
            return False

        return True

    async def handle_loop(self):
        """
        Main event loop - processes outputs from scheduler.
        Mimics TokenizerManager's handle_loop.
        """
        while not self.gracefully_exit:
            try:
                # Receive from scheduler
                recv_obj = await self.recv_from_scheduler.recv_pyobj()
                self.last_receive_tstamp = real_time()

                # Check for pause (optimized: check flag before acquiring lock)
                if self.is_pause:
                    async with self.is_pause_cond:
                        while self.is_pause:
                            await self.is_pause_cond.wait()

                # Handle different output types
                if isinstance(recv_obj, BatchTokenIDOutput):
                    await self._handle_batch_output(recv_obj)
                elif isinstance(recv_obj, BatchEmbeddingOutput):
                    await self._handle_embedding_output(recv_obj)
                elif isinstance(recv_obj, HealthCheckOutput):
                    await self._handle_health_check_output(recv_obj)
                elif isinstance(recv_obj, AbortReq):
                    await self._handle_abort_req(recv_obj)
                elif self._dispatch_communicator_output(recv_obj):
                    # Communicator responses arrive here in skip_tokenizer_init
                    # mode, where this socket is bound to tokenizer_ipc_name.
                    pass
                else:
                    logger.warning(f"Unknown output type: {type(recv_obj)}")

            except zmq.error.Again:
                # Timeout, check if we should exit
                if self.gracefully_exit:
                    break
                continue
            except zmq.error.ZMQError as e:
                # Socket closed or other ZMQ error - exit cleanly if shutting down
                if self.gracefully_exit:
                    logger.debug(f"ZMQ recv interrupted during shutdown: {e}")
                    break
                logger.error(f"ZMQ error in handle loop: {e}\n{get_exception_traceback()}")
                break
            except Exception as e:
                logger.error(f"Handle loop error: {e}\n{get_exception_traceback()}")
                if self.gracefully_exit:
                    break

    def _dispatch_communicator_output(self, recv_obj) -> bool:
        """Route a communicator response from the scheduler to its communicator.

        Returns True when the object was a recognized communicator response.
        """
        if isinstance(recv_obj, ProfileReqOutput):
            self.profile_communicator.handle_recv(recv_obj)
        elif isinstance(recv_obj, FlushCacheReqOutput):
            self.flush_cache_communicator.handle_recv(recv_obj)
        elif isinstance(recv_obj, GetInternalStateReqOutput):
            self.get_internal_state_communicator.handle_recv(recv_obj)
        else:
            return False
        return True

    async def _handle_tokenizer_loop(self):
        """Process communicator responses from the scheduler's send_to_tokenizer path.

        The scheduler sends communicator responses (ProfileReqOutput,
        FlushCacheReqOutput, ...) directly to tokenizer_ipc_name, not through
        the detokenizer. This loop only runs when recv_from_tokenizer is a
        dedicated socket — in skip_tokenizer_init mode those responses arrive
        on recv_from_scheduler and are dispatched by handle_loop instead.
        """
        while not self.gracefully_exit:
            try:
                recv_obj = await self.recv_from_tokenizer.recv_pyobj()
                if not self._dispatch_communicator_output(recv_obj):
                    logger.warning(f"Unknown type on tokenizer socket: {type(recv_obj)}")
            except zmq.error.Again:
                if self.gracefully_exit:
                    break
                continue
            except zmq.error.ZMQError as e:
                if self.gracefully_exit:
                    logger.debug(f"ZMQ recv interrupted during shutdown: {e}")
                    break
                logger.error(f"ZMQ error in tokenizer loop: {e}\n{get_exception_traceback()}")
                break
            except Exception as e:
                logger.error(f"Tokenizer loop error: {e}\n{get_exception_traceback()}")
                if self.gracefully_exit:
                    break

    def _convert_logprob_style(
        self,
        state: GrpcReqState,
        batch_out: BatchTokenIDOutput,
        batch_index: int,
    ):
        """
        Convert and accumulate logprobs from batch output to state.
        Follows the same logic as tokenizer_manager.convert_logprob_style.
        """
        # Early exit if no input logprobs at all
        if batch_out.input_token_logprobs_val is None:
            return

        # Accumulate input token logprobs (only if list is non-empty)
        if len(batch_out.input_token_logprobs_val) > 0:
            state.input_token_logprobs_val.extend(batch_out.input_token_logprobs_val[batch_index])
            state.input_token_logprobs_idx.extend(batch_out.input_token_logprobs_idx[batch_index])

        # Always accumulate output token logprobs
        state.output_token_logprobs_val.extend(batch_out.output_token_logprobs_val[batch_index])
        state.output_token_logprobs_idx.extend(batch_out.output_token_logprobs_idx[batch_index])

        # Handle top logprobs if requested
        if state.obj.top_logprobs_num > 0:
            # Accumulate input top logprobs (only if list is non-empty)
            if len(batch_out.input_top_logprobs_val) > 0:
                state.input_top_logprobs_val.extend(batch_out.input_top_logprobs_val[batch_index])
                state.input_top_logprobs_idx.extend(batch_out.input_top_logprobs_idx[batch_index])

            # Always accumulate output top logprobs
            state.output_top_logprobs_val.extend(batch_out.output_top_logprobs_val[batch_index])
            state.output_top_logprobs_idx.extend(batch_out.output_top_logprobs_idx[batch_index])

    async def _handle_batch_output(self, batch_out: BatchTokenIDOutput):
        """Handle batch generation output from scheduler."""
        # Collect all queue.put() tasks for parallel execution
        put_tasks = []
        cleanup_tasks = []

        # Process each request in the batch
        for i, rid in enumerate(batch_out.rids):
            if rid not in self.rid_to_state:
                continue

            state = self.rid_to_state[rid]

            # Skip if already aborted/finished locally (client cancelled)
            if state.finished:
                logger.debug(f"Skipping output for aborted request {rid}")
                continue

            # Update metrics
            if state.time_stats.first_token_time == 0.0:
                state.time_stats.set_first_token_time()
            else:
                state.time_stats.set_last_time()

            # Extract output for this request
            prompt_tokens = batch_out.prompt_tokens[i] if batch_out.prompt_tokens else 0
            completion_tokens = batch_out.completion_tokens[i] if batch_out.completion_tokens else 0
            cached_tokens = batch_out.cached_tokens[i] if batch_out.cached_tokens else 0
            reasoning_tokens_list = getattr(batch_out, "reasoning_tokens", None)
            reasoning_tokens = reasoning_tokens_list[i] if reasoning_tokens_list else 0
            token_ids = batch_out.output_ids[i] if batch_out.output_ids else []
            finished = batch_out.finished_reasons[i] is not None
            finish_reason = batch_out.finished_reasons[i] if batch_out.finished_reasons[i] else None

            output_data = {
                "request_id": rid,
                "token_ids": token_ids,
                "finished": finished,
                "meta_info": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": finish_reason,
                },
            }

            # Accumulate logprobs (following tokenizer_manager pattern)
            # Use getattr for safe access - not all request types have return_logprob
            # (e.g., TokenizedEmbeddingReqInput)
            if getattr(state.obj, "return_logprob", False):
                self._convert_logprob_style(state, batch_out, i)

            # Send input logprobs based if available
            if (
                getattr(state.obj, "return_logprob", False)
                and state.obj.logprob_start_len >= 0
                and state.input_token_logprobs_val
            ):
                if state.obj.stream and not state.input_logprobs_sent:
                    # Streaming: send input logprobs once in first chunk that has them
                    output_data["input_logprobs"] = {
                        "token_logprobs_val": state.input_token_logprobs_val,
                        "token_logprobs_idx": state.input_token_logprobs_idx,
                        "top_logprobs_val": state.input_top_logprobs_val,
                        "top_logprobs_idx": state.input_top_logprobs_idx,
                    }
                    state.input_logprobs_sent = True
                elif not state.obj.stream and output_data["finished"]:
                    # Non-streaming: send input logprobs in final chunk
                    output_data["input_logprobs"] = {
                        "token_logprobs_val": state.input_token_logprobs_val,
                        "token_logprobs_idx": state.input_token_logprobs_idx,
                        "top_logprobs_val": state.input_top_logprobs_val,
                        "top_logprobs_idx": state.input_top_logprobs_idx,
                    }

            # Send output logprobs if available
            if (
                getattr(state.obj, "return_logprob", False)
                and batch_out.output_token_logprobs_val
                and i < len(batch_out.output_token_logprobs_val)
            ):
                if state.obj.stream:
                    # For streaming: send incremental logprobs (only new tokens in this chunk)
                    # NOTE: this is different than TokenizerManager, which always accumulates
                    def get_part(attr_name):
                        source_list = getattr(batch_out, attr_name, None)
                        return source_list[i] if source_list and i < len(source_list) else []

                    output_data["output_logprobs"] = {
                        "token_logprobs_val": batch_out.output_token_logprobs_val[i],
                        "token_logprobs_idx": get_part("output_token_logprobs_idx"),
                        "top_logprobs_val": get_part("output_top_logprobs_val"),
                        "top_logprobs_idx": get_part("output_top_logprobs_idx"),
                    }
                elif output_data["finished"]:
                    # Non-streaming: send cumulative output logprobs in final chunk
                    output_data["output_logprobs"] = {
                        "token_logprobs_val": state.output_token_logprobs_val,
                        "token_logprobs_idx": state.output_token_logprobs_idx,
                        "top_logprobs_val": state.output_top_logprobs_val,
                        "top_logprobs_idx": state.output_top_logprobs_idx,
                    }

            # Update state for accumulation
            if output_data["token_ids"]:
                state.output_ids.extend(output_data["token_ids"])

            # Add queue.put() to parallel task list
            put_tasks.append(state.out_queue.put(output_data))

            # Handle completion
            if output_data["finished"]:
                state.finished = True
                state.time_stats.set_finished_time()
                state.stream_finished = True
                state.event.set()

                # Remove from tracking after a delay
                async def cleanup(request_id):
                    await asyncio.sleep(5.0)
                    if request_id in self.rid_to_state:
                        del self.rid_to_state[request_id]

                cleanup_tasks.append(asyncio.create_task(cleanup(rid)))

        # Execute all queue.put() operations in parallel
        if put_tasks:
            await asyncio.gather(*put_tasks, return_exceptions=True)

    async def _handle_embedding_output(self, batch_out: BatchEmbeddingOutput):
        """Handle batch embedding output from scheduler."""
        for i, rid in enumerate(batch_out.rids):
            if rid not in self.rid_to_state:
                continue

            state = self.rid_to_state[rid]

            # Create result
            result = {
                "request_id": rid,
                "embedding": batch_out.embeddings[i],
                "prompt_tokens": (batch_out.prompt_tokens[i] if batch_out.prompt_tokens else 0),
                "finish_reason": (
                    batch_out.finished_reasons[i] if batch_out.finished_reasons else None
                ),
            }

            # Send result
            await state.out_queue.put(result)

            # Mark as finished
            state.finished = True
            state.time_stats.set_finished_time()
            state.event.set()

    async def _handle_health_check_output(self, health_out: HealthCheckOutput):
        """Handle health check output from scheduler."""
        rid = health_out.rid

        if rid not in self.rid_to_state:
            logger.warning(f"Health check output for unknown request: {rid}")
            return

        state = self.rid_to_state[rid]

        # Create health check result
        result = {
            "request_id": rid,
            "healthy": True,  # If we got a response, scheduler is healthy
            "output_text": (health_out.output_str if hasattr(health_out, "output_str") else ""),
            "finish_reason": (
                health_out.finish_reason if hasattr(health_out, "finish_reason") else "stop"
            ),
        }

        # Send result
        await state.out_queue.put(result)

        # Mark as finished
        state.finished = True
        state.time_stats.set_finished_time()
        state.event.set()

    async def _handle_abort_req(self, recv_obj: AbortReq):
        """Handle abort request from scheduler.

        The scheduler sends AbortReq back to notify us that a request was aborted,
        either due to explicit abort_request() call or scheduler-initiated abort
        (priority preemption, queue full, KV cache pressure, etc).
        """
        # Skip health check requests
        if recv_obj.rid.startswith("HEALTH_CHECK"):
            return

        # Check if request still exists
        if recv_obj.rid not in self.rid_to_state:
            logger.debug(
                f"Abort request for {recv_obj.rid} not in local state (may have already finished or not started yet)"
            )
            return

        state = self.rid_to_state[recv_obj.rid]

        # Mark as finished
        state.finished = True
        state.stream_finished = True

        # Create abort response
        if recv_obj.finished_reason:
            # Scheduler provided a specific finish reason (e.g., priority preemption, queue full)
            abort_response = {
                "request_id": recv_obj.rid,
                "error": recv_obj.finished_reason.get("message", "Request aborted"),
                "finished": True,
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": recv_obj.finished_reason,
                },
            }
        else:
            # Generic abort (e.g., explicit abort_request call)
            abort_response = {
                "request_id": recv_obj.rid,
                "error": "Request aborted",
                "finished": True,
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": "Abort before prefill",
                    },
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            }

        # Send abort notification to output queue
        await state.out_queue.put(abort_response)

        # Wake up any waiting coroutines
        state.event.set()

        logger.debug(f"Handled abort request for {recv_obj.rid}")

    async def _send_to_scheduler(self, obj):
        """Send an object to the scheduler via ZMQ."""
        try:
            self.send_to_scheduler.send_pyobj(obj)
        except Exception as e:
            logger.error(f"Failed to send to scheduler: {e}")
            raise

    def record_request_for_crash_dump(self, obj):
        """Record request for potential crash dump."""
        if len(self.crash_dump_request_list) < 100:
            self.crash_dump_request_list.append(
                {
                    "time": real_time(),
                    "request_id": getattr(obj, "rid", "unknown"),
                    "type": type(obj).__name__,
                }
            )

    async def shutdown(self):
        """Gracefully shutdown the request manager."""
        logger.info("Shutting down GrpcRequestManager")
        self.gracefully_exit = True

        # Cancel all asyncio tasks FIRST - this will interrupt blocked recv() calls
        for task in list(self.asyncio_tasks):
            if not task.done():
                task.cancel()

        # Give tasks a moment to process cancellation
        if self.asyncio_tasks:
            await asyncio.gather(*list(self.asyncio_tasks), return_exceptions=True)

        # Cancel all pending requests
        for rid, state in list(self.rid_to_state.items()):
            if not state.finished:
                await state.out_queue.put({"error": "Server shutting down", "shutdown": True})
                state.finished = True
                state.event.set()

        # Wait for tasks to complete
        if self.asyncio_tasks:
            await asyncio.gather(*list(self.asyncio_tasks), return_exceptions=True)

        # Shutdown bootstrap server if running
        if self.bootstrap_server:
            logger.info("Shutting down bootstrap server")
            try:
                if hasattr(self.bootstrap_server, "shutdown"):
                    if asyncio.iscoroutinefunction(self.bootstrap_server.shutdown):
                        await self.bootstrap_server.shutdown()
                    else:
                        self.bootstrap_server.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down bootstrap server: {e}")

        try:
            self.load_snapshot_reader.close()
        except Exception as e:
            logger.warning(f"Error closing load snapshot reader: {e}")

        # Close ZMQ sockets
        self.recv_from_scheduler.close()
        if self.recv_from_tokenizer is not None:
            self.recv_from_tokenizer.close()
        self.send_to_scheduler.close()

        # Terminate the ZMQ context - this is critical for asyncio loop to exit cleanly
        self.context.term()

        logger.info("GrpcRequestManager shutdown complete")

    def get_server_info(self) -> dict[str, Any]:
        """Get server information for health checks."""
        return {
            "active_requests": len(self.rid_to_state),
            "paused": self.is_pause,
            "last_receive_time": self.last_receive_tstamp,
        }

    async def get_loads(self, include: list[str], dp_rank: int | None = None) -> list[LoadSnapshot]:
        """
        Get load metrics from the schedulers' published LoadSnapshots.

        Args:
            include: List of metric sections to include (core, memory, spec, lora, disagg, queues, all)
            dp_rank: Optional DP rank filter (None for all ranks)

        Returns:
            List of LoadSnapshot objects, one per scheduler/DP rank
        """
        sections = set(include) if include else {"all"}
        invalid = sections - LoadSnapshot.VALID_SECTIONS
        if invalid:
            raise ValueError(
                f"Invalid include sections: {sorted(invalid)}. "
                f"Valid options: {sorted(LoadSnapshot.VALID_SECTIONS)}"
            )

        snapshots = self.load_snapshot_reader.read_all()
        if dp_rank is not None:
            snapshots = [s for s in snapshots if s.dp_rank == dp_rank]
        if "all" not in sections:
            for snapshot in snapshots:
                if "memory" not in sections:
                    snapshot.memory = None
                if "spec" not in sections:
                    snapshot.speculative = None
                if "lora" not in sections:
                    snapshot.lora = None
                if "disagg" not in sections:
                    snapshot.disaggregation = None
                if "queues" not in sections:
                    snapshot.queues = None
        return snapshots

    # Communicators that callers may address through send_communicator_req.
    _PUBLIC_COMMUNICATORS = frozenset(
        {
            "profile_communicator",
            "flush_cache_communicator",
            "get_internal_state_communicator",
        }
    )

    async def send_communicator_req(self, req, communicator_name: str, timeout: float = 30.0):
        """Send a request to the scheduler via a named communicator and return responses.

        This is the generic transport method for request/response patterns that
        go through the scheduler's send_to_tokenizer path (profile, flush_cache,
        get_internal_state, ...). Business logic (request construction, response
        interpretation) belongs in the caller — sglang's HTTP sidecar relies on
        this exact signature, so treat it as a public contract.

        Args:
            req: The request object to send to the scheduler.
            communicator_name: Attribute name of the communicator
                (e.g. "profile_communicator").
            timeout: Timeout in seconds for the scheduler round-trip.

        Returns:
            List of response objects from the scheduler(s), one per DP rank.

        Raises:
            ValueError: If communicator_name is not a known communicator.
            TimeoutError: If the scheduler does not respond within timeout.
        """
        if communicator_name not in self._PUBLIC_COMMUNICATORS:
            raise ValueError(
                f"Unknown communicator '{communicator_name}'. "
                f"Allowed: {sorted(self._PUBLIC_COMMUNICATORS)}"
            )
        self.auto_create_handle_loop()
        communicator = getattr(self, communicator_name)
        return await communicator(req, timeout=timeout)

    def auto_create_handle_loop(self):
        """Automatically create and start the handle_loop task, matching TokenizerManager pattern."""
        if self.no_create_loop:
            return

        self.no_create_loop = True
        loop = get_or_create_event_loop()
        self.asyncio_tasks.add(loop.create_task(print_exception_wrapper(self.handle_loop)))
        if self.recv_from_tokenizer is not None:
            self.asyncio_tasks.add(
                loop.create_task(print_exception_wrapper(self._handle_tokenizer_loop))
            )

        self.event_loop = loop

        # We only add signal handler when the tokenizer manager is in the main thread
        # due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = GrpcSignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(signal.SIGQUIT, signal_handler.running_phase_sigquit_handler)

        self.asyncio_tasks.add(loop.create_task(print_exception_wrapper(self.sigterm_watchdog)))

    async def sigterm_watchdog(self):
        """Watchdog to handle SIGTERM gracefully, matching TokenizerManager pattern."""
        while not self.gracefully_exit:
            await asyncio.sleep(1.0)

    def _req_stats_init(
        self,
        obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
        grpc_context: grpc.ServicerContext | None = None,
    ):
        calibrate_time_diff()
        # Create and register request state
        # TODO: support log_request
        # TODO: support request tracing
        time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)
        state = GrpcReqState(
            request_id=obj.rid,
            grpc_context=grpc_context,
            out_queue=asyncio.Queue(),
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            time_stats=time_stats,
        )

        # Track session if needed
        if hasattr(obj, "session_params") and obj.session_params:
            state.session_id = obj.session_params.session_id
            state.is_session_request = True

        self.rid_to_state[obj.rid] = state
        time_stats.set_created_time()


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"GrpcRequestManager hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(func.__self__, GrpcRequestManager):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)
