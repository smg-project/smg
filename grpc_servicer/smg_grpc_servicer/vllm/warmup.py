"""The planned warm-up of the vLLM servicers: the shared plan
(:mod:`smg_grpc_servicer.warmup`) with the figures vLLM's config carries,
and the two wires it runs over.

vLLM warms a fixed set of shapes at start (its graph-capture sizes, its JIT
registry, one kernel-tuning pass at the admission step's full token count,
whose tuned configurations cover the tuner's bucket grid and nothing between)
and then watches for compiles it did not plan. A step at a token count off
that grid, or at a running-batch size the capture list skipped, still pays a
kernel search or a compile the first time, and the requests in flight pay it
with it. The plan visits those counts before the health flips, under
``SMG_VLLM_SERVICER_WARMUP``:

- the Rust path's launcher drives the pass over the servicer's own port
  (:class:`GrpcWire`) between the engine's handshake and SERVING
  (:mod:`smg_grpc_servicer.rust_lifecycle`);
- the Python servicer drives it over its in-process engine
  (:class:`EngineWire`, :func:`start_engine_warmup`) while its health service
  answers NOT_SERVING (:func:`warmup_in_progress`).

Under data parallelism the servicer fronts one engine per rank, each with its
own scheduler, and routes a request that names a ``data_parallel_rank`` to
that rank (the Rust server over its wire, the in-process engine client by the
same argument): the plan carries the rank count and pins every request, so
every rank runs every shape, and the log's count is per rank.

The plan keeps to the length classes (:attr:`WarmupInputs.every_prefill_length`
stays off): the first use of a prefill length between two classes has only
cost this engine a kernel search, never a hang, and at its default step of
16,384 tokens a pass over every length would take an hour.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from smg_grpc_servicer.vllm.config_view import EngineConfigView
from smg_grpc_servicer.warmup import (
    WarmupInputs,
    WarmupPlan,
    WarmupReport,
    WarmupRequest,
    parse_setting,
    plan_for,
    run_warmup_async,
)

logger = logging.getLogger(__name__)

WARMUP_ENV = "SMG_VLLM_SERVICER_WARMUP"
_MAX_MESSAGE_BYTES = 1024 * 1024 * 256


def warmup_inputs_from_config(vllm_config: Any) -> WarmupInputs | None:
    """What the planner needs, off the engine's config: the running window
    and the admission step as the scheduler config resolved them, the
    context and vocabulary off the model config, the graph-capture sizes off
    the compilation config, the data-parallel rank count off the parallel
    config (the window and the step are each rank's own). ``None`` for a
    model that generates nothing (a pooling runner)."""
    view = EngineConfigView(vllm_config)
    model = view.model
    if model.runner_type != "generate":
        return None
    return WarmupInputs(
        max_num_seqs=view.max_num_seqs,
        max_num_batched_tokens=view.max_num_batched_tokens,
        max_model_len=model.max_model_len,
        vocab_size=model.vocab_size,
        capture_sizes=view.cudagraph_capture_sizes,
        source="window, step and capture sizes from the engine's resolved configuration",
        data_parallel_size=view.parallel.data_parallel_size or 1,
    )


class GrpcWire:
    """The warm-up's requests over the servicer's own port, as a client sends
    them: tokenized input, greedy sampling with the stop ignored, so a request
    produces exactly the output it asked for."""

    def __init__(self, target: str, channel: Any = None) -> None:
        import grpc
        from smg_grpc_proto import vllm_engine_pb2_grpc

        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
            ],
        )
        self._stub = vllm_engine_pb2_grpc.VllmEngineStub(self._channel)

    def generate(self, request: WarmupRequest, timeout: float) -> None:
        from smg_grpc_proto import vllm_engine_pb2

        message = vllm_engine_pb2.GenerateRequest(
            request_id=request.request_id,
            tokenized=vllm_engine_pb2.TokenizedInput(
                input_ids=list(request.token_ids), original_text="warm-up"
            ),
            sampling_params=vllm_engine_pb2.SamplingParams(
                temperature=0.0,
                max_tokens=request.max_tokens,
                ignore_eos=True,
                skip_special_tokens=True,
            ),
            stream=False,
        )
        if request.data_parallel_rank is not None:
            message.data_parallel_rank = request.data_parallel_rank
        last = None
        for frame in self._stub.Generate(message, timeout=timeout):
            last = frame
        if last is None or not last.HasField("complete"):
            raise RuntimeError("the request ended without a complete frame")

    def flush_cache(self) -> bool:
        from smg_grpc_proto.generated import common_pb2

        response = self._stub.FlushCache(common_pb2.FlushCacheRequest(), timeout=60)
        success = bool(getattr(response, "success", True))
        if not success and (message := str(getattr(response, "message", "") or "")):
            # The servicer answered the refusal with its reason: raised, the
            # warm-up's end line carries the words as it does for a raised status.
            raise RuntimeError(message)
        return success

    def close(self) -> None:
        if self._owns_channel:
            self._channel.close()


def warmup_wire(target: str, channel: Any = None) -> GrpcWire:
    """The wire to the servicer at ``target`` (a loopback ``host:port``), or
    over a ``channel`` the caller already holds and closes."""
    return GrpcWire(target, channel)


class EngineWire:
    """The warm-up's requests over the in-process engine client, on its
    event loop: the same tokenized greedy requests, consumed to their end."""

    def __init__(self, async_llm: Any) -> None:
        self._engine = async_llm

    async def generate(self, request: WarmupRequest, timeout: float) -> None:
        from vllm import SamplingParams, TokensPrompt

        params = SamplingParams(
            max_tokens=request.max_tokens, ignore_eos=True, temperature=0.0, detokenize=False
        )
        # A pinned request names its rank the way the servicer's own requests
        # do; an unpinned one leaves the engine client's routing alone.
        pinned = (
            {}
            if request.data_parallel_rank is None
            else {"data_parallel_rank": request.data_parallel_rank}
        )

        async def consume() -> None:
            async for _ in self._engine.generate(
                TokensPrompt(prompt_token_ids=list(request.token_ids)),
                params,
                request.request_id,
                **pinned,
            ):
                pass

        await asyncio.wait_for(consume(), timeout)

    async def flush_cache(self) -> bool:
        from smg_grpc_servicer.vllm.admin import _reset_prefix_cache

        return bool(await _reset_prefix_cache(self._engine))

    async def close(self) -> None:
        return None


# The Python servicer's pass in flight, one per process: the health service
# reads it (``warmup_in_progress``) for as long as it runs.
_running: dict[str, asyncio.Task | None] = {"task": None}


def warmup_in_progress() -> bool:
    """Whether the Python servicer's planned warm-up is still running."""
    task = _running["task"]
    return task is not None and not task.done()


def start_engine_warmup(
    async_llm: Any,
    *,
    setting_text: str | None,
    log: logging.Logger = logger,
) -> asyncio.Task | None:
    """Start the planned warm-up over the in-process engine on the running
    event loop; returns its task, or ``None`` with the reason logged (the
    knob is off or unusable, the model takes none, no loop runs). The task
    holds the health service at NOT_SERVING while it runs
    (:func:`warmup_in_progress`); its outcome is one log line either way."""
    try:
        setting = parse_setting(setting_text, name=WARMUP_ENV)
    except ValueError as error:
        log.warning("%s; the planned warm-up is skipped", error)
        return None
    try:
        inputs = warmup_inputs_from_config(async_llm.vllm_config)
    except Exception:  # noqa: BLE001 -- an engine client without the config warms nothing
        log.warning("The engine's configuration could not be read; the planned warm-up is skipped")
        return None
    plan = plan_for(setting, inputs, knob=WARMUP_ENV, log=log)
    if plan is None:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.warning("No event loop runs here; the planned warm-up is skipped")
        return None
    task = loop.create_task(_run(plan, EngineWire(async_llm), setting.budget_secs, log))
    _running["task"] = task
    return task


async def _run(
    plan: WarmupPlan, wire: EngineWire, budget_secs: float, log: logging.Logger
) -> WarmupReport | None:
    try:
        return await run_warmup_async(plan, wire, budget_secs=budget_secs, log=log)
    except Exception:  # noqa: BLE001 -- the pass is an optimisation; the servicer serves without it
        log.exception("The planned warm-up failed; the servicer serves without it")
        return None
