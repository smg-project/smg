"""The planned warm-up of the TokenSpeed servicers: the shared plan
(:mod:`smg_grpc_servicer.warmup`) with the figures TokenSpeed's server
arguments carry, and the wire over the servicer's own gRPC port.

TokenSpeed captures its decode graphs at start and tunes its attention and
MoE kernels per shape as the shapes arrive; the first request at a token count
pays the selection. The plan visits the counts before the health flips, under
``SMG_TOKENSPEED_SERVICER_WARMUP``.

The plan here also visits every prompt length up to the chunk size, not only
the length classes (:attr:`WarmupInputs.every_prefill_length`). A prefill
chunk reaches two cuBLAS GEMMs in this engine (the compressor's fp32 router
projection falls back to cuBLAS above 32 tokens; the grouped output
projection is a batched GEMM), whose kernels the library picks per token
count and loads on first use. A chunked prefill's last chunk is any count
below the chunk size, so a long turn's second chunk is a count the engine has
not seen; under the overlap schedule its forward is launched while the first
chunk's TP collective is still in flight, and the lazy kernel load inside
that forward has deadlocked every rank (the forward threads inside the
driver's module load, the collective waiting for them). Every count used
once before SERVING, with nothing else in flight, leaves no first use to
serving. The remainders run last, in an order whose every prefix is a
uniform spread of the range, so the budget cuts evenly if it cuts at all.
They run at the ``full`` level alone (the shipped default): a start at
``fast`` takes the length classes and leaves the hazard open, which the
start log says. They are asked for when the engine runs more than one
process (its parallel layout, off the server arguments): the hang needs a
collective waiting on a peer, and a one-process engine pays a count's first
use as a pause, so it plans the length classes alone, as the other engines
do, and an 8k window is a few hundred steps rather than eight thousand. An
unknown layout keeps them.

The plan's longest prompt leaves the engine's scheduler what it keeps back
(:func:`context_reserve_tokens`): its request handler clamps a request's
output budget to ``(max_model_len - 1) - prompt - 1`` and its scheduler
refuses a negative clamp, so a prompt one below the context, which the
shared plan would otherwise end on, is a request this engine refuses
(and, today, exits on).

Under attention data parallelism the servicer routes by load and the wire
pins no rank, so a remainder, like every prefill shape, reaches one rank: the
hazard stays open on the other ranks there until the wire can pin a rank (the
warning at planning says so).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from smg_grpc_servicer.tokenspeed.config_view import EngineConfigView
from smg_grpc_servicer.warmup import WarmupInputs, WarmupRequest, unserved

logger = logging.getLogger(__name__)

WARMUP_ENV = "SMG_TOKENSPEED_SERVICER_WARMUP"
#: The engine's own prefill budget when the arguments carry neither a chunk
#: size nor a prefill budget as a count.
DEFAULT_STEP_TOKENS = 8192
#: This engine's default budget for the pass, in place of the shared
#: :data:`smg_grpc_servicer.warmup.DEFAULT_BUDGET_SECS` (300 s, which the
#: other engines keep): room for the chunk remainders at a 4,096-token chunk,
#: 4,042 one-token requests at about 0.17 s each on a four-GPU node (692 s
#: measured for the whole pass). At the engine's default step of 8,192
#: tokens (no chunk size set) the remainders are about 8,140, about 1,380 s,
#: and the budget cuts the pass about halfway, evenly (the spread order): a
#: chunk size, or a larger budget on the knob, covers every count.
DEFAULT_BUDGET_SECS = 900.0
#: The positions the engine's scheduler keeps back beyond a request's prompt
#: and its output budget: the one its request handler takes off the context
#: for the request length, and the one it takes off the output clamp.
SCHEDULER_RESERVE_TOKENS = 2
_MAX_MESSAGE_BYTES = 1024 * 1024 * 256


def warmup_inputs(server_args: Any, facts: Mapping[str, Any]) -> WarmupInputs | None:
    """What the planner needs, off the server arguments and the advertised
    model facts (``max_context_length``, ``vocab_size``): the running window,
    the step budget (the chunk size under chunked prefill, else the prefill
    budget, else the engine's default), the context. ``None`` for a
    disaggregation role (encode, prefill, decode), whose requests need
    bootstrap fields a warm-up does not carry. The plan is one rank's: the
    servicer routes this engine's requests to the least-loaded rank and reads
    no rank off a request, so under attention data parallelism each prefill
    shape reaches one rank (a WARNING line says so)."""
    args = EngineConfigView(server_args).server_args
    if args.disaggregation_mode != "null":
        return None
    ranks = args.attention_dp_size or 1
    if ranks > 1:
        logger.warning(
            "The engine runs %d scheduler ranks and the servicer routes a warm-up request to the "
            "least-loaded one: each prefill shape of the planned warm-up reaches one rank; the "
            "other ranks pay its first-time cost at their first request of that shape, and the "
            "chunk remainders, when the plan runs them (the full level), reach one rank each "
            "too, so the hazard they answer (a count's first use during serving) stays open on "
            "the other ranks until the wire can pin a rank",
            ranks,
        )
    step = args.chunked_prefill_size or args.max_prefill_tokens or DEFAULT_STEP_TOKENS
    return WarmupInputs(
        max_num_seqs=args.running_window,
        max_num_batched_tokens=step,
        max_model_len=args.max_model_len or int(facts.get("max_context_length") or 0),
        vocab_size=int(facts.get("vocab_size") or 0),
        source="window and step from the server arguments (the engine's default step when unset)",
        context_reserve_tokens=context_reserve_tokens(args),
        # Every count a chunk can hold, once, before serving (the module's
        # doc), where a count's first use can meet a collective in flight:
        # an engine of more than one process. An unknown layout keeps them.
        every_prefill_length=args.world_size != 1,
    )


def context_reserve_tokens(args: Any) -> int:
    """The positions of the context this engine keeps back beyond a request's
    prompt and its output, as the engine computes them off its server
    arguments: the scheduler's two (:data:`SCHEDULER_RESERVE_TOKENS`), or the
    frontend's input reserve when it is wider: under speculative decoding the
    engine keeps the draft window (``speculative_num_draft_tokens``) behind
    the prompt, one position otherwise, which the one-token output of a
    warm-up request already covers."""
    draft = args.speculative_num_draft_tokens if args.speculative_algorithm else None
    return max(SCHEDULER_RESERVE_TOKENS, (draft or 1) - 1)


class GrpcWire:
    """The warm-up's requests over the servicer's own port, as a client sends
    them: tokenized input, greedy sampling with the stop ignored, so a request
    produces exactly the output it asked for."""

    def __init__(self, target: str, channel: Any = None) -> None:
        import grpc
        from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc

        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
            ],
        )
        self._stub = tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(self._channel)

    def generate(self, request: WarmupRequest, timeout: float) -> None:
        from smg_grpc_proto import tokenspeed_scheduler_pb2

        message = tokenspeed_scheduler_pb2.GenerateRequest(
            request_id=request.request_id,
            tokenized=tokenspeed_scheduler_pb2.TokenizedInput(
                input_ids=list(request.token_ids), original_text="warm-up"
            ),
            sampling_params=tokenspeed_scheduler_pb2.SamplingParams(
                temperature=0.0,
                max_new_tokens=request.max_tokens,
                ignore_eos=True,
            ),
            stream=False,
        )
        last = None
        for frame in self._stub.Generate(message, timeout=timeout):
            last = frame
        if last is None or not last.HasField("complete"):
            raise RuntimeError("the request ended without a complete frame")
        if error := unserved(last.complete.finish_reason, len(last.complete.output_ids)):
            # The engine ended the request without serving it (an abort at
            # admission arrives as a completion with no token): a failed
            # request, not a covered step.
            raise error

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
