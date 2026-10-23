"""The planned warm-up of the TokenSpeed servicers: the shared plan
(:mod:`smg_grpc_servicer.warmup`) with the figures TokenSpeed's server
arguments carry, and the wire over the servicer's own gRPC port.

TokenSpeed captures its decode graphs at start and tunes its attention and
MoE kernels per shape as the shapes arrive; the first request at a token count
pays the selection. The plan visits the counts before the health flips, under
``SMG_TOKENSPEED_SERVICER_WARMUP``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from smg_grpc_servicer.tokenspeed.config_view import EngineConfigView
from smg_grpc_servicer.warmup import WarmupInputs, WarmupRequest

logger = logging.getLogger(__name__)

WARMUP_ENV = "SMG_TOKENSPEED_SERVICER_WARMUP"
#: The engine's own prefill budget when the arguments carry neither a chunk
#: size nor a prefill budget as a count.
DEFAULT_STEP_TOKENS = 8192
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
            "other ranks pay its first-time cost at their first request of that shape",
            ranks,
        )
    step = args.chunked_prefill_size or args.max_prefill_tokens or DEFAULT_STEP_TOKENS
    return WarmupInputs(
        max_num_seqs=args.running_window,
        max_num_batched_tokens=step,
        max_model_len=args.max_model_len or int(facts.get("max_context_length") or 0),
        vocab_size=int(facts.get("vocab_size") or 0),
        source="window and step from the server arguments (the engine's default step when unset)",
    )


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

    def flush_cache(self) -> bool:
        from smg_grpc_proto.generated import common_pb2

        response = self._stub.FlushCache(common_pb2.FlushCacheRequest(), timeout=60)
        return bool(getattr(response, "success", True))

    def close(self) -> None:
        if self._owns_channel:
            self._channel.close()


def warmup_wire(target: str, channel: Any = None) -> GrpcWire:
    """The wire to the servicer at ``target`` (a loopback ``host:port``), or
    over a ``channel`` the caller already holds and closes."""
    return GrpcWire(target, channel)
