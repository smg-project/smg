"""The planned warm-up of the SGLang servicers: the shared plan
(:mod:`smg_grpc_servicer.warmup`) with the figures SGLang's server arguments
carry, and the wire over the servicer's own gRPC port.

SGLang captures its decode graphs at start for its own batch-size list and
precompiles what its JIT registry knows; the kernels it selects or compiles
per token count (its fused MoE and GEMM paths on the newer GPUs) are still
reached lazily by the first request at each count. The plan visits them before
the health flips, under ``SMG_SGLANG_SERVICER_WARMUP``.

The plan keeps to the length classes (:attr:`WarmupInputs.every_prefill_length`
stays off): the first use of a prefill length between two classes has only
cost this engine a kernel search, never a hang, and at its default step of
16,384 tokens a pass over every length would take an hour.

A prefill or decode leg of a disaggregated pair plans as a single engine
does: its requests carry the engine's own stand-in bootstrap (what the
engine's warm-up sets in disaggregation mode: the fake-transfer host and room
0), a transfer to nobody on either side, so the leg warms up without its peer
and no first request of a prompt-length class pays a kernel compile under the
Router's prefill-liveness bound.
The language side of an encode-disaggregated deployment (``--language-only``)
plans as a single engine: a warm-up request carries no media, so it reaches
the scheduler untouched and the encoders see nothing of it, and the first
forward of a fresh scheduler process is paid here rather than by the first
request (the engine's own frontend warms that role on the text path for the
same reason). The encoder-only role takes no plan: the engine's entrypoint
runs it on its own encoder server, never on a servicer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from smg_grpc_servicer.sglang.config_view import EngineConfigView
from smg_grpc_servicer.warmup import WarmupInputs, WarmupRequest

logger = logging.getLogger(__name__)

WARMUP_ENV = "SMG_SGLANG_SERVICER_WARMUP"
#: The engine's own prefill budget when the arguments carry neither a chunk
#: size nor a prefill budget as a count.
DEFAULT_STEP_TOKENS = 16384
#: The positions the engine's scheduler keeps back beyond a request's prompt
#: and its output budget: it admits a prompt below its ``max_req_input_len``
#: (the context less one, less five more) and answers one at or above it
#: with an error, so a prompt and a one-token output fill at most the
#: context less six; the plan's longest prompt stops there.
CONTEXT_RESERVE_TOKENS = 6
# The engine's own gRPC server sends messages up to this size.
_MAX_MESSAGE_BYTES = 1024 * 1024 * 256


def warmup_inputs(server_args: Any, facts: Mapping[str, Any]) -> WarmupInputs | None:
    """What the planner needs, off the server arguments and the advertised
    model facts (``is_generation``, ``max_context_length``, ``vocab_size``):
    the running window (``--max-running-requests``; unknown when the scheduler
    sizes it itself), the step budget (the chunk size under chunked prefill,
    else the prefill budget, else the engine's default), the context, and on
    a prefill or decode leg the engine's stand-in bootstrap host for the
    requests to carry. ``None`` for a model that generates nothing and for the
    encoder-only role, which the engine runs on its own server; the language
    side of an encode-disaggregated deployment plans as a single engine (its
    warm-up requests carry no media). The plan is one rank's: the
    servicer routes this engine's requests to the least-loaded scheduler rank
    and reads no rank off a request, so under data parallelism each prefill
    shape reaches one rank (a WARNING line says so)."""
    if not facts.get("is_generation", True):
        return None
    args = EngineConfigView(server_args).server_args
    if args.encoder_only:
        return None
    lone_leg_bootstrap = None
    if args.disaggregation_mode != "null":
        # The engine's own warm-up convention for a disaggregation role: a
        # request bound for this host is a transfer to nobody on either leg.
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        lone_leg_bootstrap = FAKE_BOOTSTRAP_HOST
    if args.scheduler_count > 1:
        logger.warning(
            "The engine runs %d scheduler ranks and the servicer routes a warm-up request to the "
            "least-loaded one: each prefill shape of the planned warm-up reaches one rank; the "
            "other ranks pay its first-time cost at their first request of that shape",
            args.scheduler_count,
        )
    step = args.chunked_prefill_size or args.max_prefill_tokens or DEFAULT_STEP_TOKENS
    return WarmupInputs(
        max_num_seqs=args.max_running_requests or 0,
        max_num_batched_tokens=step,
        max_model_len=args.context_length or int(facts.get("max_context_length") or 0),
        vocab_size=int(facts.get("vocab_size") or 0),
        context_reserve_tokens=CONTEXT_RESERVE_TOKENS,
        source="window and step from the server arguments (the engine's default step when unset)",
        lone_leg_bootstrap=lone_leg_bootstrap,
    )


class GrpcWire:
    """The warm-up's requests over the servicer's own port, as a client sends
    them: tokenized input, greedy sampling with the stop ignored, so a request
    produces exactly the output it asked for."""

    def __init__(self, target: str, channel: Any = None) -> None:
        import grpc
        from smg_grpc_proto import sglang_scheduler_pb2_grpc

        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
            ],
        )
        self._stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(self._channel)

    def generate(self, request: WarmupRequest, timeout: float) -> None:
        from smg_grpc_proto import sglang_scheduler_pb2

        message = sglang_scheduler_pb2.GenerateRequest(
            request_id=request.request_id,
            tokenized=sglang_scheduler_pb2.TokenizedInput(
                input_ids=list(request.token_ids), original_text="warm-up"
            ),
            # The proto's scalar defaults are not valid sampling values for
            # the engine; the greedy settings are spelled out.
            sampling_params=sglang_scheduler_pb2.SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                repetition_penalty=1.0,
                n=1,
                max_new_tokens=request.max_tokens,
                ignore_eos=True,
            ),
            stream=False,
        )
        if request.lone_leg_bootstrap is not None:
            # As the engine's own warm-up sends it in disaggregation mode:
            # the stand-in host and room 0.
            message.disaggregated_params.CopyFrom(
                sglang_scheduler_pb2.DisaggregatedParams(
                    bootstrap_host=request.lone_leg_bootstrap, bootstrap_room=0
                )
            )
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
