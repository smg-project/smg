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
plans as a single engine: a text warm-up request carries no media, so it
reaches the scheduler untouched and the encoders see nothing of it, and the
first forward of a fresh scheduler process is paid here rather than by the
first request (the engine's own frontend warms that role on the text path for
the same reason). The text ladder does not reach the scheduler's media path,
whose own first use (the first forward with precomputed embeddings) costs a
fresh worker seconds once; so where that worker's frontend dispatches to the
encoders, the plan ends its shapes with one image through them
(:func:`media_warmup`: the prompt the model's own chat template renders around
one image item, its token ids, and the engine's own minimal warm-up image as a
data URL), the production path once, before SERVING. A template that renders
no image placeholder leaves the leg out, and a leg the encoders cannot serve
yet is a logged step, never the end of the start. The encoder-only role takes
no plan: the engine's entrypoint runs it on its own encoder server, never on a
servicer.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Any

from smg_grpc_servicer.sglang.config_view import EngineConfigView
from smg_grpc_servicer.warmup import WarmupInputs, WarmupRequest

logger = logging.getLogger(__name__)

WARMUP_ENV = "SMG_SGLANG_SERVICER_WARMUP"
#: The image of the media leg: the engine's own minimal warm-up image (a
#: 32x32 black PNG, what its frontend sends its vision models at start),
#: carried as a data URL so the encoders fetch nothing.
WARMUP_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRY"
    "he3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpke"
    "I77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)
#: The text beside the image in the media leg's prompt.
WARMUP_IMAGE_QUESTION = "Describe the image."
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


@dataclasses.dataclass(frozen=True)
class MediaWarmup:
    """What the media leg's request carries: the prompt the model's own chat
    template renders around one image item, its token ids, and the image as a
    data URL."""

    prompt: str
    input_ids: tuple[int, ...]
    image_url: str


def _render(tokenizer: Any, with_image: bool) -> str:
    content = ([{"type": "image"}] if with_image else []) + [
        {"type": "text", "text": WARMUP_IMAGE_QUESTION}
    ]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def media_warmup(
    tokenizer_dir: str | None,
    *,
    trust_remote_code: bool = False,
    load_tokenizer: Callable[[str], Any] | None = None,
    log: logging.Logger = logger,
) -> MediaWarmup | None:
    """The media leg's request material, or ``None`` with the reason logged:
    no tokenizer directory to render the prompt from, a tokenizer that does
    not load, or a chat template that renders no image placeholder (the leg
    would be a text request). The prompt is what the template renders around
    one image item and :data:`WARMUP_IMAGE_QUESTION`, unexpanded, as a
    client sends it to a language-side worker; the ids are its tokens."""
    if not tokenizer_dir:
        log.info("Planned warm-up: no image leg (no tokenizer directory to render the prompt from)")
        return None
    try:
        if load_tokenizer is None:
            from transformers import AutoTokenizer

            def load_tokenizer(directory: str) -> Any:
                return AutoTokenizer.from_pretrained(directory, trust_remote_code=trust_remote_code)

        tokenizer = load_tokenizer(tokenizer_dir)
        with_image = _render(tokenizer, with_image=True)
        without_image = _render(tokenizer, with_image=False)
        if with_image == without_image:
            log.info(
                "Planned warm-up: no image leg (the chat template renders no image placeholder)"
            )
            return None
        input_ids = tuple(
            int(t) for t in tokenizer(with_image, add_special_tokens=False)["input_ids"]
        )
    except Exception as error:  # noqa: BLE001 -- the leg is an optimisation; the reason is logged
        log.info("Planned warm-up: no image leg (the prompt could not be rendered: %s)", error)
        return None
    return MediaWarmup(
        prompt=with_image,
        input_ids=input_ids,
        image_url=f"data:image/png;base64,{WARMUP_IMAGE_PNG_BASE64}",
    )


class GrpcWire:
    """The warm-up's requests over the servicer's own port, as a client sends
    them: tokenized input, greedy sampling with the stop ignored, so a request
    produces exactly the output it asked for; the media leg's request with
    its image reference and the prompt rendered around it (``media``)."""

    def __init__(self, target: str, channel: Any = None, media: MediaWarmup | None = None) -> None:
        import grpc
        from smg_grpc_proto import sglang_scheduler_pb2_grpc

        self._media = media
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

        if request.media:
            if self._media is None:
                raise RuntimeError("the plan's media leg has nothing to send on this wire")
            tokenized = sglang_scheduler_pb2.TokenizedInput(
                input_ids=list(self._media.input_ids), original_text=self._media.prompt
            )
        else:
            tokenized = sglang_scheduler_pb2.TokenizedInput(
                input_ids=list(request.token_ids), original_text="warm-up"
            )
        message = sglang_scheduler_pb2.GenerateRequest(
            request_id=request.request_id,
            tokenized=tokenized,
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
        if request.media:
            # As a client names an image to a language-side worker: the
            # reference alone, the prompt unexpanded; the encoders do the rest.
            message.mm_inputs.CopyFrom(
                sglang_scheduler_pb2.MultimodalInputs(
                    image_urls=[self._media.image_url], modalities=["image"]
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


def warmup_wire(target: str, channel: Any = None, media: MediaWarmup | None = None) -> GrpcWire:
    """The wire to the servicer at ``target`` (a loopback ``host:port``), or
    over a ``channel`` the caller already holds and closes; ``media`` is what
    the plan's media leg sends, when the plan has one."""
    return GrpcWire(target, channel, media)
