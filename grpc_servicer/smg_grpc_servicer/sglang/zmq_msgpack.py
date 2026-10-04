"""The SGLang scheduler behind SMG over ZMQ, speaking msgpack.

SGLang's scheduler only ever talks to a Python tokenizer manager that binds
its sockets, picks their addresses, and decodes Python-only payload
extensions. A frontend in another process cannot join that layout, so this
module adds the one thing it lacks, discovery, and keeps everything else:

* SMG BINDS a handshake socket (ROUTER), an input socket (ROUTER) and an
  output socket (PULL); the scheduler rank that owns request I/O DIALS the
  handshake with a two-byte engine identity, learns the data-plane addresses
  from INIT, registers with its ready response and connects in;
* requests arrive as ``[type_byte, payload, *aux]`` frames whose payload is
  the scheduler's own positional ``TokenizedGenerateReqInput`` (``ADD``) or a
  plain list of request ids (``ABORT``);
* outputs leave as the positional :class:`BatchTokenIDSlimOutput`: the token
  columns a frontend that detokenizes itself needs, plus a scheduler-load
  tail, so the output batch is the one in-band load channel.

The handshake structs are msgpack maps with named keys; the data-plane
structs are positional arrays. SMG's codec relies on exactly that split. The
adapters here are installed into a running scheduler by
:mod:`smg_grpc_servicer.sglang.zmq_plugin` (an SGLang plugin), and the
scheduler ranks are started by :mod:`smg_grpc_servicer.sglang.headless`.
"""

from __future__ import annotations

import logging
import struct
import time
from array import array
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import (
        AbortReq,
        BatchTokenIDOutput,
        TokenizedGenerateReqInput,
    )
    from sglang.srt.managers.load_snapshot import LoadSnapshot
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Single-byte request-type frame ahead of the payload: the receiver dispatches
# without decoding first.
REQ_TYPE_ADD = b"\x00"
REQ_TYPE_ABORT = b"\x01"

# Raw single-frame sentinel SMG's output loop treats as terminal, so a
# scheduler that exits marks its worker dead instead of healthy-idle.
ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

# SMG allows a registering worker a long connect window, so it may start the
# handshake well after this scheduler finished loading: wait for INIT in
# slices and log progress rather than giving up early.
INIT_TIMEOUT_MS = 600_000
# How stale the piggybacked scheduler load may be (seconds).
LOAD_TAIL_REFRESH_S = 0.05
INIT_POLL_SLICE_MS = 30_000

# The engine identity is a little-endian u16.
MAX_ENGINE_INDEX = 0xFFFF


# ----------------------------------------------------------------------------
# Startup handshake: msgpack maps with named keys. Not array_like.
# ----------------------------------------------------------------------------


class WireReadyMessage(msgspec.Struct):
    """Scheduler -> SMG handshake status (``status`` = HELLO | READY)."""

    status: str | None = None
    local: bool | None = None
    headless: bool | None = None
    parallel_config_hash: str | None = None


class WireHandshakeAddresses(msgspec.Struct):
    """SMG-owned data-plane addresses delivered in INIT."""

    inputs: list[str] = []
    outputs: list[str] = []
    coordinator_input: str | None = None
    coordinator_output: str | None = None
    frontend_stats_publish_address: str | None = None


class WireHandshakeInitMessage(msgspec.Struct):
    """SMG -> scheduler INIT payload (the reply to HELLO)."""

    addresses: WireHandshakeAddresses
    # Opaque to the scheduler; decoded loosely so unknown keys are ignored.
    parallel_config: dict = {}


class WireEngineCoreReadyResponse(msgspec.Struct):
    """Scheduler -> SMG post-init facts, sent as the registration message on
    the input socket once HELLO/INIT/READY completes. Field names, not order,
    are the contract."""

    max_model_len: int = 0
    num_gpu_blocks: int = 0
    block_size: int = 0
    dp_stats_address: str | None = None
    dtype: str = "bfloat16"
    multimodal_encoder_dtype: str | None = None
    vllm_version: str = ""
    world_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    data_parallel_rank: int = 0
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    instance_id: str = ""
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None
    kv_events_config: dict | None = None


# ----------------------------------------------------------------------------
# Data plane: the slim output (positional, tag = class name at element 0).
# ----------------------------------------------------------------------------


class BatchTokenIDSlimOutput(msgspec.Struct, tag=True, array_like=True):
    """Per-step token output for SMG: the token columns of the scheduler's
    ``BatchTokenIDOutput``, each finish reason reduced to its type, message
    and matched stop, and a scheduler-load tail. SMG decodes a positional
    prefix: append new fields at the end only."""

    rids: list[str]
    output_ids: list[list[int]]
    # "" while generating, else "stop" | "length" | "abort".
    finished_reasons: list[str]
    finished_messages: list[str | None]
    # The matched stop token id or stop string when the finish was a stop.
    finished_matched: list[int | str | None]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]
    output_token_logprobs_val: list[list[float]]
    output_token_logprobs_idx: list[list[int]]
    # Producing rank, so SMG attributes outputs and load under DP.
    engine_index: int = 0
    num_running: int = 0
    num_waiting: int = 0
    kv_used_tokens: int = 0
    kv_total_tokens: int = 0
    # HTTP status an abort carries (SGLang stamps 400 on requests it refuses;
    # this wire's own rejections do too); None for other finishes.
    finished_status: list[int | None] | None = None
    # Ranked candidates per newly decoded token (`top_logprobs_num` of them),
    # parallel to the sampled-token logprob columns; empty when not requested.
    output_top_logprobs_val: list[list[list[float]]] | None = None
    output_top_logprobs_idx: list[list[list[int]]] | None = None

    @classmethod
    def from_full(
        cls,
        out: BatchTokenIDOutput,
        *,
        engine_index: int = 0,
        num_running: int = 0,
        num_waiting: int = 0,
        kv_used_tokens: int = 0,
        kv_total_tokens: int = 0,
    ) -> BatchTokenIDSlimOutput:
        if out.output_ids is None:
            raise ValueError("BatchTokenIDOutput.output_ids is required on the slim wire")
        count = len(out.rids or [])

        def column(values, default):
            values = list(values or [])
            values.extend(default for _ in range(count - len(values)))
            return values

        reasons = column(out.finished_reasons, None)

        def matched(reason):
            # A multi-token stop arrives as a list; SMG only names single stops.
            value = reason.get("matched") if reason else None
            return value if isinstance(value, (int, str)) else None

        def status(reason):
            value = reason.get("status_code") if reason else None
            return int(value) if isinstance(value, int) else None

        return cls(
            rids=list(out.rids or []),
            output_ids=[list(ids) for ids in out.output_ids],
            finished_reasons=[str((r or {}).get("type") or "") for r in reasons],
            finished_messages=[(r or {}).get("message") for r in reasons],
            finished_matched=[matched(r) for r in reasons],
            prompt_tokens=column(out.prompt_tokens, 0),
            completion_tokens=column(out.completion_tokens, 0),
            cached_tokens=column(out.cached_tokens, 0),
            output_token_logprobs_val=[
                list(v or []) for v in column(out.output_token_logprobs_val, None)
            ],
            output_token_logprobs_idx=[
                list(v or []) for v in column(out.output_token_logprobs_idx, None)
            ],
            engine_index=engine_index,
            num_running=num_running,
            num_waiting=num_waiting,
            kv_used_tokens=kv_used_tokens,
            kv_total_tokens=kv_total_tokens,
            finished_status=[status(r) for r in reasons],
            output_top_logprobs_val=[
                [list(step) for step in (v or [])]
                for v in column(out.output_top_logprobs_val, None)
            ],
            output_top_logprobs_idx=[
                [list(step) for step in (v or [])]
                for v in column(out.output_top_logprobs_idx, None)
            ],
        )


_WIRE_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "half": "float16",
    "fp16": "float16",
    "float32": "float32",
    "float": "float32",
    "fp32": "float32",
}


def wire_dtype(dtype: Any) -> str:
    """Map a torch dtype (or its string form) onto SMG's dtype enum. Raises
    for anything unmapped: misreporting the dtype is worse than not starting."""
    key = str(dtype).lower().replace("torch.", "")
    mapped = _WIRE_DTYPE_MAP.get(key)
    if mapped is None:
        raise ValueError(f"dtype {dtype!r} has no wire mapping; extend _WIRE_DTYPE_MAP")
    return mapped


_ENC = msgspec.msgpack.Encoder()
_DEC_ABORT = msgspec.msgpack.Decoder(list[str])
_DEC_INIT = msgspec.msgpack.Decoder(WireHandshakeInitMessage)
_DEC_ADD: msgspec.msgpack.Decoder | None = None


def _request_decoder() -> msgspec.msgpack.Decoder:
    """The scheduler's request struct, decoded with its own extension hooks
    plus one widening: SMG sends token ids as a plain integer array where the
    tokenizer manager would send an ``array('q')`` extension."""
    global _DEC_ADD
    if _DEC_ADD is None:
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
        from sglang.srt.utils.msgpack_utils import dec_hook, ext_hook

        def request_dec_hook(tp: type, obj: object) -> object:
            if tp is array and isinstance(obj, list):
                return array("q", obj)
            return dec_hook(tp, obj)

        _DEC_ADD = msgspec.msgpack.Decoder(
            TokenizedGenerateReqInput, dec_hook=request_dec_hook, ext_hook=ext_hook
        )
    return _DEC_ADD


def encode(obj: msgspec.Struct) -> bytes:
    return _ENC.encode(obj)


def decode_init(payload: bytes) -> WireHandshakeInitMessage:
    return _DEC_INIT.decode(payload)


def decode_request_frames(
    frames: list[bytes],
) -> list[TokenizedGenerateReqInput | AbortReq]:
    """Decode one ``[type_byte, payload, *aux]`` message into io_structs. An
    ADD yields exactly one request; an ABORT may carry several request ids
    and yields one ``AbortReq`` each."""
    from sglang.srt.managers.io_struct import AbortReq

    if len(frames) < 2:
        raise ValueError(f"request needs >= 2 frames (type, payload), got {len(frames)}")
    type_byte = bytes(frames[0])
    if type_byte == REQ_TYPE_ADD:
        return [_request_decoder().decode(frames[1])]
    if type_byte == REQ_TYPE_ABORT:
        return [AbortReq(rid=rid) for rid in _DEC_ABORT.decode(frames[1])]
    raise ValueError(f"unknown request type byte {type_byte!r}")


def _malformed_request_id(frames: list[bytes]) -> str | None:
    """The rid of an ADD whose typed decode failed: element 1 of the positional
    array, read loosely. ``None`` when even that cannot be read."""
    if len(frames) < 2 or bytes(frames[0]) != REQ_TYPE_ADD:
        return None
    try:
        payload = msgspec.msgpack.decode(frames[1])
    except Exception:
        return None
    if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], str):
        return payload[1]
    return None


def engine_identity(engine_index: int) -> bytes:
    if not 0 <= engine_index <= MAX_ENGINE_INDEX:
        raise ValueError(f"engine index {engine_index} does not fit a u16 identity")
    return struct.pack("<H", engine_index)


# ----------------------------------------------------------------------------
# Adapters the scheduler drives.
# ----------------------------------------------------------------------------


class MsgpackSendSocket:
    """The scheduler's sender in this mode, in ``SenderWrapper``'s shape.

    Created before the model loads (the scheduler hands its sender to other
    components during init) and attached to the PUSH socket once the
    handshake completes; anything sent before that is dropped with a log
    line. A ``BatchTokenIDOutput`` is sliced to :class:`BatchTokenIDSlimOutput`
    with the load tail sampled at send time; an ``AbortReq`` the scheduler
    emits for a request becomes that request's terminal output, so SMG's
    stream never hangs on a scheduler-side abort; other control replies have
    no consumer on this wire and are dropped.
    """

    def __init__(self, engine_index: int = 0) -> None:
        self.socket: zmq.Socket | None = None
        self.engine_index = engine_index
        self._load_probe: Callable[[], LoadSnapshot] | None = None

    def attach(
        self,
        socket: zmq.Socket,
        engine_index: int,
        load_probe: Callable[[], LoadSnapshot] | None,
    ) -> None:
        self.socket = socket
        self.engine_index = engine_index
        self._load_probe = load_probe
        self._load_cache: tuple[float, dict] | None = None

    def _load_tail(self) -> dict:
        """The scheduler's load, refreshed at most every ``LOAD_TAIL_REFRESH_S``:
        the snapshot walks the waiting queue, too much for every decode step
        of a busy rank, and routing tolerates a tail that old."""
        if self._load_probe is None:
            return {}
        now = time.monotonic()
        if self._load_cache is not None and now - self._load_cache[0] < LOAD_TAIL_REFRESH_S:
            return self._load_cache[1]
        try:
            load = self._load_probe()
        except Exception as exc:  # the load tail is best-effort
            logger.warning("zmq msgpack: load snapshot failed: %s", exc)
            return {}
        tail = dict(
            num_running=int(load.num_running_reqs),
            num_waiting=int(load.num_waiting_reqs),
            kv_used_tokens=int(load.num_used_tokens),
            kv_total_tokens=int(load.max_total_num_tokens),
        )
        self._load_cache = (now, tail)
        return tail

    def _send_slim(self, slim: BatchTokenIDSlimOutput) -> None:
        if self.socket is None:
            logger.warning("zmq msgpack: dropping an output for %s before the handshake", slim.rids)
            return
        self.socket.send(_ENC.encode(slim), copy=False)

    def send_output(self, output: Any, recv_obj: object | None = None) -> None:
        from sglang.srt.managers.io_struct import AbortReq, BatchTokenIDOutput

        if isinstance(output, BatchTokenIDOutput):
            self._send_slim(
                BatchTokenIDSlimOutput.from_full(
                    output, engine_index=self.engine_index, **self._load_tail()
                )
            )
        elif isinstance(output, AbortReq) and output.rid:
            # The scheduler's own abort: its status (400 when it refused the
            # request, none for an operational abort), not this side's.
            reason = output.finished_reason or {}
            status = reason.get("status_code")
            self.send_terminal_abort(
                output.rid,
                output.abort_message or reason.get("message") or "Aborted",
                status=int(status) if isinstance(status, int) else None,
            )
        else:
            logger.debug(
                "zmq msgpack: dropping control reply %s (no consumer on this wire)",
                type(output).__name__,
            )

    def send_terminal_abort(self, rid: str, message: str, status: int | None = 400) -> None:
        """One request's terminal ``abort`` output with no tokens; a request
        this side refused is the client's error (400) unless told otherwise."""
        self._send_slim(
            BatchTokenIDSlimOutput(
                rids=[rid],
                output_ids=[[]],
                finished_reasons=["abort"],
                finished_messages=[message],
                finished_matched=[None],
                prompt_tokens=[0],
                completion_tokens=[0],
                cached_tokens=[0],
                output_token_logprobs_val=[[]],
                output_token_logprobs_idx=[[]],
                engine_index=self.engine_index,
                finished_status=[status],
                **self._load_tail(),
            )
        )

    def send_engine_dead(self) -> None:
        """Best-effort ENGINE_CORE_DEAD on shutdown or crash; never raises."""
        if self.socket is None:
            return
        try:
            self.socket.send(ENGINE_CORE_DEAD, zmq.NOBLOCK)
        except Exception as exc:
            logger.warning("zmq msgpack: ENGINE_CORE_DEAD send failed: %s", exc)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None


class MsgpackRecvSocket:
    """The scheduler's request source in this mode.

    ``drain`` is the contract the scheduler's request receiver already uses
    for an in-process ring: return every decoded request currently available
    without blocking. The tokenizer manager's request-side duties this
    transport bypasses run here: ``SamplingParams.normalize``/``verify`` and
    the ``n`` fan-out SMG performs itself. A request that fails them is not
    dropped (that would hang SMG's stream); it is answered with a terminal
    abort through the sender. A malformed frame is logged and skipped for the
    same reason it must never raise: an exception here would kill the
    scheduler.
    """

    def __init__(
        self,
        socket: zmq.Socket,
        vocab_size: int,
        reject: Callable[[str, str], None],
        tokenizer: object | None = None,
    ) -> None:
        self.socket = socket
        self._vocab_size = vocab_size
        self._reject = reject
        # The scheduler's tokenizer when it keeps one (grammar-constrained
        # decoding needs it); normalization then resolves stop strings too.
        self._tokenizer = tokenizer

    def _validation_error(self, req: TokenizedGenerateReqInput) -> str | None:
        params = req.sampling_params
        try:
            params.normalize(self._tokenizer)
            params.verify(self._vocab_size)
        except ValueError as exc:
            return str(exc)
        if params.n != 1:
            return f"n={params.n} is not served on this wire; SMG fans out n > 1 itself"
        if req.input_ids is None or len(req.input_ids) == 0:
            return "input_ids are required on this wire"
        return None

    def drain(self, max_recv: int) -> list:
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        received: list = []
        while max_recv < 0 or len(received) < max_recv:
            try:
                frames = self.socket.recv_multipart(zmq.NOBLOCK, copy=False)
            except zmq.ZMQError:  # Again, or the context going away
                break
            frames = [frame.buffer for frame in frames]
            try:
                decoded = decode_request_frames(frames)
            except Exception as exc:
                # A request SMG built that this scheduler cannot decode (a
                # version-skewed field, a value the engine's validation
                # refuses) is still SMG's request: answer it so the stream
                # ends, when its id can be read, rather than dropping it.
                rid = _malformed_request_id(frames)
                if rid is not None:
                    logger.warning("zmq msgpack: rejecting undecodable request %s: %s", rid, exc)
                    self._answer(rid, f"the scheduler could not decode the request: {exc}")
                else:
                    logger.warning(
                        "zmq msgpack: dropping a malformed message (%d frames, sizes=%s): %s",
                        len(frames),
                        [len(frame) for frame in frames],
                        exc,
                    )
                continue
            for obj in decoded:
                if isinstance(obj, TokenizedGenerateReqInput):
                    reason = self._validation_error(obj)
                    if reason is not None:
                        logger.warning("zmq msgpack: rejecting request %s: %s", obj.rid, reason)
                        self._answer(obj.rid or "", reason)
                        continue
                received.append(obj)
        return received

    def _answer(self, rid: str, reason: str) -> None:
        try:
            self._reject(rid, reason)
        except zmq.ZMQError as exc:  # SMG gone mid-rejection: nothing to answer
            logger.warning("zmq msgpack: could not answer rejected request %s: %s", rid, exc)

    def close(self) -> None:
        self.socket.close()


# ----------------------------------------------------------------------------
# Handshake.
# ----------------------------------------------------------------------------


def _recv_init_with_timeout(socket: zmq.Socket) -> list[bytes]:
    waited_ms = 0
    while waited_ms < INIT_TIMEOUT_MS:
        if socket.poll(INIT_POLL_SLICE_MS, zmq.POLLIN):
            return socket.recv_multipart()
        waited_ms += INIT_POLL_SLICE_MS
        logger.info(
            "zmq msgpack handshake: waiting for SMG's INIT (%ds elapsed)", waited_ms // 1000
        )
    raise TimeoutError(f"zmq msgpack handshake: no INIT from SMG after {INIT_TIMEOUT_MS} ms")


def _unbounded(socket: zmq.Socket) -> zmq.Socket:
    """No high-water marks and no linger, as the scheduler's own channels: a
    stalled SMG queues in memory rather than blocking the event loop inside
    ``send``, and shutdown never waits on it."""
    socket.setsockopt(zmq.SNDHWM, 0)
    socket.setsockopt(zmq.RCVHWM, 0)
    socket.setsockopt(zmq.LINGER, 0)
    return socket


def connect_msgpack_engine(
    context: zmq.Context,
    handshake_address: str,
    engine_index: int,
    ready_response: WireEngineCoreReadyResponse,
    vocab_size: int,
    sender: MsgpackSendSocket,
    load_probe: Callable[[], LoadSnapshot] | None = None,
    tokenizer: object | None = None,
) -> MsgpackRecvSocket:
    """Run the startup handshake against SMG and wire both adapters.

    1. HELLO on the handshake DEALER (identity = engine index).
    2. recv INIT -> SMG's input/output addresses.
    3. READY on the handshake DEALER.
    4. connect the input DEALER and register with the ready response.
    5. connect the output PUSH and attach it to ``sender``.
    """
    identity = engine_identity(engine_index)

    handshake = context.socket(zmq.DEALER)
    handshake.setsockopt(zmq.IDENTITY, identity)
    handshake.connect(handshake_address)
    logger.info("zmq msgpack handshake: dialing %s", handshake_address)
    handshake.send(encode(WireReadyMessage(status="HELLO", local=True, headless=True)))
    init = decode_init(_recv_init_with_timeout(handshake)[-1])
    handshake.send(encode(WireReadyMessage(status="READY", local=True, headless=True)))
    if not init.addresses.inputs or not init.addresses.outputs:
        raise ValueError(
            "zmq msgpack handshake: INIT carried no input/output addresses: "
            f"inputs={init.addresses.inputs} outputs={init.addresses.outputs}"
        )
    input_address, output_address = init.addresses.inputs[0], init.addresses.outputs[0]
    logger.info("zmq msgpack handshake: INIT input=%s output=%s", input_address, output_address)

    input_socket = _unbounded(context.socket(zmq.DEALER))
    input_socket.setsockopt(zmq.IDENTITY, identity)
    input_socket.connect(input_address)
    input_socket.send(encode(ready_response))

    output_socket = _unbounded(context.socket(zmq.PUSH))
    output_socket.connect(output_address)
    sender.attach(output_socket, engine_index, load_probe)

    handshake.close()
    logger.info("zmq msgpack handshake: complete (engine_index=%d)", engine_index)
    return MsgpackRecvSocket(input_socket, vocab_size, sender.send_terminal_abort, tokenizer)


def _sglang_version() -> str:
    try:
        from sglang.version import __version__

        return __version__
    except Exception:
        return "unknown"


def scheduler_dp_rank(scheduler: Scheduler) -> int:
    """The scheduler's DP rank, 0 without DP. SGLang <= 0.5.20 keeps it on the
    scheduler's own parallel state; later versions publish it on the parallel
    context (which the older context raises for)."""
    state = getattr(scheduler, "ps", None)
    if state is None:
        from sglang.srt.runtime_context import get_parallel

        state = get_parallel()
    try:
        rank = state.dp_rank
    except AttributeError:
        rank = None
    return int(rank) if rank is not None else 0


def ready_response_for_scheduler(scheduler: Scheduler) -> WireEngineCoreReadyResponse:
    """The facts SMG reads at registration, from a loaded scheduler."""
    args = scheduler.server_args
    page_size = int(getattr(scheduler, "page_size", 1) or 1)
    max_total = int(getattr(scheduler, "max_total_num_tokens", 0) or 0)
    chunked_prefill_size = getattr(scheduler, "chunked_prefill_size", None) or 0
    tp_size, pp_size, dp_size = int(args.tp_size), int(args.pp_size), int(args.dp_size)
    # Attention DP splits the TP group; otherwise each DP rank is a full replica.
    attn_tp_size = tp_size // dp_size if args.enable_dp_attention else tp_size
    return WireEngineCoreReadyResponse(
        max_model_len=int(scheduler.model_config.context_len),
        num_gpu_blocks=max_total // page_size,
        block_size=page_size,
        dtype=wire_dtype(scheduler.model_config.dtype),
        vllm_version=f"sglang-{_sglang_version()}",
        world_size=tp_size * pp_size,
        data_parallel_size=dp_size,
        tensor_parallel_size=attn_tp_size,
        pipeline_parallel_size=pp_size,
        data_parallel_rank=scheduler_dp_rank(scheduler),
        max_num_seqs=int(getattr(scheduler, "max_running_requests", 0) or 0),
        # A disabled chunked prefill is -1 here; the wire field is a cap >= 0.
        max_num_batched_tokens=max(0, int(chunked_prefill_size)),
        instance_id=str(args.served_model_name or scheduler.model_config.model_path),
        kv_cache_size_tokens=max_total,
    )


def connect_scheduler(
    scheduler: Scheduler,
    context: zmq.Context,
    sender: MsgpackSendSocket,
    handshake_address: str,
    engine_index: int,
) -> MsgpackRecvSocket:
    """Handshake this (rank-zero) scheduler with SMG and return its request
    source. Each DP rank dials with its own identity, ``engine_index +
    dp_rank``; SMG tells ranks apart, and routes requests back, by it."""
    return connect_msgpack_engine(
        context,
        handshake_address,
        int(engine_index) + scheduler_dp_rank(scheduler),
        ready_response_for_scheduler(scheduler),
        int(scheduler.model_config.vocab_size),
        sender,
        # The load inquirer is built after this point of Scheduler.__init__.
        load_probe=lambda: scheduler.load_inquirer.get_loads(),
        tokenizer=getattr(scheduler, "tokenizer", None),
    )
