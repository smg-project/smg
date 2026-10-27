"""An SGLang plugin that puts the scheduler behind SMG over ZMQ.

SGLang loads plugins (``sglang.srt.plugins`` entry points) in every scheduler
process before it constructs the scheduler, and lets them wrap methods by
dotted path. This plugin hooks the three seams the scheduler already has for
an alternative ingress and egress, the ones its embedded Rust server uses:

* ``SchedulerIpcChannels.create``: on the rank that owns request I/O, no
  tokenizer-manager sockets; both senders are one deferred
  :class:`~smg_grpc_servicer.sglang.zmq_msgpack.MsgpackSendSocket`;
* ``Scheduler.maybe_init_rust_server``: the first point after the model and
  cache are up, so the handshake's ready response can report their geometry;
  the input socket becomes the request source;
* ``SchedulerRequestReceiver._pull_raw_reqs``: drain that source.

Plus the idle sleeper (no socket to poll until the handshake ran) and the
event loop (an ``ENGINE_CORE_DEAD`` sentinel on exit). The hooks register
only when :data:`HANDSHAKE_ENV` is set, which the headless launcher does, so
an SGLang install with this package present behaves normally otherwise.

One more set of hooks names a failure the scheduler leaves unnamed: its own
grammar paths end a request whose grammar rejected a sampled token with a
bare ``FINISH_ABORT()`` (message ``Aborted``, no status, no ``err_type``), the
same finish an abort-all gives every running request. The hooks on those
paths (:func:`register_grammar_attribution`) stamp
:data:`~smg_grpc_servicer.sglang.GRAMMAR_REJECTED_ERR_TYPE` on the finish
they see appear, so the wire's ``finished_err_types`` column and the
tokenizer manager's finish dict carry the engine's own attribution; the
servicers raise the constraint-failure status on that name only. They
register under the handshake too, and under
:data:`~smg_grpc_servicer.sglang.GRAMMAR_ATTRIBUTION_ENV` for the Python
servicer's scheduler ranks.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pkgutil
from typing import Any

from smg_grpc_servicer.sglang import (
    ENGINE_INDEX_ENV,
    GRAMMAR_ATTRIBUTION_ENV,
    GRAMMAR_REJECTED_ERR_TYPE,
    GRAMMAR_REJECTED_MESSAGE,
    HANDSHAKE_ENV,
)
from smg_grpc_servicer.sglang.zmq_msgpack import (
    MAX_ENGINE_INDEX,
    MsgpackRecvSocket,
    MsgpackSendSocket,
    connect_scheduler,
)

logger = logging.getLogger(__name__)

_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"
_CHANNELS = "sglang.srt.managers.scheduler_components.ipc_channels.SchedulerIpcChannels"
_RECEIVER = "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver"
# The scheduler's grammar paths: the per-request advances of the result
# processor (speculative runs and plain decode in `_accept_grammar_tokens`,
# the prefilled token in `_apply_prefill_grammar`) and a decode side's first
# step on a transferred prefill (`process_prebuilt`, which advances every
# grammar in the batch over the last prefilled token).
_BATCH_RESULTS = (
    "sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor"
)
_DECODE_BATCH = (
    "sglang.srt.disaggregation.decode_schedule_batch_mixin.ScheduleBatchDisaggregationDecodeMixin"
)
_BARE_ABORT_MESSAGE = "Aborted"


class _State:
    """Per-process plugin state: the deferred sender exists only on the rank
    that owns request I/O, which is how the later hooks know they apply."""

    def __init__(self) -> None:
        self.sender: MsgpackSendSocket | None = None
        self.context: Any = None


_state = _State()


def handshake_address() -> str | None:
    value = os.environ.get(HANDSHAKE_ENV, "").strip()
    return value or None


def engine_index() -> int:
    value = int(os.environ.get(ENGINE_INDEX_ENV, "0") or 0)
    if not 0 <= value <= MAX_ENGINE_INDEX:
        raise ValueError(f"{ENGINE_INDEX_ENV}={value} must be in 0..={MAX_ENGINE_INDEX}")
    return value


def active() -> bool:
    return handshake_address() is not None


def attributes_grammar_aborts() -> bool:
    """Whether the grammar-attribution hooks register in this rank: under the
    handshake (the ZMQ wire carries the name), or when whoever spawned the
    rank asked for them by :data:`GRAMMAR_ATTRIBUTION_ENV` (the Python
    servicer, whose finish dicts carry it)."""
    return active() or os.environ.get(GRAMMAR_ATTRIBUTION_ENV, "").strip() not in ("", "0")


def register() -> None:
    """The ``sglang.srt.plugins`` entry point."""
    if attributes_grammar_aborts():
        register_grammar_attribution()
    if not active():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(f"{_CHANNELS}.create", create_channels, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.init_idle_sleeper", init_idle_sleeper, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.maybe_init_rust_server", after_rust_server, HookType.AFTER)
    HookRegistry.register(f"{_RECEIVER}._pull_raw_reqs", pull_raw_reqs, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.run_event_loop", run_event_loop, HookType.AROUND)
    from smg_grpc_servicer.sglang import mm_broadcast

    if mm_broadcast.enabled():
        HookRegistry.register(
            f"{_RECEIVER}._broadcast_reqs_across_ranks",
            mm_broadcast.broadcast_with_shm,
            HookType.AROUND,
        )

    logger.info("SMG ZMQ plugin: scheduler will dial %s", handshake_address())


def create_channels(original, cls, *args, **kwargs):
    """AROUND ``SchedulerIpcChannels.create``: the rank that owns request I/O
    gets no tokenizer-manager sockets and one deferred sender in both sender
    slots; the other ranks are unchanged. The deferred sender keeps its
    identity across the handshake because components bind its
    ``send_output`` during init."""
    if not kwargs.get("is_rank_zero"):
        return original(cls, *args, **kwargs)
    import zmq

    channels = original(cls, *args, **{**kwargs, "is_rank_zero": False})
    _state.context = zmq.Context(2)
    _state.sender = MsgpackSendSocket()
    return dataclasses.replace(
        channels, send_to_tokenizer=_state.sender, send_to_detokenizer=_state.sender
    )


def init_idle_sleeper(original, scheduler, *args, **kwargs):
    """AROUND: nothing to poll until the handshake ran; ``after_rust_server``
    builds the sleeper on the handshake socket."""
    if _state.sender is None:
        return original(scheduler, *args, **kwargs)
    scheduler.idle_sleeper = None
    return None


def after_rust_server(result, scheduler, *args, **kwargs):
    """AFTER ``maybe_init_rust_server``: handshake with SMG and install the
    request source. Leaves the hooked method's result alone."""
    if _state.sender is None:
        return None
    recv = connect_scheduler(
        scheduler, _state.context, _state.sender, handshake_address(), engine_index()
    )
    scheduler.recv_from_tokenizer = recv
    if scheduler.server_args.sleep_on_idle:
        from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper

        try:
            scheduler.idle_sleeper = IdleSleeper(
                sockets=[recv.socket], can_empty_cache=lambda: not scheduler._engine_paused
            )
        except TypeError:  # SGLang <= 0.5.20: no can_empty_cache
            scheduler.idle_sleeper = IdleSleeper(sockets=[recv.socket])
    return None


def pull_raw_reqs(original, receiver, *args, **kwargs):
    """AROUND ``_pull_raw_reqs``: drain the msgpack source where it is the
    ingress; every other rank keeps the original broadcast path."""
    recv = receiver.recv_from_tokenizer
    if isinstance(recv, MsgpackRecvSocket):
        return recv.drain(receiver.max_recv_per_poll)
    return original(receiver, *args, **kwargs)


def run_event_loop(original, scheduler, *args, **kwargs):
    """AROUND ``run_event_loop``: tell SMG this rank is gone when the loop
    ends, however it ends."""
    try:
        return original(scheduler, *args, **kwargs)
    finally:
        if _state.sender is not None:
            _state.sender.send_engine_dead()


def register_grammar_attribution() -> None:
    """Hook the scheduler's own grammar paths so the abort they end a request
    with names its cause. Each path sets a bare ``FINISH_ABORT()`` on the
    request when ``req.grammar.accept_token`` raises (the token the model
    sampled is not in the grammar), which nothing tells from the bare abort
    an abort-all gives every running request; the hooks stamp
    :data:`GRAMMAR_REJECTED_ERR_TYPE` (and a message, where the path left the
    default) on the finish they see appear. A release without one of the
    paths keeps that path's aborts bare, and the servicers answer them as
    before; it is logged, not an error."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target, hook in (
        (f"{_BATCH_RESULTS}._accept_grammar_tokens", attribute_grammar_abort),
        (f"{_BATCH_RESULTS}._apply_prefill_grammar", attribute_grammar_abort),
        (f"{_DECODE_BATCH}.process_prebuilt", attribute_prebuilt_grammar_aborts),
    ):
        if not _resolves(target):
            logger.info(
                "SMG plugin: %s is not on this release; its grammar aborts stay unnamed", target
            )
            continue
        HookRegistry.register(target, hook, HookType.AROUND)


def _resolves(target: str) -> bool:
    """Whether the dotted ``module.Class.method`` names a callable on this
    release (the registry would log a traceback for a missing one)."""
    owner, _, name = target.rpartition(".")
    try:
        return callable(getattr(pkgutil.resolve_name(owner), name, None))
    except Exception:  # an import error, a renamed module: no such path
        return False


def attribute_grammar_abort(original, processor, *args, **kwargs):
    """AROUND the result processor's per-request grammar advances
    (``_accept_grammar_tokens(req, tokens)`` and
    ``_apply_prefill_grammar(*, req, next_token_id, ...)``): a finish the
    call sets on the request is the grammar's rejection of the token, the one
    thing the call aborts for."""
    req = kwargs.get("req", args[0] if args else None)
    before = getattr(req, "to_finish", None)
    result = original(processor, *args, **kwargs)
    if req is not None:
        _name_grammar_abort(req, before)
    return result


def attribute_prebuilt_grammar_aborts(original, batch, *args, **kwargs):
    """AROUND ``process_prebuilt`` (a decode side's first step on a transferred
    prefill, which advances every grammar in the batch over the last prefilled
    token): a finish it sets on a request with a grammar is that rejection.
    Batches without a grammar cost one attribute read."""
    watched = []
    if getattr(batch, "has_grammar", True):
        watched = [
            (req, getattr(req, "to_finish", None))
            for req in getattr(batch, "reqs", None) or ()
            if getattr(req, "grammar", None) is not None
        ]
    result = original(batch, *args, **kwargs)
    for req, before in watched:
        _name_grammar_abort(req, before)
    return result


def _name_grammar_abort(req, before) -> None:
    """Stamp the finish ``req`` gained during a grammar advance as the
    grammar's: ``err_type`` (the scheduler's own field for the failure's
    name, carried by its finish dict) and, where the path left the default
    ``Aborted``, the message. A finish that was there before the advance, or
    none, is not the grammar's."""
    finish = getattr(req, "to_finish", None)
    if finish is None or finish is before or not hasattr(finish, "err_type"):
        return
    finish.err_type = GRAMMAR_REJECTED_ERR_TYPE
    if getattr(finish, "message", None) in (None, _BARE_ABORT_MESSAGE):
        finish.message = GRAMMAR_REJECTED_MESSAGE
