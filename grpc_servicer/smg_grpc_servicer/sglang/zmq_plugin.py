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
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from smg_grpc_servicer.sglang.zmq_msgpack import (
    MAX_ENGINE_INDEX,
    MsgpackRecvSocket,
    MsgpackSendSocket,
    connect_scheduler,
)

logger = logging.getLogger(__name__)

# The headless launcher sets these for the scheduler ranks it spawns.
HANDSHAKE_ENV = "SMG_SGLANG_ZMQ_HANDSHAKE"
ENGINE_INDEX_ENV = "SMG_SGLANG_ZMQ_ENGINE_INDEX"

_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"
_CHANNELS = "sglang.srt.managers.scheduler_components.ipc_channels.SchedulerIpcChannels"
_RECEIVER = "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver"


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


def register() -> None:
    """The ``sglang.srt.plugins`` entry point."""
    if not active():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(f"{_CHANNELS}.create", create_channels, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.init_idle_sleeper", init_idle_sleeper, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.maybe_init_rust_server", after_rust_server, HookType.AFTER)
    HookRegistry.register(f"{_RECEIVER}._pull_raw_reqs", pull_raw_reqs, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.run_event_loop", run_event_loop, HookType.AROUND)
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
