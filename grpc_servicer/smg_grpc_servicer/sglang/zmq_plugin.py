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

import copy
import dataclasses
import hashlib
import json
import logging
import os
import pkgutil
import re
from array import array
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


# Diagnostic only: correlated public CI requests; no GPU transfers or extra requests.
_QWENVL_TRACE_SLOTS = iter(range(40))
_QWENVL_OUTPUTS = {}


def _trace_qwenvl(event, scheduler=None, req=None, **facts):
    if os.environ.get("SMG_QWENVL_REQUEST_DIAGNOSTIC") != "1":
        return
    try:
        if req is not None:
            rid = getattr(req, "rid", None)
            match = (
                re.match(r"^(smg-qwenvl-[0-9a-f]{32})(?:-|$)", rid)
                if isinstance(rid, str) and len(rid) <= 512
                else None
            )
            if match is None:
                return
            facts["client_trace_id"] = match.group(1)
            model = getattr(getattr(scheduler, "server_args", None), "model_path", None)
            mm = getattr(req, "multimodal_inputs", None)
            canonical = "Qwen/Qwen3-VL-8B-Instruct"
            local_root = os.environ.get("ROUTER_LOCAL_MODEL_PATH", "")
            local_alias = os.path.join(local_root, canonical) if local_root else None
            if model not in (canonical, local_alias) or model is None or mm is None:
                return
        # Reserve before IDs, item/grid metadata, hashing or tensor property reads.
        slot = next(_QWENVL_TRACE_SLOTS, None)
        if slot is None:
            return
        if req is not None:

            def ids(value):
                if not isinstance(value, (list, array)):
                    return {"available": False}
                result = {"length": len(value), "sha256": None}
                if len(value) <= 32768:
                    digest = hashlib.sha256()
                    for token in value:
                        if type(token) is not int or not -(2**63) <= token < 2**63:
                            return {**result, "integer_domain_valid": False}
                        digest.update(token.to_bytes(8, "little", signed=True))
                    result["sha256"] = digest.hexdigest()
                else:
                    result["hash_omitted_over_32768"] = True
                return result

            def tensor_metadata(value, *, integer_values=False):
                if value is None:
                    return None
                if type(value) is int:
                    return (
                        {"scalar": value}
                        if -(2**63) <= value < 2**63
                        else {"scalar_over_cap": True}
                    )
                dims = getattr(value, "shape", None)
                device = getattr(getattr(value, "device", None), "type", None)
                result = {
                    "shape": [dim if type(dim) is int else None for dim in dims]
                    if isinstance(dims, (list, tuple)) and len(dims) <= 8
                    else None,
                    "device_type": device if device in ("cpu", "cuda", "meta") else None,
                    "dtype": str(getattr(value, "dtype", ""))[:32],
                }
                # Only small CPU integer metadata; never .cpu(), numpy or feature copies.
                if integer_values:
                    result["values_status"] = "OMITTED_NOT_SMALL_CPU_INTEGER"
                    if (
                        device == "cpu"
                        and result["dtype"] in ("torch.int32", "torch.int64")
                        and value.numel() <= 4096
                    ):
                        values = value.reshape(-1).tolist()
                        summary = ids(values)
                        result.update(values_status="CAPTURED", values=summary, first12=values[:12])
                return result

            def item_metadata(item):
                modality = getattr(getattr(item, "modality", None), "name", None)
                data = getattr(item, "model_specific_data", None)
                return {
                    "modality": modality if modality in ("IMAGE", "VIDEO", "AUDIO") else None,
                    "model_specific_data_is_dict": isinstance(data, dict),
                    "image_grid_thw_present": isinstance(data, dict) and "image_grid_thw" in data,
                    "image_grid_thw": tensor_metadata(
                        data.get("image_grid_thw"), integer_values=True
                    )
                    if isinstance(data, dict)
                    else None,
                    "feature": tensor_metadata(getattr(item, "feature", None)),
                }

            original = ids(getattr(req, "origin_input_ids", None))
            unpadded = ids(getattr(req, "origin_input_ids_unpadded", None))
            items = getattr(mm, "mm_items", None)
            rid = getattr(req, "rid", None)
            processor = getattr(scheduler, "_mm_processor", None)
            config = getattr(processor, "config", None)
            revision = getattr(config, "_commit_hash", None)
            facts.update(
                mrope_processor_class=(
                    type(processor).__module__ + "." + type(processor).__qualname__
                )[:192]
                if processor is not None
                else None,
                mrope_processor_config_commit=revision[:128] if isinstance(revision, str) else None,
                processor_revision_scope=(
                    "Direct live mrope processor config only; "
                    "None means unavailable, not cached-snapshot proof"
                ),
                request_id_sha256=hashlib.sha256(rid.encode()).hexdigest()
                if isinstance(rid, str) and len(rid) <= 512
                else None,
                model=model,
                original_ids=original,
                unpadded_ids=unpadded,
                original_unpadded_hash_equal=(original["sha256"] == unpadded["sha256"])
                if original.get("sha256") is not None and unpadded.get("sha256") is not None
                else None,
                mm_items_count=len(items) if isinstance(items, (list, tuple)) else None,
                mm_items=[item_metadata(item) for item in items[:4]]
                if isinstance(items, (list, tuple))
                else None,
                mm_items_summary_truncated=isinstance(items, (list, tuple)) and len(items) > 4,
                mrope_positions=tensor_metadata(
                    getattr(mm, "mrope_positions", None), integer_values=True
                ),
                mrope_position_delta=tensor_metadata(
                    getattr(mm, "mrope_position_delta", None), integer_values=True
                ),
            )
        record = json.dumps({"event": event, "slot": slot, **facts}, sort_keys=True)
        if len(record) <= 8192:
            logger.warning("SMG_QWENVL_REQUEST_DIAGNOSTIC %s", record)
    except Exception:
        # Diagnostic metadata must not change engine success/failure semantics.
        return


def trace_qwenvl_output(output):
    """Observe the existing emitted token rows; bounded12 requests/512 IDs each."""
    if os.environ.get("SMG_QWENVL_REQUEST_DIAGNOSTIC") != "1":
        return
    try:
        for index in range(min(len(output.rids), 16)):
            rid = output.rids[index]
            match = (
                re.match(r"^(smg-qwenvl-[0-9a-f]{32})(?:-|$)", rid)
                if isinstance(rid, str) and len(rid) <= 512
                else None
            )
            if match is None:
                continue
            if rid not in _QWENVL_OUTPUTS and len(_QWENVL_OUTPUTS) >= 12:
                continue
            row = output.output_ids[index]
            if not isinstance(row, list) or not all(
                type(v) is int and 0 <= v < 2**63 for v in row[:513]
            ):
                continue
            tokens, truncated = _QWENVL_OUTPUTS.get(rid, ([], False))
            room = 512 - len(tokens)
            truncated = truncated or len(row) > room
            tokens.extend(row[:room])
            _QWENVL_OUTPUTS[rid] = (tokens, truncated)
            reason = output.finished_reasons[index]
            if reason:
                del _QWENVL_OUTPUTS[rid]
                _trace_qwenvl(
                    "output",
                    client_trace_id=match.group(1),
                    request_id_sha256=hashlib.sha256(rid.encode()).hexdigest(),
                    emitted_token_ids=tokens,
                    truncated=truncated,
                    finish_reason=reason if reason in ("stop", "length", "abort") else "other",
                )
    except Exception:
        return


def _trace_mrope_call(original, scheduler, req, branch, *args, **kwargs):
    _trace_qwenvl("mrope_before", scheduler, req, branch=branch)
    try:
        return original(scheduler, req, *args, **kwargs)
    finally:
        _trace_qwenvl("mrope_after", scheduler, req, branch=branch)


def register() -> None:
    """The ``sglang.srt.plugins`` entry point."""
    if attributes_grammar_aborts():
        register_grammar_attribution()
    if not active():
        _trace_qwenvl("registration", handshake_active=False, registered=False)
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(f"{_CHANNELS}.create", create_channels, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.init_idle_sleeper", init_idle_sleeper, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.maybe_init_rust_server", after_rust_server, HookType.AFTER)
    HookRegistry.register(f"{_RECEIVER}._pull_raw_reqs", pull_raw_reqs, HookType.AROUND)
    HookRegistry.register(f"{_SCHEDULER}.run_event_loop", run_event_loop, HookType.AROUND)
    target = f"{_SCHEDULER}._maybe_compute_mrope_positions"
    mrope_available = _resolves(target)
    if mrope_available:
        HookRegistry.register(target, mrope_from_unpadded, HookType.AROUND)
    _trace_qwenvl("registration", handshake_active=True, registered=mrope_available)
    from smg_grpc_servicer.sglang import mm_broadcast

    if mm_broadcast.enabled():
        target = f"{_RECEIVER}._broadcast_reqs_across_ranks"
        if _resolves(target):
            HookRegistry.register(target, mm_broadcast.broadcast_with_shm, HookType.AROUND)
        else:
            logger.info("SMG plugin: %s is unavailable; using inline media broadcast", target)

    logger.info("SMG ZMQ plugin: scheduler will dial %s", handshake_address())


def mrope_from_unpadded(original, scheduler, req, *args, **kwargs):
    """Let the SDK compute positions from image anchors, keeping hash tokens for KV identity."""
    mm = getattr(req, "multimodal_inputs", None)
    unpadded = getattr(req, "origin_input_ids_unpadded", None)
    padded = getattr(req, "origin_input_ids", None)
    if (
        mm is not None
        and getattr(mm, "mrope_positions", None) is None
        and getattr(scheduler, "_mm_processor", None) is not None
        and (isinstance(unpadded, array) and unpadded.typecode == "q" or isinstance(unpadded, list))
        and len(unpadded) > 0
        and isinstance(padded, (array, list))
        and len(unpadded) == len(padded)
        and all(type(token) is int and 0 <= token < 2**63 for token in unpadded)
    ):
        # The scheduler pads first; its saved buffer still has the expanded image anchors.
        # A shallow view shares multimodal results without rewriting the caller's cache tokens.
        req_view = copy.copy(req)
        req_view.origin_input_ids = unpadded
        _trace_qwenvl("mrope_selected_unpadded", scheduler, req, branch="unpadded")
        return _trace_mrope_call(original, scheduler, req_view, "unpadded", *args, **kwargs)
    return _trace_mrope_call(original, scheduler, req, "fallback", *args, **kwargs)


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
