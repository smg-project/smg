"""Managed-only scheduler hooks; legacy workers register no hooks from this module."""

from __future__ import annotations

import logging
import os
import sys
from array import array

from . import prepared_input as native

ENVELOPE = "__smg_managed_epd_v1"
ENABLE_ENV = "SMG_SGLANG_EPD"
ROLE_ENV = "SMG_SGLANG_MANAGED_EPD_ROLE"
EVENTS_ENV = "SMG_SGLANG_MANAGED_EPD_EVENTS"
_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"
_MM_INPUTS = "sglang.srt.managers.schedule_batch.MultimodalInputs"
_ENCODER = "sglang.srt.disaggregation.encoder.server.MMEncoder"
_registered_role = None
_sender = None
_context = None


def register():
    """Opt into native launch dispatch, or install hooks for an already configured role."""
    global _registered_role
    role = os.environ.get(ROLE_ENV)
    if not role:
        if os.environ.get(ENABLE_ENV, "0") not in ("", "0"):
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType

            main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            module = (
                "__main__"
                if getattr(main_spec, "name", None) == "sglang.launch_server"
                else "sglang.launch_server"
            )
            HookRegistry.register(f"{module}.run_server", native_launch, HookType.AROUND)
        return
    if _registered_role == role:
        return
    if role not in ("encoder", "prefill", "decode"):
        raise ValueError("managed scheduler role must be encoder, prefill, or decode")
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if role == "encoder":
        hooks = [
            (f"{_ENCODER}.__init__", encoder_initialized, HookType.AFTER),
            (
                f"{_ENCODER}._calculate_hashes_from_features",
                reuse_materialized_feature_hashes,
                HookType.AROUND,
            ),
            (f"{_ENCODER}._compute_direct_embedding", cached_direct_embedding, HookType.AROUND),
            (f"{_ENCODER}._compute_embedding", await_encoder_forward, HookType.AROUND),
        ]
    else:
        hooks = [
            (f"{_SCHEDULER}.__init__", initialized, HookType.AFTER),
            (f"{_SCHEDULER}.handle_generate_request", prepare_request, HookType.AROUND),
            (
                "smg_grpc_servicer.sglang.zmq_msgpack.ready_response_for_scheduler",
                advertise,
                HookType.AFTER,
            ),
        ]
        if role == "prefill":
            hooks.extend(
                [
                    (
                        f"{_SCHEDULER}._get_multimodal_inputs",
                        retain_receiver_owner,
                        HookType.AROUND,
                    ),
                    (f"{_SCHEDULER}._add_request_to_queue", before_queue, HookType.AROUND),
                    (
                        f"{_SCHEDULER}.process_disagg_prefill_inflight_queue",
                        release_inflight_owners,
                        HookType.AFTER,
                    ),
                    (
                        f"{_SCHEDULER}._retire_aborted_prefill_result",
                        release_aborted_prefill_owner,
                        HookType.AFTER,
                    ),
                    (
                        f"{_SCHEDULER}.handle_bootstrap_failure",
                        release_bootstrap_failure_owner,
                        HookType.AFTER,
                    ),
                    (
                        f"{_MM_INPUTS}.release_features",
                        release_mm_features_owner,
                        HookType.AFTER,
                    ),
                ]
            )
    for target, callback, kind in hooks:
        HookRegistry.register(target, callback, kind)
    _registered_role = role


def native_launch(original, server_args):
    """Intercept ``sglang.launch_server.run_server`` when ``SMG_SGLANG_EPD=1``."""
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled in ("", "0"):
        return original(server_args)
    if enabled != "1":
        raise ValueError(f"{ENABLE_ENV} must be 0 or 1")
    from .epd_launcher import run_server

    return run_server(server_args)


def load_role_hooks():
    """Load SGLang plugins, register role-specific EPD hooks, and apply them."""
    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry

    load_plugins()
    register()
    HookRegistry.apply_hooks()


def reuse_materialized_feature_hashes(original, encoder, mm_feature, *args, **kwargs):
    """Reuse Rust-derived per-item keys without re-hashing unpacked pixel tensors.

    Supports both sglang 0.5.23+ ``(mm_feature, preprocess_result, modality)`` and
    sglang v0.5.21 ``(mm_feature, grid_thw, modality, mm_inputs=None)`` signatures.
    """
    preprocess_or_grid = args[0] if args else kwargs.get("preprocess_result")
    keys = getattr(preprocess_or_grid, "_smg_item_keys", None)
    if keys is not None:
        grid_thw = getattr(preprocess_or_grid, "grid_thw", None)
        if grid_thw is not None and len(keys) == len(grid_thw):
            return list(keys)
    mm_inputs = args[2] if len(args) >= 3 else kwargs.get("mm_inputs")
    if isinstance(mm_inputs, dict):
        keys = mm_inputs.get("_smg_item_keys")
    else:
        keys = getattr(mm_inputs, "_smg_item_keys", None)
    grid_thw = args[0] if args else kwargs.get("grid_thw")
    if keys is not None and grid_thw is not None and len(keys) == len(grid_thw):
        return list(keys)
    return original(encoder, mm_feature, *args, **kwargs)


async def cached_direct_embedding(original, encoder, ctx, *args, **kwargs):
    """Serve per-item cached image embeddings around ``MMEncoder._compute_direct_embedding``."""
    from .epd_encoder import cached_compute_direct_embedding

    keep_on_gpu = kwargs.get("keep_on_gpu", args[0] if args else False)
    return await cached_compute_direct_embedding(original, encoder, ctx, keep_on_gpu=keep_on_gpu)


async def await_encoder_forward(original, encoder, *args, **kwargs):
    """Retain dispatch/encode owners while GPU readiness waits outside the control loop."""
    result = await original(encoder, *args, **kwargs)
    if result is None or not result.is_cuda:
        return result
    import torch

    from .epd_encoder import run_uninterruptible

    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(result.device))
    try:
        await run_uninterruptible(None, ready.synchronize)
    except RuntimeError:
        ready.synchronize()
        raise
    return result


def encoder_initialized(result, encoder, *args, **kwargs):
    """AFTER ``MMEncoder.__init__``: install the managed Rust preprocessor bridge."""
    from .epd_encoder import install_bridge
    from .epd_launcher import PROCESSOR_CONTRACT, resolve_max_context_bytes

    # Per-request artifacts are bounded by config["max_artifact_bytes"] in ManagedEncoderServer
    # and serve_native; RustPreprocessorBridge bounds the fused multi-request batch total across
    # all TP ranks (including spawned followers initialized before serve() computes per-model caps).
    install_bridge(encoder, PROCESSOR_CONTRACT, resolve_max_context_bytes())
    return None


def advertise(result, scheduler):
    """AFTER ``ready_response_for_scheduler``: verify contract and attach managed EPD capability."""
    import msgspec

    native.verify_contract(scheduler)
    result = msgspec.to_builtins(result)
    result["sglang_managed"] = {
        "version": 1,
        "role": os.environ[ROLE_ENV],
        "layout": native.LAYOUT,
    }
    return result


def initialized(result, scheduler, *args, **kwargs):
    """AFTER ``Scheduler.__init__``: enable managed receipts and connect P's metadata IPC push socket."""
    global _sender, _context
    from smg_grpc_servicer.sglang.zmq_plugin import _state

    if _state.sender is None:
        native.verify_contract(scheduler)
        return None
    _state.sender.managed_native_receipts = True
    if os.environ[ROLE_ENV] == "prefill":
        import zmq

        endpoint = os.environ.get(EVENTS_ENV, "")
        if not endpoint.startswith("ipc://"):
            raise ValueError("managed metadata channel must be a same-host IPC endpoint")
        _context = zmq.Context()
        _sender = _context.socket(zmq.PUSH)
        _sender.setsockopt(zmq.SNDHWM, 32)
        _sender.setsockopt(zmq.LINGER, 0)
        _sender.connect(endpoint)
    return None


def _decode_hex_field(raw, label):
    """Decode a bounded hex-encoded protobuf payload from the managed control envelope."""
    if len(raw) > native.MAX_MANIFEST_BYTES * 2:
        raise ValueError(f"managed {label} exceeds byte bound")
    return bytes.fromhex(raw)


def _envelope(req):
    """Extract and validate the private managed EPD control envelope and InputBinding."""
    params = req.sampling_params.custom_params or {}
    envelope = params.get(ENVELOPE)
    if envelope is None:
        raise ValueError("managed worker refuses an unregistered legacy Generate")
    if not isinstance(envelope, dict) or envelope.get("role") != os.environ[ROLE_ENV]:
        raise ValueError("managed request role mismatch")
    binding = native._proto().InputBinding.FromString(
        _decode_hex_field(envelope.get("binding", ""), "binding")
    )
    return envelope, binding


def _ordinary_text(req, media_field="mm_inputs"):
    """Return True when a request carries neither a managed envelope nor multimodal/embedding inputs."""
    params = getattr(getattr(req, "sampling_params", None), "custom_params", None) or {}
    return (
        ENVELOPE not in params
        and getattr(req, media_field, object()) is None
        and not getattr(req, "need_wait_for_mm_inputs", False)
        and not getattr(req, "input_embeds", None)
    )


def prepare_request(original, scheduler, recv_req, *args, **kwargs):
    """Apply validated D metadata before the native handler; no raw-media fallback."""
    if _ordinary_text(recv_req):
        return original(scheduler, recv_req, *args, **kwargs)
    try:
        envelope, binding = _envelope(recv_req)
        if envelope["role"] == "decode" and envelope.get("manifest"):
            manifest = native._proto().PreparedDecodeInput.FromString(
                _decode_hex_field(envelope["manifest"], "manifest")
            )
            if recv_req.mm_inputs is not None or recv_req.need_wait_for_mm_inputs:
                raise ValueError("managed D refuses raw media or encoder dispatch")
            recv_req.mm_inputs = native.decode_input(
                manifest,
                require_mrope=getattr(scheduler.model_config, "model_is_mrope", True),
            )
            if isinstance(recv_req.input_ids, array):
                recv_req.input_ids = array(recv_req.input_ids.typecode, manifest.unpadded_ids)
            else:
                recv_req.input_ids = list(manifest.unpadded_ids)
            recv_req.input_text = None
            logging.getLogger(__name__).info(
                "managed D metadata bound rid=%s room=%s images=%s",
                recv_req.rid,
                binding.bootstrap.room,
                len(manifest.images),
            )
    except Exception as error:
        logging.getLogger(__name__).warning("managed input refused rid=%s: %s", recv_req.rid, error)
        kwargs["mm_input_error"] = f"managed input refused: {error}"
    return original(scheduler, recv_req, *args, **kwargs)


def retain_receiver_owner(original, scheduler, received, *args, **kwargs):
    """Keep native pool ownership alive with the normalized P input, without copying embeddings."""
    prepared = original(scheduler, received, *args, **kwargs)
    if prepared is not None and prepared is not received:
        prepared._smg_managed_receiver_owner = received
    return prepared


def _release_receiver_owner(mm_inputs):
    """Drop retained MMReceiver pool ownership so weakref finalizers free GPU slots immediately."""
    if mm_inputs is not None and hasattr(mm_inputs, "_smg_managed_receiver_owner"):
        mm_inputs._smg_managed_receiver_owner = None


def _release_req_receiver_owner(req):
    """Release retained MMReceiver pool ownership held on a completed or aborted prefill Req."""
    _release_receiver_owner(getattr(req, "multimodal_inputs", None))


def release_mm_features_owner(result, mm_inputs, *args, **kwargs):
    """AFTER ``MultimodalInputs.release_features``: drop retained pool ownership immediately."""
    _release_receiver_owner(mm_inputs)
    return None


def release_inflight_owners(result, scheduler, *args, **kwargs):
    """AFTER ``Scheduler.process_disagg_prefill_inflight_queue``: free GPU slots for done reqs."""
    if result:
        for req in result:
            _release_req_receiver_owner(req)
    return None


def release_aborted_prefill_owner(result, scheduler, req, *args, **kwargs):
    """AFTER ``Scheduler._retire_aborted_prefill_result``: free GPU slot when req is retired."""
    if result:
        _release_req_receiver_owner(req)
    return None


def release_bootstrap_failure_owner(result, scheduler, req, *args, **kwargs):
    """AFTER ``Scheduler.handle_bootstrap_failure``: free GPU slot on prefill bootstrap failure."""
    _release_req_receiver_owner(req)
    return None


def _single_rank_tp(scheduler) -> bool:
    """Return True only when the scheduler is provably single-rank (TP=1, CP=1)."""
    server_args = getattr(scheduler, "server_args", None)
    tp_size = next(
        (
            v
            for v in (
                getattr(scheduler, "attn_tp_size", None),
                getattr(scheduler, "tp_size", None),
                getattr(server_args, "tp_size", None),
            )
            if v is not None
        ),
        None,
    )
    cp_size = getattr(server_args, "attn_cp_size", 1) if server_args is not None else 1
    return tp_size == 1 and cp_size == 1


def before_queue(original, scheduler, req, *args, **kwargs):
    """Export once after receiver/TP/layout preparation and before native PD bootstrap."""
    if (
        _ordinary_text(req, "multimodal_inputs")
        or req.to_finish is not None
        or getattr(req, "_smg_managed_exported", False)
    ):
        return original(scheduler, req, *args, **kwargs)
    try:
        envelope, binding = _envelope(req)
        if envelope.get("text_only"):
            return original(scheduler, req, *args, **kwargs)
        manifest = native.export_layout(
            req, binding, require_mrope=getattr(scheduler.model_config, "model_is_mrope", True)
        )
    except Exception as error:
        req.set_finish_with_abort(f"managed layout refused: {error}")
        return original(scheduler, req, *args, **kwargs)
    if _sender is not None:
        import zmq

        try:
            _sender.send_multipart(
                [req.rid.encode(), manifest.SerializeToString()], flags=zmq.NOBLOCK
            )
        except zmq.ZMQError:
            logging.getLogger(__name__).exception(
                "managed metadata publication failed rid=%s", req.rid
            )
            # Only abort locally at TP=1/CP=1; at TP/CP>1 let all ranks enqueue symmetrically
            # so the gateway's layout deadline aborts every rank uniformly.
            if _single_rank_tp(scheduler):
                req.set_finish_with_abort("managed metadata publication failed")
            return original(scheduler, req, *args, **kwargs)
        logging.getLogger(__name__).info(
            "managed P metadata exported before bootstrap rid=%s room=%s images=%s",
            req.rid,
            binding.bootstrap.room,
            len(manifest.images),
        )
    req._smg_managed_exported = True
    return original(scheduler, req, *args, **kwargs)
