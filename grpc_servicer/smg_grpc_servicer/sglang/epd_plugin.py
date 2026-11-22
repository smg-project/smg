"""Managed-only scheduler hooks; legacy workers register no hooks from this module."""

from __future__ import annotations

import logging
import os
import sys

from . import prepared_input as native

ENVELOPE = "__smg_managed_epd_v1"
ENABLE_ENV = "SMG_SGLANG_EPD"
_registered_role = None
ROLE_ENV = "SMG_SGLANG_MANAGED_EPD_ROLE"
EVENTS_ENV = "SMG_SGLANG_MANAGED_EPD_EVENTS"
_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"
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
            (
                "sglang.srt.disaggregation.encoder.server.MMEncoder.__init__",
                encoder_initialized,
                HookType.AFTER,
            ),
            (
                "sglang.srt.disaggregation.encoder.server.MMEncoder._compute_embedding",
                await_encoder_forward,
                HookType.AROUND,
            ),
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
                ]
            )
    for target, callback, kind in hooks:
        HookRegistry.register(target, callback, kind)
    _registered_role = role


def native_launch(original, server_args):
    """Redirect native `sglang.launch_server` into the managed EPD launcher when enabled."""
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled in ("", "0"):
        return original(server_args)
    if enabled != "1":
        raise ValueError(f"{ENABLE_ENV} must be 0 or 1")
    from .epd_launcher import run_server

    return run_server(server_args)


def load_role_hooks():
    """Load SGLang plugins and apply managed role-specific hooks before worker startup."""
    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry

    load_plugins()
    register()
    HookRegistry.apply_hooks()


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
    """Install the Rust artifact bridge on each encoder rank after `MMEncoder.__init__`."""
    from .epd_encoder import install_bridge

    install_bridge(
        encoder,
        os.environ["SMG_SGLANG_MANAGED_PROCESSOR_CONTRACT"],
        int(os.environ["SMG_SGLANG_MANAGED_BATCH_BYTES"]),
    )
    return None


def advertise(result, scheduler):
    """Verify native layout hooks and attach the managed contract to the ZMQ handshake."""
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
    """Enable managed native receipts and connect P's same-host metadata IPC socket."""
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


def _envelope(req):
    """Extract and validate the managed role envelope and `InputBinding` from `custom_params`."""
    params = req.sampling_params.custom_params or {}
    envelope = params.get(ENVELOPE)
    if envelope is None:
        raise ValueError("managed worker refuses an unregistered legacy Generate")
    if not isinstance(envelope, dict) or envelope.get("role") != os.environ[ROLE_ENV]:
        raise ValueError("managed request role mismatch")
    raw = envelope.get("binding", "")
    if len(raw) > native.MAX_MANIFEST_BYTES * 2:
        raise ValueError("managed binding exceeds byte bound")
    return envelope, native._proto().InputBinding.FromString(bytes.fromhex(raw))


def _ordinary_text(req, media_field="mm_inputs"):
    """Return whether a request carries no managed envelope or multimodal inputs."""
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
            raw = envelope["manifest"]
            if len(raw) > native.MAX_MANIFEST_BYTES * 2:
                raise ValueError("managed manifest exceeds byte bound")
            manifest = native._proto().PreparedDecodeInput.FromString(bytes.fromhex(raw))
            if recv_req.mm_inputs is not None or recv_req.need_wait_for_mm_inputs:
                raise ValueError("managed D refuses raw media or encoder dispatch")
            recv_req.mm_inputs = native.decode_input(
                manifest,
                require_mrope=getattr(scheduler.model_config, "model_is_mrope", True),
            )
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
    req._smg_managed_exported = True
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
            return original(scheduler, req, *args, **kwargs)
        logging.getLogger(__name__).info(
            "managed P metadata exported before bootstrap rid=%s room=%s images=%s",
            req.rid,
            binding.bootstrap.room,
            len(manifest.images),
        )
    return original(scheduler, req, *args, **kwargs)
