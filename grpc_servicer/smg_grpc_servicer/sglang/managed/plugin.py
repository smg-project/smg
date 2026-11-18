"""Managed-only scheduler hooks; legacy workers register no hooks from this module.

The SMG launcher supplies the instance fence and a same-host metadata output
endpoint. Managed requests carry a private control envelope in custom_params;
the normal API cannot supply that envelope to this worker pool.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from . import native

ENVELOPE = "__smg_managed_epd_v1"
ENABLE_ENV = "SMG_SGLANG_EPD"
_registered_role = None
ROLE_ENV = "SMG_SGLANG_MANAGED_EPD_ROLE"
INSTANCE_ENV = "SMG_SGLANG_MANAGED_EPD_GENERATION"
WORKER_ENV = "SMG_SGLANG_MANAGED_EPD_WORKER_ID"
EVENTS_ENV = "SMG_SGLANG_MANAGED_EPD_EVENTS"
_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"
_sender = None
_context = None


def register():
    """Opt into native launch dispatch, or install hooks for an already fenced role."""
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
    if role == "encoder":
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        HookRegistry.register(
            "sglang.srt.disaggregation.encoder.server.MMEncoder.__init__",
            encoder_initialized,
            HookType.AFTER,
        )
        HookRegistry.register(
            "sglang.srt.disaggregation.encoder.server.MMEncoder._compute_embedding",
            await_encoder_forward,
            HookType.AROUND,
        )
        _registered_role = role
        return
    if role not in ("prefill", "decode"):
        raise ValueError("managed scheduler role must be prefill or decode")
    if not os.environ.get(INSTANCE_ENV) or not os.environ.get(WORKER_ENV):
        raise ValueError("managed scheduler requires its instance fence")
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(f"{_SCHEDULER}.__init__", initialized, HookType.AFTER)
    HookRegistry.register(f"{_SCHEDULER}.handle_generate_request", prepare_request, HookType.AROUND)
    HookRegistry.register(
        "smg_grpc_servicer.sglang.zmq_msgpack.ready_response_for_scheduler",
        advertise,
        HookType.AFTER,
    )
    if role == "prefill":
        HookRegistry.register(
            f"{_SCHEDULER}._get_multimodal_inputs", retain_receiver_owner, HookType.AROUND
        )
        HookRegistry.register(f"{_SCHEDULER}._add_request_to_queue", before_queue, HookType.AROUND)

    _registered_role = role


def native_launch(original, server_args):
    """Select SMG lifecycle from native arguments without parsing or launching twice."""
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled in ("", "0"):
        return original(server_args)
    if enabled != "1":
        raise ValueError(f"{ENABLE_ENV} must be 0 or 1")
    from .launch import run_server

    return run_server(server_args)


def load_role_hooks():
    """Install the role discovered after native parsing, including in an already-loaded plugin."""
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

    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(result.device))
    # An executor Future is not cancelled by asyncio.run's all-Task shutdown.
    # A to_thread Task would lose its waiter while its CUDA thread was still active.
    try:
        waiter = asyncio.get_running_loop().run_in_executor(None, ready.synchronize)
    except RuntimeError:
        ready.synchronize()
        raise
    cancelled = False
    # Repeated shutdown cancellation must not release the batch while the GPU wait
    # still owns its event. Native staging retains its final sync for cloned slices.
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            cancelled = True
    waiter.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


def encoder_initialized(result, encoder, *args, **kwargs):
    """Install the same E-local artifact component on rank zero and every TP follower."""
    from .encoder_bridge import install_bridge

    install_bridge(
        encoder,
        os.environ["SMG_SGLANG_MANAGED_PROCESSOR_CONTRACT"],
        int(os.environ["SMG_SGLANG_MANAGED_BATCH_BYTES"]),
    )
    return None


def advertise(result, scheduler):
    """Legacy handshakes stay byte-for-byte unchanged; managed ranks explicitly fence their role."""
    import msgspec
    from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput

    native.verify_contract(
        scheduler.model_config.hf_config.model_type,
        MultimodalDataItem,
        MultimodalProcessorOutput,
        type(scheduler),
    )
    result = msgspec.to_builtins(result)
    result["sglang_managed"] = {
        "version": 1,
        "worker_id": os.environ[WORKER_ENV],
        "generation": os.environ[INSTANCE_ENV],
        "role": os.environ[ROLE_ENV],
        "layout": native.LAYOUT,
    }
    return result


def initialized(result, scheduler, *args, **kwargs):
    """Validate native fields and establish the bounded metadata channel after initialization."""
    global _sender, _context
    from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput

    model_type = scheduler.model_config.hf_config.model_type
    native.verify_contract(
        model_type, MultimodalDataItem, MultimodalProcessorOutput, type(scheduler)
    )
    # The existing SMG ingress determines actual I/O ownership, including nonzero DP ranks.
    from smg_grpc_servicer.sglang.zmq_plugin import _state

    if _state.sender is not None:
        _state.sender.managed_native_receipts = True
    if os.environ[ROLE_ENV] == "prefill" and _state.sender is not None:
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
    """Read SMG's private registration, not client-supplied media or endpoints."""
    params = req.sampling_params.custom_params or {}
    envelope = params.get(ENVELOPE)
    if envelope is None:
        raise ValueError("managed worker refuses an unregistered legacy Generate")
    if not isinstance(envelope, dict) or envelope.get("role") != os.environ[ROLE_ENV]:
        raise ValueError("managed request role mismatch")
    binding = native._proto().InputBinding()
    raw = envelope.get("binding", "")
    if len(raw) > native.MAX_MANIFEST_BYTES * 2:
        raise ValueError("managed binding exceeds byte bound")
    binding.ParseFromString(bytes.fromhex(raw))
    target = binding.prefill if envelope["role"] == "prefill" else binding.decode.target
    if target.generation != os.environ[INSTANCE_ENV] or target.worker_id != os.environ[WORKER_ENV]:
        raise ValueError("managed request instance mismatch")
    return envelope, binding


def _ordinary_text(req, media_field="mm_inputs"):
    """Only media-free requests without a prepared envelope may bypass layout binding."""
    params = getattr(getattr(req, "sampling_params", None), "custom_params", None) or {}
    return (
        ENVELOPE not in params
        and getattr(req, media_field, object()) is None
        and not getattr(req, "need_wait_for_mm_inputs", False)
        and not getattr(req, "input_embeds", None)
    )


def prepare_request(original, scheduler, recv_req, *args, **kwargs):
    """Apply only validated D metadata before the native handler; no placeholder Generate."""
    if _ordinary_text(recv_req):
        return original(scheduler, recv_req, *args, **kwargs)
    try:
        envelope, binding = _envelope(recv_req)
        if envelope["role"] == "prefill":
            recv_req.mm_routing_labels = {"image": list(binding.media_routing_labels)}
        if envelope["role"] == "decode" and envelope.get("manifest"):
            raw = envelope["manifest"]
            if len(raw) > native.MAX_MANIFEST_BYTES * 2:
                raise ValueError("managed manifest exceeds byte bound")
            manifest = native._proto().PreparedDecodeInput()
            manifest.ParseFromString(bytes.fromhex(raw))
            if recv_req.mm_inputs is not None or recv_req.need_wait_for_mm_inputs:
                raise ValueError("managed D refuses raw media or encoder dispatch")
            recv_req.mm_inputs = native.decode_input(manifest, binding)
            recv_req.input_ids = list(manifest.unpadded_ids)
            recv_req.input_text = None
            logging.getLogger(__name__).info(
                "managed D metadata bound rid=%s room=%s images=%s digest=%s",
                recv_req.rid,
                binding.bootstrap.room,
                len(manifest.images),
                manifest.integrity_digest.hex(),
            )
    except Exception as error:
        # Native intake retires this unadmitted request on every TP rank and
        # emits the existing abort response; no GPU work or raw-media fallback.
        logging.getLogger(__name__).warning("managed input refused rid=%s: %s", recv_req.rid, error)
        kwargs["mm_input_error"] = f"managed input refused: {error}"
    return original(scheduler, recv_req, *args, **kwargs)


def retain_receiver_owner(original, scheduler, received, *args, **kwargs):
    """Keep native pool ownership alive with the normalized P input, without copying embeddings."""
    prepared = original(scheduler, received, *args, **kwargs)
    if prepared is not None and prepared is not received:
        # Native pool release is tied to the processor output's GC, while normalization
        # retains only its tensor views. P owns that output until its input is retired.
        # Attach after native TP processing so the local owner never enters a collective.
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
        # Deterministic validation on all ranks; no new TP collective or embedding copy.
        export_started = time.monotonic_ns()
        manifest = native.export_layout(req, binding)
        export_us = (time.monotonic_ns() - export_started) // 1000
    except Exception as error:
        req.set_finish_with_abort(f"managed layout refused: {error}")
        return original(scheduler, req, *args, **kwargs)
    req._smg_managed_exported = True
    if _sender is not None:
        import zmq

        publish_started = time.monotonic_ns()
        try:
            _sender.send_multipart(
                [req.rid.encode(), manifest.SerializeToString(deterministic=True)],
                flags=zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            # Do not abort only the I/O rank or crash its unrelated requests.
            # No manifest reaches D: the existing bounded managed deadline
            # orders native Abort through the normal TP-synchronized ingress.
            logging.getLogger(__name__).exception(
                "managed metadata publication failed rid=%s", req.rid
            )
            return original(scheduler, req, *args, **kwargs)
        logging.getLogger(__name__).info(
            "managed P metadata exported before bootstrap rid=%s room=%s images=%s digest=%s "
            "layout_export_us=%s metadata_publish_us=%s",
            req.rid,
            binding.bootstrap.room,
            len(manifest.images),
            manifest.integrity_digest.hex(),
            export_us,
            (time.monotonic_ns() - publish_started) // 1000,
        )
    return original(scheduler, req, *args, **kwargs)
