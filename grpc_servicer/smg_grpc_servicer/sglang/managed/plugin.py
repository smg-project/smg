"""Managed-only scheduler hooks; legacy workers register no hooks from this module.

The SMG launcher supplies the instance fence and a same-host metadata output
endpoint. Managed requests carry a private control envelope in custom_params;
the normal API cannot supply that envelope to this worker pool.
"""

from __future__ import annotations

import logging
import os

from . import native

ENVELOPE = "__smg_managed_epd_v1"
ROLE_ENV = "SMG_SGLANG_MANAGED_EPD_ROLE"
INSTANCE_ENV = "SMG_SGLANG_MANAGED_EPD_GENERATION"
WORKER_ENV = "SMG_SGLANG_MANAGED_EPD_WORKER_ID"
EVENTS_ENV = "SMG_SGLANG_MANAGED_EPD_EVENTS"
_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"
_sender = None
_context = None


def register():
    """Register narrow hooks only for an explicitly managed P/D launcher."""
    role = os.environ.get(ROLE_ENV)
    if not role:
        return
    if role == "encoder":
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        HookRegistry.register(
            "sglang.srt.disaggregation.encoder.server.MMEncoder.__init__",
            encoder_initialized,
            HookType.AFTER,
        )
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
        HookRegistry.register(f"{_SCHEDULER}._add_request_to_queue", before_queue, HookType.AROUND)
    else:
        # A metadata-only D has no legitimate processor path. Fail closed even
        # if a future native scheduler starts calling one implicitly.
        processor = "sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor"
        for method in ("compute_mrope_positions", "get_mm_data", "process_mm_data_async"):
            HookRegistry.register(
                f"{processor}.{method}", refuse_decode_processing, HookType.AROUND
            )


def refuse_decode_processing(original, *args, **kwargs):
    """D must consume P's native metadata without any multimodal processor invocation."""
    raise RuntimeError("managed D refuses multimodal processing; complete P metadata is required")


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


def prepare_request(original, scheduler, recv_req, *args, **kwargs):
    """Apply only validated D metadata before the native handler; no placeholder Generate."""
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
    return original(scheduler, recv_req, *args, **kwargs)


def before_queue(original, scheduler, req, *args, **kwargs):
    """Export once after receiver/TP/layout preparation and before native PD bootstrap."""
    if req.to_finish is not None or getattr(req, "_smg_managed_exported", False):
        return original(scheduler, req, *args, **kwargs)
    envelope, binding = _envelope(req)
    if envelope.get("text_only"):
        return original(scheduler, req, *args, **kwargs)
    # Validate on every rank, without adding a TP collective or copying embeddings.
    manifest = native.export_layout(req, binding)
    req._smg_managed_exported = True
    if _sender is not None:
        import zmq

        # A full/broken output channel is a worker error, never a silently lost layout.
        _sender.send_multipart(
            [req.rid.encode(), manifest.SerializeToString(deterministic=True)], flags=zmq.NOBLOCK
        )
        logging.getLogger(__name__).info(
            "managed P metadata exported before bootstrap rid=%s room=%s images=%s digest=%s",
            req.rid,
            binding.bootstrap.room,
            len(manifest.images),
            manifest.integrity_digest.hex(),
        )
    return original(scheduler, req, *args, **kwargs)
