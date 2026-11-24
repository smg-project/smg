"""Opt-in, bounded observations of the existing three-image Qwen request.

No imports of Torch/SDK/model code. Tensor reads use the already-loaded Torch;
GPU readback can synchronize and is not a performance measurement. Missing,
copied, mixed, DP or graph-captured associations remain UNKNOWN, never success.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import logging
import math
import os
import re
import stat
import sys
import threading
import types
import weakref

_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_QWEN_SHA = "d565d469deb286fa531c478f7eac3814cfa7e9820dc841a6320bdf8dc4a65be3"
_VISION_SHA = "02448653065b9edfe1062e6dcae9f12bab57b4abf615180f305ed166ac477aec"
_TENSOR_BYTES = 8 * 1024**2
_TRACE_BYTES = 32 * 1024**2
_PROCESS_BYTES = 96 * 1024**2
_lock = threading.RLock()
_context = contextvars.ContextVar("smg_qwenvl_numerical_scope", default=None)
_traces = {}
_items = {}  # id -> (weak reference, set of request ids); never retains features/items.
_process_bytes = 0
_events = 0
_unknown_boundaries = set()
_installed = False
_install_attempted = False
_logger = logging.getLogger(__name__)


def _enabled():
    return os.environ.get("SMG_QWENVL_REQUEST_DIAGNOSTIC") == "1"


def _field(value, name):
    try:
        fields = object.__getattribute__(value, "__dict__")
        children = fields.get("_modules", {})
        return fields.get(name, children.get(name) if type(children) is dict else None)
    except Exception:
        return None


def _emit(rid, boundary, **facts):
    global _events
    try:
        with _lock:
            if _events >= 80:
                return
            state = _traces.get(rid)
            if state is not None:
                if state["events"] >= 20:
                    return
                state["events"] += 1
            else:
                if boundary in _unknown_boundaries:
                    return
                _unknown_boundaries.add(boundary)
            _events += 1
            event = _events
        match = re.match(r"^(smg-qwenvl-[0-9a-f]{32})(?:-|$)", rid or "")
        payload = json.dumps(
            dict(
                event="numerical_boundary",
                process_id=os.getpid(),
                sequence=event,
                boundary=boundary,
                client_trace_id=match.group(1) if match else None,
                request_id_sha256=hashlib.sha256(rid.encode()).hexdigest() if rid else None,
                **facts,
            ),
            sort_keys=True,
            allow_nan=False,
        )
        if len(payload.encode()) > 8192:
            payload = json.dumps(
                dict(
                    event="numerical_boundary",
                    process_id=os.getpid(),
                    sequence=event,
                    boundary=boundary,
                    status="UNKNOWN_RECORD_BYTE_CAP",
                ),
                sort_keys=True,
            )
        _logger.warning("SMG_QWENVL_REQUEST_DIAGNOSTIC %s", payload)
    except Exception:
        # No diagnostic error may change the original call/return/exception.
        pass


def _reserve(rid, size):
    global _process_bytes
    with _lock:
        state = _traces[rid]
        if size > _TENSOR_BYTES:
            return "UNKNOWN_TENSOR_BYTE_CAP"
        if state["bytes"] + size > _TRACE_BYTES:
            return "UNKNOWN_TRACE_BYTE_CAP"
        if _process_bytes + size > _PROCESS_BYTES:
            return "UNKNOWN_PROCESS_BYTE_CAP"
        state["bytes"] += size
        _process_bytes += size
    return None


def _tensor(rid, value, *, integers=False):
    """Full logical-C-order bytes only; never a sampled/truncated hash."""
    try:
        torch = sys.modules.get("torch")
        if torch is None or not isinstance(value, torch.Tensor) or type(value) is not torch.Tensor:
            return {"status": "UNKNOWN_NOT_PLAIN_TENSOR"}
        shape = list(value.shape)
        if len(shape) > 8 or any(type(v) is not int or v < 0 for v in shape):
            return {"status": "UNKNOWN_SHAPE"}
        dtype = str(value.dtype)
        if dtype not in (
            "torch.bfloat16",
            "torch.float16",
            "torch.float32",
            "torch.float64",
            "torch.int32",
            "torch.int64",
            "torch.uint8",
        ):
            return {"status": "UNKNOWN_DTYPE", "dtype": dtype}
        size = value.numel() * value.element_size()
        reason = _reserve(rid, size)
        metadata = dict(
            shape=shape,
            dtype=dtype,
            device=str(value.device),
            stride=list(value.stride()),
            storage_offset=value.storage_offset(),
            bytes=size,
        )
        if reason:
            return dict(metadata, status=reason, full_byte_sha256=None)
        if value.device.type not in ("cpu", "cuda") or value.layout is not torch.strided:
            return dict(metadata, status="UNKNOWN_DEVICE_OR_LAYOUT", full_byte_sha256=None)
        if value.device.type == "cuda":
            # Do not perform readbacks when the capture state itself is unknown.
            try:
                with torch.cuda.device(value.device):
                    capturing = torch.cuda.is_current_stream_capturing()
                if capturing:
                    return dict(
                        metadata, status="UNKNOWN_CUDA_GRAPH_CAPTURE", full_byte_sha256=None
                    )
            except Exception:
                return dict(metadata, status="UNKNOWN_CUDA_CAPTURE_STATE", full_byte_sha256=None)
        try:
            version = value._version
        except RuntimeError:
            version = None  # Inference tensors legitimately have no version counter.
        metadata["version_guard"] = (
            "AVAILABLE" if version is not None else "UNKNOWN_NO_VERSION_COUNTER"
        )
        with torch.no_grad():
            host = value.detach().to(device="cpu", copy=True).contiguous()
            raw = host.reshape(-1).view(torch.uint8).numpy().tobytes()
            if len(raw) != size or (version is not None and value._version != version):
                return dict(metadata, status="UNKNOWN_CHANGED_DURING_READ", full_byte_sha256=None)
            result = dict(
                metadata,
                status="CAPTURED_FULL_LOGICAL_BYTES",
                full_byte_sha256=hashlib.sha256(raw).hexdigest(),
                byte_order=sys.byteorder,
                interpretation="logical_C_order_not_storage_padding",
                coherence_scope="current_stream_readback_not_global_atomic_snapshot",
            )
            if host.is_floating_point():
                finite = torch.isfinite(host)
                result["finite_count"] = int(finite.sum().item())
                result["nonfinite_count"] = host.numel() - result["finite_count"]
                if result["nonfinite_count"] == 0 and host.numel():
                    values = host.to(torch.float64)
                    for name, scalar in (
                        ("min", values.min()),
                        ("max", values.max()),
                        ("mean", values.mean()),
                    ):
                        number = float(scalar)
                        result[name] = number if math.isfinite(number) else None
                        if result[name] is None:
                            result[name + "_status"] = "UNKNOWN_NONFINITE_REDUCTION"
            elif integers and host.numel() <= 64:
                result["integer_values"] = host.reshape(-1).tolist()
            return result
    except Exception as exc:
        return {
            "status": "UNKNOWN_TENSOR_READ",
            "error_type": type(exc).__name__,
            "full_byte_sha256": None,
        }


def _source(function, files):
    """Loaded public Python definition; no arbitrary fallback path or dynamic repr."""
    if type(function) is types.MethodType:
        function = function.__func__
    if type(function) is not types.FunctionType:
        return {"status": "UNKNOWN_NON_PYTHON_CALLABLE"}
    module = function.__module__
    package_name = module.split(".", 1)[0]
    if package_name not in ("sglang", "aiter"):
        return {"status": "UNKNOWN_CALLABLE_NAMESPACE"}
    loaded, package = sys.modules.get(module), sys.modules.get(package_name)
    path = vars(loaded).get("__file__") if loaded else None
    package_file = vars(package).get("__file__") if package else None
    result = dict(
        module=module,
        qualified_name=function.__qualname__[:192],
        code_first_line=function.__code__.co_firstlineno,
        code_bytes_sha256=hashlib.sha256(function.__code__.co_code).hexdigest(),
        argument_names=list(
            function.__code__.co_varnames[
                : function.__code__.co_argcount + function.__code__.co_kwonlyargcount
            ]
        ),
        positional_count=function.__code__.co_argcount,
        keyword_only_count=function.__code__.co_kwonlyargcount,
        varargs=bool(function.__code__.co_flags & 4),
        varkwargs=bool(function.__code__.co_flags & 8),
        signature_scope="code argument layout; defaults not evaluated",
    )
    if not isinstance(path, str) or not isinstance(package_file, str) or not path.endswith(".py"):
        return dict(result, status="UNKNOWN_PUBLIC_SOURCE_PATH")
    root, real = os.path.dirname(os.path.realpath(package_file)), os.path.realpath(path)
    if (
        os.path.commonpath((root, real)) != root
        or function.__code__.co_filename != path
        or len(path) > 256
        or len(module) > 128
    ):
        return dict(result, status="UNKNOWN_SOURCE_ASSOCIATION")
    if module not in files:
        if len(files) >= 2:
            return dict(result, status="UNKNOWN_SOURCE_FILE_CAP")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
                return dict(result, status="UNKNOWN_SOURCE_SIZE_OR_TYPE")
            body = stream.read(1024**2 + 1)
            after = os.fstat(stream.fileno())

        def identity(info):
            return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

        if len(body) != before.st_size or identity(before) != identity(after):
            return dict(result, status="UNKNOWN_SOURCE_CHANGED")
        files[module] = dict(
            status="captured", path=path, bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
        )
    return dict(result, status="CAPTURED", source=files[module])


def observe_wire(request, frames):
    """After decoding, before original normalization; never emit payload/metadata."""
    if not _enabled() or not _installed:
        return
    rid = None
    try:
        rid = getattr(request, "rid", None)
        if (
            not isinstance(rid, str)
            or len(rid) > 512
            or not re.match(r"^smg-qwenvl-[0-9a-f]{32}(?:-|$)", rid)
        ):
            return
        items = getattr(getattr(request, "mm_inputs", None), "mm_items", None)
        if (
            type(items) is not list
            or len(items) not in (1, 3)
            or any(
                getattr(getattr(item, "modality", None), "name", None) != "IMAGE" for item in items
            )
        ):
            return
        if len(items) == 1:
            data = getattr(items[0], "model_specific_data", None)
            grid = data.get("image_grid_thw") if type(data) is dict else None
            # Shape metadata only: a batch of three image grids, not grid values.
            if tuple(getattr(grid, "shape", ())) != (3, 3):
                return
        with _lock:
            if rid in _traces or len(_traces) >= 3:
                return
            _traces[rid] = dict(bytes=0, active=True, visual_calls=0, events=0)
        wire = []
        if len(frames) > 19:
            wire = [{"status": "UNKNOWN_WIRE_FRAME_CAP"}]
        else:
            for index, frame in enumerate(frames[2:-1], 1):
                size = memoryview(frame).nbytes
                reason = _reserve(rid, size)
                wire.append(
                    dict(
                        index=index,
                        bytes=size,
                        status=reason or "CAPTURED_FULL_AUX_FRAME",
                        full_byte_sha256=None if reason else hashlib.sha256(frame).hexdigest(),
                    )
                )
        _emit(
            rid,
            "wire_decoded_before_normalization",
            auxiliary_frames=wire,
            decoded_raw_features=[_tensor(rid, getattr(item, "feature", None)) for item in items],
            wire_scope="auxiliary tensor frames only; request payload and metadata never emitted",
        )
    except Exception as exc:
        _emit(
            rid,
            "wire_decoded_before_normalization",
            status="UNKNOWN",
            error_type=type(exc).__name__,
        )


def bind_scheduler(request):
    if not _enabled() or not _installed:
        return
    rid = None
    try:
        rid = getattr(request, "rid", None)
        items = getattr(getattr(request, "multimodal_inputs", None), "mm_items", None)
        with _lock:
            if rid not in _traces or not _traces[rid]["active"]:
                return
            if type(items) is not list or len(items) != 3:
                _emit(rid, "scheduler_identity", status="UNKNOWN_ITEM_COUNT")
                return
            for index, item in enumerate(items):
                key = id(item)
                row = _items.get(key)
                if row is None or row[0]() is not item:
                    _items[key] = (weakref.ref(item), {rid: index})
                else:
                    if rid in row[1] and row[1][rid] != index:
                        raise ValueError("same item in multiple positions")
                    row[1][rid] = index
            _emit(
                rid,
                "scheduler_identity",
                status="BOUND_WEAK_IDENTITY_ONLY",
                ordered_features=[_tensor(rid, getattr(item, "feature", None)) for item in items],
            )
    except Exception as exc:
        _emit(rid, "scheduler_identity", status="UNKNOWN", error_type=type(exc).__name__)


def finish(rid):
    try:
        with _lock:
            if rid in _traces:
                _traces[rid]["active"] = False
            for key, row in list(_items.items()):
                row[1].pop(rid, None)
                if not row[1] or row[0]() is None:
                    del _items[key]
    except Exception:
        pass


def _matched(items):
    if type(items) is not list or len(items) != 3:
        return None
    with _lock:
        owners = []
        for item in items:
            row = _items.get(id(item))
            if row is None or row[0]() is not item or len(row[1]) != 1:
                return None
            owners.append(next(iter(row[1])))
        rid = owners[0]
        state = _traces.get(rid)
        if len(set(owners)) != 1 or state is None or not state["active"] or state["visual_calls"]:
            return None
        positions = [_items[id(item)][1][rid] for item in items]
        if sorted(positions) != [0, 1, 2]:
            return None
        state["visual_calls"] += 1
        _emit(
            rid,
            "visual_identity",
            status="MATCHED_EXACT_WEAK_ITEMS",
            scheduler_item_order=positions,
        )
        return rid


def install(scheduler):
    """Only the exact captured model/backend implementation; no class-wide hooks."""
    global _install_attempted, _installed
    if not _enabled() or _install_attempted:
        return
    _install_attempted = True
    changes = []
    try:
        path = getattr(getattr(scheduler, "server_args", None), "model_path", None)
        root = os.environ.get("ROUTER_LOCAL_MODEL_PATH", "")
        if path != _MODEL and (not root or path != os.path.join(root, _MODEL)):
            return
        worker = _field(scheduler, "tp_worker")
        runner = None
        for owner in (worker, _field(worker, "worker")):
            runner = _field(owner, "model_runner") or _field(owner, "_model_runner")
            if runner is not None:
                break
        model = _field(runner, "model")
        visual = _field(model, "visual")
        blocks = _field(visual, "blocks")
        modules = object.__getattribute__(blocks, "__dict__").get("_modules", {})
        attention = _field(modules.get("0"), "attn")
        backend = _field(attention, "attn_backend")
        if backend is None:
            backend = _field(attention, "qkv_backend")
        if (
            type(model).__name__ != "Qwen3VLForConditionalGeneration"
            or type(backend).__name__ != "VisionAiterAttention"
        ):
            raise ValueError("unsupported loaded model or backend")
        get_visual = model._get_visual_feature.__func__
        materialize = model._materialize_visual_items.__func__
        files = {}
        model_source = _source(get_visual, files)
        vision_source = _source(backend.forward, files)
        if (
            model_source.get("source", {}).get("sha256") != _QWEN_SHA
            or vision_source.get("source", {}).get("sha256") != _VISION_SHA
        ):
            raise ValueError("unknown captured consumer implementation")
        files = {}
        helper = get_visual.__globals__.get("materialize_multimodal_features")
        flash = backend.flash_attn_varlen_func
        facts = dict(materialize_helper=_source(helper, files), bound_aiter=_source(flash, files))
        compile_flag = _field(getattr(scheduler, "server_args", None), "enable_torch_compile")
        _emit(
            None,
            "numerical_source_metadata",
            metadata=dict(
                source_files=files,
                callables=facts,
                enable_torch_compile=compile_flag if type(compile_flag) is bool else "UNKNOWN",
            ),
        )
        if (
            _field(model, "use_data_parallel") is not False
            or compile_flag is not False
            or type(flash) is not types.FunctionType
        ):
            raise ValueError("DP, compiled/unknown compile setting or unsupported callable layout")
        model_ref = weakref.ref(model)

        @functools.wraps(get_visual)
        def visual_call(*args, **kwargs):
            rid = None
            try:
                rid = _matched(args[0] if args else kwargs.get("items")) if _enabled() else None
                if rid is None:
                    _emit(
                        None, "visual_identity", status="UNKNOWN_UNMATCHED_COPIED_MIXED_OR_REPEATED"
                    )
            except Exception:
                pass
            token = _context.set(dict(rid=rid, attention_used=False) if rid else None)
            try:
                return get_visual(model_ref(), *args, **kwargs)
            except BaseException:
                if rid:
                    _emit(rid, "visual_output", status="ORIGINAL_EXCEPTION")
                raise
            finally:
                _context.reset(token)

        @functools.wraps(materialize)
        def materialize_call(*args, **kwargs):
            scope = _context.get()
            try:
                result = materialize(model_ref(), *args, **kwargs)
            except BaseException:
                if scope is not None:
                    _emit(scope["rid"], "materialized_packed", status="ORIGINAL_EXCEPTION")
                raise
            try:
                if scope is not None:
                    packed = result[0] if type(result) is tuple else result
                    _emit(scope["rid"], "materialized_packed", tensor=_tensor(scope["rid"], packed))
            except Exception as exc:
                _emit(None, "materialized_packed", status="UNKNOWN", error_type=type(exc).__name__)
            return result

        @functools.wraps(flash)
        def flash_call(*args, **kwargs):
            scope = _context.get()
            watched = scope is not None and not scope["attention_used"]
            try:
                if watched:
                    scope["attention_used"] = True
                    if args:
                        _emit(
                            scope["rid"],
                            "first_vision_aiter_input",
                            status="UNKNOWN_POSITIONAL_CALL",
                        )
                    else:
                        facts = {
                            name: _tensor(
                                scope["rid"], kwargs.get(name), integers=name.startswith("cu_")
                            )
                            for name in ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k")
                        }
                        for name in ("max_seqlen_q", "max_seqlen_k", "softmax_scale"):
                            value = kwargs.get(name)
                            facts[name] = (
                                value
                                if value is None
                                or type(value) is int
                                or (type(value) is float and math.isfinite(value))
                                else "UNKNOWN_NON_SCALAR_OR_NONFINITE"
                            )
                        _emit(scope["rid"], "first_vision_aiter_input", tensors=facts)
            except Exception as exc:
                _emit(
                    None,
                    "first_vision_aiter_input",
                    status="UNKNOWN",
                    error_type=type(exc).__name__,
                )
            try:
                result = flash(*args, **kwargs)
            except BaseException:
                if watched:
                    _emit(scope["rid"], "first_vision_aiter_output", status="ORIGINAL_EXCEPTION")
                raise
            if watched:
                _emit(
                    scope["rid"], "first_vision_aiter_output", tensor=_tensor(scope["rid"], result)
                )
            return result

        for owner, name, replacement in (
            (model, "_get_visual_feature", visual_call),
            (model, "_materialize_visual_items", materialize_call),
            (backend, "flash_attn_varlen_func", flash_call),
        ):
            values = object.__getattribute__(owner, "__dict__")
            changes.append((owner, name, name in values, values.get(name)))
            object.__setattr__(owner, name, replacement)
        _installed = True
        versions = {}
        for name in ("torch", "sglang", "aiter"):
            loaded = sys.modules.get(name)
            version = vars(loaded).get("__version__") if loaded is not None else None
            versions[name] = version[:96] if type(version) is str else "UNKNOWN"
        _emit(
            None,
            "numerical_install",
            status="INSTALLED_EXACT_CAPTURED_CONSUMER",
            runtime_versions=versions,
            parallel_scope="this loaded process and first vision block only",
            limits=dict(
                traces=3,
                tensor_bytes=_TENSOR_BYTES,
                trace_bytes=_TRACE_BYTES,
                process_bytes=_PROCESS_BYTES,
                events=80,
                events_per_trace=20,
                log_record_bytes=8192,
            ),
        )
    except Exception as exc:
        for owner, name, existed, old in reversed(changes):
            if existed:
                object.__setattr__(owner, name, old)
            else:
                object.__delattr__(owner, name)
        _emit(
            None,
            "numerical_install",
            status="UNKNOWN_UNSUPPORTED_OR_UNAVAILABLE",
            error_type=type(exc).__name__,
        )
