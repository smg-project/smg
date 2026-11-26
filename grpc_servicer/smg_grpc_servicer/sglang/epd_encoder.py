"""Native encoder batch, HTTP receiver facade, and ownership behind Rust E control."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from urllib.parse import urlsplit

import grpc

ARTIFACT_KEY = "__smg_rust_encoder_artifact_v1"
MAX_RECEIVER_HTTP_BODY_BYTES = 256 * 1024
MAX_ABANDONED_RIDS = 4096
ABANDONED_TTL_SECONDS = 125.0
_ROUTES = {
    "scheduler_receive_meta_data": "metadata",
    "scheduler_receive_url": "register",
    "send": "send",
}
# Every status must belong to `http.HTTPStatus` (SGLang calls `HTTPStatus(int(code))`
# and non-standard codes like 499 crash Prefill) and stay in `engine_abort_status`'s set.
_HTTP_STATUS = {
    grpc.StatusCode.INVALID_ARGUMENT: HTTPStatus.BAD_REQUEST,
    grpc.StatusCode.CANCELLED: HTTPStatus.NOT_FOUND,
    grpc.StatusCode.OUT_OF_RANGE: HTTPStatus.BAD_REQUEST,
    grpc.StatusCode.ALREADY_EXISTS: HTTPStatus.BAD_REQUEST,
    grpc.StatusCode.RESOURCE_EXHAUSTED: HTTPStatus.TOO_MANY_REQUESTS,
    grpc.StatusCode.DEADLINE_EXCEEDED: HTTPStatus.GATEWAY_TIMEOUT,
    grpc.StatusCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    grpc.StatusCode.INTERNAL: HTTPStatus.INTERNAL_SERVER_ERROR,
}


def _err(exc):
    """Format a bounded single-line error message from an exception."""
    return str(exc)[:1024] or type(exc).__name__


async def run_uninterruptible(executor, fn, *args):
    """Run blocking work on an executor, retaining ownership across repeated cancellation."""
    work = asyncio.get_running_loop().run_in_executor(executor, fn, *args)
    cancelled = False
    while not work.done():
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            if not cancelled:
                raise
            break
    if cancelled:
        if not work.cancelled():
            work.exception()
        raise asyncio.CancelledError
    return work.result()


def _extract_artifact(request):
    """Read the serialized Rust artifact from top-level metadata or `mm_items[0]`."""
    raw = request.get(ARTIFACT_KEY)
    if isinstance(raw, bytes) and raw:
        return raw
    mm_items = request.get("mm_items")
    if isinstance(mm_items, (list, tuple)) and mm_items and isinstance(mm_items[0], dict):
        nested = mm_items[0].get(ARTIFACT_KEY)
        if isinstance(nested, bytes) and nested:
            return nested
    return None


def _build_stable_identity_namespace(encoder, processor_contract):
    """Build a host-, path-, and rank-independent 32-byte namespace for EPD item identities."""
    model_config = getattr(encoder, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    revision = getattr(model_config, "revision", None) or getattr(
        getattr(encoder, "server_args", None), "revision", None
    )
    model_type = getattr(encoder, "model_type", None) or getattr(hf_config, "model_type", None)
    emb_dtype = getattr(encoder, "_embedding_dtype", None) or getattr(model_config, "dtype", None)
    vision_cfg = getattr(hf_config, "vision_config", None)
    if vision_cfg is not None:
        if hasattr(vision_cfg, "to_dict"):
            vision_cfg = vision_cfg.to_dict()
        elif isinstance(vision_cfg, dict):
            vision_cfg = dict(vision_cfg)
        else:
            vision_cfg = {k: v for k, v in vars(vision_cfg).items() if not k.startswith("_")}
        vision_cfg.pop("_name_or_path", None)
        vision_cfg.pop("model_path", None)
    payload = json.dumps(
        {
            "v": "smg-epd-item-id-v1",
            "processor_contract": processor_contract,
            "revision": revision,
            "model_type": model_type,
            "embedding_dtype": str(emb_dtype) if emb_dtype is not None else None,
            "vision_config": vision_cfg,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _hash_materialized_item(stable_ns, modality_name, pixel_slice, grid_row, second_per_grid=None):
    """Compute a deterministic 32-byte content digest from preprocessed item features."""
    import torch

    flat = pixel_slice.contiguous().reshape(-1)
    hasher = hashlib.sha256(stable_ns)
    for part in (
        modality_name.encode("ascii"),
        str(pixel_slice.dtype).encode("ascii"),
        b",".join(str(int(v)).encode("ascii") for v in grid_row)
        + b";"
        + str(tuple(flat.shape)).encode("ascii"),
    ):
        hasher.update(len(part).to_bytes(4, "little"))
        hasher.update(part)
    if second_per_grid is not None:
        spg_bytes = memoryview(second_per_grid.contiguous().reshape(-1).view(torch.uint8).numpy())
        hasher.update(len(spg_bytes).to_bytes(4, "little"))
        hasher.update(spg_bytes)
    else:
        hasher.update(b"\x00\x00\x00\x00")
    hasher.update(memoryview(flat.view(torch.uint8).numpy()))
    return hasher.digest()


def _hash_batch_items(stable_ns, modality, pixel, grid_rows, get_num_patches, spg_tensor=None):
    """Hash every item slice in a materialized batch; return None if shapes mismatch."""
    if pixel.ndim == 0:
        return None
    total_patches = int(pixel.shape[0])
    offset = 0
    keys = []
    for row_idx, row in enumerate(grid_rows):
        patches = int(get_num_patches(row, modality))
        if patches <= 0 or offset + patches > total_patches:
            return None
        item_spg = None
        if spg_tensor is not None:
            item_spg = (
                spg_tensor[row_idx : row_idx + 1]
                if len(spg_tensor) == len(grid_rows)
                else spg_tensor
            )
        keys.append(
            _hash_materialized_item(
                stable_ns, modality.name, pixel[offset : offset + patches], row, item_spg
            )
        )
        offset += patches
    return keys if offset == total_patches else None


class RustPreprocessorBridge:
    """Use artifacts distributed by the native TP request IPC; never read media on a rank."""

    def __init__(
        self, original, max_artifact_bytes, *, stable_identity_ns=None, identity_sink=None
    ):
        """Wrap SGLang's encoder preprocessor to consume immutable Rust-produced artifacts."""
        self.original = original
        self.max_artifact_bytes = max_artifact_bytes
        self.stable_identity_ns = (
            stable_identity_ns
            if stable_identity_ns is not None
            else hashlib.sha256(b"smg-epd-item-id-v1:default").digest()
        )
        self.identity_sink = identity_sink

    def supports_modality(self, modality):
        """Return True only for IMAGE and VIDEO modalities supported by managed EPD."""
        return modality.name in ("IMAGE", "VIDEO")

    def get_num_patches(self, grid, modality):
        """Delegate patch-count calculation to the native preprocessor for supported modalities."""
        if not self.supports_modality(modality):
            raise ValueError("managed encoder supports image and video artifacts only")
        return self.original.get_num_patches(grid, modality)

    async def process_batch_mm_items(self, requests, modality):
        """Materialize a fused batch of Rust artifacts on the preprocessor executor."""
        return await run_uninterruptible(
            self.original.preproc_executor, self._materialize_batch, requests, modality
        )

    def _batch_token_counts(self, batch, grid, modality, is_video):
        """Resolve per-item embedding token counts from Rust placeholder ranges and grid rows."""
        if not is_video:
            if len(batch.mm_placeholders) != len(grid):
                raise ValueError("Rust token counts do not match grid items")
            return [p.length for p in batch.mm_placeholders]
        if len(batch.mm_placeholders) == len(grid):
            return [p.length for p in batch.mm_placeholders]
        if len(grid) == 1 and batch.mm_placeholders:
            return [sum(p.length for p in batch.mm_placeholders)]
        counts = [self.original.get_num_tokens(row, modality) for row in grid]
        if sum(counts) != sum(p.length for p in batch.mm_placeholders):
            raise ValueError("Rust token counts do not match grid items")
        return counts

    def _materialize_batch(self, requests, modality):
        """Merge Rust artifacts in native batch order without CPU decoding or normalization."""
        import torch
        from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessResult
        from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg

        if not self.supports_modality(modality):
            raise ValueError("managed encoder refuses non-image/video preprocessing")
        is_video = modality.name == "VIDEO"
        feature_key = "pixel_values_videos" if is_video else "pixel_values"
        grid_key = "video_grid_thw" if is_video else "image_grid_thw"
        get_patches = (
            self.original.get_num_patches
            if self.original is not None
            else (lambda row, _mod: int(row[0]) * int(row[1]) * int(row[2]))
        )
        pixels, grids, counts, items_per_req, second_per_grids, all_keys = [], [], [], [], [], []
        has_second_per_grid = None
        all_keys_valid = True
        for request in requests:
            raw = _extract_artifact(request)
            if not isinstance(raw, bytes) or not raw:
                raise ValueError("managed encoder is missing its immutable Rust artifact")
            if len(raw) > self.max_artifact_bytes:
                raise ValueError("managed native request exceeds artifact budget")
            artifact = sg.GenerateRequest.FromString(raw)
            if not artifact.HasField("mm_inputs"):
                raise ValueError("managed Rust artifact has no pixels")
            item_count = 0
            req_keys = []
            req_keys_valid = True
            for batch in [artifact.mm_inputs, *artifact.extra_mm_inputs]:
                if grid_key not in batch.model_specific_tensors:
                    raise ValueError("unsupported Rust artifact model fields")
                pixel = _tensor(batch.pixel_values, copy=False)
                grid = _tensor(batch.model_specific_tensors[grid_key])
                if grid.dtype != torch.int64 or grid.ndim != 2 or grid.shape[1] != 3:
                    raise ValueError("invalid Rust media grid")
                spg_tensor = None
                if is_video:
                    batch_has_spg = "video_second_per_grid" in batch.model_specific_tensors
                    if has_second_per_grid is None:
                        has_second_per_grid = batch_has_spg
                    elif batch_has_spg != has_second_per_grid:
                        raise ValueError("inconsistent Rust video timing metadata")
                    if batch_has_spg:
                        spg_tensor = _tensor(batch.model_specific_tensors["video_second_per_grid"])
                        second_per_grids.append(spg_tensor)
                counts.extend(self._batch_token_counts(batch, grid, modality, is_video))
                if req_keys_valid:
                    batch_keys = _hash_batch_items(
                        self.stable_identity_ns,
                        modality,
                        pixel,
                        grid.tolist(),
                        get_patches,
                        spg_tensor,
                    )
                    if batch_keys is None:
                        req_keys_valid = False
                    else:
                        req_keys.extend(batch_keys)
                pixels.append(pixel)
                grids.append(grid)
                item_count += len(grid)
            if item_count != len(request["mm_items"]):
                raise ValueError("Rust artifact lost a media occurrence")
            items_per_req.append(item_count)
            if req_keys_valid and len(req_keys) == item_count:
                req_id = request.get("req_id")
                if isinstance(req_id, str) and req_id and self.identity_sink is not None:
                    self.identity_sink[req_id] = [key.hex() for key in req_keys]
                all_keys.extend(req_keys)
            else:
                all_keys_valid = False
        grid = torch.cat(grids, dim=0)
        mm_inputs = {feature_key: torch.cat(pixels, dim=0), grid_key: grid}
        if second_per_grids:
            mm_inputs["second_per_grid_ts"] = torch.cat(second_per_grids, dim=0)
        result = EncoderPreprocessResult(mm_inputs=mm_inputs, grid_thw=grid, token_counts=counts)
        if all_keys_valid and len(all_keys) == len(counts):
            result._smg_item_keys = all_keys
        return result, items_per_req


def _tensor(value, *, copy=True):
    """Materialize an E-local CPU view, validating dtype and byte length."""
    import torch

    dtypes = {
        "float32": (torch.float32, 4),
        "float16": (torch.float16, 2),
        "bfloat16": (torch.bfloat16, 2),
        "int64": (torch.int64, 8),
    }
    if value.dtype not in dtypes:
        raise ValueError("unsupported managed artifact dtype")
    dtype, width = dtypes[value.dtype]
    data = value.data
    if not value.shape or math.prod(value.shape) * width != len(data):
        raise ValueError("invalid managed artifact byte length")
    return torch.frombuffer(bytearray(data) if copy else data, dtype=dtype).reshape(
        list(value.shape)
    )


def install_bridge(encoder, processor_contract, max_artifact_bytes):
    """Install before serving on every rank; preserve native forward, TP, cache and sends."""
    if any(
        not callable(getattr(encoder.preprocessor, name, None))
        for name in ("process_batch_mm_items", "get_num_patches", "supports_modality")
    ) or not callable(
        getattr(getattr(encoder.preprocessor, "preproc_executor", None), "submit", None)
    ):
        raise ValueError("native encoder preprocessor component contract changed")
    if not processor_contract or max_artifact_bytes <= 0:
        raise ValueError("managed Rust processor requires a contract and byte limit")
    sink = {} if getattr(encoder, "rank", 0) == 0 else None
    if sink is not None:
        encoder._smg_req_item_identities = sink
    encoder.preprocessor = RustPreprocessorBridge(
        encoder.preprocessor,
        max_artifact_bytes,
        stable_identity_ns=_build_stable_identity_namespace(encoder, processor_contract),
        identity_sink=sink,
    )
    if hasattr(encoder, "cache_namespace"):
        base_ns = encoder.cache_namespace or ""
        encoder.cache_namespace = (
            "sha256:"
            + hashlib.sha256((base_ns + "\0smg-rust\0" + processor_contract).encode()).hexdigest()
        )


def launch_managed_rank(server_args, schedule_path, dist_init_method, rank):
    """Load SMG plugins inside a spawned TP follower rank before launching SGLang's encoder."""
    from sglang.srt.plugins import load_plugins

    load_plugins()
    from sglang.srt.disaggregation.encoder.server import launch_encoder

    launch_encoder(server_args, schedule_path, dist_init_method, rank)


def _configure_meta_registry_sweep_timeout(min_ttl=ABANDONED_TTL_SECONDS):
    """Ensure SGLang's metadata sweep TTL stays >= min_ttl without altering send_timeout.

    SMG defaults SGLANG_ENCODER_SEND_TIMEOUT=15 for Mooncake to keep ZMQ/ACK send
    blocking under Rust's 30s CANCEL_RECEIPT_GRACE, while SGLang's MMEncoder.__init__
    copies send_timeout into meta_registry.sweep_timeout. Raising sweep_timeout here
    to at least ABANDONED_TTL_SECONDS (125s, covering Router's 120s prepared-layout
    window) prevents premature metadata eviction when Prefill stalls >15s while
    leaving encoder.send_timeout untouched at 15s.
    """
    try:
        from sglang.srt.disaggregation.encoder.server import meta_registry
    except ImportError:
        return None
    if meta_registry is None or not hasattr(meta_registry, "sweep_timeout"):
        return None
    current = meta_registry.sweep_timeout
    if isinstance(current, bool) or not isinstance(current, (int, float)) or math.isnan(current):
        target = float(min_ttl)
    else:
        target = max(float(current), float(min_ttl))
    meta_registry.sweep_timeout = target
    return target


def launch_runtime(server_args, max_active):
    """Launch SGLang's local encoder runtime with SMG's plugin rank loader and bounded queue."""
    from sglang.srt.disaggregation.encoder import runtime
    from sglang.srt.runtime_context import publish

    if server_args.dp_size != 1 or server_args.nnodes != 1:
        raise ValueError("managed E currently requires one host and one native DP unit")
    if max_active <= 0:
        raise ValueError("managed E requires a finite native queue")
    publish(server_args, role="encoder")
    original_target = runtime.launch_encoder
    runtime.launch_encoder = launch_managed_rank
    try:
        backend = runtime.launch_local_runtime(server_args)
    finally:
        runtime.launch_encoder = original_target
    _configure_meta_registry_sweep_timeout()
    backend.scheduler.pending_queue = asyncio.Queue(maxsize=max_active)
    return backend


class OwnedBatchSubmitter:
    """Enqueue once and await real batch completion; never cancel the shared batch future."""

    def __init__(self, scheduler):
        """Bind the submitter to SGLang's encoder batch scheduler."""
        self.scheduler = scheduler

    async def submit(self, request):
        """Enqueue ``request`` onto ``pending_queue`` and shield its completion future."""
        from sglang.srt.disaggregation.encoder.runtime import PendingRequest

        pending = PendingRequest(request, asyncio.get_running_loop())
        self.scheduler.pending_queue.put_nowait(pending)
        return await asyncio.shield(pending.future)


class Rejected(Exception):
    """Typed control-plane error carrying a gRPC status code mapped to an HTTP status."""

    def __init__(self, code, message):
        """Store gRPC status ``code`` alongside the human-readable error ``message``."""
        super().__init__(message)
        self.code = code


def require(condition, message, code=grpc.StatusCode.INVALID_ARGUMENT):
    """Raise :class:`Rejected` with ``code`` when ``condition`` is falsy."""
    if not condition:
        raise Rejected(code, message)


@dataclass
class NativeEntry:
    deadline: float
    rid: str
    prefill_host: str
    part_idx: int = 0
    abort_requested: bool = False
    closing: asyncio.Future = field(default_factory=asyncio.Future)
    delivered: asyncio.Event = field(default_factory=asyncio.Event)
    active: int = 0
    idle: asyncio.Event = field(default_factory=asyncio.Event)
    sends: dict = field(default_factory=dict)
    receivers: int | None = None
    destinations: set = field(default_factory=set)
    cleanup: asyncio.Task | None = None

    @contextmanager
    def retain(self, *, cancelling=False):
        """Track an active receiver/cleanup reference so release waits until all users drain."""
        if self.closing.done() and not cancelling:
            raise Rejected(grpc.StatusCode.CANCELLED, "managed part no longer accepts work")
        self.active += 1
        self.idle.clear()
        try:
            yield
        finally:
            self.active -= 1
            if not self.active:
                self.idle.set()


class NativeEncoder:
    """Owns native GPU batches, HTTP receiver calls, and explicit release receipts."""

    def __init__(self, config, backend):
        """Initialize entry/waiter registries for one managed encoder runtime."""
        self.config, self.backend = config, backend
        self.entries = {}
        self.waiters = {}
        self.abandoned = {}
        self.closed = False
        self.submitter = OwnedBatchSubmitter(backend.scheduler)

    def _wake_waiters(self, rid, reason):
        """Resolve all pending early-arrival receiver waiters for ``rid`` with ``reason``."""
        for waiter in self.waiters.pop(rid, ()):
            if not waiter.done():
                waiter.set_result(reason)

    def _mark_abandoned(self, rid):
        """Record a bounded-TTL tombstone so late receiver HTTP calls fail fast with CANCELLED."""
        if not isinstance(rid, str) or not 0 < len(rid) <= 512:
            return
        now = time.monotonic()
        for key in list(self.abandoned):
            if self.abandoned[key] <= now:
                del self.abandoned[key]
            else:
                break
        self.abandoned.pop(rid, None)
        while len(self.abandoned) >= MAX_ABANDONED_RIDS:
            self.abandoned.pop(next(iter(self.abandoned)))
        self.abandoned[rid] = now + ABANDONED_TTL_SECONDS
        self._wake_waiters(rid, "abandoned")

    def _require_not_abandoned(self, rid):
        """Reject ``rid`` with CANCELLED if a non-expired abandoned tombstone exists."""
        expiry = self.abandoned.get(rid)
        if expiry is not None and expiry <= time.monotonic():
            self.abandoned.pop(rid, None)
            expiry = None
        require(
            expiry is None,
            "native operation aborted before execution",
            grpc.StatusCode.CANCELLED,
        )

    async def _receiver_control(self, entry, operation):
        """Race a receiver coroutine against ``entry.closing`` so cancellation aborts waits promptly."""
        work = asyncio.create_task(operation)
        try:
            done, _ = await asyncio.wait((work, entry.closing), return_when=asyncio.FIRST_COMPLETED)
            require(
                work in done and not entry.abort_requested,
                "native receiver control cancelled",
                grpc.StatusCode.CANCELLED,
            )
            return await work
        finally:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)

    async def _wait_entry(self, rid, timeout=120.0):
        """Resolve an active entry by `rid`, waiting boundedly if P's HTTP call arrives before execute."""
        require(isinstance(rid, str) and 0 < len(rid) <= 512, "missing receiver request id")
        entry = self.entries.get(rid)
        if entry is None:
            require(not self.closed, "encoder shutting down", grpc.StatusCode.CANCELLED)
            self._require_not_abandoned(rid)
            waiter = asyncio.get_running_loop().create_future()
            self.waiters.setdefault(rid, []).append(waiter)
            try:
                reason = await asyncio.wait_for(waiter, timeout)
            except asyncio.TimeoutError as exc:  # noqa: UP041 - Python 3.10
                raise Rejected(
                    grpc.StatusCode.DEADLINE_EXCEEDED, "early native receiver expired"
                ) from exc
            finally:
                if (pending := self.waiters.get(rid)) and waiter in pending:
                    pending.remove(waiter)
                    if not pending:
                        del self.waiters[rid]
            require(reason != "closed", "encoder shutting down", grpc.StatusCode.CANCELLED)
            require(
                reason != "abandoned",
                "native operation aborted before execution",
                grpc.StatusCode.CANCELLED,
            )
            entry = self.entries.get(rid)
            if entry is None:
                self._require_not_abandoned(rid)
        require(entry is not None, "native operation is not registered")
        require(not entry.abort_requested, "native part closed", grpc.StatusCode.CANCELLED)
        require(
            time.monotonic() < entry.deadline,
            "native facade deadline expired",
            grpc.StatusCode.DEADLINE_EXCEEDED,
        )
        return entry

    def _validate_destination(self, kind, request, expected_host):
        """Verify that a receiver's ZMQ or Mooncake destination matches the frozen Prefill host."""
        if kind == "register":
            require(
                self.backend.encoder.transfer_backend == "zmq_to_scheduler",
                "unexpected ZMQ receiver",
            )
            destination = request.get("receive_url")
            require(isinstance(destination, str) and bool(destination), "invalid receiver address")
            addr = urlsplit("//" + destination)
            require(
                (addr.hostname or "").lower() == expected_host.lower()
                and bool(addr.port)
                and not (
                    addr.username or addr.password or addr.path or addr.query or addr.fragment
                ),
                "receiver differs from frozen P host",
            )
            return destination
        require(self.backend.encoder.transfer_backend == "mooncake", "unexpected Mooncake receiver")
        destination = (request.get("prefill_host"), request.get("embedding_port"))
        require(
            isinstance(destination[0], str)
            and destination[0].lower() == expected_host.lower()
            and type(destination[1]) is int
            and 0 < destination[1] <= 65535,
            "receiver differs from frozen P host",
        )
        return destination

    async def facade(self, kind, request):
        """Handle one P->E HTTP facade request (``metadata``, ``register``, or ``send``)."""
        entry = await self._wait_entry(request.get("req_id"))
        from sglang.srt.disaggregation.encoder import server as native
        from sglang.srt.disaggregation.encoder.runtime import send_staged_embedding

        if kind == "metadata":
            require(request.get("part_idx") == entry.part_idx, "invalid part index")
            with entry.retain():
                try:
                    metadata = await self._receiver_control(
                        entry,
                        asyncio.wait_for(
                            native.meta_registry.wait(entry.rid),
                            max(0, entry.deadline - time.monotonic()),
                        ),
                    )
                except asyncio.TimeoutError as exc:  # noqa: UP041 - Python 3.10
                    raise Rejected(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        f"native metadata expired for {entry.rid}",
                    ) from exc
            require(
                metadata and not metadata.get("error"),
                "native preprocessing failed",
                grpc.StatusCode.INTERNAL,
            )
            response = {"req_id": entry.rid, "part_idx": entry.part_idx, **metadata}
            identities_map = getattr(self.backend.encoder, "_smg_req_item_identities", None)
            if isinstance(identities_map, dict) and entry.rid in identities_map:
                response["item_identities"] = identities_map[entry.rid]
            return response

        count = request.get("receive_count")
        require(
            type(count) is int and 0 < count <= self.config["max_receivers"],
            "invalid native receive count",
        )
        require(entry.receivers in (None, count), "native receive count changed")
        destination = self._validate_destination(kind, request, entry.prefill_host)
        key = (kind, destination)
        if key in entry.sends:
            old_request, task = entry.sends[key]
            require(
                request == old_request,
                "conflicting duplicate native receive call",
                grpc.StatusCode.ALREADY_EXISTS,
            )
            return await asyncio.shield(task)
        require(
            len(entry.sends) < count,
            "too many native destinations",
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        entry.receivers = count

        async def run():
            with entry.retain():
                if kind == "register":
                    await self._receiver_control(
                        entry,
                        self.backend.encoder.register_embedding_destinations(
                            entry.rid, count, [destination]
                        ),
                    )
                else:
                    sent = await send_staged_embedding(
                        self.backend.encoder, request, release_without_count=False
                    )
                    require(sent, "native embedding unavailable", grpc.StatusCode.INTERNAL)
                    entry.destinations.add(destination)
                    if len(entry.destinations) == entry.receivers:
                        entry.delivered.set()
            return None

        task = asyncio.create_task(run())
        entry.sends[key] = (dict(request), task)
        return await asyncio.shield(task)

    def forget(self, rid):
        """Wake early receiver waiters when Rust fails a request before dispatching execute."""
        if rid not in self.entries:
            self._mark_abandoned(rid)

    def _wake_abandoned_sends(self, rid):
        """Break _wait_for_embedding stalls if a request is abandoned before encoding stages."""
        req_states = getattr(self.backend.encoder, "req_states", None)
        state = req_states.get(rid) if isinstance(req_states, dict) else None
        if state is not None and getattr(state, "active_encodes", 0) == 0:
            ready = getattr(state, "embedding_ready", None)
            if ready is not None and not ready.is_set():
                ready.set()

    async def cancel(self, rid):
        """Cancel an active or not-yet-arrived encoder request and wake abandoned send waiters."""
        entry = self.entries.get(rid)
        if entry is None:
            self._mark_abandoned(rid)
            return
        entry.abort_requested = True
        entry.closing.cancel()
        self._wake_abandoned_sends(rid)
        if entry.cleanup is None:
            with entry.retain(cancelling=True):
                await self.backend.encoder.abandon_request(rid)

    async def execute(self, command, artifact):
        """Run one Rust-dispatched encode command through SGLang's encoder pipeline and await delivery."""
        from sglang.srt.disaggregation.encoder.runtime import execute_encode_pipeline

        try:
            from sglang.srt.managers.req_types import Modality
        except ImportError:
            from sglang.srt.managers.schedule_batch import Modality

        rid = command["rid"]
        require(rid and rid not in self.entries, "duplicate or empty native operation")
        remaining = command["remaining_ms"] / 1000
        require(0 < remaining <= 122, "invalid native deadline")
        modality_str = command.get("modality", "image")
        num_parts, part_idx = command.get("num_parts", 1), command.get("part_idx", 0)
        item_count = command.get("item_count", command.get("image_count", 0))
        require(type(num_parts) is int and 1 <= num_parts <= 2, "invalid native part count")
        require(type(part_idx) is int and 0 <= part_idx < num_parts, "invalid native part index")
        require(
            isinstance(modality_str, str) and modality_str.lower() in ("image", "video"),
            "unsupported native modality",
        )
        modality = (
            Modality.from_str(modality_str)
            if hasattr(Modality, "from_str")
            else getattr(Modality, modality_str.upper())
        )
        entry = NativeEntry(
            time.monotonic() + remaining,
            rid,
            prefill_host=command["prefill_host"],
            part_idx=part_idx,
        )
        self.abandoned.pop(rid, None)
        self.entries[rid] = entry
        self._wake_waiters(rid, "registered")
        error = None
        try:
            tokens = command["embedding_tokens"]
            require(type(tokens) is int and tokens > 0, "invalid embedding token count")
            size = (
                tokens
                * self.backend.encoder._embedding_dims[modality]
                * self.backend.encoder._element_size
            )
            require(size <= self.config["max_staging_bytes_per_request"], "staging budget exceeded")
            require(type(item_count) is int and item_count > 0, "invalid native item count")
            native = {
                "req_id": rid,
                "num_parts": num_parts,
                "part_idx": part_idx,
                "modality": modality_str,
                "mm_items": [{ARTIFACT_KEY: artifact}, *([None] * (item_count - 1))],
            }
            send_sockets = getattr(self.backend, "send_sockets", None)
            pipeline_kwargs = {"send_sockets": send_sockets} if send_sockets is not None else {}
            with entry.retain():
                await execute_encode_pipeline(
                    self.backend.encoder, self.submitter, native, **pipeline_kwargs
                )
            if self.backend.encoder.transfer_backend == "mooncake":
                try:
                    await self._receiver_control(
                        entry,
                        asyncio.wait_for(
                            entry.delivered.wait(),
                            max(0, entry.deadline - time.monotonic()),
                        ),
                    )
                except asyncio.TimeoutError as exc:  # noqa: UP041 - Python 3.10
                    raise Rejected(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        f"native embedding delivery expired for {rid}",
                    ) from exc
            require(not entry.abort_requested, "native operation cancelled")
        except (Exception, asyncio.CancelledError) as exc:
            error = _err(exc)
        return await self._finish(entry, error)

    async def _finish(self, entry, error):
        """Drain active facade users, release native staged memory, and emit a release receipt."""
        from sglang.srt.disaggregation.encoder.server import meta_registry

        rid = entry.rid
        entry.cleanup = asyncio.current_task()
        entry.closing.cancel()
        if error:
            try:
                await meta_registry.publish(rid, 0, 0, 0, error=error)
            except Exception as exc:
                error = _err(exc)
        try:
            self._wake_abandoned_sends(rid)
            await self.backend.encoder.abandon_request(rid)
            if entry.active:
                await entry.idle.wait()
            await self.backend.encoder.release_request(rid)
            self.backend.encoder.clear_abandoned_request(rid)
        except Exception as exc:
            return {"released": False, "error": _err(exc)}
        finally:
            identities_map = getattr(self.backend.encoder, "_smg_req_item_identities", None)
            if isinstance(identities_map, dict):
                identities_map.pop(rid, None)
        entry.sends.clear()
        del self.entries[rid]
        return {"released": True, "error": error}

    def close(self):
        """Mark the encoder shutting down and wake all pending receiver waiters."""
        self.closed = True
        for rid in list(self.waiters):
            self._wake_waiters(rid, "closed")
        self.waiters.clear()

    async def dispatch(self, command, artifact):
        """Route one IPC command from Rust (`execute`, `cancel`, or `forget`)."""
        operation = command["operation"]
        if operation == "execute":
            return await self.execute(command, artifact)
        if operation == "cancel":
            await self.cancel(command["rid"])
            return None
        if operation == "forget":
            self.forget(command["rid"])
            return None
        raise ValueError("unknown native encoder operation")


async def _start_http_facade(owner, config):
    """Serve P->E receiver HTTP routes directly on the native asyncio loop."""
    from aiohttp import web

    host, port_str = config["http_listen"].rsplit(":", 1)
    host = host.strip("[]")
    max_body = min(
        config.get("max_http_body_bytes", MAX_RECEIVER_HTTP_BODY_BYTES),
        MAX_RECEIVER_HTTP_BODY_BYTES,
    )
    slots = asyncio.Semaphore(config.get("max_http_connections", 528))

    async def handle_route(request):
        kind = _ROUTES.get(request.match_info["route"])
        if kind is None:
            return web.json_response(
                {"message": "unsupported native receiver operation"}, status=404
            )
        if slots.locked():
            return web.json_response({"message": "native HTTP capacity full"}, status=429)
        async with slots:
            try:
                raw = await asyncio.wait_for(request.read(), 10.0)
            except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
                return web.json_response({"message": "control request body timeout"}, status=408)
            if len(raw) > max_body:
                return web.json_response({"message": "control request too large"}, status=413)
            try:
                payload = json.loads(raw)
                require(isinstance(payload, dict), "invalid receiver request payload")
                result = await owner.facade(kind, payload)
                return web.json_response({"status": "ok"} if result is None else result)
            except Rejected as err:
                return web.json_response(
                    {"message": str(err)},
                    status=int(_HTTP_STATUS.get(err.code, HTTPStatus.BAD_REQUEST)),
                )
            except (json.JSONDecodeError, ValueError) as err:
                return web.json_response({"message": _err(err)}, status=400)
            except Exception as err:
                return web.json_response({"message": _err(err)}, status=500)

    app = web.Application(client_max_size=max_body + 1024)
    app.router.add_post("/{route}", handle_route)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, int(port_str)).start()
    return runner


async def serve_native(config, backend):
    """Run Python HTTP receiver facade and Rust execute/cancel IPC loop together."""
    import zmq
    import zmq.asyncio

    owner = NativeEncoder(config, backend)
    runner = await _start_http_facade(owner, config) if config.get("http_listen") else None
    context = zmq.asyncio.Context()
    commands, events = context.socket(zmq.PULL), context.socket(zmq.PUSH)
    commands.setsockopt(zmq.LINGER, 0)
    events.setsockopt(zmq.LINGER, 0)
    commands.connect(config["native_commands"])
    events.connect(config["native_events"])
    tasks = set()
    stopping = asyncio.get_running_loop().create_future()

    async def publish(reply):
        if stopping.done():
            return
        sending = events.send_json(reply)
        if sending.done():
            await sending
            return
        try:
            await asyncio.wait((sending, stopping), return_when=asyncio.FIRST_COMPLETED)
            if sending.done():
                await sending
        finally:
            sending.cancel()

    async def handle(frames):
        if stopping.done():
            frames.clear()
            return
        command = json.loads(frames[0])
        sequence = command.get("sequence")
        reply = {"sequence": sequence} if sequence is not None else None
        try:
            result = await owner.dispatch(command, frames[1] if len(frames) == 2 else b"")
            if reply is not None:
                reply["result"] = result
        except (Exception, asyncio.CancelledError) as error:
            if reply is not None:
                reply["error"] = _err(error)
                if isinstance(error, Rejected):
                    reply["error_code"] = error.code.value[0]
                elif isinstance(error, asyncio.CancelledError):
                    reply["error_code"] = 1
                elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
                    reply["error_code"] = 4
        frames.clear()
        if reply is not None:
            await publish(reply)

    try:
        # Signal Rust readiness only after HTTP facade is listening and both IPC sockets are connected.
        await publish({"ready": True})
        while True:
            frames = await commands.recv_multipart()
            if not 1 <= len(frames) <= 2 or len(frames[0]) > config["max_control_bytes"]:
                raise ValueError("invalid native encoder control frames")
            if len(frames) == 2 and len(frames[1]) > config["max_artifact_bytes"]:
                raise ValueError("native encoder artifact too large")
            task = asyncio.create_task(handle(frames))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    finally:
        stopping.set_result(None)
        owner.close()
        deadline = asyncio.get_running_loop().time() + config.get("shutdown_timeout_seconds", 30)

        def remaining():
            return max(0.0, deadline - asyncio.get_running_loop().time())

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(owner.cancel(rid) for rid in list(owner.entries)), return_exceptions=True
                ),
                timeout=remaining(),
            )
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=remaining()
            )
        except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
            for task in list(tasks):
                task.cancel()
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=0.5
                    )
                except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
                    pass
        finally:
            if runner is not None:
                try:
                    await asyncio.wait_for(runner.cleanup(), timeout=max(0.05, remaining()))
                except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
                    pass
            commands.close()
            events.close()
            context.term()
