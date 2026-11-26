"""Native encoder batch, HTTP receiver facade, and ownership behind Rust E control."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import mmap
import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from types import SimpleNamespace
from urllib.parse import urlsplit

import grpc

ARTIFACT_KEY = "__smg_rust_encoder_artifact_v1"
ARTIFACT_V2_KEY = "__smg_rust_encoder_artifact_v2"
_EPD_SHM_PREFIX = "smg-epd-"
_EPD_SHM_DIR = "/dev/shm"
MAX_RECEIVER_HTTP_BODY_BYTES = 256 * 1024
MAX_ABANDONED_RIDS = 4096
ABANDONED_TTL_SECONDS = 125.0
_DEFAULT_ITEM_CACHE_MB = 16384
ITEM_CACHE_MB_ENV = "SMG_SGLANG_ENCODER_ITEM_CACHE_MB"
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


def _extract_artifact_v2(request):
    """Read the compact/cache-hit v2 Rust artifact from top-level metadata or `mm_items[0]`."""
    raw = request.get(ARTIFACT_V2_KEY)
    if isinstance(raw, dict):
        return raw
    mm_items = request.get("mm_items")
    if isinstance(mm_items, (list, tuple)) and mm_items and isinstance(mm_items[0], dict):
        nested = mm_items[0].get(ARTIFACT_V2_KEY)
        if isinstance(nested, dict):
            return nested
    return None


def _validate_shm_path(name):
    """Validate an EPD `/dev/shm` segment name and return its full path."""
    if (
        not isinstance(name, str)
        or not name.startswith(_EPD_SHM_PREFIX)
        or "/" in name
        or "\x00" in name
        or name in (".", "..")
    ):
        raise ValueError("invalid managed encoder SHM segment name")
    return os.path.join(_EPD_SHM_DIR, name)


def _unlink_shm_best_effort(name):
    """Best-effort unlink of a validated EPD `/dev/shm` segment."""
    if not isinstance(name, str) or not name.startswith(_EPD_SHM_PREFIX):
        return
    try:
        path = _validate_shm_path(name)
        os.unlink(path)
    except (ValueError, OSError):
        pass


def _open_shm_mmap(name, nbytes, *, unlink_immediately):
    """Map a Rust-produced `/dev/shm` pixel segment zero-copy with COW writability."""
    path = _validate_shm_path(name)
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("invalid managed encoder SHM byte length")
    if nbytes == 0:
        if unlink_immediately:
            _unlink_shm_best_effort(name)
        return b""
    fd = os.open(path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size < nbytes:
            raise ValueError("managed encoder SHM segment truncated")
        mm = mmap.mmap(fd, nbytes, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)
    if unlink_immediately:
        _unlink_shm_best_effort(name)
    return mm


def _derive_item_key_from_rust_hash(
    stable_ns, modality_name, rust_hash_hex, grid_row, second_per_grid=None
):
    """Derive a deterministic 32-byte item key from Rust's raw-content BLAKE3 hash (<1 us)."""
    import torch

    hasher = hashlib.sha256(stable_ns)
    hasher.update(b"rust-v2\0")
    for part in (
        modality_name.encode("ascii"),
        rust_hash_hex.encode("ascii"),
        b",".join(str(int(v)).encode("ascii") for v in grid_row),
    ):
        hasher.update(len(part).to_bytes(4, "little"))
        hasher.update(part)
    if second_per_grid is not None:
        spg_bytes = memoryview(second_per_grid.contiguous().reshape(-1).view(torch.uint8).numpy())
        hasher.update(len(spg_bytes).to_bytes(4, "little"))
        hasher.update(spg_bytes)
    else:
        hasher.update(b"\x00\x00\x00\x00")
    return hasher.digest()


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
        self,
        original,
        max_artifact_bytes,
        *,
        stable_identity_ns=None,
        identity_sink=None,
        tp_size=1,
    ):
        self.original = original
        self.max_artifact_bytes = max_artifact_bytes
        self.stable_identity_ns = (
            stable_identity_ns
            if stable_identity_ns is not None
            else hashlib.sha256(b"smg-epd-item-id-v1:default").digest()
        )
        self.identity_sink = identity_sink
        self.tp_size = max(1, int(tp_size))

    def supports_modality(self, modality):
        return modality.name in ("IMAGE", "VIDEO")

    def get_num_patches(self, grid, modality):
        if not self.supports_modality(modality):
            raise ValueError("managed encoder supports image and video artifacts only")
        return self.original.get_num_patches(grid, modality)

    async def process_batch_mm_items(self, requests, modality):
        return await run_uninterruptible(
            self.original.preproc_executor, self._materialize_batch, requests, modality
        )

    def _batch_token_counts_from_lengths(self, lengths, grid, modality, is_video):
        """Derive per-item embedding token counts from Rust placeholder lengths and grid rows."""
        if not is_video:
            if len(lengths) != len(grid):
                raise ValueError("Rust token counts do not match grid items")
            return [int(length) for length in lengths]
        if len(lengths) == len(grid):
            return [int(length) for length in lengths]
        if len(grid) == 1 and lengths:
            return [sum(int(length) for length in lengths)]
        counts = [self.original.get_num_tokens(row, modality) for row in grid]
        if sum(counts) != sum(int(length) for length in lengths):
            raise ValueError("Rust token counts do not match grid items")
        return counts

    def _batch_token_counts(self, batch, grid, modality, is_video):
        """Extract per-item token counts from a legacy protobuf multimodal batch."""
        return self._batch_token_counts_from_lengths(
            [p.length for p in batch.mm_placeholders], grid, modality, is_video
        )

    def _materialize_batch(self, requests, modality):
        """Merge Rust artifacts in native batch order without CPU decoding or normalization."""
        import torch
        from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessResult

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
        need_legacy_keys = self.identity_sink is not None or _item_cache_budget_bytes() > 0
        pixels, grids, counts, items_per_req, second_per_grids, all_keys = [], [], [], [], [], []
        pinned_embs = []
        pixel_item_indices = []
        has_cache_hit = False
        has_second_per_grid = None
        total_bytes = 0
        all_keys_valid = True
        total_item_offset = 0
        sg = None

        for request in requests:
            art_v2 = _extract_artifact_v2(request)
            if art_v2 is not None:
                kind = art_v2.get("kind")
                if kind == "cache_hit":
                    if is_video:
                        raise ValueError("cache_hit artifact is supported for images only")
                    item_keys_hex = art_v2.get("item_keys") or []
                    grid_rows = art_v2.get("grid_thw") or []
                    token_counts = art_v2.get("token_counts") or []
                    req_pinned = art_v2.get("pinned_embeddings") or []
                    item_count = len(grid_rows)
                    if (
                        item_count != len(request["mm_items"])
                        or len(item_keys_hex) != item_count
                        or len(token_counts) != item_count
                        or len(req_pinned) != item_count
                    ):
                        raise ValueError("invalid managed cache_hit artifact layout")
                    grid = torch.tensor(grid_rows, dtype=torch.int64)
                    if grid.ndim != 2 or grid.shape[1] != 3:
                        raise ValueError("invalid Rust media grid")
                    req_keys = [bytes.fromhex(k) for k in item_keys_hex]
                    grids.append(grid)
                    counts.extend(int(c) for c in token_counts)
                    pinned_embs.extend(req_pinned)
                    items_per_req.append(item_count)
                    req_id = request.get("req_id")
                    if isinstance(req_id, str) and req_id and self.identity_sink is not None:
                        self.identity_sink[req_id] = list(item_keys_hex)
                    all_keys.extend(req_keys)
                    has_cache_hit = True
                    total_item_offset += item_count
                    continue

                if kind != "compact":
                    raise ValueError(f"unsupported managed artifact_v2 kind: {kind!r}")

                batches = art_v2.get("batches") or []
                if not batches:
                    raise ValueError("managed compact artifact has no batches")
                shm_name = art_v2.get("shm_name")
                if shm_name is not None:
                    shm_nbytes = art_v2.get("shm_nbytes", 0)
                    total_bytes += shm_nbytes
                    if total_bytes > self.max_artifact_bytes:
                        raise ValueError("managed native batch exceeds artifact budget")
                    source_buf = _open_shm_mmap(
                        shm_name, shm_nbytes, unlink_immediately=(self.tp_size == 1)
                    )
                    copy_pixels = False
                else:
                    source_buf = art_v2.get("inline_pixels", b"")
                    if not isinstance(source_buf, (bytes, bytearray, memoryview)):
                        raise ValueError("managed compact artifact missing inline pixel bytes")
                    total_bytes += len(source_buf)
                    if total_bytes > self.max_artifact_bytes:
                        raise ValueError("managed native batch exceeds artifact budget")
                    copy_pixels = True

                item_count = 0
                req_keys = []
                req_keys_valid = True
                for batch in batches:
                    pixel = _tensor_from_buffer(
                        source_buf,
                        batch["pixel_dtype"],
                        batch["pixel_shape"],
                        offset=batch["pixel_offset"],
                        nbytes=batch["pixel_nbytes"],
                        copy=copy_pixels,
                    )
                    grid_rows = batch["grid_thw"]
                    grid = torch.tensor(grid_rows, dtype=torch.int64)
                    if grid.dtype != torch.int64 or grid.ndim != 2 or grid.shape[1] != 3:
                        raise ValueError("invalid Rust media grid")
                    spg_tensor = None
                    if is_video:
                        batch_spg = batch.get("second_per_grid_ts")
                        batch_has_spg = batch_spg is not None
                        if has_second_per_grid is None:
                            has_second_per_grid = batch_has_spg
                        elif batch_has_spg != has_second_per_grid:
                            raise ValueError("inconsistent Rust video timing metadata")
                        if batch_has_spg:
                            spg_shape = batch.get("second_per_grid_shape") or [len(batch_spg)]
                            spg_tensor = torch.tensor(batch_spg, dtype=torch.float32).reshape(
                                spg_shape
                            )
                            second_per_grids.append(spg_tensor)
                    counts.extend(
                        self._batch_token_counts_from_lengths(
                            batch["placeholder_lengths"], grid, modality, is_video
                        )
                    )
                    item_hashes = batch.get("item_hashes") or []
                    if req_keys_valid and len(item_hashes) == len(grid_rows):
                        for row_idx, (rust_hash, row) in enumerate(zip(item_hashes, grid_rows)):
                            item_spg = None
                            if spg_tensor is not None:
                                item_spg = (
                                    spg_tensor[row_idx : row_idx + 1]
                                    if len(spg_tensor) == len(grid_rows)
                                    else spg_tensor
                                )
                            req_keys.append(
                                _derive_item_key_from_rust_hash(
                                    self.stable_identity_ns,
                                    modality.name,
                                    rust_hash,
                                    row,
                                    item_spg,
                                )
                            )
                    elif req_keys_valid and need_legacy_keys:
                        batch_keys = _hash_batch_items(
                            self.stable_identity_ns,
                            modality,
                            pixel,
                            grid_rows,
                            get_patches,
                            spg_tensor,
                        )
                        if batch_keys is None:
                            req_keys_valid = False
                        else:
                            req_keys.extend(batch_keys)
                    else:
                        req_keys_valid = False
                    pixels.append(pixel)
                    grids.append(grid)
                    item_count += len(grid)
                if item_count != len(request["mm_items"]):
                    raise ValueError("Rust artifact lost a media occurrence")
                items_per_req.append(item_count)
                pinned_embs.extend([None] * item_count)
                pixel_item_indices.extend(range(total_item_offset, total_item_offset + item_count))
                total_item_offset += item_count
                if req_keys_valid and len(req_keys) == item_count:
                    req_id = request.get("req_id")
                    if isinstance(req_id, str) and req_id and self.identity_sink is not None:
                        self.identity_sink[req_id] = [key.hex() for key in req_keys]
                    all_keys.extend(req_keys)
                else:
                    all_keys_valid = False
                continue

            # Legacy protobuf artifact path (retained for backward compatibility and unit tests).
            if sg is None:
                from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg

            raw = _extract_artifact(request)
            if not isinstance(raw, bytes) or not raw:
                raise ValueError("managed encoder is missing its immutable Rust artifact")
            total_bytes += len(raw)
            if total_bytes > self.max_artifact_bytes:
                raise ValueError("managed native batch exceeds artifact budget")
            artifact = sg.GenerateRequest.FromString(raw)
            if not artifact.HasField("mm_inputs"):
                raise ValueError("managed Rust artifact has no pixels")
            item_count = 0
            req_keys = []
            req_keys_valid = need_legacy_keys
            legacy_batches = [artifact.mm_inputs, *artifact.extra_mm_inputs]
            copy_single_legacy = len(requests) == 1 and len(legacy_batches) == 1
            for batch in legacy_batches:
                if grid_key not in batch.model_specific_tensors:
                    raise ValueError("unsupported Rust artifact model fields")
                pixel = _tensor(batch.pixel_values, copy=copy_single_legacy)
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
            pinned_embs.extend([None] * item_count)
            pixel_item_indices.extend(range(total_item_offset, total_item_offset + item_count))
            total_item_offset += item_count
            if req_keys_valid and len(req_keys) == item_count:
                req_id = request.get("req_id")
                if isinstance(req_id, str) and req_id and self.identity_sink is not None:
                    self.identity_sink[req_id] = [key.hex() for key in req_keys]
                all_keys.extend(req_keys)
            else:
                all_keys_valid = False

        grid = grids[0] if len(grids) == 1 else torch.cat(grids, dim=0)
        if pixels:
            pixel_tensor = pixels[0] if len(pixels) == 1 else torch.cat(pixels, dim=0)
        else:
            pixel_tensor = torch.empty((0,), dtype=torch.bfloat16)
        mm_inputs = {feature_key: pixel_tensor, grid_key: grid}
        if second_per_grids:
            mm_inputs["second_per_grid_ts"] = (
                second_per_grids[0]
                if len(second_per_grids) == 1
                else torch.cat(second_per_grids, dim=0)
            )
        result = EncoderPreprocessResult(mm_inputs=mm_inputs, grid_thw=grid, token_counts=counts)
        if all_keys_valid and len(all_keys) == len(counts):
            result._smg_item_keys = all_keys
        if has_cache_hit:
            result._smg_pinned_embeddings = pinned_embs
            if pixels:
                result._smg_pixel_item_indices = pixel_item_indices
        return result, items_per_req


def _item_cache_budget_bytes():
    """Return configured per-item encoder cache budget in bytes, or 0 when disabled."""
    raw = os.environ.get(ITEM_CACHE_MB_ENV, "").strip()
    try:
        mb = int(raw) if raw else _DEFAULT_ITEM_CACHE_MB
    except ValueError:
        mb = _DEFAULT_ITEM_CACHE_MB
    return max(0, mb) * 1024 * 1024


class PerItemEmbeddingCache:
    """Thread-safe LRU cache keyed by per-item preprocessed image features."""

    def __init__(self, max_bytes):
        """Initialize the LRU embedding cache with a byte capacity cap."""
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self._pending_evictions = []
        self.hits = 0
        self.misses = 0

    def __len__(self):
        """Return the number of cached item embeddings."""
        with self._lock:
            return len(self._entries)

    def contains(self, key):
        """Return whether `key` is currently resident in the cache."""
        with self._lock:
            return key in self._entries

    def get(self, key):
        """Look up a cached item embedding and mark it most-recently used."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key, tensor):
        """Insert a tensor into the cache, evicting LRU entries as needed, and return eviction count."""
        if self.max_bytes <= 0:
            return 0
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        if nbytes <= 0 or nbytes > self.max_bytes:
            return 0
        evictions = 0
        with self._lock:
            prev = self._entries.pop(key, None)
            if prev is not None:
                self.current_bytes -= int(prev.numel()) * int(prev.element_size())
            while self.current_bytes + nbytes > self.max_bytes and self._entries:
                evicted_key, evicted = self._entries.popitem(last=False)
                self.current_bytes -= int(evicted.numel()) * int(evicted.element_size())
                if isinstance(evicted_key, (bytes, bytearray)):
                    self._pending_evictions.append(evicted_key.hex())
                evictions += 1
            self._entries[key] = tensor
            self.current_bytes += nbytes
        return evictions

    def drain_evictions(self):
        """Return and clear hex-encoded keys evicted since the last drain."""
        with self._lock:
            if not self._pending_evictions:
                return []
            drained = self._pending_evictions
            self._pending_evictions = []
            return drained

    def clear(self):
        """Evict all cached entries and queue their keys for Rust pre-decode cache feedback."""
        with self._lock:
            for evicted_key in self._entries:
                if isinstance(evicted_key, (bytes, bytearray)):
                    self._pending_evictions.append(evicted_key.hex())
            self._entries.clear()
            self.current_bytes = 0


def _compute_image_item_keys(encoder, ctx):
    """Compute per-item content keys from preprocessed pixel_values and grid_thw."""
    mm_inputs = ctx.preprocess_result.mm_inputs
    if not isinstance(mm_inputs, dict) or set(mm_inputs.keys()) != {
        "pixel_values",
        "image_grid_thw",
    }:
        return None
    pixel_values, grid_thw = mm_inputs["pixel_values"], mm_inputs["image_grid_thw"]
    if pixel_values is None or grid_thw is None:
        return None
    grid_rows = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
    if len(grid_rows) != ctx.num_items:
        return None
    pixel_cpu = pixel_values.detach()
    if pixel_cpu.is_cuda:
        pixel_cpu = pixel_cpu.cpu()
    ns = hashlib.sha256(
        (getattr(encoder, "cache_namespace", "") or "").encode("utf-8", errors="replace")
    ).digest()
    return _hash_batch_items(
        ns, ctx.modality, pixel_cpu.contiguous(), grid_rows, encoder.preprocessor.get_num_patches
    )


def _record_item_cache_metrics(ctx, cache, hit_count, hit_tokens, total_tokens, evictions=0):
    """Report per-item embedding cache hits, tokens, evictions, and size to SGLang metrics."""
    srv_mod = sys.modules.get("sglang.srt.disaggregation.encoder.server")
    metrics_collector = getattr(srv_mod, "encoder_metrics_collector", None) if srv_mod else None
    if metrics_collector is None or ctx.is_health_check:
        return
    if hasattr(metrics_collector, "record_cache_tokens"):
        metrics_collector.record_cache_tokens(hit_tokens, total_tokens, modality="image")
    if hasattr(metrics_collector, "record_cache_files"):
        metrics_collector.record_cache_files(hit_count, ctx.num_items, modality="image")
    if evictions > 0 and hasattr(metrics_collector, "inc_cache_evictions"):
        metrics_collector.inc_cache_evictions(modality="image", count=evictions)
    if cache is not None and hasattr(metrics_collector, "set_cache_state"):
        metrics_collector.set_cache_state(cache.current_bytes, len(cache))


async def cached_compute_direct_embedding(original, encoder, ctx, *, keep_on_gpu=False):
    """Cache per-item image embeddings across turns and fused multi-request batches."""
    import torch

    preproc = ctx.preprocess_result
    smg_keys = getattr(preproc, "_smg_item_keys", None)
    pinned_embs = getattr(preproc, "_smg_pinned_embeddings", None)
    has_smg_keys = smg_keys is not None or pinned_embs is not None

    if (
        (getattr(encoder, "_has_native_item_cache", False) and not has_smg_keys)
        or ctx.is_health_check
        or getattr(ctx.modality, "name", "") != "IMAGE"
        or getattr(encoder, "tp_size", 1) != 1
        or ctx.num_items <= 0
    ):
        return await original(encoder, ctx, keep_on_gpu=keep_on_gpu)

    budget_bytes = _item_cache_budget_bytes()
    if budget_bytes <= 0 and pinned_embs is None:
        return await original(encoder, ctx, keep_on_gpu=keep_on_gpu)

    cache = None
    if budget_bytes > 0:
        cache = getattr(encoder, "_smg_item_cache", None)
        if cache is None or cache.max_bytes != budget_bytes:
            cache = encoder._smg_item_cache = PerItemEmbeddingCache(budget_bytes)

    keys = smg_keys if smg_keys is not None else _compute_image_item_keys(encoder, ctx)
    if keys is None or len(keys) != len(preproc.token_counts):
        return await original(encoder, ctx, keep_on_gpu=keep_on_gpu)

    token_counts = [int(c) for c in preproc.token_counts]
    if pinned_embs is not None and len(pinned_embs) == len(keys):
        cached_items = list(pinned_embs)
        if cache is not None:
            for idx, existing in enumerate(cached_items):
                if existing is None:
                    cached_items[idx] = cache.get(keys[idx])
    elif cache is not None:
        cached_items = [cache.get(key) for key in keys]
    else:
        return await original(encoder, ctx, keep_on_gpu=keep_on_gpu)

    miss_indices = [i for i, cached in enumerate(cached_items) if cached is None]
    hit_count = len(keys) - len(miss_indices)
    hit_tokens = sum(token_counts[i] for i, cached in enumerate(cached_items) if cached is not None)
    total_tokens = sum(token_counts)

    if not miss_indices:
        if keep_on_gpu:
            model_dev = (
                next((p.device for p in encoder.model.parameters()), torch.device("cuda"))
                if hasattr(encoder, "model")
                else torch.device("cuda")
            )
            out_slices = [
                t if t.is_cuda else t.to(device=model_dev, non_blocking=False) for t in cached_items
            ]
        else:
            out_slices = [t.cpu() if t.is_cuda else t for t in cached_items]
        _record_item_cache_metrics(ctx, cache, hit_count, hit_tokens, total_tokens)
        return out_slices[0].clone() if len(out_slices) == 1 else torch.cat(out_slices, dim=0)

    pixel_item_indices = getattr(preproc, "_smg_pixel_item_indices", None)
    if pixel_item_indices is not None:
        index_pos = {orig_idx: pos for pos, orig_idx in enumerate(pixel_item_indices)}
        remapped_miss_indices = [index_pos[i] for i in miss_indices]
        compact_grid = preproc.grid_thw[pixel_item_indices]
        compact_mm_inputs = dict(preproc.mm_inputs)
        compact_mm_inputs["image_grid_thw"] = compact_grid
        build_preproc = SimpleNamespace(
            mm_inputs=compact_mm_inputs,
            grid_thw=compact_grid,
            token_counts=[token_counts[i] for i in pixel_item_indices],
        )
        build_indices = remapped_miss_indices
    else:
        build_preproc = preproc
        build_indices = miss_indices

    mm_items = encoder._build_model_mm_items(
        ctx.mm_feature, build_preproc, build_indices, ctx.modality
    )
    forward_start = time.perf_counter()
    with torch.inference_mode():
        miss_embedding = ctx.get_feature_fn(mm_items)
        if not keep_on_gpu:
            miss_embedding = miss_embedding.cpu()
    if len(miss_embedding.shape) != 2:
        miss_embedding = miss_embedding.reshape(-1, miss_embedding.shape[-1])
    if miss_embedding.shape[0] != sum(token_counts[i] for i in miss_indices):
        return await original(encoder, ctx, keep_on_gpu=keep_on_gpu)

    srv_mod = sys.modules.get("sglang.srt.disaggregation.encoder.server")
    metrics_collector = getattr(srv_mod, "encoder_metrics_collector", None) if srv_mod else None
    if metrics_collector is not None and not ctx.is_health_check:
        metrics_collector.observe_model_forward(
            time.perf_counter() - forward_start, modality="image"
        )

    target_device = miss_embedding.device
    evictions = 0
    for slice_tensor, item_idx in zip(
        torch.split(miss_embedding, [token_counts[i] for i in miss_indices], dim=0),
        miss_indices,
    ):
        if cache is not None:
            evictions += cache.put(keys[item_idx], slice_tensor.detach().clone().contiguous())
        cached_items[item_idx] = slice_tensor

    _record_item_cache_metrics(ctx, cache, hit_count, hit_tokens, total_tokens, evictions=evictions)

    assembled = [
        t if t.device == target_device else t.to(device=target_device) for t in cached_items
    ]
    if len(assembled) == 1:
        return assembled[0] if 0 in miss_indices else assembled[0].clone()
    return torch.cat(assembled, dim=0)


def _tensor_from_buffer(source_buf, dtype_name, shape, *, offset=0, nbytes=None, copy=True):
    """Materialize a CPU tensor from raw bytes or mmap, validating dtype and byte length."""
    import torch

    dtypes = {
        "float32": (torch.float32, 4),
        "float16": (torch.float16, 2),
        "bfloat16": (torch.bfloat16, 2),
        "int64": (torch.int64, 8),
    }
    if dtype_name not in dtypes:
        raise ValueError("unsupported managed artifact dtype")
    dtype, width = dtypes[dtype_name]
    numel = math.prod(shape) if shape else 0
    expected = numel * width
    if not shape or expected <= 0:
        raise ValueError("invalid managed artifact shape")
    if nbytes is None:
        nbytes = len(source_buf) - offset
    if expected != nbytes or offset < 0 or offset + nbytes > len(source_buf):
        raise ValueError("invalid managed artifact byte length")
    if copy:
        buf = bytearray(memoryview(source_buf)[offset : offset + nbytes])
        return torch.frombuffer(buf, dtype=dtype, count=numel).reshape(list(shape))
    return torch.frombuffer(source_buf, dtype=dtype, count=numel, offset=offset).reshape(
        list(shape)
    )


def _tensor(value, *, copy=True):
    """Materialize an E-local CPU view, validating dtype and byte length."""
    return _tensor_from_buffer(value.data, value.dtype, value.shape, copy=copy)


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
    tp_size = getattr(encoder, "tp_size", 1)
    if tp_size == 1:
        # Unify per-item embedding cache with Rust pre-decode index so SGLang's native cache
        # never allocates a second buffer or re-hashes unpacked pixel tensors.
        if getattr(encoder, "_item_embedding_cache", None) is not None:
            encoder._item_embedding_cache = None
        budget_bytes = _item_cache_budget_bytes()
        if budget_bytes > 0 and getattr(encoder, "_smg_item_cache", None) is None:
            encoder._smg_item_cache = PerItemEmbeddingCache(budget_bytes)
    sink = {} if getattr(encoder, "rank", 0) == 0 else None
    if sink is not None:
        encoder._smg_req_item_identities = sink
    encoder.preprocessor = RustPreprocessorBridge(
        encoder.preprocessor,
        max_artifact_bytes,
        stable_identity_ns=_build_stable_identity_namespace(encoder, processor_contract),
        identity_sink=sink,
        tp_size=tp_size,
    )
    if hasattr(encoder, "cache_namespace"):
        encoder.cache_namespace = (
            "sha256:"
            + hashlib.sha256(
                ((encoder.cache_namespace or "") + "\0smg-rust\0" + processor_contract).encode()
            ).hexdigest()
        )


def launch_managed_rank(server_args, schedule_path, dist_init_method, rank):
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
        self.scheduler = scheduler

    async def submit(self, request):
        from sglang.srt.disaggregation.encoder.runtime import PendingRequest

        pending = PendingRequest(request, asyncio.get_running_loop())
        self.scheduler.pending_queue.put_nowait(pending)
        return await asyncio.shield(pending.future)


class Rejected(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, message, code=grpc.StatusCode.INVALID_ARGUMENT):
    if not condition:
        raise Rejected(code, message)


@dataclass
class NativeEntry:
    deadline: float
    rid: str
    prefill_host: str
    part_idx: int = 0
    shm_name: str | None = None
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
        self.config, self.backend = config, backend
        self.entries = {}
        self.waiters = {}
        self.abandoned = {}
        self.closed = False
        self.submitter = OwnedBatchSubmitter(backend.scheduler)

    def _wake_waiters(self, rid, reason):
        for waiter in self.waiters.pop(rid, ()):
            if not waiter.done():
                waiter.set_result(reason)

    def _mark_abandoned(self, rid):
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
        max_parts = max(2, int(self.config.get("max_images", 1024)))
        require(type(num_parts) is int and 1 <= num_parts <= max_parts, "invalid native part count")
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

        artifact_v2 = command.get("artifact_v2")
        shm_name = None
        if isinstance(artifact_v2, dict):
            kind = artifact_v2.get("kind")
            if kind == "cache_hit":
                item_keys_hex = artifact_v2.get("item_keys") or []
                budget_bytes = _item_cache_budget_bytes()
                cache = (
                    getattr(self.backend.encoder, "_smg_item_cache", None)
                    if budget_bytes > 0 and getattr(self.backend.encoder, "tp_size", 1) == 1
                    else None
                )
                pinned = []
                missing = []
                cache_miss = False
                if cache is None or item_count <= 0 or len(item_keys_hex) != item_count:
                    missing = list(item_keys_hex)
                    cache_miss = True
                else:
                    for key_hex in item_keys_hex:
                        try:
                            key_bytes = bytes.fromhex(key_hex)
                        except (TypeError, ValueError):
                            missing.append(key_hex)
                            cache_miss = True
                            continue
                        tensor = cache.get(key_bytes)
                        if tensor is None:
                            missing.append(key_hex)
                            cache_miss = True
                        else:
                            pinned.append(tensor)
                if cache_miss:
                    evicted = list(missing)
                    if cache is not None:
                        evicted.extend(cache.drain_evictions())
                    return {"released": True, "cache_miss": True, "cache_evicted": evicted}
                artifact_v2 = {**artifact_v2, "pinned_embeddings": pinned}
            elif kind == "compact":
                shm_name = artifact_v2.get("shm_name")
                if shm_name is None:
                    artifact_v2 = {**artifact_v2, "inline_pixels": artifact}

        entry = NativeEntry(
            time.monotonic() + remaining,
            rid,
            prefill_host=command["prefill_host"],
            part_idx=part_idx,
            shm_name=shm_name,
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
            first_item = (
                {ARTIFACT_V2_KEY: artifact_v2}
                if isinstance(artifact_v2, dict)
                else {ARTIFACT_KEY: artifact}
            )
            native = {
                "req_id": rid,
                "num_parts": num_parts,
                "part_idx": part_idx,
                "modality": modality_str,
                "mm_items": [first_item, *([None] * (item_count - 1))],
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
        from sglang.srt.disaggregation.encoder.server import meta_registry

        rid = entry.rid
        entry.cleanup = asyncio.current_task()
        entry.closing.cancel()
        if error:
            try:
                await meta_registry.publish(rid, 0, 0, 0, error=error)
            except Exception as exc:
                error = _err(exc)
        cache_resident = None
        cache_evicted = None
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
            if entry.shm_name is not None:
                _unlink_shm_best_effort(entry.shm_name)
                entry.shm_name = None
            identities_map = getattr(self.backend.encoder, "_smg_req_item_identities", None)
            req_identities = (
                identities_map.pop(rid, None) if isinstance(identities_map, dict) else None
            )
            cache = getattr(self.backend.encoder, "_smg_item_cache", None)
            if cache is not None:
                cache_evicted = cache.drain_evictions() or None
                if error is None and req_identities and _item_cache_budget_bytes() > 0:
                    resident = [
                        k if cache.contains(bytes.fromhex(k)) else None for k in req_identities
                    ]
                    if any(resident):
                        cache_resident = resident
        entry.sends.clear()
        del self.entries[rid]
        receipt = {"released": True, "error": error}
        if cache_resident is not None:
            receipt["cache_resident"] = cache_resident
        if cache_evicted is not None:
            receipt["cache_evicted"] = cache_evicted
        return receipt

    def close(self):
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
    slots = asyncio.Semaphore(config.get("max_http_connections", 2320))

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
        finally:
            if runner is not None:
                try:
                    await asyncio.wait_for(runner.cleanup(), timeout=max(0.05, remaining()))
                except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
                    pass
            commands.close()
            events.close()
            context.term()
