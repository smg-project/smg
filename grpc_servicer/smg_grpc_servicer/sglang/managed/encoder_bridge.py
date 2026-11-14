"""E-local immutable Rust artifacts at the native preprocessor component boundary."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager

ARTIFACT_KEY = "__smg_rust_encoder_artifact_v1"


class RustPreprocessorBridge:
    """Use artifacts distributed by the native TP request IPC; never read media on a rank."""

    def __init__(self, original, max_artifact_bytes):
        self.original = original
        self.max_artifact_bytes = max_artifact_bytes

    def supports_modality(self, modality):
        """The initial adapter exposes images only, regardless of native model capabilities."""
        return modality.name == "IMAGE"

    def get_num_patches(self, grid, modality):
        """Keep the native model's patch slicing calculation."""
        if not self.supports_modality(modality):
            raise ValueError("managed encoder supports image artifacts only")
        return self.original.get_num_patches(grid, modality)

    async def process_batch_mm_items(self, requests, modality):
        """Keep artifact materialization off the control loop; native dispatch still awaits it."""
        # Reuse the native preprocessor owner and its pool; unrelated control/send tasks
        # must not scatter large CPU tensor allocations across the default executor.
        work = asyncio.get_running_loop().run_in_executor(
            self.original.preproc_executor, self._materialize_batch, requests, modality
        )
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            # CPU tensors and immutable artifacts remain owned until the worker exits.
            # In particular, shutdown must not release the enclosing native batch early.
            await work
            raise

    def _materialize_batch(self, requests, modality):
        """Merge Rust artifacts in native batch order without CPU decoding or normalization."""
        import torch
        from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessResult
        from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg

        if not self.supports_modality(modality):
            raise ValueError("managed encoder refuses non-image preprocessing")
        pixels, grids, counts, items_per_req = [], [], [], []
        total_bytes = 0
        for request in requests:
            raw = request.get(ARTIFACT_KEY)
            if not isinstance(raw, bytes) or not raw:
                raise ValueError("managed encoder is missing its immutable Rust artifact")
            total_bytes += len(raw)
            if total_bytes > self.max_artifact_bytes:
                raise ValueError("managed native batch exceeds artifact budget")
            artifact = sg.GenerateRequest.FromString(raw)
            if not artifact.HasField("mm_inputs"):
                raise ValueError("managed Rust artifact has no pixels")
            item_count = 0
            for batch in [artifact.mm_inputs, *artifact.extra_mm_inputs]:
                if (
                    batch.image_urls
                    or batch.video_urls
                    or batch.audio_urls
                    or batch.image_data
                    or batch.video_data
                    or batch.audio_data
                ):
                    raise ValueError(
                        "managed artifact must not contain executable media references"
                    )
                if set(batch.model_specific_tensors) != {"image_grid_thw", "patches_per_image"}:
                    raise ValueError("unsupported Rust artifact model fields")
                pixel = _tensor(batch.pixel_values)
                grid = _tensor(batch.model_specific_tensors["image_grid_thw"])
                if grid.dtype != torch.int64 or grid.ndim != 2 or grid.shape[1] != 3:
                    raise ValueError("invalid Rust image grid")
                patches = _tensor(batch.model_specific_tensors["patches_per_image"])
                if patches.dtype != torch.int64 or not torch.equal(patches, grid.prod(dim=1)):
                    raise ValueError("Rust patch counts do not match image grids")
                if len(batch.mm_placeholders) != len(grid):
                    raise ValueError("Rust token counts do not match grid items")
                pixels.append(pixel)
                grids.append(grid)
                counts.extend(p.length for p in batch.mm_placeholders)
                item_count += len(grid)
            if item_count != len(request["mm_items"]):
                raise ValueError("Rust artifact lost an image occurrence")
            items_per_req.append(item_count)
        grid = torch.cat(grids, dim=0)
        return EncoderPreprocessResult(
            mm_inputs={"pixel_values": torch.cat(pixels, dim=0), "image_grid_thw": grid},
            grid_thw=grid,
            token_counts=counts,
        ), items_per_req


def _tensor(value):
    """Materialize an E-local CPU view, validating bytes before handing it to native batch code."""
    import math

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
    if not value.shape or math.prod(value.shape) * width != len(value.data):
        raise ValueError("invalid managed artifact byte length")
    # Native consumers own a writable copy; the registered artifact stays immutable.
    return torch.frombuffer(bytearray(value.data), dtype=dtype).reshape(list(value.shape))


def install_bridge(encoder, processor_contract, max_artifact_bytes):
    """Install before serving on every rank; preserve native forward, TP, cache and sends."""
    required = ("process_batch_mm_items", "get_num_patches", "supports_modality")
    if any(
        not callable(getattr(encoder.preprocessor, name, None)) for name in required
    ) or not callable(
        getattr(getattr(encoder.preprocessor, "preproc_executor", None), "submit", None)
    ):
        raise ValueError("native encoder preprocessor component contract changed")
    if not processor_contract or max_artifact_bytes <= 0:
        raise ValueError("managed Rust processor requires a contract and byte limit")
    encoder.preprocessor = RustPreprocessorBridge(encoder.preprocessor, max_artifact_bytes)
    # Both native processing semantics and the exact Rust contract scope native cache keys.
    encoder.cache_namespace = hashlib.sha256(
        (encoder.cache_namespace + "\0smg-rust\0" + processor_contract).encode()
    ).hexdigest()


class NativeReferences:
    """Retain actual CPU, batch, send and release scopes beyond cancellation receipts."""

    def __init__(self):
        self.count = 0
        self.cancelled = False
        self.idle = asyncio.Event()
        self.idle.set()

    @asynccontextmanager
    async def retain(self, *, internal_release=False):
        """A reference ends when the actual native operation exits, not its HTTP caller."""
        if self.cancelled and not internal_release:
            raise asyncio.CancelledError("managed part no longer accepts work")
        self.count += 1
        self.idle.clear()
        try:
            yield
        finally:
            self.count -= 1
            if self.count == 0:
                self.idle.set()

    def cancel(self):
        """Close new work without cancelling shared GPU batch tasks or their TP collectives."""
        self.cancelled = True

    async def wait_quiesced(self):
        """Call after closing new work and retaining the actual native release operation."""
        if not self.cancelled:
            raise RuntimeError("cannot declare an open encoder operation quiescent")
        await self.idle.wait()


def launch_managed_rank(server_args, schedule_path, dist_init_method, rank):
    """Spawn target loads the SMG component hook before native MMEncoder construction."""
    from sglang.srt.plugins import load_plugins

    load_plugins()
    from sglang.srt.disaggregation.encoder.server import launch_encoder

    launch_encoder(server_args, schedule_path, dist_init_method, rank)


def launch_runtime(server_args, max_active):
    """Reuse native TP assembly with an explicit SMG follower initialization target."""
    from sglang.srt.disaggregation.encoder import runtime
    from sglang.srt.runtime_context import publish

    if server_args.dp_size != 1 or server_args.nnodes != 1:
        raise ValueError("managed E currently requires one host and one native DP unit")
    if max_active <= 0:
        raise ValueError("managed E requires a finite native queue")
    publish(server_args, role="encoder")
    # launch_local_runtime resolves this imported component when creating Process objects.
    # The target itself has a stable SMG module path, so spawn imports and runs our hook.
    original_target = runtime.launch_encoder
    runtime.launch_encoder = launch_managed_rank
    try:
        backend = runtime.launch_local_runtime(server_args)
    finally:
        runtime.launch_encoder = original_target
    backend.scheduler.pending_queue = asyncio.Queue(maxsize=max_active)
    if not hasattr(backend.encoder.preprocessor, "max_artifact_bytes"):
        raise RuntimeError("native rank zero did not install the managed Rust bridge")
    return backend


class OwnedBatchSubmitter:
    """Keep native batch completion alive beyond request cancellation and RPC deadlines.

    Native submit() applies wait_for to PendingRequest.future. Its timeout can retire
    that future while the batch still owns CUDA/TP resources. This adapter supplies
    the existing native queue with a future that only its native batch owner resolves.
    Admission and deadlines belong to the managed registry surrounding this adapter.
    """

    def __init__(self, scheduler, on_encoded):
        self.scheduler = scheduler
        self.on_encoded = on_encoded

    async def submit(self, request):
        """Enqueue once and await real batch completion; never cancel its shared future."""
        from sglang.srt.disaggregation.encoder.runtime import PendingRequest

        pending = PendingRequest(request, asyncio.get_running_loop())
        self.scheduler.pending_queue.put_nowait(pending)
        result = await asyncio.shield(pending.future)
        if result[3] is None:
            await self.on_encoded(request["req_id"])
        return result
