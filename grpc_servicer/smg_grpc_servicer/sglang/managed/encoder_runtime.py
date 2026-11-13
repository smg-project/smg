"""SMG assembly of native encoder components, without copying the GPU/TP scheduler."""

from __future__ import annotations

import asyncio


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
