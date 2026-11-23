"""Native encoder batch, HTTP receiver facade, and ownership behind Rust E control."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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
_HTTP_STATUS = {
    grpc.StatusCode.CANCELLED: 499,
    grpc.StatusCode.OUT_OF_RANGE: 413,
    grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.INTERNAL: 500,
}


def _err(exc):
    """Format an exception message bounded to 1 KiB for IPC/HTTP error payloads."""
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


class RustPreprocessorBridge:
    """Use artifacts distributed by the native TP request IPC; never read media on a rank."""

    def __init__(self, original, max_artifact_bytes):
        self.original = original
        self.max_artifact_bytes = max_artifact_bytes

    def supports_modality(self, modality):
        """Accept image artifacts only on the managed Rust preprocessing bridge."""
        return modality.name == "IMAGE"

    def get_num_patches(self, grid, modality):
        """Delegate image patch counting to SGLang's native preprocessor."""
        if not self.supports_modality(modality):
            raise ValueError("managed encoder supports image artifacts only")
        return self.original.get_num_patches(grid, modality)

    async def process_batch_mm_items(self, requests, modality):
        """Materialize a batch of Rust-preprocessed artifacts on the executor without interruption."""
        return await run_uninterruptible(
            self.original.preproc_executor, self._materialize_batch, requests, modality
        )

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
                if "image_grid_thw" not in batch.model_specific_tensors:
                    raise ValueError("unsupported Rust artifact model fields")
                # Zero-copy view over protobuf bytes; torch.cat creates a writable owned tensor.
                pixel = _tensor(batch.pixel_values, copy=False)
                grid = _tensor(batch.model_specific_tensors["image_grid_thw"])
                if grid.dtype != torch.int64 or grid.ndim != 2 or grid.shape[1] != 3:
                    raise ValueError("invalid Rust image grid")
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


def _tensor(value, *, copy=True):
    """Materialize an E-local CPU view, validating dtype and byte length."""
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
    data = value.data
    if not value.shape or math.prod(value.shape) * width != len(data):
        raise ValueError("invalid managed artifact byte length")
    storage = bytearray(data) if copy else data
    return torch.frombuffer(storage, dtype=dtype).reshape(list(value.shape))


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
    encoder.cache_namespace = hashlib.sha256(
        (encoder.cache_namespace + "\0smg-rust\0" + processor_contract).encode()
    ).hexdigest()


def launch_managed_rank(server_args, schedule_path, dist_init_method, rank):
    """Load managed SGLang plugins before starting an encoder TP rank."""
    from sglang.srt.plugins import load_plugins

    load_plugins()
    from sglang.srt.disaggregation.encoder.server import launch_encoder

    launch_encoder(server_args, schedule_path, dist_init_method, rank)


def launch_runtime(server_args, max_active):
    """Launch SGLang's local encoder TP runtime with a bounded pending request queue."""
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
    backend.scheduler.pending_queue = asyncio.Queue(maxsize=max_active)
    return backend


class OwnedBatchSubmitter:
    """Enqueue once and await real batch completion; never cancel the shared batch future."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    async def submit(self, request):
        """Submit one request into the native encoder scheduler queue and shield its completion."""
        from sglang.srt.disaggregation.encoder.runtime import PendingRequest

        pending = PendingRequest(request, asyncio.get_running_loop())
        self.scheduler.pending_queue.put_nowait(pending)
        return await asyncio.shield(pending.future)


class Rejected(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, message, code=grpc.StatusCode.INVALID_ARGUMENT):
    """Raise `Rejected` with the given gRPC status code when `condition` is false."""
    if not condition:
        raise Rejected(code, message)


@dataclass
class NativeEntry:
    deadline: float
    rid: str
    prefill_host: str
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
        """Track an active native operation so release waits until every in-flight user exits."""
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
        for waiter in self.waiters.pop(rid, ()):
            waiter.set()

    def _is_abandoned(self, rid):
        expiry = self.abandoned.get(rid)
        if expiry is None:
            return False
        if expiry <= time.monotonic():
            self.abandoned.pop(rid, None)
            return False
        return True

    async def _receiver_control(self, entry, operation):
        """Race a receiver control future against entry cancellation and clean up on exit."""
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
            require(
                not self._is_abandoned(rid),
                "native operation aborted before execution",
                grpc.StatusCode.CANCELLED,
            )
            event = asyncio.Event()
            self.waiters.setdefault(rid, []).append(event)
            try:
                await asyncio.wait_for(event.wait(), timeout)
            except asyncio.TimeoutError as exc:  # noqa: UP041 - Python 3.10
                raise Rejected(
                    grpc.StatusCode.DEADLINE_EXCEEDED, "early native receiver expired"
                ) from exc
            finally:
                if (pending := self.waiters.get(rid)) and event in pending:
                    pending.remove(event)
                    if not pending:
                        del self.waiters[rid]
            entry = self.entries.get(rid)
            if entry is None:
                require(
                    not self._is_abandoned(rid),
                    "native operation aborted before execution",
                    grpc.StatusCode.CANCELLED,
                )
        require(entry is not None, "native operation is not registered")
        require(not entry.abort_requested, "native part closed", grpc.StatusCode.CANCELLED)
        require(
            time.monotonic() < entry.deadline,
            "native facade deadline expired",
            grpc.StatusCode.DEADLINE_EXCEEDED,
        )
        return entry

    async def facade(self, kind, request):
        """Serve one P->E receiver HTTP call (`metadata`, `register`, or `send`) with host fencing."""
        from sglang.srt.disaggregation.encoder import server as native
        from sglang.srt.disaggregation.encoder.runtime import send_staged_embedding

        entry = await self._wait_entry(request.get("req_id"))
        if kind == "metadata":
            require(request.get("part_idx") == 0, "invalid part index")
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
            return {"req_id": entry.rid, "part_idx": 0, **metadata}
        count = request.get("receive_count")
        require(
            type(count) is int and 0 < count <= self.config["max_receivers"],
            "invalid native receive count",
        )
        require(entry.receivers in (None, count), "native receive count changed")
        expected_host = entry.prefill_host
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
        else:
            require(
                self.backend.encoder.transfer_backend == "mooncake", "unexpected Mooncake receiver"
            )
            destination = (request.get("prefill_host"), request.get("embedding_port"))
            require(
                isinstance(destination[0], str)
                and destination[0].lower() == expected_host.lower()
                and type(destination[1]) is int
                and 0 < destination[1] <= 65535,
                "receiver differs from frozen P host",
            )
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

    async def cancel(self, rid):
        """Wake and abandon an active native encode entry without releasing its slot prematurely."""
        entry = self.entries.get(rid)
        if entry is not None:
            entry.abort_requested = True
            entry.closing.cancel()
            if entry.cleanup is None:
                with entry.retain(cancelling=True):
                    await self.backend.encoder.abandon_request(rid)
        else:
            self._mark_abandoned(rid)

    async def execute(self, command, artifact):
        """Run one Rust-preprocessed artifact through native GPU encoding, transfer, and release."""
        from sglang.srt.disaggregation.encoder.runtime import execute_encode_pipeline
        from sglang.srt.managers.req_types import Modality

        rid = command["rid"]
        require(rid and rid not in self.entries, "duplicate or empty native operation")
        remaining = command["remaining_ms"] / 1000
        require(0 < remaining <= 122, "invalid native deadline")
        entry = NativeEntry(time.monotonic() + remaining, rid, prefill_host=command["prefill_host"])
        self.abandoned.pop(rid, None)
        self.entries[rid] = entry
        for waiter in self.waiters.pop(rid, ()):
            waiter.set()
        error = None
        try:
            tokens = command["embedding_tokens"]
            require(type(tokens) is int and tokens > 0, "invalid embedding token count")
            size = (
                tokens
                * self.backend.encoder._embedding_dims[Modality.IMAGE]
                * self.backend.encoder._element_size
            )
            require(size <= self.config["max_staging_bytes_per_request"], "staging budget exceeded")
            native = {
                "req_id": rid,
                "num_parts": 1,
                "part_idx": 0,
                "modality": "image",
                "mm_items": [None] * command["image_count"],
                ARTIFACT_KEY: artifact,
            }
            with entry.retain():
                await execute_encode_pipeline(self.backend.encoder, self.submitter, native)
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
        """Wait for active receiver calls, release native staging state, and return the release receipt."""
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
            await self.backend.encoder.abandon_request(rid)
            if entry.active:
                await entry.idle.wait()
            await self.backend.encoder.release_request(rid)
            self.backend.encoder.clear_abandoned_request(rid)
        except Exception as exc:
            return {"released": False, "error": _err(exc)}
        entry.sends.clear()
        del self.entries[rid]
        return {"released": True, "error": error}

    def close(self):
        """Mark the encoder shutting down and wake all early HTTP facade waiters."""
        self.closed = True
        for waiters in self.waiters.values():
            for waiter in waiters:
                waiter.set()
        self.waiters.clear()

    async def dispatch(self, command, artifact):
        """Route one sequenced IPC command from Rust (`execute`, `cancel`, or `forget`)."""
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
    """Serve P->E receiver HTTP routes directly on the native asyncio loop (no ZMQ detour)."""
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
                    {"message": str(err)}, status=_HTTP_STATUS.get(err.code, 400)
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
                    *(owner.cancel(rid) for rid in list(owner.entries)),
                    return_exceptions=True,
                ),
                timeout=remaining(),
            )
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=remaining(),
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
