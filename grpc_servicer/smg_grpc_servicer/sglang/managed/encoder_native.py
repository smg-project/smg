"""Native GPU/transport ownership behind the Rust encoder's same-host ZMQ control."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import grpc

from .encoder_bridge import ARTIFACT_KEY, OwnedBatchSubmitter


class Rejected(Exception):
    """Typed control refusal, before any native state is created."""

    def __init__(self, code, message):
        """Retain a typed control error without native ownership."""
        super().__init__(message)
        self.code = code


def require(condition, message, code=grpc.StatusCode.INVALID_ARGUMENT):
    """Keep validation failures separate from unexpected native errors."""
    if not condition:
        raise Rejected(code, message)


@dataclass
class NativeEntry:
    """Native task and send references; Rust alone owns admission and scheduling."""

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
    task: asyncio.Task | None = None
    cleanup: asyncio.Task | None = None
    lease: asyncio.Task | None = None

    @contextmanager
    def retain(self, *, cancelling=False):
        """Keep a native operation live until its batch or send actually exits."""
        if self.closing.done() and not cancelling:
            raise asyncio.CancelledError("managed part no longer accepts work")
        self.active += 1
        self.idle.clear()
        try:
            yield
        finally:
            self.active -= 1
            if not self.active:
                self.idle.set()


class NativeEncoder:
    """No admission queue: Rust owns scheduling; native owners prove safe release."""

    async def _receiver_control(self, entry, operation):
        """Cancel only metadata/registration waits; these own no GPU or transport buffers."""
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

    async def facade(self, kind, request):
        """Native receive calls and Encode resolve the exact same E operation and buffer owner."""
        from sglang.srt.disaggregation.encoder import server as native
        from sglang.srt.disaggregation.encoder.runtime import send_staged_embedding

        entry = self.part(request.get("req_id"))
        if kind == "metadata":
            require(request.get("part_idx") == 0, "invalid part index")
            with entry.retain():
                metadata = await self._receiver_control(
                    entry,
                    asyncio.wait_for(
                        native.meta_registry.wait(entry.rid),
                        max(0, entry.deadline - time.monotonic()),
                    ),
                )
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
        destination = (
            request.get("receive_url")
            if kind == "register"
            else (request.get("prefill_host"), request.get("embedding_port"))
        )
        require(bool(destination), "missing native receiver")
        expected_host = entry.prefill_host
        if kind == "register":
            require(
                self.backend.encoder.transfer_backend == "zmq_to_scheduler",
                "unexpected ZMQ receiver",
            )
            # The native receiver sends NetworkAddress.to_host_port_str(), and
            # the native sender adds the transport scheme itself.
            require(isinstance(destination, str), "invalid native receiver address")
            address = urlsplit("//" + destination)
            require(
                address.hostname == expected_host
                and bool(address.port)
                and not any(
                    (
                        address.username,
                        address.password,
                        address.path,
                        address.query,
                        address.fragment,
                    )
                ),
                "receiver differs from frozen P host",
            )
        else:
            require(
                self.backend.encoder.transfer_backend == "mooncake", "unexpected Mooncake receiver"
            )
            require(
                destination[0] == expected_host
                and type(destination[1]) is int
                and 0 < destination[1] <= 65535,
                "receiver differs from frozen P host",
            )
        key = (kind, destination)
        digest = hashlib.sha256(repr(sorted(request.items())).encode()).digest()
        if key in entry.sends:
            old_digest, task = entry.sends[key]
            require(
                digest == old_digest,
                "conflicting duplicate native receive call",
                grpc.StatusCode.ALREADY_EXISTS,
            )
            return await asyncio.shield(task)
        require(
            len(entry.sends) < self.config["max_receivers"],
            "too many native destinations",
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )

        # Commit only after all receiver and duplicate/capacity checks succeed.
        entry.receivers = count

        async def run():
            """Retain the actual send even if its HTTP response consumer disconnects."""
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
        entry.sends[key] = (digest, task)
        return await asyncio.shield(task)

    def __init__(self, config, backend):
        """Keep native owners and replay receipts separate from Rust admission."""
        self.config, self.backend = config, backend
        self.entries, self.completed = {}, {}
        self.submitter = OwnedBatchSubmitter(backend.scheduler)

    def part(self, rid):
        """Resolve only a live, uncancelled native owner within its deadline."""
        entry = self.entries.get(rid)
        require(entry is not None, "native operation is not registered")
        require(not entry.abort_requested, "native part closed")
        require(time.monotonic() < entry.deadline, "native facade deadline expired")
        return entry

    def register(self, command):
        """Create one generation-bound native owner and its orphan-cleanup lease."""
        rid = command["rid"]
        now = time.monotonic()
        self.completed = {key: item for key, item in self.completed.items() if item[0] > now}
        require(rid not in self.completed, "completed native operation")
        require(rid not in self.entries, "duplicate native operation")
        require(rid.startswith(self.config["generation"] + "."), "stale native generation")
        remaining = command["remaining_ms"] / 1000
        require(0 < remaining <= 122, "invalid native deadline")
        entry = NativeEntry(
            now + remaining,
            rid,
            prefill_host=command["prefill_host"],
        )
        self.entries[rid] = entry
        entry.lease = asyncio.create_task(self._lease(entry))

    async def _lease(self, entry):
        """Orphan fail-safe: a lost Rust control loop cannot retain native waits forever."""
        await asyncio.sleep(max(0, entry.deadline - time.monotonic()))
        if self.entries.get(entry.rid) is entry:
            try:
                await self.cancel(entry.rid)
            finally:
                # A failed wakeup is not cleanup. Registrations never executed
                # have no execution owner to reach the final release barrier.
                if entry.task is None:
                    await self.finish(entry.rid, "native control lease expired")

    async def cancel(self, rid):
        """Wake native waits while retaining cancellation work until the cleanup barrier."""
        entry = self.entries.get(rid)
        if entry is not None:
            entry.abort_requested = True
            entry.closing.cancel()
            # Once cleanup starts it owns abandonment and release; do not admit a
            # new native cancellation after its quiescence barrier has passed.
            if entry.cleanup is None:
                with entry.retain(cancelling=True):
                    # This is a wakeup, never proof of GPU or send completion.
                    await self.backend.encoder.abandon_request(rid)

    async def execute(self, command, artifact):
        """Create one native execution task; disconnecting an observer does not cancel it."""
        if command["rid"] in self.completed:
            return self.completed[command["rid"]][1]
        entry = self.entries[command["rid"]]
        require(entry.task is None, "duplicate native execute")
        entry.task = asyncio.create_task(self._execute(command, artifact))
        return await asyncio.shield(entry.task)

    async def _execute(self, command, artifact):
        """Retain native batch work and delivery before obtaining the final cleanup receipt."""
        from sglang.srt.disaggregation.encoder.runtime import execute_encode_pipeline
        from sglang.srt.managers.req_types import Modality

        entry = self.entries[command["rid"]]
        error = None
        try:
            require(not entry.abort_requested, "native operation cancelled before submission")
            tokens = command["embedding_tokens"]
            require(type(tokens) is int and tokens > 0, "invalid embedding token count")
            size = (
                tokens
                * self.backend.encoder._embedding_dims[Modality.IMAGE]
                * self.backend.encoder._element_size
            )
            require(size <= self.config["max_staging_bytes_per_request"], "staging budget exceeded")
            native = dict(
                req_id=entry.rid,
                num_parts=1,
                part_idx=0,
                modality="image",
                mm_items=[None] * command["image_count"],
            )
            native[ARTIFACT_KEY] = artifact
            with entry.retain():
                await execute_encode_pipeline(self.backend.encoder, self.submitter, native)
            if self.backend.encoder.transfer_backend == "mooncake":
                await self._receiver_control(entry, entry.delivered.wait())
            require(not entry.abort_requested, "native operation cancelled")
        except (Exception, asyncio.CancelledError) as exc:
            error = str(exc)[:1024] or type(exc).__name__
        return await self.finish(entry.rid, error)

    async def finish(self, rid, error=None):
        """A single native cleanup owner; duplicate controls only observe its receipt."""
        if rid in self.completed:
            return self.completed[rid][1]
        entry = self.entries[rid]
        if entry.cleanup is None:
            entry.cleanup = asyncio.create_task(self._finish(entry, error))
        return await asyncio.shield(entry.cleanup)

    async def _finish(self, entry, error):
        """Release through one cleanup owner; preserve state if native release fails."""
        from sglang.srt.disaggregation.encoder.server import meta_registry

        rid = entry.rid
        entry.closing.cancel()
        if error:
            try:
                await meta_registry.publish(rid, 0, 0, 0, error=error)
            except Exception as exc:
                # Cancellation and release still run; publication is not a resource owner.
                error = str(exc)[:1024] or type(exc).__name__
        try:
            await self.backend.encoder.abandon_request(rid)
            if entry.active:
                await entry.idle.wait()
            await self.backend.encoder.release_request(rid)
            self.backend.encoder.clear_abandoned_request(rid)
        except Exception as exc:
            return {"released": False, "error": str(exc)[:1024] or type(exc).__name__}
        entry.sends.clear()
        # The lease can be the caller of this task: do not cancel it while it observes cleanup.
        if entry.task is not None:
            entry.lease.cancel()
        receipt = {"released": True, "error": error}
        horizon = entry.deadline + self.config["retention_ms"] / 1000
        self.completed[rid] = (horizon, receipt)
        del self.entries[rid]
        return receipt

    async def dispatch(self, command, artifact):
        """Route same-host control operations to their existing native owner."""
        operation, rid = command["operation"], command["rid"]
        if operation == "register":
            self.register(command)
        elif operation == "execute":
            return await self.execute(command, artifact)
        elif operation == "cancel":
            await self.cancel(rid)
        elif operation == "finish":
            return await self.finish(rid, command.get("error"))
        elif operation == "facade":
            return await self.facade(command["kind"], command["payload"])
        else:
            raise ValueError("unknown native encoder operation")
        return None


async def serve_native(config, backend, command_endpoint, event_endpoint):
    """One bounded Rust command stream; replies never imply release unless explicitly stated."""
    import zmq
    import zmq.asyncio

    owner = NativeEncoder(config, backend)
    context = zmq.asyncio.Context()
    commands, events = context.socket(zmq.PULL), context.socket(zmq.PUSH)
    commands.setsockopt(zmq.LINGER, 0)
    events.setsockopt(zmq.LINGER, 0)
    commands.connect(command_endpoint)
    events.connect(event_endpoint)
    tasks = set()
    stopping = asyncio.get_running_loop().create_future()

    async def publish(reply):
        """Only a completed dispatch's reply observer may be dropped on IPC shutdown."""
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
        """Dispatch native work before publishing a reply; shutdown may drop only publication."""
        if stopping.done():
            frames.clear()
            return
        command = json.loads(frames[0])
        if command.get("generation") != config["generation"]:
            return
        reply = {"sequence": command["sequence"], "generation": config["generation"]}
        try:
            reply["result"] = await owner.dispatch(command, frames[1] if len(frames) == 2 else b"")
        except (Exception, asyncio.CancelledError) as error:
            reply["error"] = str(error)[:1024] or type(error).__name__
            if isinstance(error, Rejected):
                reply["error_code"] = error.code.value[0]
            elif isinstance(error, asyncio.CancelledError):
                reply["error_code"] = 1
            elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
                reply["error_code"] = 4
        # Drop the incoming artifact before publishing a release receipt to Rust.
        frames.clear()
        await publish(reply)

    try:
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
        # Stop reply publication, not native dispatch: a missing Rust reader must
        # not pin completed work behind PUSH HWM while native owners still drain.
        stopping.set_result(None)
        # Native batch/send tasks must settle before socket or tensor teardown.
        # A failed abandonment does not excuse other owners from cancellation,
        # and is not proof of release. Their dispatches still have to settle.
        await asyncio.gather(
            *(owner.cancel(rid) for rid in list(owner.entries)), return_exceptions=True
        )
        await asyncio.gather(*tasks, return_exceptions=True)
        commands.close()
        events.close()
        context.term()
