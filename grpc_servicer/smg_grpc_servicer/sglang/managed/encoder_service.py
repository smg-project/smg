"""Bounded SMG encoder control registry and the native P receiver control facade."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import grpc
from smg_grpc_proto.generated import sglang_managed_pb2 as pb
from smg_grpc_proto.generated import sglang_managed_pb2_grpc as rpc

from .config import encoder_window, healthy
from .encoder_bridge import ARTIFACT_KEY, NativeReferences, OwnedBatchSubmitter


class Rejected(Exception):
    """Typed control refusal, before any native state is created."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, message, code=grpc.StatusCode.INVALID_ARGUMENT):
    """Keep validation failures separate from unexpected native errors."""
    if not condition:
        raise Rejected(code, message)


def same_operation(left, right):
    """Compare all instance, attempt, branch and digest fences, not an ID alone."""
    return left.SerializeToString(deterministic=True) == right.SerializeToString(deterministic=True)


@dataclass
class Entry:
    """Only this E registry owns admission, CPU/native tasks, sends and final release."""

    operation: object
    expiry: int
    deadline: float
    rid: str
    admission: int = 0
    created_ns: int = field(default_factory=time.monotonic_ns)
    cpu_started_ns: int = 0
    embedding_bytes: int = 0
    request: object = None
    request_digest: bytes = b""
    cancelled: bool = False
    quiesced: bool = False
    terminal: bool = False
    work: int = 0
    cpu_work: int = 0
    gpu_work: int = 0
    context_bytes: int = 0
    artifact_bytes: int = 0
    staging_bytes: int = 0
    send_bytes: int = 0
    events: list = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    cancellation: asyncio.Event = field(default_factory=asyncio.Event)
    delivered: asyncio.Event = field(default_factory=asyncio.Event)
    refs: NativeReferences = field(default_factory=NativeReferences)
    sends: dict = field(default_factory=dict)
    receivers: int | None = None
    destinations: set = field(default_factory=set)
    task: object = None


class FairQueue:
    """Weighted deficit round robin with bounded costs and an oldest-request escape."""

    def __init__(self, quantum, aging_seconds, weights):
        self.quantum, self.aging = quantum, aging_seconds
        self.weights = weights
        self.queues = OrderedDict()
        self.deficits = {}

    def put(self, tenant, entry):
        """The registry's finite admission bounds the number of queue entries and tenants."""
        self.queues.setdefault(tenant, deque()).append((time.monotonic(), entry))
        self.deficits.setdefault(tenant, 0)

    def pop(self):
        """Do not let a large request starve behind an indefinite stream of small work."""
        now = time.monotonic()
        oldest = min(self.queues, key=lambda key: self.queues[key][0][0])
        if now - self.queues[oldest][0][0] >= self.aging:
            return self._take(oldest)
        rounds = min(
            max(
                1,
                (
                    queue[0][1].work
                    - self.deficits[tenant]
                    + self.quantum * self.weights.get(tenant, 1)
                    - 1
                )
                // (self.quantum * self.weights.get(tenant, 1)),
            )
            for tenant, queue in self.queues.items()
        )
        for tenant in self.queues:
            self.deficits[tenant] += rounds * self.quantum * self.weights.get(tenant, 1)
        while self.queues:
            tenant = next(iter(self.queues))
            entry = self.queues[tenant][0][1]
            if entry.work <= self.deficits[tenant]:
                self.deficits[tenant] -= entry.work
                result = self._take(tenant)
                if tenant in self.queues:
                    self.queues.move_to_end(tenant)
                return result
            self.queues.move_to_end(tenant)

    def remove(self, entry):
        """A queued cancellation removes only its own work, preserving every sibling's order."""
        for tenant, queue in list(self.queues.items()):
            for queued in queue:
                if queued[1] is entry:
                    queue.remove(queued)
                    if not queue:
                        del self.queues[tenant]
                        del self.deficits[tenant]
                    return True
        return False

    def _take(self, tenant):
        """Remove empty tenant state so historical tenant labels cannot grow memory."""
        _, entry = self.queues[tenant].popleft()
        if not self.queues[tenant]:
            del self.queues[tenant]
            del self.deficits[tenant]
        return entry


class EncoderService(rpc.ManagedEncoderServicer):
    """Single-loop admission ledger shared by gRPC and the native HTTP facade."""

    def __init__(self, config, backend, processor):
        self.config, self.backend, self.processor = config, backend, processor
        self.capacity = config["max_active"]
        # Waiting requests own bounded controls, not yet materialized tensors. Reserve
        # room for every queued control, then bound the CPU/native/send window separately.
        self.execution_capacity, self.execution_bytes = encoder_window(config)
        self.execution = asyncio.BoundedSemaphore(self.execution_capacity)
        self.instance = pb.Instance(worker_id=config["worker_id"], generation=config["generation"])
        self.entries, self.parts, self.early = {}, {}, {}
        self.admission_seq = self.snapshot_seq = 0
        self.draining = False
        self.queue = FairQueue(
            config["drr_quantum"], config["aging_seconds"], config["tenant_weights"]
        )
        self.queued = asyncio.Event()
        self.cpu = asyncio.Semaphore(config["max_cpu"])
        self.cpu_target = min(2, config["max_cpu"])
        self.cpu_active = 0
        self.cpu_changed = asyncio.Event()
        self.dispatcher = None
        self.early_count = 0
        self.submitter = OwnedBatchSubmitter(backend.scheduler, self._encoded)

    async def _encoded(self, rid):
        """Record GPU completion before native delivery starts, including ZMQ inline sends."""
        # Native completion is the feedback boundary. Pending native work or an
        # occupied dispatch lock means more CPU producers cannot feed this GPU.
        if (
            self.queue.queues
            and self.cpu_active >= self.cpu_target
            and self.cpu_target < self.config["max_cpu"]
            and self.backend.scheduler.pending_queue.empty()
            and not self.backend.encoder.encode_dispatch_lock.locked()
        ):
            self.cpu_target += 1
            self.cpu_changed.set()
        self.parts[rid].gpu_work = 0
        await self._event(self.parts[rid], stage=pb.ENCODED)

    def start(self):
        """Start one admission dispatcher; request tasks remain in the bounded registry."""
        self.backend.start()
        self.dispatcher = asyncio.create_task(self._dispatch())

    def _operation(self, operation):
        """Validate the explicit instance fence before registry lookup or tombstone creation."""
        require(
            operation.target == self.instance,
            "stale E instance",
            grpc.StatusCode.FAILED_PRECONDITION,
        )
        require(operation.identity.branch_id == 0, "managed E supports branch zero only")
        for value in (
            operation.operation_id,
            operation.identity.logical_request_id,
            operation.identity.attempt_id,
        ):
            require(0 < len(value) <= 256, "invalid operation identity")
        require(
            len(operation.request_digest) == 32 and len(operation.plan_digest) == 32,
            "invalid operation digest",
        )

    def _collect(self, now):
        """Only safe terminal records past their original acceptance horizon may expire."""
        for key, entry in list(self.entries.items()):
            if entry.quiesced and now > entry.expiry + self.config["retention_ms"]:
                self.entries.pop(key)
                self.parts.pop(entry.rid, None)

    def _budget(self, budget):
        """Freeze the minimum of original wall expiry and the received remaining budget."""
        require(
            healthy(self.config["max_clock_error_ms"]),
            "E clock synchronization outside tolerance",
            grpc.StatusCode.UNAVAILABLE,
        )
        now = int(time.time() * 1000)
        remaining = min(budget.original_expiry_unix_ms - now, budget.remaining_timeout_ms)
        require(remaining > 0, "Encode expired", grpc.StatusCode.DEADLINE_EXCEEDED)
        require(
            budget.original_expiry_unix_ms - now <= self.config["max_duration_ms"],
            "expiry exceeds retention horizon",
        )
        require(0 < budget.estimated_work <= self.config["max_work"], "invalid estimated work")
        self._collect(now)
        return time.monotonic() + remaining / 1000

    def _admit(self, request):
        """Admit bounded controls; the execution window reserves tensors before media IO."""
        self._operation(request.operation)
        require(request.version == 1, "unsupported managed version")
        deadline = self._budget(request.budget)
        require(
            0 < len(request.image_references) <= self.config["max_images"],
            "invalid managed image count",
        )
        require(len(request.tenant_id) <= 256, "tenant label too long")
        binding, plan = request.receiver_binding, request.plan
        require(
            binding.prefill_rid == plan.receiver_rid and bool(plan.receiver_rid),
            "receiver binding mismatch",
        )
        require(binding.decode.identity == request.operation.identity, "attempt binding mismatch")
        require(
            binding.decode.request_digest == request.operation.request_digest
            and binding.decode.plan_digest == request.operation.plan_digest,
            "plan digest mismatch",
        )
        require(
            binding.model_revision == self.config["model_revision"]
            and binding.processor_contract == self.config["processor_contract"],
            "model/processor mismatch",
            grpc.StatusCode.FAILED_PRECONDITION,
        )
        require(
            request.transfer_backend == self.backend.encoder.transfer_backend,
            "transfer backend mismatch",
        )
        require(
            len(plan.parts) == 1 and list(plan.image_counts) == [len(request.image_references)],
            "one E must own all images",
        )
        require(
            list(plan.encoder_receive_endpoints) == [self.config["receive_endpoint"]],
            "foreign E facade",
        )
        part = plan.parts[0]
        require(
            part.part_idx == 0 and part.native_part_rid == plan.receiver_rid + "_local_part_0",
            "invalid native part mapping",
        )
        require(
            list(part.ordered_item_ids) == list(range(len(request.image_references))),
            "image occurrence order mismatch",
        )
        # Old native facades carry no generation field. Their unique native rid must fence it.
        require(
            plan.receiver_rid.startswith(self.instance.generation + "."),
            "native part must carry E instance generation",
        )
        canonical = pb.EncodeRequest()
        canonical.CopyFrom(request)
        canonical.budget.remaining_timeout_ms = 0
        encoded = canonical.SerializeToString(deterministic=True)
        require(
            len(encoded) <= self.config["max_control_bytes"],
            "control request too large",
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        digest = hashlib.sha256(encoded).digest()
        previous = self.entries.get(request.operation.operation_id)
        if previous is not None:
            require(
                same_operation(previous.operation, request.operation)
                and previous.expiry == request.budget.original_expiry_unix_ms,
                "conflicting operation",
                grpc.StatusCode.ALREADY_EXISTS,
            )
            require(
                not previous.cancelled, "operation already cancelled", grpc.StatusCode.CANCELLED
            )
            require(
                previous.request_digest == digest,
                "conflicting Encode contents",
                grpc.StatusCode.ALREADY_EXISTS,
            )
            return previous
        active = [entry for entry in self.entries.values() if not entry.quiesced]
        reserve = len(encoded)
        require(
            not self.draining
            and len(self.entries) < self.config["max_records"]
            and len(active) < self.capacity,
            "E admission full",
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        require(
            sum(entry.work for entry in active) + request.budget.estimated_work
            <= self.config["max_work"],
            "E work capacity full",
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        require(
            sum(entry.context_bytes for entry in active) + reserve
            <= self.config["max_context_bytes"],
            "E artifact capacity full",
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        require(
            part.native_part_rid not in self.parts,
            "native part already belongs to another operation",
            grpc.StatusCode.ALREADY_EXISTS,
        )
        self.admission_seq += 1
        entry = Entry(
            request.operation,
            request.budget.original_expiry_unix_ms,
            deadline,
            part.native_part_rid,
            admission=self.admission_seq,
            request=request,
            request_digest=digest,
            work=request.budget.estimated_work,
            context_bytes=reserve,
        )
        self.entries[request.operation.operation_id] = entry
        self.parts[entry.rid] = entry
        entry.artifact_bytes = reserve
        entry.cpu_work = entry.work
        # Accepted CPU work will also consume GPU capacity; do not hide it until preprocess ends.
        entry.gpu_work = entry.work
        waiter = self.early.get(entry.rid)
        if waiter is not None:
            waiter[0].set()
        self.queue.put(request.tenant_id, entry)
        self.queued.set()
        return entry

    async def _event(self, entry, *, stage=None, error=None):
        """Finite per-operation history supports duplicate observers without duplicate work."""
        event = pb.ControlEvent(
            operation=entry.operation,
            admission_seq=entry.admission,
            event_seq=len(entry.events) + 1,
            part_idx=0,
        )
        if error is not None:
            event.failed.CopyFrom(pb.Failure(code="encoder_failed", message=str(error)[:1024]))
        else:
            event.stage = stage
        logging.getLogger(__name__).info(
            "managed E event operation=%s seq=%s stage=%s refs=%s "
            "rid=%s elapsed_us=%s admission_queue_us=%s embedding_bytes=%s "
            "cpu_active=%s cpu_target=%s",
            entry.operation.operation_id,
            event.event_seq,
            pb.Stage.Name(stage) if error is None else "FAILED",
            entry.refs.count,
            entry.rid,
            (time.monotonic_ns() - entry.created_ns) // 1000,
            (entry.cpu_started_ns - entry.created_ns) // 1000 if entry.cpu_started_ns else None,
            entry.embedding_bytes,
            self.cpu_active,
            self.cpu_target,
        )
        async with entry.changed:
            require(
                len(entry.events) < 8,
                "encoder event history exceeded bound",
                grpc.StatusCode.INTERNAL,
            )
            entry.events.append(event)
            entry.changed.notify_all()

    def _release_cpu(self):
        """Release only completed CPU owners; backpressure never cancels running work."""
        self.cpu_active -= 1
        self.cpu.release()
        if self.cpu_target > min(
            2, self.config["max_cpu"]
        ) and self.backend.scheduler.pending_queue.qsize() >= min(
            self.backend.scheduler.max_batch_size, max(1, self.execution_capacity // 2)
        ):
            self.cpu_target -= 1
        self.cpu_changed.set()

    async def _dispatch(self):
        """Grant tensor and CPU slots in DRR order while completed CPU work overlaps GPU/send."""
        while True:
            await self.queued.wait()
            await self.execution.acquire()
            while self.cpu_active >= self.cpu_target:
                self.cpu_changed.clear()
                await self.cpu_changed.wait()
            await self.cpu.acquire()
            self.cpu_active += 1
            if not self.queue.queues:
                self._release_cpu()
                self.execution.release()
                self.queued.clear()
                continue
            entry = self.queue.pop()
            entry.context_bytes += self.execution_bytes
            entry.artifact_bytes += self.config["max_artifact_bytes"]
            entry.staging_bytes = self.config["max_staging_bytes_per_request"]
            entry.send_bytes = entry.staging_bytes * self.config["max_receivers"]
            entry.task = asyncio.create_task(self._run(entry, execution_held=True))
            if not self.queue.queues:
                self.queued.clear()

    async def _run(self, entry, *, cpu_held=True, execution_held=False):
        """Do not cancel CPU, a native batch or a zero-copy send when the request expires."""
        from sglang.srt.disaggregation.encoder.runtime import execute_encode_pipeline

        native_started = False

        async def abandon_cancelled_native():
            """Wake missing-destination waits through the native lifecycle owner, never cancel CUDA."""
            await entry.cancellation.wait()
            if native_started and entry.cancelled:
                await self.backend.encoder.abandon_request(entry.rid)

        cancellation = asyncio.create_task(abandon_cancelled_native())
        try:
            require(
                not entry.cancelled and time.monotonic() < entry.deadline,
                "Encode cancelled before CPU",
                grpc.StatusCode.CANCELLED,
            )
            entry.cpu_started_ns = time.monotonic_ns()
            async with entry.refs.retain():
                artifact = await asyncio.to_thread(
                    self.processor.prepare,
                    list(entry.request.image_references),
                    self.config["image_token_id"],
                )
            self._release_cpu()
            cpu_held = False
            entry.cpu_work = 0
            entry.gpu_work = entry.work
            await self._event(entry, stage=pb.PREPROCESSED)
            require(
                not entry.cancelled and time.monotonic() < entry.deadline,
                "Encode expired after CPU",
                grpc.StatusCode.CANCELLED,
            )
            from sglang.srt.managers.req_types import Modality
            from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg

            prepared = sg.GenerateRequest.FromString(artifact)
            tokens = sum(
                item.length
                for batch in [prepared.mm_inputs, *prepared.extra_mm_inputs]
                for item in batch.mm_placeholders
            )
            native_bytes = (
                tokens
                * self.backend.encoder._embedding_dims[Modality.IMAGE]
                * self.backend.encoder._element_size
            )
            require(
                native_bytes <= entry.staging_bytes,
                "embedding exceeds reserved staging budget",
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            )
            entry.embedding_bytes = native_bytes
            del prepared
            native = {
                "req_id": entry.rid,
                "num_parts": 1,
                "part_idx": 0,
                "modality": "image",
                "mm_items": [None] * len(entry.request.image_references),
                ARTIFACT_KEY: artifact,
            }
            async with entry.refs.retain():
                native_started = True
                await execute_encode_pipeline(self.backend.encoder, self.submitter, native)
            if self.backend.encoder.transfer_backend == "mooncake":
                delivered = asyncio.create_task(entry.delivered.wait())
                cancelled = asyncio.create_task(entry.cancellation.wait())
                try:
                    await asyncio.wait(
                        [delivered, cancelled],
                        timeout=max(0, entry.deadline - time.monotonic()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for waiter in (delivered, cancelled):
                        waiter.cancel()
                    await asyncio.gather(delivered, cancelled, return_exceptions=True)
                require(
                    entry.delivered.is_set() and not entry.cancelled,
                    "Encode delivery cancelled or expired",
                    grpc.StatusCode.CANCELLED,
                )
            require(
                not entry.cancelled,
                "Encode delivery cancelled",
                grpc.StatusCode.CANCELLED,
            )
            await self._event(entry, stage=pb.DELIVERED)
        except Exception as error:
            await self._event(entry, error=error)
            if not native_started:
                # A P metadata waiter may already hold a reference to this queued
                # request. Publish failure before waiting for that reader to leave.
                from sglang.srt.disaggregation.encoder.server import meta_registry

                try:
                    await meta_registry.publish(entry.rid, 0, 0, 0, error=str(error)[:1024])
                except Exception:
                    logging.getLogger(__name__).exception(
                        "native metadata failure publication failed"
                    )
        finally:
            if cpu_held:
                self._release_cpu()
            entry.refs.cancel()
            entry.cancellation.set()
            # Native abandonment wakes ZMQ destination waits but retains active encode/send
            # owners. Their scopes still have to finish before SMG removes any charges.
            await cancellation
            await entry.refs.wait_quiesced()
            try:
                await self.backend.encoder.release_request(entry.rid)
                if native_started:
                    self.backend.encoder.clear_abandoned_request(entry.rid)
            except Exception as error:
                await self._event(entry, error=error)
            else:
                # Retain only the replay fence and bounded events after native owners exit.
                entry.request = None
                # Failed send tasks retain traceback frames holding native embeddings.
                # Closed parts cannot replay these tasks after quiescence.
                entry.sends.clear()
                entry.quiesced = True
                entry.cpu_work = entry.gpu_work = 0
                entry.context_bytes = entry.artifact_bytes = entry.work = entry.staging_bytes = (
                    entry.send_bytes
                ) = 0
                if execution_held:
                    self.execution.release()
                await self._event(entry, stage=pb.QUIESCED)
            async with entry.changed:
                entry.terminal = True
                entry.changed.notify_all()

    async def Encode(self, request, context):
        """Return Accepted before CPU processing; a duplicate attaches to the same operation."""
        try:
            owns_lifetime = request.operation.operation_id not in self.entries
            entry = self._admit(request)
            if not entry.events:
                await self._event(entry, stage=pb.ACCEPTED)
        except Rejected as error:
            await context.abort(error.code, str(error))
        index = 0
        try:
            while True:
                async with entry.changed:
                    await entry.changed.wait_for(
                        lambda: index < len(entry.events) or entry.terminal
                    )
                    batch = entry.events[index:]
                    index += len(batch)
                    terminal = entry.terminal
                for event in batch:
                    yield event
                if terminal:
                    return
        finally:
            if owns_lifetime and not entry.terminal:
                self._cancel(entry)

    def _cancel(self, entry):
        """Close new work; cancellation acknowledgement never frees resource charges."""
        entry.cancelled = True
        entry.refs.cancel()
        entry.cancellation.set()
        if entry.task is None and self.queue.remove(entry):
            # There is no CPU/native execution permit to release. Any early receiver
            # still owns its references until the same normal cleanup confirms safety.
            entry.task = asyncio.create_task(self._run(entry, cpu_held=False))

    async def Cancel(self, request, context):
        """Install a bounded pre-Encode tombstone or close an existing operation."""
        try:
            self._operation(request.operation)
            entry = self.entries.get(request.operation.operation_id)
            if entry is not None:
                require(
                    same_operation(entry.operation, request.operation)
                    and entry.expiry == request.original_expiry_unix_ms,
                    "conflicting cancellation",
                    grpc.StatusCode.ALREADY_EXISTS,
                )
                self._cancel(entry)
            else:
                now = int(time.time() * 1000)
                if request.original_expiry_unix_ms <= now:
                    return pb.TerminationReceipt(recorded=True, quiesced=True)
                require(
                    request.original_expiry_unix_ms - now <= self.config["max_duration_ms"],
                    "cancellation expiry exceeds horizon",
                )
                self._collect(now)
                require(
                    len(self.entries) < self.config["max_records"],
                    "cancellation ledger full",
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                )
                entry = Entry(
                    request.operation,
                    request.original_expiry_unix_ms,
                    time.monotonic(),
                    request.native_rid,
                    cancelled=True,
                    quiesced=True,
                    terminal=True,
                )
                self.entries[request.operation.operation_id] = entry
            return pb.TerminationReceipt(recorded=True, quiesced=entry.quiesced)
        except Rejected as error:
            await context.abort(error.code, str(error))

    async def GetOperation(self, request, context):
        """Reconcile unknown acceptance with the original sequence and complete fence."""
        try:
            self._operation(request)
            entry = self.entries.get(request.operation_id)
            require(entry is not None, "unknown operation", grpc.StatusCode.NOT_FOUND)
            require(
                same_operation(entry.operation, request),
                "operation fence mismatch",
                grpc.StatusCode.FAILED_PRECONDITION,
            )
            return pb.OperationStatus(
                operation=entry.operation,
                admission_seq=entry.admission,
                stage=entry.events[-1].stage if entry.events else pb.CANCEL_REQUESTED,
                terminal=entry.terminal,
                quiesced=entry.quiesced,
            )
        except Rejected as error:
            await context.abort(error.code, str(error))

    async def WatchLoad(self, request, context):
        """Coverage and occupancy are sampled on the same owner loop as admission."""
        if request != self.instance:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "stale load instance")
        while not context.cancelled():
            active = [entry for entry in self.entries.values() if not entry.quiesced]
            self.snapshot_seq += 1
            yield pb.LoadSnapshot(
                instance=self.instance,
                snapshot_seq=self.snapshot_seq,
                covered_admission_seq=self.admission_seq,
                capacity=self.capacity
                if not self.draining and healthy(self.config["max_clock_error_ms"])
                else 0,
                active=len(active),
                cpu_work=sum(entry.cpu_work for entry in active),
                gpu_work=sum(entry.gpu_work for entry in active),
                artifact_bytes=sum(entry.artifact_bytes for entry in active),
                staging_bytes=sum(entry.staging_bytes for entry in active),
                send_bytes=sum(entry.send_bytes for entry in active),
                draining_work=sum(entry.work for entry in active if entry.cancelled),
            )
            await asyncio.sleep(self.config["load_interval_ms"] / 1000)

    async def GetCapabilities(self, request, context):
        """Publish this process fence and the explicit validated deployment contract."""
        return pb.Capabilities(
            version=1,
            instance=self.instance,
            role="encoder",
            model_revision=self.config["model_revision"],
            processor_contract=self.config["processor_contract"],
            layout_contract="sglang_native_manifest_v1",
            embedding_dtype=self.config["embedding_dtype"],
            kv_backend=self.config["kv_backend"],
            embedding_backend=self.backend.encoder.transfer_backend,
            topology_contract=self.config["topology_contract"],
            receive_endpoint=self.config["receive_endpoint"],
            capacity=self.capacity,
            tp_size=self.config["tp_size"],
            wall_time_unix_ms=int(time.time() * 1000),
            ready=not self.draining and healthy(self.config["max_clock_error_ms"]),
        )

    async def part(self, rid):
        """Bound pull-before-Encode waiters without creating speculative native state."""
        require(
            isinstance(rid, str)
            and len(rid) <= 512
            and rid.startswith(self.instance.generation + "."),
            "stale native part",
        )
        if rid not in self.parts:
            require(
                self.early_count < self.config["max_early_waiters"],
                "early facade capacity full",
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            )
            state = self.early.setdefault(rid, [asyncio.Event(), 0])
            state[1] += 1
            self.early_count += 1
            try:
                await asyncio.wait_for(state[0].wait(), self.config["early_wait_seconds"])
            finally:
                self.early_count -= 1
                state[1] -= 1
                if not state[1]:
                    self.early.pop(rid, None)
        entry = self.parts[rid]
        require(
            not entry.cancelled and not entry.quiesced,
            "native part closed",
            grpc.StatusCode.CANCELLED,
        )
        require(
            time.monotonic() < entry.deadline,
            "native facade deadline expired",
            grpc.StatusCode.DEADLINE_EXCEEDED,
        )
        return entry

    async def _receiver_control(self, entry, operation):
        """Cancel only metadata/registration waits; these own no GPU or transport buffers."""
        work = asyncio.create_task(operation)
        cancelled = asyncio.create_task(entry.cancellation.wait())
        try:
            done, _ = await asyncio.wait((work, cancelled), return_when=asyncio.FIRST_COMPLETED)
            require(
                work in done and not entry.cancelled,
                "native receiver control cancelled",
                grpc.StatusCode.CANCELLED,
            )
            return await work
        finally:
            for waiter in (work, cancelled):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(work, cancelled, return_exceptions=True)

    async def facade(self, kind, request):
        """Native receive calls and Encode resolve the exact same E operation and buffer owner."""
        from sglang.srt.disaggregation.encoder import server as native
        from sglang.srt.disaggregation.encoder.runtime import send_staged_embedding

        entry = await self.part(request.get("req_id"))
        if kind == "metadata":
            require(request.get("part_idx") == 0, "invalid part index")
            async with entry.refs.retain():
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
        entry.receivers = count
        destination = (
            request.get("receive_url")
            if kind == "register"
            else (request.get("prefill_host"), request.get("embedding_port"))
        )
        require(bool(destination), "missing native receiver")
        expected_host = entry.request.receiver_binding.bootstrap.host
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

        async def run():
            """Retain the actual send even if its HTTP response consumer disconnects."""
            async with entry.refs.retain():
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
