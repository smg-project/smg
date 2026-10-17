"""The load record a servicer's KV-event stream carries, shared by every
engine bridge in this package.

A gateway that subscribes to ``SubscribeKvEvents`` reads an ``EngineLoad``
on every ``KvEventBatch`` (``common.proto``): the same figures ``GetLoads``
answers with, at every scheduler step instead of at its poll, and a
``load_only`` batch (no events, the last sequence repeated) while the engine
is idle. The gateway counts every record as contact with the worker; a
worker whose records stop is excluded from routing after its staleness
threshold, and its rules that read the record (the engine-silence veto, the
pending-prefill allowance) apply only to workers that push one. A stream
without the record leaves its worker polled, judged by the transport's
keepalive alone.

This module mirrors the Rust servicer's relay (``crates/engine_servicer``,
``kv_events.rs``) field for field and tick for tick, so a gateway sees the
Python and the Rust servicer of one engine alike:

- :func:`engine_load_from` converts an engine's ``SchedulerLoad`` message
  (any of the three protos) to the record, as ``crates/grpc_client``'s
  ``engine_load.rs`` does;
- :class:`LoadRecorder` numbers and paces one stream's records: the record
  on every batch, telemetry on the first record and the heartbeats only,
  a ``load_only`` batch when the record moved or after
  :data:`HEARTBEAT_INTERVAL` of silence (:data:`HEARTBEAT_BACKOFF` once the
  engine has been idle for two heartbeats), checked every :data:`LOAD_TICK`;
  a source it cannot read leaves the batch without a record, the outage
  logged once and its end once.

Imports only the stdlib and the generated proto, so it is unit-tested
without any engine installed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from smg_grpc_proto.generated import common_pb2

logger = logging.getLogger(__name__)

__all__ = [
    "HEARTBEAT_BACKOFF",
    "HEARTBEAT_INTERVAL",
    "LOAD_TICK",
    "LoadRecorder",
    "LoadSource",
    "core_only",
    "engine_load_from",
    "load_changed",
    "poll_timeout_for",
]

#: How often an idle stream checks the record for a change (seconds).
LOAD_TICK = 0.1
#: Silence after which a ``load_only`` heartbeat goes out (seconds).
HEARTBEAT_INTERVAL = 1.0
#: The heartbeat interval once two heartbeats in a row carried an unchanged
#: record (seconds).
HEARTBEAT_BACKOFF = 5.0

#: The servicer's ``GetLoads`` figures for one rank (``None`` for a batch that
#: names none), as the record, or ``None`` while nothing is known (the engine
#: is not up yet). ``sample`` and ``load_only`` are the recorder's to set.
LoadSource = Callable[["int | None"], "common_pb2.EngineLoad | None"]

#: The optional sections the SGLang and TokenSpeed reports carry, under the
#: same names as the record's.
_SECTIONS = ("memory", "queues", "speculative", "lora", "disaggregation")
#: The record's telemetry beyond the routing core: what rides the heartbeats
#: and the first record, not every event batch.
_TELEMETRY = ("cache_hit_rate", "num_used_tokens", "max_total_num_tokens", *_SECTIONS)


def _unsigned(value: object) -> int:
    """A count as the record's ``uint32`` takes it: a negative reads as 0."""
    return max(0, int(value or 0))


def _copy_section(source: object, target: object) -> None:
    for field in target.DESCRIPTOR.fields:
        if field.name in source.DESCRIPTOR.fields_by_name:
            setattr(target, field.name, getattr(source, field.name))


def engine_load_from(load: object, *, queued_token_work: bool) -> common_pb2.EngineLoad:
    """The record from an engine's ``SchedulerLoad`` message.

    The core (running, waiting, KV usage, generation rate, admission window),
    the cache hit rate and the token counts come from every engine's report;
    ``active_token_usage`` and the sections (memory, queues, speculative,
    LoRA, disaggregation) when the message carries them. The queued
    token-work (``waiting_uncached_tokens``) is set only when the servicer
    reports it on ``GetLoads`` (``queued_token_work``); absent is distinct
    from zero to the gateway.
    """
    record = common_pb2.EngineLoad(
        running_requests=_unsigned(load.num_running_reqs),
        waiting_requests=_unsigned(load.num_waiting_reqs),
        token_usage=float(load.token_usage),
        gen_throughput=float(load.gen_throughput),
        max_running_requests=_unsigned(load.max_running_requests),
        cache_hit_rate=float(load.cache_hit_rate),
        num_used_tokens=int(load.num_used_tokens),
        max_total_num_tokens=int(load.max_total_num_tokens),
    )
    if queued_token_work:
        record.waiting_uncached_tokens = _unsigned(load.num_waiting_uncached_tokens)
    fields = load.DESCRIPTOR.fields_by_name
    if "active_token_usage" in fields and load.HasField("active_token_usage"):
        record.active_token_usage = load.active_token_usage
    for name in _SECTIONS:
        if name in fields and load.HasField(name):
            _copy_section(getattr(load, name), getattr(record, name))
    return record


def core_only(record: common_pb2.EngineLoad) -> None:
    """Clear the telemetry: what an event batch carries between heartbeats."""
    for name in _TELEMETRY:
        record.ClearField(name)


def _queued(record: common_pb2.EngineLoad) -> int | None:
    if record.HasField("waiting_uncached_tokens"):
        return record.waiting_uncached_tokens
    return None


def _active(record: common_pb2.EngineLoad) -> float:
    if record.HasField("active_token_usage"):
        return record.active_token_usage
    return record.token_usage


def load_changed(last: common_pb2.EngineLoad, current: common_pb2.EngineLoad) -> bool:
    """Whether the record moved enough from ``last`` to be worth a
    ``load_only`` batch: any queue, running or window change, KV usage by
    half a percent, the rate by 5 % or 50 tokens per second."""
    if (
        last.running_requests != current.running_requests
        or last.waiting_requests != current.waiting_requests
        or _queued(last) != _queued(current)
        or last.max_running_requests != current.max_running_requests
        or abs(last.token_usage - current.token_usage) > 0.005
        or abs(_active(last) - _active(current)) > 0.005
    ):
        return True
    delta = abs(last.gen_throughput - current.gen_throughput)
    return delta > 50.0 or delta > 0.05 * max(last.gen_throughput, current.gen_throughput)


def _rank_name(rank: int | None) -> str:
    """How the log names a rank; a batch that names none is the stream's own."""
    return "the stream (no rank)" if rank is None else f"rank {rank}"


class LoadRecorder:
    """One subscriber's load records: numbered, telemetry-trimmed and paced.

    :meth:`stamp` attaches the record to every batch the stream sends;
    :meth:`heartbeat` is asked between batches (every :attr:`load_tick`, see
    :meth:`poll_timeout`) and answers a ``load_only`` batch when one is due.
    ``source`` is the servicer's :data:`LoadSource`; ``clock`` is monotonic
    seconds (tests pass their own). A source that raises leaves the batch
    without a record and the heartbeat unsent (see :meth:`_read`).
    """

    def __init__(
        self,
        source: LoadSource,
        *,
        load_tick: float = LOAD_TICK,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        heartbeat_backoff: float = HEARTBEAT_BACKOFF,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self.load_tick = load_tick
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_backoff = heartbeat_backoff
        self._clock = clock
        self._sample = 0
        # The last record sent, for the change check.
        self._last_record: common_pb2.EngineLoad | None = None
        self._last_sent_at = clock()
        # Heartbeats in a row with an unchanged record.
        self._idle_heartbeats = 0
        # The rank and sequence of the last batch sent: what a heartbeat names.
        self._last_rank: int | None = None
        self._last_sequence: int | None = None
        # Failed reads of the source in a row, per rank: the outages being reported.
        self._failed_reads: dict[int | None, int] = {}

    @classmethod
    def optional(
        cls,
        source: LoadSource | None,
        *,
        load_tick: float = LOAD_TICK,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        heartbeat_backoff: float = HEARTBEAT_BACKOFF,
    ) -> LoadRecorder | None:
        """A recorder over ``source`` at the given intervals, or ``None``
        without a source: the stream then carries cache events only and the
        gateway keeps polling the worker."""
        if source is None:
            return None
        return cls(
            source,
            load_tick=load_tick,
            heartbeat_interval=heartbeat_interval,
            heartbeat_backoff=heartbeat_backoff,
        )

    def poll_timeout(self, recv_timeout: float) -> float:
        """How long the stream may wait for the next batch before the record
        is checked again."""
        return min(recv_timeout, self.load_tick)

    def _read(self, rank: int | None) -> common_pb2.EngineLoad | None:
        """The source's record for ``rank``: ``None`` when it has none yet, and
        when it cannot be read, in which case the batch goes without a record
        (the gateway keeps the worker polled). The first failed read of an
        outage is a warning with the traceback, the repeats debug lines, and
        the read that ends it an info line with their count: a read that keeps
        failing (a snapshot gone, a reader whose shape changed) is one loud
        line at the default level, not a flood and not silence. The count is
        kept per rank: one recorder serves every rank of a relayed stream, and
        a rank that fails while another reads fine is one outage, not one per
        interleaved batch."""
        try:
            record = self._source(rank)
        except Exception:  # noqa: BLE001 - a record that cannot be read leaves the batch without one
            failed = self._failed_reads.get(rank, 0) + 1
            self._failed_reads[rank] = failed
            if failed == 1:
                logger.warning(
                    "load record unavailable for %s; the KV-event stream carries cache events "
                    "only until it can be read again",
                    _rank_name(rank),
                    exc_info=True,
                )
            else:
                logger.debug(
                    "load record unavailable for %s (%d in a row)",
                    _rank_name(rank),
                    failed,
                    exc_info=True,
                )
            return None
        failed = self._failed_reads.pop(rank, 0)
        if failed:
            logger.info(
                "load record for %s available again after %d failed read(s)",
                _rank_name(rank),
                failed,
            )
        return record

    def _numbered(
        self, current: common_pb2.EngineLoad, *, load_only: bool
    ) -> common_pb2.EngineLoad:
        record = common_pb2.EngineLoad()
        record.CopyFrom(current)
        self._sample += 1
        record.sample = self._sample
        record.load_only = load_only
        # The telemetry rides the heartbeats and the stream's first record; an
        # event batch, one per scheduler step, carries the core only.
        if not load_only and self._sample > 1:
            core_only(record)
        self._last_record = current
        return record

    def stamp(self, batch: common_pb2.KvEventBatch) -> common_pb2.KvEventBatch:
        """Attach the record for the batch's rank to a batch about to go out
        and note it went (the batch is returned for convenience)."""
        rank = batch.dp_rank if batch.HasField("dp_rank") else None
        self._last_rank = rank
        self._last_sequence = batch.sequence_number
        self._last_sent_at = self._clock()
        self._idle_heartbeats = 0
        current = self._read(rank)
        if current is not None:
            batch.load.CopyFrom(self._numbered(current, load_only=False))
        return batch

    def heartbeat(self) -> common_pb2.KvEventBatch | None:
        """A ``load_only`` batch when the record moved since the last one
        sent, or when the stream has been silent for the heartbeat interval
        (the backoff interval after two unchanged heartbeats); ``None``
        otherwise. Nothing before the first batch went out: a heartbeat
        would have no sequence to repeat."""
        if self._last_sequence is None:
            return None
        current = self._read(self._last_rank)
        if current is None:
            return None
        changed = self._last_record is None or load_changed(self._last_record, current)
        due_after = (
            self.heartbeat_backoff if self._idle_heartbeats >= 2 else self.heartbeat_interval
        )
        if not changed and self._clock() - self._last_sent_at < due_after:
            return None
        record = self._numbered(current, load_only=True)
        self._last_sent_at = self._clock()
        self._idle_heartbeats = 0 if changed else self._idle_heartbeats + 1
        batch = common_pb2.KvEventBatch(sequence_number=self._last_sequence, timestamp=time.time())
        if self._last_rank is not None:
            batch.dp_rank = self._last_rank
        batch.load.CopyFrom(record)
        return batch


def poll_timeout_for(recorder: LoadRecorder | None, recv_timeout: float) -> float:
    """How long a stream with ``recorder`` (or none) may wait for the next
    batch: the recorder's :meth:`LoadRecorder.poll_timeout`, else
    ``recv_timeout`` as given."""
    return recorder.poll_timeout(recv_timeout) if recorder is not None else recv_timeout
