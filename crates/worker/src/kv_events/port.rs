//! The port between the KV event collector and the index it feeds: the
//! seven calls of a sink and the facts that cross them. The collector (one
//! subscription task a worker) reads a stream and says what it read; the
//! sink keeps the admission cursor of each rank and applies what the cursor
//! admits. Neither names the other's types: the batch and the id are the
//! sink's own.
//!
//! The policy controller's store states the same port over its own index
//! (`store::kv_events` of that crate). The two statements differ today in
//! what each host needs: the id is the sink's own type here and the
//! store's worker id there; the store holds three more reasons, a relay's
//! frames; and `Applied` is read by the cursor here (another rank's batch
//! included), by the index there.

/// The header facts of one batch, read by the cursor; the batch itself (the
/// servicer's message, with the blocks the index applies) rides beside them
/// as the host's own type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchFacts {
    /// The publisher's sequence number: monotonic per rank, each rank its
    /// own counter.
    pub sequence: u64,
    /// The batch carries the engine's own cache clear, wherever it stands
    /// among the events: the cursor's fact (a batch below the cursor that
    /// clears is a restarted publisher, since what such a batch leaves
    /// owes nothing to what came before it). The clear itself is the
    /// batch's: a collector sets this when a clear stands among what it
    /// decoded, and a sink empties the rank there.
    pub clears: bool,
    /// The batch is one chunk of a relay's state snapshot, not a batch the
    /// publisher sent.
    pub snapshot: Option<SnapshotChunk>,
}

/// One chunk of a relay's in-band snapshot, stamped with a sequence the
/// relay chose so live batches continue after the last one. Where a chunk
/// stands in the relay's count is no fact of the port: the collector counts
/// the chunks of a stream and says a snapshot's start
/// ([`ResyncReason::Snapshot`]) before the first chunk and a snapshot the
/// stream did not finish ([`ResyncReason::SnapshotAbandoned`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SnapshotChunk {
    /// The snapshot could not cover the engine's whole life: the rank's
    /// blocks are a partial set until the next resync.
    pub unknown_before: bool,
}

/// What the sink did with a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchOutcome {
    /// The cursor advanced to it: the sink took the batch, and applied it to
    /// the index where its rank is one the sink indexes for the id.
    Applied,
    /// A duplicate below the cursor, or a batch for an id the sink holds
    /// nothing for: not taken, and the cursor stays where it was.
    Skipped,
    /// A gap with no replay asked yet: `expected` is the first sequence the
    /// batch's own rank is missing. The collector subscribes again from rank
    /// 0's cursor (the servicers replay rank 0's publisher), and the batch
    /// is not applied (the replay resends it). A second gap on the same rank
    /// is applied as lost history (see [`KvCursor::degraded`]).
    Gap { expected: u64, received: u64 },
}

/// Why a rank's cursor was reset or its blocks replaced, and the label the
/// resync is counted under. This crate's collector says four: `OutOfRange`
/// and `DataLoss`, the server's own status codes, and a snapshot's two ends,
/// `Snapshot` and `SnapshotAbandoned`. A host's sink may find two with its
/// cursor, `PublisherRestart` and `GapCleared`, and count a `Snapshot`
/// where it clears for one. The store's statement holds three more, a
/// relay's frames, which no stream of this crate carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResyncReason {
    /// The server's history no longer reaches the cursor.
    OutOfRange,
    /// The server lost batches it cannot name.
    DataLoss,
    /// The stream ended or sent a malformed chunk before finishing a snapshot.
    SnapshotAbandoned,
    /// A sequence far below the cursor, or a clear below it: the publisher
    /// started over and its cache is empty.
    PublisherRestart,
    /// A gap the server could not replay, too large to keep the state.
    GapCleared,
    /// A relay served a state snapshot in place of history: the live set it
    /// carries replaces the blocks of every rank of the engine. The
    /// collector says it once, before the admit of the snapshot's first
    /// chunk.
    Snapshot,
}

impl ResyncReason {
    /// The metric label of the reason.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutOfRange => "out_of_range",
            Self::DataLoss => "data_loss",
            Self::SnapshotAbandoned => "snapshot_abandoned",
            Self::PublisherRestart => "publisher_restart",
            Self::GapCleared => "gap_cleared",
            Self::Snapshot => "snapshot",
        }
    }
}

/// One rank's cursor as a subscriber reads it before it (re)subscribes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvCursor {
    /// The last sequence applied, `0` for none: servers resume strictly
    /// after it.
    pub applied: u64,
    /// A replay was asked for and its batches have not arrived.
    pub replay_pending: bool,
    /// A gap could not be recovered: the rank's blocks are a partial set
    /// until a resync.
    pub degraded: bool,
}

/// Where KV event batches are admitted and applied.
///
/// The cursor rules, per `(worker, rank)`: the first batch sets the cursor
/// wherever it is; a contiguous batch applies; a batch at or below the cursor
/// is a duplicate and is skipped, unless it carries a clear, sits a restart
/// window (1,024) or more below, is a counter at its start, or is the first
/// batch of a fresh connection below the cursor, which read as a restarted
/// publisher: the rank's blocks are cleared and the batch applied; a batch
/// past the cursor is a gap answered once by a replay request from the
/// expected sequence; a second gap is lost history, applied with the rank
/// marked degraded (its state kept under 1,024 missed batches, cleared
/// past). A snapshot chunk is applied outside these rules and moves the
/// cursor to its stamp.
///
/// The sink's unit is what `id` names: the host hands the collector the id
/// with the worker at its subscription, and the collector gives it back on
/// every call. `rank` is the rank a batch names. What is a stream's the
/// collector says to the id: a new connection, a snapshot's start (a resync
/// for `Snapshot`, before the admit of its first chunk) and a snapshot the
/// stream did not finish.
pub trait KvEventSink: Send + Sync {
    /// The batch as the collector's wire crate defines it; a sink reads of
    /// it what its own index needs.
    type Batch: ?Sized;

    /// The sink's name for what one subscription feeds.
    type Id: Copy + Send + Sync + 'static;

    /// Run one batch of `rank` through its cursor and, when admitted, apply
    /// it to the worker's blocks. `facts` are the batch's header as the
    /// cursor reads it. A load-only batch carries no events and is not
    /// admitted: its record goes to the load sink.
    fn admit(
        &self,
        id: Self::Id,
        rank: u16,
        facts: BatchFacts,
        batch: &Self::Batch,
    ) -> BatchOutcome;

    /// A new stream connection was made for the worker: the first batch on
    /// it shows where the publisher stands.
    fn reconnected(&self, id: Self::Id);

    /// Reset the id's cursors for `reason`: its blocks go with them, and the
    /// next stream is taken from wherever it starts. A `rank` the id does
    /// not stand for resets nothing; `None` asks no check. A snapshot's two
    /// ends are a sink's to time: its start may clear in the first chunk's
    /// own batch, and an abandoned one may keep its blocks for the next
    /// snapshot to replace.
    fn resync(&self, id: Self::Id, rank: Option<u16>, reason: ResyncReason);

    /// The cursor of one rank, when the rank has been heard from.
    fn cursor(&self, id: Self::Id, rank: u16) -> Option<KvCursor>;

    /// The worker left: its blocks out of the index, its cursors gone. A
    /// same-key successor is interned only after this returns.
    fn removed(&self, id: Self::Id);

    /// The block size of record for a model, once a stored block or a
    /// registration named it.
    fn block_size(&self, model: &str) -> Option<u32>;

    /// Name a model's block size at a registration: provisional until a
    /// stored block confirms it.
    fn set_block_size(&self, model: &str, size: u32);
}
