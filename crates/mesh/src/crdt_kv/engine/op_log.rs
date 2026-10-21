//! The op-log keeper the engines share.
//!
//! An engine's CRDT strategy is its merge rule and its per-key fold; what
//! surrounds them is not strategy and is kept here once: the operation log
//! behind its lock, the log's mutation counter, the compaction trigger, the
//! local-write path's append-compact-truncate with its overflow warning, the
//! remote-merge path's append-compact (never truncate), the GC path's retain,
//! and the change emission after a remote batch.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use parking_lot::{RwLock, RwLockReadGuard};

use crate::crdt_kv::operation::{CrdtChange, Operation, OperationLog};

/// An engine's operation log with its mutation counter and the log rules
/// that are not strategy. The engine supplies its per-key fold and its
/// tracked key count; the keeper applies the trigger, the compaction and the
/// overflow backstop the same way for every engine.
pub(super) struct OpLogKeeper {
    log: RwLock<OperationLog>,
    /// Bumped on every log mutation (including relay-only appends);
    /// invalidation key for shared op-log snapshots.
    generation: AtomicU64,
    /// Named in the overflow warning.
    engine: &'static str,
}

impl OpLogKeeper {
    pub(super) fn new(engine: &'static str) -> Self {
        Self {
            log: RwLock::new(OperationLog::new()),
            generation: AtomicU64::new(0),
            engine,
        }
    }

    /// Compact once the log carries more than twice the tracked key count
    /// (min 64). Tracked means live and tombstoned: every key the engine
    /// still keeps metadata for, which is every key a compacted log holds an
    /// op for, so the trigger stays above what compaction can reach and
    /// resident log memory is bounded at O(keys) while the fold is amortized.
    /// A live-only count (the store's) would put the trigger under the
    /// compacted log's size while tombstones await GC, and every local write
    /// would compact in vain.
    fn compact_trigger(tracked_keys: usize) -> usize {
        (tracked_keys * 2).max(64)
    }

    /// The local-write path: append `op`, and past the trigger compact the
    /// log with `fold` (one op per key), then drop the oldest entries over
    /// the memory backstop with one warning. `tracked_keys` is the engine's
    /// count of keys with metadata, live and tombstoned (see
    /// `compact_trigger`), read only once the log is past the trigger's
    /// floor, so a short log skips the key count (a DashMap len() is an
    /// O(shards) scan) on this per-write path.
    /// Truncation belongs to this path alone: dropping the oldest on the
    /// remote-merge path would shed remotely-learned keys (see
    /// `OperationLog::truncate_oldest_over_threshold`).
    pub(super) fn append(
        &self,
        op: Operation,
        tracked_keys: impl FnOnce() -> usize,
        fold: impl Fn(&[Operation]) -> Option<Operation>,
    ) {
        let mut log = self.log.write();
        log.append(op);
        self.generation.fetch_add(1, Ordering::Release);
        if log.len() > 64 && log.len() > Self::compact_trigger(tracked_keys()) {
            log.compact_by_key(fold);
            let dropped = log.truncate_oldest_over_threshold();
            if dropped > 0 {
                tracing::warn!(
                    engine = self.engine,
                    dropped,
                    total = log.len(),
                    "engine log over threshold after compaction; dropped oldest entries \
                     (out-of-spec distinct-key count)"
                );
            }
        }
    }

    /// The remote-merge path: append the batch and compact it with `fold`
    /// under one write lock and one generation bump. Never truncates: that
    /// would drop remotely-learned keys that are live in state.
    pub(super) fn merge_batch(
        &self,
        ops: &[Operation],
        fold: impl Fn(&[Operation]) -> Option<Operation>,
    ) {
        let mut log = self.log.write();
        for op in ops {
            log.append(op.clone());
        }
        log.compact_by_key(fold);
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// The GC path: keep the ops `keep` says, one generation bump.
    pub(super) fn retain(&self, keep: impl FnMut(&Operation) -> bool) {
        let mut log = self.log.write();
        log.retain_ops(keep);
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// A read view of the log, for a scan that decides before it writes.
    pub(super) fn read(&self) -> RwLockReadGuard<'_, OperationLog> {
        self.log.read()
    }

    /// Every operation the log retains, in its order.
    pub(super) fn export(&self) -> Vec<Operation> {
        self.log.read().operations().to_vec()
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

/// The observable value of every key `ops` touch, read once per key through
/// `current` before the batch applies, so the engine can emit a change only
/// where `get` changes.
pub(super) fn values_before(
    ops: &[Operation],
    current: impl Fn(&str) -> Option<Vec<u8>>,
) -> HashMap<String, Option<Vec<u8>>> {
    let mut before: HashMap<String, Option<Vec<u8>>> = HashMap::new();
    for op in ops {
        // `contains_key` with a borrowed key avoids allocating a `String`
        // for keys already snapshotted (a batch may repeat a key).
        if !before.contains_key(op.key()) {
            before.insert(op.key().to_string(), current(op.key()));
        }
    }
    before
}

/// One [`CrdtChange`] per key whose observable value differs from its entry
/// in `before`, carrying the canonical post-merge value `current` gives (what
/// `get` returns). Keying off the observable value, not op acceptance,
/// suppresses spurious events from a newer-version op that rewrites
/// byte-identical bytes or from a dominated op.
pub(super) fn changes(
    before: HashMap<String, Option<Vec<u8>>>,
    current: impl Fn(&str) -> Option<Vec<u8>>,
) -> Vec<CrdtChange> {
    before
        .into_iter()
        .filter_map(|(key, prior)| {
            let value = current(&key);
            (value != prior).then_some(CrdtChange { key, value })
        })
        .collect()
}
