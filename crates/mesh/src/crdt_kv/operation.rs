use serde::{Deserialize, Serialize};

use super::replica::ReplicaId;

// ============================================================================
// Operation Type Definition - Atomic Unit of State Change
// ============================================================================

/// CRDT operation type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation {
    /// Insert operation: key, value, timestamp, replica_id
    Insert {
        key: String,
        value: Vec<u8>,
        timestamp: u64,
        replica_id: ReplicaId,
    },
    /// Remove operation: key, timestamp, replica_id
    Remove {
        key: String,
        timestamp: u64,
        replica_id: ReplicaId,
    },
}

impl Operation {
    /// Create insert operation
    pub fn insert(key: String, value: Vec<u8>, timestamp: u64, replica_id: ReplicaId) -> Self {
        Self::Insert {
            key,
            value,
            timestamp,
            replica_id,
        }
    }

    /// Create remove operation
    pub fn remove(key: String, timestamp: u64, replica_id: ReplicaId) -> Self {
        Self::Remove {
            key,
            timestamp,
            replica_id,
        }
    }

    /// Get the key of the operation
    pub fn key(&self) -> &str {
        match self {
            Self::Insert { key, .. } => key,
            Self::Remove { key, .. } => key,
        }
    }

    /// Get the timestamp of the operation
    pub fn timestamp(&self) -> u64 {
        match self {
            Self::Insert { timestamp, .. } => *timestamp,
            Self::Remove { timestamp, .. } => *timestamp,
        }
    }

    /// Get the replica ID of the operation
    pub fn replica_id(&self) -> ReplicaId {
        match self {
            Self::Insert { replica_id, .. } => *replica_id,
            Self::Remove { replica_id, .. } => *replica_id,
        }
    }
}

/// An observable change to a key's live value produced by a remote merge.
/// `value` is the canonical post-merge value (matching `get(key)`): `Some` for
/// a key that is live after the merge, `None` for a key that is now tombstoned.
/// Engines emit one `CrdtChange` per key whose live value actually changed, so
/// the merge caller can fire subscribers with the same value shape `get`
/// returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrdtChange {
    pub key: String,
    pub value: Option<Vec<u8>>,
}

// ============================================================================
// Operation Log - State Operation Pipeline
// ============================================================================

/// Strategy-agnostic append-only log of CRDT operations.
///
/// Each engine owns one `OperationLog` instance holding only its own
/// namespace's operations; the log itself carries no merge-strategy knowledge.
/// Engines drive merge (op-id collision policy) and compaction (per-key fold
/// rule) themselves via [`Self::operations`], [`Self::append`], and
/// [`Self::compact_by_key`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationLog {
    operations: Vec<Operation>,
}

impl OperationLog {
    /// Threshold at which engine-driven auto-compaction triggers. Engines
    /// check this against `len()` after each append and run their own
    /// per-key fold to bring the log back down.
    pub(super) const AUTO_COMPACT_THRESHOLD: usize = 10_000;

    /// Create empty operation log
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Build an operation log from a pre-collected vector. Used by the
    /// engine router to concatenate per-engine ops back into a single log
    /// for gossip export.
    pub(super) fn from_operations(operations: Vec<Operation>) -> Self {
        Self { operations }
    }

    /// Append an operation. Strategy-free: engines call this then run their
    /// own compaction policy on whatever threshold they want.
    pub fn append(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    /// Get all operations
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Memory backstop: if the log is still over `AUTO_COMPACT_THRESHOLD`
    /// after the caller has compacted it, drop the oldest entries down to 75%
    /// of the threshold. Returns the number of entries dropped (0 if under
    /// threshold) so the caller can log with its own context.
    ///
    /// This only fires when distinct live keys exceed the threshold - far
    /// beyond the design's expected key count. It trades convergence for the
    /// dropped keys against unbounded log growth; that trade is only
    /// acceptable because reaching it signals an out-of-spec key count in the
    /// first place.
    ///
    /// Callers MUST invoke this only on the local-write path. Dropping on a
    /// remote-merge path would shed remotely-learned keys that are live in
    /// state, breaking downstream sync from this node.
    pub(super) fn truncate_oldest_over_threshold(&mut self) -> usize {
        if self.operations.len() <= Self::AUTO_COMPACT_THRESHOLD {
            return 0;
        }
        let keep = Self::AUTO_COMPACT_THRESHOLD * 3 / 4;
        let drain_count = self.operations.len() - keep;
        self.operations.drain(..drain_count);
        drain_count
    }

    /// Drop every operation for which `keep` returns `false`. Used by
    /// tombstone GC to purge collected keys' dominated ops from the log.
    pub(super) fn retain_ops(&mut self, keep: impl FnMut(&Operation) -> bool) {
        self.operations.retain(keep);
    }

    /// Serialize to bincode bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<bincode::ErrorKind>> {
        bincode::serialize(self)
    }

    /// Deserialize from bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<bincode::ErrorKind>> {
        bincode::deserialize(bytes)
    }

    /// Get number of operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Group operations by key, fold each group via `fold`, and replace the
    /// log with the resulting per-key winners sorted by `(timestamp,
    /// replica_id)`. Strategy-agnostic - the caller's `fold` decides what
    /// "winner" means (LWW: max by version; EpochMaxWins:
    /// `epoch_max_wins::compact_operations`).
    ///
    /// Grouping is done by sorting in-place and scanning contiguous runs,
    /// avoiding the `String` allocation per op that a `HashMap<String, _>`
    /// grouping would require.
    pub(super) fn compact_by_key<F>(&mut self, fold: F)
    where
        F: Fn(&[Operation]) -> Option<Operation>,
    {
        self.operations
            .sort_unstable_by(|a, b| a.key().cmp(b.key()));
        let mut folded: Vec<Operation> = Vec::with_capacity(self.operations.len());
        let mut start = 0;
        while start < self.operations.len() {
            let key = self.operations[start].key();
            let mut end = start + 1;
            while end < self.operations.len() && self.operations[end].key() == key {
                end += 1;
            }
            if let Some(winner) = fold(&self.operations[start..end]) {
                folded.push(winner);
            }
            start = end;
        }
        folded.sort_by_key(|op| (op.timestamp(), op.replica_id()));
        self.operations = folded;
    }
}

impl Default for OperationLog {
    fn default() -> Self {
        Self::new()
    }
}
