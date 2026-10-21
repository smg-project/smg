//! CRDT OR-Map router.
//!
//! [`CrdtOrMap`] used to host both LWW and EpochMaxWins logic inline, with a
//! per-prefix strategy table that branched at every entry point. The bug
//! pattern in PR #1469 traced back to that shared shape: any strategy-specific
//! invariant had to be threaded through every shared call site, and missing
//! one site was how the same bug would resurface in a different form.
//!
//! `CrdtOrMap` is now a thin router over per-namespace engines (see
//! [`engine`](super::engine)). Each registered prefix gets its own engine
//! with its own state, log, clock, and metadata; the router just matches
//! keys to engines by longest-prefix-match and delegates. Unregistered keys
//! fall through to a built-in default LWW engine, preserving today's
//! "default LWW" semantics for callers that never call
//! `register_merge_strategy`.

use std::{cmp::Reverse, collections::BTreeMap, sync::Arc, time::Duration};

use parking_lot::RwLock;
use tracing::info;

use super::{
    engine::{EngineHandle, LwwEngine, RateLimitEngine},
    merge_strategy::MergeStrategy,
    operation::{CrdtChange, Operation, OperationLog},
    replica::{LamportClock, ReplicaId},
};

/// Default tombstone grace period for [`CrdtOrMap::gc_tombstones`]. Forwarded
/// to each engine's GC.
pub const DEFAULT_TOMBSTONE_GRACE: Duration = Duration::from_secs(300);

type EngineTable = Arc<[(String, EngineHandle)]>;

/// CRDT OR-Map. Routes operations to per-namespace engines by prefix.
#[derive(Clone)]
pub struct CrdtOrMap {
    /// Engines explicitly registered via `register_merge_strategy`, sorted by
    /// `Reverse(prefix.len())` so longest-prefix-match wins.
    engines: Arc<RwLock<EngineTable>>,
    /// Catch-all engine for keys not matching any registered prefix. LWW for
    /// backward compatibility with callers that never register a prefix
    /// (notably the in-crate tests).
    default_engine: EngineHandle,
    /// Per-node Lamport clock, shared across every engine. Op-id
    /// `(replica_id, timestamp)` must be unique across this node's operations
    /// regardless of which engine produced them; otherwise a peer that routes
    /// two op-id-colliding ops into one engine deduplicates them.
    clock: Arc<LamportClock>,
    replica_id: ReplicaId,
    /// Cached gossip snapshot keyed by the sum of engine op-generations.
    /// Rounds with no log mutations share one `Arc` instead of deep-cloning
    /// every op (the log carries full values) once per second.
    op_snapshot: OpSnapshotCache,
}

/// `(generation, snapshot)` cache slot for [`CrdtOrMap::operation_log_snapshot`].
type OpSnapshotCache = Arc<RwLock<Option<(u64, Arc<OperationLog>)>>>;

impl CrdtOrMap {
    pub fn new() -> Self {
        Self::with_replica_id(ReplicaId::new())
    }

    pub fn with_replica_id(replica_id: ReplicaId) -> Self {
        info!("Creating CRDT OR-Map, Replica ID: {}", replica_id);
        let clock = Arc::new(LamportClock::new());
        Self {
            engines: Arc::new(RwLock::new(Arc::from(Vec::new()))),
            default_engine: Arc::new(LwwEngine::new(replica_id, Arc::clone(&clock))),
            clock,
            replica_id,
            op_snapshot: Arc::new(RwLock::new(None)),
        }
    }

    /// Register the merge strategy for a key prefix. One-shot: each prefix
    /// must be registered exactly once over the lifetime of a `CrdtOrMap`.
    /// `MeshKV::configure_crdt_prefix` enforces this at the public boundary;
    /// this assert backstops in-crate callers so a re-register attempt fails
    /// loudly instead of silently orphaning the data already routed under
    /// that prefix.
    pub(crate) fn register_merge_strategy(&self, prefix: String, strategy: MergeStrategy) {
        let engine: EngineHandle = match strategy {
            MergeStrategy::LastWriterWins => {
                Arc::new(LwwEngine::new(self.replica_id, Arc::clone(&self.clock)))
            }
            MergeStrategy::EpochMaxWins => Arc::new(RateLimitEngine::new(
                self.replica_id,
                Arc::clone(&self.clock),
            )),
        };
        let mut guard = self.engines.write();
        assert!(
            !guard.iter().any(|(p, _)| p == &prefix),
            "prefix '{prefix}' is already registered; register_merge_strategy is one-shot",
        );
        let mut next: Vec<(String, EngineHandle)> = guard.iter().cloned().collect();
        next.push((prefix, engine));
        next.sort_by_key(|(prefix, _)| Reverse(prefix.len()));
        *guard = Arc::from(next);
    }

    fn engines_snapshot(&self) -> EngineTable {
        Arc::clone(&self.engines.read())
    }

    /// Return the engine handle a key routes to. Falls back to the default
    /// LWW engine if no registered prefix matches.
    fn engine_for_key(&self, key: &str) -> EngineHandle {
        let engines = self.engines_snapshot();
        for (prefix, engine) in engines.iter() {
            if key.starts_with(prefix.as_str()) {
                return Arc::clone(engine);
            }
        }
        Arc::clone(&self.default_engine)
    }

    /// Collect every engine (registered + default) so callers can fan reads
    /// across all of them.
    fn all_engines(&self) -> Vec<EngineHandle> {
        let engines = self.engines_snapshot();
        let mut out = Vec::with_capacity(engines.len() + 1);
        for (_, engine) in engines.iter() {
            out.push(Arc::clone(engine));
        }
        out.push(Arc::clone(&self.default_engine));
        out
    }

    // ---- Local writes ----
    //
    // Crate-private. External callers route through `CrdtNamespace::put` /
    // `delete`, which assert the key matches the namespace's registered
    // prefix - making "write before register" structurally unreachable from
    // outside the crate.

    pub(crate) fn insert(&self, key: String, value: Vec<u8>) -> Option<Vec<u8>> {
        self.engine_for_key(&key).put_local(&key, value)
    }

    pub(crate) fn remove(&self, key: &str) -> Option<Vec<u8>> {
        self.engine_for_key(key).delete_local(key)
    }

    // ---- Reads ----

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.engine_for_key(key).get(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.engine_for_key(key).contains_key(key)
    }

    /// Mutation generation counter. Sums per-engine generations: any change
    /// in any engine increments the sum monotonically, so callers that key
    /// off `generation()` to detect "anything changed" still work.
    pub fn generation(&self) -> u64 {
        self.all_engines().iter().map(|e| e.generation()).sum()
    }

    pub fn keys(&self) -> Vec<String> {
        let mut all = Vec::new();
        for engine in self.all_engines() {
            all.extend(engine.keys());
        }
        all
    }

    pub fn all(&self) -> BTreeMap<String, Vec<u8>> {
        let mut all = BTreeMap::new();
        for engine in self.all_engines() {
            for key in engine.keys() {
                if let Some(value) = engine.get(&key) {
                    all.insert(key, value);
                }
            }
        }
        all
    }

    pub fn len(&self) -> usize {
        self.all_engines().iter().map(|e| e.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn replica_id(&self) -> ReplicaId {
        self.replica_id
    }

    // ---- Tombstone GC ----

    pub fn gc_tombstones(&self) -> usize {
        self.gc_tombstones_with_grace(DEFAULT_TOMBSTONE_GRACE)
    }

    pub fn gc_tombstones_with_grace(&self, grace: Duration) -> usize {
        self.all_engines()
            .iter()
            .map(|e| e.gc_tombstones(grace))
            .sum()
    }

    // ---- Replication ----

    /// Snapshot the operation log seen by gossip. Concatenates each engine's
    /// log into a single [`OperationLog`].
    pub fn get_operation_log(&self) -> OperationLog {
        let mut ops = Vec::new();
        for engine in self.all_engines() {
            ops.extend(engine.export_ops());
        }
        OperationLog::from_operations(ops)
    }

    /// Shared gossip snapshot of the operation log. Rebuilt only when some
    /// engine's log mutated since the cached build; unchanged rounds return
    /// the same `Arc` (an idle node's 1 Hz round clones nothing). A racing
    /// mutation between the generation read and the rebuild can only make
    /// the cached snapshot newer than its key, forcing one extra rebuild —
    /// never a stale serve, since every mutation strictly increases the sum.
    pub fn operation_log_snapshot(&self) -> Arc<OperationLog> {
        // Sum over the engine table directly: `all_engines` would allocate a
        // Vec of handles on every 1 Hz round just to read the counters.
        let engines = self.engines_snapshot();
        let generation: u64 = engines
            .iter()
            .map(|(_, engine)| engine.op_generation())
            .sum::<u64>()
            + self.default_engine.op_generation();
        if let Some((cached_gen, snapshot)) = self.op_snapshot.read().as_ref() {
            if *cached_gen == generation {
                return Arc::clone(snapshot);
            }
        }
        let fresh = Arc::new(self.get_operation_log());
        *self.op_snapshot.write() = Some((generation, Arc::clone(&fresh)));
        fresh
    }

    /// Merge an incoming operation log. Groups ops by destination engine
    /// (longest-prefix-match) and hands each engine its slice. Engines
    /// canonicalise (merge → compact → apply) internally, so dominated ops
    /// in the incoming batch never reach live state.
    pub fn merge(&self, log: &OperationLog) -> Vec<CrdtChange> {
        info!(
            "Merging {} operations into replica {}",
            log.len(),
            self.replica_id
        );

        let engines = self.engines_snapshot();
        // Bucket index: 0..engines.len() for registered prefixes,
        // engines.len() for the default engine.
        let default_idx = engines.len();
        let mut buckets: Vec<Vec<Operation>> = (0..=default_idx).map(|_| Vec::new()).collect();

        for op in log.operations() {
            let idx = engines
                .iter()
                .position(|(prefix, _)| op.key().starts_with(prefix.as_str()))
                .unwrap_or(default_idx);
            buckets[idx].push(op.clone());
        }

        // Concatenate each engine's changed-key deltas so the caller can fire
        // subscribers once for the whole merge.
        let mut changes = Vec::new();
        for (idx, ops) in buckets.into_iter().enumerate() {
            if ops.is_empty() {
                continue;
            }
            let engine = if idx == default_idx {
                Arc::clone(&self.default_engine)
            } else {
                Arc::clone(&engines[idx].1)
            };
            changes.extend(engine.apply_remote_ops(ops));
        }
        changes
    }
}

impl Default for CrdtOrMap {
    fn default() -> Self {
        Self::new()
    }
}
