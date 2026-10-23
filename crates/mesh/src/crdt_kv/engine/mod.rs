//! Namespace CRDT engines.
//!
//! Each namespace (`worker:`, `rl:`, `config:`, ...) is owned by exactly one
//! engine that implements [`NamespaceCrdtEngine`]. The engine owns its live
//! state, metadata, operation log, per-key locks, and logical clock - all the
//! invariants that make its CRDT strategy work.
//!
//! [`crdt::CrdtOrMap`](super::crdt::CrdtOrMap) above this layer is just a
//! router: it matches each key to the right engine by registered prefix and
//! delegates. The router does not know LWW vs EpochMaxWins.
//!
//! This split exists so a new strategy (EpochMaxWins today, future ones later)
//! does not require strategy branches inside every entry point of the shared
//! store - the seam that produced most of the bug class in PR #1469.

use std::{sync::Arc, time::Duration};

use super::operation::{CrdtChange, Operation};

mod lww;
mod op_log;
mod rate_limit;

pub(super) use lww::LwwEngine;
pub(super) use rate_limit::RateLimitEngine;

/// The state machine a single namespace runs.
///
/// All methods are byte-oriented at this boundary. Engines are free to use
/// typed internal representations (e.g. `RateLimitState` inside
/// `RateLimitEngine`); the trait deliberately does not expose those types so
/// the dispatch layer stays strategy-agnostic.
pub(super) trait NamespaceCrdtEngine: Send + Sync {
    // ---- Local writes ----

    /// Apply a local put.
    /// - Accepted, displaced a previous value: returns `Some(previous_bytes)`.
    /// - Rejected (e.g. an older `(timestamp, replica_id)` than what is
    ///   already recorded): returns `Some(current_live_bytes)` — the value
    ///   that prevented the write, so the caller can see what is actually
    ///   live without an extra `get`.
    /// - No well-defined previous value (e.g. EpochMaxWins per-point shard
    ///   update where the key remains alive with a smaller shard): returns
    ///   `None`.
    fn put_local(&self, key: &str, value: Vec<u8>) -> Option<Vec<u8>>;

    /// Apply a local delete. Returns the previous live bytes when the delete
    /// removed an existing value, or `None` otherwise.
    fn delete_local(&self, key: &str) -> Option<Vec<u8>>;

    // ---- Reads ----

    fn get(&self, key: &str) -> Option<Vec<u8>>;
    fn contains_key(&self, key: &str) -> bool;
    fn keys(&self) -> Vec<String>;
    fn len(&self) -> usize;

    /// Monotonically increasing mutation counter. Increments on every accepted
    /// local or remote write that changes live state.
    fn generation(&self) -> u64;

    /// Monotonically increasing op-log mutation counter. Unlike
    /// [`Self::generation`], this also covers log-only mutations (a losing
    /// remote op is appended for relay without changing live state), so it
    /// is the correct invalidation key for shared op-log snapshots.
    fn op_generation(&self) -> u64;

    // ---- Replication ----

    /// Snapshot every operation this engine has retained, in deterministic
    /// order. The router concatenates snapshots from all engines to build the
    /// gossip-visible operation log.
    fn export_ops(&self) -> Vec<Operation>;

    /// Merge a batch of incoming operations into this engine. The engine
    /// merges into its log, canonicalises (compaction, tombstone collapse,
    /// same-op-id folding), and applies only the post-canonicalisation result
    /// to live state. This is where the "post-compaction-replay footgun" (PR
    /// #1469) gets sealed inside the engine.
    ///
    /// Takes ownership so the engine can move the batch into its operation
    /// log without an extra clone.
    ///
    /// Returns one [`CrdtChange`] per key whose live value actually changed,
    /// each carrying the canonical post-merge value (matching [`Self::get`]).
    /// The router concatenates these so the gossip receive path can fire
    /// subscribers with the same value shape `get` returns. Keys touched by
    /// dominated/idempotent ops (no observable change) are not reported.
    fn apply_remote_ops(&self, ops: Vec<Operation>) -> Vec<CrdtChange>;

    // ---- Maintenance ----

    /// Garbage-collect tombstones older than `grace`, purging the collected
    /// keys' dominated ops from the operation log so log size keeps tracking
    /// the key population. Returns the number of metadata entries removed.
    fn gc_tombstones(&self, grace: Duration) -> usize;
}

/// Strategy-agnostic engine handle. Routers hold `Arc<dyn
/// NamespaceCrdtEngine>` keyed by registered prefix.
pub(super) type EngineHandle = Arc<dyn NamespaceCrdtEngine>;
