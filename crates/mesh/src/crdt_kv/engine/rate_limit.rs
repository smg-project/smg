//! Rate-limit engine.
//!
//! Holds typed `RateLimitState` per key, matching the EpochMaxWins CRDT
//! directly: each key is either `Live(shard)` carrying a live-points frontier
//! plus an optional tombstone boundary, or `Tombstone(version)` past which
//! dominated inserts are suppressed.
//!
//! State owned by this engine:
//! - `entries: DashMap<String, ShardEntry>` — typed per-key state. Same-key
//!   writes serialise via DashMap's `entry` API (per-shard lock).
//! - `log: OpLogKeeper` — the gossip-visible operation log with its counter
//! - shared `LamportClock` (per node, cloned from `CrdtOrMap`)
//! - `generation: AtomicU64` — mutation counter for change-detection callers

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry as MapEntry, DashMap};
use tracing::debug;

use super::{
    op_log::{changes, values_before, OpLogKeeper},
    NamespaceCrdtEngine,
};
use crate::crdt_kv::{
    epoch_max_wins::{self as ratelimit, RateLimitState, RateLimitVersion},
    operation::{CrdtChange, Operation},
    replica::{LamportClock, ReplicaId},
};

struct ShardEntry {
    state: RateLimitState,
    /// Local-clock moment the entry's current tombstone version was first
    /// observed. `None` for live entries. Used by `gc_tombstones`; on a
    /// tombstone -> tombstone transition this is refreshed when the version
    /// advances and preserved when an older dominated remove arrives.
    tombstoned_at: Option<Instant>,
}

pub(crate) struct RateLimitEngine {
    entries: Arc<DashMap<String, ShardEntry>>,
    log: OpLogKeeper,
    // Shared per-node Lamport clock — same Arc held by the router and every
    // other engine. See the equivalent note in `engine::lww`.
    clock: Arc<LamportClock>,
    replica_id: ReplicaId,
    generation: AtomicU64,
}

impl RateLimitEngine {
    pub(crate) fn new(replica_id: ReplicaId, clock: Arc<LamportClock>) -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            log: OpLogKeeper::new("RateLimitEngine"),
            clock,
            replica_id,
            generation: AtomicU64::new(0),
        }
    }

    /// A local write's op: appended, and past the keeper's trigger the log
    /// is compacted per EpochMaxWins rules (`epoch_max_wins_fold`) with the
    /// memory backstop.
    fn append_op(&self, op: Operation) {
        self.log
            .append(op, || self.entries.len(), Self::epoch_max_wins_fold);
    }

    /// EpochMaxWins per-key fold: defer to `epoch_max_wins::compact_operations`,
    /// which folds every op for the key through `RateLimitState::merge`,
    /// respecting tombstone boundaries and embedding the merged
    /// `tombstone_version` into the compacted snapshot. Compaction never
    /// drops keys outright - the truncate-oldest safety valve lives only on
    /// the `append_op` local-write path.
    fn epoch_max_wins_fold(ops: &[Operation]) -> Option<Operation> {
        ratelimit::compact_operations(ops.iter())
    }

    fn current_encoded(&self, key: &str) -> Option<Vec<u8>> {
        self.entries
            .get(key)
            .and_then(|entry| entry.state.encode_live())
    }

    /// Merge an insert (value + version) into the entry for `key`. The
    /// outcome captures whether state changed plus the prior-live and
    /// new-live classifications, sampled under the per-key entry lock so
    /// callers can honour the `NamespaceCrdtEngine::put_local` contract
    /// without a racy second lookup.
    ///
    /// Payload decoding (`state_from_insert_value`) happens inside the entry
    /// guard so a malformed put serialises with concurrent valid writes to
    /// the same key. A malformed payload is reported as a no-change outcome,
    /// indistinguishable from a dominated (rejected) insert.
    fn merge_insert(&self, key: &str, value: &[u8], version: RateLimitVersion) -> MergeOutcome {
        match self.entries.entry(key.to_string()) {
            MapEntry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                let prior_live = entry.state.encode_live();
                let now_live = matches!(&entry.state, RateLimitState::Live(_));
                let Some(incoming) = ratelimit::state_from_insert_value(value, version) else {
                    return MergeOutcome {
                        changed: false,
                        prior_live,
                        new_is_live: now_live,
                    };
                };
                // `RateLimitState::merge` returns `None` only when both operands
                // carry no live points and no tombstone. Both `entry.state` and
                // `incoming` always carry content, so this can only happen on a
                // contract violation - treat as no-op rather than panicking.
                let Some(merged) = entry.state.clone().merge(incoming) else {
                    return MergeOutcome {
                        changed: false,
                        prior_live,
                        new_is_live: now_live,
                    };
                };
                let changed = merged != entry.state;
                let new_is_live = matches!(&merged, RateLimitState::Live(_));
                if changed {
                    update_entry(entry, merged);
                }
                MergeOutcome {
                    changed,
                    prior_live,
                    new_is_live,
                }
            }
            MapEntry::Vacant(vacant) => {
                let Some(incoming) = ratelimit::state_from_insert_value(value, version) else {
                    return MergeOutcome {
                        changed: false,
                        prior_live: None,
                        new_is_live: false,
                    };
                };
                let new_is_live = matches!(&incoming, RateLimitState::Live(_));
                let tombstoned_at = (!new_is_live).then(Instant::now);
                vacant.insert(ShardEntry {
                    state: incoming,
                    tombstoned_at,
                });
                MergeOutcome {
                    changed: true,
                    prior_live: None,
                    new_is_live,
                }
            }
        }
    }

    /// Merge a remove (tombstone version) into the entry for `key`. The outcome
    /// is sampled under the per-key entry lock; `prior_live` is the displaced
    /// live shard iff the delete transitioned the entry from `Live` to
    /// `Tombstone`.
    fn merge_remove(&self, key: &str, version: RateLimitVersion) -> MergeOutcome {
        let incoming = RateLimitState::Tombstone(version);
        match self.entries.entry(key.to_string()) {
            MapEntry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                let prior_live = entry.state.encode_live();
                // See `merge_insert`: `None` requires both operands to be empty,
                // which is impossible here.
                let Some(merged) = entry.state.clone().merge(incoming) else {
                    return MergeOutcome {
                        changed: false,
                        prior_live,
                        new_is_live: matches!(&entry.state, RateLimitState::Live(_)),
                    };
                };
                let changed = merged != entry.state;
                let new_is_live = matches!(&merged, RateLimitState::Live(_));
                if changed {
                    update_entry(entry, merged);
                }
                MergeOutcome {
                    changed,
                    prior_live,
                    new_is_live,
                }
            }
            MapEntry::Vacant(vacant) => {
                vacant.insert(ShardEntry {
                    state: incoming,
                    tombstoned_at: Some(Instant::now()),
                });
                MergeOutcome {
                    changed: true,
                    prior_live: None,
                    new_is_live: false,
                }
            }
        }
    }
}

/// Result of merging an op into one entry, observed under the entry lock.
struct MergeOutcome {
    /// `true` iff the post-merge state differs from the prior state.
    changed: bool,
    /// Encoded bytes of the prior live shard, if the prior state was `Live`.
    /// `None` if the prior state was `Tombstone` or the entry was vacant.
    prior_live: Option<Vec<u8>>,
    /// `true` iff the post-merge state is `Live(_)`.
    new_is_live: bool,
}

/// Apply a merged state to `entry`, adjusting `tombstoned_at` per the
/// transition:
/// - live -> tombstone: start the GC clock now.
/// - tombstone -> live: clear the GC clock.
/// - tombstone -> tombstone, version advances: restart the GC clock so the
///   newer winning remove gets its full grace period.
/// - tombstone -> tombstone, same version: preserve the existing clock.
///   An older dominated remove that arrives late must not extend grace on a
///   tombstone that would otherwise be due for collection.
fn update_entry(entry: &mut ShardEntry, merged: RateLimitState) {
    let was_tombstone_version = tombstone_version_of(&entry.state);
    let now_tombstone_version = tombstone_version_of(&merged);
    entry.state = merged;
    match (was_tombstone_version, now_tombstone_version) {
        (None, Some(_)) => entry.tombstoned_at = Some(Instant::now()),
        (Some(_), None) => entry.tombstoned_at = None,
        (Some(was), Some(now)) if was != now => {
            entry.tombstoned_at = Some(Instant::now());
        }
        // (None, None): still live. (Some(v), Some(v)): idempotent or
        // dominated remove. Either way leave the clock alone.
        _ => {}
    }
}

fn tombstone_version_of(state: &RateLimitState) -> Option<RateLimitVersion> {
    match state {
        RateLimitState::Tombstone(version) => Some(*version),
        RateLimitState::Live(_) => None,
    }
}

impl NamespaceCrdtEngine for RateLimitEngine {
    fn put_local(&self, key: &str, value: Vec<u8>) -> Option<Vec<u8>> {
        let timestamp = self.clock.tick();
        let version = RateLimitVersion::new(timestamp, self.replica_id);

        let outcome = self.merge_insert(key, &value, version);

        if outcome.changed {
            let op = Operation::insert(key.to_string(), value, timestamp, self.replica_id);
            self.append_op(op);
            self.generation.fetch_add(1, Ordering::Release);
            debug!(
                "RateLimitEngine insert: key={}, timestamp={}, replica={}",
                key, timestamp, self.replica_id
            );
        }

        match (outcome.changed, outcome.new_is_live) {
            // Rejected (dominated / idempotent / malformed payload): return
            // current live bytes (sampled under the entry lock above), which
            // is `prior_live` since state did not change.
            (false, _) => outcome.prior_live,
            // Accepted, incoming carried a tombstone bound that killed the
            // prior live shard. The displaced previous is well-defined.
            (true, false) => outcome.prior_live,
            // Accepted, key remains live. Per-point frontier update or
            // vacant -> live insert: no well-defined previous value.
            (true, true) => None,
        }
    }

    fn delete_local(&self, key: &str) -> Option<Vec<u8>> {
        let timestamp = self.clock.tick();
        let version = RateLimitVersion::new(timestamp, self.replica_id);
        debug!(
            "RateLimitEngine remove: key={}, timestamp={}, replica={}",
            key, timestamp, self.replica_id
        );
        let outcome = self.merge_remove(key, version);
        if outcome.changed {
            let op = Operation::remove(key.to_string(), timestamp, self.replica_id);
            self.append_op(op);
            self.generation.fetch_add(1, Ordering::Release);
        }
        // The trait returns prior live bytes only when the delete actually
        // removed an existing live value. For EpochMaxWins that means the
        // entry transitioned from `Live` to `Tombstone`; a delete that
        // leaves live points behind (lower-version tombstone) or arrives at
        // an already-tombstoned key returns `None`.
        if outcome.changed && !outcome.new_is_live {
            outcome.prior_live
        } else {
            None
        }
    }

    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.current_encoded(key)
    }

    fn contains_key(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .is_some_and(|entry| matches!(&entry.state, RateLimitState::Live(_)))
    }

    fn keys(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|entry| matches!(&entry.state, RateLimitState::Live(_)))
            .map(|entry| entry.key().clone())
            .collect()
    }

    fn len(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| matches!(&entry.state, RateLimitState::Live(_)))
            .count()
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn op_generation(&self) -> u64 {
        self.log.generation()
    }

    fn export_ops(&self) -> Vec<Operation> {
        self.log.export()
    }

    fn apply_remote_ops(&self, mut ops: Vec<Operation>) -> Vec<CrdtChange> {
        if ops.is_empty() {
            return Vec::new();
        }

        // EpochMaxWins always replays incoming ops to state because a
        // compacted snapshot can carry an embedded tombstone_version at the
        // same op-id as a previously-seen raw payload. `merge_insert` /
        // `merge_remove` return `changed=false` for byte-identical re-applies
        // so generation only bumps when state truly changes.
        ops.sort_by_key(|op| (op.timestamp(), op.replica_id()));

        // Merge incoming ops into the log by appending them all and letting
        // compaction fold per key. `compact_by_key` groups by key and folds
        // each group via `compact_operations`; because ops sharing an op-id
        // share a key (op-id is unique per logical op), a compacted snapshot
        // and a previously-seen raw payload at the same op-id land in the same
        // group and fold together - preserving the embedded `tombstone_version`
        // (the bug class addressed in #1469) with no separate op-id index.
        // `RateLimitState::merge` is associative, so folding the whole group at
        // once matches the prior incremental per-op-id fold; identical ops fold
        // idempotently, so no duplicate survives. The merge path does not
        // truncate, so remotely-learned keys are never dropped.
        self.log.merge_batch(&ops, Self::epoch_max_wins_fold);

        // Snapshot the observable (encoded-live) value of every key this batch
        // touches before applying, so we emit a `CrdtChange` only when `get`
        // actually changes. Tombstone-version bumps and frontier reshuffles
        // that leave `encode_live` unchanged (e.g. Tombstone -> newer
        // Tombstone, both encoding to `None`) bump generation/state but fire no
        // subscriber event.
        let before = values_before(&ops, |key| self.current_encoded(key));

        for op in ops {
            self.clock.update(op.timestamp());
            let changed = match op {
                Operation::Insert {
                    key,
                    value,
                    timestamp,
                    replica_id,
                } => {
                    let version = RateLimitVersion::new(timestamp, replica_id);
                    self.merge_insert(&key, &value, version).changed
                }
                Operation::Remove {
                    key,
                    timestamp,
                    replica_id,
                } => {
                    let version = RateLimitVersion::new(timestamp, replica_id);
                    self.merge_remove(&key, version).changed
                }
            };
            if changed {
                self.generation.fetch_add(1, Ordering::Release);
            }
        }

        // One CrdtChange per key whose observable value changed, carrying
        // the canonical post-merge value (the encoded live shard, matching
        // `get`).
        changes(before, |key| self.current_encoded(key))
    }

    fn gc_tombstones(&self, grace: Duration) -> usize {
        let now = Instant::now();
        let candidates: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| {
                matches!(&entry.state, RateLimitState::Tombstone(_))
                    && entry
                        .tombstoned_at
                        .is_some_and(|at| now.saturating_duration_since(at) >= grace)
            })
            .map(|entry| entry.key().clone())
            .collect();

        let mut removed = 0;
        // Collected keys' winning tombstone versions, for the log purge below.
        let mut purged: std::collections::HashMap<String, RateLimitVersion> =
            std::collections::HashMap::new();
        for key in candidates {
            let was_removed = self.entries.remove_if(&key, |_, entry| {
                matches!(&entry.state, RateLimitState::Tombstone(_))
                    && entry
                        .tombstoned_at
                        .is_some_and(|at| now.saturating_duration_since(at) >= grace)
            });
            if let Some((key, entry)) = was_removed {
                removed += 1;
                if let RateLimitState::Tombstone(version) = entry.state {
                    purged.insert(key, version);
                }
            }
        }
        if !purged.is_empty() {
            // Purge the collected keys' dominated ops so log size keeps
            // tracking the entry count (the compaction trigger) and dead
            // tombstones stop gossiping. Newer concurrent ops for a reused
            // key survive the version filter.
            self.log.retain(|op| {
                purged.get(op.key()).is_none_or(|tombstone| {
                    RateLimitVersion::new(op.timestamp(), op.replica_id()) > *tombstone
                })
            });
        }
        removed
    }
}
