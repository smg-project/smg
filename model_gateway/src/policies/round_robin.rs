//! Round-robin load balancing policy

use std::{
    hash::Hasher,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use dashmap::DashMap;
use rustc_hash::FxHasher;
use smg_worker::identity::basic::Worker;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};

/// Distinct candidate sets one instance keeps a rotation for; past this a
/// new set evicts the least recently used one. Sets come and go with worker
/// health, so this is a bound on memory, not a limit on fleets.
const MAX_TRACKED_SETS: usize = 4096;

/// One candidate set's rotation.
#[derive(Debug)]
struct Rotation {
    next: AtomicUsize,
    /// The policy's tick when the set was last selected from, for eviction.
    last_used: AtomicU64,
}

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
/// The rotation is kept per candidate set: one instance serves several sets
/// (the decode partners of each PD cohort, say), and each set walks its own
/// workers in turn. A single counter would advance on every call whatever
/// the set, so its position modulo one set's size would follow how the sets
/// happen to interleave, pinning each set to a subset of its workers.
///
/// A set is identified by its healthy workers in the order the caller lists
/// them, each by its handle and never by its URL. Callers hand over a stable
/// order and the same handles for the same workers (a pool snapshot, a pair
/// index's partners), which the rotation relies on either way: a worker
/// under a new handle is a new member, and the sets that hold it start
/// again. A set seen for the first time starts from a shared, advancing
/// position rather than its first worker, so a fleet whose shape keeps
/// changing does not lean on index 0.
///
/// A list the caller names whole (a routing snapshot's pool) is rotated by
/// row instead: its rotation is kept under the list's own address, and the
/// row the count lands on is the pick when the request may use that worker.
/// Nothing is asked of the other workers and nobody is listed, so the pick
/// costs the same at any pool size. While every worker is admitted that is
/// the set's own walk. A row the request may not use falls to the set's
/// rotation over the admitted workers: each of them keeps its own turns and
/// takes an even share of the refused rows', so the spread stays even while
/// the order is no longer one strict rotation. A narrowed list (the decode
/// partners of a cohort) is a set as before.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counters: DashMap<u64, Rotation>,
    next_start: AtomicUsize,
    tick: AtomicU64,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counters: DashMap::new(),
            next_start: AtomicUsize::new(0),
            tick: AtomicU64::new(0),
        }
    }

    /// The identity of a candidate set: its healthy workers in order, each
    /// by its handle's address, one word hashed a candidate. Nothing of a
    /// worker is read, so a selection's cost does not grow with its
    /// workers' URLs. A worker replaced under its URL is a new handle: its
    /// sets start again from the shared position, which is harmless, as is
    /// a freed address that a later worker takes over.
    fn set_key(workers: &[Arc<dyn Worker>], healthy: &[usize]) -> u64 {
        let mut hasher = FxHasher::default();
        for &i in healthy {
            hasher.write_usize(Arc::as_ptr(&workers[i]).addr());
        }
        hasher.finish()
    }

    /// The identity of a list the caller names whole: the list's own
    /// address, one word. A snapshot never changes its list, so the address
    /// stands for the members in their order until the registry publishes
    /// another list, whose rotation starts from the shared position while
    /// the old one ages out as a set's does. A list's key meets a set's only
    /// through an address reused after a free, which is as harmless here as
    /// it is there.
    fn list_key(list: &Arc<[Arc<dyn Worker>]>) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_usize(Arc::as_ptr(list).addr());
        hasher.finish()
    }

    /// This set's next position, creating its rotation on first sight.
    ///
    /// Age is measured in misses, since only misses evict: the steady state
    /// is a read guard, a relaxed load of the miss count and two per-set
    /// atomics, with no shared write. A miss is a new shape: at the cap it
    /// evicts the least recently used set first, a scan of the tracked sets
    /// that only a churning fleet at the cap pays. `last_used` only moves
    /// forward, so selections completing out of order cannot age a set.
    fn advance(&self, key: u64) -> usize {
        if let Some(rotation) = self.counters.get(&key) {
            rotation
                .last_used
                .fetch_max(self.tick.load(Ordering::Relaxed), Ordering::Relaxed);
            return rotation.next.fetch_add(1, Ordering::Relaxed);
        }
        let now = self.tick.fetch_add(1, Ordering::Relaxed) + 1;
        if self.counters.len() >= MAX_TRACKED_SETS {
            // The iterator's shard guards are dropped with the statement,
            // before the removal takes its shard's write lock.
            let stale = self
                .counters
                .iter()
                .min_by_key(|entry| entry.last_used.load(Ordering::Relaxed))
                .map(|entry| *entry.key());
            if let Some(stale) = stale {
                self.counters.remove(&stale);
            }
        }
        // The shared position advances only for the request that inserts,
        // so concurrent first sightings of one set do not skip starts; a
        // set another request inserted meanwhile is refreshed as used now.
        let rotation = self.counters.entry(key).or_insert_with(|| Rotation {
            next: AtomicUsize::new(self.next_start.fetch_add(1, Ordering::Relaxed)),
            last_used: AtomicU64::new(now),
        });
        rotation.last_used.fetch_max(now, Ordering::Relaxed);
        rotation.next.fetch_add(1, Ordering::Relaxed)
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        // A list named whole is rotated by row: the row's worker is the only
        // one asked anything. A refused row falls to the set's rotation.
        if let Some(named) = info.named.filter(|named| named.positions.is_none()) {
            debug_assert!(
                std::ptr::eq(&named.list[..], workers),
                "a list named whole is the slice handed"
            );
            if !workers.is_empty() {
                let row = self.advance(Self::list_key(named.list)) % workers.len();
                if info.admits(workers[row].routing_state()) {
                    return Some(row);
                }
            }
        }

        let healthy_indices = get_healthy_worker_indices(workers, info);

        if healthy_indices.is_empty() {
            return None;
        }

        let count = self.advance(Self::set_key(workers, &healthy_indices));
        let selected_idx = count % healthy_indices.len();

        Some(healthy_indices[selected_idx])
    }

    fn hears_request_complete(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.counters.clear();
        self.next_start.store(0, Ordering::Relaxed);
        self.tick.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/round_robin/tests.rs"]
mod tests;
