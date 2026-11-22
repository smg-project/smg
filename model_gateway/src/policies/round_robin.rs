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
mod tests {
    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{identity::basic::WorkerType, test_support::no_health_check};

    use super::*;
    use crate::{
        policies::{test_support::make_workers, NamedList},
        routers::grpc::backend_client::BasicWorkerBuilder,
    };

    #[test]
    fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        // Should select workers in order: 0, 1, 2, 0, 1, 2, ...
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
    }

    #[test]
    fn test_round_robin_with_unhealthy_workers() {
        let policy = RoundRobinPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        // Mark middle worker as unhealthy
        workers[1].set_status(WorkerStatus::NotReady);

        // Should skip unhealthy worker: 0, 2, 0, 2, ...
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
    }

    #[test]
    fn test_round_robin_reset() {
        let policy = RoundRobinPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        // Advance the counter
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));

        // Reset should start from beginning
        policy.reset();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    fn assert_even_two_pool_coverage(prefill: [usize; 4], decode: [usize; 4]) {
        assert_eq!(
            prefill,
            [10, 10, 10, 10],
            "even two-pool round-robin coverage: prefill"
        );
        assert_eq!(
            decode,
            [10, 10, 10, 10],
            "even two-pool round-robin coverage: decode"
        );
    }

    fn make_regular_workers(prefix: &str, n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://{prefix}{i}:8000"))
                        .worker_type(WorkerType::Regular)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_one_instance_keeps_each_candidate_set_fair() {
        // One instance serving two interleaved sets: each set gets its own
        // rotation, so the interleaving cannot pin a set to some of its
        // workers (a single counter would give each set only two of four).
        let prefill_workers = make_regular_workers("p", 4);
        let decode_workers = make_regular_workers("d", 4);
        let info = SelectWorkerInfo::default();

        let shared = RoundRobinPolicy::new();
        let mut shared_prefill = [0usize; 4];
        let mut shared_decode = [0usize; 4];
        for _ in 0..40 {
            let p = shared.select_worker(&prefill_workers, &info).unwrap();
            let d = shared.select_worker(&decode_workers, &info).unwrap();
            shared_prefill[p] += 1;
            shared_decode[d] += 1;
        }
        assert_even_two_pool_coverage(shared_prefill, shared_decode);
    }

    #[test]
    fn test_a_set_that_shrinks_and_grows_keeps_rotating() {
        // Health changes the set's identity. The shrunken set is new, so it
        // starts from the shared position (1 by then), not from worker 0
        // again; the full set resumes where it left off when it returns.
        let workers = make_regular_workers("w", 3);
        let info = SelectWorkerInfo::default();
        let policy = RoundRobinPolicy::new();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        workers[1].set_status(WorkerStatus::NotReady);
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        workers[1].set_status(WorkerStatus::Ready);
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
    }

    #[test]
    fn test_past_the_cap_a_new_set_evicts_the_least_recently_used() {
        let info = SelectWorkerInfo::default();
        let policy = RoundRobinPolicy::new();
        let hot = make_regular_workers("hot", 2);
        assert_eq!(policy.select_worker(&hot, &info), Some(0));
        // Distinct one-worker sets fill the map to just under the cap.
        let filler: Vec<Vec<Arc<dyn Worker>>> = (0..MAX_TRACKED_SETS + 3)
            .map(|i| make_regular_workers(&format!("f{i}-"), 1))
            .collect();
        for set in &filler[..MAX_TRACKED_SETS - 2] {
            policy.select_worker(set, &info);
        }
        // The hot set is used again, so it is recent when the cap is hit.
        assert_eq!(policy.select_worker(&hot, &info), Some(1));
        for set in &filler[MAX_TRACKED_SETS - 2..] {
            policy.select_worker(set, &info);
        }
        // Evictions took the oldest fillers, one per miss: the map stays at
        // the cap and the hot set kept its rotation, which continues.
        assert_eq!(policy.counters.len(), MAX_TRACKED_SETS);
        assert!(policy
            .counters
            .contains_key(&RoundRobinPolicy::set_key(&hot, &[0, 1])));
        assert!(!policy
            .counters
            .contains_key(&RoundRobinPolicy::set_key(&filler[0], &[0])));
        assert_eq!(policy.select_worker(&hot, &info), Some(0));
    }

    #[test]
    fn test_independent_counters_pass_even_two_pool_coverage() {
        let prefill_workers = make_regular_workers("p", 4);
        let decode_workers = make_regular_workers("d", 4);
        let info = SelectWorkerInfo::default();

        let prefill_policy = RoundRobinPolicy::new();
        let decode_policy = RoundRobinPolicy::new();
        let mut indep_prefill = [0usize; 4];
        let mut indep_decode = [0usize; 4];
        for _ in 0..40 {
            let p = prefill_policy
                .select_worker(&prefill_workers, &info)
                .unwrap();
            let d = decode_policy.select_worker(&decode_workers, &info).unwrap();
            indep_prefill[p] += 1;
            indep_decode[d] += 1;
        }
        assert_even_two_pool_coverage(indep_prefill, indep_decode);
    }

    #[test]
    fn a_set_is_its_handles_in_order() {
        // The same handles are one set whatever list holds them. Another
        // order, a part of them, or other workers under the same URLs are
        // other sets.
        let workers = make_regular_workers("w", 3);
        let key = RoundRobinPolicy::set_key(&workers, &[0, 1, 2]);
        let again: Vec<Arc<dyn Worker>> = workers.iter().map(Arc::clone).collect();
        assert_eq!(key, RoundRobinPolicy::set_key(&again, &[0, 1, 2]));
        let reversed: Vec<Arc<dyn Worker>> = workers.iter().rev().map(Arc::clone).collect();
        assert_ne!(key, RoundRobinPolicy::set_key(&reversed, &[0, 1, 2]));
        assert_ne!(key, RoundRobinPolicy::set_key(&workers, &[0, 2]));
        let replaced = make_regular_workers("w", 3);
        assert_ne!(key, RoundRobinPolicy::set_key(&replaced, &[0, 1, 2]));
    }

    #[test]
    fn a_set_key_is_one_address_word_a_healthy_worker() {
        // Every selection pays the key: the healthy handles' addresses in
        // order, one word each through the Fx hasher, however long the
        // workers' URLs are. Anything more written, or another hasher, is
        // another key.
        let workers = make_regular_workers("a-worker-under-a-long-host-name-", 8);
        workers[3].set_status(WorkerStatus::NotReady);
        let healthy = get_healthy_worker_indices(&workers, &SelectWorkerInfo::default());
        assert_eq!(healthy, [0, 1, 2, 4, 5, 6, 7]);
        let mut words = FxHasher::default();
        for &i in &healthy {
            words.write_usize(Arc::as_ptr(&workers[i]).addr());
        }
        assert_eq!(
            RoundRobinPolicy::set_key(&workers, &healthy),
            words.finish()
        );
    }

    /// One pick over `list` named whole, as the single-worker placement
    /// hands a routing snapshot's pool.
    fn pick_named(policy: &RoundRobinPolicy, list: &Arc<[Arc<dyn Worker>]>) -> Option<usize> {
        let info = SelectWorkerInfo {
            named: Some(NamedList {
                list,
                positions: None,
            }),
            ..Default::default()
        };
        policy.select_worker(list, &info)
    }

    #[test]
    fn a_list_named_whole_rotates_by_row_under_its_own_address() {
        // While every worker is admitted the walk is the set's own, and the
        // one rotation kept is the list's: no worker was hashed for it.
        let list: Arc<[Arc<dyn Worker>]> = make_regular_workers("w", 3).into();
        let policy = RoundRobinPolicy::new();
        let named: Vec<_> = (0..5).map(|_| pick_named(&policy, &list)).collect();
        assert_eq!(named, [Some(0), Some(1), Some(2), Some(0), Some(1)]);
        assert_eq!(policy.counters.len(), 1);
        assert!(policy
            .counters
            .contains_key(&RoundRobinPolicy::list_key(&list)));
        assert!(!policy
            .counters
            .contains_key(&RoundRobinPolicy::set_key(&list, &[0, 1, 2])));

        let unnamed = RoundRobinPolicy::new();
        let info = SelectWorkerInfo::default();
        let walked: Vec<_> = (0..5)
            .map(|_| unnamed.select_worker(&list, &info))
            .collect();
        assert_eq!(walked, named);
    }

    #[test]
    fn a_refused_row_falls_to_the_admitted_set_and_the_spread_stays_even() {
        let list: Arc<[Arc<dyn Worker>]> = make_regular_workers("w", 4).into();
        list[1].set_status(WorkerStatus::NotReady);
        let policy = RoundRobinPolicy::new();
        // Three turns of the list: nine rows are admitted and taken, three
        // fall to the admitted set's rotation, one to each of its workers.
        let mut picks = [0usize; 4];
        for _ in 0..12 {
            picks[pick_named(&policy, &list).unwrap()] += 1;
        }
        assert_eq!(picks, [4, 0, 4, 4]);
    }

    #[test]
    fn a_row_is_asked_by_the_requests_own_admit_rule() {
        // The row answers what `get_healthy_worker_indices` asks of every
        // worker: a request whose overload veto the caller lifted takes an
        // overloaded row, a request that keeps the veto steps past it.
        let list: Arc<[Arc<dyn Worker>]> = make_regular_workers("w", 3).into();
        list[0].set_overloaded(true);
        let policy = RoundRobinPolicy::new();
        let lifted = SelectWorkerInfo {
            overload_lifted: true,
            named: Some(NamedList {
                list: &list,
                positions: None,
            }),
            ..Default::default()
        };
        let walked: Vec<_> = (0..3)
            .map(|_| policy.select_worker(&list, &lifted))
            .collect();
        assert_eq!(walked, [Some(0), Some(1), Some(2)]);
        assert!(matches!(pick_named(&policy, &list), Some(1 | 2)));
    }

    #[test]
    fn two_narrowings_of_one_list_each_keep_a_set_rotation() {
        // A narrowed list is a set. Rotated under the list's address, two
        // narrowings of one list would share a count, and each would be
        // pinned to one of its two candidates.
        let list: Arc<[Arc<dyn Worker>]> = make_regular_workers("w", 4).into();
        let rows = [[0usize, 2], [1, 3]];
        let candidates: [Vec<Arc<dyn Worker>>; 2] =
            rows.map(|rows| rows.iter().map(|&row| Arc::clone(&list[row])).collect());
        let policy = RoundRobinPolicy::new();
        let mut picks = [[0usize; 2]; 2];
        for _ in 0..4 {
            for (which, positions) in rows.iter().enumerate() {
                let info = SelectWorkerInfo {
                    named: Some(NamedList {
                        list: &list,
                        positions: Some(&positions[..]),
                    }),
                    ..Default::default()
                };
                picks[which][policy.select_worker(&candidates[which], &info).unwrap()] += 1;
            }
        }
        assert_eq!(picks, [[2, 2], [2, 2]]);
    }

    #[test]
    fn a_list_published_again_starts_a_rotation_of_its_own() {
        // The key is the list's address and nothing of its workers: the same
        // handles in another list are another rotation, which starts from
        // the shared position.
        let workers = make_regular_workers("w", 3);
        let first: Arc<[Arc<dyn Worker>]> = workers.clone().into();
        let again: Arc<[Arc<dyn Worker>]> = workers.into();
        let policy = RoundRobinPolicy::new();
        assert_eq!(pick_named(&policy, &first), Some(0));
        assert_eq!(pick_named(&policy, &again), Some(1));
        assert_eq!(pick_named(&policy, &first), Some(1));
        assert_eq!(pick_named(&policy, &again), Some(2));
        assert_eq!(policy.counters.len(), 2);
    }

    #[test]
    fn an_empty_named_list_has_no_pick() {
        let list: Arc<[Arc<dyn Worker>]> = Vec::new().into();
        assert_eq!(pick_named(&RoundRobinPolicy::new(), &list), None);
    }
}
