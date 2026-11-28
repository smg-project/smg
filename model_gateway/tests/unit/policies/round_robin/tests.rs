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
