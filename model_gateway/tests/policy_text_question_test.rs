//! Policy paths a request walks without allocating, pinned: the registry's
//! question whether some policy wants the request's text (the answer is kept
//! where a policy is added or removed); the least-load decision, which
//! allocates its eligible list and nothing else once it has dispatched to a
//! worker; a request placed by a ring policy, which pays the ring's own
//! lookup and nothing besides (prefix hash also its eligible list and its
//! boundary list); a power-of-two, random or passthrough pick, which
//! allocates nothing at all over a pool the draw covers; a keyed hit on a
//! sticky pin, measured with no handle kept, which pays its namespaced key
//! under the routing-key override and its eligible list in the manual
//! policy, with the records of their metrics, and nothing besides; and the
//! branch count of the manual, consistent-hashing and prefix-hash
//! decisions, on a kept handle. A separate executable, so it can declare
//! the counting allocator the pins need.

#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
#![expect(unsafe_code, reason = "a global allocator that counts allocations")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    sync::Arc,
};

use http::HeaderMap;
use openai_protocol::worker::HealthCheckConfig;
use smg::{
    config::types::{ManualAssignmentMode, PolicyConfig, RoutingKeyOverrideConfig},
    observability::metrics::Metrics,
    policies::{
        ConsistentHashingPolicy, LeastLoadPolicy, LoadBalancingPolicy, ManualPolicy,
        PassthroughPolicy, PolicyRegistry, PowerOfTwoPolicy, PrefixHashConfig, PrefixHashPolicy,
        RandomPolicy, SelectWorkerInfo,
    },
    worker::{
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, HashRing, Worker, WorkerType,
    },
};
use smg_external_router::metrics::handles::force_on_this_thread;

struct Counting;

thread_local! {
    /// The allocation calls of this thread while one of its pins measures,
    /// `None` outside a measurement. A thread counts its own calls alone, so
    /// neither the test harness nor a sibling test on another thread reaches
    /// a pin's count. Const-initialised with no destructor, so reading it
    /// inside the allocator allocates nothing and never runs a destructor.
    static COUNTED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Counts one allocation call on this thread while it measures; a thread
/// whose local is gone (tearing down) counts nothing.
fn count() {
    let _ = COUNTED.try_with(|counted| {
        if let Some(calls) = counted.get() {
            counted.set(Some(calls + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocation calls `work` makes on this thread: nothing another thread
/// does in the meantime is counted.
fn allocations_during(work: impl FnOnce()) -> usize {
    COUNTED.with(|counted| counted.set(Some(0)));
    work();
    COUNTED.with(|counted| counted.replace(None)).unwrap_or(0)
}

/// A cache-aware default (text wanted, no eviction thread) and 64 models on
/// hinted cache-aware policies of their own: a request's question allocates
/// nothing, with and without a routing key and token ids, however many
/// models are registered.
#[test]
fn a_request_asks_whether_text_is_wanted_without_allocating() {
    let registry = PolicyRegistry::with_override(
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
        RoutingKeyOverrideConfig {
            enabled: true,
            ..Default::default()
        },
    );
    for model in 0..64 {
        registry.on_worker_added(&format!("model-{model}"), Some("cache_aware"));
    }
    let plain = HeaderMap::new();
    let mut keyed = HeaderMap::new();
    keyed.insert(
        "x-smg-routing-key",
        "session-A".parse().expect("a header value"),
    );

    assert!(registry.any_policy_needs_request_text(Some(&plain)));
    assert!(!registry.any_policy_needs_request_text(Some(&keyed)));
    assert!(!registry.any_policy_needs_request_text_with_tokens(Some(&plain)));

    let allocations = allocations_during(|| {
        for _ in 0..1_000 {
            black_box(registry.any_policy_needs_request_text(Some(&plain)));
            black_box(registry.any_policy_needs_request_text(Some(&keyed)));
            black_box(registry.any_policy_needs_request_text_with_tokens(Some(&plain)));
            black_box(registry.any_policy_needs_request_text_with_tokens(None));
        }
    });
    assert_eq!(
        allocations, 0,
        "the text questions allocated {allocations} times over 4,000 requests"
    );
}

/// Eight ready regular workers (no health checker behind them), each with a
/// published report and requests in flight, under a least-load policy that
/// has dispatched to every one of them: a decision allocates its eligible
/// list and nothing else. The winner's credit finds its tally by a borrowed
/// key; only a worker's first dispatch allocates the key. A report is
/// published before every decision, so the credit of the one before is
/// released and the tally holds at most one dispatch per worker.
#[test]
fn a_least_load_decision_allocates_its_eligible_list_and_nothing_else() {
    let policy = LeastLoadPolicy::new();
    let feed = TestFeed::attached_to(&policy);
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{i}:8000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let reports = workers
        .iter()
        .enumerate()
        .map(|(i, worker)| {
            for _ in 0..i % 3 {
                worker.increment_load();
            }
            (
                worker.url().to_string(),
                one_rank_load(512 * i as i32, 0.1 * i as f64, 2_000.0 + 100.0 * i as f64),
            )
        })
        .collect();
    feed.publish(&reports);
    let tokens: Vec<u32> = (0..1_024).collect();
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        ..Default::default()
    };
    // Every worker has been dispatched to: the tally knows each key.
    for _ in 0..64 {
        black_box(policy.select_worker(&workers, &info));
    }
    let decisions = 1_000;
    let mut allocations = 0;
    for _ in 0..decisions {
        feed.publish(&reports);
        allocations += allocations_during(|| {
            black_box(policy.select_worker(&workers, &info));
        });
    }
    // The eligible list is collected from a filter without a size hint: four
    // indices, then eight, two blocks for this pool.
    assert_eq!(
        allocations,
        2 * decisions,
        "least-load decisions allocated {allocations} times over {decisions} decisions; the \
         eligible list (two blocks for eight workers) is the budget"
    );
}

/// Eight ready regular workers (no health checker behind them) on a ring: a
/// consistent-hashing request keyed by a session pays the ring's own lookup
/// (the ring marks the workers it asked on the stack: no allocation) and
/// nothing besides; a prefix-hash request of 1,024 tokens over the boundaries 64 and
/// 256 pays one ring lookup (the deeper level's owner is under its load
/// margin), its eligible list (a filter collected without a size hint: four
/// entries, then eight) and its boundary list, and nothing besides. The
/// translation of the ring's answer back to a position allocates nothing,
/// prefix hash spells its ring key without a string, and the record of the
/// branch metric is on a kept handle (as the gateway's threads keep them
/// once the recorder is installed; the thread is told to here).
#[test]
fn a_ring_policy_request_pays_the_ring_lookup_and_nothing_besides() {
    force_on_this_thread(Some(true));
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{i}:8000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let ring = Arc::new(HashRing::new(workers.iter().map(|worker| worker.url())));
    // The ring's own cost per lookup, read on the ring alone: a keyed lookup
    // marks the workers it asked on the stack and allocates nothing.
    let ring_lookup = allocations_during(|| {
        black_box(ring.find_healthy_url("session-0001", |_| true));
    });
    eprintln!("allocation calls per keyed ring lookup: {ring_lookup}");
    assert_eq!(ring_lookup, 0, "allocation calls per keyed ring lookup");

    let consistent = ConsistentHashingPolicy::new();
    let keyed = SelectWorkerInfo {
        routing_key: Some("session-0001"),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let prefix = PrefixHashPolicy::new(PrefixHashConfig {
        cache_boundaries: vec![64, 256],
        ..Default::default()
    });
    let tokens: Vec<u32> = (0..1_024).collect();
    let prompt = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    // A first request of each: whatever is set up once is set up now.
    assert!(consistent.select_worker(&workers, &keyed).is_some());
    assert!(prefix.select_worker(&workers, &prompt).is_some());

    let requests = 1_000;
    let mut consistent_allocations = 0;
    let mut prefix_allocations = 0;
    for _ in 0..requests {
        consistent_allocations += allocations_during(|| {
            black_box(consistent.select_worker(&workers, &keyed));
        });
        prefix_allocations += allocations_during(|| {
            black_box(prefix.select_worker(&workers, &prompt));
        });
    }
    // The branch metric's record is on a kept handle: no block of its own.
    let metric_record = 0;
    assert_eq!(
        consistent_allocations,
        (ring_lookup + metric_record) * requests,
        "consistent-hashing requests allocated {consistent_allocations} times over {requests}; \
         the ring's lookup ({ring_lookup}) is the budget: the branch metric's record is on a kept \
         handle"
    );
    assert_eq!(
        prefix_allocations,
        (ring_lookup + 3 + metric_record) * requests,
        "prefix-hash requests allocated {prefix_allocations} times over {requests}; one ring \
         lookup ({ring_lookup}), the eligible list's two blocks and the boundary list are the \
         budget: the branch metric's record is on a kept handle"
    );
    force_on_this_thread(None);
}

/// A branch recorder under the pin: the policy's name, the recorder, a
/// branch label of the policy's own.
type BranchRecord = (&'static str, fn(&'static str), &'static str);

/// A policy's branch is counted on a kept handle: after a label's first
/// sight on a thread, the record allocates nothing. The manual,
/// consistent-hashing and prefix-hash recorders each built their counter's
/// key per call, one block per request of every policy that counts its
/// branch.
#[test]
fn a_policys_branch_is_counted_without_allocating() {
    // Handles are kept once the process's recorder is installed; this binary
    // installs none, so the thread is told to keep them, as the gateway's do.
    force_on_this_thread(Some(true));
    let records: [BranchRecord; 3] = [
        (
            "manual",
            Metrics::record_worker_manual_policy_branch,
            "occupied_hit",
        ),
        (
            "consistent_hashing",
            Metrics::record_worker_consistent_hashing_policy_branch,
            "routing_key_hit",
        ),
        (
            "prefix_hash",
            Metrics::record_worker_prefix_hash_policy_branch,
            "ring_hit",
        ),
    ];
    let allocations: Vec<(&str, usize)> = records
        .iter()
        .map(|&(policy, record, branch)| {
            // The label's first record builds the handle and keeps it.
            record(branch);
            let allocations = allocations_during(|| {
                for _ in 0..1_000 {
                    record(black_box(branch));
                }
            });
            (policy, allocations)
        })
        .collect();
    force_on_this_thread(None);
    assert!(
        allocations.iter().all(|&(_, count)| count == 0),
        "1,000 branch records allocate: {allocations:?}"
    );
}

/// Eight ready regular workers with reports and in-flight requests, every
/// one eligible, under a policy that has picked from the pool often enough to
/// know every worker: a pick allocates nothing. Power-of-two draws its pair
/// over the pool and credits the winner by a borrowed key; random draws its
/// pick; passthrough takes the first eligible position.
#[test]
fn a_power_of_two_random_or_passthrough_pick_allocates_nothing() {
    let policies: [(&str, Box<dyn LoadBalancingPolicy>); 3] = [
        ("power_of_two", Box::new(PowerOfTwoPolicy::new())),
        ("random", Box::new(RandomPolicy::new())),
        ("passthrough", Box::new(PassthroughPolicy::new())),
    ];
    let tokens: Vec<u32> = (0..1_024).collect();
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        ..Default::default()
    };
    let mut over_budget = Vec::new();
    for (name, policy) in &policies {
        let feed = TestFeed::attached_to(policy.as_ref());
        let workers: Vec<Arc<dyn Worker>> = (0..8)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://worker-{i}:8000"))
                        .worker_type(WorkerType::Regular)
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        let reports = workers
            .iter()
            .enumerate()
            .map(|(i, worker)| {
                for _ in 0..i % 3 {
                    worker.increment_load();
                }
                (
                    worker.url().to_string(),
                    one_rank_load(512 * i as i32, 0.1 * i as f64, 2_000.0 + 100.0 * i as f64),
                )
            })
            .collect();
        feed.publish(&reports);
        for _ in 0..256 {
            black_box(policy.select_worker(&workers, &info));
        }
        let picks = 1_000;
        let mut allocations = 0;
        for _ in 0..picks {
            // A fresh report releases the last pick's credit, so the
            // power-of-two tally holds at most one dispatch per worker.
            feed.publish(&reports);
            allocations += allocations_during(|| {
                black_box(policy.select_worker(&workers, &info));
            });
        }
        if allocations != 0 {
            over_budget.push(format!("{name}: {allocations} over {picks} picks"));
        }
    }
    assert!(
        over_budget.is_empty(),
        "picks allocated {over_budget:?}; nothing is the budget"
    );
}

/// What a keyed hit still pays once its key is read borrowed and the
/// eligible list is built only to assign, per hit: under the routing-key
/// override the namespaced key it formats (model, role and key in one
/// string, the map's key) and the records of its metrics (blocks in the
/// metrics layer, outside this path); in the manual policy its eligible list
/// (a filter collected without a size hint: four indices, then eight) and
/// the record of its branch metric. This pin keeps no handle on its thread,
/// so a record builds its counter's key and counts here; on a kept handle,
/// as the gateway's threads have it, the branch record is free (the branch
/// pin above measures that). Recorded on the unchanged code as the hit's
/// count less the key copy and the list, then held after the change.
const OVERRIDE_HIT_BUDGET: usize = 5;

const MANUAL_HIT_BUDGET: usize = 3;

/// Eight ready regular workers (no health checker behind them) and sixteen
/// conversations pinned once. A keyed hit under the routing-key override
/// (delegating its assignments to a round-robin registry) pays its budget
/// and nothing besides: the pin is read through the borrowed key, and no
/// eligible list is built for a hit. A keyed hit in the manual policy pays
/// its budget and nothing besides: its key is read borrowed too.
#[test]
fn a_keyed_hit_pays_its_key_under_the_override_and_its_eligible_list_in_manual() {
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{i}:8000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let model_id = workers[0].model_id().to_string();
    let registry = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        RoutingKeyOverrideConfig {
            enabled: true,
            assignment_mode: ManualAssignmentMode::Delegate,
            ..Default::default()
        },
    );
    let policy = registry.get_default_policy();
    let manual = ManualPolicy::new();
    let keys: Vec<String> = (0..16).map(|i| format!("conversation-{i:04}")).collect();
    let infos: Vec<SelectWorkerInfo> = keys
        .iter()
        .map(|key| SelectWorkerInfo {
            rid_key: Some(key),
            ..Default::default()
        })
        .collect();
    for info in &infos {
        assert!(registry
            .select_worker_for_model(&policy, &model_id, &workers, info)
            .is_some());
        assert!(manual.select_worker(&workers, info).is_some());
    }

    let hits = 1_000;
    let mut override_allocations = 0;
    let mut manual_allocations = 0;
    for hit in 0..hits {
        let info = &infos[hit % infos.len()];
        override_allocations += allocations_during(|| {
            black_box(registry.select_worker_for_model(&policy, &model_id, &workers, info));
        });
        manual_allocations += allocations_during(|| {
            black_box(manual.select_worker(&workers, info));
        });
    }
    assert!(
        override_allocations == OVERRIDE_HIT_BUDGET * hits
            && manual_allocations == MANUAL_HIT_BUDGET * hits,
        "keyed hits allocated {override_allocations} times under the override and \
         {manual_allocations} times in the manual policy over {hits} hits; the budgets are \
         {OVERRIDE_HIT_BUDGET} and {MANUAL_HIT_BUDGET} per hit"
    );
}
