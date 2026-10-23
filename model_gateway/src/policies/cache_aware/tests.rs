use kv_index::{compute_content_hash, SequenceHash, StoredBlock};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use openai_protocol::worker::{
    HealthCheckConfig, SchedulerLoadSnapshot, WorkerLoadResponse, WorkerStatus,
};
use tokio::sync::watch;
use tracing_test::traced_test;

use super::*;
use crate::worker::load_state::TestFeed;

#[test]
fn text_is_wanted_only_without_token_ids() {
    let policy = CacheAwarePolicy::new();
    assert!(policy.needs_request_text());
    assert!(!policy.needs_request_text_with_tokens());
}

/// Neutral tuning: decay and temperature off, no load snapshot — the
/// historical selection behavior.
fn default_tuning() -> OverlapTuning<'static> {
    OverlapTuning {
        overlap_decay: 0.0,
        selection_temperature: 0.0,
        waiting_prefill_tokens: None,
    }
}

/// Exercise the same affinity-candidate and score-group stages used by
/// production immediately before LeastLoad resolves the final worker.
fn overlap_affinity_group(
    workers: &[Arc<dyn Worker>],
    tokens: &[u32],
    healthy_indices: &[usize],
    indexer: &KvIndex,
    block_size: usize,
    tuning: &OverlapTuning<'_>,
) -> Vec<usize> {
    let candidates = CacheAwarePolicy::overlap_candidates(
        workers,
        tokens,
        healthy_indices,
        indexer,
        block_size,
        tuning,
    );
    CacheAwarePolicy::affinity_score_group(&candidates, tuning.selection_temperature)
}
use crate::{
    observability::metrics::CACHE_AWARE_MATCH_RATIO_BUCKETS,
    worker::{runtime::StallReason, BasicWorkerBuilder, WorkerBlocks, WorkerType},
};

fn no_health_check() -> HealthCheckConfig {
    HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    }
}

#[test]
fn test_cache_aware_with_balanced_load() {
    // Create policy without eviction thread for testing
    let config = CacheAwareConfig {
        eviction_interval_secs: 0, // Disable eviction thread
        ..Default::default()
    };
    let policy = CacheAwarePolicy::with_config(config);
    let workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .health_config(no_health_check())
                .build(),
        ),
        Arc::new(
            BasicWorkerBuilder::new("http://w2:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .health_config(no_health_check())
                .build(),
        ),
    ];

    // Initialize the policy with workers
    policy.init_workers(&workers);

    // First request should be distributed
    let idx1 = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("hello world"),
                ..Default::default()
            },
        )
        .unwrap();

    // Same request should go to same worker (cache hit)
    let idx2 = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("hello world"),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx1, idx2);

    // Similar request should also go to same worker
    let idx3 = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("hello"),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx1, idx3);
}

/// `RoutingState::overloaded` has exactly one production reader — the fused
/// gather below `select_worker`. Without this test nothing would fail if
/// `state.eligible()` were reverted to health + circuit breaker, or if
/// `BasicWorker::routing_state` stopped populating the field.
#[test]
fn overloaded_worker_is_vetoed_by_the_fused_gather() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    let info = SelectWorkerInfo {
        request_text: Some("a stable prefix that pins one worker"),
        ..Default::default()
    };
    let owner = policy.select_worker(&workers, &info).unwrap();
    assert_eq!(
        policy.select_worker(&workers, &info),
        Some(owner),
        "cache affinity holds while the worker is eligible"
    );

    workers[owner].set_overloaded(true);
    assert_ne!(
        policy.select_worker(&workers, &info),
        Some(owner),
        "affinity must not outrank the absolute veto"
    );

    // Every worker vetoed leaves nothing to select.
    for worker in &workers {
        worker.set_overloaded(true);
    }
    assert_eq!(policy.select_worker(&workers, &info), None);

    for worker in &workers {
        worker.set_overloaded(false);
    }
    assert!(policy.select_worker(&workers, &info).is_some());
}

/// A caller that judged the overload veto unable to steer lifts it (every
/// worker of a disaggregated pool over the thresholds): the policy then
/// places as if none were over them, prefix affinity included. The liveness
/// veto is not lifted.
#[test]
fn a_lifted_overload_veto_keeps_the_prefix_owner() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    let info = SelectWorkerInfo {
        request_text: Some("a stable prefix that pins one worker"),
        ..Default::default()
    };
    let owner = policy.select_worker(&workers, &info).unwrap();
    for worker in &workers {
        worker.set_overloaded(true);
    }
    assert_eq!(
        policy.select_worker(&workers, &info),
        None,
        "the veto applies"
    );
    let lifted = SelectWorkerInfo {
        overload_lifted: true,
        ..info.clone()
    };
    for _ in 0..4 {
        assert_eq!(
            policy.select_worker(&workers, &lifted),
            Some(owner),
            "with the veto lifted the owner keeps its prefix"
        );
    }
    workers[owner].set_stall(Some(StallReason::Wedged));
    assert_ne!(
        policy.select_worker(&workers, &lifted),
        Some(owner),
        "a stalled owner is not lifted back"
    );
}

#[test]
fn test_cache_aware_with_imbalanced_load() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 5,
        balance_rel_threshold: 2.0,
        eviction_interval_secs: 0, // Disable eviction thread
        max_tree_size: 10000,
        block_size: 16,
        balance_token_usage_threshold: 1.0,
        overload_token_usage_threshold: 1.0,
        overlap_decay: 0.0,
        selection_temperature: 0.0,
        ..Default::default()
    });

    let worker1 = BasicWorkerBuilder::new("http://w1:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    let worker2 = BasicWorkerBuilder::new("http://w2:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    // Create significant load imbalance
    for _ in 0..20 {
        worker1.increment_load();
    }
    // worker2 has load 0

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
    policy.init_workers(&workers);

    // Should select worker2 (lower load) despite cache affinity
    let info = SelectWorkerInfo {
        request_text: Some("test"),
        ..Default::default()
    };
    for _ in 0..5 {
        let idx = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(idx, 1); // Should always pick worker2
    }
}

// ---- tree size aggregation (per-model gauges + eviction-cycle log) ----

#[test]
fn string_tree_totals_sums_tenant_chars() {
    let tree = Tree::new();
    assert_eq!(string_tree_totals(&tree), (0, 0));

    // Worker registration (empty insert) registers a zero-char tenant.
    tree.insert_text("", "http://w1:8000");
    assert_eq!(string_tree_totals(&tree), (0, 1));

    tree.insert_text("hello", "http://w1:8000");
    assert_eq!(string_tree_totals(&tree), (5, 1));

    // A shared path counts its chars for every tenant on it.
    tree.insert_text("hello", "http://w2:8000");
    assert_eq!(string_tree_totals(&tree), (10, 2));

    // "help!" shares "hel", adds only "p!" for w1.
    tree.insert_text("help!", "http://w1:8000");
    assert_eq!(string_tree_totals(&tree), (12, 2));
}

#[test]
fn token_tree_totals_sums_tenant_tokens() {
    let tree = TokenTree::new();
    assert_eq!(token_tree_totals(&tree), (0, 0));

    // Worker registration (empty insert) aligns to zero pages —
    // no tenant entry.
    tree.insert_tokens(&[], "http://w1:8000");
    assert_eq!(token_tree_totals(&tree), (0, 0));

    let page = tree.page_size();
    let page_a: Vec<u32> = (0..page as u32).collect();
    tree.insert_tokens(&page_a, "http://w1:8000");
    assert_eq!(token_tree_totals(&tree), (page, 1));

    // A shared page counts its tokens for every tenant on it.
    tree.insert_tokens(&page_a, "http://w2:8000");
    assert_eq!(token_tree_totals(&tree), (2 * page, 2));

    // A disjoint page adds only for its tenant.
    let page_b: Vec<u32> = (1000..1000 + page as u32).collect();
    tree.insert_tokens(&page_b, "http://w1:8000");
    assert_eq!(token_tree_totals(&tree), (3 * page, 2));
}

// ---- is_kv_imbalanced: KV triggers (overload ∨ KV-spread) ----

/// Single-DP load snapshot with the given waiting-prefill backlog.
fn waiting_load(waiting_uncached: i32) -> WorkerLoadResponse {
    WorkerLoadResponse {
        loads: vec![SchedulerLoadSnapshot {
            num_waiting_uncached_tokens: waiting_uncached,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Single-DP load snapshot reporting the given KV utilization (0.0–1.0).
fn kv_load(token_usage: f64) -> WorkerLoadResponse {
    WorkerLoadResponse {
        loads: vec![SchedulerLoadSnapshot {
            token_usage,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Single-DP snapshot used to make live request count and LeastLoad's
/// expected-wait score disagree deterministically. A request is running, so
/// the rate gauge counts as live (and the scorer prices that request at the
/// mean prefill, the same for every worker).
fn expected_wait_load(
    queued_tokens: i32,
    token_usage: f64,
    gen_throughput: f64,
) -> WorkerLoadResponse {
    WorkerLoadResponse {
        loads: vec![SchedulerLoadSnapshot {
            num_running_reqs: 1,
            num_waiting_uncached_tokens: queued_tokens,
            token_usage,
            gen_throughput,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Healthy workers (health checks disabled) for the given URLs.
fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
    urls.iter()
        .map(|u| {
            Arc::new(
                BasicWorkerBuilder::new(*u)
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

fn update_expected_wait_loads(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    queued_tokens: &[i32],
) {
    let loads = workers
        .iter()
        .zip(queued_tokens)
        .map(|(worker, &queued)| {
            (
                worker.url().to_string(),
                expected_wait_load(queued, 0.1, 100.0),
            )
        })
        .collect();
    TestFeed::attached_to(policy).publish(&loads);
}

/// Like [`update_expected_wait_loads`], with each report counting the
/// requests the router holds on its worker (`worker.load()`) as waiting
/// requests: the engine has every request the router dispatched to it.
/// The expected-wait scorer charges a worker for requests the router
/// holds beyond what its report and since-poll credit account for (work
/// in transit the engine has not seen), so a fixture that holds requests
/// either reports them, as here, or models that in-transit backlog.
fn update_expected_wait_loads_counting_held(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    queued_tokens: &[i32],
) {
    let loads = workers
        .iter()
        .zip(queued_tokens)
        .map(|(worker, &queued)| {
            let mut load = expected_wait_load(queued, 0.1, 100.0);
            load.loads[0].num_waiting_reqs = i32::try_from(worker.load()).unwrap_or(i32::MAX);
            (worker.url().to_string(), load)
        })
        .collect();
    TestFeed::attached_to(policy).publish(&loads);
}

/// One selection for `text` through the string path.
fn route_text(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>], text: &str) -> Option<usize> {
    policy.select_worker(
        workers,
        &SelectWorkerInfo {
            request_text: Some(text),
            ..Default::default()
        },
    )
}

fn assert_only_final_worker_credited(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    final_idx: usize,
    expected_tokens: u64,
) {
    for (idx, worker) in workers.iter().enumerate() {
        assert_eq!(
            worker.processed_requests(),
            usize::from(idx == final_idx),
            "worker {idx} processed count must reflect exactly one final dispatch"
        );
        let (_, credited_tokens, credited_requests) =
            policy.load_scorer.load_state_for_test(worker.url());
        assert_eq!(
            (credited_tokens, credited_requests),
            if idx == final_idx {
                (expected_tokens, 1)
            } else {
                (0, 0)
            },
            "worker {idx} LeastLoad credit must reflect exactly one final dispatch"
        );
    }
}

/// Inject a backend KV snapshot (utilization per worker, by index). Returns
/// the sender; bind it (`let _tx = ...`) to keep the watch channel open.
fn inject_kv(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    usages: &[f64],
) -> watch::Sender<Arc<LoadSnapshot>> {
    let snapshot = LoadSnapshot::from_loads_for_test(
        workers
            .iter()
            .zip(usages)
            .map(|(w, &u)| (w.url().to_string(), kv_load(u)))
            .collect(),
    );
    let (tx, rx) = watch::channel(snapshot);
    policy.set_load_receiver(Some(rx));
    tx
}

/// Config isolating the KV triggers (count effectively disabled): `balance`
/// is the spread threshold, `overload` the ceiling.
fn kv_only_config(balance_spread: f32, overload_ceiling: f32) -> CacheAwareConfig {
    CacheAwareConfig {
        balance_abs_threshold: usize::MAX,
        eviction_interval_secs: 0,
        balance_token_usage_threshold: balance_spread,
        overload_token_usage_threshold: overload_ceiling,
        ..Default::default()
    }
}

fn all_healthy(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    (0..workers.len()).collect()
}

fn imbalanced(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>]) -> bool {
    policy.is_kv_imbalanced(workers, &all_healthy(workers))
}

#[test]
fn is_kv_imbalanced_uniform_high_kv_does_not_fire() {
    // All engines equally saturated: high utilization, zero spread.
    let policy = CacheAwarePolicy::with_config(kv_only_config(0.3, 0.95));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let _tx = inject_kv(&policy, &workers, &[0.9, 0.9, 0.9]);
    // max 0.9 < 0.95 ceiling, spread 0.0 < 0.3 → keep cache affinity.
    assert!(
        !imbalanced(&policy, &workers),
        "uniform-high KV (no cooler home) must not abandon cache affinity"
    );
}

#[test]
fn is_kv_imbalanced_one_hot_rest_idle_fires_via_spread() {
    // Same hottest engine (0.9) as the uniform case, but neighbors are idle.
    let policy = CacheAwarePolicy::with_config(kv_only_config(0.3, 0.95));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let _tx = inject_kv(&policy, &workers, &[0.9, 0.15, 0.15]);
    // spread 0.75 > 0.3 → spill toward a cooler engine.
    assert!(
        imbalanced(&policy, &workers),
        "a hot engine with idle neighbors (large KV spread) must rebalance"
    );
}

#[test]
fn is_kv_imbalanced_overload_ceiling_fires_below_spread() {
    // Critically hot engine, but the spread is under the balance threshold.
    let policy = CacheAwarePolicy::with_config(kv_only_config(0.3, 0.95));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let _tx = inject_kv(&policy, &workers, &[0.97, 0.80]);
    // spread 0.17 < 0.3 (balance quiet) but 0.97 > 0.95 ceiling → shed.
    assert!(
        imbalanced(&policy, &workers),
        "a critically-saturated engine must shed even below the spread threshold"
    );
}

#[test]
fn is_kv_imbalanced_ignores_request_count_spread() {
    // Count dispersion alone never abandons affinity fleet-wide; count
    // pressure is applied per request by the candidate gate instead.
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_abs_threshold: 5,
        balance_rel_threshold: 2.0,
        eviction_interval_secs: 0,
        balance_token_usage_threshold: 0.3,
        overload_token_usage_threshold: 0.95,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let _tx = inject_kv(&policy, &workers, &[0.3, 0.3]);
    for _ in 0..20 {
        workers[0].increment_load();
    }
    assert!(
        !imbalanced(&policy, &workers),
        "count spread alone must not disable cache affinity fleet-wide"
    );
}

#[test]
fn is_kv_imbalanced_kv_disabled_by_default_ignores_snapshot() {
    // Default config: both KV thresholds 1.0 (disabled).
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    // A massive KV spread that WOULD fire if KV balancing were enabled...
    let _tx = inject_kv(&policy, &workers, &[0.95, 0.05]);
    // ...is ignored at the 1.0 default; counts balanced → no rebalance.
    assert!(
        !imbalanced(&policy, &workers),
        "default thresholds (1.0) must ignore KV usage entirely"
    );
}

#[test]
fn cache_aware_always_requests_backend_load_updates() {
    // Cache misses, affinity ties, and spills all use the embedded
    // expected-wait scorer, so even default CacheAware configuration must
    // participate in WorkerMonitor load polling.
    let policy = CacheAwarePolicy::with_config(test_config());
    assert!(policy.needs_backend_loads());
}

#[test]
fn cache_aware_load_update_resets_atomic_inflight_credit() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    // `w2` holds one request the router dispatched, and its report counts
    // it: one waiting request of 500 uncached tokens. Right after the
    // update `w2` is the busier worker by count (1 vs 0) and the more
    // expensive by wait (5 s vs 0 s), so the first miss goes to `w1`;
    // once that miss credits `w1` with 1,024 tokens (10.24 s), `w2` is
    // the busier by count but the cheaper by wait, which is the state the
    // second-miss assertion checks. The report hides nothing, so the
    // charge for requests a report does not show stays out of this test.
    workers[1].increment_load();
    update_expected_wait_loads_counting_held(&policy, &workers, &[0, 500]);

    let route_miss = |text| {
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(text),
                ..Default::default()
            },
        )
    };
    assert_eq!(route_miss("alpha miss"), Some(0));
    assert_eq!(
        policy.load_scorer.load_state_for_test(workers[0].url()),
        (true, 1024, 1)
    );
    assert_eq!(
        route_miss("zulu miss"),
        Some(1),
        "first dispatch credit must steer the second miss away"
    );
    assert_eq!(
        policy.load_scorer.load_state_for_test(workers[1].url()),
        (true, 1024, 1)
    );

    update_expected_wait_loads_counting_held(&policy, &workers, &[0, 500]);
    for worker in &workers {
        assert_eq!(
            policy.load_scorer.load_state_for_test(worker.url()),
            (true, 0, 0),
            "fresh load update must clear exactly the since-poll credit"
        );
    }
    assert_eq!(
        route_miss("omega miss"),
        Some(0),
        "fresh backend load must reset since-poll dispatch credit"
    );
}

#[test]
fn a_miss_avoids_the_worker_whose_report_hides_the_requests_the_router_holds() {
    // `w2` holds six requests the router dispatched while its fresh
    // report shows an idle engine: dispatches still in transit, the
    // backlog that herded a fleet onto its emptiest-looking workers. `w1`
    // reports one running request and one queued request of 10 tokens.
    // Read from the reports alone, `w2` is the emptier worker and takes
    // the miss; charged six mean prefills (61.44 s at 100 tokens/s, `w1`'s
    // live rate standing in for its idle gauge) for the requests its
    // report hides, it loses every miss to `w1` until `w1`'s own credit
    // has grown to as much.
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    for _ in 0..6 {
        workers[1].increment_load();
    }
    let mut one_queued = expected_wait_load(10, 0.1, 100.0);
    one_queued.loads[0].num_waiting_reqs = 1;
    let mut idle = expected_wait_load(0, 0.1, 100.0);
    idle.loads[0].num_running_reqs = 0;
    TestFeed::attached_to(&policy).publish(&HashMap::from([
        (workers[0].url().to_string(), one_queued),
        (workers[1].url().to_string(), idle),
    ]));
    for text in ["alpha", "bravo", "charlie", "delta", "echo"] {
        assert_eq!(
            route_text(&policy, &workers, text),
            Some(0),
            "{text}: the miss went to the worker whose report hides its load"
        );
    }
    let (_, credited_tokens, credited_requests) =
        policy.load_scorer.load_state_for_test(workers[0].url());
    assert_eq!((credited_tokens, credited_requests), (5 * 1024, 5));
}

#[test]
fn a_request_the_report_already_counts_is_not_priced_again() {
    // `w1` holds three requests and its report counts them as running:
    // three mean prefills at 100 tokens/s = 30.72 s, nothing more. `w2`
    // holds nothing, reports nothing running and 3,100 queued tokens =
    // 31 s at `w1`'s live rate (its own idle gauge does not count).
    // Charged a second time for the three, `w1` would read as 61.44 s
    // and lose.
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    for _ in 0..3 {
        workers[0].increment_load();
    }
    let mut running = expected_wait_load(0, 0.1, 100.0);
    running.loads[0].num_running_reqs = 3;
    let mut queued = expected_wait_load(3_100, 0.1, 100.0);
    queued.loads[0].num_running_reqs = 0;
    TestFeed::attached_to(&policy).publish(&HashMap::from([
        (workers[0].url().to_string(), running),
        (workers[1].url().to_string(), queued),
    ]));
    assert_eq!(route_text(&policy, &workers, "alpha"), Some(0));

    // The same three requests dispatched after `w1`'s report: the
    // since-poll credit carries them (3 × 1024 tokens = 30.72 s), and
    // nothing is charged on top of the credit either. Both reports carry
    // the fixture's one running request (10.24 s each), so it cancels:
    // 40.96 s against `w2`'s 41.24 s.
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    TestFeed::attached_to(&policy).publish(&HashMap::from([
        (
            workers[0].url().to_string(),
            expected_wait_load(0, 0.1, 100.0),
        ),
        (
            workers[1].url().to_string(),
            expected_wait_load(3_100, 0.1, 100.0),
        ),
    ]));
    for text in ["bravo", "charlie", "delta"] {
        assert_eq!(route_text(&policy, &workers, text), Some(0));
        workers[0].increment_load();
    }
    assert_eq!(
        policy.load_scorer.load_state_for_test(workers[0].url()),
        (true, 3 * 1024, 3)
    );
    assert_eq!(
        route_text(&policy, &workers, "echo"),
        Some(0),
        "requests the credit carries were charged again"
    );
}

#[test]
fn a_report_the_snapshot_dropped_is_no_longer_scored() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    let feed = TestFeed::attached_to(&policy);
    feed.publish(&HashMap::from([
        (
            workers[0].url().to_string(),
            expected_wait_load(0, 0.1, 100.0),
        ),
        (
            workers[1].url().to_string(),
            expected_wait_load(1_000, 0.1, 100.0),
        ),
    ]));

    // The next snapshot drops w1 after its load fetch fails. Its old
    // report said idle, but its five live requests make the documented
    // missing-report estimate (51.2s) worse than w2's reported 10s queue.
    for _ in 0..5 {
        workers[0].increment_load();
    }
    feed.publish(&HashMap::from([(
        workers[1].url().to_string(),
        expected_wait_load(1_000, 0.1, 100.0),
    )]));

    let selected = policy.select_worker(
        &workers,
        &SelectWorkerInfo {
            request_text: Some("snapshot-pruning miss"),
            ..Default::default()
        },
    );
    assert_eq!(
        selected,
        Some(1),
        "a report the snapshot dropped must not stay scoreable"
    );
    assert_only_final_worker_credited(&policy, &workers, 1, 1024);
}

#[test]
fn the_shared_snapshot_is_the_score_source() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    let feed = TestFeed::attached_to(&policy);
    feed.publish(&HashMap::from([
        (
            workers[0].url().to_string(),
            expected_wait_load(10_000, 0.1, 100.0),
        ),
        (
            workers[1].url().to_string(),
            expected_wait_load(0, 0.1, 100.0),
        ),
    ]));
    assert_eq!(
        route_text(&policy, &workers, "alpha snapshot miss"),
        Some(1)
    );

    // The next publication turns the scores around: nothing is kept
    // policy-side, and the credit w2 took is released by its new report.
    feed.publish(&HashMap::from([
        (
            workers[0].url().to_string(),
            expected_wait_load(0, 0.1, 100.0),
        ),
        (
            workers[1].url().to_string(),
            expected_wait_load(10_000, 0.1, 100.0),
        ),
    ]));
    assert_eq!(route_text(&policy, &workers, "zulu snapshot miss"), Some(0));
}

#[test]
fn cache_aware_reset_discards_the_in_flight_credit() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    update_expected_wait_loads(&policy, &workers, &[100_000, 0]);

    let first = SelectWorkerInfo {
        request_text: Some("alpha reset miss"),
        ..Default::default()
    };
    assert_eq!(policy.select_worker(&workers, &first), Some(1));
    assert_eq!(
        policy.load_scorer.load_state_for_test(workers[1].url()),
        (true, 1024, 1)
    );

    policy.reset();
    for worker in &workers {
        assert_eq!(
            policy.load_scorer.load_state_for_test(worker.url()),
            (true, 0, 0),
            "reset must clear the in-flight credit and leave the feed alone"
        );
    }
    let second = SelectWorkerInfo {
        request_text: Some("zulu reset miss"),
        ..Default::default()
    };
    assert_eq!(policy.select_worker(&workers, &second), Some(1));
}

#[test]
fn cache_aware_worker_removal_prunes_expected_wait_state() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    // Both reports sampled before the dispatch below, so republishing w2's
    // report releases nothing.
    let sampled_at = Instant::now();
    let mut loads: HashMap<String, WorkerLoadResponse> = workers
        .iter()
        .zip([100_000, 0])
        .map(|(worker, queued)| {
            let mut load = expected_wait_load(queued, 0.1, 100.0);
            load.sampled_at = Some(sampled_at);
            (worker.url().to_string(), load)
        })
        .collect();
    let feed = TestFeed::attached_to(&policy);
    feed.publish(&loads);

    let first = SelectWorkerInfo {
        request_text: Some("alpha removal miss"),
        ..Default::default()
    };
    assert_eq!(policy.select_worker(&workers, &first), Some(1));

    // The monitor evicts the removed worker's report; the policy drops
    // its tally.
    loads.remove(workers[0].url());
    feed.publish(&loads);
    LoadBalancingPolicy::remove_worker(&policy, workers[0].url());
    assert_eq!(
        policy.load_scorer.load_state_for_test(workers[0].url()),
        (false, 0, 0),
        "worker removal must prune the in-flight credit"
    );
    assert_eq!(
        policy.load_scorer.load_state_for_test(workers[1].url()),
        (true, 1024, 1),
        "worker removal must preserve other workers' state"
    );
    // Scored like its reporting peer once the stale queue is gone, the
    // removed worker is picked again instead of shunned.
    let picked_removed = (0..50).any(|i| {
        let text = format!("{i} removal miss");
        let miss = SelectWorkerInfo {
            request_text: Some(&text),
            ..Default::default()
        };
        policy.select_worker(&workers, &miss) == Some(0)
    });
    assert!(
        picked_removed,
        "removed worker's stale heavy snapshot must not survive cleanup"
    );
}

// ---- per-request candidate gate + matched-tenant selection ----

fn seed_string_tenants(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    text: &str,
    tenant_urls: &[&str],
) -> Arc<Tree> {
    policy.init_workers(workers);
    let model_key = normalize_model_key(workers[0].model_id()).to_string();
    let tree = Arc::clone(policy.string_trees.get(&model_key).unwrap().value());
    for url in tenant_urls {
        tree.insert_text(text, url);
    }
    tree
}

/// Route once so `model_key`-scoped trees exist, then seed the token tree
/// with the given tenants for `tokens`.
fn seed_token_tenants(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    tokens: &[u32],
    tenant_urls: &[&str],
) -> Arc<TokenTree> {
    policy.init_workers(workers);
    let model_key = normalize_model_key(workers[0].model_id()).to_string();
    let tree = Arc::clone(policy.token_trees.get(&model_key).unwrap().value());
    for url in tenant_urls {
        tree.insert_tokens(tokens, url);
    }
    tree
}

#[test]
fn string_tree_unique_hit_keeps_affinity_and_credits_expected_wait() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let cached = "alpha unique cached prompt";
    seed_string_tenants(&policy, &workers, cached, &["http://w1:8000"]);
    workers[1].increment_load();
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

    let hit = SelectWorkerInfo {
        request_text: Some(cached),
        ..Default::default()
    };
    assert_eq!(policy.select_worker(&workers, &hit), Some(0));
    assert_only_final_worker_credited(&policy, &workers, 0, 1024);
}

/// A prefill cache miss favors less queued input regardless of output rate.
#[test]
fn prefill_miss_ignores_generation_rate() {
    for generated in [0.0, 0.5, 10_000.0] {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = ["http://p1:8000", "http://p2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Prefill)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        let feed = TestFeed::attached_to(&policy.load_scorer);
        feed.publish(&HashMap::from([
            (
                workers[0].url().to_string(),
                expected_wait_load(2_000, 0.0, generated),
            ),
            (
                workers[1].url().to_string(),
                expected_wait_load(4_000, 0.0, 2_000.0),
            ),
        ]));
        assert_eq!(route_text(&policy, &workers, "new prefill prompt"), Some(0));
    }
}

#[test]
fn string_tree_miss_uses_expected_wait_and_records_final_worker() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
    let text = "novel string-tree miss with no shared prefix";

    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(text),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 1, "expected wait must beat raw request count");
    assert_only_final_worker_credited(&policy, &workers, 1, 1024);

    let model = normalize_model_key(workers[0].model_id());
    let tree = policy.string_trees.get(model).unwrap();
    let matched = tree.match_prefix_with_counts(text);
    assert!(matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[1].url()));
    assert!(!matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[0].url()));
}

#[test]
fn string_tree_spill_uses_expected_wait_and_records_only_final_worker() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let text = "shared string prefix that must spill";
    let tree = seed_string_tenants(&policy, &workers, text, &["http://w1:8000"]);
    for _ in 0..100 {
        workers[0].increment_load();
    }
    for _ in 0..10 {
        workers[2].increment_load();
    }
    // Every report counts the requests its worker holds (100 queued on
    // `w1`, 10 on `w3`); `w3`'s ten queue 1,000 uncached tokens against
    // `w2`'s 10,000, so the expected wait prefers `w3` although its live
    // count is the higher: the spill must follow the wait, not the count.
    update_expected_wait_loads_counting_held(&policy, &workers, &[20_000, 10_000, 1_000]);

    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(text),
                ..Default::default()
            },
        )
        .unwrap();
    assert_only_final_worker_credited(&policy, &workers, 2, 1024);
    assert_eq!(selected, 2, "spill must use expected wait over the fleet");

    let matched = tree.match_prefix_with_counts(text);
    assert!(matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[2].url()));
    assert!(!matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[1].url()));
}

#[test]
fn string_tree_equal_affinity_tie_uses_expected_wait_only_among_holders() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let text = "equal string affinity";
    seed_string_tenants(
        &policy,
        &workers,
        text,
        &["http://w1:8000", "http://w2:8000"],
    );
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(text),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 1);
    assert_only_final_worker_credited(&policy, &workers, 1, 1024);
}

#[test]
fn token_tree_unique_hit_keeps_affinity_and_credits_expected_wait() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let cached: Vec<u32> = (0..8).collect();
    seed_token_tenants(&policy, &workers, &cached, &["http://w1:8000"]);
    workers[1].increment_load();
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

    let hit = SelectWorkerInfo {
        tokens: Some(&cached),
        ..Default::default()
    };
    assert_eq!(policy.select_worker(&workers, &hit), Some(0));
    assert_only_final_worker_credited(&policy, &workers, 0, 8);
}

#[test]
fn token_tree_miss_uses_expected_wait_and_records_final_worker() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
    let tokens: Vec<u32> = (100..108).collect();

    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 1, "expected wait must beat raw request count");
    assert_only_final_worker_credited(&policy, &workers, 1, 8);

    let model = normalize_model_key(workers[0].model_id());
    let tree = policy.token_trees.get(model).unwrap();
    let matched = tree.match_prefix_with_counts(&tokens);
    assert!(matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[1].url()));
    assert!(!matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[0].url()));
}

#[test]
fn token_tree_spill_uses_expected_wait_and_records_only_final_worker() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let tokens: Vec<u32> = (0..8).collect();
    let tree = seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
    for _ in 0..100 {
        workers[0].increment_load();
    }
    for _ in 0..10 {
        workers[2].increment_load();
    }
    // Every report counts the requests its worker holds (100 queued on
    // `w1`, 10 on `w3`); `w3`'s ten queue 1,000 uncached tokens against
    // `w2`'s 10,000, so the expected wait prefers `w3` although its live
    // count is the higher: the spill must follow the wait, not the count.
    update_expected_wait_loads_counting_held(&policy, &workers, &[20_000, 10_000, 1_000]);

    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 2, "spill must use expected wait over the fleet");
    assert_only_final_worker_credited(&policy, &workers, 2, 8);

    let matched = tree.match_prefix_with_counts(&tokens);
    assert!(matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[2].url()));
    assert!(!matched
        .matched_tenants
        .iter()
        .any(|tenant| tenant.as_ref() == workers[1].url()));
}

#[test]
fn token_tree_equal_affinity_tie_uses_expected_wait_only_among_holders() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let tokens: Vec<u32> = (0..8).collect();
    seed_token_tenants(
        &policy,
        &workers,
        &tokens,
        &["http://w1:8000", "http://w2:8000"],
    );
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 1);
    assert_only_final_worker_credited(&policy, &workers, 1, 8);
}

#[test]
fn cache_hit_prefers_least_loaded_matched_tenant() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let tokens: Vec<u32> = (0..32).collect();
    seed_token_tenants(
        &policy,
        &workers,
        &tokens,
        &["http://w1:8000", "http://w2:8000"],
    );
    for _ in 0..5 {
        workers[0].increment_load();
    }

    // w2 and w3 are equally idle, but only w1/w2 hold the prefix: the
    // selection must stay within the matched tenants and take the less
    // loaded one.
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1, "least-loaded matched tenant, not any idle worker");
}

#[test]
fn gated_hot_tenant_spills_to_expected_wait_and_replicates() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..32).collect();
    let tree = seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
    // avg = 50; w1 clears both margins (100 > 50 * 1.1 and 100 > 50 + 32).
    for _ in 0..100 {
        workers[0].increment_load();
    }

    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1, "gated selection must spill to expected wait");

    // The spill inserted for w2, so the prefix now has a second tenant.
    let result = tree.match_prefix_with_counts(&tokens);
    assert!(
        result
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == "http://w2:8000"),
        "spill target must become a tenant of the hot prefix"
    );
}

#[test]
#[traced_test]
fn token_tree_selection_emits_decision_line() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..32).collect();
    seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);

    policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(logs_contain("Cache-aware selection"));
    assert!(logs_contain("tree_match"));

    let novel: Vec<u32> = (1000..1064).collect();
    policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&novel),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(logs_contain("expected_wait_fallback"));
}

/// The branch counter and match-ratio histogram are recorded on every
/// tree decision, independent of whether the debug line is enabled, and
/// the ratio is bucketed the way `start_prometheus` registers it: an
/// exact decile (32 of 320) must land in `le="0.1"`, not widen past it.
#[test]
fn token_tree_selection_records_branch_and_match_ratio_metrics() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..32).collect();
    seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
    let novel: Vec<u32> = (1000..1064).collect();
    let decile: Vec<u32> = (0..320).collect();

    let recorder = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full(String::from("smg_cache_aware_match_ratio")),
            CACHE_AWARE_MATCH_RATIO_BUCKETS,
        )
        .unwrap()
        .build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        // A full-prefix hit (1.0), a novel prompt (0.0), and a prompt
        // sharing only the seeded 32 tokens of 320 (0.1).
        for request in [&tokens, &novel, &decile] {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(request),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    });
    let rendered = handle.render();

    for series in [
        "smg_cache_aware_policy_branch_total{branch=\"tree_match\"} 1",
        "smg_cache_aware_policy_branch_total{branch=\"expected_wait_fallback\"} 2",
        "smg_cache_aware_match_ratio_bucket{le=\"0\"} 1",
        "smg_cache_aware_match_ratio_bucket{le=\"0.1\"} 2",
        "smg_cache_aware_match_ratio_bucket{le=\"0.9\"} 2",
        "smg_cache_aware_match_ratio_bucket{le=\"1\"} 3",
        "smg_cache_aware_match_ratio_count 3",
    ] {
        assert!(
            rendered.lines().any(|l| l == series),
            "{series} missing; rendered:\n{rendered}"
        );
    }
}

#[test]
fn count_spread_elsewhere_keeps_cache_affinity() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..32).collect();
    seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
    // Fleet-wide count spread (50 vs 0) that formerly disabled affinity
    // outright — but the loaded worker is not the request's tenant.
    for _ in 0..50 {
        workers[1].increment_load();
    }

    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        idx, 0,
        "a deep queue on another worker must not break this request's affinity"
    );
}

/// The affinity pick of a two-worker pool whose prefix lives on w1, with
/// `w1_load` and `w2_load` requests in flight.
fn affinity_pick(config: CacheAwareConfig, w1_load: usize, w2_load: usize) -> usize {
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..32).collect();
    seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
    for _ in 0..w1_load {
        workers[0].increment_load();
    }
    for _ in 0..w2_load {
        workers[1].increment_load();
    }
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        ..Default::default()
    };
    policy.select_worker(&workers, &info).unwrap()
}

#[test]
fn candidate_gate_requires_both_margins() {
    // w1 load 7 vs avg 5: over the relative margin (7 > 5 * 1.1) but not
    // the absolute one (the swing at a mean of 5 is five: 7 is not
    // > 5 + 5) — affinity holds.
    assert_eq!(affinity_pick(test_config(), 7, 3), 0);
    // w1 load 12 vs avg 6: both margins clear (12 > 6.6, 12 > 6 + 5):
    // spill.
    assert_eq!(affinity_pick(test_config(), 12, 0), 1);
    // An absolute threshold under the floor lowers it: with none at all
    // the relative margin decides alone, and 7 vs avg 5 spills.
    let rel_only = CacheAwareConfig {
        balance_abs_threshold: 0,
        ..test_config()
    };
    assert_eq!(affinity_pick(rel_only, 7, 3), 1);
}

/// The launcher's defaults: a 64-request absolute margin and 1.5 times
/// the mean.
fn launcher_defaults() -> CacheAwareConfig {
    CacheAwareConfig {
        balance_abs_threshold: 64,
        balance_rel_threshold: 1.5,
        ..test_config()
    }
}

#[test]
fn a_small_pool_spills_a_worker_holding_the_relative_margin_more_than_its_peer() {
    // Two workers, 32 in flight. A 25/7 skew (the busier worker at 1.56
    // times the mean) is over the gate: 25 > 1.5 * 16 and 25 > 16 + 8.
    // Under a fixed 64-request margin the busier worker would have needed
    // 81 in flight, more than the pool holds, so this never spilled.
    assert_eq!(affinity_pick(launcher_defaults(), 25, 7), 1);
    // A 19/13 skew (1.19 times the mean) is under the relative margin:
    // affinity holds, as it should at this scale.
    assert_eq!(affinity_pick(launcher_defaults(), 19, 13), 0);
}

#[test]
fn the_absolute_margin_is_capped_at_the_counts_swing_and_floored_at_four() {
    let policy = CacheAwarePolicy::with_config(launcher_defaults());
    // (mean, margin): the floor up to a mean of 4, two standard
    // deviations of the count above it, the configured 64 from a mean
    // of 1,024 on.
    for (avg_load, margin) in [
        (0.0, 4.0),
        (1.0, 4.0),
        (4.0, 4.0),
        (5.0, 5.0),
        (16.0, 8.0),
        (64.0, 16.0),
        (128.0, 23.0),
        (400.0, 40.0),
        (1024.0, 64.0),
        (2000.0, 64.0),
    ] {
        assert_eq!(
            policy.absolute_margin(avg_load, 0.0),
            margin,
            "mean {avg_load}"
        );
    }
    // No absolute threshold at all keeps the gate relative-only.
    let rel_only = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_abs_threshold: 0,
        ..launcher_defaults()
    });
    assert_eq!(rel_only.absolute_margin(16.0, 0.0), 0.0);
}

#[test]
fn the_gate_opens_at_the_load_the_pools_mean_sets() {
    let policy = CacheAwarePolicy::with_config(launcher_defaults());
    // Pools of 2, 8 and 29 workers at 4, 8, 32 and 128 requests in
    // flight: the smallest in-flight count one worker must hold for the
    // gate to open (None: more than the pool holds). A fixed 64-request
    // margin opened only for a worker holding 81 of 128 on 8 workers and
    // 69 of 128 on 29.
    for (workers, in_flight, opens_at) in [
        (2, 4, None),
        (2, 8, None),
        (2, 32, Some(25)),
        (2, 128, Some(97)),
        (8, 4, None),
        (8, 8, Some(6)),
        (8, 32, Some(9)),
        (8, 128, Some(25)),
        (29, 4, None),
        (29, 8, Some(5)),
        (29, 32, Some(6)),
        (29, 128, Some(10)),
    ] {
        let avg_load = f64::from(in_flight) / f64::from(workers);
        let first = (0..=in_flight)
            .find(|&load| policy.over_count_pressure_gate(f64::from(load), avg_load, 0.0));
        assert_eq!(first, opens_at, "{workers} workers, {in_flight} in flight");
    }
    // At high means the relative margin decides, with either rule.
    assert!(!policy.over_count_pressure_gate(290.0, 200.0, 0.0));
    assert!(policy.over_count_pressure_gate(301.0, 200.0, 0.0));
}

/// The launcher's spill thresholds with a held prefix priced at 1,024
/// tokens a request (LeastLoad's mean prefill).
fn priced_defaults() -> CacheAwareConfig {
    CacheAwareConfig {
        balance_prefix_margin_tokens: 1_024,
        ..launcher_defaults()
    }
}

/// The affinity pick of a two-worker pool whose `prompt_tokens`-token
/// prompt is held whole by w1, with `w1_load` and `w2_load` requests in
/// flight.
fn held_prompt_pick(
    config: CacheAwareConfig,
    prompt_tokens: u32,
    w1_load: usize,
    w2_load: usize,
) -> usize {
    held_prompt_pick_as(config, WorkerRole::Single, prompt_tokens, w1_load, w2_load)
}

/// [`held_prompt_pick`] for a selection of the given PD role.
fn held_prompt_pick_as(
    config: CacheAwareConfig,
    role: WorkerRole,
    prompt_tokens: u32,
    w1_load: usize,
    w2_load: usize,
) -> usize {
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..prompt_tokens).collect();
    seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
    for _ in 0..w1_load {
        workers[0].increment_load();
    }
    for _ in 0..w2_load {
        workers[1].increment_load();
    }
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        role,
        ..Default::default()
    };
    policy.select_worker(&workers, &info).unwrap()
}

#[test]
fn a_deep_prefix_keeps_its_holder_at_a_skew_the_swing_margin_spills() {
    // Two workers, 32 in flight, 25/7: the swing-sized margin (8 at a mean
    // of 16) spills a short prompt (see above). A 65,536-token prompt held
    // whole by w1 is 64 requests' worth of prefill at a price of 1,024
    // tokens, so its margin is the configured 64 and 25 is not over 16 + 64:
    // the conversation stays with its cache.
    assert_eq!(held_prompt_pick(priced_defaults(), 65_536, 25, 7), 0);
    // Priced at 0, the held prefix does not count and the swing spills it.
    let unpriced = CacheAwareConfig {
        balance_prefix_margin_tokens: 0,
        ..priced_defaults()
    };
    assert_eq!(held_prompt_pick(unpriced, 65_536, 25, 7), 1);
}

#[test]
fn held_prefixes_are_unpriced_by_default() {
    // The default config prices nothing: the deep prefix above spills at
    // 25/7 under the swing-sized margin, as before the price existed.
    assert_eq!(CacheAwareConfig::default().balance_prefix_margin_tokens, 0);
    assert_eq!(held_prompt_pick(launcher_defaults(), 65_536, 25, 7), 1);
}

#[test]
fn a_decode_pick_keeps_the_swing_margin_whatever_is_held() {
    // A decode worker prefills nothing again, so a priced deep prefix does
    // not hold a decode pick: it spills at 25/7 as an unpriced one does,
    // while a prefill pick of the same prompt keeps its holder.
    assert_eq!(
        held_prompt_pick_as(priced_defaults(), WorkerRole::Decode, 65_536, 25, 7),
        1
    );
    assert_eq!(
        held_prompt_pick_as(priced_defaults(), WorkerRole::Prefill, 65_536, 25, 7),
        0
    );
}

#[test]
fn a_pick_the_price_keeps_counts_as_prefix_margin_held_alone() {
    // A priced deep prefix kept at 25/7 counts once on its own counter and
    // under no extra branch label; unpriced, the same pick spills and the
    // counter stays absent.
    for (price, held) in [(1_024, true), (0, false)] {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let config = CacheAwareConfig {
            balance_prefix_margin_tokens: price,
            ..priced_defaults()
        };
        let pick =
            metrics::with_local_recorder(&recorder, || held_prompt_pick(config, 65_536, 25, 7));
        assert_eq!(pick, if held { 0 } else { 1 }, "price {price}");
        let rendered = handle.render();
        assert_eq!(
            rendered
                .lines()
                .any(|l| l == "smg_cache_aware_prefix_margin_held_total 1"),
            held,
            "price {price}; rendered:\n{rendered}"
        );
        assert!(
            !rendered.contains("prefix_margin_held\""),
            "no branch label; rendered:\n{rendered}"
        );
    }
}

#[test]
fn the_absolute_margin_weighs_the_held_prefix() {
    let policy = CacheAwarePolicy::with_config(priced_defaults());
    // (mean, held tokens, margin): a prefix up to the swing's worth keeps
    // the swing; a deeper one raises the margin to its price, capped at
    // the configured 64; the floor holds under a shallow one.
    for (avg_load, held_tokens, margin) in [
        (16.0, 0.0, 8.0),
        (16.0, 8_192.0, 8.0),
        (16.0, 20_480.0, 20.0),
        (16.0, 100_000.0, 64.0),
        (4.0, 2_048.0, 4.0),
    ] {
        assert_eq!(
            policy.absolute_margin(avg_load, held_tokens),
            margin,
            "mean {avg_load}, held {held_tokens}"
        );
    }
    // A price of 0 leaves the margin to the swing.
    let unpriced = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_prefix_margin_tokens: 0,
        ..priced_defaults()
    });
    assert_eq!(unpriced.absolute_margin(16.0, 100_000.0), 8.0);
    // No absolute threshold keeps the gate relative-only, whatever is held.
    let rel_only = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_abs_threshold: 0,
        ..priced_defaults()
    });
    for (avg_load, held_tokens) in [(16.0, 0.0), (16.0, 100_000.0), (400.0, 1_000_000.0)] {
        assert_eq!(rel_only.absolute_margin(avg_load, held_tokens), 0.0);
    }
}

#[test]
fn a_deep_prefix_keeps_a_two_worker_pools_gate_shut_up_to_128_in_flight() {
    let policy = CacheAwarePolicy::with_config(priced_defaults());
    // Two workers at 32, 72 and 128 in flight: the smallest in-flight count
    // one worker must hold for the gate to open on a 4,096-token prefix
    // (the swing decides) and on a 100,000-token one (the configured 64
    // decides: no worker of these pools can be 64 above the mean; past 128
    // in flight one can).
    for (in_flight, shallow_opens_at) in [(32, Some(25)), (72, Some(55)), (128, Some(97))] {
        let avg_load = f64::from(in_flight) / 2.0;
        let opens_at = |held_tokens: f64| {
            (0..=in_flight).find(|&load| {
                policy.over_count_pressure_gate(f64::from(load), avg_load, held_tokens)
            })
        };
        assert_eq!(opens_at(4_096.0), shallow_opens_at, "{in_flight} in flight");
        assert_eq!(opens_at(100_000.0), None, "{in_flight} in flight");
    }
}

#[test]
fn the_configured_absolute_threshold_caps_the_held_prefix() {
    // An absolute threshold of 16 caps the margin of any prefix: a holder of
    // a 65,536-token prompt at 40 of 40 in flight (mean 20) is over 1.5 times
    // the mean and 16 above it, and spills.
    let tuned = CacheAwareConfig {
        balance_abs_threshold: 16,
        ..priced_defaults()
    };
    assert_eq!(held_prompt_pick(tuned, 65_536, 40, 0), 1);
}

#[test]
fn the_string_tree_prices_a_held_prefix_at_four_characters_a_token() {
    // 262,144 characters held whole are 65,536 tokens' worth: kept at 25/7.
    // 32,768 characters are 8,192 tokens, no more than the swing at a mean
    // of 16 (8 requests' worth): spilled, as a 32,768-token prefix would not
    // be.
    for (chars, expected) in [(262_144usize, 0usize), (32_768, 1)] {
        let policy = CacheAwarePolicy::with_config(priced_defaults());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let text = "a".repeat(chars);
        seed_string_tenants(&policy, &workers, &text, &["http://w1:8000"]);
        for _ in 0..25 {
            workers[0].increment_load();
        }
        for _ in 0..7 {
            workers[1].increment_load();
        }
        assert_eq!(
            route_text(&policy, &workers, &text),
            Some(expected),
            "{chars} chars"
        );
    }
}

#[test]
fn an_event_index_holder_of_a_deep_prefix_keeps_it() {
    // The event-driven path prices the blocks a holder reports: 16,384
    // blocks of 4 tokens on w1 keep a 65,536-token prompt there at 25/7;
    // priced at 0 it spills.
    let tokens: Vec<u32> = (0..65_536).collect();
    let chunks: Vec<&[u32]> = tokens.chunks(4).collect();
    for (price, expected) in [(1_024, 0usize), (0, 1)] {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_prefix_margin_tokens: price,
            ..launcher_defaults()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let monitor = Arc::new(KvEventMonitor::new());
        monitor.indexers.insert(
            "unknown".to_string(),
            setup_indexer_with_blocks("http://w1:8000", &chunks),
        );
        policy.set_kv_event_monitor(Some(monitor));
        for _ in 0..25 {
            workers[0].increment_load();
        }
        for _ in 0..7 {
            workers[1].increment_load();
        }
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_eq!(
            policy.select_worker(&workers, &info),
            Some(expected),
            "price {price}"
        );
    }
}

#[test]
fn a_hash_index_holder_is_priced_at_its_boundary() {
    // A holder recorded at the 65,536-token boundary keeps the prompt at
    // 25/7; one recorded at the 4,096-token boundary holds no more than the
    // swing's worth and spills.
    for (boundary, expected) in [(65_536usize, 0usize), (4_096, 1)] {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            balance_prefix_margin_tokens: 1_024,
            ..hash_config(&[boundary])
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let tokens: Vec<u32> = (0..boundary as u32).collect();
        policy.record_placement(
            model,
            None,
            &tokens,
            &[boundary],
            workers[0].url(),
            Instant::now(),
        );
        for _ in 0..25 {
            workers[0].increment_load();
        }
        for _ in 0..7 {
            workers[1].increment_load();
        }
        assert_eq!(
            route_tokens(&policy, &workers, &tokens),
            expected,
            "boundary {boundary}"
        );
    }
}

/// Five hits on a 100,000-token prompt whose first 99,000 tokens w1 holds,
/// then two cold prompts of 20,000 and 10,000 tokens, under equal load
/// reports: the pick of the second cold prompt.
fn cold_pick_after_cached_hits(balance_uncached_credit: bool) -> usize {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_uncached_credit,
        ..test_config()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let context: Vec<u32> = (0..99_000).collect();
    seed_token_tenants(&policy, &workers, &context, &["http://w1:8000"]);
    update_expected_wait_loads(&policy, &workers, &[0, 0]);
    let route = |tokens: &[u32]| {
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(tokens),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    for turn in 0..5u32 {
        let mut prompt = context.clone();
        prompt.extend(1_000_000 + turn * 1_000..1_000_000 + (turn + 1) * 1_000);
        assert_eq!(route(&prompt), 0, "hit {turn} stays with its holder");
    }
    let first_cold: Vec<u32> = (2_000_000..2_020_000).collect();
    assert_eq!(
        route(&first_cold),
        1,
        "the first cold prompt takes the idle worker"
    );
    let second_cold: Vec<u32> = (3_000_000..3_010_000).collect();
    route(&second_cold)
}

/// The token-work LeastLoad books on w1 after it takes a `prompt_len`-token
/// prompt whose first `held_len` tokens it holds, picked for `role`.
fn credit_booked_for_a_hit(
    balance_uncached_credit: bool,
    role: WorkerRole,
    prompt_len: u32,
    held_len: u32,
) -> u64 {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_uncached_credit,
        ..test_config()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let prompt: Vec<u32> = (0..prompt_len).collect();
    seed_token_tenants(
        &policy,
        &workers,
        &prompt[..held_len as usize],
        &["http://w1:8000"],
    );
    let info = SelectWorkerInfo {
        tokens: Some(&prompt),
        role,
        ..Default::default()
    };
    assert_eq!(
        policy.select_worker(&workers, &info),
        Some(0),
        "the holder takes it"
    );
    policy.load_scorer.load_state_for_test("http://w1:8000").1
}

#[test]
fn uncached_credit_never_books_more_than_the_whole_prompt() {
    // A 512-token prompt with 384 tokens held still needs 128, under the
    // mean prefill's floor of 1,024: the credit stops at the 512 a miss, or
    // the uncredited pick, books.
    assert_eq!(
        credit_booked_for_a_hit(true, WorkerRole::Single, 512, 384),
        512
    );
    assert_eq!(
        credit_booked_for_a_hit(false, WorkerRole::Single, 512, 384),
        512
    );
    // A long prompt books what it still needs, floored at the mean prefill.
    assert_eq!(
        credit_booked_for_a_hit(true, WorkerRole::Single, 100_000, 99_000),
        1_024
    );
    assert_eq!(
        credit_booked_for_a_hit(true, WorkerRole::Single, 100_000, 90_000),
        10_000
    );
}

#[test]
fn uncached_credit_leaves_a_decode_pick_booked_whole() {
    // A decode worker prefills nothing again, so its pick keeps the whole
    // prompt's credit; the prefill pick of the same prompt books what it
    // still needs.
    assert_eq!(
        credit_booked_for_a_hit(true, WorkerRole::Decode, 100_000, 99_000),
        100_000
    );
    assert_eq!(
        credit_booked_for_a_hit(true, WorkerRole::Prefill, 100_000, 99_000),
        1_024
    );
}

#[test]
fn uncached_credit_books_a_hit_at_the_prefill_it_still_needs() {
    // Booked whole, w1's five hits weigh 500,000 tokens against w2's cold
    // 20,000, so the second cold prompt goes to w2. Booked at the prefill
    // each still needed (1,000 tokens, floored at the mean prefill of 1,024:
    // 5,120 in all), w1 is the emptier worker and takes it.
    assert_eq!(cold_pick_after_cached_hits(false), 1);
    assert_eq!(cold_pick_after_cached_hits(true), 0);
}

#[test]
fn kv_pressure_still_forces_expected_wait_fallback() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        block_size: 4,
        ..kv_only_config(0.3, 0.95)
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..32).collect();
    seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
    workers[0].increment_load();
    // KV spread 0.8 > 0.3: shed fleet-wide despite the w1 cache hit.
    let _tx = inject_kv(&policy, &workers, &[0.9, 0.1]);

    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1, "KV pressure must still override cache affinity");
}

#[test]
fn test_cache_aware_worker_removal() {
    let config = CacheAwareConfig {
        eviction_interval_secs: 0, // Disable eviction thread
        ..Default::default()
    };
    let policy = CacheAwarePolicy::with_config(config);
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

    policy.init_workers(&workers);

    // Route some requests
    policy.select_worker(
        &workers,
        &SelectWorkerInfo {
            request_text: Some("test1"),
            ..Default::default()
        },
    );
    policy.select_worker(
        &workers,
        &SelectWorkerInfo {
            request_text: Some("test2"),
            ..Default::default()
        },
    );

    // Remove a worker
    policy.remove_worker_by_url("http://w1:8000");
    workers[0].set_status(WorkerStatus::NotReady);

    // All requests should now go to worker2
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("test1"),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1);
}

#[test]
fn test_remove_worker_purges_tree_tenants() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    // Seed both tenants explicitly; worker cleanup is the behavior under
    // test, independent of random tie-breaking in a fully dark fleet.
    let model_key = normalize_model_key(workers[0].model_id()).to_string();
    let string_tree = Arc::clone(policy.string_trees.get(&model_key).unwrap().value());
    let token_tree = Arc::clone(policy.token_trees.get(&model_key).unwrap().value());
    string_tree.insert_text("purge me please", workers[0].url());
    string_tree.insert_text("keep me around", workers[1].url());
    let tokens_a: Vec<u32> = (0..32).collect();
    let tokens_b: Vec<u32> = (1000..1032).collect();
    token_tree.insert_tokens(&tokens_a, workers[0].url());
    token_tree.insert_tokens(&tokens_b, workers[1].url());

    let char_counts = string_tree.get_tenant_char_count();
    let token_counts = token_tree.get_tenant_token_counts();
    for url in ["http://w1:8000", "http://w2:8000"] {
        assert!(char_counts.contains_key(url), "precondition: {url} routed");
        assert!(token_counts.contains_key(url), "precondition: {url} routed");
    }

    policy.remove_worker(workers[0].as_ref());

    let char_counts = string_tree.get_tenant_char_count();
    assert!(!char_counts.contains_key("http://w1:8000"));
    assert!(char_counts.contains_key("http://w2:8000"));
    let token_counts = token_tree.get_tenant_token_counts();
    assert!(!token_counts.contains_key("http://w1:8000"));
    assert!(token_counts.contains_key("http://w2:8000"));

    policy.remove_worker_by_url("http://w2:8000");
    assert!(string_tree.get_tenant_char_count().is_empty());
    assert!(token_tree.get_tenant_token_counts().is_empty());
}

#[test]
fn test_remove_workers_from_model_does_not_mutate_other_models() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        cache_boundaries: vec![16],
        ..Default::default()
    });
    let worker_url = "http://shared-url:8000";
    let tokens: Vec<u32> = (0..32).collect();

    for model_id in ["retired-model", "active-model"] {
        let string_tree = Arc::new(Tree::new());
        string_tree.insert_text("shared prefix", worker_url);
        policy
            .string_trees
            .insert(model_id.to_string(), string_tree);

        let token_tree = Arc::new(policy.new_token_tree());
        token_tree.insert_tokens(&tokens, worker_url);
        policy.token_trees.insert(model_id.to_string(), token_tree);

        policy.record_placement(model_id, None, &tokens, &[16], worker_url, Instant::now());
    }

    policy.remove_workers_from_model("retired-model", &HashSet::from([worker_url.to_string()]));

    let retired_string = policy.string_trees.get("retired-model").unwrap();
    let retired_token = policy.token_trees.get("retired-model").unwrap();
    assert!(!retired_string
        .get_tenant_char_count()
        .contains_key(worker_url));
    assert!(!retired_token
        .get_tenant_token_counts()
        .contains_key(worker_url));
    assert!(policy
        .placement_index
        .get("retired-model")
        .is_some_and(|placements| placements.is_empty()));

    let active_string = policy.string_trees.get("active-model").unwrap();
    let active_token = policy.token_trees.get("active-model").unwrap();
    assert!(active_string
        .get_tenant_char_count()
        .contains_key(worker_url));
    assert!(active_token
        .get_tenant_token_counts()
        .contains_key(worker_url));
    assert!(policy
        .placement_index
        .get("active-model")
        .is_some_and(|placements| !placements.is_empty()));
}

#[test]
fn test_apply_known_remote_insert_round_trip() {
    // Seed both kinds via `apply_repair_page` (the v2 cold-start
    // path that populates hash_index), then verify
    // `apply_known_remote_insert` resolves the hash and returns
    // true. Unknown hashes return false. Wrong-kind lookups
    // against the same hash return false (model + kind scope
    // the index).
    let config = CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    };
    let policy = CacheAwarePolicy::with_config(config);

    let text = "remote_text";
    let tokens = vec![1u32, 2, 3, 4];
    let string_page = TreeRepairPage {
        session_id: uuid::Uuid::now_v7(),
        model_id: "model1".to_string(),
        tree_kind: TreeKind::String,
        page_index: 0,
        entries: vec![RepairEntry::String {
            path: text.to_string(),
            tenants: vec![(Arc::from("http://w1"), 1)],
        }],
        next_cursor: None,
        is_last: true,
    };
    assert_eq!(policy.apply_repair_page(&string_page), 1);

    let token_page = TreeRepairPage {
        session_id: uuid::Uuid::now_v7(),
        model_id: "model1".to_string(),
        tree_kind: TreeKind::Token,
        page_index: 0,
        entries: vec![RepairEntry::Token {
            tokens: tokens.clone(),
            tenants: vec![(Arc::from("http://w1"), 1)],
        }],
        next_cursor: None,
        is_last: true,
    };
    assert_eq!(policy.apply_repair_page(&token_page), 1);

    let text_hash = kv_index::hash_node_path(text);
    let token_hash = kv_index::hash_token_path(&tokens);

    // Known hashes apply for the matching kind.
    assert!(policy.apply_known_remote_insert("model1", TreeKind::String, text_hash, "http://w2",));
    assert!(policy.apply_known_remote_insert("model1", TreeKind::Token, token_hash, "http://w2",));

    // Same hash but wrong kind doesn't alias.
    assert!(!policy.apply_known_remote_insert("model1", TreeKind::Token, text_hash, "http://w2",));

    // Unknown hash, unknown model → false.
    assert!(!policy.apply_known_remote_insert(
        "model1",
        TreeKind::String,
        0xDEAD_BEEF,
        "http://w2",
    ));
    assert!(!policy.apply_known_remote_insert(
        "unknown_model",
        TreeKind::String,
        text_hash,
        "http://w2",
    ));
}

#[test]
fn test_apply_repair_page_seeds_hash_index() {
    let config = CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    };
    let policy = CacheAwarePolicy::with_config(config);
    let text = "repaired text";
    let tokens = vec![1u32; 16];

    let string_page = TreeRepairPage {
        session_id: uuid::Uuid::now_v7(),
        model_id: "model1".to_string(),
        tree_kind: TreeKind::String,
        page_index: 0,
        entries: vec![RepairEntry::String {
            path: text.to_string(),
            tenants: vec![(Arc::from("http://w1"), 1)],
        }],
        next_cursor: None,
        is_last: true,
    };
    assert_eq!(policy.apply_repair_page(&string_page), 1);
    assert!(policy.apply_known_remote_insert(
        "model1",
        TreeKind::String,
        kv_index::hash_node_path(text),
        "http://w2",
    ));

    let token_page = TreeRepairPage {
        session_id: uuid::Uuid::now_v7(),
        model_id: "model1".to_string(),
        tree_kind: TreeKind::Token,
        page_index: 0,
        entries: vec![RepairEntry::Token {
            tokens: tokens.clone(),
            tenants: vec![(Arc::from("http://w1"), 1)],
        }],
        next_cursor: None,
        is_last: true,
    };
    assert_eq!(policy.apply_repair_page(&token_page), 1);
    assert!(policy.apply_known_remote_insert(
        "model1",
        TreeKind::Token,
        kv_index::hash_token_path(&tokens),
        "http://w2",
    ));
}

#[test]
fn test_apply_known_remote_insert_from_request_hot_path() {
    // Companion to `test_apply_known_remote_insert_round_trip`.
    // That test seeds via `apply_repair_page`, which stores
    // full text/tokens. The local request hot path
    // (`select_worker_with_text` / `_with_tokens` plus the
    // imbalanced fallback) stores the *matched prefix* shape
    // instead. A regression on the matched-prefix apply path
    // would still pass the full-path test, so seed via
    // `select_worker` here and assert apply succeeds.
    //
    // Opt into request-hot-path hash_index population — without
    // this the populate sites are no-ops and the apply call
    // below would have nothing to resolve. In production this
    // flag is flipped by the mesh wiring code; here we set it
    // directly because the test mimics the mesh consumer.
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    policy.set_populate_hash_index(true);
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    // Drive a string request through select_worker — populates
    // the string-side hash_index with a matched-prefix value.
    let text = "the quick brown fox jumps over the lazy dog";
    policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some(text),
                ..Default::default()
            },
        )
        .unwrap();
    let text_hash = kv_index::hash_node_path(text);

    // Drive a token request — populates the token-side
    // hash_index. select_worker uses the model_id from the
    // first worker's `model_id()`, which the builder leaves
    // empty → UNKNOWN_MODEL_ID after normalization.
    let tokens: Vec<u32> = (0..32).collect();
    policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    let token_hash = kv_index::hash_token_path(&tokens);

    // Both populate sites use UNKNOWN_MODEL_ID for these
    // workers (no model_id set on the builder), and the
    // resolver normalizes empty → UNKNOWN_MODEL_ID, so an
    // empty model_id resolves the same entries the populate
    // sites wrote.
    assert!(policy.apply_known_remote_insert("", TreeKind::String, text_hash, "http://remote",));
    assert!(policy.apply_known_remote_insert("", TreeKind::Token, token_hash, "http://remote",));
}

#[test]
fn test_cache_aware_without_mesh() {
    let config = CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    };
    let policy = CacheAwarePolicy::with_config(config);

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
        BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .api_key("test_api_key")
            .health_config(no_health_check())
            .build(),
    )];

    policy.init_workers(&workers);

    // Should work without mesh
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("test request"),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 0);
}

// -----------------------------------------------------------------------
// Event-driven routing tests (Type 1: KV index overlap scoring)
// -----------------------------------------------------------------------

/// Helper: create a KV index and store blocks for a worker.
/// `token_chunks` is a list of token-id slices — each becomes one block.
fn setup_indexer_with_blocks(worker_url: &str, token_chunks: &[&[u32]]) -> Arc<KvIndex> {
    let indexer = Arc::new(KvIndex::chain());
    let worker_id = indexer.intern_worker(worker_url).unwrap();
    let mut wb = WorkerBlocks::default();
    let blocks: Vec<StoredBlock> = token_chunks
        .iter()
        .enumerate()
        .map(|(i, tokens)| StoredBlock {
            seq_hash: SequenceHash(i as u64 + 1),
            content_hash: compute_content_hash(tokens),
        })
        .collect();
    indexer
        .apply_stored(worker_id, &blocks, None, &mut wb)
        .unwrap();
    indexer
}

fn test_config() -> CacheAwareConfig {
    CacheAwareConfig {
        eviction_interval_secs: 0,
        block_size: 4, // small block size for easy test setup
        ..Default::default()
    }
}

// -- overlap affinity-group unit tests --

#[test]
fn test_overlap_affinity_group_selects_best_match() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    // Store 4 blocks for w1: tokens [1..16] in blocks of 4
    let indexer = setup_indexer_with_blocks(
        "http://w1:8000",
        &[
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            &[13, 14, 15, 16],
        ],
    );

    // Query with matching tokens — should select w1
    let result = overlap_affinity_group(
        &workers,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        &[0, 1],
        &indexer,
        4,
        &default_tuning(),
    );
    assert_eq!(result, vec![0]); // w1
}

#[test]
fn test_equal_overlap_forms_one_affinity_group() {
    // Equal overlap remains one affinity group so LeastLoad, rather than
    // raw load or slice order, owns the final tie-break.
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

    let chunks: [&[u32]; 2] = [&[1, 2, 3, 4], &[5, 6, 7, 8]];
    let indexer = setup_indexer_with_blocks("http://w1:8000", &chunks);
    // Same content cached on w2 under distinct backend seq hashes.
    let w2 = indexer.intern_worker("http://w2:8000").unwrap();
    let mut wb2 = WorkerBlocks::default();
    let blocks: Vec<StoredBlock> = chunks
        .iter()
        .enumerate()
        .map(|(i, tokens)| StoredBlock {
            seq_hash: SequenceHash(100 + i as u64),
            content_hash: compute_content_hash(tokens),
        })
        .collect();
    indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();

    assert_eq!(
        overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        ),
        vec![0, 1]
    );
}

/// Two workers with identical cached blocks (the tie-test topology): both
/// fully match the request.
fn equal_overlap_fixture() -> (Vec<Arc<dyn Worker>>, Arc<KvIndex>) {
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let chunks: [&[u32]; 2] = [&[1, 2, 3, 4], &[5, 6, 7, 8]];
    let indexer = setup_indexer_with_blocks("http://w1:8000", &chunks);
    let w2 = indexer.intern_worker("http://w2:8000").unwrap();
    let mut wb2 = WorkerBlocks::default();
    let blocks: Vec<StoredBlock> = chunks
        .iter()
        .enumerate()
        .map(|(i, tokens)| StoredBlock {
            seq_hash: SequenceHash(100 + i as u64),
            content_hash: compute_content_hash(tokens),
        })
        .collect();
    indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();
    (workers, indexer)
}

#[test]
fn test_overlap_decay_prefers_less_backlogged_worker() {
    // Equal overlap, equal load — but w2 carries waiting-prefill backlog.
    // With decay on, the fleet-floor worker keeps full credit and must
    // win every draw (previously this tie was a coin flip).
    let (workers, indexer) = equal_overlap_fixture();
    let waiting = LoadSnapshot::from_loads_for_test(vec![
        ("http://w1:8000".to_string(), waiting_load(0)),
        ("http://w2:8000".to_string(), waiting_load(8)),
    ]);
    let tuning = OverlapTuning {
        overlap_decay: 4.0,
        selection_temperature: 0.0,
        waiting_prefill_tokens: Some(&waiting),
    };
    assert_eq!(
        overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1],
            &indexer,
            4,
            &tuning,
        ),
        vec![0],
        "backlogged worker must lose its affinity edge"
    );
}

#[test]
fn test_overlap_decay_missing_load_data_never_decays() {
    // Only w1 reports load (and holds the floor at zero backlog); w2 has
    // no entry. Neither may be decayed, so the equal-score tie — and its
    // random spreading — must survive.
    let (workers, indexer) = equal_overlap_fixture();
    let waiting =
        LoadSnapshot::from_loads_for_test(vec![("http://w1:8000".to_string(), waiting_load(0))]);
    let tuning = OverlapTuning {
        overlap_decay: 4.0,
        selection_temperature: 0.0,
        waiting_prefill_tokens: Some(&waiting),
    };
    assert_eq!(
        overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1],
            &indexer,
            4,
            &tuning,
        ),
        vec![0, 1],
        "workers without load data must not be decayed"
    );
}

/// w1 caches both request blocks (score 2), w2 only the first (score 1).
fn unequal_overlap_fixture() -> (Vec<Arc<dyn Worker>>, Arc<KvIndex>) {
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
    let w2 = indexer.intern_worker("http://w2:8000").unwrap();
    let mut wb2 = WorkerBlocks::default();
    let blocks = vec![StoredBlock {
        seq_hash: SequenceHash(100),
        content_hash: compute_content_hash(&[1, 2, 3, 4]),
    }];
    indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();
    (workers, indexer)
}

#[test]
fn test_selection_temperature_spreads_but_favors_better_score() {
    // At temperature 0 the better scorer wins every draw; at temperature
    // 1.0 the weaker scorer must be sampled sometimes, while the better
    // one keeps the majority (p(best) = 1/(1+e^-1) ≈ 0.73).
    let (workers, indexer) = unequal_overlap_fixture();
    for _ in 0..50 {
        let group = overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(group, vec![0], "temperature 0 must be exact argmax");
    }

    let tuning = OverlapTuning {
        overlap_decay: 0.0,
        selection_temperature: 1.0,
        waiting_prefill_tokens: None,
    };
    let mut counts = [0usize; 2];
    for _ in 0..300 {
        let group = overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1],
            &indexer,
            4,
            &tuning,
        );
        assert_eq!(group.len(), 1, "unequal scores select one score band");
        counts[group[0]] += 1;
    }
    assert!(
        counts[1] > 0,
        "temperature must spread picks to the weaker scorer"
    );
    assert!(
        counts[0] > counts[1],
        "better score must keep the majority: {counts:?}"
    );
}

#[test]
fn test_selection_temperature_preserves_equal_score_group() {
    // A temperature draw chooses an affinity score band. Equal-score
    // workers stay together so LeastLoad can resolve the tie.
    let (workers, indexer) = equal_overlap_fixture();
    let tuning = OverlapTuning {
        overlap_decay: 0.0,
        selection_temperature: 0.5,
        waiting_prefill_tokens: None,
    };
    assert_eq!(
        overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1],
            &indexer,
            4,
            &tuning,
        ),
        vec![0, 1]
    );
}

#[test]
fn test_overlap_affinity_group_no_match_is_empty() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000"]);
    policy.init_workers(&workers);

    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]]);

    // Completely different tokens — no overlap → None
    let result = overlap_affinity_group(
        &workers,
        &[100, 200, 300, 400, 500, 600, 700, 800],
        &[0],
        &indexer,
        4,
        &default_tuning(),
    );
    assert!(result.is_empty());
}

#[test]
fn test_raw_load_does_not_break_equal_affinity() {
    let policy = CacheAwarePolicy::with_config(test_config());

    let w1 = BasicWorkerBuilder::new("http://w1:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    let w2 = BasicWorkerBuilder::new("http://w2:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    // Give w1 higher load
    for _ in 0..10 {
        w1.increment_load();
    }

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
    policy.init_workers(&workers);

    // Store same blocks for both workers (equal overlap)
    let indexer = Arc::new(KvIndex::chain());
    let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
    let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
    let mut wb1 = WorkerBlocks::default();
    let mut wb2 = WorkerBlocks::default();
    let blocks = vec![StoredBlock {
        seq_hash: SequenceHash(1),
        content_hash: compute_content_hash(&[1, 2, 3, 4]),
    }];
    indexer
        .apply_stored(w1_id, &blocks, None, &mut wb1)
        .unwrap();
    let blocks2 = vec![StoredBlock {
        seq_hash: SequenceHash(1),
        content_hash: compute_content_hash(&[1, 2, 3, 4]),
    }];
    indexer
        .apply_stored(w2_id, &blocks2, None, &mut wb2)
        .unwrap();

    // Equal overlap remains one group despite different raw loads;
    // production delegates the final tie-break to LeastLoad.
    let result = overlap_affinity_group(
        &workers,
        &[1, 2, 3, 4],
        &[0, 1],
        &indexer,
        4,
        &default_tuning(),
    );
    assert_eq!(result, vec![0, 1]);
}

#[test]
fn test_tree_size_does_not_break_equal_affinity() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    let indexer = Arc::new(KvIndex::chain());
    let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
    let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
    let mut wb1 = WorkerBlocks::default();
    let mut wb2 = WorkerBlocks::default();

    // Both workers have block [1,2,3,4] (equal overlap, equal load)
    let block = vec![StoredBlock {
        seq_hash: SequenceHash(1),
        content_hash: compute_content_hash(&[1, 2, 3, 4]),
    }];
    indexer.apply_stored(w1_id, &block, None, &mut wb1).unwrap();

    // w2 has the same block plus extra blocks → larger tree
    let block2 = vec![StoredBlock {
        seq_hash: SequenceHash(1),
        content_hash: compute_content_hash(&[1, 2, 3, 4]),
    }];
    indexer
        .apply_stored(w2_id, &block2, None, &mut wb2)
        .unwrap();
    let extra = vec![StoredBlock {
        seq_hash: SequenceHash(2),
        content_hash: compute_content_hash(&[5, 6, 7, 8]),
    }];
    indexer
        .apply_stored(w2_id, &extra, Some(SequenceHash(1)), &mut wb2)
        .unwrap();

    // Equal overlap, equal load, different tree sizes: both stay in the
    // affinity group for LeastLoad to resolve.
    assert_eq!(
        overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        ),
        vec![0, 1]
    );
}

#[test]
fn test_overlap_affinity_group_short_request_is_empty() {
    let workers = make_workers(&["http://w1:8000"]);

    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]);

    // Request shorter than block_size → no full blocks → None
    let result = overlap_affinity_group(&workers, &[1, 2, 3], &[0], &indexer, 4, &default_tuning());
    assert!(result.is_empty());
}

#[test]
fn test_overlap_affinity_group_prefers_deeper_partial_match() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    let indexer = Arc::new(KvIndex::chain());
    let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
    let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
    let mut wb1 = WorkerBlocks::default();
    let mut wb2 = WorkerBlocks::default();

    // w1 has 4 blocks cached
    let blocks_w1: Vec<StoredBlock> = (0..4)
        .map(|i| StoredBlock {
            seq_hash: SequenceHash(i as u64 + 1),
            content_hash: compute_content_hash(&[
                (i * 4 + 1) as u32,
                (i * 4 + 2) as u32,
                (i * 4 + 3) as u32,
                (i * 4 + 4) as u32,
            ]),
        })
        .collect();
    indexer
        .apply_stored(w1_id, &blocks_w1, None, &mut wb1)
        .unwrap();

    // w2 has only the first 2 blocks (partial overlap with same request)
    let blocks_w2: Vec<StoredBlock> = (0..2)
        .map(|i| StoredBlock {
            seq_hash: SequenceHash(i as u64 + 1),
            content_hash: compute_content_hash(&[
                (i * 4 + 1) as u32,
                (i * 4 + 2) as u32,
                (i * 4 + 3) as u32,
                (i * 4 + 4) as u32,
            ]),
        })
        .collect();
    indexer
        .apply_stored(w2_id, &blocks_w2, None, &mut wb2)
        .unwrap();

    // Query with all 4 blocks worth of tokens → w1 wins (higher overlap: 4 vs 2)
    let result = overlap_affinity_group(
        &workers,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        &[0, 1],
        &indexer,
        4,
        &default_tuning(),
    );
    assert_eq!(result, vec![0]); // w1 (higher overlap)
}

// -- select_worker_event_driven integration tests --

/// Every event-driven decision observes its predicted overlap on the match
/// ratio histogram: a request cached whole on one worker reads 1, a novel
/// request 0, so the histogram is the production signal of the prediction.
#[test]
fn an_event_driven_decision_observes_its_match_ratio() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    update_expected_wait_loads(&policy, &workers, &[0, 0]);
    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let recorder = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full(String::from("smg_cache_aware_match_ratio")),
            CACHE_AWARE_MATCH_RATIO_BUCKETS,
        )
        .unwrap()
        .build_recorder();
    let handle = recorder.handle();
    let cached = [1, 2, 3, 4, 5, 6, 7, 8];
    let novel = [101, 102, 103, 104, 105, 106, 107, 108];
    metrics::with_local_recorder(&recorder, || {
        for request in [&cached, &novel] {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(request),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    });
    let rendered = handle.render();
    for series in [
        "smg_cache_aware_policy_branch_total{branch=\"event_hit\"} 1",
        "smg_cache_aware_policy_branch_total{branch=\"event_miss\"} 1",
        "smg_cache_aware_match_ratio_bucket{le=\"0\"} 1",
        "smg_cache_aware_match_ratio_bucket{le=\"0.9\"} 1",
        "smg_cache_aware_match_ratio_bucket{le=\"1\"} 2",
        "smg_cache_aware_match_ratio_count 2",
    ] {
        assert!(
            rendered.lines().any(|line| line == series),
            "missing {series} in:\n{rendered}"
        );
    }
}

#[test]
fn event_unique_hit_keeps_affinity_and_credits_expected_wait() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    workers[1].increment_load();
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let cached = [1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&cached),
                ..Default::default()
            }
        ),
        Some(0)
    );
    assert_only_final_worker_credited(&policy, &workers, 0, 8);
}

#[test]
fn a_worker_with_an_emptied_index_gets_the_warm_up_slice_whatever_its_age() {
    // The soaks' case in miniature: w1 holds a fleet-sized cache (1,100
    // blocks), w2's index was cleared by a resync (nothing indexed), and
    // every request shares a head with w1, so affinity alone would never
    // send w2 a request again. The slice does, on the thin overlap: w2 is
    // thin against the fleet's level whatever its age.
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    update_expected_wait_loads(&policy, &workers, &[0, 0]);
    let held: Vec<u32> = (1..=4_400).collect();
    let monitor = Arc::new(KvEventMonitor::new());
    // Stored as 1,100 blocks of four, as the index receives them, so the
    // fleet's level is 1,100 and w2 is thin against it; as one block of
    // 4,400 tokens the level would be one, below the warm-up blocks,
    // nobody would be thin, and the slice would never run.
    let indexer = Arc::new(KvIndex::chain());
    let holder = indexer.intern_worker("http://w1:8000").unwrap();
    indexer.intern_worker("http://w2:8000").unwrap();
    store_blocks(&indexer, holder, &held, 4, 1);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));
    // w1's first block (the shared head) followed by eleven novel blocks.
    let mut request: Vec<u32> = (1..=4).collect();
    request.extend(90_000..90_044);
    let picks: Vec<usize> = (0..8)
        .map(|_| {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(&request),
                        ..Default::default()
                    },
                )
                .unwrap()
        })
        .collect();
    assert!(
        picks.contains(&1),
        "the emptied worker got its slice of the thin-overlap requests: {picks:?}"
    );
    assert!(picks.contains(&0), "the holder kept the rest: {picks:?}");
}

#[test]
fn a_thin_worker_that_holds_more_than_the_gate_allows_gets_no_slice() {
    // Seven holders of 1,100 blocks each and an eighth worker with nothing
    // indexed: thin against the fleet's level, the slice's only candidate.
    // The router holds 100 requests on it that its report does not show
    // (dispatches queued in front of an engine that admits slowly), far
    // over the count-pressure gate (the mean in-flight is 12.5), so the
    // slice skips it and every thin-overlap request (one block of a
    // holder's head, eleven of its own) stays with its holder. Without
    // the gate the slice would feed it one such miss in four.
    let HolderFleet {
        policy,
        workers,
        heads,
        ..
    } = fleet_with_holders(7);
    for _ in 0..100 {
        workers[7].increment_load();
    }
    let mut request: Vec<u32> = heads[0][..4].to_vec();
    request.extend(90_000..90_044);
    let picks: Vec<usize> = (0..8)
        .map(|_| {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(&request),
                        ..Default::default()
                    },
                )
                .unwrap()
        })
        .collect();
    assert_eq!(
        picks,
        vec![0; 8],
        "the slice fed the thin worker that holds more than the gate allows"
    );
}

/// Store `tokens` for `worker` as blocks of `block` tokens, with sequence
/// hashes from `seq_base` (distinct per call).
fn store_blocks(indexer: &KvIndex, worker: u32, tokens: &[u32], block: usize, seq_base: u64) {
    let mut wb = WorkerBlocks::default();
    let blocks: Vec<StoredBlock> = tokens
        .chunks(block)
        .enumerate()
        .map(|(i, chunk)| StoredBlock {
            seq_hash: SequenceHash(seq_base + i as u64),
            content_hash: compute_content_hash(chunk),
        })
        .collect();
    indexer
        .apply_stored(worker, &blocks, None, &mut wb)
        .unwrap();
}

/// A fleet for the diversion tests: the policy, its eight workers, the
/// index they share and each holder's head (its first 240 tokens, 60
/// blocks); a request made of a head and a four-block tail of its own is
/// a deep hit on that holder (60 of 64 blocks) that no other request
/// repeats.
struct HolderFleet {
    policy: CacheAwarePolicy,
    workers: Vec<Arc<dyn Worker>>,
    indexer: Arc<KvIndex>,
    heads: Vec<Vec<u32>>,
}

/// Eight workers; the first `holders` hold 1,100 blocks each of their own
/// 4,400-token sequence, the rest nothing.
fn fleet_with_holders(holders: usize) -> HolderFleet {
    fleet_with_holders_of(holders, 1_100)
}

/// Eight workers; the first `holders` hold `blocks` blocks each of their
/// own sequence, the rest nothing.
fn fleet_with_holders_of(holders: usize, blocks: usize) -> HolderFleet {
    let policy = CacheAwarePolicy::with_config(test_config());
    let urls: Vec<String> = (0..8).map(|i| format!("http://w{i}:8000")).collect();
    let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
    let workers = make_workers(&refs);
    policy.init_workers(&workers);
    update_expected_wait_loads(&policy, &workers, &[0, 0, 0, 0, 0, 0, 0, 0]);
    let indexer = Arc::new(KvIndex::chain());
    let mut prefixes = Vec::new();
    for (h, url) in urls.iter().enumerate() {
        let id = indexer.intern_worker(url).unwrap();
        if h >= holders {
            continue;
        }
        let tokens: Vec<u32> = (0..blocks as u32 * 4)
            .map(|i| (h as u32 + 1) * 100_000 + i)
            .collect();
        store_blocks(&indexer, id, &tokens, 4, (h as u64 + 1) * 1_000_000);
        prefixes.push(tokens[..240].to_vec());
    }
    let monitor = Arc::new(KvEventMonitor::new());
    monitor
        .indexers
        .insert("unknown".to_string(), Arc::clone(&indexer));
    policy.set_kv_event_monitor(Some(monitor));
    HolderFleet {
        policy,
        workers,
        indexer,
        heads: prefixes,
    }
}

#[test]
fn an_emptied_worker_gets_its_share_of_hits_until_it_refills() {
    // The churn runs' case: every request is cached whole on one of seven
    // holders (a deep hit, never a miss), the eighth worker's index was
    // emptied by a resync. Affinity alone never sends it anything; the
    // diversion hands it one hit in eight (no shallow hit ever comes, so
    // the fallback at the window's end), and each diverted request lands
    // in its index, until it crosses half the fleet's level (1,100 -> 550).
    let HolderFleet {
        policy,
        workers,
        indexer,
        heads,
    } = fleet_with_holders(7);
    // The holders are busy, as on those runs: a deep hit is diverted
    // only from a holder with more in flight than the thin worker.
    for holder in &workers[..7] {
        holder.increment_load();
    }
    let emptied = indexer.worker_id("http://w7:8000").unwrap();
    // A request: a holder's 60-block head and a four-block tail of its own.
    let request = |i: usize| -> Vec<u32> {
        let mut tokens = heads[i % 7].clone();
        tokens.extend((0..16).map(|t| 90_000_000 + i as u32 * 16 + t));
        tokens
    };
    let mut decisions = 0usize;
    let mut first = None;
    let mut received = 0usize;
    while decisions < 1_000 && indexer.worker_block_count(emptied) < 550 {
        let tokens = request(decisions);
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        decisions += 1;
        if idx == 7 {
            received += 1;
            first.get_or_insert(decisions);
            // The request prefills on w7: its blocks join w7's index.
            store_blocks(
                &indexer,
                emptied,
                &tokens,
                4,
                9_000_000 + decisions as u64 * 100,
            );
        }
    }
    assert!(
        first.is_some_and(|first| first <= 8),
        "the emptied worker is served within the first eight hits, was {first:?}"
    );
    assert!(
        indexer.worker_block_count(emptied) >= 550,
        "refilled to the ratio: {} blocks after {decisions} decisions",
        indexer.worker_block_count(emptied)
    );
    assert!(
        decisions <= 40 * 8 + 16,
        "about forty requests of 64 blocks (seven heads of 60, then tails of 4) at one in eight: \
         {decisions} decisions, {received} received"
    );
    // At the ratio the diversion stops: the hit counter runs on without a
    // reset (w7 now holds the heads too and may win a tie as a holder,
    // which is affinity, not a diversion).
    let hits_before = policy.divert_hits.load(Ordering::Relaxed);
    for i in 0..32 {
        let tokens = request(decisions + i);
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    assert_eq!(
        policy.divert_hits.load(Ordering::Relaxed),
        hits_before + 32,
        "no diversion once the worker is no longer thin"
    );
}

#[test]
fn a_thin_worker_keeps_its_share_past_the_warm_up_blocks_until_the_ratio() {
    // Churn c3 (6ce84c9e): the fleet's level was 32,767 blocks a worker,
    // far above the warm-up blocks. Within a minute of the publisher
    // restart the emptied worker's index had regrown to 1,765 from the
    // decode blocks of its requests in flight, and the pool table,
    // applying the age rule's growth cap to a thin worker, dropped it
    // there after one diversion: idle for the 25 minutes to the next
    // fault. Here the level is 4,096 (the ratio at 2,048), the worker has
    // regrown to 1,100 before any diversion, every request is a deep hit
    // (60 of 64 blocks, held elsewhere), each diverted request stays in
    // flight for the rest of the test, and the table is rebuilt every
    // eight decisions as the second's refresh would.
    let HolderFleet {
        policy,
        workers,
        indexer,
        heads,
    } = fleet_with_holders_of(7, 4_096);
    // The holders are busy (twenty in flight each, a loaded fleet): the
    // diverted requests pile up on the thin worker, below that.
    for holder in &workers[..7] {
        for _ in 0..20 {
            holder.increment_load();
        }
    }
    let thin = indexer.worker_id("http://w7:8000").unwrap();
    // The growth baselines date from the first table, built on the empty
    // fleet as on the run; the regrowth comes after.
    policy.pool_table(&workers, &indexer, liveness::now_ms());
    let regrown: Vec<u32> = (0..4_400).map(|i| 80_000_000 + i).collect();
    store_blocks(&indexer, thin, &regrown, 4, 8_000_000);
    assert_eq!(indexer.worker_block_count(thin), 1_100);
    let request = |i: usize| -> Vec<u32> {
        let mut tokens = heads[i % 7].clone();
        tokens.extend((0..16).map(|t| 90_000_000 + i as u32 * 16 + t));
        tokens
    };
    let select = |tokens: &[u32]| -> usize {
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(tokens),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let mut rebuilds = 1u64;
    let mut decisions = 0usize;
    let mut first = None;
    let mut received = 0usize;
    while decisions < 2_000 && indexer.worker_block_count(thin) < 2_048 {
        if decisions.is_multiple_of(8) {
            rebuilds += 1;
            policy.pool_table(
                &workers,
                &indexer,
                liveness::now_ms() + rebuilds * POOL_TABLE_REFRESH_MS,
            );
        }
        let tokens = request(decisions);
        let idx = select(&tokens);
        decisions += 1;
        if idx != 7 {
            continue;
        }
        received += 1;
        store_blocks(
            &indexer,
            thin,
            &tokens,
            4,
            9_000_000 + decisions as u64 * 100,
        );
        // The diverted request runs on (tens of seconds on the trace).
        workers[7].increment_load();
        if first.is_none() {
            first = Some(decisions);
            // Inside the window, with that request in flight, no second
            // diversion: the hit counter runs on without a reset.
            let hits = policy.divert_hits.load(Ordering::Relaxed);
            for i in 0..16 {
                select(&request(10_000 + i));
            }
            assert_eq!(
                policy.divert_hits.load(Ordering::Relaxed),
                hits + 16,
                "no second diversion inside the window while the first is in flight"
            );
        }
        // The window elapses; the request is still in flight.
        workers[7].note_diverted(0);
    }
    assert!(
        first.is_some_and(|first| first <= 8),
        "the thin worker is served within the first eight hits although it \
         regrew past the warm-up blocks, was {first:?}"
    );
    assert!(
        indexer.worker_block_count(thin) >= 2_048,
        "refilled to the ratio across the rebuilds: {} blocks after {decisions} decisions",
        indexer.worker_block_count(thin)
    );
    assert!(
        received >= 15 && decisions <= 140 * 8 + 32,
        "one hit in eight all the way (seven heads of 60 blocks, then tails of 4): \
         {decisions} decisions, {received} received, {} in flight",
        workers[7].load()
    );
    // At the ratio the table rebuilt lists no thin worker, and with no
    // thin worker the diversion costs nothing: the hits are not even
    // counted.
    let table = policy.pool_table(
        &workers,
        &indexer,
        liveness::now_ms() + (rebuilds + 1) * POOL_TABLE_REFRESH_MS,
    );
    assert!(table.thin.is_empty(), "no longer thin: {:?}", table.thin);
    let hits = policy.divert_hits.load(Ordering::Relaxed);
    for i in 0..32 {
        select(&request(20_000 + i));
    }
    assert_eq!(
        policy.divert_hits.load(Ordering::Relaxed),
        hits,
        "no hit counted, let alone diverted, once no worker is thin"
    );
}

#[test]
fn a_deep_continuation_stays_with_its_idle_holder() {
    // A scale-out under one client: the joining worker is thin, and
    // every continuation of the client's sessions is a deep hit (60 of
    // 64 blocks) on a holder with nothing in flight. Diverting one
    // recomputes its whole context on the empty worker for a wait it
    // would not have had: none is diverted, however many hits pass.
    let HolderFleet {
        policy,
        workers,
        heads,
        ..
    } = fleet_with_holders(7);
    let request = |i: usize| -> Vec<u32> {
        let mut tokens = heads[i % 7].clone();
        tokens.extend((0..16).map(|t| 90_000_000 + i as u32 * 16 + t));
        tokens
    };
    let select = |tokens: &[u32]| -> usize {
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(tokens),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    for i in 0..64 {
        assert_eq!(
            select(&request(i)),
            i % 7,
            "a deep hit left its idle holder for the thin worker (decision {i})"
        );
    }
    // Busy holders: the recompute buys the request a shorter wait and
    // the thin worker its cache, so the diversion resumes at once (the
    // hits above have long passed a window).
    for holder in &workers[..7] {
        holder.increment_load();
    }
    assert_eq!(
        select(&request(64)),
        7,
        "with busy holders a due deep hit goes to the thin worker"
    );
}

#[test]
fn a_fleet_without_a_thin_worker_sees_no_diversion() {
    let fleet = fleet_with_holders(8);
    for i in 0..64 {
        let holder = i % 8;
        let idx = fleet
            .policy
            .select_worker(
                &fleet.workers,
                &SelectWorkerInfo {
                    tokens: Some(&fleet.heads[holder]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            idx, holder,
            "every hit stays with its holder (decision {i})"
        );
    }
}

#[test]
fn event_no_overlap_uses_expected_wait() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let novel = [100, 200, 300, 400];
    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&novel),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 1, "expected wait must beat raw request count");
    assert_only_final_worker_credited(&policy, &workers, 1, 4);
}

#[test]
fn event_spill_credits_only_expected_wait_final_worker() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    policy.init_workers(&workers);
    for _ in 0..100 {
        workers[0].increment_load();
    }
    for _ in 0..10 {
        workers[2].increment_load();
    }
    // Every report counts the requests its worker holds (100 queued on
    // `w1`, 10 on `w3`); `w3`'s ten queue 1,000 uncached tokens against
    // `w2`'s 10,000, so the expected wait prefers `w3` although its live
    // count is the higher: the spill must follow the wait, not the count.
    update_expected_wait_loads_counting_held(&policy, &workers, &[20_000, 10_000, 1_000]);

    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let tokens = [1, 2, 3, 4];
    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_only_final_worker_credited(&policy, &workers, 2, 4);
    assert_eq!(selected, 2, "spill must use expected wait over the fleet");
}

#[test]
fn event_equal_overlap_tie_uses_expected_wait_only_among_holders() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let (mut workers, indexer) = equal_overlap_fixture();
    workers.push(make_workers(&["http://w3:8000"]).pop().unwrap());
    policy.init_workers(&workers);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 1);
    assert_only_final_worker_credited(&policy, &workers, 1, 8);
}

#[test]
fn event_decay_keeps_strict_affinity_ahead_of_expected_wait() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        overlap_decay: 4.0,
        ..test_config()
    });
    let (workers, indexer) = equal_overlap_fixture();
    policy.init_workers(&workers);

    // The same coherent snapshot makes LeastLoad prefer w2 (fast, empty KV)
    // while its slightly larger waiting-prefill backlog decays w2 into a
    // strictly lower affinity score band. Affinity must remain primary.
    let overlap_loads = HashMap::from([
        (
            workers[0].url().to_string(),
            expected_wait_load(0, 0.9, 100.0),
        ),
        (
            workers[1].url().to_string(),
            expected_wait_load(8, 0.0, 1_000.0),
        ),
    ]);
    TestFeed::attached_to(&policy).publish(&overlap_loads);

    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 0, "strict decayed affinity must remain primary");
    assert_only_final_worker_credited(&policy, &workers, 0, 8);
}

#[test]
fn event_temperature_keeps_equal_score_group_for_expected_wait() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        selection_temperature: 0.5,
        ..test_config()
    });
    let (workers, indexer) = equal_overlap_fixture();
    policy.init_workers(&workers);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 1, "LeastLoad must break the sampled-score tie");
    assert_only_final_worker_credited(&policy, &workers, 1, 8);
}

#[test]
fn test_event_driven_overlap_selects_cached_worker() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    // Set up monitor with indexer data for "unknown" model
    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    // Full dispatch: should use event-driven and select w1
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 0); // w1 (has cached blocks)
}

#[test]
fn test_event_driven_no_overlap_uses_dark_fleet_expected_wait() {
    let policy = CacheAwarePolicy::with_config(test_config());

    let w1 = BasicWorkerBuilder::new("http://w1:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    let w2 = BasicWorkerBuilder::new("http://w2:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    // Give w1 higher live load so dark-fleet expected wait picks w2.
    for _ in 0..3 {
        w1.increment_load();
    }

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
    policy.init_workers(&workers);

    // Monitor has indexer with data, but tokens don't match
    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    // No overlap → event-driven falls back to expected wait (not token tree).
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[100, 200, 300, 400]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1); // w2 (dark-fleet expected wait), not token tree
}

#[test]
fn test_event_driven_gated_hot_winner_spills_to_expected_wait() {
    let policy = CacheAwarePolicy::with_config(test_config());

    let w1 = BasicWorkerBuilder::new("http://w1:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    let w2 = BasicWorkerBuilder::new("http://w2:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    // avg = 50; w1 clears both gate margins (100 > 50 * 1.1, 100 > 50 + 32).
    for _ in 0..100 {
        w1.increment_load();
    }

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
    policy.init_workers(&workers);

    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    // w1 wins the overlap score but is over both load margins: spill.
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[1, 2, 3, 4]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1, "gated overlap winner must spill to expected wait");
}

#[test]
fn test_event_driven_short_request_uses_dark_fleet_expected_wait() {
    let policy = CacheAwarePolicy::with_config(test_config()); // block_size=4

    let w1 = BasicWorkerBuilder::new("http://w1:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    let w2 = BasicWorkerBuilder::new("http://w2:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    for _ in 0..3 {
        w1.increment_load();
    }

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
    policy.init_workers(&workers);

    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    // Request shorter than block_size → no full blocks → expected-wait fallback.
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[1, 2, 3]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1); // w2 (dark-fleet expected wait)
}

#[test]
fn test_no_monitor_uses_token_tree() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    // No kv_monitor → has_event_indexer returns false → uses token tree
    assert!(!policy.has_event_indexer("unknown"));

    // Should still route (via token tree, not event-driven)
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[1, 2, 3, 4]),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(idx < 2); // valid worker selected
}

#[test]
fn test_set_kv_event_monitor() {
    let policy = CacheAwarePolicy::with_config(test_config());

    // Initially no monitor
    assert!(policy.kv_monitor.read().is_none());

    // Set monitor (works via &self thanks to interior mutability)
    let monitor = Arc::new(KvEventMonitor::new());
    policy.set_kv_event_monitor(Some(Arc::clone(&monitor)));
    assert!(policy.kv_monitor.read().is_some());

    // get_indexer returns None for unknown model
    assert!(monitor.get_indexer("nonexistent").is_none());

    // Clear monitor
    policy.set_kv_event_monitor(None);
    assert!(policy.kv_monitor.read().is_none());
}

#[test]
fn test_event_driven_uses_monitor_block_size() {
    // Test that event-driven routing uses monitor's learned block_size
    // instead of config default when available.
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        block_size: 4, // config default
        eviction_interval_secs: 0,
        ..Default::default()
    });

    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    let monitor = Arc::new(KvEventMonitor::new());

    // Store blocks using block_size=8 (tokens chunked in groups of 8)
    let indexer = Arc::new(KvIndex::chain());
    let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerBlocks::default();
    let block = vec![StoredBlock {
        seq_hash: SequenceHash(1),
        content_hash: compute_content_hash(&[1, 2, 3, 4, 5, 6, 7, 8]),
    }];
    indexer.apply_stored(w1_id, &block, None, &mut wb).unwrap();
    monitor
        .indexers
        .insert("unknown".to_string(), indexer.clone());

    // Set block_size=8 in monitor (simulating learned from events)
    monitor.set_block_size("unknown", 8);

    policy.set_kv_event_monitor(Some(monitor));

    // Query with 8 tokens — with block_size=8, this is one full block
    // With config block_size=4, this would be two blocks and wouldn't match
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 0); // w1 has the cached block
}

#[test]
fn test_imbalanced_skips_event_driven() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_abs_threshold: 5,
        balance_rel_threshold: 2.0,
        eviction_interval_secs: 0,
        block_size: 4,
        balance_token_usage_threshold: 1.0,
        overload_token_usage_threshold: 1.0,
        ..Default::default()
    });

    let w1 = BasicWorkerBuilder::new("http://w1:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    let w2 = BasicWorkerBuilder::new("http://w2:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();

    // Create heavy imbalance: w1 has 20 load, w2 has 0
    for _ in 0..20 {
        w1.increment_load();
    }

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
    policy.init_workers(&workers);

    // Even though we set up event monitor, imbalance check fires first
    let monitor = Arc::new(KvEventMonitor::new());
    policy.set_kv_event_monitor(Some(monitor));

    // With imbalance, select_worker should pick expected wait (w2), not event-driven.
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[1, 2, 3, 4]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, 1); // w2 (expected wait), regardless of event data
}

#[test]
fn test_empty_indexer_falls_through_to_token_tree() {
    // When the monitor has an indexer for a model but the indexer is empty
    // (startup, reconnect), routing should fall through to the token tree
    // instead of taking the event-driven path and landing on a fallback.
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);

    // Set up monitor with an empty indexer
    let monitor = Arc::new(KvEventMonitor::new());
    let empty_indexer = Arc::new(KvIndex::chain());
    monitor
        .indexers
        .insert("unknown".to_string(), empty_indexer);
    policy.set_kv_event_monitor(Some(monitor));

    // Empty indexer → has_event_indexer returns false → falls through to token tree
    assert!(!policy.has_event_indexer("unknown"));

    // Tokens must fill at least one tree page (the policy's block_size)
    // to populate the tree; shorter sequences are uncacheable and fall
    // through to expected wait.
    let tokens: Vec<u32> = (1..=16).collect();

    // First request populates the token tree for the selected worker.
    let idx = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(idx < 2); // valid worker via token tree

    // Same tokens again — token-tree cache hit routes to the same worker.
    let idx2 = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(idx, idx2); // token tree cache affinity preserved
}

// ---- hash placement index (cache_index = hash) ----

/// Two busy, empty-queued workers whose live rates differ so much that
/// expected wait alone would send every miss to w1.
fn skewed_rate_loads(workers: &[Arc<dyn Worker>]) -> HashMap<String, WorkerLoadResponse> {
    HashMap::from([
        (
            workers[0].url().to_string(),
            expected_wait_load(0, 0.1, 10_000.0),
        ),
        (
            workers[1].url().to_string(),
            expected_wait_load(0, 0.1, 100.0),
        ),
    ])
}

/// Gate at the mean (no margins), so an above-mean worker sits out.
fn gate_at_mean(config: CacheAwareConfig) -> CacheAwareConfig {
    CacheAwareConfig {
        balance_abs_threshold: 0,
        balance_rel_threshold: 1.0,
        ..config
    }
}

/// A request at or below `cache_threshold` has no holder to spill from,
/// and expected wait alone keeps no count bound: its queue and rate
/// signals are a poll old, so a burst of misses followed one worker
/// while the other had free slots. Misses route among the workers under
/// the same count-pressure gate a hit spills at.
#[test]
fn miss_burst_is_bounded_by_the_count_pressure_gate() {
    let policy = CacheAwarePolicy::with_config(gate_at_mean(test_config()));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    TestFeed::attached_to(&policy).publish(&skewed_rate_loads(&workers));

    let mut picks = [0usize; 2];
    for i in 0..8 {
        let text = format!("{i} novel prompt with nothing in common");
        let idx = route_text(&policy, &workers, &text).unwrap();
        // The router counts the dispatch in flight; the gate reads it.
        workers[idx].increment_load();
        picks[idx] += 1;
    }
    assert_eq!(picks, [4, 4]);
}

#[test]
fn event_miss_burst_is_bounded_by_the_count_pressure_gate() {
    let policy = CacheAwarePolicy::with_config(gate_at_mean(test_config()));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    TestFeed::attached_to(&policy).publish(&skewed_rate_loads(&workers));
    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert(
        "unknown".to_string(),
        setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]),
    );
    policy.set_kv_event_monitor(Some(monitor));

    let mut picks = [0usize; 2];
    for i in 0..8u32 {
        let novel = [100 + i, 200, 300, 400];
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&novel),
                    ..Default::default()
                },
            )
            .unwrap();
        workers[idx].increment_load();
        picks[idx] += 1;
    }
    assert_eq!(picks, [4, 4]);
}

#[test]
fn hash_miss_burst_is_bounded_by_the_count_pressure_gate() {
    let policy = CacheAwarePolicy::with_config(gate_at_mean(hash_config(&[4])));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    TestFeed::attached_to(&policy).publish(&skewed_rate_loads(&workers));

    let mut picks = [0usize; 2];
    for i in 0..8u32 {
        let head = [i, i, i, i, 9];
        let idx = route_tokens(&policy, &workers, &head);
        workers[idx].increment_load();
        picks[idx] += 1;
    }
    assert_eq!(picks, [4, 4]);
}

/// Hash mode keys on token ids, so every HTTP-router request is a miss
/// there; the same bound applies.
#[test]
fn hash_untokenized_burst_is_bounded_by_the_count_pressure_gate() {
    let policy = CacheAwarePolicy::with_config(gate_at_mean(hash_config(&[4])));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    TestFeed::attached_to(&policy).publish(&skewed_rate_loads(&workers));

    let mut picks = [0usize; 2];
    for i in 0..8 {
        let text = format!("{i} novel prompt with nothing in common");
        let idx = route_text(&policy, &workers, &text).unwrap();
        workers[idx].increment_load();
        picks[idx] += 1;
    }
    assert_eq!(picks, [4, 4]);
}

/// Two idle SGLang workers, one reporting the ~0.36 tok/s residual of
/// its last one-token health generation, the other nothing. Scored
/// against the residual, one dispatch read as ~2800 s of wait and the
/// whole burst piled onto the other worker. An idle rank has no live
/// rate, so both score at the default throughput and the misses
/// water-fill without the gate's help.
#[test]
fn idle_rate_residual_does_not_herd_misses() {
    let policy = CacheAwarePolicy::with_config(test_config());
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    let idle = |gen_throughput| WorkerLoadResponse {
        loads: vec![SchedulerLoadSnapshot {
            num_running_reqs: 0,
            gen_throughput,
            ..Default::default()
        }],
        ..Default::default()
    };
    TestFeed::attached_to(&policy).publish(&HashMap::from([
        (workers[0].url().to_string(), idle(0.36)),
        (workers[1].url().to_string(), idle(0.0)),
    ]));

    let mut picks = [0usize; 2];
    for i in 0..8 {
        let text = format!("{i} novel prompt with nothing in common");
        let idx = route_text(&policy, &workers, &text).unwrap();
        picks[idx] += 1;
    }
    assert_eq!(picks, [4, 4]);
}

fn hash_config(boundaries: &[usize]) -> CacheAwareConfig {
    CacheAwareConfig {
        eviction_interval_secs: 0,
        cache_index: CacheIndexKind::Hash,
        cache_boundaries: boundaries.to_vec(),
        ..Default::default()
    }
}

fn route_tokens(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>], tokens: &[u32]) -> usize {
    policy
        .select_worker(
            workers,
            &SelectWorkerInfo {
                tokens: Some(tokens),
                ..Default::default()
            },
        )
        .unwrap()
}

#[test]
fn hash_unique_hit_keeps_affinity_and_credits_expected_wait() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let model = normalize_model_key(workers[0].model_id());
    let cached: Vec<u32> = (0..16).collect();
    policy.record_placement(
        model,
        None,
        &cached,
        &[16],
        workers[0].url(),
        Instant::now(),
    );
    workers[1].increment_load();
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

    assert_eq!(route_tokens(&policy, &workers, &cached), 0);
    assert_only_final_worker_credited(&policy, &workers, 0, 16);
}

#[test]
fn hash_miss_uses_expected_wait_and_records_final_holder() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
    let tokens: Vec<u32> = (0..16).collect();

    let selected = route_tokens(&policy, &workers, &tokens);
    assert_eq!(selected, 1, "expected wait must beat raw request count");
    assert_only_final_worker_credited(&policy, &workers, 1, 16);

    let model = normalize_model_key(workers[0].model_id());
    let placements = policy.placement_index.get(model).unwrap();
    let holders = placements
        .get(&(16, hash_token_head(None, &tokens)))
        .expect("final dispatch must be recorded");
    assert_eq!(holders.len(), 1);
    assert_eq!(holders[0].worker_url, workers[1].url());
}

#[test]
fn hash_spill_uses_expected_wait_and_records_only_final_holder() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let model = normalize_model_key(workers[0].model_id());
    let tokens: Vec<u32> = (0..16).collect();
    policy.record_placement(
        model,
        None,
        &tokens,
        &[16],
        workers[0].url(),
        Instant::now(),
    );
    for _ in 0..100 {
        workers[0].increment_load();
    }
    for _ in 0..10 {
        workers[2].increment_load();
    }
    // Every report counts the requests its worker holds (100 queued on
    // `w1`, 10 on `w3`); `w3`'s ten queue 1,000 uncached tokens against
    // `w2`'s 10,000, so the expected wait prefers `w3` although its live
    // count is the higher: the spill must follow the wait, not the count.
    update_expected_wait_loads_counting_held(&policy, &workers, &[20_000, 10_000, 1_000]);

    let selected = route_tokens(&policy, &workers, &tokens);
    assert_eq!(selected, 2, "spill must use expected wait over the fleet");
    assert_only_final_worker_credited(&policy, &workers, 2, 16);

    let placements = policy.placement_index.get(model).unwrap();
    let holders = placements
        .get(&(16, hash_token_head(None, &tokens)))
        .unwrap();
    let holder_urls: HashSet<&str> = holders
        .iter()
        .map(|holder| holder.worker_url.as_str())
        .collect();
    assert_eq!(
        holder_urls,
        HashSet::from([workers[0].url(), workers[2].url()])
    );
}

#[test]
fn hash_short_request_uses_expected_wait_without_recording() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
    let tokens: Vec<u32> = (0..8).collect();

    let selected = route_tokens(&policy, &workers, &tokens);
    assert_eq!(selected, 1, "short fallback must use expected wait");
    assert_only_final_worker_credited(&policy, &workers, 1, 8);
    assert!(policy.placement_index.is_empty());
}

#[test]
fn hash_multiple_holders_use_expected_wait_only_within_affinity_set() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let model = normalize_model_key(workers[0].model_id());
    let tokens: Vec<u32> = (0..16).collect();
    let now = Instant::now();
    policy.record_placement(model, None, &tokens, &[16], workers[0].url(), now);
    policy.record_placement(model, None, &tokens, &[16], workers[1].url(), now);
    for _ in 0..5 {
        workers[1].increment_load();
    }
    update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

    let selected = route_tokens(&policy, &workers, &tokens);
    assert_eq!(selected, 1, "non-holder w3 must not break cache affinity");
    assert_only_final_worker_credited(&policy, &workers, 1, 16);
}

#[test]
fn hash_mode_never_touches_trees() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    assert!(policy.string_trees.is_empty());
    assert!(policy.token_trees.is_empty());

    let tokens: Vec<u32> = (0..32).collect();
    route_tokens(&policy, &workers, &tokens);
    policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("a text prompt long enough to insert"),
                ..Default::default()
            },
        )
        .unwrap();

    assert!(policy.string_trees.is_empty());
    assert!(policy.token_trees.is_empty());
    assert!(!policy.placement_index.is_empty());
}

#[test]
fn hash_mode_repeat_head_sticks_to_holder() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

    let tokens: Vec<u32> = (0..32).collect();
    let first = route_tokens(&policy, &workers, &tokens);
    // Mild load on the holder must not break affinity (under the gate).
    workers[first].increment_load();
    workers[first].increment_load();
    for _ in 0..5 {
        assert_eq!(route_tokens(&policy, &workers, &tokens), first);
    }

    // A different head load-balances away from the loaded holder.
    let other: Vec<u32> = (1000..1032).collect();
    assert_ne!(route_tokens(&policy, &workers, &other), first);
}

#[test]
fn hash_mode_probes_deepest_boundary_first() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16, 32]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let model = normalize_model_key(workers[0].model_id());
    let tokens: Vec<u32> = (0..40).collect();
    let now = Instant::now();

    // w2 holds the 16-token head, w1 the deeper 32-token head.
    policy.record_placement(model, None, &tokens, &[16], "http://w2:8000", now);
    policy.record_placement(model, None, &tokens, &[32], "http://w1:8000", now);
    // Even with the shallow holder strictly less loaded, the deeper
    // boundary must win.
    workers[0].increment_load();

    assert_eq!(route_tokens(&policy, &workers, &tokens), 0);
}

#[test]
fn hash_mode_records_at_every_applicable_boundary() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16, 32, 64]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let model = normalize_model_key(workers[0].model_id());

    let tokens: Vec<u32> = (0..40).collect();
    let selected = route_tokens(&policy, &workers, &tokens);

    let placements = policy.placement_index.get(model).unwrap();
    // 64 exceeds the request length: only the two applicable levels.
    assert_eq!(placements.len(), 2);
    for boundary in [16usize, 32] {
        let key = (boundary, hash_token_head(None, &tokens[..boundary]));
        let holders = placements.get(&key).unwrap();
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0].worker_url, workers[selected].url());
    }
}

#[test]
fn hash_mode_short_request_stays_dark_fleet_expected_wait() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    for _ in 0..3 {
        workers[0].increment_load();
    }

    let tokens: Vec<u32> = (0..8).collect();
    for _ in 0..5 {
        assert_eq!(route_tokens(&policy, &workers, &tokens), 1);
    }
    assert!(policy.placement_index.is_empty());
}

#[test]
fn hash_mode_untokenized_text_stays_dark_fleet_expected_wait() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    for _ in 0..3 {
        workers[0].increment_load();
    }

    let info = SelectWorkerInfo {
        request_text: Some("system prompt plus a user question"),
        ..Default::default()
    };
    for _ in 0..5 {
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
    }
    assert!(policy.placement_index.is_empty());
    assert!(policy.string_trees.is_empty());
}

#[test]
fn hash_mode_ttl_expires_holders_on_read() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let model = normalize_model_key(workers[0].model_id());
    let tokens: Vec<u32> = (0..16).collect();
    let t0 = Instant::now();
    let key = (16usize, hash_token_head(None, &tokens[..16]));

    policy.record_placement(model, None, &tokens, &[16], "http://w1:8000", t0);

    let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[0, 1]);
    // Just inside the 180s default TTL: live.
    let live_at = t0 + Duration::from_secs(179);
    assert_eq!(
        policy.live_holder_candidates(&url_to_idx, model, key, live_at),
        vec![0]
    );
    // Just past it: expired and pruned.
    let expired_at = t0 + Duration::from_secs(181);
    assert_eq!(
        policy.live_holder_candidates(&url_to_idx, model, key, expired_at),
        Vec::<usize>::new()
    );
    assert!(policy
        .placement_index
        .get(model)
        .unwrap()
        .get(&key)
        .unwrap()
        .is_empty());
}

#[test]
fn hash_mode_expired_holder_falls_back_to_expected_wait() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        cache_ttl_secs: 1,
        ..hash_config(&[16])
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

    let tokens: Vec<u32> = (0..16).collect();
    let first = route_tokens(&policy, &workers, &tokens);
    assert_eq!(route_tokens(&policy, &workers, &tokens), first);

    // Let the placement lapse, then load the former holder: a live
    // placement would still win, an expired one must load-balance away.
    std::thread::sleep(Duration::from_millis(1300));
    for _ in 0..3 {
        workers[first].increment_load();
    }
    assert_ne!(route_tokens(&policy, &workers, &tokens), first);
}

#[test]
fn hash_mode_holder_cap_evicts_stalest() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let tokens: Vec<u32> = (0..16).collect();
    let t0 = Instant::now();
    let key = (16usize, hash_token_head(None, &tokens[..16]));

    for (i, url) in [
        "http://w1:8000",
        "http://w2:8000",
        "http://w3:8000",
        "http://w4:8000",
    ]
    .iter()
    .enumerate()
    {
        policy.record_placement(
            "m",
            None,
            &tokens,
            &[16],
            url,
            t0 + Duration::from_secs(i as u64),
        );
    }

    let placements = policy.placement_index.get("m").unwrap();
    let holders = placements.get(&key).unwrap();
    assert_eq!(holders.len(), PLACEMENT_HOLDER_CAP);
    assert!(!holders.iter().any(|h| h.worker_url == "http://w1:8000"));
}

#[test]
fn hash_mode_gate_spills_and_replicates_placement() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let model = normalize_model_key(workers[0].model_id());
    let tokens: Vec<u32> = (0..16).collect();
    let key = (16usize, hash_token_head(None, &tokens[..16]));

    policy.record_placement(
        model,
        None,
        &tokens,
        &[16],
        "http://w1:8000",
        Instant::now(),
    );
    // Past both gate margins (rel 1.1 of avg 50, abs avg+32): spill.
    for _ in 0..100 {
        workers[0].increment_load();
    }

    assert_eq!(route_tokens(&policy, &workers, &tokens), 1);
    // The spill target becomes an additional holder of this head.
    let placements = policy.placement_index.get(model).unwrap();
    let holders = placements.get(&key).unwrap();
    assert!(holders.iter().any(|h| h.worker_url == "http://w2:8000"));
}

#[test]
fn hash_mode_kv_pressure_abandons_affinity() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        balance_token_usage_threshold: 0.3,
        overload_token_usage_threshold: 0.95,
        ..hash_config(&[16])
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let model = normalize_model_key(workers[0].model_id());
    let tokens: Vec<u32> = (0..16).collect();

    policy.record_placement(
        model,
        None,
        &tokens,
        &[16],
        "http://w1:8000",
        Instant::now(),
    );
    workers[0].increment_load();
    let _tx = inject_kv(&policy, &workers, &[0.9, 0.1]);

    // KV spread 0.8 > 0.3: shortest queue wins over the placement.
    assert_eq!(route_tokens(&policy, &workers, &tokens), 1);
}

#[test]
fn hash_mode_co_hashes_short_and_long_requests() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[2048]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

    // 3k- and 17k-token requests sharing the 2048-token head must land
    // on the same worker via the shared boundary key.
    let short: Vec<u32> = (0..3000).collect();
    let mut long: Vec<u32> = (0..2048).collect();
    long.extend(9_000_000..9_014_952);
    assert_eq!(long.len(), 17_000);

    let first = route_tokens(&policy, &workers, &short);
    assert_eq!(route_tokens(&policy, &workers, &long), first);
}

#[test]
fn hash_mode_removed_worker_is_purged_from_placements() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let tokens: Vec<u32> = (0..16).collect();
    let now = Instant::now();
    policy.record_placement("m", None, &tokens, &[16], "http://w1:8000", now);
    policy.record_placement("m", None, &tokens, &[16], "http://w2:8000", now);

    policy.remove_worker_by_url("http://w1:8000");

    let placements = policy.placement_index.get("m").unwrap();
    let holders = placements
        .get(&(16usize, hash_token_head(None, &tokens[..16])))
        .unwrap();
    assert_eq!(holders.len(), 1);
    assert_eq!(holders[0].worker_url, "http://w2:8000");
}

#[test]
fn sweep_placement_index_drops_expired_and_empty_keys() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let fresh: Vec<u32> = (0..16).collect();
    let stale: Vec<u32> = (500..516).collect();
    let t0 = Instant::now();
    policy.record_placement("m", None, &stale, &[16], "http://w1:8000", t0);
    policy.record_placement(
        "m",
        None,
        &fresh,
        &[16],
        "http://w1:8000",
        t0 + Duration::from_secs(120),
    );

    let live = CacheAwarePolicy::sweep_placement_index(
        &policy.placement_index,
        Duration::from_secs(180),
        t0 + Duration::from_secs(200),
    );
    assert_eq!(live, 1);
    let placements = policy.placement_index.get("m").unwrap();
    assert_eq!(placements.len(), 1);
    assert!(placements
        .get(&(16usize, hash_token_head(None, &fresh[..16])))
        .is_some());
}

#[test]
fn live_holder_resolution_skips_dead_holders_and_unknown_keys() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let tokens: Vec<u32> = (0..16).collect();
    let now = Instant::now();
    let key = (16usize, hash_token_head(None, &tokens[..16]));
    policy.record_placement("m", None, &tokens, &[16], "http://w1:8000", now);
    policy.record_placement("m", None, &tokens, &[16], "http://w2:8000", now);

    // w1 unhealthy: its holder entry must not resolve.
    let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[1]);
    assert_eq!(
        policy.live_holder_candidates(&url_to_idx, "m", key, now),
        vec![1]
    );
    // No healthy workers: no candidate.
    let empty = CacheAwarePolicy::healthy_url_index(&workers, &[]);
    assert_eq!(
        policy.live_holder_candidates(&empty, "m", key, now),
        Vec::<usize>::new()
    );
    // Unknown key: no candidate.
    assert_eq!(
        policy.live_holder_candidates(&url_to_idx, "m", (32, 7), now),
        Vec::<usize>::new()
    );
}

#[test]
fn healthy_url_index_preserves_duplicate_equal_affinity_candidates() {
    let workers = make_workers(&["http://w1:8000", "http://w1:8000", "http://w2:8000"]);
    workers[0].increment_load();

    let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[0, 1, 2]);
    assert_eq!(url_to_idx.len(), 2);
    assert_eq!(url_to_idx["http://w1:8000"], vec![0, 1]);
    assert_eq!(url_to_idx["http://w2:8000"], vec![2]);

    // Equal loads stay tied so LeastLoad can resolve them atomically.
    let tied = make_workers(&["http://w1:8000", "http://w1:8000"]);
    assert_eq!(
        CacheAwarePolicy::healthy_url_index(&tied, &[0, 1])["http://w1:8000"],
        vec![0, 1]
    );
}

#[test]
fn concurrent_record_and_sweep_keeps_live_placements() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
    let now = Instant::now();
    let ttl = Duration::from_secs(180);

    std::thread::scope(|s| {
        for t in 0..8u32 {
            let policy = &policy;
            s.spawn(move || {
                for i in 0..50u32 {
                    let base = t * 1000 + i * 16;
                    let tokens: Vec<u32> = (base..base + 16).collect();
                    policy.record_placement("m", None, &tokens, &[16], "http://w1:8000", now);
                    CacheAwarePolicy::sweep_placement_index(&policy.placement_index, ttl, now);
                }
            });
        }
    });

    let placements = policy.placement_index.get("m").unwrap();
    assert_eq!(placements.len(), 400);
    for entry in placements.iter() {
        assert_eq!(entry.value().len(), 1);
        assert_eq!(entry.value()[0].worker_url, "http://w1:8000");
    }
}

#[test]
fn with_config_normalizes_boundaries() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[64, 16, 16, 0]));
    assert_eq!(policy.config.cache_boundaries, vec![16, 64]);
}

// ---- cache namespaces (cache_salt / extra_key / LoRA partitioning) ----

fn salted(salt: &str) -> Option<CacheNamespace> {
    CacheNamespace::derive(&openai_protocol::common::CachePartition {
        cache_salt: Some(salt),
        extra_key: None,
        lora_path: None,
    })
}

fn route_namespaced(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    tokens: &[u32],
    cache_namespace: Option<CacheNamespace>,
) -> usize {
    policy
        .select_worker(
            workers,
            &SelectWorkerInfo {
                tokens: Some(tokens),
                cache_namespace,
                ..Default::default()
            },
        )
        .unwrap()
}

#[test]
fn head_hash_is_unchanged_without_a_namespace_and_keys_like_the_tree_with_one() {
    let head = [5u32, 6, 7, 8];
    assert_eq!(
        hash_token_head(None, &head),
        xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&head))
    );
    let tenant_a = salted("tenant-a").unwrap();
    let tenant_b = salted("tenant-b").unwrap();
    // A namespaced head hashes marker ‖ head (hash mode has no pages).
    let mut keyed = tenant_a.token_marker().to_vec();
    keyed.extend_from_slice(&head);
    assert_eq!(
        hash_token_head(Some(tenant_a), &head),
        xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&keyed))
    );
    assert_ne!(
        hash_token_head(Some(tenant_a), &head),
        hash_token_head(None, &head)
    );
    assert_ne!(
        hash_token_head(Some(tenant_a), &head),
        hash_token_head(Some(tenant_b), &head)
    );
}

/// The trees' keying rule is written once: a namespaced and an
/// unpartitioned request of one prompt get the keys the trees expect,
/// and the marker length that comes with a namespaced key is exactly
/// what a match count read from the tree subtracts.
#[test]
fn a_request_is_keyed_once_for_both_trees_and_the_marker_length_is_what_a_match_subtracts() {
    let block_size = 16;
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        block_size,
        ..Default::default()
    });
    let tenant_a = salted("tenant-a").unwrap();
    // Two whole pages, so the tree matches the key in full.
    let prompt: Vec<u32> = (100..132).collect();
    let text = "shared system prompt";
    let namespaced = SelectWorkerInfo {
        tokens: Some(&prompt),
        request_text: Some(text),
        cache_namespace: Some(tenant_a),
        ..Default::default()
    };
    let plain = SelectWorkerInfo {
        tokens: Some(&prompt),
        request_text: Some(text),
        ..Default::default()
    };

    let (keyed, marker_len) = policy.keyed_tokens(&namespaced, &prompt);
    assert_eq!(marker_len, CacheNamespace::token_marker_len(block_size));
    assert_eq!(
        &keyed[..],
        &tenant_a.prefixed_tokens(&prompt, block_size)[..]
    );
    assert_eq!(&keyed[marker_len..], &prompt[..]);
    let (plain_key, plain_len) = policy.keyed_tokens(&plain, &prompt);
    assert_eq!((&plain_key[..], plain_len), (&prompt[..], 0));

    // What the token tree reads back for the namespaced key is the
    // prompt plus the marker: the match count minus the marker length
    // is the prompt's own match, and the plain key sees none of it.
    let tree = policy.new_token_tree();
    tree.insert_tokens(&keyed, "http://w1:8000");
    let hit = tree.match_and_insert(&keyed, "http://w1:8000");
    assert_eq!(hit.matched_token_count - marker_len, prompt.len());
    assert_eq!(hit.input_token_count - marker_len, prompt.len());
    let miss = tree.match_and_insert(&plain_key, "http://w2:8000");
    assert_eq!(miss.matched_token_count, 0);

    let (keyed_text, text_marker_len) = CacheAwarePolicy::keyed_text(&namespaced, text);
    assert_eq!(text_marker_len, TEXT_MARKER_LEN);
    assert_eq!(&*keyed_text, tenant_a.prefixed_text(text));
    assert_eq!(&keyed_text[text_marker_len..], text);
    let (plain_text, plain_text_len) = CacheAwarePolicy::keyed_text(&plain, text);
    assert_eq!((&*plain_text, plain_text_len), (text, 0));

    let string_tree = Tree::new();
    string_tree.insert_text(&keyed_text, "http://w1:8000");
    let hit = string_tree.match_and_insert(&keyed_text, "http://w1:8000");
    assert_eq!(
        hit.matched_char_count - text_marker_len,
        text.chars().count()
    );
    let miss = string_tree.match_and_insert(&plain_text, "http://w2:8000");
    assert_eq!(miss.matched_char_count, 0);
}

#[test]
fn hash_mode_keys_placements_under_the_cache_namespace() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[4]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    let tokens = [1u32, 2, 3, 4, 5];
    let tenant_a = salted("tenant-a");
    let tenant_b = salted("tenant-b");

    // w1 is busier: tenant A's first request misses onto w2 and is
    // recorded there under its namespace.
    workers[0].increment_load();
    assert_eq!(route_namespaced(&policy, &workers, &tokens, tenant_a), 1);
    // A hash hit keeps tenant A on w2 once w2 is the busier worker...
    workers[1].increment_load();
    workers[1].increment_load();
    assert_eq!(route_namespaced(&policy, &workers, &tokens, tenant_a), 1);
    // ...while tenant B and an unpartitioned request key differently,
    // miss, and take the least-loaded w1.
    assert_eq!(route_namespaced(&policy, &workers, &tokens, tenant_b), 0);
    assert_eq!(route_namespaced(&policy, &workers, &tokens, None), 0);
}

#[test]
fn imbalance_fallback_inserts_under_the_cache_namespace() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    // One full page (the tree is page-aligned at the default block size).
    let prompt: Vec<u32> = (100..116).collect();
    let tenant_a = salted("tenant-a").unwrap();
    let model_id = normalize_model_key(workers[0].model_id());
    // Materialize the model's token tree so the fallback path has one to
    // update.
    let seed: Vec<u32> = (1..17).collect();
    route_namespaced(&policy, &workers, &seed, None);

    let info = SelectWorkerInfo {
        tokens: Some(&prompt),
        cache_namespace: Some(tenant_a),
        ..Default::default()
    };
    policy
        .select_worker_fallback(&workers, &info, &[0, 1], model_id)
        .unwrap();

    let tree = policy.token_trees.get(model_id).unwrap().value().clone();
    // The unpartitioned prompt cannot see the namespaced insert...
    assert!(tree.prefix_match_legacy(&prompt).0.is_empty());
    // ...while the namespaced key matches in full.
    let keyed = tenant_a.prefixed_tokens(&prompt, policy.config.block_size);
    assert_eq!(tree.prefix_match_legacy(&keyed).0.len(), keyed.len());
}

#[test]
fn an_unpartitioned_request_cannot_forge_its_way_into_a_namespace() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        cache_threshold: 0.5,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    let tenant_a = salted("tenant-a").unwrap();
    let prompt: Vec<u32> = (100..132).collect();

    // Tenant A's prompt lands on w2 (w1 is busier).
    workers[0].increment_load();
    assert_eq!(
        route_namespaced(&policy, &workers, &prompt, Some(tenant_a)),
        1
    );
    workers[1].increment_load();
    workers[1].increment_load();

    // An unpartitioned request whose token ids spell tenant A's marker
    // must not hit tenant A's entry: it misses and takes the least-loaded
    // w1. Same for text that spells the marker on the string tree.
    let forged = tenant_a.prefixed_tokens(&prompt, policy.config.block_size);
    assert_eq!(route_namespaced(&policy, &workers, &forged, None), 0);

    let text_policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        cache_threshold: 0.5,
        ..Default::default()
    });
    let text_workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    text_policy.init_workers(&text_workers);
    text_workers[0].increment_load();
    let text_info = SelectWorkerInfo {
        request_text: Some("shared system prompt"),
        cache_namespace: Some(tenant_a),
        ..Default::default()
    };
    assert_eq!(
        text_policy.select_worker(&text_workers, &text_info),
        Some(1)
    );
    text_workers[1].increment_load();
    text_workers[1].increment_load();
    let forged_text = tenant_a.prefixed_text("shared system prompt");
    let forged_info = SelectWorkerInfo {
        request_text: Some(&forged_text),
        ..Default::default()
    };
    assert_eq!(
        text_policy.select_worker(&text_workers, &forged_info),
        Some(0)
    );
}

#[test]
fn hash_mode_strips_marker_material_from_unpartitioned_heads() {
    let policy = CacheAwarePolicy::with_config(hash_config(&[4]));
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    let tokens = [1u32, 2, 3, 4, 5];
    let tenant_a = salted("tenant-a").unwrap();

    // Tenant A's placement lands on w2 (w1 is busier).
    workers[0].increment_load();
    assert_eq!(
        route_namespaced(&policy, &workers, &tokens, Some(tenant_a)),
        1
    );
    workers[1].increment_load();
    workers[1].increment_load();

    // An unpartitioned head that spells tenant A's marker keys as the
    // bare prompt: no holder, so it takes the least-loaded w1.
    let mut forged = tenant_a.token_marker().to_vec();
    forged.extend_from_slice(&tokens);
    assert_eq!(route_namespaced(&policy, &workers, &forged, None), 0);
}

// -- selection policy layer --

/// Random positive-overlap candidate sets, as `overlap_candidates` would
/// produce them (healthy order, scores from a small set so ties occur).
fn random_overlap_candidates(rng: &mut impl RngExt, workers: usize) -> Vec<OverlapCandidate> {
    let mut candidates = Vec::new();
    for idx in 0..workers {
        if rng.random::<f64>() >= 0.6 {
            continue;
        }
        let score = f64::from(rng.random_range(1..6u32)) * 1.5;
        let decayed = if rng.random::<f64>() < 0.3 {
            score / 1.5
        } else {
            score
        };
        candidates.push(OverlapCandidate {
            idx,
            raw_score: score,
            effective_score: decayed,
        });
    }
    candidates
}

fn policy_inputs<'a>(
    candidates: &'a [OverlapCandidate],
    urls: &'a [String],
) -> Vec<CandidateInputs<'a>> {
    candidates
        .iter()
        .map(|candidate| CandidateInputs {
            idx: candidate.idx,
            url: &urls[candidate.idx],
            device_blocks: candidate.raw_score,
            effective_score: candidate.effective_score,
        })
        .collect()
}

#[test]
fn default_policy_reproduces_affinity_score_group_at_zero_temperature() {
    let policy = cost::build(cost::DEFAULT_POLICY, 0.0).unwrap();
    let urls: Vec<String> = (0..16).map(|i| format!("http://w{i:02}:8000")).collect();
    let request = RequestInputs {
        prompt_tokens: 64,
        block_size: 4,
        request_blocks: 16,
        avg_load: 0.0,
        prefix_hashes: None,
    };
    let mut rng = rand::rng();
    for _ in 0..2_000 {
        let candidates = random_overlap_candidates(&mut rng, urls.len());
        let mut reference = CacheAwarePolicy::affinity_score_group(&candidates, 0.0);
        reference.sort_unstable();
        let inputs = policy_inputs(&candidates, &urls);
        let mut group = match policy.select(&request, &inputs) {
            Pick::Group(rows) => rows.iter().map(|&row| inputs[row].idx).collect::<Vec<_>>(),
            Pick::None => Vec::new(),
            Pick::Final(_) => panic!("the default policy never returns a single pick"),
        };
        group.sort_unstable();
        assert_eq!(group, reference);
    }
}

#[test]
fn default_policy_temperature_groups_are_score_groups() {
    let policy = cost::build(cost::DEFAULT_POLICY, 0.5).unwrap();
    let urls: Vec<String> = (0..8).map(|i| format!("http://w{i}:8000")).collect();
    let candidates: Vec<OverlapCandidate> = (0..8)
        .map(|idx| OverlapCandidate {
            idx,
            raw_score: if idx < 4 { 8.0 } else { 2.0 },
            effective_score: if idx < 4 { 8.0 } else { 2.0 },
        })
        .collect();
    let inputs = policy_inputs(&candidates, &urls);
    let request = RequestInputs {
        prompt_tokens: 32,
        block_size: 4,
        request_blocks: 8,
        avg_load: 0.0,
        prefix_hashes: None,
    };
    let mut saw_high = false;
    let mut saw_low = false;
    for _ in 0..400 {
        let Pick::Group(rows) = policy.select(&request, &inputs) else {
            panic!("expected a group");
        };
        let score = inputs[rows[0]].effective_score;
        assert!(rows.iter().all(|&row| inputs[row].effective_score == score));
        assert_eq!(rows.len(), 4, "a score group is every equal-score worker");
        if score == 8.0 {
            saw_high = true;
        } else {
            saw_low = true;
        }
    }
    assert!(
        saw_high && saw_low,
        "temperature must reach both score groups"
    );
}

#[test]
fn every_catalog_policy_routes_event_driven_hits_and_misses() {
    for name in cost::POLICY_NAMES {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            selection_policy: Some((*name).to_string()),
            ..test_config()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let monitor = Arc::new(KvEventMonitor::new());
        let indexer = setup_indexer_with_blocks("http://w2:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let hit = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                ..Default::default()
            },
        );
        assert_eq!(hit, Some(1), "{name}: the holder of every block must win");
        policy.on_request_complete("http://w2:8000", true);

        let miss = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&[100, 200, 300, 400]),
                ..Default::default()
            },
        );
        assert!(miss.is_some(), "{name}: a miss must still route");
    }
}

#[test]
fn every_catalog_policy_routes_tree_matches() {
    for name in cost::POLICY_NAMES {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            selection_policy: Some((*name).to_string()),
            ..test_config()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let text = "a shared system prompt that is long enough to match";
        let first = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
        policy.on_request_complete(workers[first].url(), true);
        let second = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(first, second, "{name}: an idle holder keeps its prefix");
    }
}

#[test]
fn accounting_books_the_dispatch_until_completion() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        selection_accounting_ttl_ms: 60_000,
        ..test_config()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    policy.init_workers(&workers);
    let monitor = Arc::new(KvEventMonitor::new());
    let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]]);
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));
    let accounting = policy.accounting.as_ref().expect("accounting enabled");

    let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let selected = policy
        .select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected, 0);
    // One block cached, one block (4 tokens) booked as uncached prefill.
    assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 4);
    let hashes: Vec<u64> = request_prefix_hashes(&compute_request_content_hashes(&tokens, 4))
        .into_iter()
        .map(|hash| hash.0)
        .collect();
    let predicted = accounting.predicted_overlaps(&hashes);
    assert_eq!(predicted.len(), 1);
    assert_eq!(&*predicted[0].0, "http://w1:8000");
    assert_eq!(
        predicted[0].1, 2.0,
        "the whole two-block prefix is predicted resident"
    );

    // A sibling that shares the first block but not the second is predicted one block on w1.
    let sibling: Vec<u64> = request_prefix_hashes(&compute_request_content_hashes(
        &[1, 2, 3, 4, 9, 9, 9, 9],
        4,
    ))
    .into_iter()
    .map(|hash| hash.0)
    .collect();
    assert_eq!(accounting.predicted_overlaps(&sibling)[0].1, 1.0);

    policy.on_request_complete("http://w1:8000", true);
    assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 0);
}

#[test]
fn reconciliation_releases_bookings_whose_completion_never_arrived() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        selection_accounting_ttl_ms: 60_000,
        ..test_config()
    });
    let accounting = policy.accounting.as_ref().expect("accounting enabled");
    accounting.record_dispatch("http://w1:8000", 100, &[]);
    accounting.record_dispatch("http://w1:8000", 200, &[]);
    accounting.record_dispatch("http://w1:8000", 400, &[]);

    // The router still holds all three: nothing to release.
    policy.reconcile_in_flight("http://w1:8000", 3);
    assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 700);

    // Two requests ended without a completion report: the two oldest
    // bookings go, the one the router still holds stays.
    policy.reconcile_in_flight("http://w1:8000", 1);
    assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 400);

    // A worker without bookings is a no-op, with or without accounting.
    policy.reconcile_in_flight("http://w2:8000", 0);
    CacheAwarePolicy::with_config(test_config()).reconcile_in_flight("http://w1:8000", 0);
}

#[test]
fn invalid_selection_policy_falls_back_to_the_default() {
    let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
        selection_policy: Some("no-such-policy".to_string()),
        ..test_config()
    });
    assert_eq!(policy.selection.name(), cost::DEFAULT_POLICY);
}

#[test]
fn event_driven_salted_request_matches_only_its_namespace() {
    use kv_index::salt::{content_hash_with_seed, namespace_seed};
    use openai_protocol::common::CachePartition;

    let policy = CacheAwarePolicy::with_config(test_config());
    let w1 = BasicWorkerBuilder::new("http://w1:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    let w2 = BasicWorkerBuilder::new("http://w2:8000")
        .worker_type(WorkerType::Regular)
        .health_config(no_health_check())
        .build();
    // w1 carries more live load, so every miss resolves to w2.
    for _ in 0..3 {
        w1.increment_load();
    }
    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
    policy.init_workers(&workers);

    // Blocks stored on w1 under (lora "adapter", salt "tenant-a"), as the
    // monitor hashes a salted KvBlocksStored event.
    let indexer = Arc::new(KvIndex::chain());
    let worker_id = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerBlocks::default();
    let seed = namespace_seed(Some("adapter"), Some("tenant-a"));
    let blocks: Vec<StoredBlock> = [[1u32, 2, 3, 4], [5, 6, 7, 8]]
        .iter()
        .enumerate()
        .map(|(i, tokens)| StoredBlock {
            seq_hash: SequenceHash(i as u64 + 1),
            content_hash: content_hash_with_seed(tokens, seed),
        })
        .collect();
    indexer
        .apply_stored(worker_id, &blocks, None, &mut wb)
        .unwrap();
    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert("unknown".to_string(), indexer);
    policy.set_kv_event_monitor(Some(monitor));

    let namespace = |salt: &'static str| {
        CacheNamespace::derive(&CachePartition {
            cache_salt: Some(salt),
            extra_key: None,
            lora_path: Some("adapter"),
        })
    };
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let route = |cache_namespace: Option<CacheNamespace>| {
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    cache_namespace,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    assert_eq!(route(namespace("tenant-a")), 0, "same namespace hits w1");
    assert_eq!(route(namespace("tenant-b")), 1, "another salt misses");
    assert_eq!(route(None), 1, "a plain request misses salted blocks");
}

// -----------------------------------------------------------------------
// Pool tables and the sampled decision over a large pool
// -----------------------------------------------------------------------

/// `count` workers, all interned in an event index that holds `chunks`
/// for the worker at `holder`, behind a policy built from `config`.
fn large_pool(
    count: usize,
    holder: usize,
    chunks: &[&[u32]],
    config: CacheAwareConfig,
) -> (CacheAwarePolicy, Vec<Arc<dyn Worker>>) {
    let urls: Vec<String> = (0..count).map(|i| format!("http://w{i}:8000")).collect();
    let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
    let workers = make_workers(&refs);
    let indexer = setup_indexer_with_blocks(&urls[holder], chunks);
    for (i, url) in urls.iter().enumerate() {
        if i != holder {
            indexer.intern_worker(url).unwrap();
        }
    }
    let policy = CacheAwarePolicy::with_config(config);
    policy.init_workers(&workers);
    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert("unknown".to_string(), indexer);
    monitor.set_block_size("unknown", 4);
    policy.set_kv_event_monitor(Some(monitor));
    (policy, workers)
}

fn tokens_info(tokens: &[u32]) -> SelectWorkerInfo<'_> {
    SelectWorkerInfo {
        tokens: Some(tokens),
        ..Default::default()
    }
}

#[test]
fn a_large_pool_routes_to_the_holder_the_index_names() {
    let (policy, workers) = large_pool(40, 37, &[&[1, 2, 3, 4], &[5, 6, 7, 8]], test_config());
    for _ in 0..8 {
        let idx = policy
            .select_worker(&workers, &tokens_info(&[1, 2, 3, 4, 5, 6, 7, 8]))
            .unwrap();
        assert_eq!(
            idx, 37,
            "the holder sits outside any sample of {FLEET_SAMPLE}; the index places it"
        );
    }
}

#[test]
fn a_large_pool_miss_finds_the_one_eligible_worker_the_sample_missed() {
    let (policy, workers) = large_pool(40, 0, &[&[1, 2, 3, 4]], test_config());
    for worker in &workers[..39] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let idx = policy.select_worker(&workers, &tokens_info(&[9, 9, 9, 9, 8, 8, 8, 8]));
    assert_eq!(idx, Some(39));
}

#[test]
fn a_large_pool_miss_spreads_over_the_pool() {
    let (policy, workers) = large_pool(40, 0, &[&[1, 2, 3, 4]], test_config());
    let mut picked = HashSet::new();
    for turn in 0..200u32 {
        let tokens = [turn + 100, 9, 9, 9, 8, 8, 8, 8];
        let idx = policy
            .select_worker(&workers, &tokens_info(&tokens))
            .unwrap();
        picked.insert(idx);
        policy.on_request_complete(workers[idx].url(), true);
    }
    assert!(
        picked.len() > 2 * FLEET_SAMPLE,
        "200 misses reached {} workers; a fresh sample per decision reaches the pool",
        picked.len()
    );
}

#[test]
fn pool_table_places_ids_and_rejects_a_worker_swapped_in_at_the_position() {
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let indexer = setup_indexer_with_blocks("http://w3:8000", &[&[1, 2, 3, 4]]);
    let w3 = indexer.worker_id("http://w3:8000").unwrap();
    let table = PoolTable::build(&workers, &indexer, liveness::now_ms());
    assert!(table.describes(&workers));
    assert!(table.fresh(liveness::now_ms()));
    assert_eq!(table.position(w3, &workers), Some(2));
    assert_eq!(table.id_at, vec![None, None, Some(w3)]);
    assert_eq!(
        table.position(w3 + 100, &workers),
        None,
        "an id never interned"
    );
    // The same url behind a new worker object at the same position: the
    // table was built from the old one and says so.
    let mut swapped = workers.clone();
    swapped[2] = make_workers(&["http://w3:8000"]).remove(0);
    assert_eq!(table.position(w3, &swapped), None);
    // Every worker was admitted just now: a young fleet lists nothing to slice.
    assert!(table.warming.is_empty());
    assert!(!table.fresh(liveness::now_ms() + POOL_TABLE_REFRESH_MS));
}

/// `count` workers that all hold the shared block `[1, 2, 3, 4]` (a chat
/// template's head), the one at `deep` also `[5, 6, 7, 8]`.
fn shared_prefix_pool(
    count: usize,
    deep: usize,
    config: CacheAwareConfig,
) -> (CacheAwarePolicy, Vec<Arc<dyn Worker>>) {
    let urls: Vec<String> = (0..count).map(|i| format!("http://w{i}:8000")).collect();
    let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
    let workers = make_workers(&refs);
    let indexer = Arc::new(KvIndex::chain());
    for (i, url) in urls.iter().enumerate() {
        let id = indexer.intern_worker(url).unwrap();
        let mut wb = WorkerBlocks::default();
        let mut blocks = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        if i == deep {
            blocks.push(StoredBlock {
                seq_hash: SequenceHash(2),
                content_hash: compute_content_hash(&[5, 6, 7, 8]),
            });
        }
        indexer.apply_stored(id, &blocks, None, &mut wb).unwrap();
    }
    let policy = CacheAwarePolicy::with_config(config);
    policy.init_workers(&workers);
    let monitor = Arc::new(KvEventMonitor::new());
    monitor.indexers.insert("unknown".to_string(), indexer);
    monitor.set_block_size("unknown", 4);
    policy.set_kv_event_monitor(Some(monitor));
    (policy, workers)
}

#[test]
fn the_default_decision_takes_the_deepest_holder_among_a_fleet_sharing_the_head() {
    let (policy, workers) = shared_prefix_pool(64, 50, test_config());
    for _ in 0..4 {
        let idx = policy
            .select_worker(&workers, &tokens_info(&[1, 2, 3, 4, 5, 6, 7, 8]))
            .unwrap();
        assert_eq!(idx, 50);
    }
    // The deepest holder down: the decision falls to the holders of the
    // head, never to a worker without the blocks.
    workers[50].set_status(WorkerStatus::NotReady);
    let idx = policy
        .select_worker(&workers, &tokens_info(&[1, 2, 3, 4, 5, 6, 7, 8]))
        .unwrap();
    assert_ne!(idx, 50);
    assert!(workers[idx].is_healthy());
}

#[test]
fn a_fleet_wide_tie_is_drawn_from_not_scored_whole() {
    let (policy, workers) = shared_prefix_pool(64, 50, test_config());
    let mut picked = HashSet::new();
    for _ in 0..200 {
        let idx = policy
            .select_worker(&workers, &tokens_info(&[1, 2, 3, 4]))
            .unwrap();
        picked.insert(idx);
        policy.on_request_complete(workers[idx].url(), true);
    }
    assert!(
        picked.len() > 2 * FLEET_SAMPLE,
        "200 decisions on a head every worker holds reached {} workers",
        picked.len()
    );
}

#[test]
fn merge_rows_unions_the_eligible_sample_with_the_holders_in_slice_order() {
    let candidates: Vec<OverlapCandidate> = [3usize, 7, 20]
        .iter()
        .map(|&idx| OverlapCandidate {
            idx,
            raw_score: 1.0,
            effective_score: 1.0,
        })
        .collect();
    assert_eq!(
        CacheAwarePolicy::merge_rows(&[1, 3, 9, 30], &candidates),
        vec![1, 3, 7, 9, 20, 30]
    );
    assert_eq!(
        CacheAwarePolicy::merge_rows(&[], &candidates),
        vec![3, 7, 20]
    );
    assert_eq!(CacheAwarePolicy::merge_rows(&[2, 4], &[]), vec![2, 4]);
}
