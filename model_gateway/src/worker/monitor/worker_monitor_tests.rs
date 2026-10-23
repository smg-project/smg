use std::collections::HashMap;

use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus},
};

use super::*;
use crate::{
    config::types::PolicyConfig,
    observability::metrics::test_support::render_with_recorder,
    policies::PolicyRegistry,
    worker::{overload::OverloadThresholds, BasicWorkerBuilder, ConnectionMode, WorkerType},
};

fn ready_worker(url: &str, model: &str) -> Arc<dyn Worker> {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .model(ModelCard::new(model))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// Build an isolated monitor and registry with round-robin routing for tests.
fn build_monitor() -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
    let registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let monitor = Arc::new(WorkerMonitor::new(
        registry.clone(),
        policy_registry,
        5,
        false,
        false,
    ));
    (registry, monitor)
}

/// Backend-wide pushes only refresh the matching virtual rank; quiet ranks keep polling.
#[tokio::test]
async fn dp8_pushes_cannot_redirect_virtual_workers_to_active_ranks() {
    let (registry, monitor) = build_monitor();
    let now = Instant::now();
    let interval = Duration::from_secs(10);
    let active = EngineLoad {
        num_used_tokens: Some(2048),
        max_total_num_tokens: Some(8192),
        max_running_requests: 64,
        ..pushed(7)
    };
    for rank in 0..8 {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:9000")
                .worker_type(WorkerType::Decode)
                .dp_config(rank, 8)
                .build(),
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        // Every subscription receives the same backend-wide stream.
        for pushing_rank in [0, 2] {
            monitor.apply_pushed_load(&worker, pushing_rank, &active, now);
        }
        let is_active = rank == 0 || rank == 2;
        assert_eq!(worker.pushes_load_records(), is_active);
        assert_eq!(
            monitor.poll_mode(worker.url(), now, interval),
            if is_active {
                PollMode::Suppressed
            } else {
                PollMode::Poll
            }
        );
        monitor.flush_pushed();
        let snapshot = monitor.load_state.snapshot();
        if is_active {
            let report = snapshot.get(worker.url()).unwrap();
            assert_eq!(report.loads.len(), 1);
            assert_eq!(report.loads[0].dp_rank, rank as i32);
        } else {
            assert!(snapshot.get(worker.url()).is_none());
        }

        // A poll supplies each quiet rank's real idle sample. A later
        // foreign push must neither overwrite it nor redirect selection.
        let loads = HashMap::from([(
            worker.url().to_string(),
            HashMap::from([(rank as isize, 0)]),
        )]);
        monitor.worker_load_manager.update_dp_loads(&loads);
        for pushing_rank in [0, 2] {
            monitor.apply_pushed_load(&worker, pushing_rank, &active, now);
        }
        let manager = &monitor.worker_load_manager;
        assert_eq!(
            manager.select_and_increment_lowest_dp_load(worker.as_ref(), 1),
            Some(rank as isize)
        );
    }
}

/// Foreign cached ranks cannot redirect a pinned worker or invent missing telemetry.
#[test]
fn virtual_rank_selection_rejects_foreign_cached_ranks() {
    let worker = BasicWorkerBuilder::new("grpc://decode:9000")
        .dp_config(7, 8)
        .build();
    let manager = Arc::new(WorkerLoadManager::new());
    let mut loads = HashMap::from([(0, 0), (2, 0)]);
    manager.update_dp_loads(&HashMap::from([(worker.url().to_string(), loads.clone())]));
    assert!(manager
        .select_and_increment_lowest_dp_load(&worker, 1)
        .is_none());
    loads.insert(7, 100);
    manager.update_dp_loads(&HashMap::from([(worker.url().to_string(), loads)]));
    assert_eq!(
        manager.select_and_increment_lowest_dp_load(&worker, 1),
        Some(7)
    );
}

/// A pushed record is a poll of its worker: the rank's entry is replaced
/// (other ranks kept), `sampled_at` is the receipt less the record's age
/// and the one-way margin, and the shared snapshot sees it at the next
/// coalesced publish.
#[tokio::test]
async fn a_pushed_load_record_becomes_the_workers_report_at_the_next_publish() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let received = Instant::now();
    let record = EngineLoad {
        running_requests: 7,
        waiting_requests: 2,
        waiting_uncached_tokens: Some(3_000),
        token_usage: 0.4,
        gen_throughput: 900.0,
        max_running_requests: 64,
        age_ms: 30,
        sample: 1,
        load_only: false,
        ..Default::default()
    };
    monitor.apply_pushed_load(&worker, 0, &record, received);
    assert_eq!(monitor.pending_pushed_len(), 1);
    assert!(
        monitor
            .load_state
            .snapshot()
            .get("grpc://w1:9000")
            .is_none(),
        "not published until the window closes"
    );
    monitor.flush_pushed();
    let snapshot = monitor.load_state.snapshot();
    let report = snapshot.get("grpc://w1:9000").expect("published");
    assert_eq!(report.version, "pushed");
    assert_eq!(report.loads.len(), 1);
    let rank0 = &report.loads[0];
    assert_eq!(
        (
            rank0.dp_rank,
            rank0.num_running_reqs,
            rank0.num_waiting_reqs,
            rank0.num_waiting_uncached_tokens,
            rank0.num_total_reqs,
            rank0.max_running_requests
        ),
        (0, 7, 2, 3_000, 9, 64)
    );
    assert!((rank0.token_usage - 0.4).abs() < f64::EPSILON);
    assert!((rank0.gen_throughput - 900.0).abs() < f64::EPSILON);
    let sampled_at = report.sampled_at.expect("sampled_at");
    assert_eq!(
        received.duration_since(sampled_at),
        Duration::from_millis(30) + PUSHED_ONE_WAY_MARGIN
    );
    assert_eq!(monitor.pending_pushed_len(), 0);

    // A second rank joins the report; a later record replaces only its rank.
    monitor.apply_pushed_load(
        &worker,
        1,
        &EngineLoad {
            running_requests: 1,
            waiting_uncached_tokens: None,
            ..record.clone()
        },
        Instant::now(),
    );
    monitor.apply_pushed_load(
        &worker,
        0,
        &EngineLoad {
            running_requests: 8,
            ..record.clone()
        },
        Instant::now(),
    );
    monitor.flush_pushed();
    let snapshot = monitor.load_state.snapshot();
    let report = snapshot.get("grpc://w1:9000").expect("published");
    let ranks: Vec<(i32, i32, i32)> = report
        .loads
        .iter()
        .map(|rank| {
            (
                rank.dp_rank,
                rank.num_running_reqs,
                rank.num_waiting_uncached_tokens,
            )
        })
        .collect();
    assert_eq!(ranks, vec![(0, 8, 3_000), (1, 1, 0)]);
    assert_eq!(report.dp_rank_count, 2);
}

/// A record for a worker the registry no longer holds as this incarnation
/// is dropped at publish by the fence, like a late poll.
#[tokio::test]
async fn a_pushed_record_from_a_replaced_worker_is_fenced_at_publish() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    registry.remove_by_url("grpc://w1:9000");
    monitor.apply_pushed_load(&worker, 0, &EngineLoad::default(), Instant::now());
    monitor.flush_pushed();
    assert!(monitor
        .load_state
        .snapshot()
        .get("grpc://w1:9000")
        .is_none());
}

#[tokio::test]
async fn bootstrap_reconcile_starts_loops_for_existing_workers() {
    let (registry, monitor) = build_monitor();
    registry
        .register(ready_worker("http://w1:8080", "llama-3"))
        .unwrap();
    registry
        .register(ready_worker("http://w2:8080", "llama-3"))
        .unwrap();
    registry
        .register(ready_worker("http://w3:8080", "gpt-4"))
        .unwrap();

    monitor.start_event_loop();

    // Two model groups should now have polling loops.
    let handles = monitor.group_handles.lock();
    assert_eq!(handles.len(), 2);
    let keys: Vec<&WorkerGroupKey> = handles.keys().collect();
    assert!(keys.iter().any(|k| k.model_id == "llama-3"));
    assert!(keys.iter().any(|k| k.model_id == "gpt-4"));
}

#[tokio::test]
async fn registered_event_starts_a_new_group() {
    let (registry, monitor) = build_monitor();
    monitor.start_event_loop();

    // Registry was empty at bootstrap, so no groups yet.
    assert!(monitor.group_handles.lock().is_empty());

    registry
        .register(ready_worker("http://w:8080", "llama-3"))
        .unwrap();

    // Give the event loop a moment to process the broadcast.
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let handles = monitor.group_handles.lock();
    assert_eq!(handles.len(), 1);
    assert!(handles
        .keys()
        .any(|k| k.model_id == "llama-3" && k.worker_type == WorkerType::Regular));
}

#[tokio::test]
async fn removed_event_stops_empty_group() {
    let (registry, monitor) = build_monitor();
    let id = registry
        .register(ready_worker("http://w:8080", "llama-3"))
        .unwrap();
    monitor.start_event_loop();
    assert_eq!(monitor.group_handles.lock().len(), 1);

    registry.remove(&id);
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert!(monitor.group_handles.lock().is_empty());
}

#[tokio::test]
async fn removed_event_evicts_cached_loads() {
    // Regression test for the bug where a removed worker's last
    // known load would persist forever in `load_tx` and the DP
    // cache because the polling loop that pruned it had been
    // stopped by the same removal.
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("http://w:8080", "llama-3");
    let url = worker.url().to_string();
    let id = registry.register(Arc::clone(&worker)).unwrap();
    monitor.start_event_loop();

    monitor.load_state.publish_group(
        &[],
        vec![(Arc::clone(&worker), Arc::new(WorkerLoadResponse::default()))],
    );
    assert!(monitor.load_state.snapshot().contains(&url));
    monitor
        .native_loads_memo
        .insert(url.clone(), NativeLoadsMemo::default());
    let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
    let mut inner = HashMap::new();
    inner.insert(0, 5);
    dp_loads.insert(url.clone(), inner);
    monitor.worker_load_manager.update_dp_loads(&dp_loads);

    registry.remove(&id);
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    // The event loop has queued the eviction; apply it deterministically
    // instead of racing the debounced flusher.
    monitor.flush_pending_evictions();

    let snapshot = monitor.load_state.snapshot();
    assert!(
        !snapshot.contains(&url),
        "the load snapshot must not retain entries for removed workers"
    );
    let cached = monitor.worker_load_manager.dp_cached_loads.read();
    assert!(
        !cached.contains_key(&url),
        "DP cache must not retain entries for removed workers"
    );
    assert!(
        !monitor.native_loads_memo.contains_key(&url),
        "probe memo must not retain entries for removed workers"
    );
}

#[tokio::test]
async fn stop_all_groups_clears_native_probe_memo() {
    // A worker removed during a `RecvError::Lagged` window never
    // fires an eviction, so the reset path must drop the memo too.
    let (_registry, monitor) = build_monitor();
    monitor
        .native_loads_memo
        .insert("http://w1:8080".to_string(), NativeLoadsMemo::default());

    monitor.stop_all_groups();

    assert!(monitor.native_loads_memo.is_empty());
}

#[tokio::test]
async fn stop_all_groups_clears_dp_cache() {
    // Regression test for the bug where a `RecvError::Lagged`
    // rebuild would leave stale DP cache entries because
    // stop_all_groups only cleared the watch snapshot.
    let (registry, monitor) = build_monitor();
    registry
        .register(ready_worker("http://w1:8080", "llama-3"))
        .unwrap();

    let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
    let mut inner = HashMap::new();
    inner.insert(0, 7);
    dp_loads.insert("http://w1:8080".to_string(), inner);
    monitor.worker_load_manager.update_dp_loads(&dp_loads);

    monitor.stop_all_groups();

    assert!(monitor.load_state.snapshot().is_empty());
    assert!(monitor
        .worker_load_manager
        .dp_cached_loads
        .read()
        .is_empty());
}

#[tokio::test]
async fn status_changed_away_from_ready_evicts_worker() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("http://w:8080", "llama-3");
    let url = worker.url().to_string();
    let id = registry.register(Arc::clone(&worker)).unwrap();
    monitor.start_event_loop();

    // Seed the load snapshot + DP cache as if a poll had succeeded.
    monitor.load_state.publish_group(
        &[],
        vec![(Arc::clone(&worker), Arc::new(WorkerLoadResponse::default()))],
    );
    assert!(monitor.load_state.snapshot().contains(&url));
    let mut dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
    let mut inner = HashMap::new();
    inner.insert(0, 5);
    dp_loads.insert(url.clone(), inner);
    monitor.worker_load_manager.update_dp_loads(&dp_loads);

    registry.transition_status(&id, WorkerStatus::NotReady);
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    // The event loop has queued the eviction; apply it deterministically
    // instead of racing the debounced flusher.
    monitor.flush_pending_evictions();

    // Load snapshot entry pruned.
    let snapshot = monitor.load_state.snapshot();
    assert!(!snapshot.contains(&url));

    // DP cache entry pruned.
    let cached = monitor.worker_load_manager.dp_cached_loads.read();
    assert!(!cached.contains_key(&url));
}

#[tokio::test]
async fn spawned_tasks_use_weak_references() {
    // Regression test for the Arc cycle bug: the spawned event
    // loop and the per-group polling loops must hold
    // `Weak<WorkerMonitor>` rather than `Arc<WorkerMonitor>`. If
    // they held strong references, the cycle through
    // `event_task` / `group_handles` would keep the strong count
    // pinned and `Drop` would never fire outside of full runtime
    // shutdown.
    let (registry, monitor) = build_monitor();
    registry
        .register(ready_worker("http://w:8080", "llama-3"))
        .unwrap();
    monitor.start_event_loop();

    // Bootstrap should have started a polling loop for the
    // worker's group, on top of the event task itself.
    assert_eq!(monitor.group_handles.lock().len(), 1);

    // After spawning, only the test's `Arc` should be strong:
    // both the event task and the group loop downgraded to
    // `Weak<Self>`.
    assert_eq!(
        Arc::strong_count(&monitor),
        1,
        "spawned tasks must not hold strong references to the monitor"
    );
    assert!(
        Arc::weak_count(&monitor) >= 2,
        "expected at least one Weak from the event task and one from the group loop"
    );
}

fn pushed(running: u32) -> EngineLoad {
    EngineLoad {
        running_requests: running,
        waiting_requests: 1,
        token_usage: 0.2,
        gen_throughput: 10.0,
        max_running_requests: 32,
        sample: 1,
        ..Default::default()
    }
}

/// A pushed record that carries the engine's own silence past the wedge
/// threshold vetoes the worker with nothing of this gateway's in flight
/// on it; the record that shows the engine producing again lifts it.
#[tokio::test]
async fn a_pushed_record_with_the_engines_silence_vetoes_the_worker() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let mut record = pushed(1);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(worker.stall_reason().is_none(), "no silence reported");
    record.engine_silence_ms = Some(5_000);
    record.engine_reports_steps = Some(true);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(
        worker.stall_reason(),
        Some(super::super::worker::StallReason::Wedged)
    );
    assert!(!worker.is_healthy_and_eligible());
    record.engine_silence_ms = None;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(worker.stall_reason().is_none(), "the engine produces again");
}

/// A pushed record from a step-reporting wire whose silence is past the
/// threshold while the servicer reports a long prefill in hand (the step
/// that starts a long multimodal prompt, or a kernel built for it) is no
/// veto until twice that prefill's time has passed; the same silence
/// with nothing pending, a frozen engine, is the veto it was.
#[tokio::test]
async fn a_pushed_record_with_a_long_pending_prefill_behind_the_engines_silence_is_no_veto() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let mut record = pushed(1);
    record.engine_reports_steps = Some(true);
    record.prefill_pending_tokens = Some(156_752);
    for silence in [3_000, 10_000, 31_000] {
        record.engine_silence_ms = Some(silence);
        monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
        assert!(
            worker.stall_reason().is_none(),
            "{silence} ms of silence inside the prefill's allowance"
        );
    }
    record.engine_silence_ms = Some(32_000);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(
        worker.stall_reason(),
        Some(super::super::worker::StallReason::Wedged),
        "silent past twice the prefill at the cold prior"
    );
    record.engine_silence_ms = None;
    record.prefill_pending_tokens = Some(0);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(worker.stall_reason().is_none(), "the engine produces again");
    // Nothing pending: the threshold, as before.
    record.engine_silence_ms = Some(3_000);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(
        worker.stall_reason(),
        Some(super::super::worker::StallReason::Wedged),
        "a frozen engine with nothing to prefill"
    );
}

/// The overload veto's edges are INFO lines with the report's figures:
/// a record at the waiting threshold sets it, the next under it clears
/// it, and a record that re-asserts the verdict the worker carries
/// writes nothing new.
#[tokio::test]
#[tracing_test::traced_test]
async fn the_overload_vetos_edges_are_logged_with_their_figures() {
    let (registry, monitor) = build_monitor();
    let worker: Arc<dyn Worker> = {
        let worker = BasicWorkerBuilder::new("grpc://w1:9000")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .model(ModelCard::new("llama-3"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .overload_defaults(OverloadThresholds {
                waiting_requests: Some(8),
                token_usage: Some(0.8),
            })
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    };
    registry.register(Arc::clone(&worker)).unwrap();
    let mut record = pushed(1);
    record.waiting_requests = 8;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(worker.is_overloaded(), "eight waiting is the threshold");
    assert!(logs_contain("Worker overload veto set"));
    assert!(logs_contain("waiting_requests=8"));
    assert!(logs_contain("pushed record"));
    assert!(!logs_contain("Worker overload veto cleared"));
    record.sample = 2;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    logs_assert(|lines: &[&str]| {
        let edges = lines
            .iter()
            .filter(|line| line.contains("Worker overload veto"))
            .count();
        if edges == 1 {
            Ok(())
        } else {
            Err(format!("{edges} edge lines for one edge"))
        }
    });
    record.waiting_requests = 0;
    record.sample = 3;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(!worker.is_overloaded(), "the queue drained");
    assert!(logs_contain("Worker overload veto cleared"));
}

/// A worker's first pushed record may be a `load_only` one the servicer
/// sends before the engine's first batch (from the subscription on, to
/// a gateway that asks for them): with the engine's silence on it, it
/// vetoes the worker like any later record, so an engine frozen on the
/// first request it ever receives leaves routing within the threshold
/// rather than never.
#[tokio::test]
async fn the_first_record_a_worker_pushes_is_judged_before_any_event_batch() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let record = EngineLoad {
        engine_silence_ms: Some(5_000),
        engine_reports_steps: Some(true),
        prefill_pending_tokens: Some(47),
        load_only: true,
        sample: 1,
        ..Default::default()
    };
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(
        worker.stall_reason(),
        Some(super::super::worker::StallReason::Wedged)
    );
    assert!(!worker.is_healthy_and_eligible());
}

/// A record's telemetry feeds the `smg_engine_*` gauges, the PD gauges
/// included, as a poll's does; a core-only record after it (an event
/// batch between heartbeats) keeps the telemetry the worker last sent
/// and updates the core, so suppression blanks no gauge and `GET /loads`
/// shows the engine's sections between heartbeats.
#[tokio::test]
async fn a_pushed_records_telemetry_feeds_the_gauges_and_outlives_core_only_records() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let full = EngineLoad {
        cache_hit_rate: Some(0.75),
        num_used_tokens: Some(2_048),
        max_total_num_tokens: Some(8_192),
        memory: Some(smg_grpc_client::common_proto::EngineMemory {
            weight_gb: 15.0,
            kv_cache_gb: 40.0,
            graph_gb: 1.5,
            token_capacity: 8_192,
        }),
        disaggregation: Some(smg_grpc_client::common_proto::EngineDisaggregation {
            mode: "prefill".to_string(),
            prefill_prealloc_queue_reqs: 4,
            prefill_inflight_queue_reqs: 5,
            kv_transfer_latency_ms: 3.5,
            ..Default::default()
        }),
        ..pushed(7)
    };
    let rendered = render_with_recorder(|| {
        monitor.apply_pushed_load(&worker, 0, &full, Instant::now());
    });
    let gauge = |name: &str| {
        rendered
            .lines()
            .find(|l| {
                l.starts_with(&format!("{name}{{")) && l.contains("worker=\"grpc://w1:9000\"")
            })
            .map(|l| l.rsplit(' ').next().unwrap_or("").to_string())
            .unwrap_or_else(|| panic!("{name} missing:\n{rendered}"))
    };
    assert_eq!(gauge("smg_engine_cache_hit_rate"), "0.75");
    assert_eq!(gauge("smg_engine_pd_kv_transfer_latency_ms"), "3.5");
    assert_eq!(gauge("smg_engine_pd_prefill_queue_reqs"), "9");

    monitor.apply_pushed_load(&worker, 0, &pushed(8), Instant::now());
    monitor.flush_pushed();
    let snapshot = monitor.load_state.snapshot();
    let rank0 = &snapshot.get("grpc://w1:9000").expect("published").loads[0];
    assert_eq!(rank0.num_running_reqs, 8);
    assert!((rank0.cache_hit_rate - 0.75).abs() < f64::EPSILON);
    assert_eq!(rank0.max_total_num_tokens, 8_192);
    assert_eq!(rank0.memory.as_ref().map(|m| m.token_capacity), Some(8_192));
    assert_eq!(rank0.disagg_mode.as_deref(), Some("prefill"));
    assert_eq!(rank0.prefill_queue_reqs, Some(9));
}

/// The poll decision reads the receipt of the worker's pushed records
/// against the tick interval: none on file polls as always, a record
/// within the interval on every pushing rank suppresses the poll, and a
/// rank whose newest record is older falls back to the poll; the
/// record's own age (its `sampled_at`) plays no part.
#[tokio::test]
async fn the_poll_decision_follows_the_receipt_of_pushed_records() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let url = "grpc://w1:9000";
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    // An old servicer: no record, ever.
    assert_eq!(monitor.poll_mode(url, t0, interval), PollMode::Poll);
    // Heartbeats flowing: no poll while the newest record is younger
    // than the interval.
    monitor.apply_pushed_load(&worker, 0, &pushed(3), t0);
    assert_eq!(
        monitor.poll_mode(url, t0 + interval / 2, interval),
        PollMode::Suppressed
    );
    // Heartbeats stopped: the poll is back within one interval.
    assert_eq!(
        monitor.poll_mode(url, t0 + interval, interval),
        PollMode::Fallback
    );
    // A second rank pushes; the oldest pushing rank decides.
    monitor.apply_pushed_load(&worker, 1, &pushed(4), t0 + interval);
    assert_eq!(
        monitor.poll_mode(url, t0 + interval + interval / 2, interval),
        PollMode::Fallback
    );
    monitor.apply_pushed_load(&worker, 0, &pushed(5), t0 + interval + interval / 2);
    assert_eq!(
        monitor.poll_mode(url, t0 + interval + interval / 2, interval),
        PollMode::Suppressed
    );
    // A stale sample in a fresh record is still a fresh record.
    let late = t0 + 2 * interval;
    monitor.apply_pushed_load(
        &worker,
        0,
        &EngineLoad {
            age_ms: 60_000,
            ..pushed(6)
        },
        late,
    );
    monitor.apply_pushed_load(&worker, 1, &pushed(6), late);
    assert_eq!(
        monitor.poll_mode(url, late + Duration::from_millis(1), interval),
        PollMode::Suppressed
    );
    // A worker that leaves takes its records with it.
    monitor.evict_worker_loads(&worker);
    assert_eq!(monitor.poll_mode(url, late, interval), PollMode::Poll);
}

/// Under data parallelism a quiet rank's record stops arriving while
/// the other rank's heartbeats keep the stream alive, and the tick
/// falls back to the poll for the worker. The poll's figure for the
/// quiet rank replaces its entry in the pushed report: a heartbeat
/// naming the busy rank no longer rebuilds the report from the quiet
/// rank's last pushed record, the rank follows the poll from then on,
/// a rank whose own record arrived since the poll began keeps it, and
/// the poll decision is unchanged (the worker stays polled until the
/// rank's records resume).
#[tokio::test]
async fn a_quiet_ranks_figures_follow_the_poll_rather_than_its_last_pushed_record() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let url = "grpc://w1:9000";
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    let record = |running: u32, hit_rate: f64| EngineLoad {
        cache_hit_rate: Some(hit_rate),
        ..pushed(running)
    };
    let poll = |rank1_hit_rate: f64| WorkerLoadResponse {
        dp_rank_count: 2,
        loads: vec![
            SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: 2,
                cache_hit_rate: 0.9,
                ..Default::default()
            },
            SchedulerLoadSnapshot {
                dp_rank: 1,
                num_running_reqs: 8,
                cache_hit_rate: rank1_hit_rate,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let ranks_of = |report: &WorkerLoadResponse| -> Vec<(i32, i32, f64)> {
        report
            .loads
            .iter()
            .map(|rank| (rank.dp_rank, rank.num_running_reqs, rank.cache_hit_rate))
            .collect()
    };

    // Both ranks push during the first passes; then rank 1's working
    // set is cached and its records stop, while rank 0 keeps pushing.
    monitor.apply_pushed_load(&worker, 0, &record(3, 0.5), t0);
    monitor.apply_pushed_load(&worker, 1, &record(8, 0.65), t0);
    let polled_at = t0 + interval;
    assert_eq!(
        monitor.poll_mode(url, polled_at, interval),
        PollMode::Fallback
    );
    // Rank 0 pushed while the poll was in flight: its own record is the
    // newer; rank 1 pushed nothing since the poll began.
    monitor.apply_pushed_load(
        &worker,
        0,
        &record(4, 0.5),
        polled_at + Duration::from_millis(1),
    );
    monitor.apply_polled_load(url, &poll(0.91), polled_at);
    // The next heartbeat names rank 0 only: the report it rebuilds and
    // the gauges carry rank 1 at the poll's figure, rank 0 at its own.
    let rendered = render_with_recorder(|| {
        monitor.apply_pushed_load(
            &worker,
            0,
            &record(5, 0.5),
            polled_at + Duration::from_millis(2),
        );
    });
    monitor.flush_pushed();
    let snapshot = monitor.load_state.snapshot();
    let report = snapshot.get(url).expect("published");
    assert_eq!(report.version, "pushed");
    assert_eq!(ranks_of(report), vec![(0, 5, 0.5), (1, 8, 0.91)]);
    let rank1_hit_rate = rendered
        .lines()
        .find(|l| {
            l.starts_with("smg_engine_cache_hit_rate{")
                && l.contains("worker=\"grpc://w1:9000\"")
                && l.contains("dp_rank=\"1\"")
        })
        .map(|l| l.rsplit(' ').next().unwrap_or("").to_string())
        .unwrap_or_else(|| panic!("rank 1's hit rate gauge missing:\n{rendered}"));
    assert_eq!(rank1_hit_rate, "0.91");

    // The receipts are the stream's: rank 1 is still quiet, the worker
    // stays polled, and each poll's figure replaces the last. A rank
    // the stream never named (here a rank 2 the poll reports) stays
    // out of the pushed report: nothing would refresh it once the
    // pushing ranks are fresh again.
    let next = polled_at + interval;
    assert_eq!(monitor.poll_mode(url, next, interval), PollMode::Fallback);
    let mut with_a_third_rank = poll(0.93);
    with_a_third_rank.loads.push(SchedulerLoadSnapshot {
        dp_rank: 2,
        num_running_reqs: 1,
        cache_hit_rate: 0.2,
        ..Default::default()
    });
    with_a_third_rank.dp_rank_count = 3;
    monitor.apply_polled_load(url, &with_a_third_rank, next);
    monitor.apply_pushed_load(&worker, 0, &record(6, 0.5), next + Duration::from_millis(1));
    monitor.flush_pushed();
    let snapshot = monitor.load_state.snapshot();
    assert_eq!(
        ranks_of(snapshot.get(url).expect("published")),
        vec![(0, 6, 0.5), (1, 8, 0.93)]
    );
    // A rollup keyed by downstream worker rather than by rank is
    // ignored: its entries both say rank 0, and rank 0's slot keeps
    // the record the stream pushed before the rollup was asked.
    let rollup = WorkerLoadResponse {
        dp_rank_count: 2,
        loads: vec![
            SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: 40,
                worker: Some("http://downstream-a:8000".to_string()),
                ..Default::default()
            },
            SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: 41,
                worker: Some("http://downstream-b:8000".to_string()),
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    monitor.apply_polled_load(url, &rollup, next + Duration::from_millis(2));
    let rank0 = monitor
        .pushed_ranks
        .lock()
        .get(url)
        .and_then(|ranks| ranks.get(&0).cloned())
        .expect("rank 0 on file");
    assert_eq!(
        (rank0.num_running_reqs, rank0.cache_hit_rate, rank0.worker),
        (6, 0.5, None)
    );
    // Rank 1's own record, once it pushes again, is the newer figure
    // and suppresses the poll again; the report it rebuilds carries
    // rank 0 as the stream left it.
    monitor.apply_pushed_load(&worker, 1, &record(7, 0.7), next + Duration::from_millis(3));
    monitor.flush_pushed();
    let snapshot = monitor.load_state.snapshot();
    assert_eq!(
        ranks_of(snapshot.get(url).expect("published")),
        vec![(0, 6, 0.5), (1, 7, 0.7)]
    );
    assert_eq!(
        monitor.poll_mode(url, next + Duration::from_millis(4), interval),
        PollMode::Suppressed
    );

    // A worker with no pushed record on file has no report to merge
    // into: the poll's report stands on its own and the worker is
    // polled as always.
    let other = ready_worker("grpc://w2:9000", "llama-3");
    registry.register(Arc::clone(&other)).unwrap();
    monitor.apply_polled_load("grpc://w2:9000", &poll(0.1), t0);
    assert!(monitor.pushed_ranks.lock().get("grpc://w2:9000").is_none());
    assert_eq!(
        monitor.poll_mode("grpc://w2:9000", t0, interval),
        PollMode::Poll
    );
}

/// A tick does not poll a worker whose stream pushed within the
/// interval, leaves the pushed report in the shared snapshot, and
/// counts the skip; the record fed the `smg_engine_*` gauges on its
/// way in, so suppression blanks nothing. Once the records are older
/// than the interval the tick polls again (here a URL nobody serves:
/// the failed poll prunes the entry, as a failed poll always has).
#[test]
fn a_tick_skips_a_worker_whose_stream_pushed_within_the_interval() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let key = WorkerGroupKey {
        model_id: "llama-3".to_string(),
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Http,
    };
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    let rendered = render_with_recorder(|| {
        runtime.block_on(async {
            monitor.apply_pushed_load(&worker, 0, &pushed(7), t0);
            monitor.flush_pushed();
            poll_group_once(&monitor, &key, interval, t0 + interval / 2).await;
        });
    });
    let snapshot = monitor.load_state.snapshot();
    let report = snapshot
        .get("grpc://w1:9000")
        .expect("kept across the tick");
    assert_eq!(report.version, "pushed");
    assert_eq!(report.loads[0].num_running_reqs, 7);
    let line = |name: &str, label: &str| {
        rendered
            .lines()
            .find(|l| l.starts_with(&format!("{name}{{")) && l.contains(label))
            .map(str::to_string)
            .unwrap_or_else(|| panic!("{name} with {label} missing:\n{rendered}"))
    };
    assert!(
        line("smg_engine_running_requests", "worker=\"grpc://w1:9000\"").ends_with(" 7"),
        "{rendered}"
    );
    assert!(
        line("smg_engine_load_polls_total", "mode=\"skipped_fresh_push\"").ends_with(" 1"),
        "{rendered}"
    );

    let rendered = render_with_recorder(|| {
        runtime.block_on(poll_group_once(&monitor, &key, interval, t0 + 2 * interval));
    });
    assert!(
        monitor
            .load_state
            .snapshot()
            .get("grpc://w1:9000")
            .is_none(),
        "the fallback poll of an unreachable worker prunes its entry"
    );
    assert!(
        rendered
            .lines()
            .any(|l| l.starts_with("smg_engine_load_polls_total{")
                && l.contains("mode=\"fallback\"")
                && l.ends_with(" 1")),
        "{rendered}"
    );
}
