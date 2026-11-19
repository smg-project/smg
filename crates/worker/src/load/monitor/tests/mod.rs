mod prom_scrape;
mod worker_load_manager;

use std::{collections::HashMap, sync::mpsc};

use super::*;
use crate::{
    common_proto as common,
    health::{manager::WorkerManager, overload::OverloadThresholds},
    identity::basic::{ConnectionMode, WorkerType},
    model_card::ModelCard,
    runtime::StallReason,
    test_support::{
        assert_metric, ready_worker, render_with_recorder, BasicWorkerBuilder, LoadReaders,
        ScriptedAnswers, WorkerMonitor, WorkerRegistry,
    },
    worker::{
        EngineMemoryMetricsSnapshot, EngineQueueMetricsSnapshot, HealthCheckConfig, NamedValue,
        WorkerStatus,
    },
};

/// Build an isolated monitor and registry beside a routing side that reads
/// no loads.
fn build_monitor() -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
    let registry = Arc::new(WorkerRegistry::new());
    let monitor = Arc::new(WorkerMonitor::new(
        registry.clone(),
        Arc::new(LoadReaders::Blind),
        5,
        false,
        false,
    ));
    (registry, monitor)
}

/// A record whose every reader fails the case that calls it.
struct Unread;

impl LoadRecord for Unread {
    fn age(&self) -> Duration {
        panic!("the age of an ignored record was read")
    }

    fn telemetry(&self) -> LoadTelemetry {
        panic!("the telemetry of an ignored record was read")
    }

    fn snapshot(&self) -> SchedulerLoadSnapshot {
        panic!("the snapshot of an ignored record was built")
    }

    fn values_sent(&self) -> usize {
        panic!("the named values of an ignored record were counted")
    }

    fn engine_silence(&self) -> Option<Duration> {
        panic!("the silence of an ignored record was read")
    }

    fn reports_every_step(&self) -> bool {
        panic!("the step word of an ignored record was read")
    }

    fn prefill_pending(&self) -> Option<u64> {
        panic!("the pending prefill of an ignored record was read")
    }
}

/// A pushed record as plain data: the six readings the monitor takes of a
/// [`LoadRecord`], so a case states what the monitor is told and names no
/// wire message. The setters keep the entry's own relations: its total is
/// the running and the waiting summed, its utilization is its token usage,
/// and a figure the record carries sets that figure's flag. The wire
/// message's reader holds the same relations, by its own cases in the gRPC
/// client crate (`crates/grpc_client/src/engine_load.rs`).
#[derive(Clone)]
pub(super) struct PushedRecord {
    pub(super) age: Duration,
    pub(super) telemetry: LoadTelemetry,
    pub(super) snapshot: SchedulerLoadSnapshot,
    pub(super) engine_silence: Option<Duration>,
    pub(super) reports_every_step: bool,
    pub(super) prefill_pending: Option<u64>,
}

impl Default for PushedRecord {
    fn default() -> Self {
        Self {
            age: Duration::ZERO,
            telemetry: LoadTelemetry::empty(),
            snapshot: SchedulerLoadSnapshot::default(),
            engine_silence: None,
            reports_every_step: false,
            prefill_pending: None,
        }
    }
}

impl PushedRecord {
    /// A record whose core reports `running` and `waiting` requests.
    pub(super) fn requests(running: i32, waiting: i32) -> Self {
        Self::default().with_requests(running, waiting)
    }

    /// The record with another count of running and waiting requests.
    pub(super) fn with_requests(mut self, running: i32, waiting: i32) -> Self {
        self.snapshot.num_running_reqs = running;
        self.snapshot.num_waiting_reqs = waiting;
        self.snapshot.num_total_reqs = running + waiting;
        self
    }

    /// The record with `usage` as its KV usage.
    pub(super) fn token_usage(mut self, usage: f64) -> Self {
        self.snapshot.token_usage = usage;
        self.snapshot.utilization = usage;
        self
    }

    /// The record carrying `rate` as its cache hit rate.
    pub(super) fn cache_hit_rate(mut self, rate: f64) -> Self {
        self.snapshot.cache_hit_rate = rate;
        self.telemetry |= LoadTelemetry::CACHE_HIT_RATE;
        self
    }

    /// The record carrying `used` as its count of used tokens.
    pub(super) fn used_tokens(mut self, used: i32) -> Self {
        self.snapshot.num_used_tokens = used;
        self.telemetry |= LoadTelemetry::USED_TOKENS;
        self
    }

    /// The record carrying `total` as its token capacity.
    pub(super) fn total_tokens(mut self, total: i32) -> Self {
        self.snapshot.max_total_num_tokens = total;
        self.telemetry |= LoadTelemetry::TOTAL_TOKENS;
        self
    }

    /// The record carrying a queues section of `waiting` waiting requests.
    pub(super) fn queues_waiting(mut self, waiting: i32) -> Self {
        self.snapshot.queues = Some(EngineQueueMetricsSnapshot {
            waiting,
            ..Default::default()
        });
        self.telemetry |= LoadTelemetry::QUEUES;
        self
    }

    /// The record carrying `memory` as its memory section.
    pub(super) fn memory(mut self, memory: EngineMemoryMetricsSnapshot) -> Self {
        self.snapshot.memory = Some(memory);
        self.telemetry |= LoadTelemetry::MEMORY;
        self
    }

    /// The record carrying a disaggregation section: the KV transfer's
    /// latency and speed, the prefill and decode queue depths and the mode.
    pub(super) fn disaggregation(
        mut self,
        latency_ms: f64,
        speed_gb_s: f64,
        prefill_queue: i32,
        decode_queue: i32,
        mode: &str,
    ) -> Self {
        self.snapshot.kv_transfer_latency_ms = Some(latency_ms);
        self.snapshot.kv_transfer_speed_gb_s = Some(speed_gb_s);
        self.snapshot.prefill_queue_reqs = Some(prefill_queue);
        self.snapshot.decode_queue_reqs = Some(decode_queue);
        self.snapshot.disagg_mode = Some(mode.to_string());
        self.telemetry |= LoadTelemetry::DISAGGREGATION;
        self
    }
}

impl LoadRecord for PushedRecord {
    fn age(&self) -> Duration {
        self.age
    }

    fn telemetry(&self) -> LoadTelemetry {
        self.telemetry
    }

    fn snapshot(&self) -> SchedulerLoadSnapshot {
        self.snapshot.clone()
    }

    fn values_sent(&self) -> usize {
        self.snapshot.values.len()
    }

    fn engine_silence(&self) -> Option<Duration> {
        self.engine_silence
    }

    fn reports_every_step(&self) -> bool {
        self.reports_every_step
    }

    fn prefill_pending(&self) -> Option<u64> {
        self.prefill_pending
    }
}

/// A record of a negative rank, or of a rank that is another virtual
/// worker's, is dropped before any reader of it is called: the monitor
/// builds no snapshot for a record it ignores.
#[tokio::test]
async fn an_ignored_record_is_never_read() {
    let (registry, monitor) = build_monitor();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:9000")
            .worker_type(WorkerType::Decode)
            .dp_config(1, 8)
            .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    registry.register(Arc::clone(&worker)).unwrap();
    let now = Instant::now();
    monitor.apply_pushed_load(&worker, -1, &Unread, now);
    monitor.apply_pushed_load(&worker, 0, &Unread, now);
    assert_eq!(monitor.pending_pushed_len(), 0);
    assert!(!worker.runtime_state().unwrap().pushes_load_records());
}

/// Backend-wide pushes only refresh the matching virtual rank; quiet ranks keep polling.
#[tokio::test]
async fn dp8_pushes_cannot_redirect_virtual_workers_to_active_ranks() {
    let (registry, monitor) = build_monitor();
    let now = Instant::now();
    let interval = Duration::from_secs(10);
    let mut active = pushed(7).used_tokens(2048).total_tokens(8192);
    active.snapshot.max_running_requests = 64;
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
        assert_eq!(
            worker.runtime_state().unwrap().pushes_load_records(),
            is_active
        );
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
    let record = PushedRecord {
        age: Duration::from_millis(30),
        snapshot: SchedulerLoadSnapshot {
            num_waiting_uncached_tokens: 3_000,
            gen_throughput: 900.0,
            max_running_requests: 64,
            ..Default::default()
        },
        ..Default::default()
    }
    .with_requests(7, 2)
    .token_usage(0.4);
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
    let mut second = record.clone().with_requests(1, 2);
    second.snapshot.num_waiting_uncached_tokens = 0;
    monitor.apply_pushed_load(&worker, 1, &second, Instant::now());
    monitor.apply_pushed_load(
        &worker,
        0,
        &record.clone().with_requests(8, 2),
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
    monitor.apply_pushed_load(&worker, 0, &PushedRecord::default(), Instant::now());
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

fn pushed(running: i32) -> PushedRecord {
    PushedRecord {
        snapshot: SchedulerLoadSnapshot {
            gen_throughput: 10.0,
            max_running_requests: 32,
            ..Default::default()
        },
        ..Default::default()
    }
    .with_requests(running, 1)
    .token_usage(0.2)
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
    record.engine_silence = Some(Duration::from_millis(5_000));
    record.reports_every_step = true;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(worker.stall_reason(), Some(StallReason::Wedged));
    assert!(!worker.is_healthy_and_eligible());
    record.engine_silence = None;
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
    record.reports_every_step = true;
    record.prefill_pending = Some(156_752);
    for silence in [3_000, 10_000, 31_000] {
        record.engine_silence = Some(Duration::from_millis(silence));
        monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
        assert!(
            worker.stall_reason().is_none(),
            "{silence} ms of silence inside the prefill's allowance"
        );
    }
    record.engine_silence = Some(Duration::from_millis(32_000));
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(
        worker.stall_reason(),
        Some(StallReason::Wedged),
        "silent past twice the prefill at the cold prior"
    );
    record.engine_silence = None;
    record.prefill_pending = Some(0);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(worker.stall_reason().is_none(), "the engine produces again");
    // Nothing pending: the threshold, as before.
    record.engine_silence = Some(Duration::from_millis(3_000));
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(
        worker.stall_reason(),
        Some(StallReason::Wedged),
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
    let mut record = pushed(1).with_requests(1, 8);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(worker.is_overloaded(), "eight waiting is the threshold");
    assert!(logs_contain("Worker overload veto set"));
    assert!(logs_contain("waiting_requests=8"));
    assert!(logs_contain("pushed record"));
    assert!(logs_contain(
        "smg_worker::load::monitor: Worker overload veto set"
    ));
    assert!(!logs_contain("Worker overload veto cleared"));
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
    record = record.with_requests(1, 0);
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
    let record = PushedRecord {
        engine_silence: Some(Duration::from_millis(5_000)),
        reports_every_step: true,
        prefill_pending: Some(47),
        ..Default::default()
    };
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(worker.stall_reason(), Some(StallReason::Wedged));
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
    // The memory section and the disaggregation section, each whole: what a
    // section leaves at zero reads as zero.
    let full = pushed(7)
        .cache_hit_rate(0.75)
        .used_tokens(2_048)
        .total_tokens(8_192)
        .memory(EngineMemoryMetricsSnapshot {
            weight_gb: 15.0,
            kv_cache_gb: 40.0,
            graph_gb: 1.5,
            token_capacity: 8_192,
        })
        .disaggregation(3.5, 0.0, 9, 0, "prefill");
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

/// A memory section of `token_capacity` tokens.
fn memory_of(token_capacity: i32) -> EngineMemoryMetricsSnapshot {
    EngineMemoryMetricsSnapshot {
        token_capacity,
        ..Default::default()
    }
}

/// Rank 0 of `url` as published once the pushed records are flushed.
fn published_rank0(monitor: &WorkerMonitor, url: &str) -> SchedulerLoadSnapshot {
    monitor.flush_pushed();
    let snapshot = monitor.load_state.snapshot();
    snapshot.get(url).expect("published").loads[0].clone()
}

/// A record that carries other telemetry and neither its memory section nor
/// its queues section keeps both as the last record that carried them said.
#[tokio::test]
async fn a_record_keeps_the_memory_and_queues_sections_it_does_not_carry() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let first = pushed(7).memory(memory_of(8_192)).queues_waiting(4);
    monitor.apply_pushed_load(&worker, 0, &first, Instant::now());
    let other = pushed(8).cache_hit_rate(0.5);
    monitor.apply_pushed_load(&worker, 0, &other, Instant::now());

    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(rank0.num_running_reqs, 8);
    assert!((rank0.cache_hit_rate - 0.5).abs() < f64::EPSILON);
    assert_eq!(rank0.memory.map(|m| m.token_capacity), Some(8_192));
    assert_eq!(rank0.queues.map(|q| q.waiting), Some(4));
}

/// A record that carries its memory section or its queues section replaces
/// the last one, each by its own flag: the section it does not carry stays.
#[tokio::test]
async fn a_record_replaces_the_memory_or_queues_section_it_carries() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let first = pushed(7).memory(memory_of(8_192)).queues_waiting(4);
    monitor.apply_pushed_load(&worker, 0, &first, Instant::now());

    let memory_only = pushed(8).memory(memory_of(4_096));
    monitor.apply_pushed_load(&worker, 0, &memory_only, Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(rank0.memory.map(|m| m.token_capacity), Some(4_096));
    assert_eq!(rank0.queues.map(|q| q.waiting), Some(4));

    let queues_only = pushed(9).queues_waiting(6);
    monitor.apply_pushed_load(&worker, 0, &queues_only, Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(rank0.memory.map(|m| m.token_capacity), Some(4_096));
    assert_eq!(rank0.queues.map(|q| q.waiting), Some(6));
}

/// A record's named values are that record's alone. The next record's list
/// takes their place in the rank's entry, where the fixed telemetry a record
/// leaves out stays: a reader that wants a value across records keeps it
/// itself.
#[tokio::test]
async fn a_records_named_values_take_the_place_of_the_ranks_last() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let named = |name: &'static str, value: f64| NamedValue {
        name: name.into(),
        unit: "items".into(),
        value,
        age_ms: None,
    };
    let mut heartbeat = pushed(7).used_tokens(2_048);
    heartbeat.snapshot.values = vec![named("queue_depth", 4.0), named("ready", 1.0)];
    monitor.apply_pushed_load(&worker, 0, &heartbeat, Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(rank0.values, heartbeat.snapshot.values);

    // A record that names one of the two: its list stands whole.
    let mut other = pushed(8);
    other.snapshot.values = vec![named("ready", 0.0)];
    monitor.apply_pushed_load(&worker, 0, &other, Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(rank0.values, other.snapshot.values);
    assert_eq!(rank0.num_used_tokens, 2_048);

    // A core-only record names none: the entry holds none, and the fixed
    // telemetry of the heartbeat stays.
    monitor.apply_pushed_load(&worker, 0, &pushed(9), Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert!(rank0.values.is_empty());
    assert_eq!(rank0.num_used_tokens, 2_048);
    assert_eq!(rank0.num_running_reqs, 9);
}

/// The named values a pushed record's snapshot drops at a bound are counted
/// by the worker: the record's list less the snapshot's. A record whose
/// values all stand writes no series.
#[tokio::test]
async fn a_pushed_record_counts_the_named_values_its_snapshot_drops() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let value = |name: &str, value: f64| common::NamedValue {
        name: name.to_string(),
        unit: "items".to_string(),
        value,
        age_ms: None,
    };
    let whole = common::EngineLoad {
        running_requests: 7,
        values: vec![value("queue_depth", 4.0)],
        ..Default::default()
    };
    let rendered = render_with_recorder(|| {
        monitor.apply_pushed_load(&worker, 0, &whole, Instant::now());
    });
    assert!(
        !rendered.contains("smg_engine_load_values_dropped_total{"),
        "{rendered}"
    );

    // One stands. An empty name, a value that is not finite and a name the
    // record states again are dropped: three of each record's four.
    let dropping = common::EngineLoad {
        running_requests: 8,
        values: vec![
            value("queue_depth", 5.0),
            value("", 1.0),
            value("ready", f64::NAN),
            value("queue_depth", 6.0),
        ],
        ..Default::default()
    };
    let rendered = render_with_recorder(|| {
        monitor.apply_pushed_load(&worker, 0, &dropping, Instant::now());
        monitor.apply_pushed_load(&worker, 0, &dropping, Instant::now());
    });
    assert_metric(
        &rendered,
        "smg_engine_load_values_dropped_total",
        &["worker=\"grpc://w1:9000\""],
        "6",
    );
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(rank0.values.len(), 1);
    assert_eq!(
        (rank0.values[0].name.as_ref(), rank0.values[0].value),
        ("queue_depth", 5.0)
    );
}

/// What a rank's entry offers by name is its own record's. A count the entry
/// keeps from a heartbeat is not offered under a later record that did not
/// carry it, and the entry still holds it.
#[tokio::test]
async fn a_core_record_offers_no_fixed_count_the_entry_kept() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let names = |rank: &SchedulerLoadSnapshot| -> Vec<String> {
        rank.named_values()
            .map(|value| value.name.to_string())
            .collect()
    };
    let heartbeat = pushed(7).used_tokens(2_048).total_tokens(8_192);
    monitor.apply_pushed_load(&worker, 0, &heartbeat, Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(names(&rank0), ["kv_tokens_used", "kv_tokens_capacity"]);

    monitor.apply_pushed_load(&worker, 0, &pushed(8), Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert!(names(&rank0).is_empty());
    assert_eq!(
        (rank0.num_used_tokens, rank0.max_total_num_tokens),
        (2_048, 8_192)
    );

    // A record that carries one of the two offers that one.
    let capacity = pushed(9).total_tokens(4_096);
    monitor.apply_pushed_load(&worker, 0, &capacity, Instant::now());
    let rank0 = published_rank0(&monitor, "grpc://w1:9000");
    assert_eq!(names(&rank0), ["kv_tokens_capacity"]);
    assert_eq!(
        (rank0.num_used_tokens, rank0.max_total_num_tokens),
        (2_048, 4_096)
    );
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
        &PushedRecord {
            age: Duration::from_millis(60_000),
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

/// A worker's first record files its receipt under the record's own rank.
/// Under another rank the receipt would age with no record to renew it,
/// and one interval later the worker would fall back to the poll with its
/// stream at work.
#[tokio::test]
async fn a_workers_first_record_files_its_receipt_under_its_own_rank() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    let url = "grpc://w1:9000";
    let interval = Duration::from_secs(10);
    let t0 = Instant::now();
    monitor.apply_pushed_load(&worker, 1, &pushed(3), t0);
    assert!(monitor.has_receipt(url, 1));
    assert!(!monitor.has_receipt(url, 0), "rank 0 pushed nothing");
    monitor.apply_pushed_load(&worker, 1, &pushed(4), t0 + interval);
    assert_eq!(
        monitor.poll_mode(url, t0 + interval, interval),
        PollMode::Suppressed
    );
}

/// Under data parallelism a quiet rank's record stops arriving while
/// the other rank's heartbeats keep the stream alive, and the tick
/// falls back to the poll for the worker. The poll's figure for the
/// quiet rank replaces its entry in the pushed report: a heartbeat
/// naming the busy rank no longer publishes a report with the quiet
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
    let record = |running: i32, hit_rate: f64| pushed(running).cache_hit_rate(hit_rate);
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
    // The next heartbeat names rank 0 only: the report it publishes carries
    // rank 1 at the poll's figure and rank 0 at its own, and the record sets
    // its own rank's gauges alone (rank 1's took the poll's figure from the
    // poll itself, as every polled rank's do).
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
    let hit_rate_of = |rank: &str| {
        rendered
            .lines()
            .find(|l| {
                l.starts_with("smg_engine_cache_hit_rate{")
                    && l.contains("worker=\"grpc://w1:9000\"")
                    && l.contains(&format!("dp_rank=\"{rank}\""))
            })
            .map(|l| l.rsplit(' ').next().unwrap_or("").to_string())
    };
    assert_eq!(hit_rate_of("0").as_deref(), Some("0.5"), "{rendered}");
    assert_eq!(
        hit_rate_of("1"),
        None,
        "a record touches no other rank's series:\n{rendered}"
    );

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
        .pushed_reports
        .lock()
        .get(url)
        .and_then(|file| {
            file.report
                .loads
                .iter()
                .find(|load| load.dp_rank == 0)
                .cloned()
        })
        .expect("rank 0 on file");
    assert_eq!(
        (rank0.num_running_reqs, rank0.cache_hit_rate, rank0.worker),
        (6, 0.5, None)
    );
    // Rank 1's own record, once it pushes again, is the newer figure
    // and suppresses the poll again; the report it publishes carries
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
    assert!(monitor
        .pushed_reports
        .lock()
        .get("grpc://w2:9000")
        .is_none());
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

/// The quarter-second sweep visits the registry's membership as it stands at
/// each tick: a worker registered after the loop started is swept on the next
/// tick, a removed one no longer. A sweep is observed through the load sample
/// it stores on the worker (`swap_load_sample`), which every sweep writes
/// before it judges anything.
#[tokio::test(start_paused = true)]
async fn the_liveness_sweep_follows_the_registrys_membership_tick_by_tick() {
    let (registry, monitor) = build_monitor();
    // The sleeps end just after a tick's deadline, so the paused clock fires
    // the tick before the assertion runs.
    let past_a_tick = liveness::SWEEP_INTERVAL + Duration::from_millis(1);

    let first = ready_worker("http://w1:8000", "m");
    first.runtime_state().unwrap().note_tracked_started();
    registry.register(Arc::clone(&first)).unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "the test aborts the loop before it returns"
    )]
    let sweep = tokio::spawn(liveness_sweep_loop(Arc::downgrade(&monitor)));
    tokio::time::sleep(past_a_tick).await;
    assert_eq!(
        first.runtime_state().unwrap().swap_load_sample(0),
        1,
        "a worker registered before the loop started is swept"
    );

    let late = ready_worker("http://w2:8000", "m");
    late.runtime_state().unwrap().note_tracked_started();
    late.runtime_state().unwrap().note_tracked_started();
    let late_id = registry.register(Arc::clone(&late)).unwrap();
    tokio::time::sleep(past_a_tick).await;
    assert_eq!(
        late.runtime_state().unwrap().swap_load_sample(0),
        2,
        "a worker registered after the loop started is swept on the next tick"
    );
    assert_eq!(
        first.runtime_state().unwrap().swap_load_sample(0),
        1,
        "the first worker is swept on the same tick"
    );

    registry.remove(&late_id).unwrap();
    tokio::time::sleep(past_a_tick).await;
    assert_eq!(
        late.runtime_state().unwrap().swap_load_sample(0),
        0,
        "a removed worker is not swept"
    );
    assert_eq!(
        first.runtime_state().unwrap().swap_load_sample(0),
        1,
        "the worker that stays is swept on the tick after the removal"
    );
    sweep.abort();
}

/// What a recording sink was told, in order, with whether the monitor's
/// reports were locked at the call.
#[derive(Clone, Debug, PartialEq)]
enum Said {
    Record {
        url: String,
        rank: i32,
        rows: usize,
        received_at: Instant,
        sampled_at: Instant,
        locked: bool,
    },
    PollMode(String),
    /// Each member's URL, the rows of its answer and the rows on file.
    Report(Vec<(String, Option<usize>, Option<usize>)>, bool),
    ReportEmpty(Vec<(String, Option<usize>, Option<usize>)>, bool),
    FetchFailed(Vec<String>, bool),
    /// The worker's URL, and whether the lock of the reports was held.
    Forget(String, bool),
    Reset,
    /// The Ready workers of the group, then each report's URL, its rows and
    /// whether its answer carried rank capacity.
    Tick(usize, Vec<(String, usize, bool)>),
}

/// A sink that only records: what the monitor says is kept in order, each
/// question gets the answer the case set, and nothing is written anywhere.
/// It reads the lock of the monitor's reports at each call, so a case tells
/// which calls the monitor makes under it.
struct RecordingLoadSink {
    said: Mutex<Vec<Said>>,
    /// The workers the sink was told to forget, in order.
    forgotten: Mutex<Vec<Arc<dyn Worker>>>,
    polled_values: Mutex<Vec<NamedValue>>,
    mode: PollMode,
    ranks: PolledRanks,
    needs: bool,
    monitor: std::sync::OnceLock<Weak<WorkerMonitor>>,
}

impl RecordingLoadSink {
    fn locked(&self) -> bool {
        self.monitor
            .get()
            .and_then(Weak::upgrade)
            .is_some_and(|monitor| monitor.pushed_reports.is_locked())
    }

    fn members(answer: &[PolledMember<'_>]) -> Vec<(String, Option<usize>, Option<usize>)> {
        answer
            .iter()
            .map(|member| {
                (
                    member.worker.url().to_string(),
                    member.answer.map(|load| load.loads.len()),
                    member.on_file.map(|load| load.loads.len()),
                )
            })
            .collect()
    }
}

impl LoadSink for RecordingLoadSink {
    fn record(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        received_at: Instant,
        sampled_at: Instant,
    ) {
        let locked = self.locked();
        self.said.lock().push(Said::Record {
            url: worker.url().to_string(),
            rank,
            rows: report.loads.len(),
            received_at,
            sampled_at,
            locked,
        });
    }

    fn poll_mode(&self, worker: &Arc<dyn Worker>, _now: Instant, _interval: Duration) -> PollMode {
        self.said
            .lock()
            .push(Said::PollMode(worker.url().to_string()));
        self.mode
    }

    fn report(&self, answer: &[PolledMember<'_>], _polled_at: Instant) -> PolledRanks {
        let locked = self.locked();
        self.polled_values.lock().extend(
            answer
                .iter()
                .filter_map(|member| member.answer)
                .flat_map(|report| &report.loads)
                .flat_map(|rank| &rank.values)
                .cloned(),
        );
        self.said
            .lock()
            .push(Said::Report(Self::members(answer), locked));
        self.ranks.clone()
    }

    fn report_empty(&self, answer: &[PolledMember<'_>], _polled_at: Instant) -> Vec<i32> {
        let locked = self.locked();
        self.said
            .lock()
            .push(Said::ReportEmpty(Self::members(answer), locked));
        self.ranks.leaving.clone()
    }

    fn fetch_failed(&self, members: &[Arc<dyn Worker>]) {
        let locked = self.locked();
        let urls = members
            .iter()
            .map(|worker| worker.url().to_string())
            .collect();
        self.said.lock().push(Said::FetchFailed(urls, locked));
    }

    fn forget(&self, worker: &Arc<dyn Worker>) {
        self.said
            .lock()
            .push(Said::Forget(worker.url().to_string(), self.locked()));
        self.forgotten.lock().push(Arc::clone(worker));
    }

    fn reset(&self) {
        self.said.lock().push(Said::Reset);
    }

    fn needs_loads(&self) -> bool {
        self.needs
    }

    fn tick(&self, tick: &GroupTick<'_>) {
        let reports = tick
            .reports
            .iter()
            .map(|polled| {
                (
                    polled.worker.url().to_string(),
                    polled.report.loads.len(),
                    polled.carries_rank_capacity,
                )
            })
            .collect();
        self.said
            .lock()
            .push(Said::Tick(tick.workers.len(), reports));
    }
}

/// A monitor over a recording sink that answers `mode` to the poll question
/// and `ranks` to a poll's answer, with the sink and the registry.
fn build_recording(
    mode: PollMode,
    ranks: PolledRanks,
    needs: bool,
    conditional_polling: bool,
) -> (
    Arc<WorkerRegistry>,
    Arc<WorkerMonitor>,
    Arc<RecordingLoadSink>,
) {
    let registry = Arc::new(WorkerRegistry::new());
    let sink = Arc::new(RecordingLoadSink {
        said: Mutex::new(Vec::new()),
        forgotten: Mutex::new(Vec::new()),
        polled_values: Mutex::new(Vec::new()),
        mode,
        ranks,
        needs,
        monitor: std::sync::OnceLock::new(),
    });
    let monitor = Arc::new(WorkerMonitor::with_sink(
        registry.clone(),
        sink.clone(),
        5,
        false,
        conditional_polling,
    ));
    assert!(sink.monitor.set(Arc::downgrade(&monitor)).is_ok());
    (registry, monitor, sink)
}

/// The rank and the running requests of each row, in order.
fn rows(load: &WorkerLoadResponse) -> Vec<(i32, i32)> {
    load.loads
        .iter()
        .map(|row| (row.dp_rank, row.num_running_reqs))
        .collect()
}

/// An answer that names `rank` alone, with `running` requests.
fn answer_of(rank: i32, running: i32) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: rank,
            num_running_reqs: running,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[tokio::test]
async fn a_connections_named_values_reach_the_load_sink_and_fleet_response() {
    let (registry, monitor, sink) =
        build_recording(PollMode::Poll, PolledRanks::default(), true, false);
    let expected = vec![
        NamedValue {
            name: "blocked_cycles".into(),
            unit: "cycles".into(),
            value: 17.0,
            age_ms: Some(29),
        },
        NamedValue {
            name: "wait_time".into(),
            unit: "ms".into(),
            value: 2.75,
            age_ms: None,
        },
    ];
    let mut answer = answer_of(2, 3);
    answer.loads[0].values.clone_from(&expected);
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://worker:9132")
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("m"))
            .engine_connection(ScriptedAnswers::answering_loads(answer))
            .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    registry.register(worker).unwrap();
    let group = WorkerGroupKey {
        model_id: "m".into(),
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Grpc,
    };
    poll_group_once(&monitor, &group, Duration::from_secs(5), Instant::now()).await;
    assert_eq!(*sink.polled_values.lock(), expected);

    let fleet = WorkerManager::fleet_loads(&registry, &monitor.load_snapshot(), None, "test");
    assert_eq!(fleet.loads[0].values, expected);
    let json = serde_json::to_value(fleet).unwrap();
    assert_eq!(
        json["loads"][0]["values"],
        serde_json::json!([
            {"name": "blocked_cycles", "unit": "cycles", "value": 17.0, "age_ms": 29},
            {"name": "wait_time", "unit": "ms", "value": 2.75},
        ])
    );
    let decoded: WorkerLoadResponse = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.loads[0].values, expected);
}

/// A pushed record crosses the port once, after the monitor laid it in the
/// worker's report and under the lock of the reports. The sink is told the
/// worker, the rank, the report as it stands, the receipt, and a sample
/// instant that is the receipt less the record's age and the one-way margin.
#[tokio::test]
async fn a_pushed_record_crosses_the_port_under_the_lock_of_the_reports() {
    let (registry, monitor, sink) =
        build_recording(PollMode::Poll, PolledRanks::default(), true, false);
    let worker = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&worker)).unwrap();
    let received_at = Instant::now();
    let mut aged = PushedRecord::requests(3, 2);
    aged.age = Duration::from_millis(30);

    monitor.apply_pushed_load(&worker, 1, &aged, received_at);
    monitor.apply_pushed_load(&worker, 0, &PushedRecord::requests(1, 0), received_at);

    assert_eq!(
        *sink.said.lock(),
        vec![
            Said::Record {
                url: "http://w:1".to_string(),
                rank: 1,
                rows: 1,
                received_at,
                sampled_at: received_at - Duration::from_millis(30) - PUSHED_ONE_WAY_MARGIN,
                locked: true,
            },
            Said::Record {
                url: "http://w:1".to_string(),
                rank: 0,
                rows: 2,
                received_at,
                sampled_at: received_at - PUSHED_ONE_WAY_MARGIN,
                locked: true,
            },
        ]
    );
}

/// One engine's answer crosses whole and under the lock of the reports: the
/// member with its answer's rows and the rows on file. The file is then laid
/// by the ranks the sink names: the rank whose stream is the newer keeps its
/// row and hands it to the answer, the rank that leaves loses its row, and
/// the other takes the answer's entry.
#[tokio::test]
async fn an_engines_answer_crosses_whole_and_the_file_is_laid_by_the_sinks_ranks() {
    let ranks = PolledRanks {
        newer: vec![1],
        leaving: vec![2],
    };
    let (registry, monitor, sink) = build_recording(PollMode::Fallback, ranks, true, false);
    let worker = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&worker)).unwrap();
    for rank in 0..3 {
        let record = PushedRecord::requests(10 + rank, 0);
        monitor.apply_pushed_load(&worker, rank, &record, Instant::now());
    }
    sink.said.lock().clear();

    let mut engine = vec![(Arc::clone(&worker), Some(answer_of(0, 7)))];
    let carries_rank_capacity = monitor.reconcile_engine(&mut engine, Instant::now());

    assert_eq!(
        *sink.said.lock(),
        vec![Said::Report(
            vec![("http://w:1".to_string(), Some(1), Some(3))],
            true
        )]
    );
    {
        let reports = monitor.pushed_reports.lock();
        let file = &reports["http://w:1"].report;
        assert_eq!(rows(file), vec![(0, 7), (1, 11)]);
        assert_eq!(file.dp_rank_count, 2);
    }
    let answer = engine[0].1.as_ref().expect("the answer stays");
    assert_eq!(rows(answer), vec![(0, 7), (1, 11)]);
    assert_eq!(
        carries_rank_capacity,
        vec![false],
        "an answer with no token capacity lists no ranks for a cache"
    );
}

/// An answer that brought no row and a fetch that failed are said apart: the
/// first under the lock of the reports, with the file laid by the ranks the
/// sink says leave; the second with no lock held and nothing laid.
#[tokio::test]
async fn an_answer_with_no_row_and_a_failed_fetch_are_said_apart() {
    let ranks = PolledRanks {
        newer: Vec::new(),
        leaving: vec![0],
    };
    let (registry, monitor, sink) = build_recording(PollMode::Poll, ranks, true, false);
    let worker = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&worker)).unwrap();
    monitor.apply_pushed_load(&worker, 0, &PushedRecord::requests(1, 0), Instant::now());
    monitor.apply_pushed_load(&worker, 1, &PushedRecord::requests(2, 0), Instant::now());
    sink.said.lock().clear();

    let mut failed = vec![(Arc::clone(&worker), None)];
    monitor.reconcile_engine(&mut failed, Instant::now());
    assert_eq!(
        rows(&monitor.pushed_reports.lock()["http://w:1"].report),
        vec![(0, 1), (1, 2)],
        "a failed fetch lays nothing"
    );
    let mut empty = vec![(Arc::clone(&worker), Some(WorkerLoadResponse::default()))];
    monitor.reconcile_engine(&mut empty, Instant::now());

    assert_eq!(
        *sink.said.lock(),
        vec![
            Said::FetchFailed(vec!["http://w:1".to_string()], false),
            Said::ReportEmpty(vec![("http://w:1".to_string(), Some(0), Some(2))], true),
        ]
    );
    assert_eq!(
        rows(&monitor.pushed_reports.lock()["http://w:1"].report),
        vec![(1, 2)],
        "the rank the sink named left the file"
    );
}

/// A tick that polled nobody still ends at the sink, with every Ready worker
/// of the group and no report: a sink that trims what its readers booked
/// does it on every tick. Nothing is fetched and nothing is published.
#[tokio::test]
async fn a_tick_that_polled_nobody_ends_at_the_sink_with_every_ready_worker() {
    let (registry, monitor, sink) =
        build_recording(PollMode::Suppressed, PolledRanks::default(), true, false);
    let worker = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&worker)).unwrap();
    let key = WorkerGroupKey {
        model_id: "m".to_string(),
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Http,
    };

    poll_group_once(&monitor, &key, Duration::from_secs(5), Instant::now()).await;

    assert_eq!(
        *sink.said.lock(),
        vec![
            Said::PollMode("http://w:1".to_string()),
            Said::Tick(1, Vec::new()),
        ]
    );
    assert!(monitor.load_state.snapshot().is_empty());
}

/// Under the conditional poll gate the monitor asks its sink whether a
/// reader wants loads, and takes the answer.
#[tokio::test]
async fn the_conditional_gate_asks_the_sink_whether_loads_are_wanted() {
    let worker = ready_worker("http://w:1", "m");
    for needs in [false, true] {
        let (_registry, monitor, _sink) =
            build_recording(PollMode::Poll, PolledRanks::default(), needs, true);
        assert_eq!(
            monitor.loads_wanted(std::slice::from_ref(&worker)),
            needs,
            "the sink says {needs}"
        );
    }
}

/// A sink that writes the seven calls with no default and keeps the port's
/// answer at the two that have one.
struct BareSink;

impl LoadSink for BareSink {
    fn record(&self, _: &Arc<dyn Worker>, _: i32, _: &WorkerLoadResponse, _: Instant, _: Instant) {}

    fn poll_mode(&self, _: &Arc<dyn Worker>, _: Instant, _: Duration) -> PollMode {
        PollMode::Poll
    }

    fn report(&self, _: &[PolledMember<'_>], _: Instant) -> PolledRanks {
        PolledRanks::default()
    }

    fn report_empty(&self, _: &[PolledMember<'_>], _: Instant) -> Vec<i32> {
        Vec::new()
    }

    fn fetch_failed(&self, _: &[Arc<dyn Worker>]) {}

    fn forget(&self, _: &Arc<dyn Worker>) {}

    fn reset(&self) {}
}

/// A sink that does not say whether its readers want loads is fed: the
/// port's own answer is that they do, so a collector told to fetch on
/// demand alone still polls for it.
#[test]
fn a_sink_that_keeps_the_ports_answer_is_fed_under_the_conditional_gate() {
    let monitor = WorkerMonitor::with_sink(
        Arc::new(WorkerRegistry::new()),
        Arc::new(BareSink),
        5,
        false,
        true,
    );
    let worker = ready_worker("http://w:1", "m");
    assert!(monitor.loads_wanted(std::slice::from_ref(&worker)));
}

/// The monitor running over a recording sink, with one Ready worker
/// registered and the event loop started.
fn recording_with_a_worker() -> (
    Arc<WorkerRegistry>,
    Arc<WorkerMonitor>,
    Arc<RecordingLoadSink>,
    crate::registry::WorkerId,
) {
    let (registry, monitor, sink) =
        build_recording(PollMode::Suppressed, PolledRanks::default(), true, false);
    let id = registry.register(ready_worker("http://w:1", "m")).unwrap();
    monitor.start_event_loop();
    (registry, monitor, sink, id)
}

/// Whether the sink has been told to forget the worker at `url`.
fn told_to_forget(sink: &RecordingLoadSink, url: &str) -> bool {
    sink.said
        .lock()
        .iter()
        .any(|said| matches!(said, Said::Forget(told, _) if told == url))
}

/// The event loop's turn: it runs on its own task.
async fn settle() {
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(50)).await;
}

/// Wait, up to two seconds, until `done` holds: what the event loop does
/// arrives on its own task.
async fn within_two_seconds(mut done: impl FnMut() -> bool) -> bool {
    for _ in 0..200 {
        if done() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    done()
}

/// A worker's removal forgets its loads at the sink.
#[tokio::test]
async fn a_removal_forgets_the_workers_loads_at_the_sink() {
    let (registry, _monitor, sink, id) = recording_with_a_worker();
    settle().await;
    assert!(!told_to_forget(&sink, "http://w:1"));
    registry.remove(&id);
    assert!(within_two_seconds(|| told_to_forget(&sink, "http://w:1")).await);
}

/// A worker's replacement forgets the old worker's loads at the sink.
#[tokio::test]
async fn a_replacement_forgets_the_old_workers_loads_at_the_sink() {
    let (registry, _monitor, sink, id) = recording_with_a_worker();
    settle().await;
    assert!(!told_to_forget(&sink, "http://w:1"));
    assert!(registry.replace(&id, ready_worker("http://w:1", "m")));
    assert!(within_two_seconds(|| told_to_forget(&sink, "http://w:1")).await);
}

/// A status that leaves `Ready` forgets the worker's loads at the sink.
#[tokio::test]
async fn a_status_that_leaves_ready_forgets_the_workers_loads_at_the_sink() {
    let (registry, _monitor, sink, id) = recording_with_a_worker();
    settle().await;
    assert!(!told_to_forget(&sink, "http://w:1"));
    registry.transition_status(&id, WorkerStatus::NotReady);
    assert!(within_two_seconds(|| told_to_forget(&sink, "http://w:1")).await);
}

/// A Ready worker under overload thresholds: eight waiting requests, or a
/// token usage of 0.8.
fn overload_worker(url: &str) -> Arc<dyn Worker> {
    overload_member(url, None)
}

/// [`overload_worker`]. With a `rank` it builds that rank worker of a
/// data-parallel engine of two at `url`.
fn overload_member(url: &str, rank: Option<usize>) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .model(ModelCard::new("m"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .overload_defaults(OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.8),
        });
    if let Some(rank) = rank {
        builder = builder.dp_config(rank, 2);
    }
    let worker = builder.build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// A worker whose status leaves `Ready` has no load feed and so no verdict:
/// the monitor's own sink clears the overload flag, which nothing else
/// clears on that path. A record comes before the event loop starts, so the
/// first tick finds a fresh receipt and polls nobody, and no failed fetch
/// clears the flag in the eviction's place. The loop's start resets every
/// flag, so the record that sets this one comes after it.
#[tokio::test]
async fn a_status_that_leaves_ready_clears_the_overload_flag_at_the_monitors_own_sink() {
    let (registry, monitor) = build_monitor();
    let worker = overload_worker("http://w:1");
    let id = registry.register(Arc::clone(&worker)).unwrap();
    let record = pushed(1).with_requests(1, 8);
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    monitor.start_event_loop();
    settle().await;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert!(worker.is_overloaded(), "eight waiting is the threshold");
    registry.transition_status(&id, WorkerStatus::NotReady);
    assert!(
        within_two_seconds(|| !worker.is_overloaded()).await,
        "no load feed, no verdict"
    );
}

/// The collector's reset crosses the port: the sink is told, and the
/// collector itself clears no overload flag. The flag here was set by
/// another hand and the recording sink writes nothing, so it still stands.
#[test]
fn a_reset_is_said_to_the_sink_and_the_collector_clears_no_flag_itself() {
    let (registry, monitor, sink) =
        build_recording(PollMode::Suppressed, PolledRanks::default(), true, false);
    let worker = overload_worker("http://w:1");
    registry.register(Arc::clone(&worker)).unwrap();
    registry.set_worker_overloaded(&worker, true);

    monitor.stop_all_groups();

    assert_eq!(*sink.said.lock(), vec![Said::Reset]);
    assert!(worker.is_overloaded(), "the flag is the sink's to clear");
}

/// At a reset the monitor's own sink fails every overload veto open and
/// says so, under the monitor's log target as before the port.
#[test]
#[tracing_test::traced_test]
fn the_monitors_own_sink_fails_the_vetoes_open_at_a_reset_and_says_so() {
    let (registry, monitor) = build_monitor();
    let worker = overload_worker("http://w:1");
    registry.register(Arc::clone(&worker)).unwrap();
    registry.set_worker_overloaded(&worker, true);

    monitor.stop_all_groups();

    assert!(!worker.is_overloaded());
    assert!(logs_contain(
        "smg_worker::load::monitor: Overload vetoes cleared with the load feed"
    ));
}

/// The receipts of a pusher that stays outlive a reset, with its pushed
/// report: the collector stops its polls, not the workers' streams. A
/// worker whose stream pushed within the interval is still not polled.
#[tokio::test]
async fn a_reset_keeps_the_receipts_at_the_monitors_own_sink() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&worker)).unwrap();
    let now = Instant::now();
    monitor.apply_pushed_load(&worker, 0, &pushed(1), now);

    monitor.stop_all_groups();

    assert!(monitor.pushed_reports.lock().contains_key("http://w:1"));
    assert_eq!(
        monitor.poll_mode("http://w:1", now, Duration::from_secs(5)),
        PollMode::Suppressed
    );
}

/// A dropped collector says nothing to its sink: its drop aborts its tasks
/// and makes no call. A sink's code inside a drop would run on whatever
/// thread lets the last handle go, under the locks that thread holds. The
/// host that drops a collector owns the sink too: it voids its store when
/// it chooses, or it starts the next collector over the same sink, whose
/// start says the reset.
#[tokio::test]
async fn a_dropped_collector_says_nothing_to_its_sink() {
    let (_registry, monitor, sink, _id) = recording_with_a_worker();
    settle().await;
    let said_before = sink.said.lock().clone();
    assert_eq!(
        said_before.first(),
        Some(&Said::Reset),
        "the start says the reset"
    );

    drop(monitor);
    settle().await;

    assert!(
        sink.monitor.get().and_then(Weak::upgrade).is_none(),
        "the collector is gone"
    );
    assert_eq!(*sink.said.lock(), said_before, "the drop says nothing");
}

/// Register `worker` and put it under the overload veto by another hand.
fn veto(registry: &WorkerRegistry, worker: &Arc<dyn Worker>) {
    registry.register(Arc::clone(worker)).unwrap();
    assert!(registry.set_worker_overloaded(worker, true));
}

/// Absent means no opinion. An engine whose fetch failed for every member
/// brought none, and the monitor's own sink clears the overload veto: a
/// worker whose feed went away is not left out of routing.
#[test]
fn a_fetch_that_failed_whole_clears_the_veto_at_the_monitors_own_sink() {
    let (registry, monitor) = build_monitor();
    let worker = overload_worker("http://w:1");
    veto(&registry, &worker);

    monitor.reconcile_engine(&mut [(Arc::clone(&worker), None)], Instant::now());

    assert!(!worker.is_overloaded());
}

/// An answer with no row is no opinion either: the veto is cleared.
#[test]
fn an_answer_with_no_row_clears_the_veto_at_the_monitors_own_sink() {
    let (registry, monitor) = build_monitor();
    let worker = overload_worker("http://w:1");
    veto(&registry, &worker);
    let no_row = Some(WorkerLoadResponse::default());

    monitor.reconcile_engine(&mut [(Arc::clone(&worker), no_row)], Instant::now());

    assert!(!worker.is_overloaded());
}

/// A rank worker whose own fetch failed beside one that answered has no
/// opinion of its own: its veto is cleared, whatever the other's rows say.
#[test]
fn a_members_failed_fetch_beside_an_answer_clears_its_veto_at_the_monitors_own_sink() {
    let (registry, monitor) = build_monitor();
    let failed = overload_member("http://w:1", Some(0));
    let answered = overload_member("http://w:1", Some(1));
    veto(&registry, &failed);
    registry.register(Arc::clone(&answered)).unwrap();
    let mut engine = [
        (Arc::clone(&failed), None),
        (answered, Some(answer_of(1, 1))),
    ];

    monitor.reconcile_engine(&mut engine, Instant::now());

    assert!(!failed.is_overloaded());
}

/// A registered worker whose stream pushed a record of rank 0 and one of
/// rank 1, under the monitor's own sink and with no event loop: no registry
/// event reaches the monitor, as when the events are lost.
fn a_pusher_no_event_reaches() -> (
    Arc<WorkerRegistry>,
    Arc<WorkerMonitor>,
    crate::registry::WorkerId,
) {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("http://w:1", "m");
    let id = registry.register(Arc::clone(&worker)).unwrap();
    for rank in [0, 1] {
        monitor.apply_pushed_load(&worker, rank, &pushed(1), Instant::now());
    }
    (registry, monitor, id)
}

/// Whether the worker at `http://w:1` has a pushed report on file, and a
/// receipt of rank 1.
fn on_file(monitor: &WorkerMonitor) -> (bool, bool) {
    (
        monitor.pushed_reports.lock().contains_key("http://w:1"),
        monitor.has_receipt("http://w:1", 1),
    )
}

/// A pusher removed while registry events were lost was never evicted. The
/// reset evicts it: its pushed report and its receipts go, which nothing
/// else drops.
#[tokio::test]
async fn a_reset_evicts_a_pusher_that_left_while_events_were_lost() {
    let (registry, monitor, id) = a_pusher_no_event_reaches();
    registry.remove(&id);
    assert_eq!(on_file(&monitor), (true, true), "no event evicted it");

    monitor.stop_all_groups();

    assert_eq!(on_file(&monitor), (false, false));
}

/// A pusher whose status left `Ready` while registry events were lost is
/// evicted at the reset, as its event would have.
#[tokio::test]
async fn a_reset_evicts_a_pusher_that_left_ready_while_events_were_lost() {
    let (registry, monitor, id) = a_pusher_no_event_reaches();
    registry.transition_status(&id, WorkerStatus::NotReady);

    monitor.stop_all_groups();

    assert_eq!(on_file(&monitor), (false, false));
}

/// A pusher replaced while registry events were lost is evicted at the
/// reset: the replacement is polled until its own stream pushes.
#[tokio::test]
async fn a_reset_evicts_a_pusher_that_was_replaced_while_events_were_lost() {
    let (registry, monitor, id) = a_pusher_no_event_reaches();
    assert!(registry.replace(&id, ready_worker("http://w:1", "m")));

    monitor.stop_all_groups();

    assert_eq!(on_file(&monitor), (false, false));
    assert_eq!(
        monitor.poll_mode("http://w:1", Instant::now(), Duration::from_secs(5)),
        PollMode::Poll
    );
}

/// A worker registered at the address of a pusher that left while registry
/// events were lost starts a report of its own after the reset: its first
/// record publishes no rank of its predecessor.
#[tokio::test]
async fn a_worker_at_a_lost_pushers_address_starts_a_report_of_its_own() {
    let (registry, monitor, id) = a_pusher_no_event_reaches();
    registry.remove(&id);
    let next = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&next)).unwrap();

    monitor.stop_all_groups();
    monitor.apply_pushed_load(&next, 0, &pushed(3), Instant::now());

    assert_eq!(
        rows(&monitor.pushed_reports.lock()["http://w:1"].report),
        vec![(0, 3)],
        "the predecessor's rank 1 is gone"
    );
    assert!(!monitor.has_receipt("http://w:1", 1));
}

/// A worker registered at a lost pusher's address that pushes before the
/// reset starts a report of its own with that record: the predecessor's
/// rank and receipts are gone at once, and the reset keeps the report.
#[tokio::test]
async fn a_record_ahead_of_the_reset_starts_the_successors_own_report() {
    let (registry, monitor, id) = a_pusher_no_event_reaches();
    registry.remove(&id);
    let next = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&next)).unwrap();

    monitor.apply_pushed_load(&next, 0, &pushed(3), Instant::now());

    assert_eq!(on_file(&monitor), (true, false), "no receipt of rank 1");
    assert_eq!(published_rows(&monitor), vec![(0, 3)]);
    monitor.stop_all_groups();
    assert_eq!(
        rows(&monitor.pushed_reports.lock()["http://w:1"].report),
        vec![(0, 3)],
        "the reset keeps the report of the registry's worker"
    );
}

/// A pusher of ranks 0 and 1 that was replaced, with its event still to
/// come: the monitor, the replaced worker and its replacement.
fn a_replaced_pusher_ahead_of_its_event() -> (Arc<WorkerMonitor>, Arc<dyn Worker>, Arc<dyn Worker>)
{
    let (registry, monitor, id) = a_pusher_no_event_reaches();
    let replaced = registry.get(&id).unwrap();
    let next = ready_worker("http://w:1", "m");
    assert!(registry.replace(&id, Arc::clone(&next)));
    (monitor, replaced, next)
}

/// A successor's stream can push before the monitor handles its
/// predecessor's event. Its first record starts a report of its own: it
/// holds and publishes its own rank alone, and the predecessor's receipts
/// are gone.
#[tokio::test]
async fn a_successors_record_ahead_of_its_predecessors_event_starts_its_own_report() {
    let (monitor, _replaced, next) = a_replaced_pusher_ahead_of_its_event();

    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());

    assert_eq!(on_file(&monitor), (true, false), "no receipt of rank 1");
    assert_eq!(published_rows(&monitor), vec![(0, 9)]);
}

/// A record of the replaced worker that lands behind its successor's
/// report is dropped: a report holds its pusher's records alone.
#[tokio::test]
async fn a_replaced_workers_record_behind_its_successors_report_is_dropped() {
    let (monitor, replaced, next) = a_replaced_pusher_ahead_of_its_event();
    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());

    monitor.apply_pushed_load(&replaced, 1, &pushed(4), Instant::now());

    assert_eq!(
        rows(&monitor.pushed_reports.lock()["http://w:1"].report),
        vec![(0, 9)]
    );
    assert!(!monitor.has_receipt("http://w:1", 1));
}

/// The sink is told to forget the worker whose report a successor's record
/// replaces: that worker and no other, ahead of that record and under the
/// same hold of the reports. The replaced worker's own eviction comes
/// later and says nothing more: the URL holds its successor's report.
#[tokio::test]
async fn the_sink_forgets_a_replaced_pusher_ahead_of_its_successors_record() {
    let (registry, monitor, sink) =
        build_recording(PollMode::Suppressed, PolledRanks::default(), true, false);
    let first = ready_worker("http://w:1", "m");
    let id = registry.register(Arc::clone(&first)).unwrap();
    monitor.apply_pushed_load(&first, 1, &pushed(1), Instant::now());
    let next = ready_worker("http://w:1", "m");
    assert!(registry.replace(&id, Arc::clone(&next)));
    sink.said.lock().clear();

    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());

    let said = sink.said.lock();
    assert_eq!(said.len(), 2, "{said:?}");
    assert_eq!(said[0], Said::Forget("http://w:1".to_string(), true));
    assert!(
        Arc::ptr_eq(&sink.forgotten.lock()[0], &first),
        "the replaced pusher, not its successor"
    );
    assert!(
        matches!(
            said[1],
            Said::Record {
                rank: 0,
                rows: 1,
                ..
            }
        ),
        "{said:?}"
    );
    drop(said);

    monitor.evict_worker_loads(&first);

    assert_eq!(sink.said.lock().len(), 2, "nothing at the late eviction");
}

/// A replaced worker's event can come behind its successor's first record.
/// Its eviction takes what is its own: the report at the URL is its
/// successor's and stays, with its receipt and its published row.
#[tokio::test]
async fn a_replaced_workers_late_eviction_leaves_its_successors_report() {
    let (monitor, replaced, next) = a_replaced_pusher_ahead_of_its_event();
    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());

    monitor.evict_worker_loads(&replaced);

    assert!(monitor.pushed_reports.lock().contains_key("http://w:1"));
    assert!(monitor.has_receipt("http://w:1", 0), "its receipt");
    assert_eq!(published_rows(&monitor), vec![(0, 9)]);
}

/// The monitor's own sink with a stop inside `forget`: a case holds the
/// collector there while another thread pushes a record.
struct StoppedAtForget {
    classic: ClassicLoadSink<crate::test_support::Members>,
    entered: mpsc::Sender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}

impl LoadSink for StoppedAtForget {
    fn record(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        received_at: Instant,
        sampled_at: Instant,
    ) {
        self.classic
            .record(worker, rank, report, received_at, sampled_at);
    }

    fn poll_mode(&self, worker: &Arc<dyn Worker>, now: Instant, interval: Duration) -> PollMode {
        self.classic.poll_mode(worker, now, interval)
    }

    fn report(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> PolledRanks {
        self.classic.report(answer, polled_at)
    }

    fn report_empty(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> Vec<i32> {
        self.classic.report_empty(answer, polled_at)
    }

    fn fetch_failed(&self, members: &[Arc<dyn Worker>]) {
        self.classic.fetch_failed(members);
    }

    fn forget(&self, worker: &Arc<dyn Worker>) {
        self.entered.send(()).unwrap();
        self.resume
            .lock()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        self.classic.forget(worker);
    }

    fn reset(&self) {
        self.classic.reset();
    }
}

/// A reset's eviction and the first record of the next worker at the URL
/// can cross. The sink forgets under the lock of the reports, so the
/// record waits for it and its receipt is kept: no poll runs beside the
/// next worker's stream.
#[tokio::test]
async fn a_record_that_crosses_a_resets_eviction_keeps_its_receipt() {
    let (entered, in_forget) = mpsc::channel();
    let (resume, at_resume) = mpsc::channel();
    let (recorded, record_done) = mpsc::channel();
    let registry = Arc::new(WorkerRegistry::new());
    let sink = Arc::new(StoppedAtForget {
        classic: ClassicLoadSink::new(
            Arc::clone(&registry),
            Arc::new(LoadReaders::Blind),
            Arc::new(WorkerLoadManager::new()),
        ),
        entered,
        resume: Mutex::new(at_resume),
    });
    let monitor = Arc::new(WorkerMonitor::with_sink(
        Arc::clone(&registry),
        sink.clone(),
        5,
        false,
        false,
    ));
    let lost = ready_worker("http://w:1", "m");
    let id = registry.register(Arc::clone(&lost)).unwrap();
    monitor.apply_pushed_load(&lost, 1, &pushed(4), Instant::now());
    registry.remove(&id);
    let next = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&next)).unwrap();
    let now = Instant::now();
    let runtime = tokio::runtime::Handle::current();

    let waited = std::thread::scope(|scope| {
        scope.spawn(|| monitor.stop_all_groups());
        in_forget.recv_timeout(Duration::from_secs(5)).unwrap();
        scope.spawn(|| {
            let _context = runtime.enter();
            monitor.apply_pushed_load(&next, 0, &pushed(9), now);
            recorded.send(()).unwrap();
        });
        // A record that does not wait for the sink is done by now.
        let waited = record_done
            .recv_timeout(Duration::from_millis(100))
            .is_err();
        resume.send(()).unwrap();
        waited
    });

    assert!(waited, "the record waits for the sink to forget");
    assert_eq!(
        sink.poll_mode(&next, now, Duration::from_secs(5)),
        PollMode::Suppressed,
        "the next worker's receipt"
    );
}

/// A removed worker's late record that lands behind the next worker's
/// report is dropped: the next flush still publishes the next worker's
/// rank, and its predecessor's rank never joins it.
#[tokio::test]
async fn a_removed_workers_late_record_behind_the_next_workers_report_is_dropped() {
    let (registry, monitor, gone) = a_pusher_its_event_evicted();
    let next = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&next)).unwrap();
    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());

    monitor.apply_pushed_load(&gone, 1, &pushed(4), Instant::now());

    assert_eq!(published_rows(&monitor), vec![(0, 9)], "its own rank");
    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());
    assert_eq!(published_rows(&monitor), vec![(0, 9)]);
}

/// A pusher that was removed and that its event evicted: nothing of it is
/// on file. The removed worker is handed back, as its stream holds it.
fn a_pusher_its_event_evicted() -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>, Arc<dyn Worker>) {
    let (registry, monitor, id) = a_pusher_no_event_reaches();
    monitor.flush_pushed();
    let gone = registry.remove(&id).unwrap();
    monitor.evict_worker_loads(&gone);
    monitor.flush_pending_evictions();
    (registry, monitor, gone)
}

/// Whether anything still holds `worker`, which is let go here.
fn held(worker: Arc<dyn Worker>) -> bool {
    let weak = Arc::downgrade(&worker);
    drop(worker);
    weak.upgrade().is_some()
}

/// The rows the worker at `http://w:1` publishes at the next flush.
fn published_rows(monitor: &Arc<WorkerMonitor>) -> Vec<(i32, i32)> {
    monitor.flush_pushed();
    rows(monitor.load_state.snapshot().get("http://w:1").unwrap())
}

/// A host ends a worker's stream after the registry let the worker go, so
/// a record can land behind the worker's eviction. It starts no report:
/// nothing holds the worker that is gone, and the next worker at the
/// address publishes its own rank alone.
#[tokio::test]
async fn a_record_behind_its_workers_eviction_starts_no_report() {
    let (registry, monitor, gone) = a_pusher_its_event_evicted();

    monitor.apply_pushed_load(&gone, 1, &pushed(4), Instant::now());

    assert_eq!(on_file(&monitor), (false, false));
    assert!(!held(gone), "nothing holds the worker");
    let next = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&next)).unwrap();
    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());
    assert_eq!(published_rows(&monitor), vec![(0, 9)]);
}

/// The start test reads the worker, not its address alone: with the next
/// worker registered there, a record of the removed worker still starts no
/// report, nothing holds the removed worker, and the next worker publishes
/// its own rank alone.
#[tokio::test]
async fn a_removed_workers_record_behind_the_next_registration_starts_no_report() {
    let (registry, monitor, gone) = a_pusher_its_event_evicted();
    let next = ready_worker("http://w:1", "m");
    registry.register(Arc::clone(&next)).unwrap();

    monitor.apply_pushed_load(&gone, 1, &pushed(4), Instant::now());

    assert_eq!(on_file(&monitor), (false, false));
    assert!(!held(gone), "nothing holds the removed worker");
    monitor.apply_pushed_load(&next, 0, &pushed(9), Instant::now());
    assert_eq!(published_rows(&monitor), vec![(0, 9)]);
}

/// A record that is dropped still marks its worker as one whose stream
/// pushes, as any record does: a worker the registry does not hold starts
/// no report, and nothing of it reaches the sink.
#[tokio::test]
async fn a_dropped_record_still_marks_its_worker_as_one_whose_stream_pushes() {
    let (_registry, monitor, sink) =
        build_recording(PollMode::Poll, PolledRanks::default(), true, false);
    let worker = ready_worker("http://w:1", "m");

    monitor.apply_pushed_load(&worker, 0, &pushed(1), Instant::now());

    assert!(monitor.pushed_reports.lock().is_empty());
    assert!(sink.said.lock().is_empty(), "nothing reaches the sink");
    assert!(
        worker
            .runtime_state()
            .is_some_and(|runtime| runtime.pushes_load_records()),
        "its worker is marked as one whose stream pushes"
    );
}

/// A report is on file for the registry's worker at its URL, `Ready` or
/// not: the record of a registered worker that is not `Ready` starts its
/// report and reaches the sink, and the publish fence holds its rows back.
/// Its next record once it is `Ready` publishes them.
#[tokio::test]
async fn a_record_of_a_worker_that_is_not_ready_is_filed_and_not_published() {
    let (registry, monitor, sink) =
        build_recording(PollMode::Poll, PolledRanks::default(), true, false);
    let worker = ready_worker("http://w:1", "m");
    let id = registry.register(Arc::clone(&worker)).unwrap();
    registry.transition_status(&id, WorkerStatus::NotReady);

    monitor.apply_pushed_load(&worker, 0, &pushed(1), Instant::now());

    assert!(
        matches!(sink.said.lock().as_slice(), [Said::Record { rank: 0, .. }]),
        "the record reaches the sink"
    );
    assert_eq!(
        rows(&monitor.pushed_reports.lock()["http://w:1"].report),
        vec![(0, 1)]
    );
    monitor.flush_pushed();
    assert!(
        monitor.load_state.snapshot().get("http://w:1").is_none(),
        "nothing is published"
    );

    registry.transition_status(&id, WorkerStatus::Ready);
    monitor.apply_pushed_load(&worker, 0, &pushed(1), Instant::now());

    assert_eq!(published_rows(&monitor), vec![(0, 1)]);
}

/// A worker the wedged rule vetoed can leave `Ready` and produce again
/// before health takes it back. Its record is heard while the registry
/// holds it, whatever its status: the veto is lifted, and nothing keeps
/// the worker out of routing once it is `Ready`.
#[tokio::test]
async fn a_record_of_a_worker_that_is_not_ready_lifts_its_wedge_veto() {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    let id = registry.register(Arc::clone(&worker)).unwrap();
    let mut record = pushed(1);
    record.engine_silence = Some(Duration::from_millis(5_000));
    record.reports_every_step = true;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    assert_eq!(worker.stall_reason(), Some(StallReason::Wedged));
    registry.transition_status(&id, WorkerStatus::NotReady);
    monitor.evict_worker_loads(&worker);

    record.engine_silence = None;
    monitor.apply_pushed_load(&worker, 0, &record, Instant::now());
    registry.transition_status(&id, WorkerStatus::Ready);

    assert!(worker.stall_reason().is_none(), "the engine produces again");
}

/// A pusher that left `Ready` and came back, with the eviction its leave
/// queued still to be applied: it reported 7 running before it left and 9
/// since its return, and published both.
fn a_pusher_back_ahead_of_its_queued_eviction() -> Arc<WorkerMonitor> {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("http://w:1", "m");
    let id = registry.register(Arc::clone(&worker)).unwrap();
    monitor.apply_pushed_load(&worker, 0, &pushed(7), Instant::now());
    monitor.flush_pushed();
    registry.transition_status(&id, WorkerStatus::NotReady);
    monitor.evict_worker_loads(&worker);
    registry.transition_status(&id, WorkerStatus::Ready);
    monitor.apply_pushed_load(&worker, 0, &pushed(9), Instant::now());
    monitor.flush_pushed();
    monitor
}

/// The running-requests series of the worker at `http://w:1`, as rendered.
fn running_series(rendered: &str) -> &str {
    rendered
        .lines()
        .find(|l| l.starts_with("smg_engine_running_requests{") && l.contains("http://w:1"))
        .unwrap_or_else(|| panic!("no running series:\n{rendered}"))
}

/// An eviction is for what its worker published before it was queued. The
/// flush that applies the queue leaves the row a worker published since
/// its return, and writes no sentinel over its series.
#[tokio::test]
async fn a_queued_eviction_leaves_the_row_its_worker_published_since() {
    let rendered = render_with_recorder(|| {
        let monitor = a_pusher_back_ahead_of_its_queued_eviction();

        monitor.flush_pending_evictions();

        assert_eq!(published_rows(&monitor), vec![(0, 9)]);
    });
    assert!(running_series(&rendered).ends_with(" 9"), "{rendered}");
}

/// A reset applies the queue too: a worker that came back keeps the series
/// of what it reported since, and its receipt.
#[tokio::test]
async fn a_reset_leaves_the_series_a_returned_worker_reported_since() {
    let rendered = render_with_recorder(|| {
        let monitor = a_pusher_back_ahead_of_its_queued_eviction();

        monitor.stop_all_groups();

        assert!(monitor.has_receipt("http://w:1", 0));
    });
    assert!(running_series(&rendered).ends_with(" 9"), "{rendered}");
}

/// The series is held back with the rows. A record of a worker that is not
/// `Ready` leaves the sentinel its eviction wrote: a stream that ends
/// there leaves no load that reads as live.
#[tokio::test]
async fn a_record_of_a_worker_that_is_not_ready_leaves_the_sentinel_on_its_series() {
    let rendered = render_with_recorder(|| {
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:1", "m");
        let id = registry.register(Arc::clone(&worker)).unwrap();
        monitor.apply_pushed_load(&worker, 0, &pushed(7), Instant::now());
        monitor.flush_pushed();
        registry.transition_status(&id, WorkerStatus::NotReady);
        monitor.evict_worker_loads(&worker);
        monitor.flush_pending_evictions();

        monitor.apply_pushed_load(&worker, 0, &pushed(9), Instant::now());
    });
    assert!(running_series(&rendered).ends_with(" -1"), "{rendered}");
}

/// A worker that left `Ready` and still pushes starts its report again,
/// and the fence refuses its publish. That publish prunes nothing: the row
/// from before the leave is the eviction's to remove, and the eviction
/// writes the sentinel of the series with it.
#[tokio::test]
async fn a_refused_publish_leaves_the_old_row_to_the_eviction() {
    let rendered = render_with_recorder(|| {
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:1", "m");
        let id = registry.register(Arc::clone(&worker)).unwrap();
        monitor.apply_pushed_load(&worker, 0, &pushed(7), Instant::now());
        monitor.flush_pushed();
        registry.transition_status(&id, WorkerStatus::NotReady);
        monitor.evict_worker_loads(&worker);
        monitor.apply_pushed_load(&worker, 0, &pushed(9), Instant::now());

        assert_eq!(published_rows(&monitor), vec![(0, 7)], "the old row");
        monitor.flush_pending_evictions();

        assert!(monitor.load_state.snapshot().get("http://w:1").is_none());
    });
    assert!(running_series(&rendered).ends_with(" -1"), "{rendered}");
}

/// A queued eviction can be applied between a returned worker's first
/// record and its publish. It evicts nothing and writes no sentinel: the
/// worker is the registry's `Ready` worker again, and its series reads the
/// record made since.
#[tokio::test]
async fn a_batch_ahead_of_a_returned_workers_publish_writes_no_sentinel() {
    let rendered = render_with_recorder(|| {
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:1", "m");
        let id = registry.register(Arc::clone(&worker)).unwrap();
        monitor.apply_pushed_load(&worker, 0, &pushed(7), Instant::now());
        monitor.flush_pushed();
        registry.transition_status(&id, WorkerStatus::NotReady);
        monitor.evict_worker_loads(&worker);
        registry.transition_status(&id, WorkerStatus::Ready);
        monitor.apply_pushed_load(&worker, 0, &pushed(9), Instant::now());

        monitor.flush_pending_evictions();

        assert_eq!(published_rows(&monitor), vec![(0, 9)]);
    });
    assert!(running_series(&rendered).ends_with(" 9"), "{rendered}");
}

/// A leave can be handled behind the worker's return. A worker that is the
/// registry's `Ready` worker again keeps its loads: the sink hears no
/// `forget`.
#[tokio::test]
async fn a_leave_handled_behind_the_workers_return_forgets_nothing() {
    let (registry, _monitor, sink, id) = recording_with_a_worker();
    settle().await;
    registry.transition_status(&id, WorkerStatus::NotReady);
    registry.transition_status(&id, WorkerStatus::Ready);

    settle().await;

    assert!(!told_to_forget(&sink, "http://w:1"));
}

/// A worker can leave `Ready`, come back ahead of the batch and leave
/// again before it reports. The batch evicts no worker that is `Ready`
/// again, so the row from before the first leave stands for the second
/// eviction, which removes it and writes the sentinel of the series.
#[tokio::test]
async fn a_second_leave_ahead_of_any_report_still_writes_the_sentinel() {
    let rendered = render_with_recorder(|| {
        let (registry, monitor) = build_monitor();
        let worker = ready_worker("http://w:1", "m");
        let id = registry.register(Arc::clone(&worker)).unwrap();
        monitor.apply_pushed_load(&worker, 0, &pushed(7), Instant::now());
        monitor.flush_pushed();
        registry.transition_status(&id, WorkerStatus::NotReady);
        monitor.evict_worker_loads(&worker);
        registry.transition_status(&id, WorkerStatus::Ready);
        monitor.flush_pending_evictions();
        assert_eq!(published_rows(&monitor), vec![(0, 7)], "not evicted");

        registry.transition_status(&id, WorkerStatus::NotReady);
        monitor.evict_worker_loads(&worker);
        monitor.flush_pending_evictions();

        assert!(monitor.load_state.snapshot().get("http://w:1").is_none());
    });
    assert!(running_series(&rendered).ends_with(" -1"), "{rendered}");
}

/// A pusher that left `Ready` while registry events were lost stays
/// registered, so its engine series would read its last load as live. The
/// reset's eviction writes the series' sentinel, as its event's would.
#[test]
fn a_reset_writes_the_sentinel_of_a_lost_pushers_engine_series() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://w1:9000", "llama-3");
    let id = registry.register(Arc::clone(&worker)).unwrap();
    let rendered = render_with_recorder(|| {
        runtime.block_on(async {
            monitor.apply_pushed_load(&worker, 0, &pushed(7), Instant::now());
            monitor.flush_pushed();
            registry.transition_status(&id, WorkerStatus::NotReady);
            monitor.stop_all_groups();
        });
    });
    let running = rendered
        .lines()
        .find(|l| l.starts_with("smg_engine_running_requests{") && l.contains("grpc://w1:9000"))
        .unwrap_or_else(|| panic!("no running series:\n{rendered}"));
    assert!(running.ends_with(" -1"), "{rendered}");
}

/// At a reset the collector says `forget` for a pusher an event it lost
/// would have evicted, under the lock of the reports, and then the reset.
#[tokio::test]
async fn a_reset_says_forget_for_a_lost_pusher_and_then_the_reset() {
    let (registry, monitor, sink) =
        build_recording(PollMode::Suppressed, PolledRanks::default(), true, false);
    let worker = ready_worker("http://w:1", "m");
    let id = registry.register(Arc::clone(&worker)).unwrap();
    monitor.apply_pushed_load(&worker, 0, &pushed(1), Instant::now());
    registry.remove(&id);
    sink.said.lock().clear();

    monitor.stop_all_groups();

    assert_eq!(
        *sink.said.lock(),
        vec![Said::Forget("http://w:1".to_string(), true), Said::Reset]
    );
}
