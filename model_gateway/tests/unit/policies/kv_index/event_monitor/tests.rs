use std::time::Duration;

use kv_index::{ContentHash, SequenceHash, StoredBlock};
use openai_protocol::{
    worker::{ConnectionMode, HealthCheckConfig, RuntimeType, WorkerSpec, WorkerType},
    UNKNOWN_MODEL_ID,
};
use smg_worker::kv_events::{MAX_CONCURRENT_SINK_REMOVALS, SINK_REMOVAL_PERMITS};
use tracing_test::traced_test;

use super::*;
use crate::{
    policies::kv_index::index_backend::WorkerBlocks,
    routers::grpc::backend_client::BasicWorkerBuilder,
};

async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    for _ in 0..200 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting until {what}");
}

/// The cases' worker address: a closed port, so a task loops on
/// reconnects and answers shutdown with no engine involved.
const REFUSING_URL: &str = "grpc://127.0.0.1:2";

/// A gRPC worker at [`REFUSING_URL`].
fn refusing_grpc_worker() -> Arc<dyn Worker> {
    refusing_grpc_worker_at(REFUSING_URL)
}

/// A gRPC worker at `url`, a closed port.
fn refusing_grpc_worker_at(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

/// A worker removed and added again under the same URL while the old
/// subscription's cleanup is still running: the index hands the URL its
/// old id until that cleanup has released it, so the re-add must wait
/// for the removal to finish rather than intern into the old id and
/// lose its blocks and its id to the cleanup.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_same_url_re_add_waits_for_the_old_subscriptions_cleanup() {
    // The chain index releases a removed worker's id and name, which is
    // what makes the window observable.
    let monitor = Arc::new(KvEventMonitor::new());
    let worker = refusing_grpc_worker();
    let url = worker.url().to_string();
    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    let index = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the model's index");
    wait_until("the subscription interns its worker", || {
        index.worker_id(&url).is_some()
    })
    .await;

    // Hold every cleanup permit: the old task's cleanup cannot finish.
    let permits = SINK_REMOVAL_PERMITS
        .acquire_many(MAX_CONCURRENT_SINK_REMOVALS as u32)
        .await
        .expect("the removal semaphore is open");
    #[expect(
        clippy::disallowed_methods,
        reason = "the removal and the re-add are awaited below"
    )]
    let removal = tokio::spawn({
        let monitor = Arc::clone(&monitor);
        let url = url.clone();
        async move { monitor.on_worker_removed(&url).await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!removal.is_finished(), "the removal waits for the cleanup");
    #[expect(
        clippy::disallowed_methods,
        reason = "the removal and the re-add are awaited below"
    )]
    let re_add = tokio::spawn({
        let monitor = Arc::clone(&monitor);
        let worker = Arc::clone(&worker);
        async move {
            monitor
                .on_worker_added(&worker, SubscriptionKind::Events)
                .await;
        }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !re_add.is_finished(),
        "the re-add waits for the old subscription's cleanup"
    );

    drop(permits);
    removal.await.expect("removal task");
    assert!(
        index.worker_id(&url).is_none(),
        "the removal returns once the old subscription has released its id"
    );
    re_add.await.expect("re-add task");
    // The worker was the model's last, so the removal dropped the model's
    // index and the re-add created a new one; the new subscription
    // interned the URL there, after the release, with an id of its own.
    let current = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the re-added worker's model has an index");
    wait_until("the new subscription interns its worker", || {
        current.worker_id(&url).is_some()
    })
    .await;
    monitor.stop().await;
    assert!(
        current.worker_id(&url).is_none(),
        "stop takes the new subscription out of the index"
    );
}

/// A caller may drop `on_worker_removed` before the old task has ended:
/// the registry's removal step runs under a timeout, and a cleanup
/// waiting on the removal permits can outlast it. The reservation the
/// removal left must not outlive the task: once the task is gone, the
/// model's last index is gone with it and an add of the URL goes
/// through, with nobody awaiting the removal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_removal_leaves_no_reservation_behind() {
    let monitor = Arc::new(KvEventMonitor::new());
    let worker = refusing_grpc_worker();
    let url = worker.url().to_string();
    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    let index = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the model's index");
    wait_until("the subscription interns its worker", || {
        index.worker_id(&url).is_some()
    })
    .await;

    // Hold every cleanup permit: the old task's cleanup cannot finish.
    let permits = SINK_REMOVAL_PERMITS
        .acquire_many(MAX_CONCURRENT_SINK_REMOVALS as u32)
        .await
        .expect("the removal semaphore is open");
    // The removal is dropped while it awaits the old task, as a timeout
    // around it would drop it.
    let removal =
        tokio::time::timeout(Duration::from_millis(200), monitor.on_worker_removed(&url)).await;
    assert!(
        removal.is_err(),
        "the removal was still waiting for the old task's cleanup"
    );
    // The cleanup is still held: a re-add waits, and nothing has moved.
    let re_add = tokio::time::timeout(
        Duration::from_millis(200),
        monitor.on_worker_added(&worker, SubscriptionKind::Events),
    )
    .await;
    assert!(
        re_add.is_err(),
        "the re-add waits while the old task's cleanup is held"
    );
    assert!(
        monitor.get_indexer(UNKNOWN_MODEL_ID).is_some(),
        "the model's index stays until the old task has finished"
    );

    drop(permits);
    // Nobody awaits the removal any more: the old task finishes it on
    // its own, releasing its id and dropping the model's last index.
    wait_until("the old task drops the model's index", || {
        monitor.get_indexer(UNKNOWN_MODEL_ID).is_none()
    })
    .await;
    assert!(
        index.worker_id(&url).is_none(),
        "the old task released its id"
    );
    tokio::time::timeout(
        Duration::from_secs(5),
        monitor.on_worker_added(&worker, SubscriptionKind::Events),
    )
    .await
    .expect("the re-add completes once the old task has ended");
    let current = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the re-added worker's model has an index");
    wait_until("the new subscription interns its worker", || {
        current.worker_id(&url).is_some()
    })
    .await;
    monitor.stop().await;
}

/// An add can find a reservation whose task is gone and that nobody
/// lifted. It lifts it before it starts the worker's stream: behind the
/// model's last events subscription the model's index is dropped and the
/// add makes a fresh one, and the worker's gauge is lowered.
#[test]
fn an_add_behind_a_dead_reservation_builds_the_model_a_fresh_index() {
    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

    fn gauge(handle: &PrometheusHandle, worker_url: &str) -> Option<f64> {
        let prefix = format!("smg_kv_events_unavailable{{worker=\"{worker_url}\"}}");
        handle
            .render()
            .lines()
            .find(|line| line.starts_with(&prefix))
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
    }

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    // The add publishes from the runtime's thread, which must be the one
    // holding the local recorder.
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let monitor = KvEventMonitor::new();
            let worker = refusing_grpc_worker_at("grpc://127.0.0.1:4");
            let url = worker.url();
            monitor
                .on_worker_added(&worker, SubscriptionKind::Events)
                .await;
            let before = monitor
                .get_indexer(UNKNOWN_MODEL_ID)
                .expect("the model's index");
            // Raised as an engine's refusal of the stream raises it.
            Metrics::set_kv_events_unavailable(url, true);
            assert_eq!(gauge(&handle, url), Some(1.0));
            let (stranded, _) = monitor.slot_ids(url).await;
            monitor.strand_reservation(url).await;
            assert_eq!(monitor.slot_ids(url).await, (None, stranded));

            monitor
                .on_worker_added(&worker, SubscriptionKind::Events)
                .await;
            let after = monitor
                .get_indexer(UNKNOWN_MODEL_ID)
                .expect("the re-added worker's model has an index");
            assert!(
                !Arc::ptr_eq(&before, &after),
                "the lift dropped the index of the model's last events subscription; the \
                     add made a fresh one"
            );
            let (active, removing) = monitor.slot_ids(url).await;
            assert!(active.is_some_and(|id| Some(id) != stranded));
            assert_eq!(removing, None, "the add lifted the dead reservation");
            assert_eq!(gauge(&handle, url), Some(0.0), "the lift lowers the gauge");
            monitor.stop().await;
        });
    });
}

/// The same add while another worker of the model still feeds the index:
/// the model keeps its index.
#[tokio::test]
async fn an_add_behind_a_dead_reservation_keeps_an_index_another_worker_feeds() {
    let monitor = KvEventMonitor::new();
    let stays = refusing_grpc_worker_at("grpc://127.0.0.1:5");
    let worker = refusing_grpc_worker_at("grpc://127.0.0.1:6");
    let url = worker.url();
    monitor
        .on_worker_added(&stays, SubscriptionKind::Events)
        .await;
    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    let before = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the model's index");
    monitor.strand_reservation(url).await;

    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    let (active, removing) = monitor.slot_ids(url).await;
    assert!(active.is_some(), "the add starts a stream of its own");
    assert_eq!(removing, None, "the add lifted the dead reservation");
    let after = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the model's index");
    assert!(
        Arc::ptr_eq(&before, &after),
        "another events worker of the model is active: its index stays"
    );
    monitor.stop().await;
}

#[test]
fn a_new_monitor_holds_no_index() {
    let monitor = KvEventMonitor::new();
    assert!(monitor.indexers.is_empty());
}

/// The `Events` restart takes the whole removal path: the old task
/// purges the URL's blocks and, as the model's only events subscription,
/// drops the model's index; the new subscription builds a fresh one and
/// interns its worker there.
#[traced_test]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_re_registration_of_an_events_worker_rebuilds_the_models_index() {
    let monitor = Arc::new(KvEventMonitor::new());
    let first = refusing_grpc_worker();
    let url = first.url().to_string();
    monitor
        .on_worker_added(&first, SubscriptionKind::Events)
        .await;
    let before = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the model's index");
    wait_until("the first subscription interns its worker", || {
        before.worker_id(&url).is_some()
    })
    .await;

    let second = refusing_grpc_worker();
    monitor
        .on_worker_added(&second, SubscriptionKind::Events)
        .await;
    assert!(logs_contain(
        "Restarting KV event subscription for a re-registered worker"
    ));
    let after = monitor
        .get_indexer(UNKNOWN_MODEL_ID)
        .expect("the re-added worker's model has an index");
    assert!(
        !Arc::ptr_eq(&before, &after),
        "the old task dropped the model's index with its last events subscription; the \
             restart built a fresh one"
    );
    assert!(
        before.worker_id(&url).is_none(),
        "the old index released the URL's id before it was dropped"
    );
    wait_until(
        "the new subscription interns its worker in the fresh index",
        || after.worker_id(&url).is_some(),
    )
    .await;
    monitor.stop().await;
}

#[tokio::test]
async fn an_unknown_model_has_no_index() {
    let monitor = KvEventMonitor::new();
    assert!(monitor.get_indexer("nonexistent").is_none());
}

/// Grand Central's reads of the monitor. A block that a worker's
/// event stored in a model's index is what the engine's lookup answers,
/// under the index's own number of the worker, and the model's block
/// size is the one the monitor holds. The index the engine is handed is
/// the monitor's own, not a copy. A model with no index answers none.
#[test]
fn the_engine_reads_the_index_an_event_wrote() {
    use kv_index::compute_request_content_hashes;
    use smg_grpc_client::common_proto::{
        kv_cache_event::Data, KvBlock, KvBlocksStored, KvCacheEvent, KvEventBatch,
    };

    let monitor = KvEventMonitor::new();
    let index = Arc::new(KvIndex::chain());
    monitor.set_index("llama", Arc::clone(&index));
    monitor.set_block_size("llama", 4);
    let tokens = [10, 20, 30, 40];
    let stored = KvBlocksStored {
        blocks: vec![KvBlock {
            block_hash: 1,
            token_ids: tokens.to_vec(),
            block_size: 4,
            ..Default::default()
        }],
        ..Default::default()
    };
    let batch = KvEventBatch {
        events: vec![KvCacheEvent {
            event_id: 1,
            data: Some(Data::Stored(stored)),
        }],
        ..Default::default()
    };
    let mut feed = bench_support::IndexFeed::new(&index, "grpc://w1:9000").unwrap();
    feed.apply(&index, &batch);

    let read: &dyn CacheIndexes = &monitor;
    assert_eq!(read.block_size("llama"), Some(4));
    let lent = read.index("llama").expect("the model's index");
    assert!(
        std::ptr::addr_eq(Arc::as_ptr(&lent), Arc::as_ptr(&index)),
        "the monitor's own index"
    );
    let number = feed.worker_id();
    assert_eq!(lent.worker_number("grpc://w1:9000"), Some(number));
    assert_eq!(lent.worker_blocks(number), 1);
    let hashes = compute_request_content_hashes(&tokens, 4);
    let request: Vec<u64> = hashes.iter().map(|hash| hash.0).collect();
    let mut kept = Vec::new();
    lent.deepest_into(&request, 4, &[1 << number], 16, &mut |_| 0, &mut kept);
    assert_eq!(kept, [(number, 1)]);

    assert!(read.index("nonexistent").is_none());
    assert_eq!(read.block_size("nonexistent"), None);
}

/// The engine is lent the model's index whenever the model has one,
/// and the index says itself that it holds no block: from the
/// enrolment of the model's first worker to its first event, and again
/// once the last holder left with its blocks. The engine reads such an
/// index as none, and its request is then decided by the token tree
/// the gateway lends. An index lent before the first event says the
/// event's block. The block size is answered whatever the index holds.
///
/// The classic policy reads an index that holds no block as no index
/// too (`event_index_for`,
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn the_engine_is_lent_an_index_that_says_whether_it_holds_a_block() {
    let monitor = KvEventMonitor::new();
    let index = Arc::new(KvIndex::chain());
    monitor.set_index("llama", Arc::clone(&index));
    monitor.set_block_size("llama", 4);
    let read: &dyn CacheIndexes = &monitor;
    let empty = read.index("llama").expect("the model's index");
    assert!(!empty.holds_a_block(), "no event yet");
    assert_eq!(read.block_size("llama"), Some(4));

    let worker = index.intern_worker("grpc://w1:9000").unwrap();
    let mut held = WorkerBlocks::default();
    let block = StoredBlock {
        seq_hash: SequenceHash(1),
        content_hash: ContentHash(101),
    };
    index
        .apply_stored(worker, &[block], None, &mut held)
        .unwrap();
    let lent = read.index("llama").expect("the index holds a block");
    assert!(
        std::ptr::addr_eq(Arc::as_ptr(&lent), Arc::as_ptr(&index)),
        "the monitor's own index"
    );
    assert!(lent.holds_a_block() && empty.holds_a_block());

    index.remove_worker(worker, held);
    assert!(!lent.holds_a_block(), "the last holder left");
    assert_eq!(read.block_size("llama"), Some(4));
}

#[test]
fn set_block_size_keeps_the_first_value() {
    let monitor = KvEventMonitor::new();

    // Initially no block_size
    assert!(monitor.block_size("llama").is_none());

    // Set it
    monitor.set_block_size("llama", 32);
    assert_eq!(monitor.block_size("llama"), Some(32));

    // set_block_size doesn't overwrite existing value
    monitor.set_block_size("llama", 64);
    assert_eq!(monitor.block_size("llama"), Some(32));
}

/// The add of an events worker seeds the model's block size from the
/// worker's spec, and the size leaves with the model's last events
/// subscription: the model's next worker seeds or learns its own.
#[tokio::test]
async fn the_block_size_comes_with_the_add_and_leaves_with_the_model() {
    let monitor = KvEventMonitor::new();
    let mut spec = WorkerSpec::new("grpc://127.0.0.1:7");
    spec.kv_block_size = Some(32);
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::from_spec(spec)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    assert_eq!(
        monitor.block_size(UNKNOWN_MODEL_ID),
        Some(32),
        "the add seeds the size from the spec"
    );
    monitor.on_worker_removed(worker.url()).await;
    assert!(
        monitor.block_size(UNKNOWN_MODEL_ID).is_none(),
        "the size left with the model's last events subscription"
    );
}

#[tokio::test]
async fn stop_forgets_the_block_sizes() {
    let monitor = KvEventMonitor::new();
    monitor.set_block_size("llama", 16);
    assert_eq!(monitor.block_size("llama"), Some(16));

    monitor.stop().await;
    assert!(monitor.block_size("llama").is_none());
}

/// The index-shape gauges come from the index's own counters, by model:
/// memberships and entries, runs, blocks, arena and slab bytes, moved
/// hashes and engine conflicts.
#[test]
fn index_shape_gauges_follow_the_index() {
    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

    fn gauge(handle: &PrometheusHandle, name: &str, model: &str) -> Option<f64> {
        let prefix = format!("{name}{{model=\"{model}\"}}");
        handle
            .render()
            .lines()
            .find(|line| line.starts_with(&prefix))
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
    }

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let monitor = KvEventMonitor::new();
        let index = Arc::new(KvIndex::chain());
        let worker = index.intern_worker("grpc://w1:9000").unwrap();
        let mut held = WorkerBlocks::default();
        let blocks: Vec<StoredBlock> = (1..=3u64)
            .map(|i| StoredBlock {
                seq_hash: SequenceHash(i),
                content_hash: ContentHash(100 + i),
            })
            .collect();
        index
            .apply_stored(worker, &blocks, None, &mut held)
            .unwrap();
        monitor.indexers.insert("chain".to_string(), index);
        KvEventMonitor::publish_stats(&monitor.indexers);
        assert_eq!(
            gauge(&handle, "smg_kv_index_memberships", "chain"),
            Some(3.0)
        );
        assert_eq!(gauge(&handle, "smg_kv_index_entries", "chain"), Some(3.0));
        assert_eq!(gauge(&handle, "smg_kv_index_runs_live", "chain"), Some(1.0));
        assert_eq!(
            gauge(&handle, "smg_kv_index_blocks_live", "chain"),
            Some(3.0)
        );
        assert_eq!(
            gauge(&handle, "smg_kv_index_moved_hashes", "chain"),
            Some(0.0)
        );
        assert_eq!(
            gauge(&handle, "smg_kv_index_engine_conflicts", "chain"),
            Some(0.0)
        );
        assert_eq!(
            gauge(&handle, "smg_kv_index_cap_refusals", "chain"),
            Some(0.0)
        );
        assert!(gauge(&handle, "smg_kv_index_arena_bytes", "chain").is_some_and(|b| b > 0.0));
        // The arena holds whole chunks: what it holds is at least what it handed out.
        let handed_out = gauge(&handle, "smg_kv_index_arena_bytes", "chain").expect("arena");
        assert!(gauge(&handle, "smg_kv_index_arena_chunk_bytes", "chain")
            .is_some_and(|held| held >= handed_out));
        assert!(gauge(&handle, "smg_kv_index_slab_bytes", "chain").is_some_and(|b| b > 0.0));
    });
}

/// A worker's integrity series and its model's index gauges exist from
/// the subscription's start, at zero: a scrape of a fresh gateway tells
/// "nothing happened" from "nothing exported".
#[test]
fn a_new_subscription_publishes_its_series_at_zero() {
    use metrics_exporter_prometheus::PrometheusBuilder;

    fn sample(body: &str, family: &str, labels: &[(&str, &str)]) -> Option<f64> {
        let prefix = format!("{family}{{");
        body.lines()
            .find(|line| {
                line.starts_with(&prefix)
                    && labels
                        .iter()
                        .all(|(key, value)| line.contains(&format!("{key}=\"{value}\"")))
            })
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
    }

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let monitor = KvEventMonitor::new();
            let worker = refusing_grpc_worker();
            monitor
                .on_worker_added(&worker, SubscriptionKind::Events)
                .await;
            let body = handle.render();
            let url = worker.url();
            for (family, labels) in [
                (
                    "smg_kv_event_gaps_total",
                    vec![("worker", url), ("outcome", "replay_requested")],
                ),
                (
                    "smg_kv_event_gaps_total",
                    vec![("worker", url), ("outcome", "unrecovered_cleared")],
                ),
                ("smg_kv_event_missed_batches_total", vec![("worker", url)]),
                (
                    "smg_kv_event_resyncs_total",
                    vec![("worker", url), ("reason", "snapshot")],
                ),
                (
                    "smg_kv_event_resyncs_total",
                    vec![("worker", url), ("reason", "publisher_restart")],
                ),
                (
                    "smg_kv_event_subscription_failures_total",
                    vec![("worker", url), ("reason", "panic")],
                ),
                (
                    "smg_kv_event_parentless_stores_total",
                    vec![("worker", url)],
                ),
                (
                    "smg_kv_event_parentless_blocks_total",
                    vec![("worker", url)],
                ),
                ("smg_kv_event_sub_block_stores_total", vec![("worker", url)]),
                ("smg_kv_event_sub_block_blocks_total", vec![("worker", url)]),
                (
                    "smg_kv_event_cap_refused_stores_total",
                    vec![("worker", url)],
                ),
                (
                    "smg_kv_event_cap_refused_blocks_total",
                    vec![("worker", url)],
                ),
                ("smg_kv_event_degraded_ranks", vec![("worker", url)]),
                (
                    "smg_kv_index_moved_hashes",
                    vec![("model", UNKNOWN_MODEL_ID)],
                ),
                (
                    "smg_kv_index_engine_conflicts",
                    vec![("model", UNKNOWN_MODEL_ID)],
                ),
                (
                    "smg_kv_index_cap_refusals",
                    vec![("model", UNKNOWN_MODEL_ID)],
                ),
                (
                    "smg_kv_index_memberships",
                    vec![("model", UNKNOWN_MODEL_ID)],
                ),
            ] {
                assert_eq!(
                    sample(&body, family, &labels),
                    Some(0.0),
                    "{family} {labels:?} on a fresh gateway:\n{body}"
                );
            }
            monitor.stop().await;
        });
    });
}

/// A model's index goes with its last subscription, and nothing publishes
/// the model's gauges after that: the removal itself zeroes them, or they
/// keep the removed worker's count until a worker of the model joins
/// again (an engine replaced behind a one-engine gateway).
#[test]
fn the_models_last_removal_zeroes_its_index_gauges() {
    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

    fn gauge(handle: &PrometheusHandle, name: &str, model: &str) -> Option<f64> {
        let prefix = format!("{name}{{model=\"{model}\"}}");
        handle
            .render()
            .lines()
            .find(|line| line.starts_with(&prefix))
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
    }

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    // The removal task publishes from the runtime's thread, which must
    // be the one holding the local recorder.
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let monitor = Arc::new(KvEventMonitor::new());
            let worker = refusing_grpc_worker();
            let url = worker.url().to_string();
            monitor
                .on_worker_added(&worker, SubscriptionKind::Events)
                .await;
            let index = monitor
                .get_indexer(UNKNOWN_MODEL_ID)
                .expect("the model's index");
            wait_until("the subscription interns its worker", || {
                index.worker_id(&url).is_some()
            })
            .await;
            let worker_id = index.worker_id(&url).unwrap();
            let blocks: Vec<StoredBlock> = (1..=3u64)
                .map(|i| StoredBlock {
                    seq_hash: SequenceHash(i),
                    content_hash: ContentHash(100 + i),
                })
                .collect();
            index
                .apply_stored(worker_id, &blocks, None, &mut WorkerBlocks::default())
                .unwrap();
            KvEventMonitor::publish_stats(&monitor.indexers);
            assert_eq!(
                gauge(&handle, "smg_kv_index_memberships", UNKNOWN_MODEL_ID),
                Some(3.0)
            );
            assert_eq!(
                gauge(&handle, "smg_kv_index_blocks_live", UNKNOWN_MODEL_ID),
                Some(3.0)
            );

            monitor.on_worker_removed(&url).await;
            assert!(
                monitor.get_indexer(UNKNOWN_MODEL_ID).is_none(),
                "the model's last worker takes its index with it"
            );
            for name in [
                "smg_kv_index_memberships",
                "smg_kv_index_entries",
                "smg_kv_index_blocks_live",
                "smg_kv_index_runs_live",
                "smg_kv_index_arena_bytes",
                "smg_kv_index_arena_chunk_bytes",
                "smg_kv_index_slab_bytes",
                "smg_kv_block_size_tokens",
            ] {
                assert_eq!(gauge(&handle, name, UNKNOWN_MODEL_ID), Some(0.0), "{name}");
            }
        });
    });
}

#[test]
fn start_stats_task_holds_the_task() {
    let monitor = KvEventMonitor::new();
    assert!(monitor.stats_task.lock().is_none());
    monitor.start_stats_task();
    assert!(monitor.stats_task.lock().is_some());
}
