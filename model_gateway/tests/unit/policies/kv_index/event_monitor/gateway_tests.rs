//! The gateway's cases over the worker crate's KV event collector: the
//! monitor's subscriptions through the gateway's own client and the generated
//! service to the rounds, and from them into the index. They read the
//! gateway beside the collector, so they stay with the gateway.

use std::{
    net::SocketAddr,
    pin::Pin,
    sync::Mutex,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use futures::Stream;
use openai_protocol::{
    model_card::ModelCard,
    worker::{ConnectionMode, HealthCheckConfig, RuntimeType, WorkerType},
    UNKNOWN_MODEL_ID,
};
use smg_grpc_client::{
    common_proto::{self as common, kv_cache_event, KvEventBatch},
    tokenspeed_scheduler::tokenspeed_proto::{
        self as ts,
        token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
    },
};
use tokio::task::JoinHandle;
use tonic::{transport::Server, Request, Response, Status};
use tracing_test::traced_test;

use super::*;
use crate::routers::grpc::{backend_client::BasicWorkerBuilder, test_engine::serve_on_loopback};

/// The second subscribe call a scheduler noted, by which the first call's
/// stream was drained to its end.
async fn second_request(
    seen: &Mutex<Vec<common::SubscribeKvEventsRequest>>,
) -> common::SubscribeKvEventsRequest {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(request) = seen.lock().unwrap().get(1) {
            return *request;
        }
        assert!(
            Instant::now() < deadline,
            "the subscription never asked again"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The index the monitor keeps for `worker`'s model.
fn index_of(monitor: &KvEventMonitor, worker: &Arc<dyn Worker>) -> Arc<KvIndex> {
    monitor
        .get_indexer(worker.model_id())
        .or_else(|| monitor.get_indexer(UNKNOWN_MODEL_ID))
        .expect("the worker's model has its index")
}

/// One batch of stored events, one block per event, `(hash, block_size)` each.
fn stored_batch(blocks: &[(i64, usize)]) -> KvEventBatch {
    KvEventBatch {
        sequence_number: 1,
        events: blocks
            .iter()
            .map(|&(hash, block_size)| common::KvCacheEvent {
                event_id: hash as u64,
                data: Some(kv_cache_event::Data::Stored(common::KvBlocksStored {
                    blocks: vec![common::KvBlock {
                        block_hash: hash,
                        token_ids: (0..block_size as u32).collect(),
                        block_size: block_size as i32,
                        ..Default::default()
                    }],
                    ..Default::default()
                })),
            })
            .collect(),
        ..Default::default()
    }
}

fn learned(sizes: &DashMap<String, KvBlockSize>) -> Option<usize> {
    sizes.get("m").map(|size| size.tokens)
}

#[test]
fn a_partial_entry_never_sets_the_block_size_below_the_full_block() {
    let sizes: DashMap<String, KvBlockSize> = DashMap::new();
    // The engine's first batch names a partial entry before a full block.
    KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(1, 64), (2, 128)]));
    assert_eq!(learned(&sizes), Some(128));
    // A partial entry on its own never lowers the size.
    KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(3, 64)]));
    assert_eq!(learned(&sizes), Some(128));
    // A larger block raises it.
    KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(4, 256)]));
    assert_eq!(learned(&sizes), Some(256));
}

#[test]
fn the_first_stored_block_replaces_a_provisional_seed() {
    let sizes: DashMap<String, KvBlockSize> = DashMap::new();
    sizes.insert(
        "m".to_string(),
        KvBlockSize {
            tokens: 1152,
            from_events: false,
        },
    );
    KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(1, 128)]));
    assert_eq!(
        sizes.get("m").map(|size| *size),
        Some(KvBlockSize {
            tokens: 128,
            from_events: true,
        })
    );
    // Empty blocks and batches without stores change nothing.
    KvEventMonitor::learn_block_size(&sizes, "m", &stored_batch(&[(2, 0)]));
    KvEventMonitor::learn_block_size(&sizes, "m", &KvEventBatch::default());
    assert_eq!(learned(&sizes), Some(128));
}

/// What the test scheduler's `SubscribeKvEvents` does.
enum Subscribe {
    /// Accepts the call and never answers it, as an engine still
    /// starting behind an open port does.
    Hang,
    /// Answers `UNIMPLEMENTED`, as a backend whose KV event publisher is
    /// off does.
    Unimplemented,
    /// Notes the request, then never answers it, like `Hang`.
    RecordRequest(Arc<Mutex<Option<common::SubscribeKvEventsRequest>>>),
    /// Notes each request; answers the first at once with a stream that
    /// yields the script's items, then ends, and never answers a later one,
    /// like `Hang`.
    Once(
        Arc<Mutex<Vec<common::SubscribeKvEventsRequest>>>,
        Vec<Result<KvEventBatch, Status>>,
    ),
}

struct TestScheduler(Subscribe);

type ServerStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for TestScheduler {
    type GenerateStream = ServerStream<ts::GenerateResponse>;
    type SubscribeKvEventsStream = ServerStream<KvEventBatch>;
    type GetTokenizerStream = ServerStream<common::GetTokenizerChunk>;

    async fn generate(
        &self,
        _: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn health_check(
        &self,
        _: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn abort(
        &self,
        _: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn get_model_info(
        &self,
        _: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn get_server_info(
        &self,
        _: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn get_loads(
        &self,
        _: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        match &self.0 {
            Subscribe::Hang => std::future::pending().await,
            Subscribe::RecordRequest(seen) => {
                *seen.lock().unwrap() = Some(request.into_inner());
                std::future::pending().await
            }
            Subscribe::Unimplemented => Err(Status::unimplemented("KV cache events not enabled")),
            Subscribe::Once(seen, script) => {
                let first = {
                    let mut seen = seen.lock().unwrap();
                    seen.push(request.into_inner());
                    seen.len() == 1
                };
                if first {
                    Ok(Response::new(Box::pin(futures::stream::iter(
                        script.clone(),
                    ))))
                } else {
                    std::future::pending().await
                }
            }
        }
    }

    async fn flush_cache(
        &self,
        _: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn start_profile(
        &self,
        _: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn stop_profile(
        &self,
        _: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }

    async fn get_tokenizer(
        &self,
        _: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("test scheduler"))
    }
}

fn grpc_worker(addr: SocketAddr) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(format!("grpc://{addr}"))
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

/// A rank worker of a data-parallel engine at `addr`: rank `rank` of `size`.
fn rank_worker(addr: SocketAddr, rank: usize, size: usize) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(format!("grpc://{addr}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .dp_config(rank, size)
            .build(),
    )
}

/// The test scheduler served on a loopback port bound before it is
/// handed out, as its address and its server task.
async fn serve(behaviour: Subscribe) -> (SocketAddr, JoinHandle<()>) {
    let router =
        Server::builder().add_service(TokenSpeedSchedulerServer::new(TestScheduler(behaviour)));
    serve_on_loopback(router).await
}

/// A loads-only subscription asks the servicer for its records alone,
/// from the subscription on, and creates no index for the model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_loads_only_subscription_asks_for_records_alone_and_keeps_no_index() {
    let seen = Arc::new(Mutex::new(None));
    let (addr, server) = serve(Subscribe::RecordRequest(Arc::clone(&seen))).await;
    let worker = grpc_worker(addr);
    let monitor = KvEventMonitor::new();
    monitor
        .on_worker_added(&worker, SubscriptionKind::Loads)
        .await;
    let deadline = Instant::now() + Duration::from_secs(5);
    let request = loop {
        if let Some(request) = *seen.lock().unwrap() {
            break request;
        }
        assert!(
            Instant::now() < deadline,
            "the subscribe call never reached the scheduler"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(request.loads_only, "{request:?}");
    assert!(request.load_records_before_first_batch, "{request:?}");
    assert_eq!(request.start_sequence_number, 0);
    assert!(
        monitor.get_indexer(worker.model_id()).is_none()
            && monitor.get_indexer(UNKNOWN_MODEL_ID).is_none(),
        "a loads-only subscription builds no index"
    );
    tokio::time::timeout(
        Duration::from_secs(5),
        monitor.on_worker_removed(worker.url()),
    )
    .await
    .expect("removal returns while the subscribe call hangs");
    server.abort();
}

/// A data-parallel engine's stream names every rank's batches; a
/// rank worker indexes its rank's and the unranked ones, never another
/// rank's, so its overlap is its own cache, while every rank's batch
/// moves its cursor, so the resubscribe cursor is the unpinned worker's.
/// An unexpanded worker indexes them all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rank_worker_indexes_its_ranks_batches_only() {
    // Three blocks of distinct content (the chain index is keyed by the
    // tokens, not the engine's hash), numbered per admission cursor: a
    // batch that names no rank runs on rank 0's cursor.
    let ranked = |hash: i64, first_token: u32, sequence_number: u64, dp_rank: Option<i32>| {
        let mut batch = stored_batch(&[(hash, 16)]);
        if let Some(kv_cache_event::Data::Stored(stored)) = batch.events[0].data.as_mut() {
            stored.blocks[0].token_ids = (first_token..first_token + 16).collect();
        }
        batch.sequence_number = sequence_number;
        batch.dp_rank = dp_rank;
        batch
    };
    let batches = || {
        vec![
            Ok(ranked(42, 1_000, 1, Some(0))),
            Ok(ranked(43, 2_000, 1, Some(1))),
            Ok(ranked(44, 3_000, 2, None)),
        ]
    };
    for (pinned, indexed) in [(Some(1), 2), (Some(0), 2), (None, 3)] {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (addr, server) = serve(Subscribe::Once(Arc::clone(&seen), batches())).await;
        let worker = match pinned {
            Some(rank) => rank_worker(addr, rank, 2),
            None => grpc_worker(addr),
        };
        let monitor = KvEventMonitor::new();
        monitor
            .on_worker_added(&worker, SubscriptionKind::Events)
            .await;
        // The subscription asks again once it has drained the stream to its
        // end, from rank 0's cursor.
        let resumed = second_request(&seen).await;
        let indexer = index_of(&monitor, &worker);
        let worker_id = indexer.intern_worker(worker.url()).unwrap();
        assert_eq!(
            indexer.worker_block_count(worker_id),
            indexed,
            "pinned={pinned:?}"
        );
        assert_eq!(
            resumed.start_sequence_number, 2,
            "rank 0's cursor moved by rank 0's and the unranked batch, pinned={pinned:?}"
        );
        monitor.on_worker_removed(worker.url()).await;
        server.abort();
    }
}

/// A subscribe call that never answers, retried at every contact with
/// the worker, must still see the worker's removal: with no shutdown arm
/// on the call the task lived on and `on_worker_removed` never returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn removal_ends_a_subscribe_call_that_never_answers() {
    let (addr, server) = serve(Subscribe::Hang).await;

    let worker = grpc_worker(addr);
    let monitor = KvEventMonitor::new();
    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    // Let the task connect and park in the subscribe call.
    tokio::time::sleep(Duration::from_millis(500)).await;
    // The worker keeps being heard from on other paths; every contact
    // retries the parked call well inside its deadline.
    let pinger = {
        let worker = Arc::clone(&worker);
        #[expect(
            clippy::disallowed_methods,
            reason = "the contact source is stopped at the end of the test"
        )]
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                worker.runtime_state().unwrap().note_contact();
            }
        })
    };

    tokio::time::timeout(
        Duration::from_secs(5),
        monitor.on_worker_removed(worker.url()),
    )
    .await
    .expect("removal returns while the subscribe call hangs");

    pinger.abort();
    server.abort();
}

/// The subscription asks the servicer for load records before the
/// publisher's first batch: this gateway admits a `load_only` batch to
/// no cursor, and the silence of an engine frozen on the first request
/// it ever receives reaches the wedge rule through no other record.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_subscription_asks_for_load_records_before_the_first_batch() {
    let seen = Arc::new(Mutex::new(None));
    let (addr, server) = serve(Subscribe::RecordRequest(Arc::clone(&seen))).await;
    let worker = grpc_worker(addr);
    let monitor = KvEventMonitor::new();
    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    let deadline = Instant::now() + Duration::from_secs(5);
    let request = loop {
        if let Some(request) = *seen.lock().unwrap() {
            break request;
        }
        assert!(
            Instant::now() < deadline,
            "the subscribe call never reached the scheduler"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(request.load_records_before_first_batch, "{request:?}");
    assert_eq!(request.start_sequence_number, 0);
    tokio::time::timeout(
        Duration::from_secs(5),
        monitor.on_worker_removed(worker.url()),
    )
    .await
    .expect("removal returns while the subscribe call hangs");
    server.abort();
}

/// A backend that serves no KV events (an engine whose publisher is off)
/// leaves one WARN line, which names the model, carries the servicer's
/// own reason and says that the model reaches the approximate tree only
/// when no worker of it publishes; the worker carries a gauge from the
/// answer until it leaves. Its removal finds the task ended and frees the
/// slot itself, with the index of the model's last events subscription.
#[traced_test]
#[test]
fn a_backend_without_kv_events_is_a_gauge_until_the_worker_leaves() {
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
    // The subscription task publishes from the runtime's thread, which
    // must be the one holding the local recorder.
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let (addr, server) = serve(Subscribe::Unimplemented).await;

            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("grpc://{addr}"))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .model(ModelCard::new("hybrid-m"))
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            );
            let monitor = KvEventMonitor::new();
            monitor
                .on_worker_added(&worker, SubscriptionKind::Events)
                .await;
            for _ in 0..500 {
                if gauge(&handle, worker.url()) == Some(1.0) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(
                gauge(&handle, worker.url()),
                Some(1.0),
                "the answer raises the gauge"
            );
            // The WARN line itself names the model, carries the servicer's
            // reason and says when the model reaches the tree; the
            // subscription's start line names the model too, so the
            // checks are scoped to the refusal's line.
            logs_assert(|lines: &[&str]| {
                let line = lines
                    .iter()
                    .find(|line| line.contains("does not implement SubscribeKvEvents"))
                    .ok_or_else(|| "no refusal line was logged".to_string())?;
                for wanted in [
                    "WARN",
                    "hybrid-m",
                    "KV cache events not enabled",
                    "approximate prefix tree only when no worker of it publishes events",
                ] {
                    if !line.contains(wanted) {
                        return Err(format!("the refusal line lacks {wanted:?}: {line}"));
                    }
                }
                Ok(())
            });

            assert!(
                monitor.get_indexer("hybrid-m").is_some(),
                "the add made the model's index"
            );
            monitor.on_worker_removed(worker.url()).await;
            assert_eq!(
                gauge(&handle, worker.url()),
                Some(0.0),
                "the worker's removal lowers it"
            );
            // The task had ended at the engine's answer, before the removal
            // began: the removal itself frees the slot and drops the index
            // of the model's last events subscription.
            assert_eq!(
                monitor.slot_ids(worker.url()).await,
                (None, None),
                "the removal leaves no reservation"
            );
            assert!(
                monitor.get_indexer("hybrid-m").is_none(),
                "the model's last events subscription takes its index"
            );
            server.abort();
        });
    });
}

/// A worker whose connection is not the gateway's client carries no
/// event stream: its subscription is refused as unimplemented, the
/// worker's gauge rises, and no second connect is asked for.
#[test]
fn a_connection_that_is_not_the_gateways_client_carries_no_event_stream() {
    use metrics_exporter_prometheus::PrometheusBuilder;
    use smg_worker::{connection::engine::EngineConnector, test_support::CountingConnector};

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    // The subscription task publishes from the runtime's thread, which
    // must be the one holding the local recorder.
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let connector = Arc::new(CountingConnector::default());
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("grpc://no-engine:9000")
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .model(ModelCard::new("m"))
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .engine_connector(Arc::clone(&connector) as Arc<dyn EngineConnector>)
                    .build(),
            );
            let prefix = format!("smg_kv_events_unavailable{{worker=\"{}\"}}", worker.url());
            let gauge = || {
                handle
                    .render()
                    .lines()
                    .find(|line| line.starts_with(&prefix))
                    .and_then(|line| line.rsplit(' ').next()?.parse::<f64>().ok())
            };
            let monitor = KvEventMonitor::new();
            monitor
                .on_worker_added(&worker, SubscriptionKind::Events)
                .await;
            for _ in 0..500 {
                if gauge() == Some(1.0) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(gauge(), Some(1.0), "the refusal raises the gauge");
            assert_eq!(connector.connects(), 1, "no second connect");
            // The connect count cannot tell a second subscription over the
            // kept handle from an end: the loop's own task says it ended.
            let mut ended = false;
            for _ in 0..500 {
                ended = monitor.task_ended(worker.url()).await;
                if ended {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(ended, "the refusal ends the loop");
            monitor.on_worker_removed(worker.url()).await;
        });
    });
}

#[tokio::test]
async fn a_removed_workers_blocks_leave_the_index_off_the_runtime() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let script = vec![Ok(stored_batch(&[(1, 16)]))];
    let (addr, server) = serve(Subscribe::Once(Arc::clone(&seen), script)).await;
    let worker = grpc_worker(addr);
    let monitor = KvEventMonitor::new();
    monitor
        .on_worker_added(&worker, SubscriptionKind::Events)
        .await;
    second_request(&seen).await;
    let indexer = index_of(&monitor, &worker);
    assert_ne!(indexer.current_size(), 0, "the stored block is indexed");

    monitor.on_worker_removed(worker.url()).await;

    assert_eq!(indexer.current_size(), 0);
    server.abort();
}
