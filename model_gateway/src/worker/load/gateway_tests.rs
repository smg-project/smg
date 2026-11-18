//! The gateway's case over the worker's load monitor: it reads the gateway
//! beside it (the gateway's own connector, polling an engine on the wire), so
//! it stays with the gateway.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, RuntimeType, WorkerGroupKey, WorkerType},
};

use crate::worker::{
    load::monitor::poll_group_once_for_test, test_support::LoadReaders, BasicWorkerBuilder,
    ConnectionMode, Worker, WorkerMonitor, WorkerRegistry,
};

/// The rank workers of one gRPC engine share one `GetLoads` per tick: the
/// engine answers once, and every rank worker gets its contact and its
/// own rank's part of the answer, as its own poll would have had it (the
/// canned engine names rank 0 only, so rank 1 has no report).
#[tokio::test]
async fn the_rank_workers_of_one_grpc_engine_share_one_poll() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let cfg = Arc::new(mock_worker::config::Config {
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        model_id: "a".to_string(),
        tokenizer_path: "a".to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 3,
        realistic: false,
        ..mock_worker::config::Config::default()
    });
    // A port an earlier test's engine served on keeps that engine's
    // overrides; this engine starts from fresh ones.
    mock_worker::overrides::fresh_for_port(port);
    #[expect(
        clippy::disallowed_methods,
        reason = "test engine: the server task lives for the test process"
    )]
    let _engine = tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    let registry = Arc::new(WorkerRegistry::new());
    let monitor = Arc::new(WorkerMonitor::new(
        registry.clone(),
        Arc::new(LoadReaders::Blind),
        1,
        false,
        false,
    ));
    let rank_worker = |rank: usize| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .model(ModelCard::new("a"))
                .dp_config(rank, 2)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    };
    let workers = [rank_worker(0), rank_worker(1)];
    for worker in &workers {
        registry.register(Arc::clone(worker)).unwrap();
    }
    let key = WorkerGroupKey {
        model_id: "a".to_string(),
        worker_type: WorkerType::Regular,
        connection_mode: ConnectionMode::Grpc,
    };
    let polls_before = mock_worker::overrides::for_port(port).loads_polls();
    // A transport failure noted on each rank worker: the one GetLoads'
    // contact must clear it on both.
    for worker in &workers {
        worker.runtime_state().unwrap().note_transport_failure();
    }
    poll_group_once_for_test(&monitor, &key, Duration::from_secs(10), Instant::now()).await;
    assert_eq!(
        mock_worker::overrides::for_port(port).loads_polls() - polls_before,
        1,
        "one GetLoads for the engine's two rank workers"
    );

    let snapshot = monitor.load_snapshot();
    let report = snapshot
        .get(workers[0].url())
        .expect("the rank the answer names has its report");
    assert_eq!((report.loads.len(), report.loads[0].dp_rank), (1, 0));
    assert!(
        snapshot.get(workers[1].url()).is_none(),
        "a rank the answer does not name has no report, as its own poll would have had it"
    );
    for worker in &workers {
        assert!(
            !worker.runtime_state().unwrap().transport_failure_pending(),
            "the contact is every member's: {}",
            worker.url()
        );
    }
}
