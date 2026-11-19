//! The gateway's cases over the worker's health modules: each reads something
//! of the gateway beside them (the gRPC client's transport question, the
//! routers' retry rule, the gateway's own connector over an engine on the
//! wire, the connector mode the request path mints), so it stays with the
//! gateway.

use axum::http::StatusCode;
use openai_protocol::worker::{ResilienceUpdate, RuntimeType, WorkerType};
use smg_grpc_client::{h2, is_transport_failure};
use smg_worker::{
    health::{
        bootstrap_probe::effective_bootstrap_port_for_test, circuit_breaker::CircuitBreakerConfig,
        liveness::is_connection_code, manager::read_kv_engine_id_for_test,
        resilience::resolve_resilience,
    },
    identity::basic::{ConnectionMode, Worker, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR},
    settings::RetryConfig,
};

use crate::routers::{
    common::{
        kv_transfer::{kv_connector_mode, KvConnectorMode},
        retry::is_retryable_status,
    },
    grpc::{backend_client::BasicWorkerBuilder, test_engine::mock_engine_on_wire},
};

/// The gRPC client's `is_transport_failure` answers another question
/// (is this status the HTTP/2 layer's doing rather than the worker's
/// answer?) from the status's source chain, and the two disagree by
/// design in both directions.
#[test]
fn a_connection_code_and_a_transport_failure_are_different_questions() {
    // An Unavailable the worker answered with: unreachable for work, yet
    // not the transport's doing.
    let answered = tonic::Status::unavailable("shutting down");
    assert!(is_connection_code(answered.code()));
    assert!(!is_transport_failure(&answered));

    // An Internal tonic made of a protocol fault: the transport's doing,
    // yet its code is the one a worker's own failure carries.
    let fault = tonic::Status::from_error(Box::new(h2::Error::from(h2::Reason::PROTOCOL_ERROR)));
    assert_eq!(fault.code(), tonic::Code::Internal);
    assert!(is_transport_failure(&fault));
    assert!(!is_connection_code(fault.code()));

    // A refused stream is both: an Unavailable with an HTTP/2 cause.
    let refused = tonic::Status::from_error(Box::new(h2::Error::from(h2::Reason::REFUSED_STREAM)));
    assert_eq!(refused.code(), tonic::Code::Unavailable);
    assert!(is_transport_failure(&refused));
    assert!(is_connection_code(refused.code()));
}

#[test]
fn partial_retryable_override_leaves_capacity_pushback_retryable() {
    // Pins why the capacity set is not merged into the retryable set: a
    // per-worker retryable override that drops 429 cannot make pushback
    // non-retryable, because the retry gate is router-global.
    let overrides = ResilienceUpdate {
        retryable_status_codes: Some(vec![500, 502]),
        ..Default::default()
    };
    let (resolved, _cb) = resolve_resilience(
        &RetryConfig::default(),
        &CircuitBreakerConfig::default(),
        true,
        true,
        &overrides,
    );

    // The override replaces the default set verbatim - 429 is not merged in.
    assert!(!resolved.retryable_status_codes.contains(&429));
    // Yet 429 is still retried, because retryability never consults this set.
    assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
    // And it still records no circuit-breaker sample, because
    // `record_outcome` returns on the capacity set before it ever reads
    // `retryable_status_codes`.
    assert!(resolved.capacity_status_codes.contains(&429));
}

/// The manager's engine id read end to end, through the gateway's own
/// connector, over a mock engine that reports an id: a plain row, a rank row
/// and a worker registered under an HTTP scheme each read the reported id,
/// and an id reported empty reads as none.
#[tokio::test]
async fn the_engine_id_read_answers_the_id_an_engine_reports() {
    let (url, engine) = mock_engine_on_wire("m", mock_worker::config::SchedulerWire::Vllm).await;
    let port: u16 = url
        .rsplit(':')
        .next()
        .and_then(|port| port.parse().ok())
        .expect("the mock engine's port");
    let report = |id: &str| {
        let mut patch = serde_json::Map::new();
        patch.insert("kv_engine_id".to_string(), id.into());
        mock_worker::overrides::for_port(port).patch_info(&patch);
    };
    let row = |address: &str, rank: Option<(usize, usize)>| {
        let mut builder = BasicWorkerBuilder::new(address)
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm);
        if let Some((rank, size)) = rank {
            builder = builder.dp_config(rank, size);
        }
        builder.build()
    };
    let grpc = url.replacen("http://", "grpc://", 1);
    let reported = || Ok::<_, String>(Some("eng".to_string()));

    report("eng");
    assert_eq!(
        read_kv_engine_id_for_test(&row(&grpc, None)).await,
        reported()
    );
    assert_eq!(
        read_kv_engine_id_for_test(&row(&grpc, Some((1, 2)))).await,
        reported()
    );
    assert_eq!(
        read_kv_engine_id_for_test(&row(&url, None)).await,
        reported()
    );

    report("");
    assert_eq!(
        read_kv_engine_id_for_test(&row(&grpc, None)).await,
        Ok(None)
    );
    engine.abort();
}

/// For a prefill worker of the runtime that names a KV
/// connector the probe's decision is the request path's: the
/// default is probed exactly when the connector mode minted for the worker
/// hands the decode worker a bootstrap address on that port.
#[test]
fn the_default_port_decision_follows_the_connector_mode_the_request_path_mints() {
    for connector in [
        None,
        Some(MOONCAKE_CONNECTOR),
        Some("NixlConnector"),
        Some("MoRIIOConnector"),
        Some("LMCacheConnector"),
    ] {
        let mut builder = BasicWorkerBuilder::new("http://prefill-1:8000")
            .worker_type(WorkerType::Prefill)
            .runtime_type(RuntimeType::Vllm);
        if let Some(connector) = connector {
            builder = builder.kv_connector(connector);
        }
        let worker = builder.build();
        let minted = matches!(
            kv_connector_mode(connector, worker.bootstrap_host(), None, None),
            KvConnectorMode::Mooncake { port, .. } if port == u32::from(DEFAULT_BOOTSTRAP_PORT)
        );
        assert_eq!(
            effective_bootstrap_port_for_test(&worker).is_some(),
            minted,
            "{connector:?}"
        );
    }
}
