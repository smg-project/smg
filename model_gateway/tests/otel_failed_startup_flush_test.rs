//! A refused start-up flushes the tracer. The spans the batch exporter still
//! holds when the gateway refuses to start (here: the HTTP port another
//! process holds) reach the collector before the error is returned, the
//! tracer is shut down, and the error is the bind's own. The shutdown used
//! to follow the result's `?`, so a refusal skipped it and the spans were
//! lost with the process.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{
    net::TcpListener,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use opentelemetry_proto::tonic::collector::trace::v1::{
    trace_service_server::{TraceService, TraceServiceServer},
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use smg::{
    config::RouterConfig,
    observability::otel_trace,
    server::{self, ServerConfig},
};
use tonic::{
    transport::{server::TcpIncoming, Server},
    Request, Response, Status,
};
use tracing::info_span;
use tracing_subscriber::prelude::*;

/// An OTLP collector that counts the spans it is sent.
#[derive(Clone, Default)]
struct Collector {
    spans: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl TraceService for Collector {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        let spans = request
            .into_inner()
            .resource_spans
            .iter()
            .flat_map(|resource| &resource.scope_spans)
            .map(|scope| scope.spans.len())
            .sum();
        self.spans.fetch_add(spans, Ordering::SeqCst);
        Ok(Response::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

/// The tracer is process-wide and initialised once: this binary runs the one
/// start-up that needs it.
#[test]
fn a_refused_startup_flushes_the_spans_the_exporter_holds() {
    // The collector, on a runtime of its own, bound before the tracer dials it.
    let collector = Collector::default();
    let collector_runtime = tokio::runtime::Runtime::new().unwrap();
    let endpoint = collector_runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let service = TraceServiceServer::new(collector.clone());
        #[expect(
            clippy::disallowed_methods,
            reason = "the collector serves for the rest of the test"
        )]
        let _serving = tokio::spawn(
            Server::builder()
                .add_service(service)
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        endpoint
    });

    // The HTTP port the gateway is refused: held by this test throughout.
    let held = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = held.local_addr().unwrap().port();

    // The gateway's runtime, with the tracer initialised on it as `startup`
    // does: the exporter's connection lives on the runtime that dialled it.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime
        .block_on(async { otel_trace::otel_tracing_init(true, Some(&endpoint)) })
        .expect("the tracer initialises against a bound collector");

    // A span the exporter holds when the start-up fails, recorded through the
    // gateway's own layer as a request span is, and closed: it sits in the
    // batch, half a second short of the timer's export.
    let layer = otel_trace::get_otel_layer().expect("the layer follows the tracer");
    tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), || {
        info_span!(target: "smg::otel-trace", "before_the_refusal").in_scope(|| {});
    });

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port,
        health_check_port: None,
        runtime_worker_threads: None,
        secret_sources: Default::default(),
        router_config: RouterConfig::builder()
            .regular_mode(vec![])
            .host("127.0.0.1")
            .port(port)
            .build()
            .expect("a valid configuration"),
        max_payload_size: 1024 * 1024,
        log_dir: None,
        log_level: None,
        log_json: false,
        service_discovery_config: None,
        mesh_discovery_config: None,
        prometheus_config: None,
        request_timeout_secs: 60,
        request_id_headers: None,
        shutdown_grace_period_secs: 1,
        control_plane_auth: None,
        mesh_server_config: None,
        webrtc_bind_addr: None,
        webrtc_stun_server: None,
    };
    let started = Instant::now();
    let refused = server::run(&runtime, config);
    let elapsed = started.elapsed();
    // Read at once: the spans were to arrive before the result did.
    let received = collector.spans.load(Ordering::SeqCst);

    let error = refused.expect_err("the port is held").to_string();
    assert!(
        error.contains(&format!(
            "Failed to bind the HTTP listener on 127.0.0.1:{port}"
        )),
        "the refusal is the bind's own: {error}"
    );
    assert_eq!(
        received, 1,
        "the span the exporter held reached the collector before the refusal returned \
         (the start-up was refused after {elapsed:?})"
    );
    assert!(
        !otel_trace::is_otel_enabled(),
        "the tracer is shut down after a refused start-up"
    );
    drop(held);
}
