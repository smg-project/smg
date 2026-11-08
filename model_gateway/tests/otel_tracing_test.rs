use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use opentelemetry::trace::TraceContextExt;
use portpicker::pick_unused_port;
use serial_test::serial;
use smg::{observability::otel_trace, routers::grpc::client::GrpcClient};
use smg_grpc_client::tokenspeed_proto as ts;
use smg_test_harness::engine_answers::{TokenSpeedAnswers, TokenSpeedGenerateStream};
use tokio::sync::oneshot;
use tonic::{metadata::MetadataMap, transport::Server, Request as TonicRequest, Response, Status};
use tracing::{info_span, instrument::WithSubscriber, Instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::prelude::*;

use crate::collector::start_collector;

/// The collector is a target for the exporter here, nothing more: the span count is asserted
/// by the `otel_router_export_test` binary, whose one test initializes the tracer itself.
#[expect(
    dead_code,
    reason = "the count is read by the export binary; these tests need the collector as a target only"
)]
#[path = "otel/collector.rs"]
mod collector;

// ============================================================================
// gRPC Trace Context Injection Tests
// ============================================================================

/// Comprehensive test for gRPC trace context injection.
///
/// This test validates:
/// 1. W3C trace context headers are properly injected into gRPC metadata
/// 2. traceparent format is correct (version-traceid-spanid-flags)
/// 3. All metadata keys are lowercase (gRPC requirement)
///
/// Note: This test handles the case where OTEL may already be initialized
/// by a previous test (since tests run sequentially with #[serial]).
#[expect(clippy::print_stdout, reason = "test diagnostic output")]
#[tokio::test]
#[serial]
async fn test_grpc_trace_context_injection() {
    // 1. Start the OTLP collector (needed even if OTEL is already initialized,
    //    as a target for any spans that might be exported)
    let port = pick_unused_port().expect("Failed to pick unused port");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let _collector = start_collector(port, shutdown_rx)
        .await
        .expect("Failed to start collector");
    let collector_endpoint = format!("0.0.0.0:{port}");

    // 2. Initialize OTEL if not already enabled
    // Note: otel_tracing_init will fail if already initialized (OnceLock),
    // but that's fine - we just need OTEL to be enabled
    let already_enabled = otel_trace::is_otel_enabled();
    if !already_enabled {
        let init_result = otel_trace::otel_tracing_init(true, Some(&collector_endpoint));
        assert!(
            init_result.is_ok(),
            "Failed to initialize OTEL: {:?}",
            init_result.err()
        );
    }

    // Verify OTEL is enabled (either from this test or a previous one)
    assert!(otel_trace::is_otel_enabled(), "OTEL should be enabled");

    // 3. Set up tracing subscriber with OTEL layer
    let otel_layer = otel_trace::get_otel_layer().expect("Failed to get OTEL layer");
    let subscriber = tracing_subscriber::registry().with(otel_layer);

    // 4. Test within a span context
    tracing::subscriber::with_default(subscriber, || {
        // Create a span that will be exported to OTEL
        let span = info_span!(target: "smg::otel-trace", "test_grpc_span");
        let _guard = span.enter();

        // Create empty gRPC metadata
        let mut metadata = MetadataMap::new();

        // Inject trace context
        otel_trace::inject_trace_context_grpc(&mut metadata);

        // === Test 1: Verify traceparent header was injected ===
        let traceparent = metadata.get("traceparent");
        assert!(
            traceparent.is_some(),
            "traceparent header should be present in gRPC metadata"
        );

        // === Test 2: Verify traceparent format (version-traceid-spanid-flags) ===
        let traceparent_value = traceparent.unwrap().to_str().unwrap();
        let parts: Vec<&str> = traceparent_value.split('-').collect();
        assert_eq!(
            parts.len(),
            4,
            "traceparent should have 4 parts: version-traceid-spanid-flags"
        );
        assert_eq!(parts[0], "00", "traceparent version should be 00");
        assert_eq!(parts[1].len(), 32, "trace ID should be 32 hex characters");
        assert_eq!(parts[2].len(), 16, "span ID should be 16 hex characters");

        println!("Successfully injected traceparent: {traceparent_value}");

        // === Test 3: Verify all keys are lowercase (gRPC metadata requirement) ===
        for key_and_value in metadata.iter() {
            match key_and_value {
                tonic::metadata::KeyAndValueRef::Ascii(key, _) => {
                    let key_str = key.as_str();
                    assert_eq!(
                        key_str,
                        key_str.to_lowercase(),
                        "gRPC metadata key '{key_str}' should be lowercase"
                    );
                }
                tonic::metadata::KeyAndValueRef::Binary(key, _) => {
                    let key_str = key.as_str();
                    assert_eq!(
                        key_str,
                        key_str.to_lowercase(),
                        "gRPC metadata key '{key_str}' should be lowercase"
                    );
                }
            }
        }

        println!("All gRPC metadata keys are lowercase as required");
    });

    // Cleanup - don't shutdown OTEL since tests share global state (OnceLock)
    // and other tests may need to use the already-initialized OTEL
    let _ = shutdown_tx.send(());

    println!("test_grpc_trace_context_injection: All assertions passed!");
}

// ============================================================================
// The engine clients the gateway connects carry the trace context
// ============================================================================

/// A TokenSpeed engine stub that records the `traceparent` of every generate
/// call's metadata and answers nothing else.
#[derive(Clone, Default)]
struct MetadataStub {
    traceparents: Arc<Mutex<Vec<Option<String>>>>,
}

#[tonic::async_trait]
impl TokenSpeedAnswers for MetadataStub {
    async fn generate(
        &self,
        request: TonicRequest<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        let traceparent = request
            .metadata()
            .get("traceparent")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        self.traceparents
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(traceparent);
        Ok(Response::new(Box::pin(futures::stream::empty())))
    }

    async fn health_check(
        &self,
        _request: TonicRequest<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Ok(Response::new(ts::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: TonicRequest<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }
}

#[expect(
    clippy::disallowed_methods,
    clippy::expect_used,
    clippy::panic,
    reason = "test stub; the server lives for the test"
)]
async fn spawn_metadata_stub(stub: MetadataStub) -> u16 {
    let port = pick_unused_port().expect("free port for the metadata stub");
    let addr = format!("127.0.0.1:{port}").parse().expect("stub addr");
    tokio::spawn(async move {
        Server::builder()
            .add_service(stub.into_server())
            .serve(addr)
            .await
            .expect("metadata stub server");
    });
    for _ in 0..100 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return port;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("metadata stub on port {port} never came up");
}

/// A generate call made under the gateway's upstream span carries that span's
/// W3C trace context in its gRPC metadata: the client `GrpcClient::connect`
/// builds injects it, so the engine side can continue the gateway's trace.
#[tokio::test]
#[serial]
async fn engine_calls_carry_the_current_span_trace_context() {
    let port = pick_unused_port().expect("Failed to pick unused port");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let _collector = start_collector(port, shutdown_rx)
        .await
        .expect("Failed to start collector");
    let collector_endpoint = format!("0.0.0.0:{port}");
    if !otel_trace::is_otel_enabled() {
        otel_trace::otel_tracing_init(true, Some(&collector_endpoint)).expect("OTEL init");
    }
    let stub = MetadataStub::default();
    let stub_port = spawn_metadata_stub(stub.clone()).await;

    let otel_layer = otel_trace::get_otel_layer().expect("Failed to get OTEL layer");
    let subscriber = tracing_subscriber::registry().with(otel_layer);
    let trace_id = async {
        let span = info_span!(target: "smg::otel-trace", "grpc_execute");
        let trace_id = span.context().span().span_context().trace_id().to_string();
        let client = GrpcClient::connect(&format!("grpc://127.0.0.1:{stub_port}"), "tokenspeed")
            .instrument(span.clone())
            .await
            .expect("connect to the stub");
        let stream = client
            .as_tokenspeed()
            .generate(ts::GenerateRequest {
                request_id: "trace-context-1".to_string(),
                ..Default::default()
            })
            .instrument(span)
            .await
            .expect("generate against the stub");
        stream.mark_completed();
        trace_id
    }
    .with_subscriber(subscriber)
    .await;

    let traceparents = stub
        .traceparents
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(traceparents.len(), 1, "one generate call");
    let traceparent = traceparents[0]
        .as_deref()
        .expect("the generate call carries traceparent");
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4, "traceparent {traceparent}");
    assert_eq!(parts[1], trace_id, "traceparent {traceparent}");
    let _ = shutdown_tx.send(());
}
