mod common;

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use axum::{body::Body, extract::Request, http::StatusCode};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use futures::Stream;
use opentelemetry::trace::TraceContextExt;
use opentelemetry_proto::tonic::collector::trace::v1::{
    trace_service_server::{TraceService, TraceServiceServer},
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use portpicker::pick_unused_port;
use serde_json::json;
use serial_test::serial;
use smg::{
    config::{RouterConfig, TraceConfig},
    observability::{logging, otel_trace},
    routers::{grpc::client::GrpcClient, RouterFactory},
    workflow::Job,
};
use smg_grpc_client::{common_proto, tokenspeed_proto as ts};
use tokio::sync::oneshot;
use tonic::{metadata::MetadataMap, transport::Server, Request as TonicRequest, Response, Status};
use tower::ServiceExt;
use tracing::{info_span, instrument::WithSubscriber, Instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::prelude::*;
use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

#[derive(Clone)]
struct TestOtelCollector {
    span_count: Arc<AtomicUsize>,
}

impl TestOtelCollector {
    fn new() -> Self {
        Self {
            span_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn get_span_count(&self) -> usize {
        self.span_count.load(Ordering::SeqCst)
    }
}

#[tonic::async_trait]
impl TraceService for TestOtelCollector {
    async fn export(
        &self,
        request: TonicRequest<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        let req = request.into_inner();

        let mut total_spans = 0;

        for resource_span in &req.resource_spans {
            for scope_span in &resource_span.scope_spans {
                total_spans += scope_span.spans.len();
            }
        }

        self.span_count.fetch_add(total_spans, Ordering::SeqCst);

        Ok(Response::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

#[expect(clippy::disallowed_methods, reason = "test infrastructure")]
async fn start_collector(
    port: u16,
    shutdown_rx: oneshot::Receiver<()>,
) -> Result<TestOtelCollector, Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{port}").parse()?;
    let collector = TestOtelCollector::new();
    let collector_clone = collector.clone();

    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(TraceServiceServer::new(collector_clone))
            .serve_with_shutdown(addr, async {
                shutdown_rx.await.ok();
            })
            .await;
    });

    tokio::time::sleep(Duration::from_millis(200)).await;

    Ok(collector)
}

#[expect(clippy::print_stdout, reason = "test diagnostic output")]
#[tokio::test]
#[serial]
async fn test_router_with_tracing() {
    // 1. Start the OTLP collector
    let port = pick_unused_port().expect("Failed to pick unused port");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let collector = start_collector(port, shutdown_rx)
        .await
        .expect("Failed to start collector");
    let collector_endpoint = format!("0.0.0.0:{port}");
    println!("OTLP Collector started on: {collector_endpoint}");

    // 2. create the mock worker
    let mut mock_worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });

    let worker_url = mock_worker.start().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    println!("Mock worker started on: {worker_url}");

    // 3. create router config and enable tracing
    let mut router_config = RouterConfig::builder()
        .regular_mode(vec![worker_url.clone()])
        .random_policy()
        .host("0.0.0.0")
        .port(0)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .enable_trace(&collector_endpoint)
        .build_unchecked();
    router_config.health_check.disable_health_check = true;

    // 4. Initialize the OTLP client (check if already initialized by another test)
    let otel_initialized_by_this_test = if otel_trace::is_otel_enabled() {
        println!(
            "OpenTelemetry already initialized by previous test (spans will go to that collector)"
        );
        false
    } else {
        let init_result = otel_trace::otel_tracing_init(true, Some(&collector_endpoint));
        assert!(
            init_result.is_ok(),
            "Failed to initialize OTEL: {:?}",
            init_result.err()
        );
        println!("OpenTelemetry initialized successfully");
        true
    };

    let trace_config = TraceConfig {
        enable_trace: true,
        otlp_traces_endpoint: collector_endpoint.clone(),
    };
    let _log_guard = logging::init_logging(
        logging::LoggingConfig {
            level: tracing::Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: false,
            log_file_name: "test-otel".to_string(),
            log_targets: Some(vec!["smg".to_string()]),
        },
        Some(trace_config),
    );
    println!("Logging initialized with OTEL layer");

    // 5. Create a span and sleep for a while
    let _span = info_span!(target: "smg::otel-trace", "test_router_with_tracing");
    tokio::time::sleep(Duration::from_secs(1)).await;
    drop(_span);

    // 6. create app context and router
    let app_context = common::create_test_context(router_config.clone()).await;

    // 7. initialize worker
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");

    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(router_config.clone()),
    };

    job_queue
        .submit(job)
        .await
        .expect("Failed to submit worker init job");

    // 8. wait for worker initialization (detection + registration can take several seconds)
    tokio::time::sleep(Duration::from_millis(3000)).await;
    println!("Workers initialized");

    // 9. create router
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("Failed to create router");

    println!("Router created");

    // 10. create app (middleware::create_logging_layer() will use the already initialized OTEL layer)
    let app =
        common::test_app::create_test_app_with_context(Arc::from(router), app_context.clone());

    println!("App created with logging middleware");

    // 10. send request
    let request_body = json!({
        "model": "mock-model",
        "messages": [
            {"role": "user", "content": "Hello, test OpenTelemetry tracing!"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(request_body.to_string()))
        .unwrap();

    println!("Sending request to router...");
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK, "Request should succeed");

    println!("Request completed successfully");
    drop(response);

    // 11. Wait for spans to be exported
    match otel_trace::flush_spans_async().await {
        Ok(()) => println!("Spans flushed successfully"),
        Err(e) => println!("Failed to flush spans: {e:?}"),
    }

    // 12. Verify that the spans were exported to the collector
    let span_count = collector.get_span_count();
    println!("Total spans received by collector: {span_count}");

    // Only assert span count if we initialized OTEL with our own collector
    // When OTEL was pre-initialized by another test, spans go to that collector instead
    if otel_initialized_by_this_test {
        assert!(
            span_count == 2,
            "Expected to receive exactly 2 spans, but got {span_count}. \
            This indicates that tracing data is not being exported to the OTLP collector."
        );
        println!("Test passed! Collector received {span_count} spans");
    } else {
        println!(
            "Skipping span count assertion - OTEL was pre-initialized by another test. \
            Spans went to that collector. Received {span_count} spans on this test's collector."
        );
    }

    // 13. cleanup
    let _ = shutdown_tx.send(());
    mock_worker.stop().await;

    println!("Cleanup completed");
}

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

type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
type KvEventStream = Pin<Box<dyn Stream<Item = Result<common_proto::KvEventBatch, Status>> + Send>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common_proto::GetTokenizerChunk, Status>> + Send>>;

/// A TokenSpeed engine stub that records the `traceparent` of every generate
/// call's metadata and answers nothing else.
#[derive(Clone, Default)]
struct MetadataStub {
    traceparents: Arc<Mutex<Vec<Option<String>>>>,
}

#[tonic::async_trait]
impl TokenSpeedScheduler for MetadataStub {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: TonicRequest<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
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

    async fn get_model_info(
        &self,
        _request: TonicRequest<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("metadata stub"))
    }

    async fn get_server_info(
        &self,
        _request: TonicRequest<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("metadata stub"))
    }

    async fn get_loads(
        &self,
        _request: TonicRequest<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("metadata stub"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: TonicRequest<common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("metadata stub"))
    }

    async fn flush_cache(
        &self,
        _request: TonicRequest<common_proto::FlushCacheRequest>,
    ) -> Result<Response<common_proto::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("metadata stub"))
    }

    async fn start_profile(
        &self,
        _request: TonicRequest<common_proto::StartProfileRequest>,
    ) -> Result<Response<common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("metadata stub"))
    }

    async fn stop_profile(
        &self,
        _request: TonicRequest<common_proto::StopProfileRequest>,
    ) -> Result<Response<common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("metadata stub"))
    }

    async fn get_tokenizer(
        &self,
        _request: TonicRequest<common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("metadata stub"))
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
            .add_service(TokenSpeedSchedulerServer::new(stub))
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
