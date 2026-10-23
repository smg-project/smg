//! The trace of a request through the gRPC pipeline: the request span, its
//! stage children, the dispatch span under it stamped with the engine's
//! first output, and, for a disaggregated dispatch, the prefill and decode
//! calls under the dispatch span, each to the side's first message.
//!
//! Its own test binary: the OpenTelemetry exporter is set up once per
//! process, here towards this binary's own collector, so every span the
//! requests below make lands where the assertions read.

use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{body::Body, extract::Request, http::StatusCode};
use common::grpc_harness::{
    grpc_router, grpc_router_config, grpc_test_context, grpc_worker, serve_canned_worker, Canned,
};
use openai_protocol::model_card::ModelCard;
use opentelemetry_proto::tonic::{
    collector::trace::v1::{
        trace_service_server::{TraceService, TraceServiceServer},
        ExportTraceServiceRequest, ExportTraceServiceResponse,
    },
    common::v1::{any_value, AnyValue},
};
use portpicker::pick_unused_port;
use serde_json::json;
use smg::{
    config::TraceConfig,
    observability::{logging, otel_trace},
    worker::{RuntimeType, WorkerType},
};
use smg_test_harness as common;
use tonic::{transport::Server, Request as TonicRequest, Response, Status};
use tower::ServiceExt;

const MODEL: &str = "span-tree-test-model";

/// The canned engines' delay before their first output: the dispatch
/// span's `first_output_us` has to cover it.
const GEN_DELAY: Duration = Duration::from_millis(40);

/// The stages every completion passes through, as the pipeline names their
/// spans.
const STAGES: [&str; 6] = [
    "preparation",
    "rate_limit",
    "worker_selection",
    "client_acquisition",
    "request_building",
    "response_processing",
];

/// One exported span: its name, its id, its parent's id and its attributes
/// in export order, each with its integer value when it holds one (a key
/// stamped twice shows twice).
#[derive(Clone, Debug)]
struct Exported {
    name: String,
    span_id: Vec<u8>,
    parent_span_id: Vec<u8>,
    attributes: Vec<(String, Option<i64>)>,
}

/// The integer of an attribute value, if that is what it holds.
fn int_value(value: Option<&AnyValue>) -> Option<i64> {
    match value?.value.as_ref()? {
        any_value::Value::IntValue(int) => Some(*int),
        _ => None,
    }
}

/// The collector the exporter sends to: every span it received.
#[derive(Clone, Default)]
struct Collector(Arc<Mutex<Vec<Exported>>>);

impl Collector {
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    fn spans(&self) -> Vec<Exported> {
        self.0.lock().expect("spans").clone()
    }
}

#[tonic::async_trait]
impl TraceService for Collector {
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    async fn export(
        &self,
        request: TonicRequest<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        let mut spans = self.0.lock().expect("spans");
        for resource in request.into_inner().resource_spans {
            for scope in resource.scope_spans {
                for span in scope.spans {
                    spans.push(Exported {
                        name: span.name,
                        span_id: span.span_id,
                        parent_span_id: span.parent_span_id,
                        attributes: span
                            .attributes
                            .iter()
                            .map(|kv| (kv.key.clone(), int_value(kv.value.as_ref())))
                            .collect(),
                    });
                }
            }
        }
        Ok(Response::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

/// The exported spans as a tree: the children of each span, by the parent's
/// id.
struct Tree {
    spans: Vec<Exported>,
    children: HashMap<Vec<u8>, Vec<usize>>,
}

impl Tree {
    fn of(spans: Vec<Exported>) -> Self {
        let mut children: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        for (index, span) in spans.iter().enumerate() {
            children
                .entry(span.parent_span_id.clone())
                .or_default()
                .push(index);
        }
        Self { spans, children }
    }

    /// The children of `span`, as names.
    fn child_names(&self, span: &Exported) -> BTreeSet<&str> {
        self.children
            .get(&span.span_id)
            .map(|indices| {
                indices
                    .iter()
                    .map(|&index| self.spans[index].name.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The one child of `span` named `name`.
    fn child(&self, span: &Exported, name: &str) -> Option<&Exported> {
        let mut found = self
            .children
            .get(&span.span_id)?
            .iter()
            .map(|&index| &self.spans[index])
            .filter(|child| child.name == name);
        let first = found.next()?;
        assert!(
            found.next().is_none(),
            "one {name} span under {}: {:?}",
            span.name,
            self.spans
        );
        Some(first)
    }

    fn named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Exported> + 'a {
        self.spans.iter().filter(move |span| span.name == name)
    }

    /// Whether every request's tree has arrived whole: the spans close in
    /// no fixed order (a dispatch span closes when the engine stream is
    /// released, which can be after the request span closed with the body),
    /// so the collector is read until the three request spans, their
    /// dispatch spans and the two calls are there.
    fn settled(&self) -> bool {
        let requests: Vec<&Exported> = self.named("http_request").collect();
        requests.len() == 3
            && requests
                .iter()
                .all(|request| self.child(request, "grpc_execute").is_some())
            && self.named("prefill_call").count() == 1
            && self.named("decode_call").count() == 1
    }
}

/// The canned mock engine: it answers every request with three tokens at
/// once.
async fn canned_worker() -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: GEN_DELAY,
        output_tokens: 3,
    })
    .await
}

/// A completion request through `app`; the body is read to its end, so the
/// dispatch span closes.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn complete(app: axum::Router, stream: bool, prompts: usize) {
    let prompt = if prompts == 1 {
        json!("trace me")
    } else {
        json!((0..prompts)
            .map(|i| format!("trace me {i}"))
            .collect::<Vec<_>>())
    };
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 3,
                "stream": stream,
            })
            .to_string(),
        ))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK, "stream={stream}");
    axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the body to its end");
}

/// A request's trace: the request span with one child per stage, the
/// dispatch span under it stamped with the engine's first output, and the
/// prefill and decode calls under the dispatch span of a disaggregated
/// request, each to the side's first message; a regular request's dispatch
/// span has no such children.
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test infrastructure: the collector task lives for the process"
)]
async fn a_request_trace_has_its_stages_its_dispatch_and_its_legs() {
    // The collector, and the exporter towards it.
    let port = pick_unused_port().expect("free port");
    let endpoint = format!("127.0.0.1:{port}");
    let collector = Collector::default();
    let service = collector.clone();
    let addr = endpoint.parse().expect("collector address");
    tokio::spawn(async move {
        Server::builder()
            .add_service(TraceServiceServer::new(service))
            .serve(addr)
            .await
            .expect("the collector serves");
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    otel_trace::otel_tracing_init(true, Some(&endpoint)).expect("tracing init");
    let _log_guard = logging::init_logging(
        logging::LoggingConfig {
            level: tracing::Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: false,
            log_file_name: "span-tree-test".to_string(),
            log_targets: Some(vec!["smg".to_string()]),
        },
        Some(TraceConfig {
            enable_trace: true,
            otlp_traces_endpoint: endpoint.clone(),
        }),
    );

    // A regular gateway over one canned engine: one streamed completion.
    let regular_port = canned_worker().await;
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .enable_trace(&endpoint)
        .build_unchecked();
    let context = grpc_test_context(config, MODEL).await;
    let router = grpc_router(
        &context,
        vec![Arc::new(
            grpc_worker(
                regular_port,
                WorkerType::Regular,
                RuntimeType::TokenSpeed,
                ModelCard::new(MODEL),
            )
            .build(),
        )],
    )
    .await;
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    complete(app.clone(), true, 1).await;
    // A batched completion: one dispatch span over two sub-requests, stamped
    // once with the batch's first output.
    complete(app, false, 2).await;

    // A disaggregated gateway over a canned prefill and a canned decode
    // engine: one buffered completion.
    let (prefill_port, decode_port) = (canned_worker().await, canned_worker().await);
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .random_policy()
        .enable_trace(&endpoint)
        .build_unchecked();
    let context = grpc_test_context(config, MODEL).await;
    let router = grpc_router(
        &context,
        vec![
            Arc::new(
                grpc_worker(
                    prefill_port,
                    WorkerType::Prefill,
                    RuntimeType::TokenSpeed,
                    ModelCard::new(MODEL),
                )
                .build(),
            ),
            Arc::new(
                grpc_worker(
                    decode_port,
                    WorkerType::Decode,
                    RuntimeType::TokenSpeed,
                    ModelCard::new(MODEL),
                )
                .build(),
            ),
        ],
    )
    .await;
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    complete(app, false, 1).await;

    // The spans reach the collector a moment after they close, in no fixed
    // order: flush until every tree is whole (bounded; the assertions
    // below print the tree they got).
    let mut tree = Tree::of(Vec::new());
    for _ in 0..50 {
        otel_trace::flush_spans_async()
            .await
            .expect("the spans flushed");
        tree = Tree::of(collector.spans());
        if tree.settled() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let requests: Vec<&Exported> = tree.named("http_request").collect();
    assert_eq!(
        requests.len(),
        3,
        "one request span per request: {:?}",
        tree.spans
    );
    let mut legs_seen = 0;
    for request in requests {
        let children = tree.child_names(request);
        for stage in STAGES {
            assert!(
                children.contains(stage),
                "the {stage} stage under the request span {:?}; children {children:?}; spans {:?}",
                request.span_id,
                tree.spans
            );
        }
        let dispatch = tree.child(request, "grpc_execute").unwrap_or_else(|| {
            panic!(
                "the dispatch span under the request span {:?}: {:?}",
                request.span_id, tree.spans
            )
        });
        let stamps: Vec<i64> = dispatch
            .attributes
            .iter()
            .filter(|(key, _)| key == "first_output_us")
            .map(|(_, value)| value.expect("first_output_us is an integer"))
            .collect();
        assert_eq!(
            stamps.len(),
            1,
            "the dispatch span is stamped once with the engine's first output: {:?}",
            dispatch.attributes
        );
        assert!(
            u128::try_from(stamps[0]).unwrap_or(0) >= GEN_DELAY.as_micros(),
            "first_output_us ({}) runs from the dispatch, so it covers the engine's {:?} delay",
            stamps[0],
            GEN_DELAY
        );
        let legs = tree.child_names(dispatch);
        if legs.is_empty() {
            continue;
        }
        legs_seen += 1;
        assert_eq!(
            legs,
            BTreeSet::from(["decode_call", "prefill_call"]),
            "the disaggregated dispatch's two calls"
        );
    }
    assert_eq!(
        legs_seen, 1,
        "exactly the disaggregated request's dispatch has legs: {:?}",
        tree.spans
    );
}
