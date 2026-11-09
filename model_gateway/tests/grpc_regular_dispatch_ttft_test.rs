//! Regular gRPC latency starts at dispatch, including a worker that holds
//! response headers until its first chunk. Each case captures metrics on a
//! current-thread runtime, as the PD TTFT regressions do.

use std::time::Duration;

use common::{
    decode_stream::{chunk, complete},
    engine_answers::{VllmAnswers, VllmGenerateStream},
    grpc_harness::{build_grpc_router, grpc_router_config, grpc_workers, serve_engine},
    metrics_capture::render_after,
};
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use smg_test_harness as common;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

const MODEL: &str = "regular-dispatch-ttft-test-model";
const OUTPUT_TOKENS: u32 = 3;
const HELD_HEADERS_DELAY: Duration = Duration::from_millis(300);
const POST_FIRST_DELAY: Duration = Duration::from_millis(120);

#[derive(Clone)]
struct HeldHeadersEngine;

#[tonic::async_trait]
impl VllmAnswers for HeldHeadersEngine {
    #[expect(
        clippy::disallowed_methods,
        reason = "test mock - the producer ends with its response stream"
    )]
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(8);
        // The RPC await absorbs this delay, with its first chunk already
        // buffered when it returns, like a Python servicer holding headers.
        tokio::time::sleep(HELD_HEADERS_DELAY).await;
        tx.send(Ok(chunk(1)))
            .await
            .map_err(|_| Status::cancelled("stream closed"))?;
        tokio::spawn(async move {
            tokio::time::sleep(POST_FIRST_DELAY).await;
            for id in 2..=OUTPUT_TOKENS {
                if tx.send(Ok(chunk(id))).await.is_err() {
                    return;
                }
            }
            let _ = tx.send(Ok(complete("stop", OUTPUT_TOKENS))).await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

async fn regular_router() -> Box<dyn RouterTrait> {
    let port = serve_engine(HeldHeadersEngine.into_server()).await;
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .build_unchecked();
    build_grpc_router(
        config,
        MODEL,
        grpc_workers(&[(port, WorkerType::Regular)], RuntimeType::Vllm, MODEL),
    )
    .await
}

#[derive(Clone, Copy)]
enum Route {
    Chat,
    Messages,
    Generate,
    Completions,
    BatchedCompletions,
}

impl Route {
    fn endpoint(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Messages => "messages",
            Self::Generate => "generate",
            Self::Completions | Self::BatchedCompletions => "completions",
        }
    }
}

#[expect(clippy::expect_used, reason = "test helper")]
async fn request(router: &dyn RouterTrait, route: Route, streaming: bool) -> Option<f64> {
    let tenant = TenantRequestMeta::new(TenantKey::new("regular-dispatch-ttft-test-tenant"));
    let response = match route {
        Route::Chat => {
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 8,
                "stream": streaming,
            }))
            .expect("chat request");
            router.route_chat(None, &tenant, request, MODEL).await
        }
        Route::Messages => {
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 8,
                "stream": streaming,
            }))
            .expect("Messages request");
            router.route_messages(None, &tenant, request, MODEL).await
        }
        Route::Generate => {
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "text": "Hello world",
                "sampling_params": {"max_new_tokens": 8},
                "stream": streaming,
            }))
            .expect("generate request");
            router.route_generate(None, &tenant, request, MODEL).await
        }
        Route::Completions | Route::BatchedCompletions => {
            let prompt = if matches!(route, Route::BatchedCompletions) {
                serde_json::json!(["Hello world", "Hello again"])
            } else {
                serde_json::json!("Hello world")
            };
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 8,
                "stream": streaming,
            }))
            .expect("completions request");
            router.route_completion(None, &tenant, request, MODEL).await
        }
    };
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("UTF-8 body");
    assert_eq!(status, http::StatusCode::OK, "{body}");
    if streaming {
        let ended = match route {
            Route::Chat | Route::Generate | Route::Completions | Route::BatchedCompletions => {
                body.contains("data: [DONE]")
            }
            Route::Messages => body.contains("event: message_stop"),
        };
        assert!(ended, "{body}");
    } else {
        let response: serde_json::Value = serde_json::from_str(&body).expect("JSON response");
        let output_tokens = match route {
            Route::Messages => response["usage"]["output_tokens"].as_u64(),
            Route::Generate => response["meta_info"]["completion_tokens"].as_u64(),
            Route::Chat | Route::Completions | Route::BatchedCompletions => {
                response["usage"]["completion_tokens"].as_u64()
            }
        };
        let expected_tokens = if matches!(route, Route::BatchedCompletions) {
            2 * OUTPUT_TOKENS
        } else {
            OUTPUT_TOKENS
        };
        assert_eq!(output_tokens, Some(u64::from(expected_tokens)), "{body}");
    }
    if matches!(route, Route::Generate) && streaming {
        return Some(
            body.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| {
                    serde_json::from_str::<serde_json::Value>(data).expect("generate event")
                })
                .find_map(|event| event["meta_info"]["e2e_latency"].as_f64())
                .expect("the terminal generate event carries e2e_latency"),
        );
    }
    None
}

fn sample(rendered: &str, family: &str, field: &str, endpoint: &str) -> Option<f64> {
    let prefix = format!("{family}_{field}{{");
    let endpoint = format!("endpoint=\"{endpoint}\"");
    rendered
        .lines()
        .find(|line| {
            line.starts_with(&prefix)
                && line.contains("backend_type=\"regular\"")
                && line.contains(&endpoint)
        })
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}

#[expect(clippy::expect_used, reason = "test helper")]
fn observation(rendered: &str, family: &str, endpoint: &str) -> f64 {
    assert_eq!(
        sample(rendered, family, "count", endpoint),
        Some(1.0),
        "{endpoint}: one {family} observation:\n{rendered}"
    );
    sample(rendered, family, "sum", endpoint).expect("histogram sum")
}

#[expect(
    clippy::print_stderr,
    reason = "report the measured intervals alongside the regression result"
)]
fn held_headers_are_included(route: Route, streaming: bool) {
    let mut e2e_latency = None;
    let rendered = render_after(|| async {
        let router = regular_router().await;
        e2e_latency = request(router.as_ref(), route, streaming).await;
    });
    let endpoint = route.endpoint();
    let generation = observation(
        &rendered,
        "smg_router_generation_duration_seconds",
        endpoint,
    );
    let ttft = sample(&rendered, "smg_router_ttft_seconds", "sum", endpoint);
    let tpot = sample(&rendered, "smg_router_tpot_seconds", "sum", endpoint);
    eprintln!(
        "regular {endpoint} (stream={streaming}): ttft={ttft:?} generation={generation} tpot={tpot:?}"
    );
    assert!(
        generation >= (HELD_HEADERS_DELAY + POST_FIRST_DELAY).as_secs_f64(),
        "{endpoint}: generation {generation} must include the held-header and decode delays"
    );
    if let Some(e2e_latency) = e2e_latency {
        assert!(
            generation - e2e_latency >= HELD_HEADERS_DELAY.as_secs_f64(),
            "generate: response e2e_latency {e2e_latency} keeps its processor-entry clock, \
             while Prometheus generation {generation} includes the held-header delay"
        );
    }
    if streaming {
        let ttft = observation(&rendered, "smg_router_ttft_seconds", endpoint);
        let tpot = observation(&rendered, "smg_router_tpot_seconds", endpoint);
        assert!(
            ttft >= HELD_HEADERS_DELAY.as_secs_f64(),
            "{endpoint}: TTFT {ttft} must include the held-header delay"
        );
        assert!(
            generation >= ttft,
            "{endpoint}: generation {generation} < TTFT {ttft}"
        );
        let expected = (generation - ttft) / f64::from(OUTPUT_TOKENS - 1);
        assert!(
            (tpot - expected).abs() < 1e-6,
            "{endpoint}: TPOT {tpot} != (generation {generation} - TTFT {ttft}) / 2"
        );
        assert!(
            tpot.is_finite() && tpot >= 0.0,
            "{endpoint}: TPOT {tpot} must be finite and nonnegative"
        );
    } else {
        // Collected responses retain their existing generation-only semantics.
        assert!(ttft.is_none(), "{endpoint}: collected TTFT {ttft:?}");
        assert!(tpot.is_none(), "{endpoint}: collected TPOT {tpot:?}");
    }
}

#[test]
fn streaming_chat_includes_the_held_header_delay() {
    held_headers_are_included(Route::Chat, true);
}

#[test]
fn streaming_messages_includes_the_held_header_delay() {
    held_headers_are_included(Route::Messages, true);
}

#[test]
fn streaming_generate_includes_the_held_header_delay() {
    held_headers_are_included(Route::Generate, true);
}

#[test]
fn streaming_completions_includes_the_held_header_delay() {
    held_headers_are_included(Route::Completions, true);
}

#[test]
fn collected_chat_includes_the_held_header_delay() {
    held_headers_are_included(Route::Chat, false);
}

#[test]
fn collected_messages_includes_the_held_header_delay() {
    held_headers_are_included(Route::Messages, false);
}

#[test]
fn collected_completions_includes_the_held_header_delay() {
    held_headers_are_included(Route::Completions, false);
}

#[test]
fn collected_batched_completions_includes_the_held_header_delay() {
    held_headers_are_included(Route::BatchedCompletions, false);
}
