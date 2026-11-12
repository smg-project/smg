//! The PD path's time to first token splits into phases the gateway bounds
//! itself, checked against mock prefill and decode workers whose delays are
//! known.
//!
//! Sequential dispatch (the gateway relays the sides): the prefill, relay and
//! decode phases add up to `smg_pd_ttft_seconds` exactly and each phase carries
//! the delay its own worker added, whether the decode worker returns its
//! response headers at once or holds them until its first token. Parallel
//! dispatch (both sides sent together): the decode phase is the whole TTFT.
//!
//! Every streaming route (chat, Messages, completions) anchors its own time to
//! first token and generation duration at the prefill dispatch as well: on the
//! sequential pair they carry both workers' delays, not the decode worker's
//! alone. A collected answer, with no first token to time, anchors its one
//! generation record there too.
//!
//! Each case runs on a current-thread runtime under a thread-local Prometheus
//! recorder, so the request path's emissions land in the rendered text.

use std::time::Duration;

use common::{
    decode_stream::HeldHeadersEngine,
    engine_answers::VllmAnswers,
    grpc_harness::{
        build_grpc_router, grpc_router_config, grpc_workers, serve_canned_worker, serve_engine,
        Canned,
    },
    metrics_capture::{self, render_after},
};
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};
use smg_test_harness as common;

const MODEL: &str = "pd-ttft-phases-test-model";
/// Tokens every mock answers with; ids 1..=3 are in the mock tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Delay the prefill worker adds before it answers.
const PREFILL_DELAY: Duration = Duration::from_millis(200);
/// Delay the decode worker adds before its first token.
const DECODE_DELAY: Duration = Duration::from_millis(400);
/// Scheduling slack for a loaded machine. Below either delay, so a phase that
/// absorbed the other worker's delay fails its upper bound.
const SLACK: Duration = Duration::from_millis(250);

/// Serve a sequential-dispatch mock engine in-process; returns its port. The
/// engine answers every generate call with `OUTPUT_TOKENS` chunks and a
/// completion after its delay: `HeldHeadersEngine::holding` waits it out
/// before the call returns the stream, `HeldHeadersEngine::eager` returns the
/// stream at once and the delay runs inside it.
async fn start_mock_engine(engine: HeldHeadersEngine) -> u16 {
    serve_engine(engine.into_server()).await
}

/// Serve the mock-worker crate's canned worker in-process (a parallel-dispatch
/// runtime once registered as one): one up-front `delay`, then `OUTPUT_TOKENS`
/// tokens; returns its port.
async fn start_mock_grpc_worker(delay: Duration) -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: delay,
        output_tokens: OUTPUT_TOKENS,
    })
    .await
}

/// A PD gRPC router over one prefill and one decode worker registered as
/// `runtime`, which picks the dispatch shape.
async fn build_pd_router(
    prefill_port: u16,
    decode_port: u16,
    runtime: RuntimeType,
) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .random_policy()
        .build_unchecked();
    let workers = [
        (prefill_port, WorkerType::Prefill),
        (decode_port, WorkerType::Decode),
    ];
    build_grpc_router(config, MODEL, grpc_workers(&workers, runtime, MODEL)).await
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-ttft-phases-test-tenant"))
}

/// A streaming route of the router, with the endpoint label its stream
/// metrics carry.
#[derive(Clone, Copy)]
enum Route {
    Chat,
    Messages,
    Completions,
}

impl Route {
    fn endpoint(self) -> &'static str {
        match self {
            Route::Chat => "chat",
            Route::Messages => "messages",
            Route::Completions => "completions",
        }
    }
}

/// One streaming request on `route` through `router`, read to its end.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_streaming_request(router: &dyn RouterTrait, route: Route) -> String {
    let response = match route {
        Route::Chat => {
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 8,
                "stream": true,
                "stream_options": {"include_usage": true},
            }))
            .unwrap();
            router.route_chat(None, &tenant(), request, MODEL).await
        }
        Route::Messages => {
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 8,
                "stream": true,
            }))
            .unwrap();
            router.route_messages(None, &tenant(), request, MODEL).await
        }
        Route::Completions => {
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "prompt": "Hello world",
                "max_tokens": 8,
                "stream": true,
            }))
            .unwrap();
            router
                .route_completion(None, &tenant(), request, MODEL)
                .await
        }
    };
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf-8 body");
    assert_eq!(status, http::StatusCode::OK, "{body}");
    let ended = match route {
        Route::Chat | Route::Completions => body.contains("data: [DONE]"),
        Route::Messages => body.contains("event: message_stop"),
    };
    assert!(ended, "{body}");
    body
}

/// A chat stream's content deltas and the completion count of its usage
/// chunk: every `data:` line but the terminator is one chunk.
#[expect(
    clippy::expect_used,
    reason = "test helper - a chunk that is not JSON fails the test"
)]
fn chat_stream_counts(body: &str) -> (u32, Option<u64>) {
    let mut deltas = 0;
    let mut completion_tokens = None;
    for data in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if data == "[DONE]" {
            continue;
        }
        let chunk: serde_json::Value = serde_json::from_str(data).expect("a JSON chunk");
        if chunk["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|content| !content.is_empty())
        {
            deltas += 1;
        }
        if let Some(count) = chunk["usage"]["completion_tokens"].as_u64() {
            completion_tokens = Some(count);
        }
    }
    (deltas, completion_tokens)
}

/// One streaming chat through `router`, read to its `[DONE]`: the body.
async fn one_streaming_chat(router: &dyn RouterTrait) -> String {
    one_streaming_request(router, Route::Chat).await
}

/// The `_sum` or `_count` sample of `family` for `runtime`, if it was emitted.
fn sample(rendered: &str, family: &str, field: &str, runtime: &str) -> Option<f64> {
    let runtime = format!("runtime=\"{runtime}\"");
    metrics_capture::sample(rendered, family, field, &[runtime.as_str()])
}

/// The four PD TTFT families of one request, in seconds.
struct Phases {
    prefill: Option<f64>,
    relay: Option<f64>,
    decode: Option<f64>,
    ttft: Option<f64>,
}

impl Phases {
    fn of(rendered: &str, runtime: &str) -> Self {
        let sum = |family: &str| sample(rendered, family, "sum", runtime);
        for family in [
            "smg_pd_prefill_duration_seconds",
            "smg_pd_kv_transfer_duration_seconds",
            "smg_pd_decode_ttft_seconds",
            "smg_pd_ttft_seconds",
        ] {
            let count = sample(rendered, family, "count", runtime);
            assert!(
                count.is_none_or(|count| count == 1.0),
                "{family} recorded {count:?} times for runtime {runtime}:\n{rendered}"
            );
        }
        Self {
            prefill: sum("smg_pd_prefill_duration_seconds"),
            relay: sum("smg_pd_kv_transfer_duration_seconds"),
            decode: sum("smg_pd_decode_ttft_seconds"),
            ttft: sum("smg_pd_ttft_seconds"),
        }
    }

    /// Share of the TTFT the prefill and relay families alone explain, in percent.
    fn explained_without_decode(&self) -> f64 {
        let (Some(ttft), Some(prefill)) = (self.ttft, self.prefill) else {
            return 0.0;
        };
        (prefill + self.relay.unwrap_or(0.0)) / ttft * 100.0
    }
}

fn within(value: f64, at_least: Duration, slack: Duration) -> bool {
    value >= at_least.as_secs_f64() && value < (at_least + slack).as_secs_f64()
}

/// Print one case's phases: the measured values are the evidence this suite
/// produces, next to its pass/fail.
#[expect(
    clippy::print_stderr,
    reason = "the measured phases are the evidence this suite produces"
)]
fn report(label: &str, phases: &Phases) {
    eprintln!(
        "pd phases ({label}): prefill={:?} relay={:?} decode={:?} ttft={:?} \
         explained without the decode phase={:.1}%",
        phases.prefill,
        phases.relay,
        phases.decode,
        phases.ttft,
        phases.explained_without_decode()
    );
}

/// Sequential dispatch with a worker that returns its stream at once: the
/// prefill phase carries the prefill worker's delay, the relay phase is the
/// gateway's own work, the decode phase carries the decode worker's delay,
/// and the three add up to the TTFT exactly. The stream carries one content
/// delta per scripted token and the decode's completion count in its usage.
#[test]
fn sequential_dispatch_phases_add_up_to_the_pd_ttft() {
    let rendered = render_after(|| async {
        let prefill =
            start_mock_engine(HeldHeadersEngine::eager(PREFILL_DELAY, OUTPUT_TOKENS)).await;
        let decode = start_mock_engine(HeldHeadersEngine::eager(DECODE_DELAY, OUTPUT_TOKENS)).await;
        let router = build_pd_router(prefill, decode, RuntimeType::Vllm).await;
        let body = one_streaming_chat(router.as_ref()).await;
        let (deltas, completion_tokens) = chat_stream_counts(&body);
        assert_eq!(
            deltas, OUTPUT_TOKENS,
            "one content delta per scripted token: {body}"
        );
        assert_eq!(
            completion_tokens,
            Some(u64::from(OUTPUT_TOKENS)),
            "the usage chunk carries the decode's completion count: {body}"
        );
    });
    let phases = Phases::of(&rendered, RuntimeType::Vllm.as_str());
    report("sequential, eager headers", &phases);
    let (Some(prefill), Some(relay), Some(decode), Some(ttft)) =
        (phases.prefill, phases.relay, phases.decode, phases.ttft)
    else {
        panic!("a PD phase family is missing:\n{rendered}");
    };
    assert!(
        within(prefill, PREFILL_DELAY, SLACK),
        "prefill phase {prefill}"
    );
    assert!(relay < SLACK.as_secs_f64(), "relay phase {relay}");
    assert!(within(decode, DECODE_DELAY, SLACK), "decode phase {decode}");
    assert!(
        (ttft - (prefill + relay + decode)).abs() < 1e-6,
        "ttft {ttft} != prefill {prefill} + relay {relay} + decode {decode}"
    );
}

/// A decode worker that holds its response headers until its first token puts
/// that wait in the decode phase, not in the relay phase.
#[test]
fn a_held_header_decode_side_stays_in_the_decode_phase() {
    let rendered = render_after(|| async {
        let prefill =
            start_mock_engine(HeldHeadersEngine::eager(PREFILL_DELAY, OUTPUT_TOKENS)).await;
        let decode =
            start_mock_engine(HeldHeadersEngine::holding(DECODE_DELAY, OUTPUT_TOKENS)).await;
        let router = build_pd_router(prefill, decode, RuntimeType::Vllm).await;
        one_streaming_chat(router.as_ref()).await;
    });
    let phases = Phases::of(&rendered, RuntimeType::Vllm.as_str());
    report("sequential, held headers", &phases);
    let (Some(prefill), Some(relay), Some(decode), Some(ttft)) =
        (phases.prefill, phases.relay, phases.decode, phases.ttft)
    else {
        panic!("a PD phase family is missing:\n{rendered}");
    };
    assert!(
        within(prefill, PREFILL_DELAY, SLACK),
        "prefill phase {prefill}"
    );
    assert!(relay < SLACK.as_secs_f64(), "relay phase {relay}");
    assert!(within(decode, DECODE_DELAY, SLACK), "decode phase {decode}");
    assert!(
        (ttft - (prefill + relay + decode)).abs() < 1e-6,
        "ttft {ttft} != prefill {prefill} + relay {relay} + decode {decode}"
    );
}

/// Parallel dispatch sends both sides together: the decode phase is the TTFT
/// itself, and the prefill and relay phases of the sequential shape are not
/// recorded.
#[test]
fn parallel_dispatch_reports_the_decode_phase_as_the_whole_ttft() {
    let rendered = render_after(|| async {
        let prefill = start_mock_grpc_worker(DECODE_DELAY).await;
        let decode = start_mock_grpc_worker(DECODE_DELAY).await;
        let router = build_pd_router(prefill, decode, RuntimeType::TokenSpeed).await;
        one_streaming_chat(router.as_ref()).await;
    });
    let phases = Phases::of(&rendered, RuntimeType::TokenSpeed.as_str());
    report("parallel", &phases);
    let (Some(decode), Some(ttft)) = (phases.decode, phases.ttft) else {
        panic!("the decode phase or the TTFT is missing:\n{rendered}");
    };
    assert!(ttft >= DECODE_DELAY.as_secs_f64(), "ttft {ttft}");
    assert!(
        (decode - ttft).abs() < 1e-9,
        "decode phase {decode} != ttft {ttft}"
    );
    assert!(
        phases.prefill.is_none() && phases.relay.is_none(),
        "parallel dispatch has no prefill or relay phase:\n{rendered}"
    );
}

/// A streaming route anchors its own time to first token and generation
/// duration at the prefill dispatch: on the sequential pair the route's TTFT
/// carries the prefill worker's delay and the decode worker's, and the PD
/// first-token pair is recorded once. A stream clocked from its own start
/// would read the time to the decode side's first buffered chunk alone. With
/// `decode_holds` the decode worker holds its response headers through its
/// delay, so the hold sits inside the RPC await and not inside the stream.
#[expect(
    clippy::panic,
    reason = "test helper - panicking on failure is intentional"
)]
fn route_anchors_at_the_prefill_dispatch(route: Route, decode_holds: bool) {
    let rendered = render_after(|| async {
        let prefill =
            start_mock_engine(HeldHeadersEngine::eager(PREFILL_DELAY, OUTPUT_TOKENS)).await;
        let decode_engine = if decode_holds {
            HeldHeadersEngine::holding(DECODE_DELAY, OUTPUT_TOKENS)
        } else {
            HeldHeadersEngine::eager(DECODE_DELAY, OUTPUT_TOKENS)
        };
        let decode = start_mock_engine(decode_engine).await;
        let router = build_pd_router(prefill, decode, RuntimeType::Vllm).await;
        one_streaming_request(router.as_ref(), route).await;
    });
    let endpoint = route.endpoint();
    let label = format!("endpoint=\"{endpoint}\"");
    let labels = [label.as_str()];
    let (Some(count), Some(ttft), Some(generation)) = (
        metrics_capture::sample(&rendered, "smg_router_ttft_seconds", "count", &labels),
        metrics_capture::sample(&rendered, "smg_router_ttft_seconds", "sum", &labels),
        metrics_capture::sample(
            &rendered,
            "smg_router_generation_duration_seconds",
            "sum",
            &labels,
        ),
    ) else {
        panic!("{endpoint}: the stream metrics are missing:\n{rendered}");
    };
    let pd_pair = sample(
        &rendered,
        "smg_pd_ttft_seconds",
        "count",
        RuntimeType::Vllm.as_str(),
    );
    report(
        &format!("{endpoint} route: ttft={ttft} generation={generation}"),
        &Phases::of(&rendered, RuntimeType::Vllm.as_str()),
    );
    assert_eq!(count, 1.0, "{endpoint}: one TTFT record:\n{rendered}");
    let floor = (PREFILL_DELAY + DECODE_DELAY).as_secs_f64();
    assert!(
        ttft >= floor,
        "{endpoint}: TTFT {ttft} s carries both workers' delays ({floor} s)"
    );
    assert!(
        generation >= ttft,
        "{endpoint}: generation duration {generation} s shares the TTFT's base ({ttft} s)"
    );
    assert_eq!(
        pd_pair,
        Some(1.0),
        "{endpoint}: the PD pair recorded once:\n{rendered}"
    );
    let Some(tpot) = metrics_capture::sample(&rendered, "smg_router_tpot_seconds", "sum", &labels)
    else {
        panic!("{endpoint}: the TPOT record is missing:\n{rendered}");
    };
    let expected = (generation - ttft) / f64::from(OUTPUT_TOKENS - 1);
    assert!(
        (tpot - expected).abs() < 1e-6,
        "{endpoint}: TPOT {tpot} != (generation {generation} - TTFT {ttft}) / {}",
        OUTPUT_TOKENS - 1
    );
}

#[test]
fn the_chat_route_anchors_its_ttft_at_the_prefill_dispatch() {
    route_anchors_at_the_prefill_dispatch(Route::Chat, false);
}

#[test]
fn the_messages_route_anchors_its_ttft_at_the_prefill_dispatch() {
    route_anchors_at_the_prefill_dispatch(Route::Messages, false);
}

#[test]
fn the_completions_route_anchors_its_ttft_at_the_prefill_dispatch() {
    route_anchors_at_the_prefill_dispatch(Route::Completions, false);
}

#[test]
fn the_chat_route_anchors_at_the_prefill_dispatch_under_held_headers() {
    route_anchors_at_the_prefill_dispatch(Route::Chat, true);
}

#[test]
fn the_messages_route_anchors_at_the_prefill_dispatch_under_held_headers() {
    route_anchors_at_the_prefill_dispatch(Route::Messages, true);
}

#[test]
fn the_completions_route_anchors_at_the_prefill_dispatch_under_held_headers() {
    route_anchors_at_the_prefill_dispatch(Route::Completions, true);
}

/// A collected (non-streaming) chat answer on the sequential pair: its one
/// record, the generation duration, is anchored at the prefill dispatch, so it
/// carries both workers' delays. A clock read where the anchor is derived
/// would carry the decode worker's delay alone, since the sequential pair
/// drains the prefill stream before the answer path reads its anchor; the
/// case then falls short of its floor by the prefill delay.
#[test]
fn a_collected_chat_answer_anchors_its_generation_at_the_prefill_dispatch() {
    let rendered = render_after(|| async {
        let prefill =
            start_mock_engine(HeldHeadersEngine::eager(PREFILL_DELAY, OUTPUT_TOKENS)).await;
        let decode = start_mock_engine(HeldHeadersEngine::eager(DECODE_DELAY, OUTPUT_TOKENS)).await;
        let router = build_pd_router(prefill, decode, RuntimeType::Vllm).await;
        let request = serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello world"}],
            "max_tokens": 8,
        }))
        .unwrap();
        let response = router.route_chat(None, &tenant(), request, MODEL).await;
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("response body");
        assert_eq!(
            status,
            http::StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
    });
    let labels = ["endpoint=\"chat\""];
    let (Some(count), Some(generation)) = (
        metrics_capture::sample(
            &rendered,
            "smg_router_generation_duration_seconds",
            "count",
            &labels,
        ),
        metrics_capture::sample(
            &rendered,
            "smg_router_generation_duration_seconds",
            "sum",
            &labels,
        ),
    ) else {
        panic!("chat: the generation record is missing:\n{rendered}");
    };
    report(
        &format!("collected chat: generation={generation}"),
        &Phases::of(&rendered, RuntimeType::Vllm.as_str()),
    );
    assert_eq!(count, 1.0, "chat: one generation record:\n{rendered}");
    let floor = (PREFILL_DELAY + DECODE_DELAY).as_secs_f64();
    assert!(
        generation >= floor,
        "chat: generation duration {generation} s carries both workers' delays ({floor} s)"
    );
}
