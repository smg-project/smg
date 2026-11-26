//! `GET /policy/explain/{request_id}` on the production route table: a kept
//! routing decision is read back by its request's id, or by the gateway's
//! own id of a request that states none, and an id that finds none is
//! answered 404 with its reason. The gateway states its own id under a
//! private request header, which no worker is sent.

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        HeaderMap, HeaderValue, StatusCode,
    },
    Json, Router,
};
use openai_protocol::model_card::ModelCard;
use serde_json::{json, Value};
use smg::{
    config::{RetryConfig, RouterConfig},
    routers::{grpc::backend_client::BasicWorkerBuilder, RouterTrait},
};
use smg_grand_central::document::PolicySpec;
use smg_worker::identity::basic::{RuntimeType, Worker, WorkerType};
use tokio::{net::TcpListener, sync::Mutex};
use tower::ServiceExt;

use crate::common::{
    self,
    grpc_harness::{
        grpc_router, grpc_router_config, grpc_test_context, grpc_worker, grpc_workers,
        health_checks_off, serve_canned_worker, Canned,
    },
    test_app::create_test_app_with_context,
    AppTestContext, TestWorkerConfig,
};

/// The request header the gateway restates its own id of a request under,
/// where its policy spec keeps a sample.
const PRIVATE_HEADER: &str = "x-smg-request-id";

/// The model a recording worker serves ([`registered`]).
const RECORDED_MODEL: &str = "mock-model";

/// `config` with Grand Central on under the `round_robin` preset as a
/// policy spec that keeps every `sample`th decision, and none where it is
/// zero. The routing settings of `config` must be the default ones, which a
/// supplied policy spec stands beside.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn sampled(config: &mut RouterConfig, sample: u32) {
    let spec = PolicySpec::preset("round_robin").expect("a built-in preset");
    sampled_under(config, spec, sample);
}

/// `config` with Grand Central on under `spec` as a policy spec that keeps
/// every `sample`th decision, and none where it is zero.
fn sampled_under(config: &mut RouterConfig, mut spec: PolicySpec, sample: u32) {
    spec.generation = 1;
    spec.explain.sample = sample;
    config.grand_central = true;
    config.policy_spec = Some(spec);
}

/// A gateway of two workers whose routing settings are the default ones.
/// With a `sample` it runs Grand Central under a policy spec that keeps a
/// sample ([`sampled`]). Without one it leaves Grand Central off.
async fn gateway(sample: Option<u32>) -> AppTestContext {
    gateway_with(sample, |_| ()).await
}

/// [`gateway`], with what `turned` changes in its settings beside the
/// routing ones.
async fn gateway_with(
    sample: Option<u32>,
    turned: impl FnOnce(&mut RouterConfig),
) -> AppTestContext {
    gateway_over(2, sample, turned).await
}

/// [`gateway_with`] over `mock_workers` workers of the harness: none where
/// a case registers recording workers of its own ([`registered`]).
async fn gateway_over(
    mock_workers: usize,
    sample: Option<u32>,
    turned: impl FnOnce(&mut RouterConfig),
) -> AppTestContext {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .host("127.0.0.1")
        .port(0)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    turned(&mut config);
    if let Some(sample) = sample {
        sampled(&mut config, sample);
    }
    let workers = (0..mock_workers)
        .map(|_| TestWorkerConfig::healthy(0))
        .collect();
    AppTestContext::new_with_config(config, workers).await
}

/// What a [`recording_worker`] was sent: the path and the headers of each
/// request, oldest first.
type Recorded = Arc<Mutex<Vec<(String, HeaderMap)>>>;

/// A [`recording_worker`]'s answer to every request: it records the
/// request's path and headers, reads its body whole, and answers the
/// status and the JSON body the worker was started with.
async fn record(
    State((recorded, status, answer)): State<(Recorded, StatusCode, Value)>,
    request: Request,
) -> (StatusCode, Json<Value>) {
    let (parts, body) = request.into_parts();
    // Read to its end, so that a body the gateway streams is sent whole.
    let _body = axum::body::to_bytes(body, usize::MAX).await;
    let path = parts.uri.path().to_owned();
    recorded.lock().await.push((path, parts.headers));
    (status, Json(answer))
}

/// A worker on a loopback port that answers every request `status` with
/// `answer` as its JSON body and records what it was sent ([`record`]):
/// its address and its record. Its task ends with the test's runtime.
#[expect(
    clippy::disallowed_methods,
    clippy::expect_used,
    reason = "test helper: failures should panic, and the worker's task ends with the test's runtime"
)]
async fn recording_worker(status: StatusCode, answer: Value) -> (String, Recorded) {
    let recorded = Recorded::default();
    let app = Router::new()
        .fallback(record)
        .with_state((Arc::clone(&recorded), status, answer));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port");
    let address = listener.local_addr().expect("the port's address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("the worker serves");
    });
    (format!("http://{address}"), recorded)
}

/// The worker at `url` registered with `gateway` as one of `kind` and
/// `runtime` that serves [`RECORDED_MODEL`], ready as it is: no health
/// check asks it anything.
fn registered(gateway: &AppTestContext, url: &str, kind: WorkerType, runtime: RuntimeType) {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(kind)
        .runtime_type(runtime)
        .model(ModelCard::new(RECORDED_MODEL))
        .health_config(health_checks_off())
        .build();
    let worker: Arc<dyn Worker> = Arc::new(worker);
    gateway.app_context.worker_registry.register(worker);
}

/// What a worker recorded, as text for a failed assertion's message: one
/// `path id` a request, the id being the request-id header it was sent
/// (`-` for none).
fn listed(recorded: &[(String, HeaderMap)]) -> Vec<String> {
    let line = |(path, headers): &(String, HeaderMap)| {
        let id = headers.get("x-request-id").and_then(|id| id.to_str().ok());
        format!("{path} {}", id.unwrap_or("-"))
    };
    recorded.iter().map(line).collect()
}

/// The headers of each request to `path` a worker recorded. A worker also
/// records the gateway's own reads of its load, which are no route's and
/// arrive when the gateway's monitor asks.
fn sent_to<'a>(recorded: &'a [(String, HeaderMap)], path: &str) -> Vec<&'a HeaderMap> {
    let to_path = recorded.iter().filter(|(sent, _)| sent == path);
    to_path.map(|(_, headers)| headers).collect()
}

/// Whether any request a worker recorded carries the private header: a
/// route's request, or one of the gateway's own reads of the worker's load.
fn holds_the_private_header(recorded: &[(String, HeaderMap)]) -> bool {
    let private = |(_, headers): &(String, HeaderMap)| headers.contains_key(PRIVATE_HEADER);
    recorded.iter().any(private)
}

/// One request to `app`: its status, its headers and its JSON body, null
/// where the body is no JSON.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn sent(app: &Router, request: Request) -> (StatusCode, HeaderMap, Value) {
    let response = app.clone().oneshot(request).await.expect("an answer");
    let (status, headers) = (response.status(), response.headers().clone());
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a whole body");
    let body = serde_json::from_slice(&body).unwrap_or(Value::Null);
    (status, headers, body)
}

/// The JSON `body` posted to `path`.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn posted(path: &str, body: &Value) -> Request {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("a request")
}

/// The gateway's own id of the request `headers` answer: the one the
/// response's `x-request-id` header shows.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn own_id(headers: &HeaderMap) -> String {
    let id = headers["x-request-id"].to_str().expect("a visible id");
    id.to_owned()
}

/// The address of the one target of the decision kept under `id`, read on
/// `app`: the read is answered 200 and the decision names one target.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn one_target(app: &Router, id: &str) -> Value {
    let (status, _, kept) = sent(app, explain(id)).await;
    assert_eq!(status, StatusCode::OK, "{id}: {kept}");
    let targets = kept["decision"]["pick"]["targets"].as_array();
    let targets = targets.expect("the decision's targets");
    assert_eq!(targets.len(), 1, "{id}: {kept}");
    targets[0]["url"].clone()
}

/// A generate request whose body states `rid` as its id.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn generate(rid: &str) -> Request {
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"text": "one request", "stream": false, "rid": rid}).to_string(),
        ))
        .expect("a request")
}

/// A generate request whose body states no id.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn generate_with_no_id() -> Request {
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"text": "one request", "stream": false}).to_string(),
        ))
        .expect("a request")
}

/// The read of the decision kept of the request `id` names.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn explain(id: &str) -> Request {
    Request::builder()
        .method("GET")
        .uri(format!("/policy/explain/{id}"))
        .body(Body::empty())
        .expect("a request")
}

/// Under a policy spec that keeps every decision, a request is read back by
/// the id its body states. The answer names one target, by the address of
/// the worker the request was sent to: the one the gateway names in the
/// response's routed-worker header. It holds the names of the plan and the
/// profile that decided, and states no id of the request.
#[tokio::test]
async fn a_kept_decision_is_read_by_its_requests_id_with_the_address_it_was_sent_to() {
    let gateway = gateway(Some(1)).await;
    let app = gateway.create_app();

    for id in ["explain-a", "explain-b", "explain-c"] {
        let (status, headers, _) = sent(&app, generate(id)).await;
        assert_eq!(status, StatusCode::OK, "{id} is served");
        let sent_to = headers["x-smg-routed-worker-id"].to_str().unwrap();
        assert!(gateway.worker_urls.iter().any(|url| url == sent_to));

        let (status, _, kept) = sent(&app, explain(id)).await;
        assert_eq!(status, StatusCode::OK, "{id}: {kept}");
        let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
        assert_eq!(targets.len(), 1, "{id}: {kept}");
        assert_eq!(targets[0]["url"], json!(sent_to), "{id}: {kept}");
        assert!(targets[0]["id"].is_string(), "{id}: {kept}");
        assert_eq!(kept["profile_name"], json!("round_robin"), "{id}: {kept}");
        assert!(kept["plan_name"].is_string(), "{id}: {kept}");
        assert_eq!(kept.get("request_id"), None, "{id}: {kept}");
    }

    gateway.shutdown().await;
}

/// A request whose body states no id is read back by the gateway's own id
/// of it: the one its response's `x-request-id` header shows, which is the
/// client's where the client sent that header. A request whose body states
/// an id is found by that id and not by the gateway's: a body's id wins.
#[tokio::test]
async fn a_request_that_states_no_id_is_read_by_the_gateways_own_id() {
    let gateway = gateway(Some(1)).await;
    let app = gateway.create_app();

    // The gateway's own id, which it made.
    let (status, headers, _) = sent(&app, generate_with_no_id()).await;
    assert_eq!(status, StatusCode::OK, "the request is served");
    let own = headers["x-request-id"].to_str().unwrap().to_owned();
    let sent_to = headers["x-smg-routed-worker-id"].to_str().unwrap();
    let (status, _, kept) = sent(&app, explain(&own)).await;
    assert_eq!(status, StatusCode::OK, "{own}: {kept}");
    let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 1, "{own}: {kept}");
    assert_eq!(targets[0]["url"], json!(sent_to), "{own}: {kept}");

    // The gateway's own id, which the client sent in the header.
    let mut named = generate_with_no_id();
    let header = HeaderValue::from_static("explain-by-header");
    named.headers_mut().insert("x-request-id", header);
    let (status, headers, _) = sent(&app, named).await;
    assert_eq!(status, StatusCode::OK, "the request is served");
    assert_eq!(headers["x-request-id"], "explain-by-header");
    let (status, _, kept) = sent(&app, explain("explain-by-header")).await;
    assert_eq!(status, StatusCode::OK, "{kept}");

    // A body's id wins: the record stands under it alone.
    let (status, headers, _) = sent(&app, generate("explain-by-body")).await;
    assert_eq!(status, StatusCode::OK, "the request is served");
    let own = headers["x-request-id"].to_str().unwrap().to_owned();
    let (status, _, kept) = sent(&app, explain("explain-by-body")).await;
    assert_eq!(status, StatusCode::OK, "{kept}");
    let (status, _, refused) = sent(&app, explain(&own)).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{own}: {refused}");
    assert_eq!(refused["code"], json!("DECISION_NOT_KEPT"), "{own}");

    gateway.shutdown().await;
}

/// A request the gateway forwards with its body unread states no id of its
/// own, whatever the body holds. It is read back by the gateway's own id of
/// it, and the id in its body finds nothing. The gateway forwards a body
/// unread where its retries are off and nothing of it must be read: the
/// request names its length, as a client's does.
#[tokio::test]
async fn a_request_forwarded_unread_is_read_by_the_gateways_own_id() {
    let gateway = gateway_with(Some(1), |config| config.disable_retries = true).await;
    let app = gateway.create_app();

    let asked = json!({"text": "one request", "stream": false, "rid": "explain-unread"});
    let asked = asked.to_string();
    let request = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, asked.len())
        .body(Body::from(asked))
        .expect("a request");
    let (status, headers, _) = sent(&app, request).await;
    assert_eq!(status, StatusCode::OK, "the request is served");
    let own = headers["x-request-id"].to_str().unwrap().to_owned();
    let sent_to = headers["x-smg-routed-worker-id"].to_str().unwrap();

    let (status, _, kept) = sent(&app, explain(&own)).await;
    assert_eq!(status, StatusCode::OK, "{own}: {kept}");
    let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 1, "{own}: {kept}");
    assert_eq!(targets[0]["url"], json!(sent_to), "{own}: {kept}");

    // The body went on unread: the id in it is no id of a kept decision.
    let (status, _, refused) = sent(&app, explain("explain-unread")).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{refused}");
    assert_eq!(refused["code"], json!("DECISION_NOT_KEPT"), "{refused}");

    gateway.shutdown().await;
}

/// A route that hands its placement the request's headers and no word of
/// its own is read back the same way: a rerank request whose body states no
/// id and an audio transcription, each by the id its response's
/// `x-request-id` header shows.
#[tokio::test]
async fn a_rerank_and_a_transcription_request_are_read_by_the_gateways_own_id() {
    let gateway = gateway(Some(1)).await;
    let app = gateway.create_app();
    let is_a_worker = |target: &Value| gateway.worker_urls.iter().any(|url| json!(url) == *target);

    let asked = json!({"query": "a query", "documents": ["one", "two"], "model": "mock-model"});
    let (status, headers, answer) = sent(&app, posted("/rerank", &asked)).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let target = one_target(&app, &own_id(&headers)).await;
    assert!(is_a_worker(&target), "rerank: {target}");

    let boundary = "explain-form";
    let form = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\nRIFFmock\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nmock-model\r\n\
         --{boundary}--\r\n"
    );
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(form))
        .expect("a request");
    let (status, headers, answer) = sent(&app, request).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let target = one_target(&app, &own_id(&headers)).await;
    assert!(is_a_worker(&target), "transcription: {target}");

    gateway.shutdown().await;
}

/// So are a count of tokens, a decisions request and a SystemOne request,
/// each sent to a worker that records it: the decision kept under the id
/// the response's `x-request-id` header shows names that worker, which was
/// sent the request once and no header of the private name.
#[tokio::test]
async fn count_tokens_decisions_and_systemone_requests_are_read_by_the_gateways_own_id() {
    let model = RECORDED_MODEL;
    let routes = [
        (
            "/v1/messages/count_tokens",
            json!({"model": model, "messages": [{"role": "user", "content": "count me"}]}),
            json!({"input_tokens": 3}),
        ),
        (
            "/v1/decisions",
            json!({"model": model, "input": "The sky is blue.", "questions": [
                {"type": "predicate", "instructions": "Is the sky blue?"}]}),
            json!({"model": model, "answers": [
                {"type": "predicate", "name": null, "probability": 0.9, "native_answer": true}],
                "usage": {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7,
                    "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 0}}}),
        ),
        (
            "/v1/systemone",
            json!({"model": model, "state": "The package arrived intact.", "questions": {
                "delivered": {"type": "noul", "instructions": "Did the package arrive?",
                    "criteria": {"true": "Arrived", "false": null}}}}),
            json!({"model": model, "answers": {"delivered": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 27, "output_tokens": 0}}),
        ),
    ];
    for (path, asked, answer) in routes {
        let gateway = gateway_over(0, Some(1), |_| ()).await;
        let (url, recorded) = recording_worker(StatusCode::OK, answer).await;
        registered(&gateway, &url, WorkerType::Regular, RuntimeType::Generic);
        let app = gateway.create_app();

        let (status, headers, answer) = sent(&app, posted(path, &asked)).await;
        assert_eq!(status, StatusCode::OK, "{path}: {answer}");
        let target = one_target(&app, &own_id(&headers)).await;
        assert_eq!(target, json!(url), "{path}");
        let recorded = recorded.lock().await;
        let routed = sent_to(&recorded, path).len();
        assert_eq!(routed, 1, "{path}: {:?}", listed(&recorded));
        let private = holds_the_private_header(&recorded);
        assert!(!private, "{path}: {:?}", listed(&recorded));
        drop(recorded);

        gateway.shutdown().await;
    }
}

/// No worker is sent the gateway's private header. A worker that records
/// its headers gets the request-id header the client sent and none under
/// the private name, whatever the client sent there: where the gateway
/// reads the body, and where it forwards the body unread (its retries off,
/// so the worker gets no length). The decision is found by the client's id
/// both times.
#[tokio::test]
async fn a_worker_is_sent_no_header_of_the_private_name() {
    for unread in [false, true] {
        let gateway = gateway_over(0, Some(1), |config| config.disable_retries = unread).await;
        let (url, recorded) = recording_worker(StatusCode::OK, json!({"text": "ok"})).await;
        registered(
            &gateway,
            &url,
            WorkerType::Regular,
            RuntimeType::Unspecified,
        );
        let app = gateway.create_app();

        let asked = json!({"text": "one request", "stream": false}).to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header(CONTENT_LENGTH, asked.len())
            .header("x-request-id", "explain-recorded")
            .header(PRIVATE_HEADER, "a-clients-value")
            .body(Body::from(asked))
            .expect("a request");
        let (status, _, answer) = sent(&app, request).await;
        assert_eq!(status, StatusCode::OK, "unread {unread}: {answer}");

        let recorded = recorded.lock().await;
        let routed = sent_to(&recorded, "/generate");
        assert_eq!(routed.len(), 1, "unread {unread}: {:?}", listed(&recorded));
        let headers = routed[0];
        assert_eq!(headers["x-request-id"], "explain-recorded", "{headers:?}");
        assert_eq!(headers.contains_key(CONTENT_LENGTH), !unread, "{headers:?}");
        let private = holds_the_private_header(&recorded);
        assert!(!private, "unread {unread}: {:?}", listed(&recorded));
        drop(recorded);
        let target = one_target(&app, "explain-recorded").await;
        assert_eq!(target, json!(url), "unread {unread}");

        gateway.shutdown().await;
    }
}

/// On the HTTP pair route no worker is sent it either. The prefill and the
/// decode of a pair record what they are sent of a generation request and
/// of a count of tokens: each request carries the request-id header the
/// client sent and none under the private name. Both requests are found by
/// the client's id, with the prefill as the one target.
#[tokio::test]
async fn a_pairs_workers_are_sent_no_header_of_the_private_name() {
    let mut config = RouterConfig::builder()
        .prefill_decode_mode(vec![], vec![])
        .host("127.0.0.1")
        .port(0)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    // The policy spec the pair mode's own settings mean: it binds a single
    // request, the count of tokens, behind its pairs.
    let spec = config.policy_document();
    sampled_under(&mut config, spec.expect("the pair mode's policy spec"), 1);
    let gateway = AppTestContext::new_with_config(config, vec![]).await;
    let (prefill, prefill_got) = recording_worker(StatusCode::OK, json!({"text": "ok"})).await;
    let (decode, decode_got) = recording_worker(StatusCode::OK, json!({"text": "ok"})).await;
    registered(
        &gateway,
        &prefill,
        WorkerType::Prefill,
        RuntimeType::Unspecified,
    );
    registered(
        &gateway,
        &decode,
        WorkerType::Decode,
        RuntimeType::Unspecified,
    );
    let app = gateway.create_app();

    let generation = json!({"text": "one request", "stream": false});
    let count = json!({
        "model": RECORDED_MODEL,
        "messages": [{"role": "user", "content": "count me"}],
    });
    for (path, asked, id) in [
        ("/generate", generation, "explain-recorded-pair"),
        ("/v1/messages/count_tokens", count, "explain-recorded-count"),
    ] {
        let mut request = posted(path, &asked);
        let headers = request.headers_mut();
        headers.insert("x-request-id", HeaderValue::from_static(id));
        headers.insert(PRIVATE_HEADER, HeaderValue::from_static("a-clients-value"));
        let (status, _, answer) = sent(&app, request).await;
        assert_eq!(status, StatusCode::OK, "{path}: {answer}");
        let target = one_target(&app, id).await;
        assert_eq!(target, json!(prefill), "{path}");
    }

    let (prefill_got, decode_got) = (prefill_got.lock().await, decode_got.lock().await);
    for (side, got) in [("prefill", &prefill_got), ("decode", &decode_got)] {
        let generations = sent_to(got, "/generate");
        assert!(!generations.is_empty(), "{side}: {:?}", listed(got));
        for headers in generations {
            assert_eq!(headers["x-request-id"], "explain-recorded-pair", "{side}");
        }
        assert!(!holds_the_private_header(got), "{side}: {:?}", listed(got));
    }
    // A count of tokens goes to the prefill alone.
    let counts = sent_to(&prefill_got, "/v1/messages/count_tokens");
    assert_eq!(counts.len(), 1, "{:?}", listed(&prefill_got));
    assert_eq!(counts[0]["x-request-id"], "explain-recorded-count");
    let counted_by_decode = sent_to(&decode_got, "/v1/messages/count_tokens");
    assert!(counted_by_decode.is_empty(), "{:?}", listed(&decode_got));
    drop((prefill_got, decode_got));

    gateway.shutdown().await;
}

/// A retry's decision stands under the gateway's own id too. A request
/// first sent to a worker that fails is sent again to another, with the
/// same headers: the newest decision kept under the id names the worker
/// that served the request, and neither worker is sent the private header.
#[tokio::test]
async fn a_retried_request_is_read_by_the_gateways_own_id() {
    let gateway = gateway_over(0, Some(1), |config| {
        config.retry = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 100,
            ..Default::default()
        };
    })
    .await;
    let failure = json!({"error": "a failing worker"});
    let (failing, failed) = recording_worker(StatusCode::INTERNAL_SERVER_ERROR, failure).await;
    let (serving, served) = recording_worker(StatusCode::OK, json!({"text": "ok"})).await;
    registered(
        &gateway,
        &failing,
        WorkerType::Regular,
        RuntimeType::Unspecified,
    );
    registered(
        &gateway,
        &serving,
        WorkerType::Regular,
        RuntimeType::Unspecified,
    );
    let app = gateway.create_app();

    let ids = [
        "explain-retry-a",
        "explain-retry-b",
        "explain-retry-c",
        "explain-retry-d",
    ];
    for id in ids {
        let mut request = generate_with_no_id();
        let header = HeaderValue::from_static(id);
        request.headers_mut().insert("x-request-id", header);
        let (status, headers, answer) = sent(&app, request).await;
        assert_eq!(status, StatusCode::OK, "{id}: {answer}");
        assert_eq!(headers["x-smg-routed-worker-id"], serving.as_str(), "{id}");
        let target = one_target(&app, id).await;
        assert_eq!(target, json!(serving), "{id}");
    }

    // Round robin sent some request to the failing worker first, and each
    // request reached the serving worker once.
    let (failed, served) = (failed.lock().await, served.lock().await);
    let sent_first = sent_to(&failed, "/generate").len();
    assert!(
        sent_first > 0,
        "no request failed first: {:?}",
        listed(&failed)
    );
    let served_once = sent_to(&served, "/generate").len();
    assert_eq!(served_once, ids.len(), "{:?}", listed(&served));
    assert!(!holds_the_private_header(&failed), "{:?}", listed(&failed));
    assert!(!holds_the_private_header(&served), "{:?}", listed(&served));
    drop((failed, served));

    gateway.shutdown().await;
}

/// A client cannot name the id a decision is kept under by the gateway's
/// private header: under a policy spec that keeps a sample the gateway
/// replaces what the client sent there with the id it chose.
#[tokio::test]
async fn a_clients_value_of_the_private_header_names_no_decision() {
    let gateway = gateway(Some(1)).await;
    let app = gateway.create_app();

    let mut request = generate_with_no_id();
    let forged = HeaderValue::from_static("forged-own");
    request.headers_mut().insert("x-smg-request-id", forged);
    let (status, headers, _) = sent(&app, request).await;
    assert_eq!(status, StatusCode::OK, "the request is served");
    let own = headers["x-request-id"].to_str().unwrap().to_owned();
    assert_ne!(own, "forged-own");
    let (status, _, kept) = sent(&app, explain(&own)).await;
    assert_eq!(status, StatusCode::OK, "{own}: {kept}");
    let (status, _, refused) = sent(&app, explain("forged-own")).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{refused}");
    assert_eq!(refused["code"], json!("DECISION_NOT_KEPT"), "{refused}");

    gateway.shutdown().await;
}

/// On the HTTP pair route too, a pair's request whose body states no id is
/// read back by the id its response's `x-request-id` header shows. The
/// decision kept is the pair's first one: one target, the prefill, with
/// its decode beside it. A pair's rerank request is read back the same way.
#[tokio::test]
async fn a_pairs_request_that_states_no_id_is_read_by_the_gateways_own_id() {
    let prefill = "http://127.0.0.1:19880".to_string();
    let decode = "http://127.0.0.1:19881".to_string();
    let mut config = RouterConfig::builder()
        .prefill_decode_mode(vec![(prefill.clone(), None)], vec![decode.clone()])
        .host("127.0.0.1")
        .port(0)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let spec = PolicySpec::pair_presets("round_robin", "round_robin");
    sampled_under(&mut config, spec.expect("the pair presets"), 1);
    let workers = vec![
        TestWorkerConfig::prefill(19880),
        TestWorkerConfig::decode(19881),
    ];
    let gateway = AppTestContext::new_with_config(config, workers).await;
    let app = gateway.create_app();

    let (status, headers, answer) = sent(&app, generate_with_no_id()).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let own = headers["x-request-id"].to_str().unwrap().to_owned();
    let (status, _, kept) = sent(&app, explain(&own)).await;
    assert_eq!(status, StatusCode::OK, "{own}: {kept}");
    assert_eq!(kept["plan_name"], json!("pd-pair"), "{own}: {kept}");
    let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 1, "{own}: {kept}");
    assert_eq!(targets[0]["url"], json!(prefill), "{own}: {kept}");
    let decode_url = &targets[0]["pair"]["decode_url"];
    assert_eq!(decode_url, &json!(decode), "{own}: {kept}");

    let asked = json!({"query": "a query", "documents": ["one", "two"], "model": "mock-model"});
    let (status, headers, answer) = sent(&app, posted("/rerank", &asked)).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let target = one_target(&app, &own_id(&headers)).await;
    assert_eq!(target, json!(prefill), "rerank");

    gateway.shutdown().await;
}

/// On the gRPC route too, a request whose body states no id, sent to a
/// gRPC worker, is read back by the id its response's `x-request-id`
/// header shows.
#[tokio::test]
async fn a_grpc_request_that_states_no_id_is_read_by_the_gateways_own_id() {
    const MODEL: &str = "explain-grpc-model";
    let port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 2,
    })
    .await;
    let mut config = grpc_router_config().regular_mode(vec![]).build_unchecked();
    sampled(&mut config, 1);
    let context = grpc_test_context(config, MODEL).await;
    let card = ModelCard::new(MODEL);
    let worker = grpc_worker(port, WorkerType::Regular, RuntimeType::TokenSpeed, card);
    let worker: Arc<dyn Worker> = Arc::new(worker.build());
    let router: Arc<dyn RouterTrait> = Arc::from(grpc_router(&context, vec![worker]).await);
    let app = create_test_app_with_context(router, context);

    let asked = json!({"model": MODEL, "text": "Hello world", "stream": false});
    let request = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(asked.to_string()))
        .expect("a request");
    let (status, headers, answer) = sent(&app, request).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let own = headers["x-request-id"].to_str().unwrap().to_owned();
    let (status, _, kept) = sent(&app, explain(&own)).await;
    assert_eq!(status, StatusCode::OK, "{own}: {kept}");
    let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 1, "{own}: {kept}");
}

/// On the gRPC pair route too, a pair's request whose body states no id,
/// sent to a gRPC prefill and decode, is read back by the id its response's
/// `x-request-id` header shows, where its body states an output limit and
/// where it states none. The decision kept is the pair's first one: one
/// target, the prefill.
#[tokio::test]
async fn a_grpc_pairs_request_that_states_no_id_is_read_by_the_gateways_own_id() {
    const MODEL: &str = "explain-grpc-pair-model";
    let canned = || {
        serve_canned_worker(Canned {
            model: MODEL,
            gen_delay: Duration::ZERO,
            output_tokens: 2,
        })
    };
    let (prefill, decode) = (canned().await, canned().await);
    let mut config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .build_unchecked();
    let spec = PolicySpec::pair_presets("round_robin", "round_robin");
    sampled_under(&mut config, spec.expect("the pair presets"), 1);
    let context = grpc_test_context(config, MODEL).await;
    let workers = grpc_workers(
        &[(prefill, WorkerType::Prefill), (decode, WorkerType::Decode)],
        RuntimeType::TokenSpeed,
        MODEL,
    );
    let router: Arc<dyn RouterTrait> = Arc::from(grpc_router(&context, workers).await);
    let app = create_test_app_with_context(router, context);

    // With a limit in the body and with none.
    for limit in [json!({}), json!({"max_new_tokens": 2})] {
        let asked = json!({
            "model": MODEL,
            "text": "Hello world",
            "stream": false,
            "sampling_params": limit,
        });
        let request = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(asked.to_string()))
            .expect("a request");
        let (status, headers, answer) = sent(&app, request).await;
        assert_eq!(status, StatusCode::OK, "{limit}: {answer}");
        let own = headers["x-request-id"].to_str().unwrap().to_owned();
        let (status, _, kept) = sent(&app, explain(&own)).await;
        assert_eq!(status, StatusCode::OK, "{limit}: {own}: {kept}");
        assert_eq!(kept["plan_name"], json!("pd-pair"), "{limit}: {kept}");
        let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
        assert_eq!(targets.len(), 1, "{limit}: {kept}");
        let sent_to = format!("grpc://127.0.0.1:{prefill}");
        assert_eq!(targets[0]["url"], json!(sent_to), "{limit}: {kept}");
    }
}

/// A run in a task of its own states the gateway's id too. A Responses
/// request with `background: true` is answered at once, and its run is
/// placed behind the answer, in a spawned task that holds the request's
/// headers. The run's decision is read by the id the answer's
/// `x-request-id` header shows.
#[tokio::test]
async fn a_background_run_is_read_by_the_gateways_own_id() {
    const MODEL: &str = "explain-grpc-background-model";
    let port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 2,
    })
    .await;
    let mut config = grpc_router_config().regular_mode(vec![]).build_unchecked();
    sampled(&mut config, 1);
    let context = grpc_test_context(config, MODEL).await;
    let card = ModelCard::new(MODEL);
    let worker = grpc_worker(port, WorkerType::Regular, RuntimeType::TokenSpeed, card);
    let worker: Arc<dyn Worker> = Arc::new(worker.build());
    let router: Arc<dyn RouterTrait> = Arc::from(grpc_router(&context, vec![worker]).await);
    let app = create_test_app_with_context(router, context);

    let asked = json!({"model": MODEL, "input": "Hello world", "background": true});
    let (status, headers, queued) = sent(&app, posted("/v1/responses", &asked)).await;
    assert_eq!(status, StatusCode::OK, "{queued}");
    assert_eq!(queued["status"], json!("queued"), "{queued}");
    let own = own_id(&headers);

    // The run places its request behind the answer: its decision is waited
    // for, ten seconds at most.
    let mut read = sent(&app, explain(&own)).await;
    for _ in 0..500 {
        if read.0 == StatusCode::OK {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        read = sent(&app, explain(&own)).await;
    }
    let (status, _, kept) = read;
    assert_eq!(status, StatusCode::OK, "{own}: {kept}");
    let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 1, "{own}: {kept}");
    let sent_to = format!("grpc://127.0.0.1:{port}");
    assert_eq!(targets[0]["url"], json!(sent_to), "{own}: {kept}");
}

/// An id that finds no kept decision is answered 404 in the control plane's
/// error envelope, with one of three reasons: Grand Central is off,
/// the policy spec keeps no sample, or the sample holds no decision of that
/// id. A run that turns Grand Central on for every context reads
/// the second reason in the first one's place.
#[tokio::test]
async fn an_id_that_finds_no_kept_decision_is_a_404_with_its_reason() {
    let off = if common::grand_central_forced() {
        "EXPLAIN_SAMPLE_OFF"
    } else {
        "GRAND_CENTRAL_OFF"
    };
    for (sample, code) in [
        (None, off),
        (Some(0), "EXPLAIN_SAMPLE_OFF"),
        (Some(1), "DECISION_NOT_KEPT"),
    ] {
        let gateway = gateway(sample).await;
        let app = gateway.create_app();
        let (status, _, _) = sent(&app, generate("explain-kept")).await;
        assert_eq!(status, StatusCode::OK, "{sample:?}: the request is served");

        let (status, _, refused) = sent(&app, explain("explain-other")).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{sample:?}: {refused}");
        assert_eq!(refused["code"], json!(code), "{sample:?}: {refused}");
        assert!(refused["error"].is_string(), "{sample:?}: {refused}");

        gateway.shutdown().await;
    }
}
