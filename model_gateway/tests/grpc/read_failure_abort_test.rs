//! A regular request whose engine stream fails mid-generation (the stream's
//! error status after some tokens, before the choice's `Complete`) ends at
//! the client with the failure's event, and ends the engine's request with
//! the stream's drop: the stream is dropped un-marked, so the drop sends the
//! engine the abort, and a generation the engine may still be running gives
//! its slot back within a step. A stream marked completed at the failure
//! would send no abort and leave such a generation to run to `max_tokens`
//! for nobody. The chat stream and the Messages stream, each on an engine
//! that records the abort for its request. A stream the gateway's own stop
//! closed, whose engine has sent neither its `Complete` nor a chunk past the
//! stop, is the same case: the engine may still be generating, so a read
//! that fails there aborts it too (the chat stream once released it as
//! completed, counting the stop's close as the engine's).

use std::{sync::atomic::Ordering, time::Duration};

use axum::{body::to_bytes, http::StatusCode};
use common::{
    decode_stream::chunk,
    engine_answers::{Aborts, VllmAnswers, VllmGenerateStream},
    grpc_harness::{build_grpc_router, grpc_router_config, grpc_workers, serve_engine},
};
use futures::stream;
use serde_json::{json, Value};
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use tokio::time::timeout;
use tonic::{Request, Response, Status};

use crate::common;

const MODEL: &str = "read-failure-abort-test-model";
/// The tokens the engine streams before its failure: the mock tokenizer's
/// `Hello` and `world`.
const TOKENS_BEFORE_THE_FAILURE: u32 = 2;
/// The engine's second token, as the mock tokenizer decodes it: what a
/// request's stop names.
const SECOND_TOKEN: &str = "world";
/// How long the abort may take to reach the engine once the client's stream
/// has ended: the drop sends it at once, the bound covers a loaded machine.
const ABORT_WITHIN: Duration = Duration::from_secs(5);
/// How long the client's stream may take to end after the failure.
const STREAM_ENDS_WITHIN: Duration = Duration::from_secs(10);

/// An engine whose every generate stream fails after
/// [`TOKENS_BEFORE_THE_FAILURE`] chunks, before any `Complete`, and that
/// records the abort for the request.
#[derive(Clone, Default)]
struct FailsMidGeneration {
    aborts: Aborts,
}

#[tonic::async_trait]
impl VllmAnswers for FailsMidGeneration {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        self.aborts.calls.fetch_add(1, Ordering::SeqCst);
        let items: Vec<Result<proto::GenerateResponse, Status>> = (1..=TOKENS_BEFORE_THE_FAILURE)
            .map(|id| Ok(chunk(id)))
            .chain(std::iter::once(Err(Status::internal(
                "the engine lost the request mid-generation",
            ))))
            .collect();
        Ok(Response::new(Box::pin(stream::iter(items))))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        self.aborts.note();
        Ok(Response::new(proto::AbortResponse::default()))
    }
}

/// The streams under the pin.
#[derive(Clone, Copy)]
enum Route {
    Chat,
    Messages,
}

/// Whether the request names a stop the gateway matches in the engine's
/// second token: the choice is then closed by the gateway's own stop when
/// the engine's stream fails, with no `Complete` and no chunk past the stop.
#[derive(Clone, Copy)]
enum Stop {
    None,
    Matched,
}

/// A router over one regular worker, `engine` behind it.
async fn router_over(engine: FailsMidGeneration) -> Box<dyn RouterTrait> {
    let port = serve_engine(engine.into_server()).await;
    build_grpc_router(
        grpc_router_config()
            .regular_mode(vec![])
            .random_policy()
            .build_unchecked(),
        MODEL,
        grpc_workers(&[(port, WorkerType::Regular)], RuntimeType::Vllm, MODEL),
    )
    .await
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("read-failure-abort-test-tenant"))
}

/// One streamed request, the same words on both routes, with the stop under
/// the route's own key when one is asked.
fn streamed_request(route: Route, stop: Stop) -> Value {
    let mut request = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": true,
    });
    if let Stop::Matched = stop {
        let key = match route {
            Route::Chat => "stop",
            Route::Messages => "stop_sequences",
        };
        request[key] = json!([SECOND_TOKEN]);
    }
    request
}

/// One streamed request on `route`, read to the end of its body: the stream
/// opened on the engine's first chunk, so the status is 200, and the body
/// ends with the failure's event once the engine's stream has failed.
#[expect(clippy::expect_used, reason = "test helper")]
async fn streamed_to_the_end(router: &dyn RouterTrait, route: Route, stop: Stop) -> String {
    let tenant = tenant();
    let request = streamed_request(route, stop);
    let response = match route {
        Route::Chat => {
            let request = serde_json::from_value(request).expect("chat request");
            router.route_chat(None, &tenant, request, MODEL).await
        }
        Route::Messages => {
            let request = serde_json::from_value(request).expect("Messages request");
            router.route_messages(None, &tenant, request, MODEL).await
        }
    };
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = timeout(STREAM_ENDS_WITHIN, to_bytes(response.into_body(), 1 << 20))
        .await
        .expect("the client's stream ends after the engine's failure")
        .expect("the body");
    String::from_utf8(bytes.to_vec()).expect("an SSE body")
}

/// The failure on `route` ends the client's stream, and the engine is asked
/// to abort the request within the bound: the stream was dropped un-marked.
async fn assert_the_failure_aborts_the_engines_request(route: Route, stop: Stop) {
    let engine = FailsMidGeneration::default();
    let aborts = engine.aborts.clone();
    let router = router_over(engine).await;
    let body = streamed_to_the_end(router.as_ref(), route, stop).await;
    assert!(
        aborts.first_within(ABORT_WITHIN).await.is_some(),
        "the engine was not asked to abort the request within {ABORT_WITHIN:?} of the \
         failure; the client read:\n{body}"
    );
    assert_eq!(
        aborts.calls.load(Ordering::SeqCst),
        1,
        "one generate call: a stream that failed after its first token is not retried"
    );
}

/// A chat stream whose read fails mid-generation sends the engine the abort
/// for the request, as a client's leaving does.
#[tokio::test]
async fn a_chat_read_that_fails_mid_generation_aborts_the_engines_request() {
    assert_the_failure_aborts_the_engines_request(Route::Chat, Stop::None).await;
}

/// The same for the Messages stream.
#[tokio::test]
async fn a_messages_read_that_fails_mid_generation_aborts_the_engines_request() {
    assert_the_failure_aborts_the_engines_request(Route::Messages, Stop::None).await;
}

/// A chat stream whose read fails after the gateway's own stop closed the
/// choice, before the engine's `Complete` or a chunk past the stop, sends the
/// engine the abort too: the stop's close is not the engine's.
#[tokio::test]
async fn a_chat_read_that_fails_after_the_gateways_stop_aborts_the_engines_request() {
    assert_the_failure_aborts_the_engines_request(Route::Chat, Stop::Matched).await;
}

/// The same for the Messages stream.
#[tokio::test]
async fn a_messages_read_that_fails_after_the_gateways_stop_aborts_the_engines_request() {
    assert_the_failure_aborts_the_engines_request(Route::Messages, Stop::Matched).await;
}
