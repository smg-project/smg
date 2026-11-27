use axum::http::StatusCode;

use super::*;

/// Minimal stub implementing RouterTrait so we can test the default
/// `is_pd_mode` logic without spinning up a real router.
#[derive(Debug)]
struct StubRouter {
    type_name: &'static str,
}

#[async_trait]
impl RouterTrait for StubRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn router_type(&self) -> &'static str {
        self.type_name
    }
}

/// The status of an answer and the code and message of its JSON error object.
pub(crate) async fn error_shape(answer: Response) -> (StatusCode, String, String) {
    let status = answer.status();
    let body = axum::body::to_bytes(answer.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
    let field = |name: &str| {
        body["error"][name]
            .as_str()
            .unwrap_or_else(|| panic!("error.{name} in {body}"))
            .to_owned()
    };
    (status, field("code"), field("message"))
}

fn body<T: serde::de::DeserializeOwned>(json: serde_json::Value) -> T {
    serde_json::from_value(json).expect("a request body")
}

/// Every default answer of the trait is the JSON error object, a 501
/// under the endpoint's own code, like the refusals the routers write.
#[tokio::test]
async fn every_default_answer_is_the_json_error_object_under_its_code() {
    use serde_json::json;

    use crate::middleware::{RouteRequestMeta, TenantKey};

    let router = StubRouter { type_name: "stub" };
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let empty = || Request::new(Body::empty());
    let audio = AudioFile {
        bytes: bytes::Bytes::new(),
        file_name: "a.wav".to_owned(),
        content_type: None,
    };
    let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let answers = [
        (
            router.health_generate(empty()).await,
            "health_generate_not_implemented",
            "Health generate not implemented",
        ),
        (
            router.get_server_info(empty()).await,
            "server_info_not_implemented",
            "Server info not implemented",
        ),
        (
            router.get_model_info(empty()).await,
            "model_info_not_implemented",
            "Get model info not implemented",
        ),
        (
            router
                .route_generate(None, &meta, body(json!({"text": "hi"})), "m")
                .await,
            "generate_not_implemented",
            "Generate endpoint not implemented",
        ),
        (
            router.route_chat(None, &meta, Arc::new(body(chat.clone())), "m").await,
            "chat_completions_not_implemented",
            "Chat completions not implemented",
        ),
        (
            router
                .route_completion(None, &meta, body(json!({"model": "m", "prompt": "hi"})), "m")
                .await,
            "completions_not_implemented",
            "Completion endpoint not implemented",
        ),
        (
            router
                .route_responses(None, &meta, body(json!({"model": "m", "input": "hi"})), "m")
                .await,
            "responses_not_implemented",
            "Responses endpoint not implemented",
        ),
        (
            router.cancel_response(None, &meta, "resp_1").await,
            "cancel_response_not_implemented",
            "Cancel response not implemented",
        ),
        (
            router
                .route_embeddings(None, &meta, body(json!({"model": "m", "input": "hi"})), "m")
                .await,
            "embeddings_not_implemented",
            "Embeddings not implemented",
        ),
        (
            router
                .route_classify(None, &meta, body(json!({"model": "m", "input": "hi"})), "m")
                .await,
            "classify_not_implemented",
            "Classify not implemented",
        ),
        (
            router
                .route_audio_transcriptions(
                    None,
                    &meta,
                    &TranscriptionRequest::default(),
                    audio,
                    "m",
                )
                .await,
            "audio_transcriptions_not_implemented",
            "Audio transcriptions not implemented",
        ),
        (
            router
                .route_rerank(
                    None,
                    &meta,
                    body(json!({"query": "q", "documents": ["d"], "model": "m"})),
                    "m",
                )
                .await,
            "rerank_not_implemented",
            "Rerank is served by regular HTTP workers that answer POST /v1/rerank with \
                 ranked results; this router kind has no rerank path",
        ),
        (
            router
                .route_messages(
                    None,
                    &meta,
                    Arc::new(body(json!({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}))),
                    "m",
                )
                .await,
            "messages_not_implemented",
            "Messages API not yet implemented for this router",
        ),
        (
            router
                .route_messages_count_tokens(
                    None,
                    &meta,
                    body(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})),
                    "m",
                )
                .await,
            "count_message_tokens_not_implemented",
            "Messages token counting not implemented for this router",
        ),
        (
            router
                .route_interactions(None, &meta, body(json!({"input": "hi"})), Some("m"))
                .await,
            "interactions_not_implemented",
            "Interactions API not implemented for this router",
        ),
        (
            router
                .route_realtime_session(None, &body(json!({"type": "realtime"})))
                .await,
            "realtime_sessions_not_implemented",
            "Realtime sessions not implemented",
        ),
        (
            router
                .route_realtime_client_secret(None, &body(json!({"session": {"type": "realtime"}})))
                .await,
            "realtime_client_secrets_not_implemented",
            "Realtime client secrets not implemented",
        ),
        (
            router
                .route_realtime_transcription_session(None, &body(json!({"type": "transcription"})))
                .await,
            "realtime_transcription_sessions_not_implemented",
            "Realtime transcription sessions not implemented",
        ),
        (
            router.route_realtime_ws(empty(), "m").await,
            "realtime_websocket_not_implemented",
            "Realtime WebSocket not implemented",
        ),
        (
            router.route_realtime_webrtc(empty(), "m").await,
            "realtime_webrtc_not_implemented",
            "Realtime WebRTC not implemented",
        ),
    ];
    for (answer, code, message) in answers {
        assert_eq!(
            error_shape(answer).await,
            (
                StatusCode::NOT_IMPLEMENTED,
                code.to_owned(),
                message.to_owned()
            )
        );
    }
}

#[test]
fn test_is_pd_mode_for_pd_router_types() {
    // "pd" (HTTP PD) and "grpc_pd" should both be recognized as PD mode
    let pd = StubRouter { type_name: "pd" };
    assert!(pd.is_pd_mode());

    let grpc_pd = StubRouter {
        type_name: "grpc_pd",
    };
    assert!(grpc_pd.is_pd_mode());
}

#[test]
fn test_is_pd_mode_false_for_non_pd_router_types() {
    for name in &[
        "openai",
        "regular",
        "grpc",
        "gemini",
        "anthropic",
        "manager",
    ] {
        let router = StubRouter { type_name: name };
        assert!(
            !router.is_pd_mode(),
            "router_type {name:?} should NOT be pd mode"
        );
    }
}
