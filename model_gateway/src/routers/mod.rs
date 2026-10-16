//! Router implementations

use std::fmt::Debug;

use async_trait::async_trait;
use axum::{body::Body, extract::Request, http::HeaderMap, response::Response};
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    decisions::DecisionsRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::{CountMessageTokensRequest, CreateMessageRequest},
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::RerankRequest,
    responses::ResponsesRequest,
    systemone::SystemOneRequest,
    transcription::{AudioFile, TranscriptionRequest},
};

use crate::middleware::TenantRequestMeta;

pub mod common;
pub use smg_external_router::error;
pub mod external;
pub mod factory;
pub mod gateway;
pub mod grpc;
pub mod http;
pub(crate) mod provider_support;
pub mod response_cache;

pub use common::body_policy::BodyPolicy;
pub use factory::RouterFactory;
// Re-export HTTP routers for convenience
pub use http::{pd_router, pd_types, router};

pub(crate) const PD_PREFILL_QUEUE_FULL: &str = "pd_prefill_queue_full";
pub(crate) const PD_PREFILL_QUEUE_TIMEOUT: &str = "pd_prefill_queue_timeout";

/// The client answer when the Prefill admission queue has no room.
pub(crate) fn prefill_queue_full() -> Response {
    error::too_many_requests(PD_PREFILL_QUEUE_FULL, "Prefill admission queue is full")
}

/// The client answer when a queued request waited out its Prefill admission
/// timeout.
pub(crate) fn prefill_queue_timeout() -> Response {
    error::too_many_requests(
        PD_PREFILL_QUEUE_TIMEOUT,
        "Timed out waiting for Prefill admission",
    )
}

/// Core trait for all router implementations
///
/// This trait provides a unified interface for routing requests,
/// regardless of whether it's a regular router or PD router.
#[async_trait]
pub trait RouterTrait: Send + Sync + Debug {
    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Buffering is always correct and is the default; override only when
    /// the family forwards bodies verbatim.
    fn request_body_policy(&self) -> BodyPolicy {
        BodyPolicy::MustBuffer(self.router_type())
    }

    /// Route a health generate request
    async fn health_generate(&self, _req: Request<Body>) -> Response {
        error::not_implemented(
            "health_generate_not_implemented",
            "Health generate not implemented",
        )
    }

    /// Get server information
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        error::not_implemented("server_info_not_implemented", "Server info not implemented")
    }

    /// Get model information
    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        error::not_implemented(
            "model_info_not_implemented",
            "Get model info not implemented",
        )
    }

    /// Route a generate request
    ///
    /// Typed-JSON route methods take the parsed body by value: the dispatching
    /// router owns it and can free it as soon as the upstream bytes exist,
    /// instead of the handler pinning a copy for the whole response.
    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: GenerateRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "generate_not_implemented",
            "Generate endpoint not implemented",
        )
    }

    /// Route a chat completion request
    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ChatCompletionRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "chat_completions_not_implemented",
            "Chat completions not implemented",
        )
    }

    /// Route a completion request
    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CompletionRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "completions_not_implemented",
            "Completion endpoint not implemented",
        )
    }

    /// Route a responses request
    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ResponsesRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "responses_not_implemented",
            "Responses endpoint not implemented",
        )
    }

    /// Cancel a background response by id, for the caller `tenant_meta`
    /// describes (a response of another tenant answers as an unknown id does).
    async fn cancel_response(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _response_id: &str,
    ) -> Response {
        error::not_implemented(
            "cancel_response_not_implemented",
            "Cancel response not implemented",
        )
    }

    /// Route embedding requests (OpenAI-compatible /v1/embeddings)
    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: EmbeddingRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented("embeddings_not_implemented", "Embeddings not implemented")
    }

    /// Route classification requests (OpenAI-compatible /v1/classify)
    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ClassifyRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented("classify_not_implemented", "Classify not implemented")
    }

    /// Route OpenAI Decisions requests on regular HTTP workers.
    async fn route_decisions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: DecisionsRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "decisions_not_supported",
            "Decisions is supported by regular HTTP and SGLang gRPC routers",
        )
    }

    /// Route native SystemOne requests on regular HTTP workers.
    async fn route_systemone(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: SystemOneRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "systemone_not_supported",
            "SystemOne is supported only by the regular HTTP router",
        )
    }

    /// Route audio transcription requests (OpenAI-compatible /v1/audio/transcriptions).
    ///
    /// Unlike the JSON-bodied endpoints, `/v1/audio/transcriptions` uses
    /// multipart/form-data: the server handler parses the form, packs text
    /// fields into `body` and the audio part into `audio`, and routers forward
    /// both to a worker capable of audio transcription.
    async fn route_audio_transcriptions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: &TranscriptionRequest,
        _audio: AudioFile,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "audio_transcriptions_not_implemented",
            "Audio transcriptions not implemented",
        )
    }

    /// Route rerank requests
    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: RerankRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented("rerank_not_implemented", "Rerank not implemented")
    }

    /// Route Anthropic Messages API requests (/v1/messages)
    async fn route_messages(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CreateMessageRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "messages_not_implemented",
            "Messages API not yet implemented for this router",
        )
    }

    /// Route Anthropic Messages token counting (/v1/messages/count_tokens)
    async fn route_messages_count_tokens(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CountMessageTokensRequest,
        _model_id: &str,
    ) -> Response {
        error::not_implemented(
            "count_message_tokens_not_implemented",
            "Messages token counting not implemented for this router",
        )
    }

    /// Route Gemini Interactions API requests (/v1/interactions)
    async fn route_interactions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: InteractionsRequest,
        _model_id: Option<&str>,
    ) -> Response {
        error::not_implemented(
            "interactions_not_implemented",
            "Interactions API not implemented for this router",
        )
    }

    /// Route a realtime session create request (/v1/realtime/sessions)
    async fn route_realtime_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeSessionCreateRequest,
    ) -> Response {
        error::not_implemented(
            "realtime_sessions_not_implemented",
            "Realtime sessions not implemented",
        )
    }

    /// Route a realtime client secret create request (/v1/realtime/client_secrets)
    async fn route_realtime_client_secret(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        error::not_implemented(
            "realtime_client_secrets_not_implemented",
            "Realtime client secrets not implemented",
        )
    }

    /// Route a realtime transcription session create request (/v1/realtime/transcription_sessions)
    async fn route_realtime_transcription_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        error::not_implemented(
            "realtime_transcription_sessions_not_implemented",
            "Realtime transcription sessions not implemented",
        )
    }

    /// Route a realtime WebSocket upgrade request
    async fn route_realtime_ws(&self, _req: Request<Body>, _model_id: &str) -> Response {
        error::not_implemented(
            "realtime_websocket_not_implemented",
            "Realtime WebSocket not implemented",
        )
    }

    /// Route a realtime WebRTC upgrade request
    async fn route_realtime_webrtc(&self, _req: Request<Body>, _model_id: &str) -> Response {
        error::not_implemented(
            "realtime_webrtc_not_implemented",
            "Realtime WebRTC not implemented",
        )
    }

    /// Get router type name
    fn router_type(&self) -> &'static str;

    /// Check if this is a PD router
    fn is_pd_mode(&self) -> bool {
        matches!(self.router_type(), "pd" | "grpc_pd")
    }
}

#[cfg(test)]
mod tests {
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
    async fn error_shape(answer: Response) -> (StatusCode, String, String) {
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
                router.route_chat(None, &meta, body(chat.clone()), "m").await,
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
                "Rerank not implemented",
            ),
            (
                router
                    .route_messages(
                        None,
                        &meta,
                        body(json!({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})),
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
}
