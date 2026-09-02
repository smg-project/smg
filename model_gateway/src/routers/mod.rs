//! Router implementations

use std::fmt::Debug;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    rerank::RerankRequest,
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};

use crate::middleware::TenantRequestMeta;

pub mod anthropic;
pub mod common;
pub mod conversations;
pub mod error;
pub mod factory;
pub mod gemini;
pub mod grpc;
pub mod http;
pub mod openai;
pub mod parse;
pub mod responses;
pub mod router_manager;
pub mod tokenize;

pub use common::body_policy::BodyPolicy;
pub use factory::RouterFactory;
// Re-export HTTP routers for convenience
pub use http::{pd_router, pd_types, router};

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
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not implemented",
        )
            .into_response()
    }

    /// Get server information
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Server info not implemented").into_response()
    }

    /// Get available models
    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Get models not implemented").into_response()
    }

    /// Get model information
    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Get model info not implemented",
        )
            .into_response()
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
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not implemented",
        )
            .into_response()
    }

    /// Route a chat completion request
    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ChatCompletionRequest,
        _model_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Chat completions not implemented",
        )
            .into_response()
    }

    /// Route a completion request
    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CompletionRequest,
        _model_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented",
        )
            .into_response()
    }

    /// Route a responses request
    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ResponsesRequest,
        _model_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses endpoint not implemented",
        )
            .into_response()
    }

    /// Cancel a background response by id
    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Cancel response not implemented",
        )
            .into_response()
    }

    /// Route embedding requests (OpenAI-compatible /v1/embeddings)
    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: EmbeddingRequest,
        _model_id: &str,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Embeddings not implemented").into_response()
    }

    /// Route classification requests (OpenAI-compatible /v1/classify)
    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ClassifyRequest,
        _model_id: &str,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Classify not implemented").into_response()
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
        (
            StatusCode::NOT_IMPLEMENTED,
            "Audio transcriptions not implemented",
        )
            .into_response()
    }

    /// Route rerank requests
    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: RerankRequest,
        _model_id: &str,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Rerank not implemented").into_response()
    }

    /// Route Anthropic Messages API requests (/v1/messages)
    async fn route_messages(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CreateMessageRequest,
        _model_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Messages API not yet implemented for this router",
        )
            .into_response()
    }

    /// Route Gemini Interactions API requests (/v1/interactions)
    async fn route_interactions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: InteractionsRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Interactions API not implemented for this router",
        )
            .into_response()
    }

    /// Route a realtime session create request (/v1/realtime/sessions)
    async fn route_realtime_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeSessionCreateRequest,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Realtime sessions not implemented",
        )
            .into_response()
    }

    /// Route a realtime client secret create request (/v1/realtime/client_secrets)
    async fn route_realtime_client_secret(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Realtime client secrets not implemented",
        )
            .into_response()
    }

    /// Route a realtime transcription session create request (/v1/realtime/transcription_sessions)
    async fn route_realtime_transcription_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Realtime transcription sessions not implemented",
        )
            .into_response()
    }

    /// Route a realtime WebSocket upgrade request
    async fn route_realtime_ws(&self, _req: Request<Body>, _model_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Realtime WebSocket not implemented",
        )
            .into_response()
    }

    /// Route a realtime WebRTC upgrade request
    async fn route_realtime_webrtc(&self, _req: Request<Body>, _model_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Realtime WebRTC not implemented",
        )
            .into_response()
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
