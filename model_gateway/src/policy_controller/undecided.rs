//! The router of a process the policy controller was picked for, while the
//! controller decides nothing.

use std::{any::Any, sync::Arc};

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

use crate::{
    middleware::TenantRequestMeta,
    routers::{error, RouterTrait},
};

/// The code of the refusal, as a client reads it in the error object.
pub(crate) const UNDECIDED: &str = "not_decided_by_policy_controller";

/// The one router of a process the policy controller was picked for, mounted
/// in the place of every classic router: it refuses each request by name.
/// No classic router is built beside it, so no classic policy decides.
#[derive(Debug)]
pub(crate) struct Undecided;

/// The refusal: the request is not served, and a retry changes nothing.
fn undecided() -> Response {
    error::not_implemented(
        UNDECIDED,
        "The policy controller is picked for this gateway and decides no request yet",
    )
}

#[async_trait]
impl RouterTrait for Undecided {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn router_type(&self) -> &'static str {
        "policy_controller_undecided"
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        undecided()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        undecided()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        undecided()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: GenerateRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: Arc<ChatCompletionRequest>,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CompletionRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ResponsesRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn cancel_response(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _response_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: EmbeddingRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ClassifyRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_decisions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: DecisionsRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_systemone(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: SystemOneRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_audio_transcriptions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: &TranscriptionRequest,
        _audio: AudioFile,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: RerankRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_messages(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: Arc<CreateMessageRequest>,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_messages_count_tokens(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CountMessageTokensRequest,
        _model_id: &str,
    ) -> Response {
        undecided()
    }

    async fn route_interactions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: InteractionsRequest,
        _model_id: Option<&str>,
    ) -> Response {
        undecided()
    }

    async fn route_realtime_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeSessionCreateRequest,
    ) -> Response {
        undecided()
    }

    async fn route_realtime_client_secret(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        undecided()
    }

    async fn route_realtime_transcription_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        undecided()
    }

    async fn route_realtime_ws(&self, _req: Request<Body>, _model_id: &str) -> Response {
        undecided()
    }

    async fn route_realtime_webrtc(&self, _req: Request<Body>, _model_id: &str) -> Response {
        undecided()
    }
}
