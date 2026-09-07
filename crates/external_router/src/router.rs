//! The contract between the gateway and a router for a third-party provider.

use std::{any::Any, fmt::Debug, future::Future, pin::Pin, sync::Arc};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    interactions::InteractionsRequest,
    messages::CreateMessageRequest,
    realtime_session::{
        RealtimeClientSecretCreateRequest, RealtimeSessionCreateRequest,
        RealtimeTranscriptionSessionCreateRequest,
    },
    responses::ResponsesRequest,
    worker::ProviderType,
};

use crate::{tenant::TenantRequestMeta, ExternalContext};

/// A router that fronts a third-party provider. Implement only the
/// endpoints the provider serves; the rest answer 501.
#[async_trait]
pub trait ExternalRouter: Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;

    /// The name reported as the router type.
    fn router_type(&self) -> &'static str;

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        not_implemented("Health generate not implemented")
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        not_implemented("Server info not implemented")
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ChatCompletionRequest,
        _model_id: &str,
    ) -> Response {
        not_implemented("Chat completions not implemented")
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: ResponsesRequest,
        _model_id: &str,
    ) -> Response {
        not_implemented("Responses endpoint not implemented")
    }

    async fn route_messages(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: CreateMessageRequest,
        _model_id: &str,
    ) -> Response {
        not_implemented("Messages API not yet implemented for this router")
    }

    async fn route_interactions(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: InteractionsRequest,
        _model_id: Option<&str>,
    ) -> Response {
        not_implemented("Interactions API not implemented for this router")
    }

    async fn route_realtime_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeSessionCreateRequest,
    ) -> Response {
        not_implemented("Realtime sessions not implemented")
    }

    async fn route_realtime_client_secret(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeClientSecretCreateRequest,
    ) -> Response {
        not_implemented("Realtime client secrets not implemented")
    }

    async fn route_realtime_transcription_session(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RealtimeTranscriptionSessionCreateRequest,
    ) -> Response {
        not_implemented("Realtime transcription sessions not implemented")
    }

    async fn route_realtime_ws(&self, _req: Request<Body>, _model_id: &str) -> Response {
        not_implemented("Realtime WebSocket not implemented")
    }

    async fn route_realtime_webrtc(&self, _req: Request<Body>, _model_id: &str) -> Response {
        not_implemented("Realtime WebRTC not implemented")
    }
}

fn not_implemented(message: &'static str) -> Response {
    (StatusCode::NOT_IMPLEMENTED, message).into_response()
}

/// The future a router constructor returns.
pub type BuildFuture =
    Pin<Box<dyn Future<Output = Result<Arc<dyn ExternalRouter>, String>> + Send>>;

/// How the gateway learns about one external router: its identity, the
/// providers it takes, and how to build it.
#[derive(Debug, Clone, Copy)]
pub struct ExternalRouterSpec {
    /// The id the gateway registers the router under.
    pub router_id: &'static str,
    /// The `--backend` / routing-mode name that selects this router alone.
    pub backend: &'static str,
    /// Human name for logs.
    pub label: &'static str,
    /// The gateway Cargo feature that compiles the router in.
    pub feature: &'static str,
    /// Whether the router takes workers of a provider.
    pub serves: fn(&ProviderType) -> bool,
    /// Whether the router takes external workers that name no provider.
    pub fallback: bool,
    pub build: fn(ExternalContext) -> BuildFuture,
}

impl ExternalRouterSpec {
    /// Whether the router takes a worker of `provider` (`None`: an external
    /// worker that names no provider).
    pub fn takes(&self, provider: Option<&ProviderType>) -> bool {
        match provider {
            Some(provider) => (self.serves)(provider),
            None => self.fallback,
        }
    }
}

/// Router ids of the built-in external routers, known whether or not they
/// are compiled in, so the gateway can name what is missing.
pub mod ids {
    pub const OPENAI: &str = "http-openai";
    pub const ANTHROPIC: &str = "http-anthropic";
    pub const GEMINI: &str = "http-gemini";
}

/// Every external router this build carries.
pub fn builtin_routers() -> Vec<ExternalRouterSpec> {
    [openai_spec(), anthropic_spec(), gemini_spec()]
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(feature = "openai")]
fn openai_spec() -> Option<ExternalRouterSpec> {
    Some(crate::openai::spec())
}

#[cfg(not(feature = "openai"))]
fn openai_spec() -> Option<ExternalRouterSpec> {
    None
}

#[cfg(feature = "anthropic")]
fn anthropic_spec() -> Option<ExternalRouterSpec> {
    Some(crate::anthropic::spec())
}

#[cfg(not(feature = "anthropic"))]
fn anthropic_spec() -> Option<ExternalRouterSpec> {
    None
}

#[cfg(feature = "gemini")]
fn gemini_spec() -> Option<ExternalRouterSpec> {
    Some(crate::gemini::spec())
}

#[cfg(not(feature = "gemini"))]
fn gemini_spec() -> Option<ExternalRouterSpec> {
    None
}

/// The compiled-in router that `--backend <name>` selects alone.
pub fn spec_for_backend(backend: &str) -> Option<ExternalRouterSpec> {
    builtin_routers().into_iter().find(|s| s.backend == backend)
}

/// The compiled-in router that takes workers of `provider`.
pub fn spec_for_provider(provider: Option<&ProviderType>) -> Option<ExternalRouterSpec> {
    builtin_routers().into_iter().find(|s| s.takes(provider))
}
