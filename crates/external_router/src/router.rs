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

/// How the gateway knows one external router: its identity, the providers it
/// takes, whether this build carries it, and how to build it.
#[derive(Debug, Clone, Copy)]
pub struct ExternalRouterSpec {
    /// The id the gateway registers the router under.
    pub router_id: &'static str,
    /// The `--backend` / routing-mode name that selects this router alone.
    pub backend: &'static str,
    /// Human name for logs and messages.
    pub label: &'static str,
    /// The gateway Cargo feature that compiles the router in.
    pub feature: &'static str,
    /// Whether the router takes workers of a provider.
    pub serves: fn(&ProviderType) -> bool,
    /// Whether the router takes external workers that name no provider.
    pub fallback: bool,
    /// Whether this build carries the router.
    pub compiled: bool,
    /// Build the router. On a build without it, the error names the feature.
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

    /// Why a build without this router cannot serve it.
    pub fn not_compiled(&self) -> String {
        format!(
            "{} routing is not compiled into this build; rebuild with the `{}` Cargo feature",
            self.label, self.feature
        )
    }
}

/// Router ids of the built-in external routers.
pub mod ids {
    pub const OPENAI: &str = "http-openai";
    pub const ANTHROPIC: &str = "http-anthropic";
    pub const GEMINI: &str = "http-gemini";
}

/// The built-in routers, known whether or not this build carries them, so
/// admission and dispatch resolve a provider the same way and a missing
/// router can be named.
pub mod known {
    use openai_protocol::worker::ProviderType;

    use super::{ids, BuildFuture, ExternalContext, ExternalRouterSpec};

    /// The OpenAI-compatible router: OpenAI, xAI and custom providers, and
    /// the fallback for external workers that name no provider.
    pub const OPENAI: ExternalRouterSpec = ExternalRouterSpec {
        router_id: ids::OPENAI,
        backend: "openai",
        label: "OpenAI",
        feature: "provider-openai",
        serves: openai_serves,
        fallback: true,
        compiled: cfg!(feature = "openai"),
        build: openai_build,
    };

    pub const ANTHROPIC: ExternalRouterSpec = ExternalRouterSpec {
        router_id: ids::ANTHROPIC,
        backend: "anthropic",
        label: "Anthropic",
        feature: "provider-anthropic",
        serves: anthropic_serves,
        fallback: false,
        compiled: cfg!(feature = "anthropic"),
        build: anthropic_build,
    };

    pub const GEMINI: ExternalRouterSpec = ExternalRouterSpec {
        router_id: ids::GEMINI,
        backend: "gemini",
        label: "Gemini",
        feature: "provider-gemini",
        serves: gemini_serves,
        fallback: false,
        compiled: cfg!(feature = "gemini"),
        build: gemini_build,
    };

    /// Every built-in router, in resolution order.
    pub fn all() -> [ExternalRouterSpec; 3] {
        [OPENAI, ANTHROPIC, GEMINI]
    }

    fn openai_serves(provider: &ProviderType) -> bool {
        matches!(
            provider,
            ProviderType::OpenAI | ProviderType::XAI | ProviderType::Custom(_)
        )
    }

    fn anthropic_serves(provider: &ProviderType) -> bool {
        matches!(provider, ProviderType::Anthropic)
    }

    fn gemini_serves(provider: &ProviderType) -> bool {
        matches!(provider, ProviderType::Gemini)
    }

    #[cfg(feature = "openai")]
    fn openai_build(ctx: ExternalContext) -> BuildFuture {
        use std::sync::Arc;

        use super::ExternalRouter;
        Box::pin(async move {
            crate::openai::OpenAIRouter::new(&ctx)
                .await
                .map(|router| Arc::new(router) as Arc<dyn ExternalRouter>)
        })
    }

    #[cfg(not(feature = "openai"))]
    fn openai_build(_ctx: ExternalContext) -> BuildFuture {
        Box::pin(async { Err(OPENAI.not_compiled()) })
    }

    #[cfg(feature = "anthropic")]
    fn anthropic_build(ctx: ExternalContext) -> BuildFuture {
        use std::sync::Arc;

        use super::ExternalRouter;
        Box::pin(async move {
            crate::anthropic::AnthropicRouter::new(&ctx)
                .map(|router| Arc::new(router) as Arc<dyn ExternalRouter>)
        })
    }

    #[cfg(not(feature = "anthropic"))]
    fn anthropic_build(_ctx: ExternalContext) -> BuildFuture {
        Box::pin(async { Err(ANTHROPIC.not_compiled()) })
    }

    #[cfg(feature = "gemini")]
    fn gemini_build(ctx: ExternalContext) -> BuildFuture {
        use std::sync::Arc;

        use super::ExternalRouter;
        Box::pin(async move {
            crate::gemini::GeminiRouter::new(&ctx)
                .map(|router| Arc::new(router) as Arc<dyn ExternalRouter>)
        })
    }

    #[cfg(not(feature = "gemini"))]
    fn gemini_build(_ctx: ExternalContext) -> BuildFuture {
        Box::pin(async { Err(GEMINI.not_compiled()) })
    }
}

/// Every external router this build carries.
pub fn builtin_routers() -> Vec<ExternalRouterSpec> {
    known::all().into_iter().filter(|s| s.compiled).collect()
}

/// The router that `--backend <name>` selects alone, compiled in or not.
pub fn spec_for_backend(backend: &str) -> Option<ExternalRouterSpec> {
    known::all().into_iter().find(|s| s.backend == backend)
}

/// The router among `specs` that takes workers of `provider`: a router that
/// names the provider beats the fallback, so a provider-specific router can
/// coexist with the OpenAI-compatible one taking every custom provider.
pub fn home_among(
    specs: &[ExternalRouterSpec],
    provider: Option<&ProviderType>,
) -> Option<ExternalRouterSpec> {
    match provider {
        Some(provider) => specs
            .iter()
            .find(|s| !s.fallback && (s.serves)(provider))
            .or_else(|| specs.iter().find(|s| s.fallback && (s.serves)(provider)))
            .copied(),
        None => specs.iter().find(|s| s.fallback).copied(),
    }
}

/// The built-in router that takes workers of `provider`, compiled in or not.
pub fn spec_for_provider(provider: Option<&ProviderType>) -> Option<ExternalRouterSpec> {
    home_among(&known::all(), provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn providers() -> Vec<ProviderType> {
        vec![
            ProviderType::OpenAI,
            ProviderType::XAI,
            ProviderType::Anthropic,
            ProviderType::Gemini,
            ProviderType::Custom("oci".to_string()),
        ]
    }

    #[test]
    fn every_provider_has_exactly_one_specific_home_or_the_fallback() {
        let specs = known::all();
        assert_eq!(specs.iter().filter(|s| s.fallback).count(), 1);
        for provider in providers() {
            let specific = specs
                .iter()
                .filter(|s| !s.fallback && (s.serves)(&provider))
                .count();
            assert!(
                specific <= 1,
                "{provider:?} is claimed by {specific} routers"
            );
            assert!(
                spec_for_provider(Some(&provider)).is_some(),
                "{provider:?} has no home"
            );
        }
        assert_eq!(spec_for_provider(None).map(|s| s.backend), Some("openai"));
    }

    #[test]
    fn a_specific_router_beats_the_fallback() {
        assert_eq!(
            spec_for_provider(Some(&ProviderType::Anthropic)).map(|s| s.backend),
            Some("anthropic")
        );
        assert_eq!(
            spec_for_provider(Some(&ProviderType::XAI)).map(|s| s.backend),
            Some("openai")
        );
        // A future provider-specific router wins over the fallback even
        // though the fallback also takes custom providers.
        let mut specs = known::all().to_vec();
        let oci = ExternalRouterSpec {
            router_id: "http-oci",
            backend: "oci",
            label: "OCI",
            feature: "provider-oci",
            serves: |p| matches!(p, ProviderType::Custom(name) if name == "oci"),
            fallback: false,
            compiled: true,
            build: |_ctx| Box::pin(async { Err("not built in tests".to_string()) }),
        };
        specs.push(oci);
        let custom = ProviderType::Custom("oci".to_string());
        assert_eq!(
            home_among(&specs, Some(&custom)).map(|s| s.backend),
            Some("oci")
        );
        let other = ProviderType::Custom("together".to_string());
        assert_eq!(
            home_among(&specs, Some(&other)).map(|s| s.backend),
            Some("openai")
        );
    }

    #[test]
    fn a_missing_router_names_its_feature() {
        assert!(known::ANTHROPIC
            .not_compiled()
            .contains("`provider-anthropic`"));
        assert_eq!(spec_for_backend("gemini").map(|s| s.label), Some("Gemini"));
        assert!(spec_for_backend("oci").is_none());
    }
}
