//! The chat route's body: the shared extractor's validation plus the one rule
//! the request's own fields cannot settle.
//!
//! `store: true` and `web_search_options` run on a hosted provider. A request
//! an external provider's router takes is forwarded with both fields; a
//! request an engine's router takes runs neither and is refused by name.
//! Neither the model name nor the fleet tells the two apart: `sonar-pro` and
//! `prod-chat` read alike, a keyless passthrough gateway forwards every model
//! through a wildcard worker, and a single-router gateway sends every request
//! to its default router whatever an external worker lists. The router
//! selection does. So this extractor asks the state's router where the
//! request goes and hands [`ProviderProfile::validate_hosted_features`] the
//! answer, only when one of the two fields is present: the ordinary request
//! pays nothing here.

use std::sync::Arc;

use axum::{
    extract::{FromRequest, Request},
    response::Response,
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    profile::ProviderProfile,
    validated::{rejection_of, ValidatedJson},
};

use super::AppState;

/// What the chat extractor asks of its state: whether a request for
/// `model_id` goes to an external provider's router, the router selection's
/// own answer ([`crate::routers::RouterTrait::forwards_to_provider`]).
pub(crate) trait RoutesToProvider {
    fn routes_to_provider(&self, model_id: &str) -> bool;
}

impl RoutesToProvider for Arc<AppState> {
    fn routes_to_provider(&self, model_id: &str) -> bool {
        self.router.forwards_to_provider(model_id)
    }
}

/// A validated chat request whose hosted features were judged against the
/// router its request goes to. Everything else is [`ValidatedJson`]'s: the
/// body is parsed, normalized and validated there, with the same rejection
/// envelope for every refusal.
pub(crate) struct ValidatedChat(pub(crate) ChatCompletionRequest);

impl<S> FromRequest<S> for ValidatedChat
where
    S: RoutesToProvider + Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let ValidatedJson(body) =
            ValidatedJson::<ChatCompletionRequest>::from_request(req, state).await?;
        ProviderProfile::validate_hosted_features(&body, || state.routes_to_provider(&body.model))
            .map_err(rejection_of)?;
        Ok(Self(body))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{Arc, OnceLock},
    };

    use axum::{
        body::{to_bytes, Body},
        http::{header::CONTENT_TYPE, Request, StatusCode},
        routing::post,
        Router,
    };
    use llm_tokenizer::registry::TokenizerRegistry;
    use serde_json::Value;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use tower::ServiceExt;

    use super::*;
    use crate::{
        app_context::AppContext, config::RouterConfig, health::ProbeState,
        policies::PolicyRegistry, routers::RouterTrait, worker::WorkerRegistry,
    };

    /// The app state answers with its router's answer: the wiring between
    /// the extractor and the gateway's selection, as production reads it.
    #[tokio::test]
    async fn the_app_state_asks_its_router() {
        #[derive(Debug)]
        struct ProviderFor(&'static str);

        #[async_trait::async_trait]
        impl RouterTrait for ProviderFor {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn router_type(&self) -> &'static str {
                "stub"
            }

            fn forwards_to_provider(&self, model_id: &str) -> bool {
                model_id == self.0
            }
        }

        let router_config = RouterConfig::default();
        let context = Arc::new(
            AppContext::builder()
                .client(crate::server::http_client())
                .rate_limiter(None)
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(None)
                .tool_parser_factory(None)
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(router_config.policy.clone())))
                .router_config(router_config)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_monitor(None)
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .unwrap(),
        );
        let state = Arc::new(AppState {
            secret_sources: Default::default(),
            secrets: None,
            router: Arc::new(ProviderFor("sonar-pro")),
            probe_state: ProbeState::new(context.inflight_tracker.clone()),
            context,
            admission_queue: None,
            gateway: None,
            mesh_handler: None,
            mesh_adapters: None,
        });
        assert!(state.routes_to_provider("sonar-pro"));
        assert!(!state.routes_to_provider("prod-chat"));
    }

    /// The routes a test knows: the models whose requests go to a provider's
    /// router.
    #[derive(Clone)]
    struct Fleet(Arc<HashSet<&'static str>>);

    impl RoutesToProvider for Fleet {
        fn routes_to_provider(&self, model_id: &str) -> bool {
            self.0.contains(model_id)
        }
    }

    async fn answer(fleet: &Fleet, model: &str, extra: &str) -> (StatusCode, Value) {
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|ValidatedChat(_): ValidatedChat| async { StatusCode::OK }),
            )
            .with_state(fleet.clone());
        let body = format!(
            r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#
        );
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .expect("a request"),
            )
            .await
            .expect("an answer");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    const SEARCH: &str = r#","web_search_options":{"search_context_size":"low"}"#;
    const STORE: &str = r#","store":true"#;

    /// A request bound for a provider's router keeps `web_search_options` and
    /// `store: true` for the provider; the same request bound for an engine is
    /// refused by name, in the extractor's own envelope; a request without the
    /// two fields passes for either.
    #[tokio::test]
    async fn a_request_bound_for_a_provider_keeps_its_hosted_features() {
        let fleet = Fleet(Arc::new(HashSet::from(["sonar-pro"])));
        for extra in [SEARCH, STORE] {
            let (status, json) = answer(&fleet, "sonar-pro", extra).await;
            assert_eq!(status, StatusCode::OK, "{extra}: {json}");
        }
        for (extra, param) in [(SEARCH, "web_search_options"), (STORE, "store")] {
            let (status, json) = answer(&fleet, "prod-chat", extra).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {json}");
            assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
            assert_eq!(json["error"]["code"], "unsupported_parameter", "{json}");
            assert_eq!(json["error"]["param"], param, "{json}");
        }
        for model in ["sonar-pro", "prod-chat"] {
            let (status, json) = answer(&fleet, model, "").await;
            assert_eq!(status, StatusCode::OK, "{model}: {json}");
        }
    }

    /// The shared validation still runs first and answers in the same
    /// envelope: a malformed request never reaches the route read.
    #[tokio::test]
    async fn the_shared_validation_runs_first() {
        let fleet = Fleet(Arc::new(HashSet::new()));
        let (status, json) = answer(&fleet, "prod-chat", r#","temperature":"hot""#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
        assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
    }
}
