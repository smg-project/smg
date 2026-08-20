//! GeminiRouter — entry point for the Gemini Interactions API.

use std::{any::Any, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use openai_protocol::interactions::InteractionsRequest;

use super::{
    context::{RequestContext, SharedComponents},
    driver,
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    routers::{
        common::retry::{is_retryable_response, RetryExecutor},
        RouterTrait,
    },
};

pub struct GeminiRouter {
    shared_components: Arc<SharedComponents>,
    retry_config: RetryConfig,
}

impl std::fmt::Debug for GeminiRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiRouter").finish()
    }
}

impl GeminiRouter {
    /// Create a new `GeminiRouter` from the application context.
    pub fn new(ctx: Arc<AppContext>) -> Result<Self, String> {
        let mcp_orchestrator = ctx
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "Gemini router requires MCP orchestrator".to_string())?
            .clone();

        let request_timeout = Duration::from_secs(ctx.router_config.request_timeout_secs);

        let shared_components = Arc::new(SharedComponents {
            client: ctx.client.clone(),
            worker_registry: ctx.worker_registry.clone(),
            mcp_orchestrator,
            mcp_format_registry: ctx.mcp_format_registry.clone(),
            request_timeout,
        });
        let retry_config = ctx.router_config.effective_retry_config();

        Ok(Self {
            shared_components,
            retry_config,
        })
    }
}

// ============================================================================
// RouterTrait implementation
// ============================================================================

#[async_trait]
impl RouterTrait for GeminiRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn router_type(&self) -> &'static str {
        "gemini"
    }

    async fn route_interactions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: InteractionsRequest,
        model_id: Option<&str>,
    ) -> Response {
        let request = Arc::new(body);
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(String::from);
        let components = self.shared_components.clone();

        // Use per-model retry config if set by a worker, otherwise fall back to router default.
        let per_model_retry_config =
            model_id.and_then(|id| self.shared_components.worker_registry.get_retry_config(id));
        let retry_config = per_model_retry_config
            .as_ref()
            .unwrap_or(&self.retry_config);

        RetryExecutor::execute_response_with_retry(
            retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta.clone();
                async move {
                    let mut ctx =
                        RequestContext::new(request, headers, model_id, tenant_meta, components);
                    driver::execute(&mut ctx).await
                }
            },
            |res, _attempt| is_retryable_response(res),
            |_delay, _attempt| {
                // TODO: record retry metrics when Gemini metrics are added
            },
            || {
                // TODO: record retries-exhausted metric when Gemini metrics are added
            },
        )
        .await
    }
}
