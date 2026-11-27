//! Storage request-context middleware.
//!
//! Maps configured request headers onto a [`StorageRequestContext`] so the
//! storage layer can read tenant/user/etc. fields without re-parsing headers.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use smg_data_connector::{with_request_context, RequestContext as StorageRequestContext};

use crate::{config::RouterConfig, server::AppState};

fn extract_header_str(headers: &http::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn build_storage_request_context(
    config: &RouterConfig,
    headers: &http::HeaderMap,
) -> Option<StorageRequestContext> {
    let mut ctx = StorageRequestContext::new();

    for (header_name, context_key) in &config.storage_context_headers {
        let header_name = header_name.trim();
        let context_key = context_key.trim();

        if header_name.is_empty() || context_key.is_empty() {
            continue;
        }

        if let Some(value) = extract_header_str(headers, header_name) {
            ctx.set(context_key, value);
        }
    }

    (!ctx.data().is_empty()).then_some(ctx)
}

pub async fn storage_context_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if state
        .context
        .router_config
        .storage_context_headers
        .is_empty()
    {
        return next.run(request).await;
    }

    match build_storage_request_context(&state.context.router_config, request.headers()) {
        Some(ctx) => with_request_context(ctx, next.run(request)).await,
        None => next.run(request).await,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/storage_context/tests.rs"]
mod tests;
