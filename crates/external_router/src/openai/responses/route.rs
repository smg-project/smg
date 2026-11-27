//! Responses API routing orchestration.
//!
//! Mirrors the delegation pattern in `chat.rs`: the `ExternalRouter` method in
//! `router.rs` packs borrowed references into [`ResponsesRouterContext`] and
//! delegates to [`route_responses`].

use std::{sync::Arc, time::Instant};

use axum::{http::HeaderMap, response::Response};
use openai_protocol::{
    common::to_value_exact,
    model_type::Endpoint,
    responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest},
};

use super::{
    super::{
        context::{
            ComponentRefs, PayloadState, RequestContext, ResponsesComponents,
            ResponsesPayloadState, WorkerSelection,
        },
        provider::ProviderRegistry,
        router::resolve_provider,
    },
    handle_non_streaming_response, handle_streaming_response,
};
use crate::{
    error, known,
    metrics::{bool_to_static_str, error_reason_for_refused_selection, metrics_labels, Metrics},
    tenant::{storage_tenant, TenantRequestMeta},
    worker::{SelectWorkerRequest, WorkerSource},
};

/// Shared context passed to responses routing functions.
pub(in crate::openai) struct ResponsesRouterContext<'a> {
    pub workers: &'a Arc<dyn WorkerSource>,
    pub provider_registry: &'a ProviderRegistry,
    pub responses_components: &'a Arc<ResponsesComponents>,
}

/// Route a responses API request to the appropriate upstream worker.
pub(in crate::openai) async fn route_responses(
    deps: &ResponsesRouterContext<'_>,
    headers: Option<&HeaderMap>,
    tenant_meta: &TenantRequestMeta,
    body: &ResponsesRequest,
    model_id: &str,
) -> Response {
    let start = Instant::now();
    let model = model_id;
    let streaming = body.stream.unwrap_or(false);

    Metrics::record_router_request(
        metrics_labels::ROUTER_OPENAI,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        metrics_labels::ENDPOINT_RESPONSES,
        bool_to_static_str(streaming),
    );

    let worker = match deps
        .workers
        .select(&SelectWorkerRequest {
            model_id: model,
            headers,
            router: Some(known::OPENAI),
            ..Default::default()
        })
        .await
    {
        Ok(w) => w,
        Err(response) => {
            super::record_error(
                model,
                metrics_labels::ERROR_NO_WORKERS,
                error_reason_for_refused_selection(response.status()),
            );
            return response;
        }
    };

    // Validate mutual exclusivity of conversation and previous_response_id
    // Treat empty strings as unset to match other metadata paths. The
    // `conversation` field is a `Option<ConversationRef>` union (bare string
    // or `{ id }` object); `ConversationRef::is_empty` covers both shapes.
    let conversation = body.conversation.as_ref().filter(|c| !c.is_empty());
    let has_previous_response = body
        .previous_response_id
        .as_ref()
        .is_some_and(|s| !s.is_empty());
    if conversation.is_some() && has_previous_response {
        super::record_error(
            model,
            metrics_labels::ERROR_VALIDATION,
            metrics_labels::REASON_INVALID_REQUEST,
        );
        return error::bad_request(
            "invalid_request",
            "Cannot specify both 'conversation' and 'previous_response_id'".to_string(),
        );
    }

    let mut request_body = body.clone();
    request_body.model = model_id.to_string();
    request_body.conversation = None;

    // The stored state as the caller's tenant may see it.
    let tenant = storage_tenant(tenant_meta);
    let loaded_history = match super::history::load_input_history(
        deps.responses_components,
        conversation.map(|c| c.as_id()),
        &mut request_body,
        model,
        tenant.as_deref(),
    )
    .await
    {
        Ok(id) => id,
        Err(response) => return response,
    };

    request_body.store = Some(false);
    if let ResponseInput::Items(ref mut items) = request_body.input {
        items.retain(|item| !matches!(item, ResponseInputOutputItem::Reasoning { .. }));
    }

    let mut payload = match to_value_exact(&request_body) {
        Ok(v) => v,
        Err(e) => {
            super::record_error(
                model,
                metrics_labels::ERROR_VALIDATION,
                metrics_labels::REASON_INVALID_REQUEST,
            );
            return error::bad_request(
                "invalid_request",
                format!("Failed to serialize request: {e}"),
            );
        }
    };

    let provider = resolve_provider(deps.provider_registry, worker.as_ref(), model);
    if let Err(e) = provider.transform_request(&mut payload, Endpoint::Responses) {
        super::record_error(
            model,
            metrics_labels::ERROR_VALIDATION,
            metrics_labels::REASON_INVALID_REQUEST,
        );
        return error::bad_request("invalid_request", format!("Provider transform error: {e}"));
    }

    let mut ctx = RequestContext::for_responses(
        Arc::new(body.clone()),
        headers.cloned(),
        Some(model_id.to_string()),
        ComponentRefs::Responses(Arc::clone(deps.responses_components)),
    );
    ctx.storage_request_context = smg_data_connector::current_request_context();
    ctx.tenant_request_meta = Some(tenant_meta.clone());

    ctx.state.worker = Some(WorkerSelection {
        worker: Arc::clone(&worker),
        provider: Arc::clone(&provider),
    });

    ctx.state.payload = Some(PayloadState {
        json: payload,
        url: format!("{}/v1/responses", worker.url()),
    });
    ctx.state.responses_payload = Some(ResponsesPayloadState {
        previous_response_id: loaded_history.previous_response_id,
        existing_mcp_list_tools_labels: loaded_history.existing_mcp_list_tools_labels,
    });

    let response = if ctx.is_streaming() {
        handle_streaming_response(ctx).await
    } else {
        handle_non_streaming_response(ctx).await
    };

    if response.status().is_success() {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_RESPONSES,
            start.elapsed(),
        );
    }

    response
}

#[cfg(test)]
#[path = "../../../tests/unit/openai/responses/route/tests.rs"]
mod tests;
