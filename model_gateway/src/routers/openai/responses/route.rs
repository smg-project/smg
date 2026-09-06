//! Responses API routing orchestration.
//!
//! Mirrors the delegation pattern in `chat.rs`: the `RouterTrait` method in
//! `router.rs` packs borrowed references into [`ResponsesRouterContext`] and
//! delegates to [`route_responses`].

use std::{sync::Arc, time::Instant};

use axum::{http::HeaderMap, response::Response};
use openai_protocol::responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest};
use serde_json::to_value;

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
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::{
        common::worker_selection::{SelectWorkerRequest, WorkerSelector},
        error,
    },
    worker::{Endpoint, ProviderType, WorkerRegistry},
};

/// Shared context passed to responses routing functions.
pub(in crate::routers::openai) struct ResponsesRouterContext<'a> {
    pub worker_registry: &'a WorkerRegistry,
    pub provider_registry: &'a ProviderRegistry,
    pub responses_components: &'a Arc<ResponsesComponents>,
}

/// Route a responses API request to the appropriate upstream worker.
pub(in crate::routers::openai) async fn route_responses(
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

    let worker = match WorkerSelector::new(deps.worker_registry)
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers,
            provider: Some(ProviderType::OpenAI),
            ..Default::default()
        })
        .await
    {
        Ok(w) => w,
        Err(response) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_NO_WORKERS,
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
        Metrics::record_router_error(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_RESPONSES,
            metrics_labels::ERROR_VALIDATION,
        );
        return error::bad_request(
            "invalid_request",
            "Cannot specify both 'conversation' and 'previous_response_id'".to_string(),
        );
    }

    let mut request_body = body.clone();
    request_body.model = model_id.to_string();
    request_body.conversation = None;

    let loaded_history = match super::history::load_input_history(
        deps.responses_components,
        conversation.map(|c| c.as_id()),
        &mut request_body,
        model,
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

    let mut payload = match to_value(&request_body) {
        Ok(v) => v,
        Err(e) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_OPENAI,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_labels::ENDPOINT_RESPONSES,
                metrics_labels::ERROR_VALIDATION,
            );
            return error::bad_request(
                "invalid_request",
                format!("Failed to serialize request: {e}"),
            );
        }
    };

    let provider = resolve_provider(deps.provider_registry, worker.as_ref(), model);
    if let Err(e) = provider.transform_request(&mut payload, Endpoint::Responses) {
        Metrics::record_router_error(
            metrics_labels::ROUTER_OPENAI,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_labels::ENDPOINT_RESPONSES,
            metrics_labels::ERROR_VALIDATION,
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
mod tests {
    //! R1 wire-contract tests: the OpenAI-compat Responses router forwards
    //! the caller's request body to the upstream provider by serialising the
    //! `ResponsesRequest` value (see [`route_responses`] around the
    //! `to_value(&request_body)` site). These tests lock the shape that the
    //! post-P1 content-part variants produce, so any future change to the
    //! serde layer surfaces here before it reaches an upstream.
    use openai_protocol::{
        common::Detail,
        responses::{
            Annotation, FileDetail, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponsesRequest,
        },
    };
    use serde_json::{json, to_value};

    fn build_request_with_mixed_content() -> ResponsesRequest {
        ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_r1".to_string(),
                role: "user".to_string(),
                content: vec![
                    ResponseContentPart::InputText {
                        text: "what is in this image and this file?".to_string(),
                    },
                    ResponseContentPart::InputImage {
                        detail: Some(Detail::Auto),
                        file_id: Some("file-img".to_string()),
                        image_url: Some("https://example.com/dog.jpg".to_string()),
                    },
                    ResponseContentPart::InputFile {
                        detail: Some(FileDetail::High),
                        file_data: Some("JVBERi0xLjQK".to_string()),
                        file_id: Some("file-pdf".to_string()),
                        file_url: Some("https://example.com/report.pdf".to_string()),
                        filename: Some("report.pdf".to_string()),
                    },
                    ResponseContentPart::Refusal {
                        refusal: "I cannot process that request.".to_string(),
                    },
                ],
                status: Some("completed".to_string()),
                phase: None,
            }]),
            ..Default::default()
        }
    }

    /// Exercises the exact `to_value(&request_body)` step `route_responses`
    /// uses to build the upstream payload — see `route.rs` handler body.
    fn serialize_like_router(req: &ResponsesRequest) -> serde_json::Value {
        to_value(req).expect("router serializes ResponsesRequest without error")
    }

    #[test]
    fn router_serialization_preserves_input_image_fields() {
        let req = build_request_with_mixed_content();
        let payload = serialize_like_router(&req);
        let content = &payload["input"][0]["content"];

        assert_eq!(content[1]["type"], json!("input_image"));
        assert_eq!(content[1]["detail"], json!("auto"));
        assert_eq!(content[1]["file_id"], json!("file-img"));
        assert_eq!(
            content[1]["image_url"],
            json!("https://example.com/dog.jpg")
        );
    }

    #[test]
    fn router_serialization_preserves_input_file_fields() {
        let req = build_request_with_mixed_content();
        let payload = serialize_like_router(&req);
        let content = &payload["input"][0]["content"];

        assert_eq!(content[2]["type"], json!("input_file"));
        assert_eq!(content[2]["detail"], json!("high"));
        assert_eq!(content[2]["file_data"], json!("JVBERi0xLjQK"));
        assert_eq!(content[2]["file_id"], json!("file-pdf"));
        assert_eq!(
            content[2]["file_url"],
            json!("https://example.com/report.pdf")
        );
        assert_eq!(content[2]["filename"], json!("report.pdf"));
    }

    #[test]
    fn router_serialization_preserves_refusal() {
        let req = build_request_with_mixed_content();
        let payload = serialize_like_router(&req);
        let content = &payload["input"][0]["content"];

        assert_eq!(content[3]["type"], json!("refusal"));
        assert_eq!(
            content[3]["refusal"],
            json!("I cannot process that request.")
        );
    }

    #[test]
    fn router_serialization_omits_empty_input_image_fields() {
        // `file_id` / `image_url` / `detail` are all optional; the wire
        // payload must not carry `null`s when the caller leaves them unset
        // (the #[serde(skip_serializing_if = "Option::is_none")] attributes
        // on ResponseContentPart guarantee this).
        let req = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_sparse".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputImage {
                    detail: None,
                    file_id: Some("file-only".to_string()),
                    image_url: None,
                }],
                status: None,
                phase: None,
            }]),
            ..Default::default()
        };
        let payload = serialize_like_router(&req);
        let image = &payload["input"][0]["content"][0];

        assert_eq!(image["type"], json!("input_image"));
        assert_eq!(image["file_id"], json!("file-only"));
        assert!(
            image.get("detail").is_none(),
            "detail should be omitted when None"
        );
        assert!(
            image.get("image_url").is_none(),
            "image_url should be omitted when None"
        );
    }

    #[test]
    fn router_serialization_round_trips_typed_annotations_on_output_text() {
        // Assistant turns replayed from storage carry `OutputText` with typed
        // annotations. R1 must preserve the annotation union end-to-end.
        let req = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_prior".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "Here are three citations.".to_string(),
                    annotations: vec![
                        Annotation::FileCitation {
                            file_id: "file-1".to_string(),
                            filename: "spec.pdf".to_string(),
                            index: 0,
                        },
                        Annotation::UrlCitation {
                            url: "https://example.com".to_string(),
                            title: "Example".to_string(),
                            start_index: 10,
                            end_index: 24,
                        },
                        Annotation::FilePath {
                            file_id: "file-2".to_string(),
                            index: 2,
                        },
                    ],
                    logprobs: None,
                }],
                status: Some("completed".to_string()),
                phase: None,
            }]),
            ..Default::default()
        };

        let payload = serialize_like_router(&req);
        let annotations = &payload["input"][0]["content"][0]["annotations"];
        assert_eq!(annotations[0]["type"], json!("file_citation"));
        assert_eq!(annotations[0]["filename"], json!("spec.pdf"));
        assert_eq!(annotations[1]["type"], json!("url_citation"));
        assert_eq!(annotations[1]["url"], json!("https://example.com"));
        assert_eq!(annotations[1]["start_index"], json!(10));
        assert_eq!(annotations[2]["type"], json!("file_path"));
        assert_eq!(annotations[2]["index"], json!(2));
    }

    #[test]
    fn router_serialization_preserves_empty_annotations_on_output_text() {
        let req = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_prior".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "Prior answer.".to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: Some("completed".to_string()),
                phase: None,
            }]),
            ..Default::default()
        };

        let payload = serialize_like_router(&req);
        let output_text = &payload["input"][0]["content"][0];

        assert_eq!(output_text["type"], json!("output_text"));
        assert_eq!(output_text["annotations"], json!([]));
    }
}
