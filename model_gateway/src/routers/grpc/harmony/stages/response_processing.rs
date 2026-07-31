//! Harmony Response Processing Stage: Parse Harmony channels to ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::super::{HarmonyResponseProcessor, HarmonyStreamingProcessor};
use crate::{
    rate_limit::ReservationAttachment,
    routers::{
        error,
        grpc::{
            common::stages::{PipelineStage, RateLimitCell, RateLimitOutcome},
            context::{FinalResponse, RequestContext, RequestType},
        },
    },
    worker::AttachedBody,
};

/// String `stop` sequences the ROUTER must enforce, as reported by the
/// backend client during request building (its residual obligation: strings
/// the engine will never match). Empty for engines that match server-side —
/// no transport inspection here.
fn router_stop_strings(ctx: &RequestContext) -> Vec<String> {
    ctx.state.response.router_stop_obligations.clone()
}

/// Harmony Response Processing stage: Parse and format Harmony responses
///
/// Takes output tokens from execution and parses them using HarmonyParserAdapter
/// to extract analysis, tool calls, and final response text from Harmony channels.
pub(crate) struct HarmonyResponseProcessingStage {
    processor: HarmonyResponseProcessor,
    streaming_processor: Arc<HarmonyStreamingProcessor>,
}

impl HarmonyResponseProcessingStage {
    /// Create a new Harmony response processing stage
    pub fn new() -> Self {
        Self {
            processor: HarmonyResponseProcessor::new(),
            streaming_processor: Arc::new(HarmonyStreamingProcessor::new()),
        }
    }
}

impl Default for HarmonyResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for HarmonyResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Check request type to determine which processor method to call
        match &ctx.input.request_type {
            RequestType::Chat(_) => {
                // Get execution result (output tokens from model)
                let execution_result =
                    ctx.state.response.execution_result.take().ok_or_else(|| {
                        error!(
                            function = "HarmonyResponseProcessingStage::execute",
                            request_type = "Chat",
                            "No execution result available"
                        );
                        error::internal_error("no_execution_result", "No execution result")
                    })?;

                let dispatch = ctx.state.dispatch.as_ref().cloned().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::execute",
                        request_type = "Chat",
                        "Dispatch metadata not set"
                    );
                    error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
                })?;

                // For streaming, delegate to streaming processor and return SSE response
                if is_streaming {
                    // Reserved (if tenant rate limiting is enabled): settled
                    // with real usage inside the streaming processor on
                    // success, or abandoned via ReservationAttachment's Drop
                    // below on early disconnect/error.
                    let reservation = ctx
                        .input
                        .rate_limit_cell
                        .as_deref()
                        .and_then(RateLimitCell::peek)
                        .and_then(|outcome| match outcome {
                            RateLimitOutcome::Admitted(handle) => Some(handle),
                            RateLimitOutcome::Denied => None,
                        });

                    let response = self
                        .streaming_processor
                        .clone()
                        .process_streaming_chat_response(
                            execution_result,
                            ctx.chat_request_arc(),
                            dispatch,
                            router_stop_strings(ctx),
                            reservation.clone(),
                        );

                    // Attach load guards (and the reservation's
                    // disconnect/error safety net) to the response body for
                    // proper RAII lifecycle.
                    let response = match (ctx.state.load_guards.take(), reservation) {
                        (Some(guards), Some(handle)) => AttachedBody::wrap_response(
                            response,
                            (guards, ReservationAttachment::new(handle)),
                        ),
                        (Some(guards), None) => AttachedBody::wrap_response(response, guards),
                        (None, Some(handle)) => AttachedBody::wrap_response(
                            response,
                            ReservationAttachment::new(handle),
                        ),
                        (None, None) => response,
                    };

                    return Ok(Some(response));
                }

                // For non-streaming, delegate to Harmony response processor to build ChatCompletionResponse
                let chat_request = ctx.chat_request_arc();
                let stops = router_stop_strings(ctx);
                let response = self
                    .processor
                    .process_non_streaming_chat_response(
                        execution_result,
                        chat_request,
                        dispatch,
                        &stops,
                    )
                    .await?;

                ctx.state.response.final_response = Some(FinalResponse::Chat(response));
                Ok(None)
            }
            RequestType::Responses(_) => {
                // For streaming Responses API, leave execution_result in context
                // for external streaming processor (serve_harmony_responses_stream)
                if is_streaming {
                    // Don't take execution_result - let the caller handle it
                    return Ok(None);
                }

                // For non-streaming, process normally
                let execution_result =
                    ctx.state.response.execution_result.take().ok_or_else(|| {
                        error!(
                            function = "HarmonyResponseProcessingStage::execute",
                            request_type = "Responses",
                            "No execution result available"
                        );
                        error::internal_error("no_execution_result", "No execution result")
                    })?;

                let dispatch = ctx.state.dispatch.as_ref().cloned().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::execute",
                        request_type = "Responses",
                        "Dispatch metadata not set"
                    );
                    error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
                })?;

                let responses_request = ctx.responses_request_arc();
                let iteration_result = self
                    .processor
                    .process_responses_iteration(execution_result, responses_request, dispatch)
                    .await?;

                ctx.state.response.responses_iteration_result = Some(iteration_result);
                Ok(None)
            }
            request_type @ (RequestType::Generate(_)
            | RequestType::Completion(_)
            | RequestType::Embedding(_)
            | RequestType::Classify(_)
            | RequestType::Messages(_)) => {
                error!(
                    function = "HarmonyResponseProcessingStage::execute",
                    request_type = %request_type,
                    "{request_type} request type not supported in Harmony pipeline"
                );
                Err(error::internal_error(
                    "not_supported_in_harmony",
                    format!("{request_type} requests not supported in Harmony pipeline"),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "HarmonyResponseProcessing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response_processing_stage_creation() {
        let stage = HarmonyResponseProcessingStage::new();
        assert_eq!(stage.name(), "HarmonyResponseProcessing");
    }
}
