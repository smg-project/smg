//! Harmony Response Processing Stage: Parse Harmony channels to ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::super::{HarmonyResponseProcessor, HarmonyStreamingProcessor};
use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, ProcessStage, RateLimitCell},
        context::{DispatchContext, FinalResponse},
        spec::{HarmonyResponseSpec, ResponseSpec},
    },
};

/// Harmony Response Processing stage: Parse and format Harmony responses
///
/// Takes output tokens from execution and parses them using HarmonyParserAdapter
/// to extract analysis, tool calls, and final response text from Harmony channels.
/// The Harmony spec owns its request handle — the one deliberate post-build reader.
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
impl ProcessStage for HarmonyResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Harmony(harmony_spec) = spec else {
            error!(
                function = "HarmonyResponseProcessingStage::process",
                "Wrong response spec"
            );
            return Err(error::internal_error(
                "wrong_response_spec",
                "Wrong response spec",
            ));
        };

        let is_streaming = ctx.streaming;

        // String `stop` sequences the ROUTER must enforce, as reported by the
        // backend client during request building (its residual obligation:
        // strings the engine will never match). Empty for engines that match
        // server-side.
        let router_stops = ctx.response.router_stop_obligations.clone();

        match harmony_spec {
            HarmonyResponseSpec::Chat(chat_request) => {
                // Get execution result (output tokens from model)
                let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::process",
                        request_type = "Chat",
                        "No execution result available"
                    );
                    error::internal_error("no_execution_result", "No execution result")
                })?;

                let dispatch = ctx.dispatch.as_ref().cloned().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::process",
                        request_type = "Chat",
                        "Dispatch metadata not set"
                    );
                    error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
                })?;

                // For streaming, delegate to streaming processor and return SSE response
                if is_streaming {
                    // Reserved (if tenant rate limiting is enabled): settled
                    // with real usage inside the streaming processor on
                    // success, or abandoned via the attached
                    // ReservationAttachment's Drop below on early
                    // disconnect/error.
                    let reservation = ctx
                        .rate_limit_cell
                        .as_deref()
                        .and_then(RateLimitCell::take_for_streaming_handoff);

                    let response = self
                        .streaming_processor
                        .clone()
                        .process_streaming_chat_response(
                            execution_result,
                            chat_request,
                            dispatch,
                            router_stops,
                            reservation.clone(),
                        )
                        .await;

                    // Attach load guards (and the reservation's
                    // disconnect/error safety net) to the response body for
                    // proper RAII lifecycle.
                    let response = helpers::attach_response_guards(
                        response,
                        ctx.load_guards.take(),
                        reservation,
                    );

                    return Ok(Some(response));
                }

                // For non-streaming, delegate to Harmony response processor to build ChatCompletionResponse
                let response = self
                    .processor
                    .process_non_streaming_chat_response(
                        execution_result,
                        chat_request,
                        dispatch,
                        &router_stops,
                    )
                    .await?;

                ctx.response.final_response = Some(FinalResponse::Chat(response));
                Ok(None)
            }
            HarmonyResponseSpec::Responses(responses_request) => {
                // For streaming Responses API, leave execution_result in context
                // for external streaming processor (serve_harmony_responses_stream)
                if is_streaming {
                    // Don't take execution_result - let the caller handle it
                    return Ok(None);
                }

                // For non-streaming, process normally
                let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::process",
                        request_type = "Responses",
                        "No execution result available"
                    );
                    error::internal_error("no_execution_result", "No execution result")
                })?;

                let dispatch = ctx.dispatch.as_ref().cloned().ok_or_else(|| {
                    error!(
                        function = "HarmonyResponseProcessingStage::process",
                        request_type = "Responses",
                        "Dispatch metadata not set"
                    );
                    error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
                })?;

                let iteration_result = self
                    .processor
                    .process_responses_iteration(execution_result, responses_request, dispatch)
                    .await?;

                ctx.response.responses_iteration_result = Some(iteration_result);
                Ok(None)
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
