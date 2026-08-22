//! Completion response processing stage
//!
//! Stage 7 for the `/v1/completions` pipeline
//!
//! - For streaming: spawns background task and returns SSE response (early exit)
//! - For non-streaming: collects the backend response, converts it to
//!   `CompletionResponse`, and stores it as `FinalResponse::Completion`.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, ProcessStage, RateLimitCell},
        context::{DispatchContext, FinalResponse},
        regular::{processor, streaming},
        spec::ResponseSpec,
    },
};

/// Completion response processing stage
pub(crate) struct CompletionResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl CompletionResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
        }
    }
}

#[async_trait]
impl ProcessStage for CompletionResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Completion(completion_request) = spec else {
            error!(
                function = "CompletionResponseProcessingStage::process",
                "Wrong response spec"
            );
            return Err(error::internal_error(
                "wrong_response_spec",
                "Wrong response spec",
            ));
        };

        let is_streaming = ctx.streaming;

        let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::process",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        let dispatch = ctx
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "CompletionResponseProcessingStage::process",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::process",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            // Reserved (if tenant rate limiting is enabled): settled with real
            // usage inside the streaming processor on success, or abandoned
            // via the attached ReservationAttachment's Drop below on early
            // disconnect/error.
            let reservation = ctx
                .rate_limit_cell
                .as_deref()
                .and_then(RateLimitCell::take_for_streaming_handoff);

            let response = self
                .streaming_processor
                .clone()
                .process_completion_streaming_response(
                    execution_result,
                    completion_request,
                    dispatch,
                    tokenizer,
                    reservation.clone(),
                )
                .await;

            // Attach load guards (and the reservation's disconnect/error
            // safety net) to the response body for proper RAII lifecycle.
            let response =
                helpers::attach_response_guards(response, ctx.load_guards.take(), reservation);

            return Ok(Some(response));
        }

        // Non-streaming path
        let stop_decoder = ctx.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "CompletionResponseProcessingStage::process",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        let response = self
            .processor
            .process_non_streaming_completion_response(
                execution_result,
                completion_request,
                dispatch,
                tokenizer,
                stop_decoder,
            )
            .await?;

        ctx.response.final_response = Some(FinalResponse::Completion(response));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "CompletionResponseProcessing"
    }
}
