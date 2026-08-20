//! Generate response processing stage: Handles both streaming and non-streaming responses

use std::{sync::Arc, time::Instant};

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

/// Generate response processing stage
///
/// Extracts generate-specific response processing logic from the old unified ResponseProcessingStage.
pub(crate) struct GenerateResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl GenerateResponseProcessingStage {
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
impl ProcessStage for GenerateResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Generate(generate_request) = spec else {
            error!(
                function = "GenerateResponseProcessingStage::process",
                "Wrong response spec"
            );
            return Err(error::internal_error(
                "wrong_response_spec",
                "Wrong response spec",
            ));
        };

        let start_time = Instant::now();
        let is_streaming = ctx.streaming;

        // Extract execution result
        let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "GenerateResponseProcessingStage::process",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "GenerateResponseProcessingStage::process",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Get cached tokenizer (resolved once in preparation stage)
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "GenerateResponseProcessingStage::process",
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

            // Streaming: Use StreamingProcessor and return SSE response
            let response = self
                .streaming_processor
                .clone()
                .process_streaming_generate(
                    execution_result,
                    generate_request,
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

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = generate_request.return_logprob;

        let stop_decoder = ctx.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "GenerateResponseProcessingStage::process",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        let result_array = self
            .processor
            .process_non_streaming_generate_response(
                execution_result,
                dispatch,
                stop_decoder,
                request_logprobs,
                start_time,
            )
            .await?;

        // Store the final response
        ctx.response.final_response = Some(FinalResponse::Generate(result_array));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "GenerateResponseProcessing"
    }
}
