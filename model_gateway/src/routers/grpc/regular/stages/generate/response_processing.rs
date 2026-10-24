//! Generate response processing stage: Handles both streaming and non-streaming responses

use std::{sync::Arc, time::Instant};

use axum::response::Response;
use tracing::error;

use crate::{
    observability::metrics::metrics_labels,
    routers::{
        error,
        grpc::{
            common::stages::RateLimitCell,
            context::{DispatchContext, FinalResponse},
            regular::{processor, streaming},
            spec::ResponseSpec,
            utils::StreamSource,
        },
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

impl GenerateResponseProcessingStage {
    /// The stage's work. The chat+generate dispatcher calls this directly, so
    /// the stage's future lives inline in the dispatcher's: one box per stage
    /// call, not a second one for a boxed trait method.
    pub(crate) async fn process_result(
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
            // on early disconnect/error by the reservation's safety net the
            // processor hands the response body (ReservationAttachment's Drop).
            let reservation = ctx
                .rate_limit_cell
                .as_deref()
                .and_then(RateLimitCell::take_for_streaming_handoff);

            // Streaming: Use StreamingProcessor and return SSE response
            let source = StreamSource::new(
                ctx.workers.as_ref(),
                &dispatch,
                metrics_labels::ENDPOINT_GENERATE,
            );
            let response = self
                .streaming_processor
                .clone()
                .process_streaming_generate(
                    execution_result,
                    generate_request,
                    dispatch,
                    tokenizer,
                    reservation,
                    ctx.load_guards.take(),
                    source,
                )
                .await;

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
                metrics_labels::ENDPOINT_GENERATE,
            )
            .await?;

        // Store the final response
        ctx.response.final_response = Some(FinalResponse::Generate(result_array));

        Ok(None)
    }
}
