//! Chat response processing stage: Handles both streaming and non-streaming responses
//!
//! - For streaming: Spawns background task and returns SSE response (early exit)
//! - For non-streaming: Collects all responses and builds final ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    rate_limit::ReservationAttachment,
    routers::{
        error,
        grpc::{
            common::stages::{PipelineStage, RateLimitCell, RateLimitOutcome},
            context::{FinalResponse, RequestContext},
            regular::{processor, streaming},
        },
    },
    worker::AttachedBody,
};

/// Chat response processing stage
pub(crate) struct ChatResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl ChatResponseProcessingStage {
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
impl PipelineStage for ChatResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        self.process_chat_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "ChatResponseProcessing"
    }
}

impl ChatResponseProcessingStage {
    async fn process_chat_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "ChatResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Get cached tokenizer (resolved once in preparation stage)
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::process_chat_response",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            // Read derived skip_special_tokens (set in preparation, survives request_building .take())
            let skip_special_tokens = ctx
                .state
                .response
                .skip_special_tokens
                .unwrap_or_else(|| ctx.chat_request().skip_special_tokens);

            // Reserved (if tenant rate limiting is enabled): settled with real
            // usage inside the streaming processor on success, or abandoned
            // via ReservationAttachment's Drop below on early disconnect/error.
            let reservation = ctx
                .input
                .rate_limit_cell
                .as_deref()
                .and_then(RateLimitCell::peek)
                .and_then(|outcome| match outcome {
                    RateLimitOutcome::Admitted(handle) => Some(handle),
                    RateLimitOutcome::Denied => None,
                });

            // Streaming: Use StreamingProcessor and return SSE response
            let response = self.streaming_processor.clone().process_streaming_response(
                execution_result,
                ctx.chat_request_arc(), // Cheap Arc clone (8 bytes)
                dispatch,
                tokenizer,
                skip_special_tokens,
                reservation.clone(),
            );

            // Attach load guards (and the reservation's disconnect/error
            // safety net) to the response body for proper RAII lifecycle.
            let response = match (ctx.state.load_guards.take(), reservation) {
                (Some(guards), Some(handle)) => AttachedBody::wrap_response(
                    response,
                    (guards, ReservationAttachment::new(handle)),
                ),
                (Some(guards), None) => AttachedBody::wrap_response(response, guards),
                (None, Some(handle)) => {
                    AttachedBody::wrap_response(response, ReservationAttachment::new(handle))
                }
                (None, None) => response,
            };

            return Ok(Some(response));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = ctx.chat_request().logprobs;

        let chat_request = ctx.chat_request_arc();

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        let response = self
            .processor
            .process_non_streaming_chat_response(
                execution_result,
                chat_request,
                dispatch,
                tokenizer,
                stop_decoder,
                request_logprobs,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Chat(response));

        Ok(None)
    }
}
