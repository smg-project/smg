//! Chat response processing stage: Handles both streaming and non-streaming responses
//!
//! - For streaming: Spawns background task and returns SSE response (early exit)
//! - For non-streaming: Collects all responses and builds final ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage, RateLimitCell},
        context::{FinalResponse, RequestContext},
        regular::{processor, streaming, views},
    },
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

        // Response-phase view set by request building; the payload itself may
        // already be released.
        let Some(views::RequestView::Chat(chat_request)) = ctx.state.response.request_view.take()
        else {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "Request view not set"
            );
            return Err(error::internal_error(
                "request_view_not_set",
                "Request view not set",
            ));
        };

        if is_streaming {
            // Read derived skip_special_tokens (set in preparation, survives request_building .take())
            let skip_special_tokens = ctx
                .state
                .response
                .skip_special_tokens
                .unwrap_or(chat_request.skip_special_tokens);

            // Reserved (if tenant rate limiting is enabled): settled with real
            // usage inside the streaming processor on success, or abandoned
            // via the attached ReservationAttachment's Drop below on early
            // disconnect/error.
            let reservation = ctx
                .input
                .rate_limit_cell
                .as_deref()
                .and_then(RateLimitCell::take_for_streaming_handoff);

            // Streaming: Use StreamingProcessor and return SSE response. The
            // stream task consumes the view, never the parsed request.
            let response = self
                .streaming_processor
                .clone()
                .process_streaming_response(
                    execution_result,
                    chat_request,
                    dispatch,
                    tokenizer,
                    skip_special_tokens,
                    reservation.clone(),
                )
                .await;

            // Attach load guards (and the reservation's disconnect/error
            // safety net) to the response body for proper RAII lifecycle.
            let response = helpers::attach_response_guards(
                response,
                ctx.state.load_guards.take(),
                reservation,
            );

            return Ok(Some(response));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = chat_request.logprobs;

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
