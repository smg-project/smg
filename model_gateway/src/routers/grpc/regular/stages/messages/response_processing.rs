//! Message response processing stage: streaming and non-streaming response processing
//!
//! - For streaming: Spawns background task and returns SSE response (early exit)
//! - For non-streaming: Collects the backend response, converts it to an Anthropic `Message`,
//!   and stores it as FinalResponse::Messages.

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

/// Message response processing stage
pub(crate) struct MessageResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

impl MessageResponseProcessingStage {
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
impl PipelineStage for MessageResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "MessageResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Get cached tokenizer
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::execute",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        // Response-phase view set by request building; the payload itself may
        // already be released.
        let Some(views::RequestView::Messages(messages_request)) =
            ctx.state.response.request_view.take()
        else {
            error!(
                function = "MessageResponseProcessingStage::execute",
                "Request view not set"
            );
            return Err(error::internal_error(
                "request_view_not_set",
                "Request view not set",
            ));
        };

        if is_streaming {
            // Read derived skip_special_tokens (set in preparation, survives request_building .take())
            let skip_special_tokens = ctx.state.response.skip_special_tokens.unwrap_or(true);

            // Reserved (if tenant rate limiting is enabled): settled with real
            // usage inside the streaming processor on success, or abandoned
            // via the attached ReservationAttachment's Drop below on early
            // disconnect/error.
            let reservation = ctx
                .input
                .rate_limit_cell
                .as_deref()
                .and_then(RateLimitCell::take_for_streaming_handoff);

            // Streaming: use StreamingProcessor and return SSE response. The
            // stream task consumes the view, never the parsed request.
            let response = self
                .streaming_processor
                .clone()
                .process_messages_streaming_response(
                    execution_result,
                    messages_request,
                    dispatch,
                    tokenizer,
                    skip_special_tokens,
                    reservation.clone(),
                )
                .await;

            // Attach load guards (and the reservation's disconnect/error
            // safety net) for RAII lifecycle.
            let response = helpers::attach_response_guards(
                response,
                ctx.state.load_guards.take(),
                reservation,
            );

            return Ok(Some(response));
        }

        // Non-streaming: delegate to ResponseProcessor
        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        let response = self
            .processor
            .process_non_streaming_messages_response(
                execution_result,
                messages_request,
                dispatch,
                tokenizer,
                stop_decoder,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Messages(response));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "MessageResponseProcessing"
    }
}
