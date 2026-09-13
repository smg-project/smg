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
        common::stages::{helpers, ProcessStage, RateLimitCell},
        context::{DispatchContext, FinalResponse},
        regular::{processor, streaming},
        spec::ResponseSpec,
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
impl ProcessStage for MessageResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Messages(messages_request) = spec else {
            error!(
                function = "MessageResponseProcessingStage::process",
                "Wrong response spec"
            );
            return Err(error::internal_error(
                "wrong_response_spec",
                "Wrong response spec",
            ));
        };

        let is_streaming = ctx.streaming;

        // Extract execution result
        let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::process",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata
        let dispatch = ctx
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "MessageResponseProcessingStage::process",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Get cached tokenizer
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::process",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            // Read derived skip_special_tokens (set in preparation, survives request_building .take())
            let skip_special_tokens = ctx.response.skip_special_tokens.unwrap_or(true);

            // Reserved (if tenant rate limiting is enabled): settled with real
            // usage inside the streaming processor on success, or abandoned
            // via the attached ReservationAttachment's Drop below on early
            // disconnect/error.
            let reservation = ctx
                .rate_limit_cell
                .as_deref()
                .and_then(RateLimitCell::take_for_streaming_handoff);

            // Streaming: use StreamingProcessor and return SSE response
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
            let response =
                helpers::attach_response_guards(response, ctx.load_guards.take(), reservation);

            return Ok(Some(response));
        }

        // Non-streaming: delegate to ResponseProcessor
        let stop_decoder = ctx.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "MessageResponseProcessingStage::process",
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
        ctx.response.final_response = Some(FinalResponse::Messages(response));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "MessageResponseProcessing"
    }
}
