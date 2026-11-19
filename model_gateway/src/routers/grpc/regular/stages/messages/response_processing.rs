//! Message response processing stage: streaming and non-streaming response processing
//!
//! - For streaming: Builds the reading of the engine's answer and returns the SSE response
//!   whose body drives it (early exit)
//! - For non-streaming: Collects the backend response, converts it to an Anthropic `Message`,
//!   and stores it as FinalResponse::Messages.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;

use crate::{
    observability::metrics::metrics_labels,
    routers::grpc::{
        common::stages::{ProcessStage, RateLimitCell},
        context::{stage_output, wrong_response_spec, DispatchContext, FinalResponse},
        regular::{processor, streaming},
        spec::ResponseSpec,
        utils::StreamSource,
    },
};

const FUNCTION: &str = "MessageResponseProcessingStage::process";

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
            return Err(wrong_response_spec(FUNCTION));
        };
        let messages_request = *messages_request;

        let is_streaming = ctx.streaming;

        let execution_result = stage_output(ctx.response.execution_result.take(), FUNCTION)?;

        let dispatch = stage_output(ctx.dispatch.take(), FUNCTION)?;

        let tokenizer = stage_output(ctx.tokenizer_arc(), FUNCTION)?;

        if is_streaming {
            // Read derived skip_special_tokens (set in preparation, survives request_building .take())
            let skip_special_tokens = ctx.response.skip_special_tokens.unwrap_or(true);

            let reservation = RateLimitCell::stream_reservation(ctx.rate_limit_cell.as_deref());

            // Streaming: use StreamingProcessor and return SSE response
            let source = StreamSource::new(
                ctx.workers.as_ref(),
                &dispatch,
                metrics_labels::ENDPOINT_MESSAGES,
            );
            let response = self
                .streaming_processor
                .clone()
                .process_messages_streaming_response(
                    execution_result,
                    messages_request,
                    dispatch,
                    tokenizer,
                    skip_special_tokens,
                    reservation,
                    ctx.load_guards.take(),
                    source,
                )
                .await;

            return Ok(Some(response));
        }

        // Non-streaming: delegate to ResponseProcessor
        let stop_decoder = stage_output(ctx.response.stop_decoder.as_mut(), FUNCTION)?;

        let (response, counts) = self
            .processor
            .process_non_streaming_messages_response(
                execution_result,
                messages_request,
                dispatch,
                tokenizer,
                stop_decoder,
            )
            .await?;

        // Store the final response, and the engine's counters for the settlement
        ctx.response.final_response = Some(FinalResponse::Messages(Box::new(response)));
        ctx.response.engine_counts = Some(counts);

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "MessageResponseProcessing"
    }
}
