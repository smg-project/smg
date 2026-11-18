//! Chat response processing stage: Handles both streaming and non-streaming responses
//!
//! - For streaming: Builds the reading of the engine's answer and returns the SSE response
//!   whose body drives it (early exit)
//! - For non-streaming: Collects all responses and builds final ChatCompletionResponse

use std::sync::Arc;

use axum::response::Response;

use crate::{
    observability::metrics::metrics_labels,
    routers::grpc::{
        common::stages::RateLimitCell,
        context::{stage_output, wrong_response_spec, DispatchContext, FinalResponse},
        regular::{processor, streaming},
        spec::ResponseSpec,
        utils::StreamSource,
    },
};

const FUNCTION: &str = "ChatResponseProcessingStage::process";

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

impl ChatResponseProcessingStage {
    /// The stage's work. The chat+generate dispatcher calls this directly, so
    /// the stage's future lives inline in the dispatcher's: one box per stage
    /// call, not a second one for a boxed trait method.
    pub(crate) async fn process_result(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Chat(chat_request) = spec else {
            return Err(wrong_response_spec(FUNCTION));
        };

        let is_streaming = ctx.streaming;

        let execution_result = stage_output(ctx.response.execution_result.take(), FUNCTION)?;

        let dispatch = stage_output(ctx.dispatch.take(), FUNCTION)?;

        let tokenizer = stage_output(ctx.tokenizer_arc(), FUNCTION)?;

        if is_streaming {
            // Read derived skip_special_tokens (set in preparation, survives request_building .take())
            let skip_special_tokens = ctx
                .response
                .skip_special_tokens
                .unwrap_or(chat_request.skip_special_tokens);

            let reservation = RateLimitCell::stream_reservation(ctx.rate_limit_cell.as_deref());

            // Streaming: Use StreamingProcessor and return SSE response. The
            // reading consumes the spec, never the parsed request.
            let source = StreamSource::new(
                ctx.workers.as_ref(),
                &dispatch,
                metrics_labels::ENDPOINT_CHAT,
            );
            let response = self
                .streaming_processor
                .clone()
                .process_streaming_response(
                    execution_result,
                    *chat_request,
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

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = chat_request.logprobs;

        let stop_decoder = stage_output(ctx.response.stop_decoder.as_mut(), FUNCTION)?;

        let (response, counts) = self
            .processor
            .process_non_streaming_chat_response(
                execution_result,
                *chat_request,
                dispatch,
                tokenizer,
                stop_decoder,
                request_logprobs,
            )
            .await?;

        // Store the final response, and the engine's counters for the settlement
        ctx.response.final_response = Some(FinalResponse::Chat(response));
        ctx.response.engine_counts = Some(counts);

        Ok(None)
    }
}
