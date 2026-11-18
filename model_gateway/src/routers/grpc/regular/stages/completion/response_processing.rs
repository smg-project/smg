//! Completion response processing stage
//!
//! Stage 7 for the `/v1/completions` pipeline
//!
//! - For streaming: builds the reading of the engine's answer and returns the SSE response
//!   whose body drives it (early exit)
//! - For non-streaming: collects the backend response, converts it to
//!   `CompletionResponse`, and stores it as `FinalResponse::Completion`.

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

const FUNCTION: &str = "CompletionResponseProcessingStage::process";

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
            return Err(wrong_response_spec(FUNCTION));
        };

        let is_streaming = ctx.streaming;

        let execution_result = stage_output(ctx.response.execution_result.take(), FUNCTION)?;

        let dispatch = stage_output(ctx.dispatch.take(), FUNCTION)?;

        let tokenizer = stage_output(ctx.tokenizer_arc(), FUNCTION)?;

        if is_streaming {
            let reservation = RateLimitCell::stream_reservation(ctx.rate_limit_cell.as_deref());

            let source = StreamSource::new(
                ctx.workers.as_ref(),
                &dispatch,
                metrics_labels::ENDPOINT_COMPLETIONS,
            );
            let response = self
                .streaming_processor
                .clone()
                .process_completion_streaming_response(
                    execution_result,
                    completion_request,
                    dispatch,
                    tokenizer,
                    reservation,
                    ctx.load_guards.take(),
                    source,
                )
                .await;

            return Ok(Some(response));
        }

        // Non-streaming path
        let stop_decoder = stage_output(ctx.response.stop_decoder.as_mut(), FUNCTION)?;

        let response = self
            .processor
            .process_non_streaming_completion_response(
                execution_result,
                completion_request,
                dispatch,
                &tokenizer,
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
