//! Generate response processing stage: Handles both streaming and non-streaming responses

use std::{sync::Arc, time::Instant};

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

const FUNCTION: &str = "GenerateResponseProcessingStage::process";

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
            return Err(wrong_response_spec(FUNCTION));
        };

        let start_time = Instant::now();
        let is_streaming = ctx.streaming;

        let execution_result = stage_output(ctx.response.execution_result.take(), FUNCTION)?;

        let dispatch = stage_output(ctx.dispatch.take(), FUNCTION)?;

        let tokenizer = stage_output(ctx.tokenizer_arc(), FUNCTION)?;

        if is_streaming {
            let reservation = RateLimitCell::stream_reservation(ctx.rate_limit_cell.as_deref());

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

        let stop_decoder = stage_output(ctx.response.stop_decoder.as_mut(), FUNCTION)?;

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
