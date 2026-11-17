//! Transcription response processing: decode plain text (no tool/reasoning
//! parsing), post-process with the family parser, store the transcript.

use std::time::Instant;

use async_trait::async_trait;
use axum::response::Response;

use crate::{
    observability::metrics::metrics_labels,
    routers::{
        error,
        grpc::{
            common::{response_formatting::TokenCounts, stages::ProcessStage},
            context::{stage_output, wrong_response_spec, DispatchContext, FinalResponse},
            regular::processor,
            spec::ResponseSpec,
        },
    },
};

const FUNCTION: &str = "TranscriptionResponseProcessingStage::process";

/// Transcription response processing stage.
pub(crate) struct TranscriptionResponseProcessingStage {
    processor: processor::ResponseProcessor,
}

impl TranscriptionResponseProcessingStage {
    pub fn new(processor: processor::ResponseProcessor) -> Self {
        Self { processor }
    }
}

#[async_trait]
impl ProcessStage for TranscriptionResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Transcription(spec) = spec else {
            return Err(wrong_response_spec(FUNCTION));
        };

        let start_time = Instant::now();

        let execution_result = stage_output(ctx.response.execution_result.take(), FUNCTION)?;

        let dispatch = stage_output(ctx.dispatch.take(), FUNCTION)?;

        let stop_decoder = stage_output(ctx.response.stop_decoder.as_mut(), FUNCTION)?;

        // Plain-text decode via the generate processor — no tool/reasoning
        // parsing, whole-file (transcription rejects streaming in preparation).
        let responses = self
            .processor
            .process_non_streaming_generate_response(
                execution_result,
                dispatch,
                stop_decoder,
                /* request_logprobs */ false,
                start_time,
                metrics_labels::ENDPOINT_AUDIO_TRANSCRIPTIONS,
            )
            .await?;

        // Whitespace-only decode counts as empty, matching the chat path's
        // assistant-content normalization the old endpoint sat on.
        let raw = responses
            .first()
            .map(|response| response.text.as_str())
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                error::internal_error(
                    "empty_transcription_response",
                    format!("{} returned no transcription text", spec.family.name()),
                )
            })?;

        let text = spec.family.parse_transcript(raw);
        // The transcript, and the engine's counters for the settlement: the
        // answer itself carries no usage.
        ctx.response.engine_counts = Some(TokenCounts::of_generated(&responses));
        ctx.response.final_response = Some(FinalResponse::Transcription {
            text,
            format: spec.format,
        });
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "TranscriptionResponseProcessing"
    }
}
