//! Transcription response processing: decode plain text (no tool/reasoning
//! parsing), post-process with the family parser, store the transcript.

use std::time::Instant;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::ProcessStage,
        context::{DispatchContext, FinalResponse},
        regular::processor,
        spec::ResponseSpec,
    },
};

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
            error!(
                function = "TranscriptionResponseProcessingStage::process",
                "Wrong response spec"
            );
            return Err(error::internal_error(
                "wrong_response_spec",
                "Wrong response spec",
            ));
        };

        let start_time = Instant::now();

        let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "TranscriptionResponseProcessingStage::process",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        let dispatch = ctx
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "TranscriptionResponseProcessingStage::process",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        let stop_decoder = ctx.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "TranscriptionResponseProcessingStage::process",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

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
            )
            .await?;

        let raw = responses
            .first()
            .map(|response| response.text.as_str())
            .filter(|text| !text.is_empty())
            .ok_or_else(|| {
                error::internal_error(
                    "empty_transcription_response",
                    format!("{} returned no transcription text", spec.family.name()),
                )
            })?;

        let text = spec.family.parse_transcript(raw);
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
