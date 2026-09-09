//! Transcription request building: reuse the chat backend-request build path
//! on the request synthesized in preparation, producing a transcription spec.

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::BuildStage,
        context::{BuildOutput, ExecutionPlanKind, PreparationOutput, RequestContext},
        regular::stages::chat::build_chat_backed_plan,
        spec::{ResponseSpec, TranscriptionResponseSpec},
    },
};

/// Transcription request building stage.
///
/// Transcription is Regular-only (the pipeline is never built under PD/EPD),
/// so the backend plan is always a single request with no PD metadata — this
/// stage carries no disaggregation parameters.
pub(crate) struct TranscriptionRequestBuildingStage;

impl TranscriptionRequestBuildingStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl BuildStage for TranscriptionRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "TranscriptionRequestBuildingStage::build",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let PreparationOutput::Transcription {
            token_ids,
            processed_messages,
            chat_request,
            format,
            family,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Transcription variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Transcription preparation output",
            ));
        };

        // The backend request is chat-shaped: reuse the chat build path on the
        // request synthesized in preparation. Regular-only, so a single-plan
        // request with no PD metadata and no tool constraints.
        let (plan, stamp) = build_chat_backed_plan(
            ctx,
            &chat_request,
            processed_messages.text,
            token_ids,
            None,
            "transcription-",
            /* inject_pd_metadata */ false,
            ExecutionPlanKind::Single,
        )
        .await?;

        Ok(BuildOutput {
            plan,
            spec: ResponseSpec::Transcription(TranscriptionResponseSpec { format, family }),
            stamp,
        })
    }

    fn name(&self) -> &'static str {
        "TranscriptionRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        "TranscriptionRequestBuildingStage".to_string()
    }
}
