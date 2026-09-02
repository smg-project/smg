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
pub(crate) struct TranscriptionRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl TranscriptionRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
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
        // request synthesized in preparation. Transcription carries no tool
        // constraints.
        let (plan, stamp) = build_chat_backed_plan(
            ctx,
            &chat_request,
            processed_messages.text,
            token_ids,
            None,
            "transcription-",
            self.inject_pd_metadata,
            self.plan_kind,
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
        format!(
            "TranscriptionRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}
