//! Request building stage for chat and generate endpoints

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatRequestBuildingStage, generate::GenerateRequestBuildingStage};
use crate::routers::{
    error as grpc_error,
    grpc::{
        common::stages::BuildStage,
        context::{BuildOutput, ExecutionPlanKind, RequestContext, RequestType},
    },
};

/// Request building stage for chat and generate pipelines
///
/// These two request types share a single pipeline instance and are dispatched
/// here. All other request types have dedicated pipelines and wire their own
/// request building stages directly.
pub(crate) struct ChatGenerateRequestBuildingStage {
    chat_stage: ChatRequestBuildingStage,
    generate_stage: GenerateRequestBuildingStage,
}

impl ChatGenerateRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            chat_stage: ChatRequestBuildingStage::new(inject_pd_metadata, plan_kind),
            generate_stage: GenerateRequestBuildingStage::new(inject_pd_metadata, plan_kind),
        }
    }
}

#[async_trait]
impl BuildStage for ChatGenerateRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.build(ctx).await,
            RequestType::Generate(_) => self.generate_stage.build(ctx).await,
            request_type => {
                error!(
                    function = "ChatGenerateRequestBuildingStage::build",
                    request_type = %request_type,
                    "{request_type} should not reach this stage"
                );
                Err(grpc_error::internal_error(
                    "wrong_pipeline",
                    format!("{request_type} should use its dedicated pipeline"),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "ChatGenerateRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "ChatGenerateRequestBuildingStage({}, {})",
            self.chat_stage.signature(),
            self.generate_stage.signature()
        )
    }
}
