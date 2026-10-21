//! Preparation stage for the chat + generate pipeline
//!
//! Dispatches to ChatPreparationStage or GeneratePreparationStage based on
//! request type.

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatPreparationStage, generate::GeneratePreparationStage};
use crate::{
    routers::{
        error as grpc_error,
        grpc::{
            common::stages::PipelineStage,
            context::{RequestContext, RequestType},
        },
    },
    worker::RoutingPool,
};

/// Preparation stage for chat + generate pipelines
pub(crate) struct ChatGeneratePreparationStage {
    chat_stage: ChatPreparationStage,
    generate_stage: GeneratePreparationStage,
}

impl ChatGeneratePreparationStage {
    /// `wire_pools`: the pools the pipeline's mode selects from, for the
    /// multimodal wire dtype.
    pub fn new(wire_pools: &'static [RoutingPool]) -> Self {
        Self {
            chat_stage: ChatPreparationStage::new(wire_pools),
            generate_stage: GeneratePreparationStage,
        }
    }
}

#[async_trait]
impl PipelineStage for ChatGeneratePreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
            request_type => {
                error!(
                    function = "ChatGeneratePreparationStage::execute",
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
        "ChatGeneratePreparation"
    }
}
