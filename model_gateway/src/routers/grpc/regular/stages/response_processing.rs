//! Response processing stage for the chat + generate pipeline
//!
//! Dispatches to ChatResponseProcessingStage or GenerateResponseProcessingStage
//! based on request type.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatResponseProcessingStage, generate::GenerateResponseProcessingStage};
use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{RequestContext, RequestKind},
        regular::{processor, streaming},
    },
};

/// Response processing stage for chat + generate pipelines
pub(crate) struct ChatGenerateResponseProcessingStage {
    chat_stage: ChatResponseProcessingStage,
    generate_stage: GenerateResponseProcessingStage,
}

impl ChatGenerateResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
    ) -> Self {
        Self {
            chat_stage: ChatResponseProcessingStage::new(
                processor.clone(),
                streaming_processor.clone(),
            ),
            generate_stage: GenerateResponseProcessingStage::new(processor, streaming_processor),
        }
    }
}

#[async_trait]
impl PipelineStage for ChatGenerateResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Dispatch on the kind, which survives payload release.
        match ctx.input.request_type.kind() {
            RequestKind::Chat => self.chat_stage.execute(ctx).await,
            RequestKind::Generate => self.generate_stage.execute(ctx).await,
            request_kind => {
                error!(
                    function = "ChatGenerateResponseProcessingStage::execute",
                    request_type = %request_kind,
                    "{request_kind} should not reach this stage"
                );
                Err(error::internal_error(
                    "wrong_pipeline",
                    format!("{request_kind} should use its dedicated pipeline"),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "ChatGenerateResponseProcessing"
    }
}
