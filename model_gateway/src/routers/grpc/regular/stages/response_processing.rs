//! Response processing stage for the chat + generate pipeline
//!
//! Dispatches to ChatResponseProcessingStage or GenerateResponseProcessingStage
//! based on the response spec.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::{chat::ChatResponseProcessingStage, generate::GenerateResponseProcessingStage};
use crate::routers::{
    error,
    grpc::{
        common::stages::ProcessStage,
        context::DispatchContext,
        regular::{processor, streaming},
        spec::ResponseSpec,
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
impl ProcessStage for ChatGenerateResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        // Dispatch on the spec: the only request-derived signal post-build.
        match spec {
            ResponseSpec::Chat(_) => self.chat_stage.process(ctx, spec).await,
            ResponseSpec::Generate(_) => self.generate_stage.process(ctx, spec).await,
            _ => {
                error!(
                    function = "ChatGenerateResponseProcessingStage::process",
                    "response spec should not reach this stage"
                );
                Err(error::internal_error(
                    "wrong_pipeline",
                    "response spec should use its dedicated pipeline",
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "ChatGenerateResponseProcessing"
    }
}
