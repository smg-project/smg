//! Preparation stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::common::GenerationRequest;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext, RequestType},
        utils,
    },
};

pub(crate) struct EmbeddingPreparationStage;

impl EmbeddingPreparationStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl PipelineStage for EmbeddingPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        // Extract text from embedding or classify request (both use same preparation)
        let text = match &ctx.input.request_type {
            RequestType::Embedding(req) => req.extract_text_for_routing(),
            RequestType::Classify(req) => req.extract_text_for_routing(),
            _ => {
                error!(
                    function = "EmbeddingPreparationStage::execute",
                    "Invalid request type: expected Embedding or Classify"
                );
                return Err(error::internal_error(
                    "invalid_request_type",
                    "Expected Embedding or Classify request",
                ));
            }
        };
        if text.is_empty() {
            return Err(error::bad_request(
                "empty_input",
                "Input text cannot be empty",
            ));
        }

        // Resolve tokenizer from registry (cached for potential reuse)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "EmbeddingPreparationStage::execute").map_err(|e| *e)?;

        // Tokenize with special tokens (BOS/EOS) for embeddings
        // This matches Python's transformers behavior which reads add_bos_token/add_eos_token from tokenizer_config.json
        let token_ids = utils::encode_blocking(tokenizer.clone(), text.clone(), true)
            .await
            .map_err(|e| {
                error!(
                    function = "EmbeddingPreparationStage::execute",
                    error = %e,
                    "Tokenization failed"
                );
                error::bad_request("tokenization_failed", format!("Tokenization failed: {e}"))
            })?
            .token_ids()
            .to_vec();

        // Store preparation output
        ctx.state.preparation = Some(PreparationOutput::Embedding {
            original_text: text,
            token_ids,
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "EmbeddingPreparation"
    }
}
