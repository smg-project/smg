//! Generate preparation stage: Resolve input, tokenize, create stop decoder

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{
    common::{InputIds, StringOrArray},
    generate::GenerateRequest,
};
use tracing::error;

use crate::{
    routers::{
        error,
        grpc::{
            context::{PreparationOutput, RequestContext},
            regular::stages::token_ids::{reject_unknown_token_id, reported_embedding_rows},
            utils,
        },
    },
    worker::WorkerRegistry,
};

/// Generate preparation stage
///
/// Extracts generate-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct GeneratePreparationStage;

impl GeneratePreparationStage {
    /// The stage's work. The chat+generate dispatcher calls this directly, so
    /// the stage's future lives inline in the dispatcher's: one box per stage
    /// call, not a second one for a boxed trait method.
    pub(crate) async fn prepare(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let request = ctx.generate_request_arc();
        self.prepare_generate(ctx, &request).await?;
        Ok(())
    }
}

impl GeneratePreparationStage {
    async fn prepare_generate(
        &self,
        ctx: &mut RequestContext,
        request: &GenerateRequest,
    ) -> Result<(), Response> {
        // The route declares media fields, and the paths that forward the
        // body as given carry them to the engine; this pipeline sends text
        // and token ids only. A request that names media is refused as the
        // chat route refuses media no worker takes, instead of being served
        // without it.
        let media = request.media_fields();
        if !media.is_empty() {
            return Err(error::bad_request(
                "multimodal_not_supported",
                format!(
                    "media input ({}) is not supported on the generate route over the gateway's \
                     gRPC path: the route carries text and token ids to the engine, no media; send \
                     media through the chat route",
                    media.join(", ")
                ),
            ));
        }

        // Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer = utils::resolve_tokenizer(ctx, "GeneratePreparationStage::prepare_generate")
            .map_err(|e| *e)?;

        let (original_text, token_ids) = match self
            .resolve_generate_input(
                request,
                &tokenizer,
                &ctx.components.worker_registry,
                &ctx.input.model_id,
            )
            .await
        {
            Ok(res) => res,
            Err(msg) => {
                error!(function = "GeneratePreparationStage::execute", error = %msg, "Failed to resolve generate input");
                return Err(error::bad_request("resolve_input_failed", msg));
            }
        };

        // Create stop sequence decoder for generate requests
        let params = request.sampling_params.as_ref();
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            params.and_then(|p| p.stop.as_ref()),
            params.and_then(|p| p.stop_token_ids.as_ref()),
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            params.and_then(|p| p.no_stop_trim).unwrap_or(false),
            params.and_then(|p| p.ignore_eos).unwrap_or(false),
        );

        ctx.state.preparation = Some(PreparationOutput::Generate {
            original_text,
            token_ids,
        });

        // Store stop decoder
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    async fn resolve_generate_input(
        &self,
        request: &GenerateRequest,
        tokenizer: &Arc<dyn Tokenizer>,
        worker_registry: &WorkerRegistry,
        model_id: &str,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        // The engine's order: the ids before the text, so a body that carries
        // both is served from the ids as the engine serves it (the text rode
        // along for the client's own reading); a text-first read dropped the
        // ids without a word.
        if let Some(input_ids) = &request.input_ids {
            return match input_ids {
                InputIds::Single(ids) => {
                    let converted = ids
                        .iter()
                        .map(|&id| u32::try_from(id))
                        .collect::<Result<Vec<u32>, _>>()
                        .map_err(|_| "input_ids must be non-negative".to_string())?;
                    reject_unknown_token_id(tokenizer.as_ref(), &converted, "input_ids", || {
                        reported_embedding_rows(worker_registry, model_id)
                    })?;
                    Ok((None, converted))
                }
                InputIds::Batch(_) => {
                    Err("Batch input_ids are not supported over gRPC generate yet".to_string())
                }
            };
        }

        if let Some(text) = single_text(request)? {
            return self
                .tokenize_single_text(tokenizer, text)
                .await
                .map(|(original, ids)| (Some(original), ids));
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    async fn tokenize_single_text(
        &self,
        tokenizer: &Arc<dyn Tokenizer>,
        text: &str,
    ) -> Result<(String, Vec<u32>), String> {
        // Don't add special tokens - raw text generation uses text as-is
        let encoding = utils::encode_blocking(tokenizer.clone(), text, false)
            .await
            .map_err(|e| format!("Tokenization failed: {e}"))?;
        Ok((text.to_string(), encoding.into_token_ids()))
    }
}

/// The request's text when it names one prompt. A batch of texts is refused
/// by name, as a batch of `input_ids` is: this pipeline sends one prompt.
fn single_text(request: &GenerateRequest) -> Result<Option<&str>, String> {
    match &request.text {
        Some(StringOrArray::String(text)) => Ok(Some(text)),
        Some(StringOrArray::Array(_)) => {
            Err("Batch text is not supported over gRPC generate yet".to_string())
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(extra: serde_json::Value) -> GenerateRequest {
        let mut body = json!({"model": "m", "text": "hi"});
        body.as_object_mut()
            .expect("an object")
            .extend(extra.as_object().expect("an object").clone());
        serde_json::from_value(body).expect("a generate request")
    }

    /// A batch of texts is refused by name, as a batch of `input_ids` is:
    /// the pipeline sends one prompt. One text is the prompt, and a body of
    /// ids names none.
    #[test]
    fn a_batch_of_texts_is_refused_by_name() {
        assert_eq!(
            single_text(&request(json!({"text": ["a", "b"]}))),
            Err("Batch text is not supported over gRPC generate yet".to_string())
        );
        assert_eq!(single_text(&request(json!({}))), Ok(Some("hi")));
        let ids: GenerateRequest =
            serde_json::from_value(json!({"model": "m", "input_ids": [1, 2]})).expect("ids");
        assert_eq!(single_text(&ids), Ok(None));
    }
}
