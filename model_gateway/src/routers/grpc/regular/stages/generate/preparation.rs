//! Generate preparation stage: Resolve input, tokenize, create stop decoder

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{common::InputIds, generate::GenerateRequest};
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
        let media = media_fields(request);
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
        if let Some(text) = &request.text {
            return self
                .tokenize_single_text(tokenizer, text)
                .await
                .map(|(original, ids)| (Some(original), ids));
        }

        // Handle input_ids - validate and convert
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

/// The media fields of `request` that carry something: `image_data`,
/// `video_data` and `audio_data` when present and naming an item (a null, an
/// empty list, or a list of nulls and empty lists is no media, as the engine
/// reads it).
fn media_fields(request: &GenerateRequest) -> Vec<&'static str> {
    [
        ("image_data", &request.image_data),
        ("video_data", &request.video_data),
        ("audio_data", &request.audio_data),
    ]
    .into_iter()
    .filter(|(_, value)| value.as_ref().is_some_and(carries_media))
    .map(|(name, _)| name)
    .collect()
}

/// Whether `value` names media as the engine reads it: anything but a null,
/// or a list, nested to any depth, one of whose items does.
fn carries_media(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Array(items) => items.iter().any(carries_media),
        _ => true,
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

    /// A body names media when one of the three fields carries something; a
    /// text-only body, a null, an empty list and a list of nulls and empty
    /// lists do not.
    #[test]
    fn the_media_fields_that_carry_something_are_named() {
        assert!(media_fields(&request(json!({}))).is_empty());
        assert!(media_fields(&request(json!({"image_data": null}))).is_empty());
        assert!(media_fields(&request(json!({"image_data": []}))).is_empty());
        assert!(media_fields(&request(json!({"image_data": [null]}))).is_empty());
        assert!(media_fields(&request(json!({"video_data": [[]]}))).is_empty());
        assert!(media_fields(&request(json!({"audio_data": [null, [], [null]]}))).is_empty());
        assert_eq!(
            media_fields(&request(json!({"image_data": [[null, "a.png"]]}))),
            ["image_data"]
        );
        assert_eq!(
            media_fields(&request(
                json!({"image_data": "data:image/png;base64,iVBORw0KGgo="})
            )),
            ["image_data"]
        );
        assert_eq!(
            media_fields(&request(
                json!({"video_data": ["clip.mp4"], "audio_data": "a.wav"})
            )),
            ["video_data", "audio_data"]
        );
    }
}
