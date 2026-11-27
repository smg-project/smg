//! Completion preparation stage: resolve prompt(s), tokenize, create stop decoder.
//!
//! This is the `/v1/completions` Stage 1 equivalent. It intentionally builds on top of
//! the native completion pipeline typing introduced in PR #840. It keeps
//! `CompletionRequest` native in the request context instead of laundering it
//! through `GenerateRequest`.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::completion::CompletionPrompt;
use tracing::{error, warn};

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{CompletionItem, PreparationOutput, RequestContext},
        regular::stages::token_ids::{
            decodable_ids, reject_unknown_token_id, reported_embedding_rows,
            validate_logit_bias_token_ids,
        },
        utils,
    },
};

pub(crate) struct CompletionPreparationStage;

#[async_trait]
impl PipelineStage for CompletionPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let request = ctx.completion_request_arc();

        let tokenizer =
            utils::resolve_tokenizer(ctx, "CompletionPreparationStage::execute").map_err(|e| *e)?;

        // The embedding table the model's workers report, read only after an id the
        // tokenizer does not know (the bound of the `logit_bias` keys and of `token_prompt`).
        let worker_registry = &ctx.components.worker_registry;
        let model_id = ctx.input.model_id.as_str();
        let embedding_rows = || reported_embedding_rows(worker_registry, model_id);

        validate_logit_bias_token_ids(
            request.logit_bias.as_ref(),
            tokenizer.as_ref(),
            embedding_rows,
        )
        .map_err(|e| error::bad_request("logit_bias_invalid_key", e))?;

        // Empty arrays are rejected at the boundary (`validate_completion_prompt`).
        let items = match &request.prompt {
            CompletionPrompt::Text(text) => {
                encode_prompts(&tokenizer, std::slice::from_ref(text)).await?
            }
            CompletionPrompt::Texts(texts) => encode_prompts(&tokenizer, texts).await?,
            CompletionPrompt::TokenIds(ids) => {
                let fields = ["prompt".to_string()];
                token_prompts(
                    &tokenizer,
                    std::slice::from_ref(ids),
                    &fields,
                    embedding_rows,
                )
                .await?
            }
            CompletionPrompt::TokenIdBatches(batches) => {
                let fields: Vec<String> =
                    (0..batches.len()).map(|b| format!("prompt[{b}]")).collect();
                token_prompts(&tokenizer, batches, &fields, embedding_rows).await?
            }
        };

        let joined_routing_text = (items.len() > 1).then(|| {
            items
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        });

        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
            request.ignore_eos,
        );

        ctx.state.preparation = Some(PreparationOutput::Completion {
            items,
            joined_routing_text,
        });

        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    fn name(&self) -> &'static str {
        "CompletionPreparation"
    }
}

/// The text prompts encoded (without special tokens: a completion's text is
/// sent as written), one item per prompt.
async fn encode_prompts(
    tokenizer: &Arc<dyn Tokenizer>,
    prompts: &[String],
) -> Result<Vec<CompletionItem>, Response> {
    let encodings = utils::encode_texts_blocking(tokenizer.clone(), prompts, false)
        .await
        .map_err(|e| {
            error!(
                function = "CompletionPreparationStage::execute",
                error = %e,
                "Tokenization failed"
            );
            error::bad_request("tokenization_failed", format!("Tokenization failed: {e}"))
        })?;

    Ok(prompts
        .iter()
        .zip(encodings)
        .map(|(text, encoding)| CompletionItem {
            text: text.clone(),
            token_ids: encoding.into_token_ids(),
        })
        .collect())
}

/// The token prompts of a call, one item per list. A prompt given as token
/// ids goes to the engine as given, with no encode, once every id is bound to
/// a row of the model's embedding table ([`reject_unknown_token_id`], the rule
/// `/generate` applies to `input_ids`: on the headless lanes an id past the
/// table is an out-of-range index on the engine, not a refused request). Its
/// text, for `echo` and for the policies that route on text, is the decode of
/// the ids the tokenizer knows ([`decodable_ids`]) with special tokens kept,
/// as the engines' own servers echo a token prompt: an id known only to the
/// engine's table rides the wire and leaves no text, on every decoder (the
/// byte-level and HF decoders would drop it, the tiktoken decoder would
/// refuse it). An id inside the tokenizer's range that its encoding never
/// assigned (a hole of a tiktoken id space) is the decode's to find, and is
/// refused as the client's error. The lists are checked first, the first
/// refusal ending the call, then decoded as one pass of the call
/// ([`utils::decode_ids_blocking`]): in place up to the bound, on the pool
/// past it, as a call's texts are ([`encode_prompts`]). `fields` names each
/// list in a refusal: `prompt`, or `prompt[b]` for one list of a batch.
async fn token_prompts(
    tokenizer: &Arc<dyn Tokenizer>,
    batches: &[Vec<u32>],
    fields: &[String],
    embedding_rows: impl FnOnce() -> Option<usize> + Copy,
) -> Result<Vec<CompletionItem>, Response> {
    let mut decodable = Vec::with_capacity(batches.len());
    for (ids, field) in batches.iter().zip(fields) {
        reject_unknown_token_id(tokenizer.as_ref(), ids, field, embedding_rows).map_err(
            |message| {
                warn!(
                    function = "CompletionPreparationStage::execute",
                    error = %message,
                    "Prompt token id outside the model's vocabulary"
                );
                error::bad_request("invalid_prompt_token", message)
            },
        )?;
        decodable.push(decodable_ids(tokenizer.as_ref(), ids));
    }
    let texts = utils::decode_ids_blocking(tokenizer.clone(), decodable, false)
        .await
        .map_err(|e| {
            // The range check behind `decodable_ids` is a fast path, not a guarantee: a
            // tiktoken encoding leaves holes below its reported size (ids with neither a
            // rank nor a special token), which only the decode finds. Such an id is the
            // client's to drop, as one past the bound is.
            warn!(
                function = "CompletionPreparationStage::execute",
                error = %e,
                "Prompt token ids cannot be decoded"
            );
            error::bad_request(
                "invalid_prompt_token",
                format!("prompt token ids cannot be decoded: {e}"),
            )
        })?;
    Ok(batches
        .iter()
        .zip(texts)
        .map(|(ids, text)| CompletionItem {
            text,
            token_ids: ids.to_vec(),
        })
        .collect())
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/routers/grpc/regular/stages/completion/preparation/tests.rs"]
mod tests;
