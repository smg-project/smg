//! Completion preparation stage: resolve prompt(s), tokenize, create stop decoder.
//!
//! This is the `/v1/completions` Stage 1 equivalent. It intentionally builds on top of
//! the native completion pipeline typing introduced in PR #840. It keeps
//! `CompletionRequest` native in the request context instead of laundering it
//! through `GenerateRequest`.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use futures::future::try_join_all;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::completion::CompletionPrompt;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{CompletionItem, PreparationOutput, RequestContext},
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

        // Empty arrays are rejected at the boundary (`validate_completion_prompt`).
        let items = match &request.prompt {
            CompletionPrompt::Text(text) => {
                encode_prompts(&tokenizer, std::slice::from_ref(text)).await?
            }
            CompletionPrompt::Texts(texts) => encode_prompts(&tokenizer, texts).await?,
            CompletionPrompt::TokenIds(ids) => vec![token_prompt(tokenizer.as_ref(), ids)?],
            CompletionPrompt::TokenIdBatches(batches) => batches
                .iter()
                .map(|ids| token_prompt(tokenizer.as_ref(), ids))
                .collect::<Result<_, _>>()?,
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

/// Tokenize text prompts (without special tokens, as a completion's text is
/// sent as written).
async fn encode_prompts(
    tokenizer: &Arc<dyn Tokenizer>,
    prompts: &[String],
) -> Result<Vec<CompletionItem>, Response> {
    let encodings = try_join_all(
        prompts
            .iter()
            .map(|prompt| utils::encode_blocking(tokenizer.clone(), prompt.clone(), false)),
    )
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
            token_ids: encoding.token_ids().to_vec(),
        })
        .collect())
}

/// A prompt given as token ids goes to the engine as given; its text (for
/// `echo` and routing) is their decode with special tokens kept. An id the
/// tokenizer does not know is refused: it has no text and no logprob column.
fn token_prompt(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Result<CompletionItem, Response> {
    if let Some(&id) = ids.iter().find(|&&id| tokenizer.id_to_token(id).is_none()) {
        return Err(error::bad_request(
            "invalid_prompt_token",
            format!("prompt token id {id} is not in the tokenizer's vocabulary"),
        ));
    }
    let text = tokenizer.decode(ids, false).map_err(|e| {
        error::bad_request(
            "invalid_prompt_token",
            format!("prompt token ids cannot be decoded: {e}"),
        )
    })?;
    Ok(CompletionItem {
        text,
        token_ids: ids.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use llm_tokenizer::{traits::Decoder, MockTokenizer};

    use super::*;

    #[test]
    fn a_token_prompt_goes_as_given_and_echoes_its_decode() {
        let tokenizer = MockTokenizer::new();
        let ids: Vec<u32> = ["Hello", "world"]
            .iter()
            .map(|token| tokenizer.token_to_id(token).expect("mock vocab"))
            .collect();
        let item = token_prompt(&tokenizer, &ids).expect("known ids");
        assert_eq!(item.token_ids, ids);
        assert_eq!(item.text, tokenizer.decode(&ids, false).expect("decode"));
        assert!(!item.text.is_empty());
    }

    #[test]
    fn a_token_prompt_with_an_unknown_id_is_refused() {
        let tokenizer = MockTokenizer::new();
        let known = tokenizer.token_to_id("Hello").expect("mock vocab");
        let unknown = u32::MAX;
        assert!(tokenizer.id_to_token(unknown).is_none());
        let response = token_prompt(&tokenizer, &[known, unknown])
            .err()
            .expect("unknown id must be refused");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
