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

/// The text prompts encoded (without special tokens: a completion's text is
/// sent as written), one item per prompt.
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
            token_ids: encoding.into_token_ids(),
        })
        .collect())
}

/// A prompt given as token ids goes to the engine as given, with no encode.
/// Its text, for `echo` and for the policies that route on text, is the ids'
/// decode with special tokens kept, as the engines' own servers echo a token
/// prompt. An id the vocabulary lacks has no text here; the engine refuses
/// it, as it does on `/generate`.
fn token_prompt(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Result<CompletionItem, Response> {
    let text = tokenizer.decode(ids, false).map_err(|e| {
        error!(
            function = "CompletionPreparationStage::execute",
            error = %e,
            "Prompt token ids cannot be decoded"
        );
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
    use llm_tokenizer::MockTokenizer;

    use super::*;

    /// A token prompt keeps its ids and takes their decode as its text, the
    /// text `echo` prepends and the text-reading policies route on.
    #[test]
    fn a_token_prompt_goes_as_given_with_its_decode_as_text() {
        let tokenizer = MockTokenizer::new();
        let ids: Vec<u32> = ["Hello", "world"]
            .iter()
            .map(|token| tokenizer.token_to_id(token).expect("mock vocab"))
            .collect();

        let item = token_prompt(&tokenizer, &ids).expect("known ids");

        assert_eq!(item.token_ids, ids);
        assert_eq!(item.text, "Hello world");
    }

    /// An id the vocabulary lacks decodes to nothing and still rides the
    /// prompt: the engine, not the gateway, refuses it.
    #[test]
    fn an_unknown_id_has_no_text_and_still_rides_the_prompt() {
        let tokenizer = MockTokenizer::new();

        let item = token_prompt(&tokenizer, &[1, 424_242]).expect("decode keeps what it knows");

        assert_eq!(item.token_ids, vec![1, 424_242]);
        assert_eq!(item.text, "Hello");
    }
}
