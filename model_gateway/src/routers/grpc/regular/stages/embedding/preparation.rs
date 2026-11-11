//! Preparation stage for embedding and classification requests: the inputs
//! of the request, one per member of its batch, each tokenized by the gateway
//! or taken as the token ids the client sent.

use async_trait::async_trait;
use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::completion::CompletionPrompt;
use serde::Deserialize;
use serde_json::Value;
use tracing::{error, warn};

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{EmbeddingInput, PreparationOutput, RequestContext, RequestType},
        regular::stages::token_ids::{reject_unknown_token_id, reported_embedding_rows},
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
        // The input of an embedding or classify request (both are prepared here).
        let input = match &ctx.input.request_type {
            RequestType::Embedding(req) => &req.input,
            RequestType::Classify(req) => &req.input,
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
        let input = embedding_inputs(input)?;

        // Resolve tokenizer from registry (cached for potential reuse)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "EmbeddingPreparationStage::execute").map_err(|e| *e)?;

        // The embedding table the model's workers report, read only after an
        // id the tokenizer does not know (the bound of a token-id input).
        let worker_registry = &ctx.components.worker_registry;
        let model_id = ctx.input.model_id.as_str();
        let embedding_rows = || reported_embedding_rows(worker_registry, model_id);

        let inputs = match input {
            CompletionPrompt::Text(text) => encode_inputs(&tokenizer, vec![text]).await?,
            CompletionPrompt::Texts(texts) => encode_inputs(&tokenizer, texts).await?,
            CompletionPrompt::TokenIds(ids) => {
                vec![token_input(
                    tokenizer.as_ref(),
                    ids,
                    "input",
                    embedding_rows,
                )?]
            }
            CompletionPrompt::TokenIdBatches(batches) => batches
                .into_iter()
                .enumerate()
                .map(|(b, ids)| {
                    token_input(
                        tokenizer.as_ref(),
                        ids,
                        &format!("input[{b}]"),
                        embedding_rows,
                    )
                })
                .collect::<Result<_, _>>()?,
        };

        ctx.state.preparation = Some(PreparationOutput::Embedding { inputs });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "EmbeddingPreparation"
    }
}

/// The client-visible code of an input of a shape the contract has no form
/// for (an object, a mixed array, a negative id).
pub(crate) const INVALID_INPUT_CODE: &str = "invalid_input";
/// The client-visible code of an input with nothing to embed: an empty text
/// or id list, or a batch with an empty member.
pub(crate) const EMPTY_INPUT_CODE: &str = "empty_input";
/// The client-visible code of a token-id input with an id the model has no
/// row for.
pub(crate) const INVALID_INPUT_TOKEN_CODE: &str = "invalid_input_token";

/// The request's `input` in the four shapes of the public contract, the ones
/// a completion's `prompt` takes: one text, a batch of texts, one input of
/// token ids, or a batch of token-id inputs. Any other shape, an empty input
/// and an empty member of a batch are refused by name, before a tokenizer or
/// a worker is spoken to.
fn embedding_inputs(input: &Value) -> Result<CompletionPrompt, Response> {
    let prompt = CompletionPrompt::deserialize(input).map_err(|e| {
        error::bad_request(
            INVALID_INPUT_CODE,
            format!(
                "input must be a string, an array of strings, an array of token ids or an \
                 array of token-id arrays: {e}"
            ),
        )
    })?;
    let empty = match &prompt {
        CompletionPrompt::Text(text) => text.is_empty().then(|| "input".to_owned()),
        CompletionPrompt::TokenIds(ids) => ids.is_empty().then(|| "input".to_owned()),
        CompletionPrompt::Texts(texts) => empty_member(texts.iter().map(String::is_empty)),
        CompletionPrompt::TokenIdBatches(batches) => {
            empty_member(batches.iter().map(Vec::is_empty))
        }
    };
    match empty {
        Some(field) => Err(error::bad_request(
            EMPTY_INPUT_CODE,
            format!("{field} cannot be empty"),
        )),
        None => Ok(prompt),
    }
}

/// The name of the first empty member of a batch, or of the batch itself
/// when it has none.
fn empty_member(members: impl ExactSizeIterator<Item = bool>) -> Option<String> {
    if members.len() == 0 {
        return Some("input".to_owned());
    }
    members
        .enumerate()
        .find_map(|(i, empty)| empty.then(|| format!("input[{i}]")))
}

/// The text inputs encoded with the model's special tokens (as the
/// transformers tokenizers add them for embeddings), one input per text: one
/// text takes the single encode, a batch the in-place pass over its texts.
async fn encode_inputs(
    tokenizer: &std::sync::Arc<dyn Tokenizer>,
    texts: Vec<String>,
) -> Result<Vec<EmbeddingInput>, Response> {
    let encodings = match texts.as_slice() {
        [text] => utils::encode_blocking(tokenizer.clone(), text, true)
            .await
            .map(|encoding| vec![encoding]),
        _ => utils::encode_texts_blocking(tokenizer.clone(), &texts, true).await,
    };
    let encodings = encodings.map_err(|e| {
        error!(
            function = "EmbeddingPreparationStage::execute",
            error = %e,
            "Tokenization failed"
        );
        error::bad_request("tokenization_failed", format!("Tokenization failed: {e}"))
    })?;
    Ok(texts
        .into_iter()
        .zip(encodings)
        .map(|(text, encoding)| EmbeddingInput {
            text: Some(text),
            token_ids: encoding.into_token_ids(),
        })
        .collect())
}

/// An input given as token ids: it goes to the engine as given, with no
/// encode and no text beside it, once every id is bound to a row of the
/// model's embedding table ([`reject_unknown_token_id`], the rule the
/// completion and `/generate` prompts apply). `field` names the input in a
/// refusal: `input`, or `input[b]` for one member of a batch.
fn token_input(
    tokenizer: &dyn Tokenizer,
    ids: Vec<u32>,
    field: &str,
    embedding_rows: impl FnOnce() -> Option<usize>,
) -> Result<EmbeddingInput, Response> {
    reject_unknown_token_id(tokenizer, &ids, field, embedding_rows).map_err(|message| {
        warn!(
            function = "EmbeddingPreparationStage::execute",
            error = %message,
            "Input token id outside the model's vocabulary"
        );
        error::bad_request(INVALID_INPUT_TOKEN_CODE, message)
    })?;
    Ok(EmbeddingInput {
        text: None,
        token_ids: ids,
    })
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::*;

    /// The error code of a refusal.
    async fn code_of(response: Response) -> (StatusCode, String) {
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("the body");
        let body: Value = serde_json::from_slice(&body).expect("the error envelope");
        (
            status,
            body["error"]["code"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        )
    }

    #[tokio::test]
    async fn the_four_input_shapes_parse_and_the_rest_are_refused_by_name() {
        let texts = embedding_inputs(&json!(["hello", "world"])).expect("a batch of texts");
        assert_eq!(
            texts,
            CompletionPrompt::Texts(vec!["hello".to_owned(), "world".to_owned()])
        );
        assert_eq!(
            embedding_inputs(&json!("hello")).expect("one text"),
            CompletionPrompt::Text("hello".to_owned())
        );
        assert_eq!(
            embedding_inputs(&json!([1, 2, 3])).expect("one input of ids"),
            CompletionPrompt::TokenIds(vec![1, 2, 3])
        );
        assert_eq!(
            embedding_inputs(&json!([[1, 2], [3]])).expect("a batch of id inputs"),
            CompletionPrompt::TokenIdBatches(vec![vec![1, 2], vec![3]])
        );
        for (shape, code) in [
            (json!({"text": "hello"}), INVALID_INPUT_CODE),
            (json!(["hello", 1]), INVALID_INPUT_CODE),
            (json!([-1, 2]), INVALID_INPUT_CODE),
            (json!(""), EMPTY_INPUT_CODE),
            (json!([]), EMPTY_INPUT_CODE),
            (json!(["hello", ""]), EMPTY_INPUT_CODE),
            (json!([[1, 2], []]), EMPTY_INPUT_CODE),
        ] {
            let refused = embedding_inputs(&shape).expect_err("refused");
            let (status, got) = code_of(refused).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{shape}");
            assert_eq!(got, code, "{shape}");
        }
    }

    #[tokio::test]
    async fn an_empty_member_is_refused_by_its_position() {
        let refused = embedding_inputs(&json!(["hello", "", "world"])).expect_err("refused");
        let body = axum::body::to_bytes(refused.into_body(), 4096)
            .await
            .expect("the body");
        let body: Value = serde_json::from_slice(&body).expect("the error envelope");
        assert_eq!(
            body["error"]["message"], "input[1] cannot be empty",
            "{body}"
        );
    }
}
