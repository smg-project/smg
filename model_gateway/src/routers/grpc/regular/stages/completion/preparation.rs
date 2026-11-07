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
mod tests {
    use axum::http::StatusCode;
    use llm_tokenizer::{MockTokenizer, TiktokenModel, TiktokenTokenizer};

    use super::*;

    /// The mock's base vocabulary is its token count; an added token sits far above it.
    fn mock() -> Arc<dyn Tokenizer> {
        Arc::new(MockTokenizer::new().with_tokens(&[("<|added|>", 50_000)]))
    }

    /// One token prompt, `field` naming it, through the stage's batch form.
    async fn token_prompt(
        tokenizer: &Arc<dyn Tokenizer>,
        ids: &[u32],
        field: &str,
        embedding_rows: impl FnOnce() -> Option<usize> + Copy,
    ) -> Result<CompletionItem, Response> {
        let (lists, fields) = ([ids.to_vec()], [field.to_string()]);
        token_prompts(tokenizer, &lists, &fields, embedding_rows)
            .await
            .map(|mut items| items.remove(0))
    }

    /// The refusal of `ids` under `field`: a 400 of this route's code, its body returned.
    async fn refusal(tokenizer: &Arc<dyn Tokenizer>, ids: &[u32], field: &str) -> String {
        let response = match token_prompt(tokenizer, ids, field, || None).await {
            Ok(_) => panic!("{ids:?} is refused"),
            Err(response) => response,
        };
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body = String::from_utf8_lossy(&body).into_owned();
        assert!(body.contains("invalid_prompt_token"), "{body}");
        body
    }

    /// A token prompt keeps its ids and takes their decode as its text, the
    /// text `echo` prepends and the text-reading policies route on.
    #[tokio::test]
    async fn a_token_prompt_goes_as_given_with_its_decode_as_text() {
        let tokenizer = mock();
        let ids: Vec<u32> = ["Hello", "world"]
            .iter()
            .map(|token| tokenizer.token_to_id(token).expect("mock vocab"))
            .collect();

        let item = token_prompt(&tokenizer, &ids, "prompt", || None)
            .await
            .expect("known ids");

        assert_eq!(item.token_ids, ids);
        assert_eq!(item.text, "Hello world");
    }

    /// A batch of token prompts charges one in-place budget, as a batch of
    /// texts does: its lists decode on the calling thread up to the bound and
    /// on the pool past it.
    #[tokio::test]
    async fn a_token_prompt_batch_decodes_in_place_up_to_the_bound_and_the_rest_on_the_pool() {
        use crate::routers::grpc::testing::thread_recording_tokenizer;

        let (tokenizer, ran_on) = thread_recording_tokenizer();
        let hello = tokenizer.token_to_id("Hello").expect("mock vocab");
        // Sixty-four lists of four thousand ids: each under the bound alone (16,000
        // bytes), the batch forty times over it.
        let lists = vec![vec![hello; 4_000]; 64];
        let fields: Vec<String> = (0..lists.len()).map(|b| format!("prompt[{b}]")).collect();
        let here = std::thread::current().id();

        let items = token_prompts(&tokenizer, &lists, &fields, || None)
            .await
            .expect("known ids");

        assert_eq!(items.len(), 64);
        let threads = std::mem::take(&mut *ran_on.lock().unwrap());
        assert_eq!(threads.len(), 64, "every list of the batch is decoded");
        assert_eq!(
            threads.iter().filter(|t| **t == here).count(),
            1,
            "one list of the batch fits the bound in place; the other sixty-three go to the pool"
        );
    }

    /// A prompt long enough to decode off the worker thread reads the same
    /// as the short one.
    #[tokio::test]
    async fn a_long_token_prompt_decodes_to_the_same_text() {
        let tokenizer = mock();
        let hello = tokenizer.token_to_id("Hello").expect("mock vocab");
        let ids = vec![hello; 1024];

        let item = token_prompt(&tokenizer, &ids, "prompt", || None)
            .await
            .expect("known ids");

        assert_eq!(item.token_ids, ids);
        assert_eq!(item.text, tokenizer.decode(&ids, false).expect("decode"));
    }

    /// An added token above the base vocabulary is known by lookup and rides
    /// the prompt as given.
    #[tokio::test]
    async fn an_added_token_above_the_vocabulary_rides_the_prompt() {
        let tokenizer = mock();
        let hello = tokenizer.token_to_id("Hello").expect("mock vocab");
        let ids = vec![hello, 50_000];

        let item = token_prompt(&tokenizer, &ids, "prompt", || None)
            .await
            .expect("an added token is a token");

        assert_eq!(item.token_ids, ids);
        assert_eq!(item.text, tokenizer.decode(&ids, false).expect("decode"));
    }

    /// An id past the vocabulary is refused before the decode, naming its
    /// position, its value and the bound: the engine never sees it.
    #[tokio::test]
    async fn an_id_past_the_vocabulary_is_refused_before_the_decode() {
        let tokenizer = mock();
        let vocab = tokenizer.vocab_size();

        let body = refusal(&tokenizer, &[1, 424_242], "prompt").await;

        assert!(
            body.contains("prompt[1] = 424242 is not a token of this model"),
            "{body}"
        );
        assert!(body.contains(&format!("holds {vocab} tokens")), "{body}");
    }

    /// One list of a batch is refused under its own name.
    #[tokio::test]
    async fn a_list_of_a_batch_is_refused_under_its_own_name() {
        let body = refusal(&mock(), &[2, 424_242], "prompt[1]").await;

        assert!(body.contains("prompt[1][1] = 424242 "), "{body}");
    }

    /// An id the tokenizer does not know but the workers' reported embedding
    /// table holds passes, as it would on the engine; its decode is empty.
    #[tokio::test]
    async fn an_id_inside_the_reported_embedding_table_passes() {
        let tokenizer = mock();
        let hello = tokenizer.token_to_id("Hello").expect("mock vocab");

        let item = token_prompt(&tokenizer, &[hello, 1_500], "prompt", || Some(2_000))
            .await
            .expect("an id below the table is one the model can emit");

        assert_eq!(item.token_ids, vec![hello, 1_500], "the ids ride as given");
        assert_eq!(item.text, "Hello", "the id without a token leaves no text");
    }

    /// On a tiktoken model with a padded table, an id in the gap rides the prompt and leaves
    /// no text: the decoder, which refuses such an id, never sees it.
    #[tokio::test]
    async fn a_padded_table_id_rides_a_tiktoken_prompt_and_leaves_no_text() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::new(TiktokenModel::Cl100kBase).expect("built-in encoding"));
        let vocab = tokenizer.vocab_size();
        let mut ids = tokenizer
            .encode("Hello world", false)
            .expect("encode")
            .into_token_ids();
        ids.push(vocab as u32 + 500);

        let item = token_prompt(&tokenizer, &ids, "prompt", || Some(vocab + 1_000))
            .await
            .expect("an id inside the reported table is one the model can emit");

        assert_eq!(item.token_ids, ids, "the ids ride as given");
        assert_eq!(item.text, "Hello world", "the gap id leaves no text");
    }

    /// A hole of a tiktoken encoding, an id below its reported size with neither a rank nor
    /// a special token (o200k's 199998), passes the bound by range and is refused by the
    /// decode, as the client's error.
    #[tokio::test]
    async fn a_hole_in_a_tiktoken_encoding_is_refused_by_its_decode() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::new(TiktokenModel::O200kBase).expect("built-in encoding"));
        let known = tokenizer
            .encode("Hello", false)
            .expect("encode")
            .into_token_ids();
        assert!(
            199_998 < tokenizer.vocab_size(),
            "the hole sits inside the reported range"
        );

        let body = refusal(&tokenizer, &[known[0], 199_998], "prompt").await;

        assert!(
            body.contains("prompt token ids cannot be decoded"),
            "{body}"
        );
    }

    /// A real tokenizer bounds the prompt by its own vocabulary.
    #[tokio::test]
    async fn a_real_tokenizer_bounds_the_prompt_by_its_vocabulary() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::new(TiktokenModel::Cl100kBase).expect("built-in encoding"));
        let known = tokenizer
            .encode("Hello", false)
            .expect("encode")
            .into_token_ids();

        let body = refusal(&tokenizer, &[known[0], 4_000_000_000], "prompt").await;

        assert!(
            body.contains("prompt[1] = 4000000000 is not a token of this model"),
            "{body}"
        );
    }
}
