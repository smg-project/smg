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
