//! A prompt that arrives as token ids instead of text, on the two routes that take one: the
//! native `/generate` (`input_ids`) and `/v1/completions` (`prompt` as an array of ids, or of
//! id arrays). The ids reach the worker as they were sent, the tokenizer encodes nothing, the
//! engine counts them as the prompt, and the answer decodes as it does for a text prompt. The
//! text request on the same path keeps its one encode. On `/generate` the shapes the path does
//! not take (a batch of id lists, a negative id) are refused before any worker is spoken to; on
//! `/v1/completions` a batch of id lists is one engine request per prompt, as a batch of texts
//! is, and an `echo` opens the answer with the ids' decode. On both routes an id past the
//! model's vocabulary (or past the embedding table its workers report) is refused before any
//! worker is spoken to; an added token the tokenizer knows by lookup passes.

use std::sync::atomic::{AtomicUsize, Ordering};

use llm_tokenizer::{
    traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
    MockTokenizer, TokenizerRegistry,
};
use openai_protocol::generate::GenerateRequest;
use serde_json::{json, Value};

use super::*;

/// Ids that map to no text in the mock vocabulary but sit inside the vocabulary the mock
/// reports ([`VOCAB`]): nothing on the path can rebuild them from a text, so the worker sees
/// them only if the gateway forwards what it was given.
const IDS: [u32; 7] = [11, 12, 13, 14, 15, 16, 17];

/// The vocabulary size the counting tokenizer reports: wide enough for [`IDS`], below
/// [`ADDED`].
const VOCAB: usize = 1_000;

/// An added token above the reported vocabulary, known to the tokenizer by lookup alone.
const ADDED: u32 = 5_000;

/// The mock vocabulary's "Hello world": ids whose decode an `echo` can show.
const HELLO_WORLD: [u32; 2] = [1, 2];

/// The mock tokenizer behind a count of its encode calls, with the stub's answer token (100)
/// in its vocabulary so a decoded answer reads as a word and a skipped decode as nothing.
struct CountingTokenizer {
    inner: MockTokenizer,
    encodes: AtomicUsize,
}

impl CountingTokenizer {
    fn new() -> Self {
        Self {
            inner: MockTokenizer::new().with_tokens(&[("Paris", 100), ("<|added|>", ADDED)]),
            encodes: AtomicUsize::new(0),
        }
    }

    fn encodes(&self) -> usize {
        self.encodes.load(Ordering::SeqCst)
    }
}

impl Encoder for CountingTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
        self.encodes.fetch_add(1, Ordering::SeqCst);
        self.inner.encode(input, add_special_tokens)
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        self.encodes.fetch_add(inputs.len(), Ordering::SeqCst);
        self.inner.encode_batch(inputs, add_special_tokens)
    }
}

impl Decoder for CountingTokenizer {
    /// As the tiktoken decoder: an id this tokenizer knows neither by its range nor by lookup
    /// refuses the decode, so a table-only id reaching it fails the row instead of vanishing.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let known = |id: u32| {
            usize::try_from(id).is_ok_and(|id| id < VOCAB) || self.inner.id_to_token(id).is_some()
        };
        if let Some(id) = token_ids.iter().find(|&&id| !known(id)) {
            anyhow::bail!("no token for id {id}");
        }
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CountingTokenizer {
    /// The mock counts its few tokens as its vocabulary; this bed reports one wide enough for
    /// the ids it sends, so the gateway's bound admits them while nothing maps them to text.
    fn vocab_size(&self) -> usize {
        VOCAB
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// One route's pipeline over one stub worker, with the counting tokenizer loaded for the
/// model and the stub's record of what it received.
struct Bed {
    pipeline: RequestPipeline,
    components: Arc<SharedComponents>,
    tokenizer: Arc<CountingTokenizer>,
    seen_ids: Arc<Mutex<Vec<Vec<u32>>>>,
    seen_texts: Arc<Mutex<Vec<String>>>,
}

impl Bed {
    /// `Endpoint::Chat` is the chat+generate pipeline; `Endpoint::Completion` the completions one.
    async fn new(endpoint: Endpoint) -> Self {
        Self::with_worker_labels(endpoint, &[]).await
    }

    /// The bed over a worker that carries `labels` (an engine's model-info fields, as
    /// discovery keeps them).
    async fn with_worker_labels(endpoint: Endpoint, labels: &[(&str, &str)]) -> Self {
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let seen_texts = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            seen_original_texts: Arc::clone(&seen_texts),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_with_labels(&worker_registry, port, WorkerType::Regular, labels);
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(endpoint, Mode::Regular, &deps).expect("the route's pipeline");
        let tokenizer = Arc::new(CountingTokenizer::new());
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let loaded = Arc::clone(&tokenizer) as Arc<dyn Tokenizer>;
        tokenizer_registry
            .load("tokenizer-id", MODEL, "test", || async move { Ok(loaded) })
            .await
            .expect("load the counting tokenizer");
        let components = shared_components(tokenizer_registry, worker_registry, false);
        Self {
            pipeline,
            components,
            tokenizer,
            seen_ids,
            seen_texts,
        }
    }

    /// The pipeline's answer to a `/generate` `body`: its status and its bytes.
    async fn generate(&self, body: Value) -> (http::StatusCode, bytes::Bytes) {
        let request: Arc<GenerateRequest> =
            Arc::new(serde_json::from_value(body).expect("generate request"));
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            self.pipeline.execute_generate(
                request,
                None,
                MODEL.to_string(),
                Arc::clone(&self.components),
                None,
                None,
                None,
            ),
        )
        .await
        .expect("the pipeline answers within its budget");
        drain(response).await
    }

    /// The pipeline's answer to a `/v1/completions` `body`: its status and its bytes.
    async fn complete(&self, body: Value) -> (http::StatusCode, bytes::Bytes) {
        let request: Arc<CompletionRequest> =
            Arc::new(serde_json::from_value(body).expect("completion request"));
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            self.pipeline.execute_completion(
                request,
                None,
                MODEL.to_string(),
                Arc::clone(&self.components),
                None,
                None,
                None,
            ),
        )
        .await
        .expect("the pipeline answers within its budget");
        drain(response).await
    }

    fn seen_ids(&self) -> Vec<Vec<u32>> {
        self.seen_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn seen_texts(&self) -> Vec<String> {
        self.seen_texts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

async fn drain(response: Response) -> (http::StatusCode, bytes::Bytes) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");
    (status, body)
}

fn ids_request(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "input_ids": IDS,
        "sampling_params": {"max_new_tokens": 4},
        "stream": stream,
    })
}

fn json_body(body: &bytes::Bytes) -> Value {
    serde_json::from_slice(body)
        .unwrap_or_else(|e| panic!("json body: {e}: {}", String::from_utf8_lossy(body)))
}

/// The `data:` payloads of an SSE body that ends with `[DONE]`.
fn sse_events(body: &bytes::Bytes) -> Vec<Value> {
    let body = String::from_utf8_lossy(body);
    assert!(
        body.ends_with("data: [DONE]\n\n"),
        "the stream finishes: {body}"
    );
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("stream event"))
        .collect()
}

#[tokio::test]
async fn ids_reach_the_worker_as_sent_without_an_encode() {
    let bed = Bed::new(Endpoint::Chat).await;

    let (status, body) = bed.generate(ids_request(false)).await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let body = json_body(&body);
    assert!(body.is_object(), "one prompt answers as one object: {body}");
    assert_eq!(
        bed.seen_ids(),
        vec![IDS.to_vec()],
        "the ids ride the wire as sent"
    );
    assert_eq!(
        bed.seen_texts(),
        vec![String::new()],
        "no prompt text is rebuilt next to the ids"
    );
    assert_eq!(bed.tokenizer.encodes(), 0, "the prompt is never encoded");
    assert_eq!(
        body["meta_info"]["prompt_tokens"],
        json!(IDS.len()),
        "the worker's prompt count is the ids' length: {body}"
    );
    assert_eq!(body["output_ids"], json!([100]), "{body}");
    assert_eq!(body["text"], "Paris", "the answer still decodes: {body}");
}

#[tokio::test]
async fn a_text_request_on_the_same_path_encodes_once() {
    let bed = Bed::new(Endpoint::Chat).await;

    let (status, body) = bed
        .generate(json!({
            "model": MODEL,
            "text": "Hello world",
            "sampling_params": {"max_new_tokens": 4},
            "stream": false,
        }))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let body = json_body(&body);
    assert_eq!(
        bed.tokenizer.encodes(),
        1,
        "a text prompt is encoded exactly once"
    );
    assert_eq!(
        bed.seen_ids(),
        vec![vec![1, 2]],
        "the encoded ids ride the wire"
    );
    assert_eq!(body["meta_info"]["prompt_tokens"], json!(2), "{body}");
    assert_eq!(body["output_ids"], json!([100]), "{body}");
    assert_eq!(body["text"], "Paris", "{body}");
}

#[tokio::test]
async fn a_streamed_ids_request_reports_the_ids_count_on_every_event() {
    let bed = Bed::new(Endpoint::Chat).await;

    let (status, body) = bed.generate(ids_request(true)).await;

    assert_eq!(status, http::StatusCode::OK);
    let events = sse_events(&body);
    assert!(!events.is_empty(), "{}", String::from_utf8_lossy(&body));
    for event in &events {
        assert_eq!(
            event["meta_info"]["prompt_tokens"],
            json!(IDS.len()),
            "{event}"
        );
    }
    // Every event carries the text decoded so far; the last one is the whole answer.
    assert_eq!(
        events.last().map(|event| &event["text"]),
        Some(&json!("Paris")),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(bed.seen_ids(), vec![IDS.to_vec()]);
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn a_batch_of_id_lists_is_refused_before_any_worker_call() {
    let bed = Bed::new(Endpoint::Chat).await;

    let (status, body) = bed
        .generate(json!({
            "model": MODEL,
            "input_ids": [[1, 2], [3]],
            "stream": false,
        }))
        .await;

    let body = json_body(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "resolve_input_failed", "{body}");
    assert!(bed.seen_ids().is_empty(), "no worker is spoken to");
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn a_negative_id_is_refused_before_any_worker_call() {
    let bed = Bed::new(Endpoint::Chat).await;

    let (status, body) = bed
        .generate(json!({
            "model": MODEL,
            "input_ids": [1, -2, 3],
            "stream": false,
        }))
        .await;

    let body = json_body(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "resolve_input_failed", "{body}");
    assert!(bed.seen_ids().is_empty(), "no worker is spoken to");
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn an_id_past_the_vocabulary_is_refused_before_any_worker_call() {
    let bed = Bed::new(Endpoint::Chat).await;
    // Above the reported vocabulary and none of the mock's added tokens (its own sit at
    // 999-1003, the bed's at ADDED), so neither the range nor the lookup knows it.
    let unknown = VOCAB as u32 + 500;

    let (status, body) = bed
        .generate(json!({
            "model": MODEL,
            "input_ids": [1, 2, unknown],
            "stream": false,
        }))
        .await;

    let body = json_body(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "resolve_input_failed", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(&format!("input_ids[2] = {unknown} ")),
        "the refusal names the position and the value: {body}"
    );
    assert!(
        message.contains(&format!("holds {VOCAB} tokens")),
        "the refusal names the vocabulary: {body}"
    );
    assert!(bed.seen_ids().is_empty(), "no worker is spoken to");
    assert_eq!(bed.tokenizer.encodes(), 0);
}

/// The engine's embedding table is padded past the tokenizer's last id and the engine samples
/// over the padded logits, so an id from that gap is one the model itself can emit: when a
/// worker reports its table size, such an id passes as it would on the engine.
#[tokio::test]
async fn an_id_inside_the_reported_embedding_table_reaches_the_worker() {
    let bed = Bed::with_worker_labels(Endpoint::Chat, &[("vocab_size", "2000")]).await;
    let gap = VOCAB as u32 + 500;
    let ids = vec![1, gap, 2];

    let (status, body) = bed
        .generate(json!({
            "model": MODEL,
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 4},
            "stream": false,
        }))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        bed.seen_ids(),
        vec![ids],
        "an id below the table rides the wire as sent"
    );
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn an_id_past_the_reported_embedding_table_is_refused_naming_the_table() {
    let bed = Bed::with_worker_labels(Endpoint::Chat, &[("vocab_size", "2000")]).await;

    let (status, body) = bed
        .generate(json!({
            "model": MODEL,
            "input_ids": [1, 2, 2000],
            "stream": false,
        }))
        .await;

    let body = json_body(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "resolve_input_failed", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("input_ids[2] = 2000 "), "{body}");
    assert!(
        message.contains("embedding table holds 2000 rows"),
        "{body}"
    );
    assert!(bed.seen_ids().is_empty(), "no worker is spoken to");
}

#[tokio::test]
async fn an_added_token_above_the_vocabulary_reaches_the_worker() {
    let bed = Bed::new(Endpoint::Chat).await;
    let ids = vec![11, ADDED, 12];

    let (status, body) = bed
        .generate(json!({
            "model": MODEL,
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 4},
            "stream": false,
        }))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        bed.seen_ids(),
        vec![ids],
        "an added token the tokenizer knows by lookup rides the wire as sent"
    );
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn a_completion_token_prompt_reaches_the_worker_as_sent_without_an_encode() {
    let bed = Bed::new(Endpoint::Completion).await;

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": IDS, "max_tokens": 4}))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let body = json_body(&body);
    assert_eq!(body["object"], "text_completion", "{body}");
    assert_eq!(
        bed.seen_ids(),
        vec![IDS.to_vec()],
        "the ids ride the wire as sent"
    );
    assert_eq!(
        bed.seen_texts(),
        vec![String::new()],
        "no prompt text is rebuilt next to the ids"
    );
    assert_eq!(bed.tokenizer.encodes(), 0, "the prompt is never encoded");
    assert_eq!(
        body["usage"]["prompt_tokens"],
        json!(IDS.len()),
        "the worker's prompt count is the ids' length: {body}"
    );
    assert_eq!(body["choices"].as_array().map(Vec::len), Some(1), "{body}");
    assert_eq!(body["choices"][0]["index"], json!(0), "{body}");
    assert_eq!(
        body["choices"][0]["text"], "Paris",
        "the answer still decodes: {body}"
    );
}

#[tokio::test]
async fn a_completion_text_prompt_on_the_same_path_encodes_once() {
    let bed = Bed::new(Endpoint::Completion).await;

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": "Hello world", "max_tokens": 4}))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let body = json_body(&body);
    assert_eq!(
        bed.tokenizer.encodes(),
        1,
        "a text prompt is encoded exactly once"
    );
    assert_eq!(bed.seen_ids(), vec![HELLO_WORLD.to_vec()]);
    assert_eq!(body["usage"]["prompt_tokens"], json!(2), "{body}");
    assert_eq!(body["choices"][0]["text"], "Paris", "{body}");
}

#[tokio::test]
async fn a_batch_of_completion_token_prompts_is_one_engine_request_per_prompt() {
    let bed = Bed::new(Endpoint::Completion).await;

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": [IDS, HELLO_WORLD], "max_tokens": 4}))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let body = json_body(&body);
    // The batch's requests are dispatched together; the stub records them as they land.
    let mut seen = bed.seen_ids();
    seen.sort();
    let mut sent = vec![IDS.to_vec(), HELLO_WORLD.to_vec()];
    sent.sort();
    assert_eq!(seen, sent, "every prompt's ids ride the wire as sent");
    assert_eq!(
        bed.tokenizer.encodes(),
        0,
        "no prompt of the batch is encoded"
    );
    let choices = body["choices"].as_array().expect("choices");
    assert_eq!(
        choices
            .iter()
            .map(|choice| choice["index"].as_u64())
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)],
        "one choice per prompt, in the prompts' order: {body}"
    );
    assert!(
        choices.iter().all(|c| c["text"] == "Paris"),
        "every prompt is answered: {body}"
    );
    assert_eq!(
        body["usage"]["prompt_tokens"],
        json!(IDS.len() + HELLO_WORLD.len()),
        "the usage counts every prompt's ids: {body}"
    );
}

#[tokio::test]
async fn an_echoed_completion_token_prompt_opens_with_its_decode() {
    let bed = Bed::new(Endpoint::Completion).await;

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": HELLO_WORLD, "max_tokens": 4, "echo": true}))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let body = json_body(&body);
    assert_eq!(
        body["choices"][0]["text"], "Hello worldParis",
        "the echo is the ids' decode, then the answer: {body}"
    );
    assert_eq!(bed.seen_ids(), vec![HELLO_WORLD.to_vec()]);
    assert_eq!(
        bed.tokenizer.encodes(),
        0,
        "an echo decodes, it never encodes"
    );
    assert_eq!(body["usage"]["prompt_tokens"], json!(2), "{body}");
}

#[tokio::test]
async fn a_streamed_completion_token_prompt_reports_the_ids_count_in_its_usage() {
    let bed = Bed::new(Endpoint::Completion).await;

    let (status, body) = bed
        .complete(json!({
            "model": MODEL,
            "prompt": IDS,
            "max_tokens": 4,
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .await;

    assert_eq!(status, http::StatusCode::OK);
    let events = sse_events(&body);
    assert!(!events.is_empty(), "{}", String::from_utf8_lossy(&body));
    let text: String = events
        .iter()
        .flat_map(|event| event["choices"].as_array().into_iter().flatten())
        .filter_map(|choice| choice["text"].as_str())
        .collect();
    assert_eq!(text, "Paris", "{}", String::from_utf8_lossy(&body));
    let usage = events
        .iter()
        .filter_map(|event| event.get("usage"))
        .find(|usage| !usage.is_null())
        .expect("the stream states its usage");
    assert_eq!(usage["prompt_tokens"], json!(IDS.len()), "{usage}");
    assert_eq!(bed.seen_ids(), vec![IDS.to_vec()]);
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn a_completion_id_past_the_vocabulary_is_refused_before_any_worker_call() {
    let bed = Bed::new(Endpoint::Completion).await;
    // Above the reported vocabulary and none of the mock's added tokens, as on `/generate`.
    let unknown = VOCAB as u32 + 500;

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": [1, 2, unknown], "max_tokens": 4}))
        .await;

    let body = json_body(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_prompt_token", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(&format!("prompt[2] = {unknown} ")),
        "the refusal names the position and the value: {body}"
    );
    assert!(
        message.contains(&format!("holds {VOCAB} tokens")),
        "the refusal names the vocabulary: {body}"
    );
    assert!(bed.seen_ids().is_empty(), "no worker is spoken to");
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn a_completion_batch_with_an_id_past_the_vocabulary_is_refused_naming_its_list() {
    let bed = Bed::new(Endpoint::Completion).await;
    let unknown = VOCAB as u32 + 500;

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": [[1, 2], [3, unknown]], "max_tokens": 4}))
        .await;

    let body = json_body(&body);
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_prompt_token", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(&format!("prompt[1][1] = {unknown} ")),
        "the refusal names the list and the position: {body}"
    );
    assert!(bed.seen_ids().is_empty(), "no worker is spoken to");
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn an_added_token_above_the_vocabulary_reaches_the_worker_on_the_completion_route() {
    let bed = Bed::new(Endpoint::Completion).await;
    let ids = vec![11, ADDED, 12];

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": ids, "max_tokens": 4}))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        bed.seen_ids(),
        vec![ids],
        "an added token the tokenizer knows by lookup rides the wire as sent"
    );
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn a_completion_id_inside_the_reported_embedding_table_reaches_the_worker() {
    let bed = Bed::with_worker_labels(Endpoint::Completion, &[("vocab_size", "2000")]).await;
    let gap = VOCAB as u32 + 500;
    let ids = vec![1, gap, 2];

    let (status, body) = bed
        .complete(json!({"model": MODEL, "prompt": ids, "max_tokens": 4}))
        .await;

    assert_eq!(
        status,
        http::StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        bed.seen_ids(),
        vec![ids],
        "an id below the reported table rides the wire as sent"
    );
    assert_eq!(bed.tokenizer.encodes(), 0);
}
