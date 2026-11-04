//! A generate request that carries token ids instead of text: the ids reach the worker as they
//! were sent, the tokenizer encodes nothing, the engine counts them as the prompt, and the
//! answer decodes as it does for a text prompt. The text request on the same path keeps its one
//! encode, and the shapes the path does not take (a batch of id lists, a negative id) are refused
//! before any worker is spoken to.

use std::sync::atomic::{AtomicUsize, Ordering};

use llm_tokenizer::{
    traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
    MockTokenizer, TokenizerRegistry,
};
use openai_protocol::generate::GenerateRequest;
use serde_json::{json, Value};

use super::*;

/// Ids outside the mock vocabulary: nothing on the path can rebuild them from a text, so the
/// worker sees them only if the gateway forwards what it was given.
const IDS: [u32; 7] = [11, 12, 13, 14, 15, 16, 17];

/// The mock tokenizer behind a count of its encode calls, with the stub's answer token (100)
/// in its vocabulary so a decoded answer reads as a word and a skipped decode as nothing.
struct CountingTokenizer {
    inner: MockTokenizer,
    encodes: AtomicUsize,
}

impl CountingTokenizer {
    fn new() -> Self {
        Self {
            inner: MockTokenizer::new().with_tokens(&[("Paris", 100)]),
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
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CountingTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
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

/// The chat+generate pipeline over one stub worker, with the counting tokenizer loaded for the
/// model and the stub's record of what it received.
struct Bed {
    pipeline: RequestPipeline,
    components: Arc<SharedComponents>,
    tokenizer: Arc<CountingTokenizer>,
    seen_ids: Arc<Mutex<Vec<Vec<u32>>>>,
    seen_texts: Arc<Mutex<Vec<String>>>,
}

impl Bed {
    async fn new() -> Self {
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let seen_texts = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            seen_original_texts: Arc::clone(&seen_texts),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, port, WorkerType::Regular);
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline = RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps)
            .expect("chat+generate pipeline");
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

    /// The pipeline's answer to `body`: its status and its bytes.
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
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        (status, body)
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

#[tokio::test]
async fn ids_reach_the_worker_as_sent_without_an_encode() {
    let bed = Bed::new().await;

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
    let bed = Bed::new().await;

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
    let bed = Bed::new().await;

    let (status, body) = bed.generate(ids_request(true)).await;

    assert_eq!(status, http::StatusCode::OK);
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.ends_with("data: [DONE]\n\n"),
        "the stream finishes: {body}"
    );
    let events: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("generate event"))
        .collect();
    assert!(!events.is_empty(), "{body}");
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
        "{body}"
    );
    assert_eq!(bed.seen_ids(), vec![IDS.to_vec()]);
    assert_eq!(bed.tokenizer.encodes(), 0);
}

#[tokio::test]
async fn a_batch_of_id_lists_is_refused_before_any_worker_call() {
    let bed = Bed::new().await;

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
    let bed = Bed::new().await;

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
