//! An embeddings (or classify) request whose `input` is a batch: one engine
//! call per input, in the request's order, answered as one list with each
//! vector at its input's index and the usage summed. An input given as token
//! ids reaches the engine as given, with no encode and no text. The shapes
//! the contract has no form for, an empty input or member, and an id past
//! the vocabulary are refused by name before any worker is spoken to. The
//! engine is a scripted stub of the vLLM wire: it records the ids and the
//! text of every `Embed` call and answers a vector that names the call
//! (`[ids.len(), ids[0]]`), its `prompt_tokens` the ids' count.

use std::sync::{Arc, Mutex, PoisonError};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{classify::ClassifyRequest, embedding::EmbeddingRequest};
use serde_json::{json, Value};
use vllm_proto::vllm_engine_server::{VllmEngine, VllmEngineServer};

use super::*;

/// The engine's answer to one call: the ids' count and the first id, as two
/// dimensions a test reads the call back from.
fn vector_of(ids: &[u32]) -> Vec<f32> {
    vec![ids.len() as f32, ids.first().copied().unwrap_or(0) as f32]
}

/// The vLLM-wire stub: `Embed` recorded and answered per call, the rest of
/// the wire unused.
#[derive(Clone, Default)]
struct EmbedStub {
    seen_ids: Arc<Mutex<Vec<Vec<u32>>>>,
    seen_texts: Arc<Mutex<Vec<String>>>,
    /// The call whose ids are this list is failed with UNAVAILABLE.
    fail_ids: Option<Vec<u32>>,
    /// The call whose ids are this list answers late (a sibling still in
    /// flight when another call fails).
    slow_ids: Option<Vec<u32>>,
    /// The ids of every call that ran to its answer, in the order they ended.
    finished: Arc<Mutex<Vec<Vec<u32>>>>,
}

#[tonic::async_trait]
impl VllmEngine for EmbedStub {
    type GenerateStream = ReceiverStream<Result<vllm_proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: TonicRequest<vllm_proto::GenerateRequest>,
    ) -> Result<TonicResponse<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    async fn embed(
        &self,
        request: TonicRequest<vllm_proto::EmbedRequest>,
    ) -> Result<TonicResponse<vllm_proto::EmbedResponse>, Status> {
        let tokenized = request.into_inner().tokenized.unwrap_or_default();
        self.seen_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(tokenized.input_ids.clone());
        self.seen_texts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(tokenized.original_text);
        if self.slow_ids.as_deref() == Some(tokenized.input_ids.as_slice()) {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        self.finished
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(tokenized.input_ids.clone());
        if self.fail_ids.as_deref() == Some(tokenized.input_ids.as_slice()) {
            return Err(Status::unavailable("the stub fails this input"));
        }
        Ok(TonicResponse::new(vllm_proto::EmbedResponse {
            embedding: vector_of(&tokenized.input_ids),
            prompt_tokens: tokenized.input_ids.len() as u32,
            embedding_dim: 2,
        }))
    }

    async fn health_check(
        &self,
        _request: TonicRequest<vllm_proto::HealthCheckRequest>,
    ) -> Result<TonicResponse<vllm_proto::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    async fn flush_cache(
        &self,
        _request: TonicRequest<common::FlushCacheRequest>,
    ) -> Result<TonicResponse<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    async fn start_profile(
        &self,
        _request: TonicRequest<common::StartProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    async fn stop_profile(
        &self,
        _request: TonicRequest<common::StopProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    async fn abort(
        &self,
        _request: TonicRequest<vllm_proto::AbortRequest>,
    ) -> Result<TonicResponse<vllm_proto::AbortResponse>, Status> {
        Ok(TonicResponse::new(vllm_proto::AbortResponse::default()))
    }

    async fn get_model_info(
        &self,
        _request: TonicRequest<vllm_proto::GetModelInfoRequest>,
    ) -> Result<TonicResponse<vllm_proto::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    async fn get_server_info(
        &self,
        _request: TonicRequest<vllm_proto::GetServerInfoRequest>,
    ) -> Result<TonicResponse<vllm_proto::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    async fn get_loads(
        &self,
        _request: TonicRequest<vllm_proto::GetLoadsRequest>,
    ) -> Result<TonicResponse<vllm_proto::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    type GetTokenizerStream = ReceiverStream<Result<common::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: TonicRequest<common::GetTokenizerRequest>,
    ) -> Result<TonicResponse<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }

    type SubscribeKvEventsStream = ReceiverStream<Result<common::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: TonicRequest<common::SubscribeKvEventsRequest>,
    ) -> Result<TonicResponse<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("an embedding stub"))
    }
}

/// One route's pipeline over one stub worker of the vLLM wire, the mock
/// tokenizer loaded for the model and the stub's record of the calls.
struct Bed {
    pipeline: RequestPipeline,
    components: Arc<SharedComponents>,
    tokenizer: Arc<dyn Tokenizer>,
    stub: EmbedStub,
}

impl Bed {
    async fn new(endpoint: Endpoint) -> Self {
        Self::over(endpoint, EmbedStub::default()).await
    }

    async fn over(endpoint: Endpoint, stub: EmbedStub) -> Self {
        let router = Server::builder().add_service(VllmEngineServer::new(stub.clone()));
        let port = serve_on_loopback(router).await.0.port();
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_of_runtime(
            &worker_registry,
            port,
            WorkerType::Regular,
            RuntimeType::Vllm,
            &[],
        );
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(endpoint, Mode::Regular, &deps).expect("the route's pipeline");
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let loaded = Arc::clone(&tokenizer);
        tokenizer_registry
            .load("tokenizer-id", MODEL, "test", || async move { Ok(loaded) })
            .await
            .expect("load the mock tokenizer");
        let components = shared_components(tokenizer_registry, worker_registry, false);
        Self {
            pipeline,
            components,
            tokenizer,
            stub,
        }
    }

    /// The pipeline's answer to a `/v1/embeddings` `body`: its status and its JSON.
    async fn embed(&self, body: Value) -> (http::StatusCode, Value) {
        let request: Arc<EmbeddingRequest> =
            Arc::new(serde_json::from_value(body).expect("embedding request"));
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            self.pipeline.execute_embeddings(
                request,
                None,
                MODEL.to_string(),
                Arc::clone(&self.components),
                None,
                None,
            ),
        )
        .await
        .expect("the pipeline answers within its budget");
        json_of(response).await
    }

    /// The pipeline's answer to a `/v1/classify` `body`: its status and its JSON.
    async fn classify(&self, body: Value) -> (http::StatusCode, Value) {
        let request: Arc<ClassifyRequest> =
            Arc::new(serde_json::from_value(body).expect("classify request"));
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            self.pipeline.execute_classify(
                request,
                None,
                MODEL.to_string(),
                Arc::clone(&self.components),
                None,
                None,
            ),
        )
        .await
        .expect("the pipeline answers within its budget");
        json_of(response).await
    }

    /// The ids the gateway's tokenizer gives `text`, as the stage encodes an
    /// input (with the model's special tokens).
    fn ids_of(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, true)
            .expect("the mock encodes")
            .token_ids()
            .to_vec()
    }

    fn seen_ids(&self) -> Vec<Vec<u32>> {
        self.stub
            .seen_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The ids of the calls that ran to their answer, in the order they ended.
    fn finished_ids(&self) -> Vec<Vec<u32>> {
        self.stub
            .finished
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn seen_texts(&self) -> Vec<String> {
        self.stub
            .seen_texts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

async fn json_of(response: Response) -> (http::StatusCode, Value) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let json = serde_json::from_slice(&body)
        .unwrap_or_else(|e| panic!("a JSON body ({e}): {}", String::from_utf8_lossy(&body)));
    (status, json)
}

/// The `(index, embedding)` of every item of an embeddings answer, in order.
fn items(body: &Value) -> Vec<(u64, Vec<f32>)> {
    body["data"]
        .as_array()
        .unwrap_or_else(|| panic!("a data list: {body}"))
        .iter()
        .map(|item| {
            let embedding = item["embedding"]
                .as_array()
                .expect("a vector")
                .iter()
                .map(|v| v.as_f64().expect("a number") as f32)
                .collect();
            (item["index"].as_u64().expect("an index"), embedding)
        })
        .collect()
}

#[tokio::test]
async fn a_batch_of_texts_is_one_call_per_text_answered_in_order() {
    let bed = Bed::new(Endpoint::Embeddings).await;
    let texts = ["Hello", "world", "test"];
    let expected_ids: Vec<Vec<u32>> = texts.iter().map(|text| bed.ids_of(text)).collect();

    let (status, body) = bed.embed(json!({"model": MODEL, "input": texts})).await;

    assert_eq!(status, http::StatusCode::OK, "{body}");
    assert_eq!(body["object"], "list", "{body}");
    assert_eq!(
        items(&body),
        expected_ids
            .iter()
            .enumerate()
            .map(|(i, ids)| (i as u64, vector_of(ids)))
            .collect::<Vec<_>>(),
        "each input's vector at its index: {body}"
    );
    let total: usize = expected_ids.iter().map(Vec::len).sum();
    assert_eq!(body["usage"]["prompt_tokens"], json!(total), "{body}");
    assert_eq!(body["usage"]["total_tokens"], json!(total), "{body}");
    assert_eq!(bed.seen_ids(), expected_ids, "one call per input, in order");
    assert_eq!(bed.seen_texts(), texts, "each call carries its own text");
}

#[tokio::test]
async fn a_single_text_is_one_vector_at_index_zero() {
    let bed = Bed::new(Endpoint::Embeddings).await;
    let expected = bed.ids_of("Hello world");

    let (status, body) = bed
        .embed(json!({"model": MODEL, "input": "Hello world"}))
        .await;

    assert_eq!(status, http::StatusCode::OK, "{body}");
    assert_eq!(items(&body), vec![(0, vector_of(&expected))], "{body}");
    assert_eq!(
        body["usage"]["prompt_tokens"],
        json!(expected.len()),
        "{body}"
    );
    assert_eq!(bed.seen_ids(), vec![expected]);
}

#[tokio::test]
async fn token_id_inputs_reach_the_engine_as_given() {
    let bed = Bed::new(Endpoint::Embeddings).await;

    let (status, body) = bed
        .embed(json!({"model": MODEL, "input": [[1, 2], [3]]}))
        .await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    assert_eq!(
        items(&body),
        vec![(0, vector_of(&[1, 2])), (1, vector_of(&[3]))],
        "{body}"
    );
    assert_eq!(body["usage"]["prompt_tokens"], json!(3), "{body}");
    assert_eq!(bed.seen_ids(), vec![vec![1, 2], vec![3]], "the ids as sent");
    assert_eq!(bed.seen_texts(), ["", ""], "no text beside the ids");

    let (status, body) = bed.embed(json!({"model": MODEL, "input": [4, 2, 1]})).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    assert_eq!(items(&body), vec![(0, vector_of(&[4, 2, 1]))], "{body}");
    assert_eq!(bed.seen_ids().last(), Some(&vec![4, 2, 1]));
}

#[tokio::test]
async fn the_shapes_the_contract_has_no_form_for_are_refused_before_any_worker_call() {
    let bed = Bed::new(Endpoint::Embeddings).await;
    let unknown = 5_000;
    let cases = [
        (json!({"text": "Hello"}), "invalid_input", "input must be"),
        (json!(["Hello", 1]), "invalid_input", "input must be"),
        (json!(""), "empty_input", "input cannot be empty"),
        (json!([]), "empty_input", "input cannot be empty"),
        (
            json!(["Hello", ""]),
            "empty_input",
            "input[1] cannot be empty",
        ),
        (
            json!([[1, 2], []]),
            "empty_input",
            "input[1] cannot be empty",
        ),
        (
            json!([1, unknown]),
            "invalid_input_token",
            "input[1] = 5000 ",
        ),
        (
            json!([[1], [2, unknown]]),
            "invalid_input_token",
            "input[1][1] = 5000 ",
        ),
    ];
    for (input, code, message) in cases {
        let (status, body) = bed.embed(json!({"model": MODEL, "input": input})).await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{input}: {body}");
        assert_eq!(body["error"]["code"], code, "{input}: {body}");
        let got = body["error"]["message"].as_str().unwrap_or_default();
        assert!(got.contains(message), "{input}: {body}");
    }
    assert!(bed.seen_ids().is_empty(), "no worker is spoken to");
}

#[tokio::test]
async fn a_classify_batch_is_one_result_per_input() {
    let bed = Bed::new(Endpoint::Classify).await;
    let texts = ["Hello", "world"];
    let expected_ids: Vec<Vec<u32>> = texts.iter().map(|text| bed.ids_of(text)).collect();

    let (status, body) = bed.classify(json!({"model": MODEL, "input": texts})).await;

    assert_eq!(status, http::StatusCode::OK, "{body}");
    let data = body["data"].as_array().expect("a data list");
    assert_eq!(data.len(), 2, "{body}");
    for (i, item) in data.iter().enumerate() {
        assert_eq!(item["index"], json!(i), "{body}");
        assert_eq!(item["num_classes"], json!(2), "{body}");
        assert!(item["label"].is_string(), "{body}");
    }
    let total: usize = expected_ids.iter().map(Vec::len).sum();
    assert_eq!(body["usage"]["prompt_tokens"], json!(total), "{body}");
    assert_eq!(bed.seen_ids(), expected_ids, "one call per input, in order");
}

#[tokio::test]
async fn a_failed_call_fails_the_whole_request() {
    let stub = EmbedStub {
        fail_ids: Some(vec![3]),
        slow_ids: Some(vec![4]),
        ..EmbedStub::default()
    };
    let bed = Bed::over(Endpoint::Embeddings, stub).await;

    let (status, body) = bed
        .embed(json!({"model": MODEL, "input": [[1, 2], [3], [4]]}))
        .await;

    assert!(status.is_server_error(), "{status}: {body}");
    assert_eq!(body["error"]["code"], "start_embedding_failed", "{body}");
    assert!(body.get("data").is_none(), "no partial list: {body}");
    assert!(
        bed.seen_ids().contains(&vec![3]),
        "the failing input was called: {:?}",
        bed.seen_ids()
    );
    // The late sibling ran to its answer before the request failed: a failure
    // never drops a call the worker already admitted.
    assert!(
        bed.finished_ids().contains(&vec![4]),
        "the slow sibling ran to its end: {:?}",
        bed.finished_ids()
    );
}
