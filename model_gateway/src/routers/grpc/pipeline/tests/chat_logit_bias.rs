//! A chat request's `logit_bias` through the pipelines: the keys are judged
//! before the engine sees the map, and the map reaches the engine on the
//! Harmony pipeline too, which resolves no tokenizer of its own for a request.

use super::*;
use crate::routers::grpc::mode::Mode;

/// The pipeline of `endpoint` over one stub worker, the components over
/// `tokenizer_registry`, and the stub that saw the requests.
async fn pipeline_over(
    endpoint: Endpoint,
    tokenizer_registry: Arc<TokenizerRegistry>,
) -> (RequestPipeline, Arc<SharedComponents>, GatedScheduler) {
    let scheduler = GatedScheduler::default();
    let port = spawn_stub(scheduler.clone()).await;
    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker(&worker_registry, port, WorkerType::Regular);
    let deps = PipelineDeps::pair(
        worker_registry.clone(),
        Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline = RequestPipeline::build(endpoint, Mode::Regular, &deps)
        .unwrap_or_else(|| panic!("a {endpoint:?} pipeline"));
    let components = shared_components(tokenizer_registry, worker_registry, false);
    (pipeline, components, scheduler)
}

/// The registry with the mock tokenizer under the stub's model.
async fn mock_tokenizer_registry() -> Arc<TokenizerRegistry> {
    let registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .expect("load mock tokenizer");
    registry
}

/// A chat request for the stub's model that carries `bias`.
fn chat_with_bias(bias: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "logit_bias": bias,
    })
}

/// The bias maps the stub saw, one per request, each sorted by token.
fn bias_seen(scheduler: &GatedScheduler) -> Vec<ReceivedBias> {
    scheduler
        .seen_logit_bias
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// The Harmony pipeline encodes with its own Harmony encoding and resolves
/// no tokenizer for a request, so the keys are judged against the model's
/// registered tokenizer looked up for the purpose; the map then reaches the
/// engine, as it does from the regular chat pipeline.
#[tokio::test]
async fn a_chat_request_s_logit_bias_reaches_the_engine_from_the_harmony_pipeline_too() {
    for endpoint in [Endpoint::Harmony, Endpoint::Chat] {
        let name = format!("{endpoint:?}");
        let (pipeline, components, scheduler) =
            pipeline_over(endpoint, mock_tokenizer_registry().await).await;
        let (status, body) = chat_error_body(
            &pipeline,
            components,
            chat_with_bias(serde_json::json!({"1": -100})),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK, "{name}: {body}");
        assert_eq!(
            bias_seen(&scheduler),
            vec![vec![("1".to_string(), -100.0)]],
            "{name}"
        );
    }
}

/// With no tokenizer registered for the model the Harmony pipeline cannot
/// judge the keys and refuses a request that carries a bias, saying so.
#[tokio::test]
async fn a_harmony_chat_request_s_logit_bias_is_refused_when_the_model_has_no_tokenizer() {
    let (pipeline, components, scheduler) =
        pipeline_over(Endpoint::Harmony, Arc::new(TokenizerRegistry::new())).await;
    let (status, body) = chat_error_body(
        &pipeline,
        components,
        chat_with_bias(serde_json::json!({"1": -100})),
    )
    .await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "logit_bias_invalid_key", "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("no tokenizer is registered"), "{message}");
    assert!(
        bias_seen(&scheduler).is_empty(),
        "the engine saw no request"
    );
}

/// A key the model's tokenizer does not know, with no worker reporting an
/// embedding table, is refused before the engine on both pipelines, naming
/// the key and the bound.
#[tokio::test]
async fn a_chat_request_s_logit_bias_key_past_the_model_is_refused_before_the_engine() {
    let unknown = MockTokenizer::new().vocab_size() as u32 + 7;
    for endpoint in [Endpoint::Harmony, Endpoint::Chat] {
        let name = format!("{endpoint:?}");
        let (pipeline, components, scheduler) =
            pipeline_over(endpoint, mock_tokenizer_registry().await).await;
        let (status, body) = chat_error_body(
            &pipeline,
            components,
            chat_with_bias(serde_json::json!({unknown.to_string(): -100})),
        )
        .await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{name}: {body}");
        assert_eq!(
            body["error"]["code"], "logit_bias_invalid_key",
            "{name}: {body}"
        );
        let message = body["error"]["message"].as_str().expect("message");
        assert!(
            message.contains(&format!("'{unknown}' is not a token of this model")),
            "{name}: {message}"
        );
        assert!(
            bias_seen(&scheduler).is_empty(),
            "{name}: the engine saw no request"
        );
    }
}
