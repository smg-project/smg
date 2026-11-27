use std::{any::Any, sync::Arc};

use openai_protocol::{
    chat::ChatCompletionRequest,
    generate::GenerateRequest,
    worker::{ConnectionMode, WorkerStatus},
};
use serde_json::{json, Value};
use smg_worker::{
    connection::engine::EngineConnection,
    health::{circuit_breaker::CircuitState, resilience::ResolvedResilience},
    identity::{
        basic::{BasicWorker, WorkerMetadata, WorkerType},
        error::WorkerResult,
    },
};

use super::*;
use crate::routers::{grpc::backend_client::BasicWorkerBuilder, http::set_request_model};

fn worker() -> BasicWorker {
    BasicWorkerBuilder::new("http://worker:8080")
        .worker_type(WorkerType::Regular)
        .build()
}

fn dp_worker() -> BasicWorker {
    BasicWorkerBuilder::new("http://worker:8080")
        .worker_type(WorkerType::Regular)
        .dp_config(3, 8)
        .build()
}

/// The `Value` pipeline, verbatim: the produced bytes are the wire
/// contract the fast path must reproduce.
fn value_path_bytes<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
) -> Vec<u8> {
    let mut json_val = request_to_value(typed_req, None).unwrap();
    if let Some(canonical_model) = canonical_model {
        set_request_model(&mut json_val, canonical_model);
    }
    let mut json_val = worker.prepare_request(json_val).unwrap();
    strip_default_sglang_fields(&mut json_val);
    serde_json::to_vec(&json_val).unwrap()
}

fn generate_request(mut extra: Value) -> GenerateRequest {
    let mut body = json!({
        "model": "alias-model",
        "input_ids": [101, 7592, 2088, 1010, 2129, 2024, 2017, 2651, 1029,
                      102, 2003, 2023, 1037, 2200, 2146, 3793, 6251, 102],
        "sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 32},
        "stream": false,
        "rid": "req-1"
    });
    body.as_object_mut()
        .unwrap()
        .append(extra.as_object_mut().unwrap());
    serde_json::from_value(body).unwrap()
}

#[test]
fn plain_generate_body_is_byte_identical_with_value_path() {
    let worker = worker();
    assert!(!worker.mutates_request());
    let req = generate_request(json!({}));

    let body = serialize_request_body(&req, None, &worker, None).unwrap();

    assert_eq!(body, value_path_bytes(&req, None, &worker));
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert!(parsed.get("return_hidden_states").is_none());
}

#[test]
fn aliased_model_is_rewritten_to_canonical() {
    let worker = worker();
    let req = generate_request(json!({}));

    let body = serialize_request_body(&req, Some("canonical-model"), &worker, None).unwrap();

    assert_eq!(
        body,
        value_path_bytes(&req, Some("canonical-model"), &worker)
    );
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["model"], "canonical-model");
}

#[test]
fn dp_aware_worker_still_gets_prepare_request() {
    let worker = dp_worker();
    assert!(worker.mutates_request());
    let req = generate_request(json!({}));

    let body = serialize_request_body(&req, None, &worker, None).unwrap();

    assert_eq!(body, value_path_bytes(&req, None, &worker));
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["data_parallel_rank"], 3);
}

#[test]
fn explicit_sglang_defaults_are_stripped() {
    let worker = worker();
    let req = generate_request(json!({
        "ignore_eos": false,
        "top_k": null,
        "separate_reasoning": true,
        "no_stop_trim": true,
        "priority": 5,
        "min_p": 0.0
    }));

    let body = serialize_request_body(&req, None, &worker, None).unwrap();

    assert_eq!(body, value_path_bytes(&req, None, &worker));
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert!(parsed.get("ignore_eos").is_none());
    assert!(parsed.get("top_k").is_none());
    assert!(parsed.get("separate_reasoning").is_none());
    assert_eq!(parsed["no_stop_trim"], true);
    assert_eq!(parsed["priority"], 5);
    assert_eq!(parsed["min_p"], 0.0);
}

#[test]
fn default_generate_body_reuses_the_direct_serialization() {
    let worker = worker();
    // No default-noise field survives serialization, so the strip is a
    // no-op and the first serialization goes out as-is.
    let req = generate_request(json!({}));

    let body = serialize_request_body(&req, None, &worker, None).unwrap();

    assert_eq!(body, value_path_bytes(&req, None, &worker));
    assert_eq!(body, serialize_json_sized(&req, None).unwrap());
}

#[test]
fn untouched_body_reuses_the_direct_serialization() {
    let worker = worker();
    // `return_hidden_states: true` survives the strip, so nothing in this
    // body needs editing.
    let req = generate_request(json!({"return_hidden_states": true}));

    let body = serialize_request_body(&req, None, &worker, None).unwrap();

    assert_eq!(body, value_path_bytes(&req, None, &worker));
    assert_eq!(body, serialize_json_sized(&req, None).unwrap());
}

#[cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
#[test]
fn a_body_no_edit_applies_to_allocates_only_its_bytes() {
    use crate::routers::grpc::testing::bytes_allocated_during;

    let worker = worker();
    let req = generate_request(json!({}));

    let direct = bytes_allocated_during(|| {
        serialize_json_sized(&req, None).unwrap();
    });
    let routed = bytes_allocated_during(|| {
        serialize_request_body(&req, None, &worker, None).unwrap();
    });

    assert_eq!(routed, direct);
}

#[test]
fn f32_fields_are_forwarded_as_the_client_wrote_them() {
    let generate = generate_request(json!({}));
    let chat: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "alias-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "top_p": 0.95,
        "min_p": 0.05
    }))
    .unwrap();

    // `f32` fields must not widen (0.95f32 is 0.949999988079071 as f64)
    // on the direct path or on the Value path `prepare_request` uses.
    for worker in [worker(), dp_worker()] {
        let body = serialize_request_body(&generate, None, &worker, None).unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["sampling_params"]["temperature"], json!(0.7));
        assert_eq!(body["sampling_params"]["top_p"], json!(0.9));

        let body = serialize_request_body(&chat, None, &worker, None).unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["top_p"], json!(0.95));
        assert_eq!(body["min_p"], json!(0.05));
    }
}

#[test]
fn chat_completion_body_is_byte_identical_with_value_path() {
    let worker = worker();
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "alias-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7
    }))
    .unwrap();

    let plain = serialize_request_body(&req, None, &worker, None).unwrap();
    assert_eq!(plain, value_path_bytes(&req, None, &worker));
    // Default-noise flags are omitted at serialization, so the strip is a
    // no-op and the first serialization goes out as-is.
    assert_eq!(plain, serialize_json_sized(&req, None).unwrap());
    let parsed: Value = serde_json::from_slice(&plain).unwrap();
    assert!(parsed.get("separate_reasoning").is_none());
    assert!(parsed.get("skip_special_tokens").is_none());

    let aliased = serialize_request_body(&req, Some("canonical-model"), &worker, None).unwrap();
    assert_eq!(
        aliased,
        value_path_bytes(&req, Some("canonical-model"), &worker)
    );
}

/// `separate_reasoning`, `stream_reasoning` and `skip_special_tokens`
/// default to `true` on the engine, so an explicit `false` is the value
/// the client means; it must reach the worker on both body paths.
#[test]
fn explicit_false_on_true_default_fields_reaches_the_worker() {
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "alias-model",
        "messages": [{"role": "user", "content": "hello"}],
        "separate_reasoning": false,
        "stream_reasoning": false,
        "skip_special_tokens": false
    }))
    .unwrap();

    for worker in [worker(), dp_worker()] {
        let body = serialize_request_body(&req, None, &worker, None).unwrap();
        assert_eq!(body, value_path_bytes(&req, None, &worker));
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["separate_reasoning"], false);
        assert_eq!(parsed["stream_reasoning"], false);
        assert_eq!(parsed["skip_special_tokens"], false);
    }
}

#[test]
fn chat_completion_body_preserves_unknown_content_parts() {
    let content = json!([
        {"type": "text", "text": "Describe this attachment"},
        {"type": "vendor_special", "payload": {"items": [1, null, true]}, "option": "keep"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    ]);
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "alias-model",
        "messages": [{"role": "user", "content": content}]
    }))
    .unwrap();

    // Exercise both direct serialization and the worker's Value-based
    // prepare_request path, with and without a model rewrite.
    for worker in [worker(), dp_worker()] {
        for canonical_model in [None, Some("canonical-model")] {
            let body = serialize_request_body(&req, canonical_model, &worker, None).unwrap();
            let parsed: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(parsed["messages"][0]["content"], content);
            assert_eq!(parsed["model"], canonical_model.unwrap_or("alias-model"));
        }
    }
}

#[test]
fn default_completion_body_reuses_the_direct_serialization() {
    let worker = worker();
    let req: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
        "model": "alias-model",
        "prompt": "hello",
        "temperature": 0.7
    }))
    .unwrap();

    let body = serialize_request_body(&req, None, &worker, None).unwrap();

    assert_eq!(body, value_path_bytes(&req, None, &worker));
    assert_eq!(body, serialize_json_sized(&req, None).unwrap());
}

/// A prompt of token ids rides the completion body to the engine as
/// sent, in both of its shapes.
#[test]
fn a_token_prompt_rides_the_completion_body_as_sent() {
    let worker = worker();
    for prompt in [json!([11, 12, 13]), json!([[11, 12], [13]])] {
        let req: openai_protocol::completion::CompletionRequest = serde_json::from_value(json!({
            "model": "alias-model",
            "prompt": prompt,
            "max_tokens": 4
        }))
        .unwrap();

        let body = serialize_request_body(&req, None, &worker, None).unwrap();

        assert_eq!(body, value_path_bytes(&req, None, &worker));
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["prompt"], prompt);
    }
}

#[test]
fn non_object_body_falls_back_to_the_value_pipeline() {
    let worker = worker();
    let req = vec![1, 2, 3];

    // The raw editor rejects the shape, so this exercises the fallback.
    let direct = serialize_json_sized(&req, None).unwrap();
    assert!(serde_json::from_slice::<RawBody>(&direct).is_err());

    let body = serialize_request_body(&req, Some("canonical-model"), &worker, None).unwrap();

    assert_eq!(
        body,
        value_path_bytes(&req, Some("canonical-model"), &worker)
    );
    assert_eq!(body, b"[1,2,3]");
}

#[test]
fn presized_body_capacity_stays_within_slack_of_len() {
    let worker = worker();
    let req = generate_request(json!({ "text": "x".repeat(32 << 20) }));
    let raw_len = serde_json::to_vec(&req).unwrap().len();

    let body = serialize_request_body(&req, None, &worker, Some(raw_len)).unwrap();

    assert!(body.len() > 32 << 20);
    assert!(
        body.capacity() <= body.len() + body.len() / 8 + 1024,
        "capacity {} must stay within slack of len {}",
        body.capacity(),
        body.len()
    );
}

#[test]
fn dp_aware_worker_edits_keep_the_value_pipeline_field_order() {
    // The rank is appended after the model rewrite and the strip
    // swap-removes after it, the sequence the Value pipeline runs.
    let worker = dp_worker();
    let req = generate_request(json!({
        "ignore_eos": false,
        "top_k": null,
        "return_hidden_states": true
    }));

    let body = serialize_request_body(&req, Some("canonical-model"), &worker, None).unwrap();

    assert_eq!(
        body,
        value_path_bytes(&req, Some("canonical-model"), &worker)
    );
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["data_parallel_rank"], 3);
    assert_eq!(parsed["model"], "canonical-model");
    assert!(parsed.get("ignore_eos").is_none());
    assert_eq!(parsed["return_hidden_states"], true);
}

#[test]
fn canonical_model_already_in_the_body_reuses_the_direct_serialization() {
    let worker = worker();
    let req = generate_request(json!({}));

    let body = serialize_request_body(&req, Some("alias-model"), &worker, None).unwrap();

    assert_eq!(body, value_path_bytes(&req, Some("alias-model"), &worker));
    assert_eq!(body, serialize_json_sized(&req, None).unwrap());
}

#[test]
fn raw_edits_mirror_a_preserve_order_map() {
    let bytes = br#"{"a":1,"b":2,"c":3,"d":4}"#;
    let mut raw = RawBody::parse(bytes).unwrap();
    assert!(!raw.mutated());
    let mut oracle: Value = serde_json::from_slice(bytes).unwrap();
    let map = oracle.as_object_mut().unwrap();

    raw.insert("b", to_raw_value(&20).unwrap());
    map.insert("b".to_owned(), json!(20));
    raw.insert("e", to_raw_value(&5).unwrap());
    map.insert("e".to_owned(), json!(5));
    assert!(raw.remove("a"));
    map.remove("a");
    assert!(!raw.remove("zz"));

    assert_eq!(
        raw.to_bytes(None).unwrap(),
        serde_json::to_vec(&oracle).unwrap()
    );
    assert_eq!(
        String::from_utf8(raw.to_bytes(None).unwrap()).unwrap(),
        r#"{"e":5,"b":20,"c":3,"d":4}"#,
        "insert replaces in place or appends; remove swap-removes"
    );
    assert!(raw.mutated());
    assert_eq!(raw.get("c").map(RawValue::get), Some("3"));
    assert!(raw.contains("d"));
    assert!(!raw.contains("a"));
    assert!(RawBody::parse(b"[1,2]").is_err());
}

/// A name the parser had to unescape is owned and a plain one stays
/// borrowed from the bytes; the edits read both alike.
#[test]
fn an_unescaped_name_is_owned_and_a_plain_one_stays_borrowed() {
    let bytes = br#"{"a\"b":1,"c":2}"#;
    let mut raw = RawBody::parse(bytes).unwrap();
    assert!(matches!(raw.fields[0].0, Cow::Owned(_)));
    assert!(matches!(raw.fields[1].0, Cow::Borrowed(_)));
    assert_eq!(raw.get("a\"b").map(RawValue::get), Some("1"));

    let mut oracle: Value = serde_json::from_slice(bytes).unwrap();
    raw.insert("d", to_raw_value(&3).unwrap());
    oracle
        .as_object_mut()
        .unwrap()
        .insert("d".to_owned(), json!(3));
    assert!(raw.remove("a\"b"));
    oracle.as_object_mut().unwrap().remove("a\"b");
    assert_eq!(
        raw.to_bytes(None).unwrap(),
        serde_json::to_vec(&oracle).unwrap()
    );
}

/// The prefill side always sets `stream: false`; a non-streaming request
/// already carries it, and marking the body mutated for that re-serialized
/// a body the parsed bytes could have served as they were.
#[test]
fn insert_of_an_unchanged_value_does_not_mark_the_body() {
    let mut raw = RawBody::parse(br#"{"stream":false}"#).unwrap();
    raw.insert("stream", to_raw_value(&false).unwrap());
    assert!(!raw.mutated());
    raw.insert("stream", to_raw_value(&true).unwrap());
    assert!(raw.mutated());
}

#[test]
fn set_model_only_marks_a_body_whose_model_changes() {
    let bytes = br#"{"model":"m","stream":false}"#;
    let mut raw = RawBody::parse(bytes).unwrap();
    raw.set_model(to_raw_value("m").unwrap());
    assert!(!raw.mutated());
    raw.set_model(to_raw_value("canonical").unwrap());
    assert!(raw.mutated());
    assert_eq!(raw.get("model").map(RawValue::get), Some(r#""canonical""#));

    let mut without_model = RawBody::parse(br#"{"stream":false}"#).unwrap();
    without_model.set_model(to_raw_value("canonical").unwrap());
    assert!(
        !without_model.mutated(),
        "no model field, nothing to replace"
    );
}

/// A worker with its own `prepare_request` on top of a DP rank. It keeps
/// the trait's default `uses_builtin_prepare_request` (`false`), so the
/// raw DP-rank shortcut must not apply or its edit would be lost.
#[derive(Debug)]
struct CustomPrepare(BasicWorker);

#[async_trait::async_trait]
impl Worker for CustomPrepare {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn url(&self) -> &str {
        self.0.url()
    }
    fn api_key(&self) -> Option<&String> {
        self.0.api_key()
    }
    fn worker_type(&self) -> &WorkerType {
        self.0.worker_type()
    }
    fn connection_mode(&self) -> &ConnectionMode {
        self.0.connection_mode()
    }
    fn status(&self) -> WorkerStatus {
        self.0.status()
    }
    fn set_status(&self, status: WorkerStatus) {
        self.0.set_status(status);
    }
    fn load(&self) -> usize {
        self.0.load()
    }
    fn increment_load(&self) {
        self.0.increment_load();
    }
    fn decrement_load(&self) {
        self.0.decrement_load();
    }
    fn routing_key_load(&self) -> usize {
        self.0.routing_key_load()
    }
    fn routing_key_inflight(&self, routing_key: &str) -> usize {
        self.0.routing_key_inflight(routing_key)
    }
    fn increment_routing_key_load(&self, routing_key: &str) {
        self.0.increment_routing_key_load(routing_key);
    }
    fn decrement_routing_key_load(&self, routing_key: &str) {
        self.0.decrement_routing_key_load(routing_key);
    }
    fn processed_requests(&self) -> usize {
        self.0.processed_requests()
    }
    fn increment_processed(&self) {
        self.0.increment_processed();
    }
    fn metadata(&self) -> &WorkerMetadata {
        self.0.metadata()
    }
    fn circuit_breaker_state(&self) -> CircuitState {
        self.0.circuit_breaker_state()
    }
    fn circuit_breaker_can_execute(&self) -> bool {
        self.0.circuit_breaker_can_execute()
    }
    fn record_circuit_breaker_outcome(&self, success: bool) {
        self.0.record_circuit_breaker_outcome(success);
    }
    fn resilience(&self) -> &ResolvedResilience {
        self.0.resilience()
    }
    fn http_client(&self) -> &reqwest::Client {
        self.0.http_client()
    }
    fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
        self.0.http_client_handle_if_initialized()
    }
    async fn check_health_async(&self) -> WorkerResult<()> {
        self.0.check_health_async().await
    }
    async fn engine_connection(&self) -> WorkerResult<Option<Arc<dyn EngineConnection>>> {
        self.0.engine_connection().await
    }
    fn prepare_request(&self, mut req: Value) -> WorkerResult<Value> {
        if let Some(obj) = req.as_object_mut() {
            obj.insert("custom".to_string(), Value::Bool(true));
        }
        self.0.prepare_request(req)
    }
    fn mutates_request(&self) -> bool {
        true
    }
}

#[test]
fn a_custom_prepare_request_keeps_the_value_pipeline() {
    assert!(dp_worker().uses_builtin_prepare_request());
    let custom = CustomPrepare(dp_worker());
    assert!(!custom.uses_builtin_prepare_request());
    let req = generate_request(json!({}));

    let body = serialize_request_body(&req, Some("canonical-model"), &custom, None).unwrap();

    assert_eq!(
        body,
        value_path_bytes(&req, Some("canonical-model"), &custom)
    );
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["custom"], true, "the custom edit ran");
    assert_eq!(
        parsed["data_parallel_rank"], 3,
        "and so did the built-in one"
    );
}

#[test]
fn a_custom_prepare_keeps_its_edits_when_the_budget_is_translated() {
    let worker = CustomPrepare(
        BasicWorkerBuilder::new("http://worker")
            .runtime_type(smg_worker::identity::basic::RuntimeType::Vllm)
            .build(),
    );
    let req = json!({"model":"alias","custom_params":{"thinking_budget":0,"keep":true}});
    let bytes =
        serialize_request_body_with_edits(&req, Some("model"), &worker, None, true, Some(0))
            .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap(),
        json!({"model":"model","custom":true,"thinking_token_budget":0,"custom_params":{"keep":true}})
    );
}

#[test]
fn preserving_fields_retains_extensions_with_alias_and_dp_rank() {
    let req = json!({
        "model": "alias-model",
        "skip_special_tokens": false,
        "separate_reasoning": true,
        "chat_template_kwargs": null,
        "native": {"keep": [false, null]}
    });
    let body = serialize_request_body_with_edits(
        &req,
        Some("canonical-model"),
        &dp_worker(),
        None,
        false,
        None,
    )
    .unwrap();
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        parsed,
        json!({
            "model": "canonical-model",
            "skip_special_tokens": false,
            "separate_reasoning": true,
            "chat_template_kwargs": null,
            "native": {"keep": [false, null]},
            "data_parallel_rank": 3
        })
    );
}

#[test]
fn preserving_fields_retains_extensions_through_custom_prepare() {
    let req = json!({
        "model": "alias-model",
        "skip_special_tokens": false,
        "separate_reasoning": true,
        "chat_template_kwargs": null
    });
    let body = serialize_request_body_with_edits(
        &req,
        Some("canonical-model"),
        &CustomPrepare(dp_worker()),
        None,
        false,
        None,
    )
    .unwrap();
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        parsed,
        json!({
            "model": "canonical-model",
            "skip_special_tokens": false,
            "separate_reasoning": true,
            "chat_template_kwargs": null,
            "custom": true,
            "data_parallel_rank": 3
        })
    );
}
