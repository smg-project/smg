//! Outbound proxy body construction for typed requests.
//!
//! Serializes the typed request straight to bytes and edits the top-level
//! object as borrowed [`RawValue`] slices, so token-heavy payloads
//! (`input_ids`, messages) are never materialized as a `serde_json::Value`
//! tree. The DP-aware rank and the PD router's bootstrap, rank and KV-handoff
//! fields are inserted the same way (see [`RawBody`]). Only a body the raw
//! editor cannot parse (in practice: a non-object) takes the `Value` path,
//! whose bytes every raw edit reproduces.

use std::borrow::Cow;

use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::value::{to_raw_value, RawValue};

use crate::{
    routers::common::{
        request_to_value, serialize_json_sized,
        sglang_fields::{is_stripped_sglang_default, strip_default_sglang_fields, SGLANG_FIELDS},
    },
    worker::{Worker, WorkerError},
};

#[derive(Debug)]
pub(crate) enum RequestBodyError {
    Serialize(serde_json::Error),
    Prepare(WorkerError),
}

/// The field the built-in [`Worker::prepare_request`] adds for a DP-aware
/// worker.
const DATA_PARALLEL_RANK: &str = "data_parallel_rank";

/// Serialize a typed request into the exact bytes the `Value`-mediated
/// pipeline (`to_value` → model rewrite → `prepare_request` → strip →
/// `to_vec`) produces.
pub(crate) fn serialize_request_body<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, RequestBodyError> {
    let bytes = serialize_json_sized(typed_req, raw_len).map_err(RequestBodyError::Serialize)?;

    // A body the raw editor cannot parse (in practice: non-objects) takes
    // the Value pipeline rather than skipping the hooks.
    let Ok(mut body) = RawBody::parse(&bytes) else {
        return value_request_body(typed_req, canonical_model, worker, raw_len);
    };
    // Edits only swap the model id, append the DP rank, or strip fields, so
    // the first pass plus the two inserted values bounds the reserialization.
    let mut extra = 0;
    if let Some(model) = canonical_model {
        let model = to_raw_value(model).map_err(RequestBodyError::Serialize)?;
        extra += model.get().len();
        body.set_model(model);
    }
    // `Worker::prepare_request` on the raw slices, for the built-in edit
    // (`Worker::uses_builtin_prepare_request`): the DP rank goes in after
    // the model rewrite and before the strip, the order the Value pipeline
    // runs, so the wire bytes are the same. A worker with its own
    // `prepare_request` keeps the Value pipeline so its edits and errors
    // still apply.
    if worker.mutates_request() {
        let rank = match worker.dp_rank() {
            Some(rank) if worker.uses_builtin_prepare_request() => rank,
            _ => return value_request_body(typed_req, canonical_model, worker, raw_len),
        };
        let rank = to_raw_value(&rank).map_err(RequestBodyError::Serialize)?;
        extra += DATA_PARALLEL_RANK.len() + rank.get().len() + 4;
        body.insert(DATA_PARALLEL_RANK, rank);
    }
    body.strip_default_sglang_fields();
    if body.mutated() {
        let mut out = Vec::with_capacity(bytes.len() + extra);
        serde_json::to_writer(&mut out, &body).map_err(RequestBodyError::Serialize)?;
        Ok(out)
    } else {
        Ok(bytes)
    }
}

/// The `Value` pipeline, kept for bodies [`RawBody`] cannot parse and for a
/// worker whose `prepare_request` is not the built-in DP-rank edit.
fn value_request_body<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, RequestBodyError> {
    let mut json_val = request_to_value(typed_req, raw_len).map_err(RequestBodyError::Serialize)?;
    if let Some(canonical_model) = canonical_model {
        super::set_request_model(&mut json_val, canonical_model);
    }
    let mut json_val = worker
        .prepare_request(json_val)
        .map_err(RequestBodyError::Prepare)?;
    strip_default_sglang_fields(&mut json_val);
    serialize_json_sized(&json_val, raw_len).map_err(RequestBodyError::Serialize)
}

/// Top-level fields of a serialized JSON object; values stay borrowed raw
/// JSON until an edit replaces one.
///
/// Every edit does what the same edit does to a `serde_json::Map` under
/// `preserve_order`: [`Self::insert`] replaces in place or appends,
/// [`Self::remove`] swap-removes (the last field moves into the freed slot).
/// A body built here is therefore byte-identical to one built through a
/// `Value`, which the tests use as the oracle. Cloning copies field names
/// and slice pointers, not the payload, which is what the PD router's two
/// legs need.
#[derive(Clone)]
pub(crate) struct RawBody<'a> {
    fields: Vec<(String, Cow<'a, RawValue>)>,
    mutated: bool,
}

impl<'a> RawBody<'a> {
    /// Parse the top level of a JSON object; anything else is an error.
    pub(crate) fn parse(bytes: &'a [u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }

    /// Whether an edit changed the body since it was parsed. A body that
    /// did not change can go out as the bytes it was parsed from.
    pub(crate) fn mutated(&self) -> bool {
        self.mutated
    }

    /// The raw JSON of a top-level field.
    pub(crate) fn get(&self, name: &str) -> Option<&RawValue> {
        self.fields
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value.as_ref())
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.fields.iter().any(|(field, _)| field == name)
    }

    /// Mirrors [`super::set_request_model`]: only replaces an existing
    /// field. A body that already names the canonical model is left as it
    /// was parsed.
    pub(crate) fn set_model(&mut self, canonical_model: Box<RawValue>) {
        if let Some((_, value)) = self.fields.iter_mut().find(|(name, _)| name == "model") {
            if value.get() != canonical_model.get() {
                *value = Cow::Owned(canonical_model);
                self.mutated = true;
            }
        }
    }

    /// `serde_json::Map::insert` under `preserve_order`: an existing field
    /// keeps its position, a new one goes last.
    pub(crate) fn insert(&mut self, name: &str, value: Box<RawValue>) {
        match self.fields.iter_mut().find(|(field, _)| field == name) {
            Some((_, slot)) => *slot = Cow::Owned(value),
            None => self.fields.push((name.to_owned(), Cow::Owned(value))),
        }
        self.mutated = true;
    }

    /// `serde_json::Map::remove` under `preserve_order`: the last field moves
    /// into the removed one's slot. Returns whether the field was present.
    pub(crate) fn remove(&mut self, name: &str) -> bool {
        match self.fields.iter().position(|(field, _)| field == name) {
            Some(index) => {
                self.fields.swap_remove(index);
                self.mutated = true;
                true
            }
            None => false,
        }
    }

    /// Mirrors [`strip_default_sglang_fields`], including the swap-remove
    /// ordering of `serde_json::Map::remove` under `preserve_order`.
    pub(crate) fn strip_default_sglang_fields(&mut self) {
        for field in SGLANG_FIELDS {
            let index = self.fields.iter().position(|(name, value)| {
                name == field && is_stripped_sglang_default(field, value.get())
            });
            if let Some(index) = index {
                self.fields.swap_remove(index);
                self.mutated = true;
            }
        }
    }

    /// The wire bytes, into a buffer pre-sized from the incoming request
    /// length (see [`serialize_json_sized`]).
    pub(crate) fn to_bytes(&self, raw_len: Option<usize>) -> serde_json::Result<Vec<u8>> {
        serialize_json_sized(self, raw_len)
    }
}

impl<'de> Deserialize<'de> for RawBody<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RawBodyVisitor;

        impl<'de> Visitor<'de> for RawBodyVisitor {
            type Value = RawBody<'de>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut fields = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((name, value)) = map.next_entry::<String, &'de RawValue>()? {
                    fields.push((name, Cow::Borrowed(value)));
                }
                Ok(RawBody {
                    fields,
                    mutated: false,
                })
            }
        }

        deserializer.deserialize_map(RawBodyVisitor)
    }
}

impl Serialize for RawBody<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.fields.len()))?;
        for (name, value) in &self.fields {
            map.serialize_entry(name, value.as_ref())?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use std::{any::Any, sync::Arc};

    use openai_protocol::{
        chat::ChatCompletionRequest,
        generate::GenerateRequest,
        worker::{ConnectionMode, WorkerStatus},
    };
    use serde_json::{json, Value};

    use super::*;
    use crate::{
        routers::{grpc::backend_client::BackendClient, http::set_request_model},
        worker::{
            circuit_breaker::CircuitState, worker::WorkerMetadata, BasicWorker, BasicWorkerBuilder,
            ResolvedResilience, WorkerResult, WorkerType,
        },
    };

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
        assert_eq!(parsed["skip_special_tokens"], true);

        let aliased = serialize_request_body(&req, Some("canonical-model"), &worker, None).unwrap();
        assert_eq!(
            aliased,
            value_path_bytes(&req, Some("canonical-model"), &worker)
        );
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
        fn consecutive_failures_increment(&self) -> usize {
            self.0.consecutive_failures_increment()
        }
        fn consecutive_failures_reset(&self) {
            self.0.consecutive_failures_reset();
        }
        fn consecutive_successes_increment(&self) -> usize {
            self.0.consecutive_successes_increment()
        }
        fn consecutive_successes_reset(&self) {
            self.0.consecutive_successes_reset();
        }
        fn total_pending_probes(&self) -> usize {
            self.0.total_pending_probes()
        }
        fn total_pending_probes_increment(&self) -> usize {
            self.0.total_pending_probes_increment()
        }
        fn total_pending_probes_reset(&self) {
            self.0.total_pending_probes_reset();
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
        async fn get_backend_client(&self) -> WorkerResult<Option<Arc<BackendClient>>> {
            self.0.get_backend_client().await
        }
        async fn grpc_health_check(&self) -> WorkerResult<bool> {
            self.0.grpc_health_check().await
        }
        async fn zmq_health_check(&self) -> WorkerResult<bool> {
            self.0.zmq_health_check().await
        }
        async fn http_health_check(&self) -> WorkerResult<bool> {
            self.0.http_health_check().await
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
}
