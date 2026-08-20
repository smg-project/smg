//! Outbound proxy body construction for typed requests.
//!
//! Serializes the typed request straight to bytes and edits the top-level
//! object as borrowed [`RawValue`] slices, so token-heavy payloads
//! (`input_ids`, messages) are never materialized as a `serde_json::Value`
//! tree. Workers whose `prepare_request` rewrites the body still take the
//! `Value` path (that hook is defined on `Value`), as does any body the
//! raw editor cannot parse.

use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::value::{to_raw_value, RawValue};

use crate::{
    routers::{
        common::{serialize_json_sized, serialized_capacity},
        openai::{is_stripped_sglang_default, strip_default_sglang_fields, SGLANG_FIELDS},
    },
    worker::{Worker, WorkerError},
};

#[derive(Debug)]
pub(crate) enum RequestBodyError {
    Serialize(serde_json::Error),
    Prepare(WorkerError),
}

/// Serialize a typed request into the exact bytes the `Value`-mediated
/// pipeline (`to_value` → model rewrite → `prepare_request` → strip →
/// `to_vec`) produces.
pub(crate) fn serialize_request_body<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, RequestBodyError> {
    if worker.mutates_request() {
        return value_request_body(typed_req, canonical_model, worker, raw_len);
    }

    let bytes = to_vec_value_compatible(typed_req, raw_len).map_err(RequestBodyError::Serialize)?;
    let canonical_raw = canonical_model
        .map(to_raw_value)
        .transpose()
        .map_err(RequestBodyError::Serialize)?;

    // A body the raw editor cannot parse (in practice: non-objects) takes
    // the Value pipeline rather than skipping the hooks.
    let mut body = match serde_json::from_slice::<RawBody>(&bytes) {
        Ok(body) => body,
        Err(_) => return value_request_body(typed_req, canonical_model, worker, raw_len),
    };
    if let Some(model) = canonical_raw.as_deref() {
        body.set_model(model);
    }
    body.strip_default_sglang_fields();
    if body.mutated {
        // Edits only strip fields or swap the model id, so the first pass
        // plus the canonical id bounds the reserialization.
        let capacity = bytes.len() + canonical_raw.as_deref().map_or(0, |m| m.get().len());
        let mut out = Vec::with_capacity(capacity);
        serde_json::to_writer(&mut out, &body).map_err(RequestBodyError::Serialize)?;
        Ok(out)
    } else {
        Ok(bytes)
    }
}

/// The `Value` pipeline, kept for workers whose `prepare_request` edits the
/// body ([`Worker::mutates_request`]) and as the fallback for bodies
/// [`RawBody`] cannot parse.
fn value_request_body<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, RequestBodyError> {
    let mut json_val = serde_json::to_value(typed_req).map_err(RequestBodyError::Serialize)?;
    if let Some(canonical_model) = canonical_model {
        super::set_request_model(&mut json_val, canonical_model);
    }
    let mut json_val = worker
        .prepare_request(json_val)
        .map_err(RequestBodyError::Prepare)?;
    strip_default_sglang_fields(&mut json_val);
    serialize_json_sized(&json_val, raw_len).map_err(RequestBodyError::Serialize)
}

/// `serde_json::to_value` stores `f32` widened to `f64`, so the `Value`
/// pipeline has always emitted the widened decimal form. The plain writer
/// emits the shorter `f32` form instead; widen here to keep wire bytes
/// identical.
struct F32WideningFormatter;

impl serde_json::ser::Formatter for F32WideningFormatter {
    fn write_f32<W>(&mut self, writer: &mut W, value: f32) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        self.write_f64(writer, f64::from(value))
    }
}

fn to_vec_value_compatible<T: Serialize>(
    value: &T,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, serde_json::Error> {
    let mut buf = Vec::with_capacity(raw_len.map_or(128, serialized_capacity));
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, F32WideningFormatter);
    value.serialize(&mut ser)?;
    Ok(buf)
}

/// Top-level fields of a serialized request; values stay borrowed raw JSON.
struct RawBody<'a> {
    fields: Vec<(String, &'a RawValue)>,
    mutated: bool,
}

impl<'a> RawBody<'a> {
    /// Mirrors [`super::set_request_model`]: only replaces an existing field.
    fn set_model(&mut self, canonical_model: &'a RawValue) {
        if let Some((_, value)) = self.fields.iter_mut().find(|(name, _)| name == "model") {
            *value = canonical_model;
            self.mutated = true;
        }
    }

    /// Mirrors [`strip_default_sglang_fields`], including the swap-remove
    /// ordering of `serde_json::Map::remove` under `preserve_order`.
    fn strip_default_sglang_fields(&mut self) {
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
                while let Some(entry) = map.next_entry::<String, &'de RawValue>()? {
                    fields.push(entry);
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
            map.serialize_entry(name, value)?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::{chat::ChatCompletionRequest, generate::GenerateRequest};
    use serde_json::{json, Value};

    use super::*;
    use crate::{
        routers::http::set_request_model,
        worker::{BasicWorker, BasicWorkerBuilder, WorkerType},
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

    /// The pre-existing pipeline, verbatim: the produced bytes are the wire
    /// contract the fast path must reproduce.
    fn value_path_bytes<T: Serialize>(
        typed_req: &T,
        canonical_model: Option<&str>,
        worker: &dyn Worker,
    ) -> Vec<u8> {
        let mut json_val = serde_json::to_value(typed_req).unwrap();
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
        assert_eq!(body, to_vec_value_compatible(&req, None).unwrap());
    }

    #[test]
    fn untouched_body_reuses_the_direct_serialization() {
        let worker = worker();
        // `return_hidden_states: true` survives the strip, so nothing in this
        // body needs editing.
        let req = generate_request(json!({"return_hidden_states": true}));

        let body = serialize_request_body(&req, None, &worker, None).unwrap();

        assert_eq!(body, value_path_bytes(&req, None, &worker));
        assert_eq!(body, to_vec_value_compatible(&req, None).unwrap());
    }

    #[test]
    fn f32_fields_keep_the_value_path_widening() {
        let worker = worker();
        let req = generate_request(json!({}));

        let body =
            String::from_utf8(serialize_request_body(&req, None, &worker, None).unwrap()).unwrap();

        // `to_value` widens `f32` to `f64`; the plain writer would emit the
        // shorter `0.7` and change wire bytes.
        let widened = serde_json::to_string(&Value::from(0.7f32)).unwrap();
        assert_ne!(widened, "0.7");
        assert!(body.contains(&widened));
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
        assert_eq!(plain, to_vec_value_compatible(&req, None).unwrap());
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
        assert_eq!(body, to_vec_value_compatible(&req, None).unwrap());
    }

    #[test]
    fn non_object_body_falls_back_to_the_value_pipeline() {
        let worker = worker();
        let req = vec![1, 2, 3];

        // The raw editor rejects the shape, so this exercises the fallback.
        let direct = to_vec_value_compatible(&req, None).unwrap();
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
}
