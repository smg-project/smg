//! SGLang 0.5.21 translation for the public OpenAI Decisions contract.
//!
//! System One scores the answer labels without generating reasoning. Its
//! confidence is SGLang's documented statistic; it is not asserted to be
//! numerically equivalent to OpenAI's confidence.
//! System One schedules within its own data-parallel group and does not honor
//! SMG's `data_parallel_rank` pinning, so per-rank cache affinity is unavailable.

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, HeaderValue},
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::decisions::{
    DecisionAnswer, DecisionChoiceProbability, DecisionInput, DecisionInputContent,
    DecisionInputPart, DecisionInputTokensDetails, DecisionOutputTokensDetails, DecisionQuestion,
    DecisionResponse, DecisionScoreProbability, DecisionUsage, DecisionValue, DecisionsRequest,
};
use serde::{de, Deserialize, Deserializer};
use serde_json::{json, Map, Value};

use crate::routers::error;

pub(super) const UPSTREAM_ROUTE: &str = "/v1/systemone";

pub(super) struct SglangDecisionAdapter {
    request: Value,
    questions: Vec<MappedQuestion>,
}

enum MappedQuestion {
    Predicate {
        name: Option<String>,
    },
    Choice {
        name: Option<String>,
        choices: Vec<DecisionValue>,
    },
    Score {
        name: Option<String>,
        labels: Vec<String>,
    },
}

impl MappedQuestion {
    fn name(&self) -> Option<&str> {
        match self {
            Self::Predicate { name } | Self::Choice { name, .. } | Self::Score { name, .. } => {
                name.as_deref()
            }
        }
    }
}

impl SglangDecisionAdapter {
    pub fn new(request: &DecisionsRequest) -> Result<Self, String> {
        reject_extensions(&request.other, "request")?;
        if let DecisionInput::Messages(messages) = &request.input {
            for message in messages {
                reject_extensions(&message.other, "input message")?;
                if let DecisionInputContent::Parts(parts) = &message.content {
                    for part in parts {
                        match part {
                            DecisionInputPart::InputText { other, .. } => {
                                reject_extensions(other, "input text part")?;
                            }
                            DecisionInputPart::InputImage { .. } => {
                                return Err(
                                    "SGLang 0.5.21 Decisions does not support image input".into()
                                );
                            }
                        }
                    }
                }
            }
        }
        if request.input.text().trim().is_empty() {
            return Err("SGLang Decisions requires nonempty text input".into());
        }
        let mut questions = Map::new();
        for (index, question) in request.questions.iter().enumerate() {
            if question.instructions().trim().is_empty() {
                return Err(format!(
                    "SGLang Decisions question {index} requires nonempty instructions"
                ));
            }
            let mapped = match question {
                DecisionQuestion::Predicate {
                    instructions,
                    other,
                    ..
                } => {
                    reject_extensions(other, "predicate question")?;
                    json!({"type":"noul", "instructions":instructions})
                }
                DecisionQuestion::Choice {
                    instructions,
                    choices,
                    other,
                    ..
                } => {
                    reject_extensions(other, "choice question")?;
                    if choices.is_empty() || choices.len() > 255 {
                        return Err(format!(
                            "SGLang Decisions question {index} requires 1 to 255 choices"
                        ));
                    }
                    let mut criteria = Map::new();
                    for (option_index, choice) in choices.iter().enumerate() {
                        reject_extensions(&choice.other, "choice option")?;
                        let mut description = json!({"value":choice.value});
                        if let Some(value) = &choice.description {
                            description["description"] = json!(value);
                        }
                        criteria.insert(format!("o{option_index}"), description);
                    }
                    json!({"type":"choice", "instructions":instructions, "criteria":criteria})
                }
                DecisionQuestion::Score {
                    instructions,
                    levels,
                    other,
                    ..
                } => {
                    reject_extensions(other, "score question")?;
                    if levels.is_empty() || levels.len() > 10 {
                        return Err(format!(
                            "SGLang Decisions question {index} requires 1 to 10 score levels"
                        ));
                    }
                    let mut criteria = Vec::with_capacity(levels.len());
                    for level in levels {
                        reject_extensions(&level.other, "score level")?;
                        let mut description = json!({"label":level.label});
                        if let Some(value) = &level.description {
                            description["description"] = json!(value);
                        }
                        criteria.push(description);
                    }
                    json!({"type":"score", "instructions":instructions, "criteria":criteria})
                }
            };
            questions.insert(format!("q{index}"), mapped);
        }
        if questions.is_empty() {
            return Err("SGLang Decisions requires at least one question".into());
        }
        // Structured text evidence is rendered by System One as ordered JSON,
        // preserving message/part boundaries and roles. Images are rejected above.
        // safety_identifier is consumed at the gateway boundary, not a backend field.
        Ok(Self {
            request: json!({"model":request.model, "state":request.input, "questions":questions}),
            questions: request
                .questions
                .iter()
                .map(|question| {
                    let name = question.name().map(str::to_owned);
                    match question {
                        DecisionQuestion::Predicate { .. } => MappedQuestion::Predicate { name },
                        DecisionQuestion::Choice { choices, .. } => MappedQuestion::Choice {
                            name,
                            choices: choices.iter().map(|choice| choice.value.clone()).collect(),
                        },
                        DecisionQuestion::Score { levels, .. } => MappedQuestion::Score {
                            name,
                            labels: levels.iter().map(|level| level.label.clone()).collect(),
                        },
                    }
                })
                .collect(),
        })
    }

    pub fn request(&self) -> &Value {
        &self.request
    }

    /// Drop evidence, instructions and descriptions after dispatch serialization.
    /// Only the original names, typed choices and score labels remain for mapping.
    pub fn release_request(&mut self) {
        self.request = Value::Null;
    }

    pub async fn convert_response(self, response: Response, max_payload_size: usize) -> Response {
        let (mut parts, body) = response.into_parts();
        let converted = match to_bytes(body, max_payload_size).await {
            Err(_) => error::bad_gateway(
                "invalid_decisions_response",
                "SGLang Decisions response exceeds the payload limit or could not be read",
            ),
            Ok(bytes) => {
                let parsed = serde_json::from_slice::<UniqueValue>(&bytes).map(|value| value.0);
                if parts.status.is_success() {
                    match parsed
                        .map_err(|err| format!("Invalid SGLang Decisions JSON: {err}"))
                        .and_then(|value| self.map_response(&value))
                    {
                        Ok(value) => match serde_json::to_vec(&value) {
                            Ok(bytes) if bytes.len() <= max_payload_size => {
                                let mut response = Response::new(Body::from(bytes));
                                *response.status_mut() = parts.status;
                                response
                            }
                            Ok(_) => error::bad_gateway(
                                "invalid_decisions_response",
                                "Converted Decisions response exceeds the payload limit",
                            ),
                            Err(err) => error::bad_gateway(
                                "invalid_decisions_response",
                                format!("Could not encode Decisions response: {err}"),
                            ),
                        },
                        Err(message) => error::bad_gateway("invalid_decisions_response", message),
                    }
                } else {
                    let value = parsed.ok();
                    let message = value
                        .as_ref()
                        .and_then(|value| {
                            value
                                .pointer("/error/message")
                                .and_then(Value::as_str)
                                .or_else(|| value.get("detail").and_then(Value::as_str))
                        })
                        .map(str::to_owned)
                        .unwrap_or_else(|| {
                            value
                                .as_ref()
                                .and_then(|value| value.get("detail"))
                                .map(Value::to_string)
                                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned())
                        });
                    let code = value
                        .as_ref()
                        .and_then(|value| value.pointer("/error/code"))
                        .and_then(Value::as_str)
                        .unwrap_or("upstream_decisions_error");
                    let mut response = error::create_error(parts.status, code, message);
                    if let Some(value) = value.filter(|value| {
                        value
                            .pointer("/error/message")
                            .and_then(Value::as_str)
                            .is_some()
                    }) {
                        // Keep the backend's OpenAI error type, param and extensions.
                        // The helper response still supplies trusted metric extensions.
                        *response.body_mut() = Json(value).into_response().into_body();
                    }
                    response
                }
            }
        };
        // Rewriting the body invalidates lengths, encodings and entity validators.
        // Keep routing/request identifiers, Retry-After, and other safe headers.
        for name in [
            "content-length",
            "content-encoding",
            "transfer-encoding",
            "etag",
            "last-modified",
            "content-md5",
            "digest",
            "content-digest",
            "repr-digest",
            "content-range",
            "accept-ranges",
        ] {
            parts.headers.remove(name);
        }
        parts
            .headers
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let (converted_parts, body) = converted.into_parts();
        parts.status = converted_parts.status;
        parts.extensions.extend(converted_parts.extensions);
        for (name, value) in &converted_parts.headers {
            if name != CONTENT_TYPE && name != "content-length" {
                parts.headers.insert(name.clone(), value.clone());
            }
        }
        Response::from_parts(parts, body)
    }

    fn map_response(&self, value: &Value) -> Result<DecisionResponse, String> {
        let model = string_field(value, "model")?.to_owned();
        let upstream_answers = object_field(value, "answers")?;
        if upstream_answers.len() != self.questions.len() {
            return Err("SGLang Decisions returned an unexpected number of answers".into());
        }
        let mut answers = Vec::with_capacity(self.questions.len());
        for (index, question) in self.questions.iter().enumerate() {
            let id = format!("q{index}");
            let answer = upstream_answers
                .get(&id)
                .ok_or_else(|| format!("SGLang Decisions omitted answer {id}"))?;
            let name = question.name().map(str::to_owned);
            let kind = string_field(answer, "type")?;
            let mapped = match question {
                MappedQuestion::Predicate { .. } if kind == "noul" => DecisionAnswer::Predicate {
                    name,
                    probability: probability_field(answer, "noul")?,
                },
                MappedQuestion::Choice { choices, .. } if kind == "choice" => {
                    let scores = object_field(answer, "probabilities")?;
                    if scores.len() != choices.len() {
                        return Err(format!(
                            "SGLang Decisions returned invalid probabilities for {id}"
                        ));
                    }
                    let chosen = string_field(answer, "choice")?;
                    let mut choice = None;
                    let mut chosen_probability = None;
                    let mut probabilities = Vec::with_capacity(choices.len());
                    for (option_index, option) in choices.iter().enumerate() {
                        let key = format!("o{option_index}");
                        if chosen == key {
                            choice = Some(option.clone());
                        }
                        let probability =
                            probability_value(scores.get(&key).ok_or_else(|| {
                                format!("SGLang Decisions omitted probability {id}/{key}")
                            })?)?;
                        if chosen == key {
                            chosen_probability = Some(probability);
                        }
                        probabilities.push(DecisionChoiceProbability {
                            value: option.clone(),
                            probability,
                        });
                    }
                    validate_probability_sum(probabilities.iter().map(|value| value.probability))?;
                    let top = probabilities
                        .iter()
                        .map(|value| value.probability)
                        .fold(0.0, f64::max);
                    if chosen_probability.is_some_and(|probability| probability < top) {
                        return Err(format!(
                            "SGLang Decisions returned a choice that is not most probable for {id}"
                        ));
                    }
                    DecisionAnswer::Choice {
                        name,
                        choice: choice.ok_or_else(|| {
                            format!("SGLang Decisions returned unknown choice {chosen} for {id}")
                        })?,
                        confidence: probability_field(answer, "confidence")?,
                        probabilities,
                    }
                }
                MappedQuestion::Score { labels, .. } if kind == "score" => {
                    let scores = object_field(answer, "probabilities")?;
                    if scores.len() != labels.len() {
                        return Err(format!(
                            "SGLang Decisions returned invalid probabilities for {id}"
                        ));
                    }
                    let score = number_field(answer, "score")?;
                    if !(0.0..=(labels.len() - 1) as f64).contains(&score) {
                        return Err(format!(
                            "SGLang Decisions returned an out-of-range score for {id}"
                        ));
                    }
                    let mut probabilities = Vec::with_capacity(labels.len());
                    for (level_index, level) in labels.iter().enumerate() {
                        let key = level_index.to_string();
                        probabilities.push(DecisionScoreProbability {
                            value: level_index as i64,
                            label: level.clone(),
                            probability: probability_value(scores.get(&key).ok_or_else(|| {
                                format!("SGLang Decisions omitted probability {id}/{key}")
                            })?)?,
                        });
                    }
                    validate_probability_sum(probabilities.iter().map(|value| value.probability))?;
                    let expected: f64 = probabilities
                        .iter()
                        .map(|value| value.value as f64 * value.probability)
                        .sum();
                    if (score - expected).abs()
                        > PROBABILITY_TOLERANCE * ((labels.len() - 1) as f64).max(1.0)
                    {
                        return Err(format!("SGLang Decisions returned a score inconsistent with its probabilities for {id}"));
                    }
                    DecisionAnswer::Score {
                        name,
                        score,
                        confidence: probability_field(answer, "confidence")?,
                        probabilities,
                    }
                }
                _ => {
                    return Err(format!(
                        "SGLang Decisions returned an unexpected answer type for {id}"
                    ))
                }
            };
            answers.push(mapped);
        }
        let usage = value.get("usage").ok_or("SGLang Decisions omitted usage")?;
        let input_tokens = usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .ok_or("SGLang Decisions returned invalid input_tokens")?;
        let output_tokens = usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .ok_or("SGLang Decisions returned invalid output_tokens")?;
        if output_tokens != 0 {
            return Err("SGLang Decisions scoring unexpectedly reported output tokens".into());
        }
        let total_tokens = input_tokens
            .checked_add(output_tokens)
            .ok_or("SGLang Decisions token count overflow")?;
        Ok(DecisionResponse {
            model,
            answers,
            usage: DecisionUsage {
                input_tokens,
                output_tokens,
                total_tokens,
                // System One does not report cache counters. Zero denotes
                // unreported cache usage, not a measured absence of cache activity.
                input_tokens_details: DecisionInputTokensDetails {
                    cached_tokens: 0,
                    cache_write_tokens: 0,
                },
                // Label scoring generates no reasoning or output tokens.
                output_tokens_details: DecisionOutputTokensDetails {
                    reasoning_tokens: 0,
                },
            },
        })
    }
}

fn reject_extensions(other: &Map<String, Value>, context: &str) -> Result<(), String> {
    if let Some(field) = other.keys().next() {
        Err(format!(
            "SGLang Decisions cannot honor field {field:?} on {context}"
        ))
    } else {
        Ok(())
    }
}

fn object_field<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("SGLang Decisions returned invalid or missing {field}"))
}

fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("SGLang Decisions returned invalid or missing {field}"))
}

fn number_field(value: &Value, field: &str) -> Result<f64, String> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("SGLang Decisions returned invalid or missing {field}"))
}

fn probability_field(value: &Value, field: &str) -> Result<f64, String> {
    probability_value(
        value
            .get(field)
            .ok_or_else(|| format!("SGLang Decisions omitted {field}"))?,
    )
}

// Allow float32 softmax roundoff without accepting inconsistent distributions.
const PROBABILITY_TOLERANCE: f64 = 1e-6;

fn validate_probability_sum(probabilities: impl Iterator<Item = f64>) -> Result<(), String> {
    if (probabilities.sum::<f64>() - 1.0).abs() > PROBABILITY_TOLERANCE {
        Err("SGLang Decisions probabilities do not sum to one".into())
    } else {
        Ok(())
    }
}

fn probability_value(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|number| number.is_finite() && (0.0..=1.0).contains(number))
        .ok_or_else(|| "SGLang Decisions returned an invalid probability or confidence".into())
}

/// Reject duplicate JSON keys before they can silently replace an answer or
/// candidate probability. The normal Value deserializer retains only the last.
struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = UniqueValue;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("JSON without duplicate object keys")
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Bool(value)))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(json!(value)))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(json!(value)))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|value| UniqueValue(Value::Number(value)))
                    .ok_or_else(|| E::custom("non-finite number"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value.to_owned())))
            }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value)))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut access: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = access.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueValue(Value::Array(values)))
            }
            fn visit_map<A: de::MapAccess<'de>>(
                self,
                mut access: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = access.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom(format!("duplicate key {key:?}")));
                    }
                    values.insert(key, access.next_value::<UniqueValue>()?.0);
                }
                Ok(UniqueValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, http::StatusCode, response::IntoResponse, Json};
    use serde_json::json;

    use super::*;

    fn request(input: Value, questions: Value) -> DecisionsRequest {
        let mut fields = Map::new();
        fields.insert("model".into(), json!("test"));
        fields.insert("input".into(), input);
        fields.insert("questions".into(), questions);
        serde_json::from_value(Value::Object(fields)).unwrap()
    }

    fn all_questions() -> Value {
        json!([
            {"type":"predicate","instructions":"Is it ready?","name":"same"},
            {"type":"choice","instructions":"Select","choices":[{"value":true},{"value":"true","description":"Literal text"}],"name":"same"},
            {"type":"score","instructions":"Rate","levels":[{"label":"low"},{"label":"high","description":"Fully ready"}]}
        ])
    }

    fn upstream() -> Value {
        json!({"model":"served","answers":{
            "q2":{"type":"score","score":0.75,"confidence":0.5,"probabilities":{"0":0.25,"1":0.75},"legend":{"0":{"label":"low"},"1":{"label":"high","description":"Fully ready"}},"x_label_mass":0.9},
            "q1":{"type":"choice","choice":"o1","confidence":0.6,"probabilities":{"o1":0.8,"o0":0.2},"x_label_mass":0.9},
            "q0":{"type":"noul","noul":0.8,"x_label_mass":0.9}
        },"usage":{"input_tokens":22,"output_tokens":0}})
    }

    async fn body(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 100_000).await.unwrap()).unwrap()
    }

    #[test]
    fn maps_typed_values_and_duplicate_names_to_private_ids() {
        let adapter =
            SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
        assert_eq!(
            adapter.request(),
            &json!({"model":"test","state":"Evidence","questions":{
                "q0":{"type":"noul","instructions":"Is it ready?"},
                "q1":{"type":"choice","instructions":"Select","criteria":{"o0":{"value":true},"o1":{"value":"true","description":"Literal text"}}},
                "q2":{"type":"score","instructions":"Rate","criteria":[{"label":"low"},{"label":"high","description":"Fully ready"}]}
            }})
        );
    }

    #[test]
    fn structured_text_evidence_retains_message_and_part_order() {
        let input = json!([
            {"role":"user","type":"message","content":[{"type":"input_text","text":"first"},{"type":"input_text","text":"second"}]},
            {"role":"user","content":"third"}
        ]);
        let adapter = SglangDecisionAdapter::new(&request(input.clone(), all_questions())).unwrap();
        assert_eq!(adapter.request()["state"], input);
    }

    #[test]
    fn rejects_images_and_extensions_instead_of_dropping_them() {
        let image = json!([{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AQ=="}]}]);
        let error = SglangDecisionAdapter::new(&request(image, all_questions()))
            .err()
            .unwrap();
        assert!(error.contains("image"));
        for value in [
            json!({"model":"test","input":"x","questions":[{"type":"predicate","instructions":"x"}],"temperature":1}),
            json!({"model":"test","input":[{"role":"user","content":"x","extra":true}],"questions":[{"type":"predicate","instructions":"x"}]}),
            json!({"model":"test","input":[{"role":"user","content":[{"type":"input_text","text":"x","extra":true}]}],"questions":[{"type":"predicate","instructions":"x"}]}),
            json!({"model":"test","input":"x","questions":[{"type":"predicate","instructions":"x","extra":true}]}),
            json!({"model":"test","input":"x","questions":[{"type":"choice","instructions":"x","choices":[{"value":"a","extra":true}]}]}),
            json!({"model":"test","input":"x","questions":[{"type":"score","instructions":"x","levels":[{"label":"a","extra":true}]}]}),
        ] {
            let request = serde_json::from_value(value).unwrap();
            assert!(
                SglangDecisionAdapter::new(&request)
                    .err()
                    .unwrap()
                    .contains("extra")
                    || SglangDecisionAdapter::new(&request)
                        .err()
                        .unwrap()
                        .contains("temperature")
            );
        }
    }

    #[test]
    fn enforces_sglang_bounds_and_accepts_singletons() {
        for (kind, field, item, limit) in [
            ("choice", "choices", json!({"value":"a"}), 255),
            ("score", "levels", json!({"label":"a"}), 10),
        ] {
            for size in [0, limit + 1] {
                let mut question = json!({"type":kind,"instructions":"x"});
                question[field] = json!(vec![item.clone(); size]);
                assert!(
                    SglangDecisionAdapter::new(&request(json!("x"), json!([question]))).is_err()
                );
            }
            let mut question = json!({"type":kind,"instructions":"x"});
            question[field] = json!([item]);
            assert!(SglangDecisionAdapter::new(&request(json!("x"), json!([question]))).is_ok());
        }
        assert!(SglangDecisionAdapter::new(&request(json!(" \n"), all_questions())).is_err());
        assert!(SglangDecisionAdapter::new(&request(
            json!("x"),
            json!([{ "type":"predicate","instructions":" " }])
        ))
        .is_err());
    }

    #[tokio::test]
    async fn converts_answers_in_request_order_with_typed_values_and_usage() {
        let adapter =
            SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
        let response = adapter
            .convert_response(Json(upstream()).into_response(), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            body(response).await,
            json!({"model":"served","answers":[
            {"type":"predicate","name":"same","probability":0.8},
            {"type":"choice","name":"same","choice":"true","confidence":0.6,"probabilities":[{"value":true,"probability":0.2},{"value":"true","probability":0.8}]},
            {"type":"score","name":null,"score":0.75,"confidence":0.5,"probabilities":[{"value":0,"label":"low","probability":0.25},{"value":1,"label":"high","probability":0.75}]}
        ],"usage":{"input_tokens":22,"output_tokens":0,"total_tokens":22,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}})
        );
    }

    #[tokio::test]
    async fn preserves_headers_and_status_while_removing_body_validators() {
        let adapter =
            SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
        let response = Response::builder()
            .status(StatusCode::CREATED)
            .header("x-request-id", "request-42")
            .header("x-smg-routed-worker", "worker")
            .header("content-length", "999")
            .header("content-encoding", "gzip")
            .header("etag", "old")
            .header("content-md5", "old")
            .body(Body::from(upstream().to_string()))
            .unwrap();
        let response = adapter.convert_response(response, 100_000).await;
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(response.headers()["x-request-id"], "request-42");
        assert_eq!(response.headers()["x-smg-routed-worker"], "worker");
        for header in ["content-length", "content-encoding", "etag", "content-md5"] {
            assert!(!response.headers().contains_key(header));
        }
        assert_eq!(response.headers()["content-type"], "application/json");
    }

    #[tokio::test]
    async fn upstream_validation_errors_remain_errors_with_useful_messages() {
        for fixture in [
            json!({"error":{"message":"unsupported reasoning template","type":"invalid_request_error","code":"invalid_model"}}),
            json!({"detail":"question q0 label is not a single token"}),
        ] {
            let adapter =
                SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
            let response = adapter
                .convert_response(
                    (StatusCode::BAD_REQUEST, Json(fixture)).into_response(),
                    100_000,
                )
                .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let value = body(response).await;
            assert!(
                value["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("unsupported")
                    || value["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("single token")
            );
            assert!(value.get("answers").is_none());
        }
    }

    #[tokio::test]
    async fn malformed_successes_fail_as_bad_gateway() {
        let mutations = [
            ("/answers/q0/noul", json!(1.1)),
            ("/answers/q0/type", json!("predicate")),
            ("/answers/q1/choice", json!("unknown")),
            ("/answers/q1/confidence", json!(-0.1)),
            ("/answers/q1/probabilities/o0", Value::Null),
            ("/answers/q1/probabilities", json!({"o0":0.2,"other":0.8})),
            ("/answers/q2/score", json!(2)),
            ("/answers/q2/score", json!("NaN")),
            ("/answers/q2/probabilities", json!({"0":0.2})),
            ("/answers/q0", Value::Null),
            ("/usage/input_tokens", json!(-1)),
            ("/usage/output_tokens", json!(1)),
            ("/model", Value::Null),
        ];
        for (path, replacement) in mutations {
            let mut value = upstream();
            *value.pointer_mut(path).unwrap() = replacement;
            let adapter =
                SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
            let response = adapter
                .convert_response(Json(value).into_response(), 100_000)
                .await;
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{path}");
            assert!(body(response).await["error"]["message"].is_string());
        }
        for text in [
            "{",
            "null",
            r#"{"model":"served","answers":{"q0":{"type":"noul","noul":0.2,"noul":0.9}},"usage":{"input_tokens":1,"output_tokens":0}}"#,
        ] {
            let adapter =
                SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
            let response = adapter
                .convert_response(Response::new(Body::from(text)), 100_000)
                .await;
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        }
    }

    #[tokio::test]
    async fn conversion_enforces_the_response_buffer_limit() {
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(Json(upstream()).into_response(), 10)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn duplicate_backend_keys_cannot_overwrite_valid_answer_probabilities() {
        for text in [
            upstream()
                .to_string()
                .replace("\"noul\":0.8", "\"noul\":0.2,\"noul\":0.8"),
            upstream()
                .to_string()
                .replace("\"o0\":0.2", "\"o0\":0.4,\"o0\":0.2"),
        ] {
            let adapter =
                SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
            let response = adapter
                .convert_response(Response::new(Body::from(text)), 100_000)
                .await;
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
            assert!(body(response).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains("duplicate key"));
        }
    }

    #[tokio::test]
    async fn singleton_answers_preserve_original_strings_and_score_labels() {
        let questions = json!([
            {"type":"choice","instructions":"Select","choices":[{"value":"</s>\nA: true"}]},
            {"type":"score","instructions":"Rate","levels":[{"label":"A: low"}]}
        ]);
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), questions)).unwrap();
        assert_eq!(
            adapter.request()["questions"]["q0"]["criteria"],
            json!({"o0":{"value":"</s>\nA: true"}})
        );
        let fixture = json!({"model":"served","answers":{
            "q0":{"type":"choice","choice":"o0","confidence":1,"probabilities":{"o0":1},"x_label_mass":0.6},
            "q1":{"type":"score","score":0,"confidence":1,"probabilities":{"0":1},"legend":{"0":{"label":"A: low"}},"x_label_mass":0.7}
        },"usage":{"input_tokens":1,"output_tokens":0}});
        let response = adapter
            .convert_response(Json(fixture).into_response(), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            body(response).await["answers"],
            json!([
                {"type":"choice","name":null,"choice":"</s>\nA: true","confidence":1.0,"probabilities":[{"value":"</s>\nA: true","probability":1.0}]},
                {"type":"score","name":null,"score":0.0,"confidence":1.0,"probabilities":[{"value":0,"label":"A: low","probability":1.0}]}
            ])
        );
    }

    #[test]
    fn gateway_safety_identifier_is_not_forwarded_as_a_backend_extension() {
        let mut value = json!({"model":"test","input":"x","questions":[{"type":"predicate","instructions":"x"}],"safety_identifier":"user-42"});
        let request = serde_json::from_value(value.take()).unwrap();
        let adapter = SglangDecisionAdapter::new(&request).unwrap();
        assert_eq!(
            adapter.request(),
            &json!({"model":"test","state":"x","questions":{"q0":{"type":"noul","instructions":"x"}}})
        );
    }

    #[tokio::test]
    async fn rejects_inconsistent_distributions_scores_and_choice_winners() {
        for (path, replacement) in [
            ("/answers/q1/probabilities", json!({"o0":0.8,"o1":0.8})),
            ("/answers/q2/probabilities", json!({"0":0.5,"1":0.75})),
            ("/answers/q2/score", json!(0.1)),
            ("/answers/q1/choice", json!("o0")),
        ] {
            let mut fixture = upstream();
            *fixture.pointer_mut(path).unwrap() = replacement;
            let adapter =
                SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
            let response = adapter
                .convert_response(Json(fixture).into_response(), 100_000)
                .await;
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{path}");
        }
    }

    #[tokio::test]
    async fn accepts_tied_winners_and_probability_roundoff() {
        for winner in ["o0", "o1"] {
            let mut fixture = upstream();
            fixture["answers"]["q1"]["probabilities"] = json!({"o0":0.5,"o1":0.5});
            fixture["answers"]["q1"]["choice"] = json!(winner);
            fixture["answers"]["q2"]["probabilities"] = json!({"0":0.25,"1":0.75000001});
            let adapter =
                SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
            let response = adapter
                .convert_response(Json(fixture).into_response(), 100_000)
                .await;
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn preserves_existing_openai_error_envelopes() {
        let fixture = json!({"error":{"message":"unsupported template","type":"invalid_request_error","code":"unsupported_model","param":"model","backend_context":{"retry":false}},"trace":"abc"});
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), all_questions())).unwrap();
        let response = adapter
            .convert_response(
                (StatusCode::BAD_REQUEST, Json(fixture.clone())).into_response(),
                100_000,
            )
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body(response).await, fixture);
    }

    #[tokio::test]
    async fn bounds_the_encoded_public_response_including_original_labels() {
        let questions = json!([{ "type":"choice", "instructions":"Select", "choices":[{"value":"x".repeat(2000)}]}]);
        let fixture = json!({"model":"served","answers":{"q0":{"type":"choice","choice":"o0","confidence":1,"probabilities":{"o0":1},"x_label_mass":0.5}},"usage":{"input_tokens":1,"output_tokens":0}});
        let limit = fixture.to_string().len() + 20;
        let adapter = SglangDecisionAdapter::new(&request(json!("x"), questions)).unwrap();
        let response = adapter
            .convert_response(Json(fixture).into_response(), limit)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn releasing_dispatch_evidence_retains_answer_mapping() {
        let mut adapter =
            SglangDecisionAdapter::new(&request(json!("Evidence"), all_questions())).unwrap();
        adapter.release_request();
        assert!(adapter.request().is_null());
        let response = adapter
            .convert_response(Json(upstream()).into_response(), 100_000)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let answers = body(response).await["answers"].clone();
        assert_eq!(answers[0]["name"], "same");
        assert_eq!(answers[1]["choice"], "true");
        assert_eq!(answers[2]["probabilities"][1]["label"], "high");
    }
}
