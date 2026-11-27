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

use crate::{
    observability::metrics::error_reason_from_status,
    routers::{common::attempt_ledger::UpstreamAnswer, error, grpc::utils::mark_error_reason},
};

pub(crate) const UPSTREAM_ROUTE: &str = "/v1/systemone";

#[derive(Clone)]
pub(crate) struct SglangDecisionAdapter {
    request: Value,
    questions: Vec<MappedQuestion>,
}

#[derive(Clone)]
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
                    // A worker's own answer (not the gateway's failed send or
                    // body read), re-minted under its code or the relay's: the
                    // error counter's reason is the worker's status.
                    if parts.extensions.get::<UpstreamAnswer>().is_some() {
                        mark_error_reason(&mut response, error_reason_from_status(parts.status));
                    }
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

    pub(crate) fn map_response(&self, value: &Value) -> Result<DecisionResponse, String> {
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
#[path = "../../../tests/unit/routers/common/decisions/tests.rs"]
mod tests;
