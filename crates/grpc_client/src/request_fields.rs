//! The parts of a request every engine client reads alike: the schema a chat
//! request's `response_format` asks for, and a plain generate request's sampling
//! parameters with the shared defaults applied, and the one structured constraint
//! a completion or plain request names. Each client places the values in its own
//! proto; what differs between engines (the constraint variant, how a disabled
//! `top_k` is spelled, which fields its proto has) stays in the client.

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, StringOrArray},
    completion::CompletionRequest,
    sampling_params::SamplingParams as GenerateSamplingParams,
};

/// The JSON schema a chat request's `response_format` constrains the output to:
/// `{"type":"object"}` for `json_object`, the given schema for `json_schema`,
/// nothing for `text` or none.
pub fn response_format_schema(request: &ChatCompletionRequest) -> Result<Option<String>, String> {
    let json_object;
    let schema = match &request.response_format {
        Some(ResponseFormat::JsonObject) => {
            json_object = serde_json::json!({"type": "object"});
            &json_object
        }
        Some(ResponseFormat::JsonSchema { json_schema }) => &json_schema.schema,
        Some(ResponseFormat::Text) | None => return Ok(None),
    };
    serde_json::to_string(schema)
        .map(Some)
        .map_err(|e| format!("Failed to serialize JSON schema: {e}"))
}

/// The stop sequences as the engines take them: one string or several, none
/// when the request has no `stop`.
pub fn stop_strings(stop: Option<&StringOrArray>) -> Vec<String> {
    stop.map_or_else(Vec::new, StringOrArray::to_vec)
}

/// The kind of structured constraint a request names: a JSON schema, a regex
/// or an EBNF grammar. Each client spells it in its own proto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    JsonSchema,
    Regex,
    Ebnf,
}

/// The one structured constraint among `json_schema`, `regex` and `ebnf`, with
/// its text: none when the request names none, the refusal when it names more
/// than one.
pub fn single_constraint(
    json_schema: Option<&str>,
    regex: Option<&str>,
    ebnf: Option<&str>,
) -> Result<Option<(ConstraintKind, String)>, String> {
    let mut named = [
        (ConstraintKind::JsonSchema, json_schema),
        (ConstraintKind::Regex, regex),
        (ConstraintKind::Ebnf, ebnf),
    ]
    .into_iter()
    .filter_map(|(kind, value)| value.map(|value| (kind, value.to_string())));
    match (named.next(), named.next()) {
        (None, _) => Ok(None),
        (Some(one), None) => Ok(Some(one)),
        (Some(_), Some(_)) => Err("Multiple structured constraints are not allowed".to_string()),
    }
}

/// A completion request's structured constraint, by [`single_constraint`].
pub fn completion_constraint(
    request: &CompletionRequest,
) -> Result<Option<(ConstraintKind, String)>, String> {
    single_constraint(
        request.json_schema.as_deref(),
        request.regex.as_deref(),
        request.ebnf.as_deref(),
    )
}

/// A plain generate request's structured constraint, by [`single_constraint`].
pub fn plain_constraint(
    params: &GenerateSamplingParams,
) -> Result<Option<(ConstraintKind, String)>, String> {
    single_constraint(
        params.json_schema.as_deref(),
        params.regex.as_deref(),
        params.ebnf.as_deref(),
    )
}

/// A plain generate request's sampling parameters as every engine client reads
/// them: the shared defaults applied, the stop sequences flattened. Fields an
/// engine's proto keeps optional stay optional here, so an absent value is still
/// absent on the wire; `top_k` stays as given because each engine spells
/// "disabled" its own way.
#[derive(Debug, Clone, PartialEq)]
pub struct PlainSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<i32>,
    pub repetition_penalty: f32,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub min_p: Option<f32>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub no_stop_trim: bool,
    pub stop: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub max_new_tokens: Option<u32>,
    pub min_new_tokens: Option<u32>,
    pub n: u32,
    pub sampling_seed: Option<u64>,
}

impl PlainSampling {
    /// The defaults when the request names no value: neutral sampling, one
    /// sequence, special tokens skipped.
    pub fn read(params: Option<&GenerateSamplingParams>) -> Self {
        let Some(p) = params else {
            return Self::default();
        };
        Self {
            temperature: p.temperature.unwrap_or(1.0),
            top_p: p.top_p.unwrap_or(1.0),
            top_k: p.top_k,
            repetition_penalty: p.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: p.frequency_penalty,
            presence_penalty: p.presence_penalty,
            min_p: p.min_p,
            ignore_eos: p.ignore_eos.unwrap_or(false),
            skip_special_tokens: p.skip_special_tokens.unwrap_or(true),
            no_stop_trim: p.no_stop_trim.unwrap_or(false),
            stop: stop_strings(p.stop.as_ref()),
            stop_token_ids: p.stop_token_ids.clone().unwrap_or_default(),
            max_new_tokens: p.max_new_tokens,
            min_new_tokens: p.min_new_tokens,
            n: p.n.unwrap_or(1),
            sampling_seed: p.sampling_seed,
        }
    }
}

impl Default for PlainSampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            frequency_penalty: None,
            presence_penalty: None,
            min_p: None,
            ignore_eos: false,
            skip_special_tokens: true,
            no_stop_trim: false,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            max_new_tokens: None,
            min_new_tokens: None,
            n: 1,
            sampling_seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(extra: serde_json::Value) -> ChatCompletionRequest {
        let mut body =
            serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn response_format_schema_per_kind() {
        assert_eq!(
            response_format_schema(&chat(serde_json::json!({}))),
            Ok(None)
        );
        assert_eq!(
            response_format_schema(&chat(
                serde_json::json!({"response_format": {"type": "text"}})
            )),
            Ok(None)
        );
        assert_eq!(
            response_format_schema(&chat(
                serde_json::json!({"response_format": {"type": "json_object"}})
            )),
            Ok(Some(r#"{"type":"object"}"#.to_string()))
        );
        let schema =
            serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        assert_eq!(
            response_format_schema(&chat(serde_json::json!({
                "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}
            }))),
            Ok(Some(serde_json::to_string(&schema).unwrap()))
        );
    }

    #[test]
    fn plain_sampling_defaults_and_values() {
        assert_eq!(PlainSampling::read(None), PlainSampling::default());
        let empty: GenerateSamplingParams = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(PlainSampling::read(Some(&empty)), PlainSampling::default());

        let full: GenerateSamplingParams = serde_json::from_value(serde_json::json!({
            "temperature": 0.7, "top_p": 0.9, "top_k": -1, "repetition_penalty": 1.2,
            "frequency_penalty": 0.5, "presence_penalty": -0.5, "min_p": 0.05,
            "ignore_eos": true, "skip_special_tokens": false, "no_stop_trim": true,
            "stop": ["a", "b"], "stop_token_ids": [1, 2], "max_new_tokens": 32,
            "min_new_tokens": 2, "n": 2, "sampling_seed": 7
        }))
        .unwrap();
        assert_eq!(
            PlainSampling::read(Some(&full)),
            PlainSampling {
                temperature: 0.7,
                top_p: 0.9,
                top_k: Some(-1),
                repetition_penalty: 1.2,
                frequency_penalty: Some(0.5),
                presence_penalty: Some(-0.5),
                min_p: Some(0.05),
                ignore_eos: true,
                skip_special_tokens: false,
                no_stop_trim: true,
                stop: vec!["a".to_string(), "b".to_string()],
                stop_token_ids: vec![1, 2],
                max_new_tokens: Some(32),
                min_new_tokens: Some(2),
                n: 2,
                sampling_seed: Some(7),
            }
        );
        let one: GenerateSamplingParams =
            serde_json::from_value(serde_json::json!({"stop": "END"})).unwrap();
        assert_eq!(
            PlainSampling::read(Some(&one)).stop,
            vec!["END".to_string()]
        );
    }

    #[test]
    fn the_structured_constraint_is_one_of_three() {
        assert_eq!(single_constraint(None, None, None), Ok(None));
        assert_eq!(
            single_constraint(Some("{}"), None, None),
            Ok(Some((ConstraintKind::JsonSchema, "{}".to_string())))
        );
        assert_eq!(
            single_constraint(None, Some("[a-z]+"), None),
            Ok(Some((ConstraintKind::Regex, "[a-z]+".to_string())))
        );
        assert_eq!(
            single_constraint(None, None, Some("root ::= \"a\"")),
            Ok(Some((ConstraintKind::Ebnf, "root ::= \"a\"".to_string())))
        );
        let refused: Result<Option<(ConstraintKind, String)>, String> =
            Err("Multiple structured constraints are not allowed".to_string());
        assert_eq!(single_constraint(Some("{}"), Some("a"), None), refused);
        assert_eq!(single_constraint(Some("{}"), None, Some("g")), refused);
        assert_eq!(single_constraint(None, Some("a"), Some("g")), refused);
        assert_eq!(single_constraint(Some("{}"), Some("a"), Some("g")), refused);
    }

    #[test]
    fn the_completion_and_plain_readers_read_each_of_the_three_fields() {
        for (field, kind) in [
            ("json_schema", ConstraintKind::JsonSchema),
            ("regex", ConstraintKind::Regex),
            ("ebnf", ConstraintKind::Ebnf),
        ] {
            let completion: CompletionRequest = serde_json::from_value(
                serde_json::json!({"model": "m", "prompt": "hi", field: "v"}),
            )
            .unwrap();
            let plain: GenerateSamplingParams =
                serde_json::from_value(serde_json::json!({field: "v"})).unwrap();
            let read = Ok(Some((kind, "v".to_string())));
            assert_eq!(completion_constraint(&completion), read, "{field}");
            assert_eq!(plain_constraint(&plain), read, "{field}");
        }
    }
}
