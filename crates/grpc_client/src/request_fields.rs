//! The parts of a request every engine client reads alike: the constraint a
//! chat request's `response_format` asks for, and a plain generate request's sampling
//! parameters with the shared defaults applied, and the one structured constraint
//! a completion or plain request names. Each client places the values in its own
//! proto; what differs between engines (the constraint variant, how a disabled
//! `top_k` is spelled, which fields its proto has) stays in the client.

use std::collections::HashMap;

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, StringOrArray},
    completion::CompletionRequest,
    sampling_params::SamplingParams as GenerateSamplingParams,
};

/// The constraint a chat request's `response_format` puts on the output, with
/// its text: a JSON schema (`{"type":"object"}` for `json_object`, the given
/// schema for `json_schema`), or a structural tag (the object as it arrived,
/// `type` included, which is what the engines' own servers hand their grammar
/// compiler); nothing for `text` or none.
pub fn response_format_constraint(
    request: &ChatCompletionRequest,
) -> Result<Option<(ConstraintKind, String)>, String> {
    let (kind, text) = match &request.response_format {
        Some(ResponseFormat::JsonObject) => (
            ConstraintKind::JsonSchema,
            serde_json::to_string(&serde_json::json!({"type": "object"})),
        ),
        Some(ResponseFormat::JsonSchema { json_schema }) => (
            ConstraintKind::JsonSchema,
            serde_json::to_string(&json_schema.schema),
        ),
        Some(format @ ResponseFormat::StructuralTag { .. }) => {
            (ConstraintKind::StructuralTag, serde_json::to_string(format))
        }
        Some(ResponseFormat::Text) | None => return Ok(None),
    };
    text.map(|text| Some((kind, text)))
        .map_err(|e| format!("Failed to serialize the response_format constraint: {e}"))
}

/// Which of a chat body's own format fields request building set aside: the
/// tool constraint it hands the builder answers them (a table's tag stands in
/// for a JSON `response_format`; the whole turn's grammar for the `regex` and
/// the `ebnf` as well), so the engine's request carries that constraint alone
/// and no client weighs the two against each other. A field set aside reads as
/// absent, which is what each engine's precedence between the body's format
/// and a tool constraint sees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FormatSetAside {
    /// The constraint answers the `response_format`.
    pub response_format: bool,
    /// The constraint answers the `regex` and the `ebnf`.
    pub grammar: bool,
}

/// What constrains the output beside the body's own format fields: the tool
/// constraint request building shaped for the worker (its kind and its text),
/// if any, and which of the body's own format fields it answers. A Messages
/// body names no format of its own, so its builders read the tool constraint
/// alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputConstraint {
    pub tool: Option<(String, String)>,
    pub set_aside: FormatSetAside,
}

impl OutputConstraint {
    /// A tool constraint that answers none of the body's own format fields.
    pub fn tool(tool: Option<(String, String)>) -> Self {
        Self {
            tool,
            set_aside: FormatSetAside::default(),
        }
    }
}

/// A chat body's own format constraints as the engine's request reads them:
/// the constraint its `response_format` asks for
/// ([`response_format_constraint`]), its `ebnf` and its `regex`, each absent
/// when set aside.
pub struct BodyFormat<'a> {
    pub format: Option<(ConstraintKind, String)>,
    pub ebnf: Option<&'a str>,
    pub regex: Option<&'a str>,
}

/// The body's own format constraints less what `set_aside` names.
pub fn body_format(
    request: &ChatCompletionRequest,
    set_aside: FormatSetAside,
) -> Result<BodyFormat<'_>, String> {
    Ok(BodyFormat {
        format: if set_aside.response_format {
            None
        } else {
            response_format_constraint(request)?
        },
        ebnf: request.ebnf.as_deref().filter(|_| !set_aside.grammar),
        regex: request.regex.as_deref().filter(|_| !set_aside.grammar),
    })
}

/// The stop sequences as the engines take them: one string or several, none
/// when the request has no `stop`.
pub fn stop_strings(stop: Option<&StringOrArray>) -> Vec<String> {
    stop.map_or_else(Vec::new, StringOrArray::to_vec)
}

/// A request's `logit_bias` for a proto keyed by token id: the keys parsed
/// (request validation proved them non-negative integers), empty when the
/// request has none.
pub fn logit_bias_by_token_id(bias: Option<&HashMap<String, f32>>) -> HashMap<i32, f32> {
    bias.map_or_else(HashMap::new, |map| {
        map.iter()
            .filter_map(|(key, bias)| key.parse().ok().map(|id| (id, *bias)))
            .collect()
    })
}

/// A request's `logit_bias` for a proto keyed by the token id's text: the map
/// as sent, empty when the request has none.
pub fn logit_bias_by_token_text(bias: Option<&HashMap<String, f32>>) -> HashMap<String, f32> {
    bias.map_or_else(HashMap::new, HashMap::clone)
}

/// The kind of structured constraint a request names: a JSON schema, a regex,
/// an EBNF grammar or a structural tag. Each client spells it in its own proto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    JsonSchema,
    Regex,
    Ebnf,
    StructuralTag,
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
    fn response_format_constraint_per_kind() {
        assert_eq!(
            response_format_constraint(&chat(serde_json::json!({}))),
            Ok(None)
        );
        assert_eq!(
            response_format_constraint(&chat(
                serde_json::json!({"response_format": {"type": "text"}})
            )),
            Ok(None)
        );
        assert_eq!(
            response_format_constraint(&chat(
                serde_json::json!({"response_format": {"type": "json_object"}})
            )),
            Ok(Some((
                ConstraintKind::JsonSchema,
                r#"{"type":"object"}"#.to_string()
            )))
        );
        let schema =
            serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        assert_eq!(
            response_format_constraint(&chat(serde_json::json!({
                "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}
            }))),
            Ok(Some((
                ConstraintKind::JsonSchema,
                serde_json::to_string(&schema).unwrap()
            )))
        );
        // a structural tag is the object as it arrived, `type` included
        let tag = serde_json::json!({
            "type": "structural_tag",
            "structures": [{"begin": "<w>", "schema": schema, "end": "</w>"}],
            "triggers": ["<w>"]
        });
        assert_eq!(
            response_format_constraint(&chat(serde_json::json!({"response_format": tag}))),
            Ok(Some((
                ConstraintKind::StructuralTag,
                serde_json::to_string(&tag).unwrap()
            )))
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
    fn a_format_field_set_aside_reads_as_absent() {
        let request = chat(serde_json::json!({
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}},
            "regex": "[a-z]+",
            "ebnf": "root ::= \"x\""
        }));

        let all = body_format(&request, FormatSetAside::default()).expect("read");
        assert_eq!(
            all.format,
            Some((
                ConstraintKind::JsonSchema,
                r#"{"type":"object"}"#.to_string()
            ))
        );
        assert_eq!(
            (all.ebnf, all.regex),
            (Some("root ::= \"x\""), Some("[a-z]+"))
        );

        let answered = body_format(
            &request,
            FormatSetAside {
                response_format: true,
                grammar: false,
            },
        )
        .expect("read");
        assert_eq!(
            answered.format, None,
            "the schema the tag answers is not read"
        );
        assert_eq!(
            (answered.ebnf, answered.regex),
            (Some("root ::= \"x\""), Some("[a-z]+"))
        );

        let whole_turn = body_format(
            &request,
            FormatSetAside {
                response_format: true,
                grammar: true,
            },
        )
        .expect("read");
        assert!(
            whole_turn.format.is_none() && whole_turn.ebnf.is_none() && whole_turn.regex.is_none()
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
