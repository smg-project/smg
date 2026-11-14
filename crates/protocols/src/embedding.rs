use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::{
    common::{invalid, routing_text_of_input, GenerationRequest, UsageInfo},
    validated::Normalizable,
};

// ============================================================================
// Embedding API
// ============================================================================

/// A request to the embeddings route: the model, the input (a text, texts,
/// token ids, a batch of them, or the serving engine's structured members) and
/// the encoding format of the vectors. Its rules are shape rules: `input` in
/// one of the five shapes `validate_embedding_input` names, `dimensions` at
/// least one. The validated extractor answers a body outside them in the
/// gateway's error envelope before any worker sees it; what a path does with a
/// body that passes is the path's own.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
pub struct EmbeddingRequest {
    /// ID of the model to use
    pub model: String,

    /// A string, an array of strings, an array of token ids, an array of token
    /// id arrays, or an array of objects (the serving engine's structured
    /// members, forwarded unread); no string or array empty.
    #[validate(custom(function = "validate_embedding_input"))]
    pub input: Value,

    /// Optional encoding format of the vectors: `float`, `base64`, or one the
    /// serving engine answers; the gateway does not read it.
    pub encoding_format: Option<String>,

    /// Optional user identifier
    pub user: Option<String>,

    /// Optional number of dimensions for the embedding, at least one
    #[validate(range(min = 1))]
    pub dimensions: Option<u32>,

    /// Engine extension: request id for tracking
    pub rid: Option<String>,

    /// Additional fields not explicitly defined above (e.g. engine-specific parameters)
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

impl Normalizable for EmbeddingRequest {}

/// The shapes the public API takes for an `input`, the classify route's whole
/// rule and the base of the embeddings route's: a string, an array of strings,
/// an array of token ids (non-negative integers), or an array of token id
/// arrays. No string and no array is empty; the array's first member sets its
/// kind, and a member of another kind is refused by its index. The rule reads
/// shape only: how long a batch may be is the serving engine's to bound. The
/// codes are the envelope's existing ones: `empty_array` and `empty_input` for
/// the empty shapes, `integer_below_min_value` for a negative token id,
/// `invalid_type` for every other.
pub(crate) fn validate_input(input: &Value) -> Result<(), validator::ValidationError> {
    validate_input_shapes(input, false)
}

/// The embeddings route's rule: `validate_input`'s four shapes and a fifth the
/// serving engine takes, an array of objects (a structured member each: a text,
/// an image or another medium it embeds), which the gateway forwards without
/// reading a field of them. The first member sets the kind here too: a member
/// after an object that is not one is refused by its index.
pub(crate) fn validate_embedding_input(input: &Value) -> Result<(), validator::ValidationError> {
    validate_input_shapes(input, true)
}

fn validate_input_shapes(
    input: &Value,
    structured_members: bool,
) -> Result<(), validator::ValidationError> {
    let shapes = if structured_members {
        "a string, an array of strings, an array of token ids, an array of token id arrays or an array of objects"
    } else {
        "a string, an array of strings, an array of token ids or an array of token id arrays"
    };
    let members = match input {
        Value::String(text) if text.is_empty() => {
            return Err(invalid(
                "empty_input",
                "Invalid 'input': empty string. Expected a string with minimum length 1, but got an empty string instead.",
            ));
        }
        Value::String(_) => return Ok(()),
        Value::Array(members) => members,
        _ => {
            return Err(invalid(
                "invalid_type",
                format!("Invalid 'input': expected {shapes}."),
            ));
        }
    };
    let Some(first) = members.first() else {
        return Err(invalid(
            "empty_array",
            "Invalid 'input': empty array. Expected an array with minimum length 1, but got an empty array instead.",
        ));
    };
    match first {
        Value::String(_) => {
            for (index, member) in members.iter().enumerate() {
                match member {
                    Value::String(text) if text.is_empty() => {
                        return Err(invalid(
                            "empty_input",
                            format!("Invalid 'input[{index}]': empty string. Expected a string with minimum length 1, but got an empty string instead."),
                        ));
                    }
                    Value::String(_) => {}
                    _ => {
                        return Err(invalid(
                            "invalid_type",
                            format!("Invalid 'input[{index}]': expected a string, as the array's first member is."),
                        ));
                    }
                }
            }
            Ok(())
        }
        Value::Number(_) => {
            if let Some(index) = members.iter().position(|member| member.as_u64().is_none()) {
                if let Some(id) = members[index].as_i64() {
                    return Err(invalid(
                        "integer_below_min_value",
                        format!("Invalid 'input[{index}]': a token id must be non-negative, but got {id} instead."),
                    ));
                }
                let why = if index == 0 {
                    ""
                } else {
                    ", as the array's first member is"
                };
                return Err(invalid(
                    "invalid_type",
                    format!("Invalid 'input[{index}]': expected a token id (a non-negative integer){why}."),
                ));
            }
            Ok(())
        }
        Value::Array(_) => {
            for (index, member) in members.iter().enumerate() {
                let Value::Array(ids) = member else {
                    return Err(invalid(
                        "invalid_type",
                        format!("Invalid 'input[{index}]': expected an array of token ids, as the array's first member is."),
                    ));
                };
                if ids.is_empty() {
                    return Err(invalid(
                        "empty_array",
                        format!("Invalid 'input[{index}]': empty array. Expected an array of token ids with minimum length 1, but got an empty array instead."),
                    ));
                }
                if let Some(position) = ids.iter().position(|id| id.as_u64().is_none()) {
                    if let Some(id) = ids[position].as_i64() {
                        return Err(invalid(
                            "integer_below_min_value",
                            format!("Invalid 'input[{index}][{position}]': a token id must be non-negative, but got {id} instead."),
                        ));
                    }
                    return Err(invalid(
                        "invalid_type",
                        format!("Invalid 'input[{index}][{position}]': expected a token id (a non-negative integer)."),
                    ));
                }
            }
            Ok(())
        }
        Value::Object(_) if structured_members => {
            if let Some(index) = members.iter().position(|member| !member.is_object()) {
                return Err(invalid(
                    "invalid_type",
                    format!("Invalid 'input[{index}]': expected an object, as the array's first member is."),
                ));
            }
            Ok(())
        }
        _ => Err(invalid(
            "invalid_type",
            format!("Invalid 'input[0]': expected {shapes}."),
        )),
    }
}

impl GenerationRequest for EmbeddingRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_deref()
    }

    fn is_stream(&self) -> bool {
        // Embeddings are non-streaming
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        routing_text_of_input(&self.input)
    }
}

/// One embedding of the answer, object `embedding`: the vector and the index of
/// the input it embeds.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingObject {
    pub object: String, // "embedding"
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// The answer of the embeddings route, object `list`: one embedding per input,
/// the model and the usage.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EmbeddingResponse {
    pub object: String, // "list"
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: UsageInfo,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::classify::ClassifyRequest;

    fn request(value: Value) -> EmbeddingRequest {
        serde_json::from_value(value).expect("an embeddings request")
    }

    /// The code of the first violation the rules file under `field`.
    fn code_at(errors: &validator::ValidationErrors, field: &str) -> String {
        let field_errors = errors.field_errors();
        let found = field_errors
            .get(field)
            .unwrap_or_else(|| panic!("a violation under '{field}': {errors:?}"));
        found[0].code.to_string()
    }

    /// The message of the first violation the rules file under `field`.
    fn message_at(errors: &validator::ValidationErrors, field: &str) -> String {
        let field_errors = errors.field_errors();
        let found = field_errors
            .get(field)
            .unwrap_or_else(|| panic!("a violation under '{field}': {errors:?}"));
        found[0].message.as_deref().unwrap_or_default().to_string()
    }

    /// The shapes the public API takes pass, and on this route the serving
    /// engine's array of objects, a long batch among them, with a `dimensions`
    /// of one and any encoding format: the door reads shapes, not the options
    /// the serving engine bounds.
    #[test]
    fn the_shapes_the_public_api_takes_pass() {
        let many: Vec<Value> = vec![json!("x"); 4096];
        for input in [
            json!("hello"),
            json!(["a", "b"]),
            json!([1, 2, 3]),
            json!([[1, 2], [3]]),
            // the serving engine's structured members, a fifth shape on this route
            json!([{"text": "a"}, {"image": "https://example.com/a.jpg"}]),
            Value::Array(many),
        ] {
            let within = request(json!({"model": "m", "input": input}));
            assert!(within.validate().is_ok(), "{}", within.input);
        }
        for options in [
            json!({"dimensions": 1, "encoding_format": "float"}),
            json!({"encoding_format": "base64"}),
            json!({"encoding_format": "hex"}),
        ] {
            let mut body = json!({"model": "m", "input": "hello"});
            body.as_object_mut()
                .expect("an object")
                .extend(options.as_object().expect("an object").clone());
            assert!(request(body.clone()).validate().is_ok(), "{body}");
        }
    }

    /// Every shape the public API refuses is refused under `input` with the
    /// envelope's existing code for it.
    #[test]
    fn the_shapes_the_public_api_refuses_are_refused_under_input() {
        for (input, code) in [
            (json!([]), "empty_array"),
            (json!(""), "empty_input"),
            (json!(["a", ""]), "empty_input"),
            (json!({"text": "x"}), "invalid_type"),
            (json!(null), "invalid_type"),
            (json!(42), "invalid_type"),
            (json!(true), "invalid_type"),
            (json!(["a", 1, null]), "invalid_type"),
            (json!([1, "a"]), "invalid_type"),
            (json!([1, -2]), "integer_below_min_value"),
            (json!([-1]), "integer_below_min_value"),
            (json!([1.5]), "invalid_type"),
            (json!([[1, 2], "a"]), "invalid_type"),
            (json!([[1, 2], [3, -4]]), "integer_below_min_value"),
            (json!([[1, 2], []]), "empty_array"),
            (json!([[1, 2.5]]), "invalid_type"),
            (json!([null]), "invalid_type"),
            (json!([{"text": "a"}, "b"]), "invalid_type"),
            (json!([{"text": "a"}, null]), "invalid_type"),
        ] {
            let refused = request(json!({"model": "m", "input": input}));
            let errors = refused
                .validate()
                .expect_err("a shape the public API refuses is refused");
            assert_eq!(code_at(&errors, "input"), code, "{}", refused.input);
        }
    }

    /// The message names the member at fault by its index; at index 0 it
    /// does not point at "the array's first member", which is that member.
    /// A negative id is an integer below the field's minimum, not a wrong
    /// type: it gets the code and sentence the generate route gives one.
    #[test]
    fn a_bad_token_id_is_named_by_its_index() {
        let first = request(json!({"model": "m", "input": [1.5, 2]}));
        let errors = first
            .validate()
            .expect_err("a fractional token id is refused");
        assert_eq!(
            message_at(&errors, "input"),
            "Invalid 'input[0]': expected a token id (a non-negative integer)."
        );

        let negative = request(json!({"model": "m", "input": [-1, 2]}));
        let errors = negative
            .validate()
            .expect_err("a negative token id is refused");
        assert_eq!(
            message_at(&errors, "input"),
            "Invalid 'input[0]': a token id must be non-negative, but got -1 instead."
        );

        let nested = request(json!({"model": "m", "input": [[1, 2], [3, -4]]}));
        let errors = nested
            .validate()
            .expect_err("a negative token id in a nested array is refused");
        assert_eq!(
            message_at(&errors, "input"),
            "Invalid 'input[1][1]': a token id must be non-negative, but got -4 instead."
        );

        let later = request(json!({"model": "m", "input": [1, 2.5]}));
        let errors = later
            .validate()
            .expect_err("a fractional token id is refused");
        assert_eq!(
            message_at(&errors, "input"),
            "Invalid 'input[1]': expected a token id (a non-negative integer), as the array's first member is."
        );

        let mixed = request(json!({"model": "m", "input": [{"text": "a"}, "b"]}));
        let errors = mixed
            .validate()
            .expect_err("a member after an object that is not one is refused");
        assert_eq!(
            message_at(&errors, "input"),
            "Invalid 'input[1]': expected an object, as the array's first member is."
        );
    }

    /// `dimensions` under one is refused under its own field.
    #[test]
    fn dimensions_is_at_least_one() {
        let zero = request(json!({"model": "m", "input": "hello", "dimensions": 0}));
        let errors = zero.validate().expect_err("dimensions 0 is refused");
        assert_eq!(code_at(&errors, "dimensions"), "range");
    }

    /// The classify route runs the base `input` rule, without the object kind.
    #[test]
    fn the_classify_request_runs_the_base_input_rule() {
        let classify = |input: Value| -> ClassifyRequest {
            serde_json::from_value(json!({"model": "m", "input": input}))
                .expect("a classify request")
        };
        for served in [json!("hello"), json!(["a", "b"]), json!([1, 2, 3])] {
            assert!(classify(served.clone()).validate().is_ok(), "{served}");
        }
        for (input, code) in [
            (json!([]), "empty_array"),
            (json!(""), "empty_input"),
            (json!({"a": 1}), "invalid_type"),
            (json!([{"text": "a"}]), "invalid_type"),
            (json!(["a", 1]), "invalid_type"),
        ] {
            let errors = classify(input.clone())
                .validate()
                .expect_err("a shape the public API refuses is refused");
            assert_eq!(code_at(&errors, "input"), code, "{input}");
        }
    }
}
