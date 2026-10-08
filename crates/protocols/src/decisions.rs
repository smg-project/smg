//! OpenAI Decisions API request and response types for `/v1/decisions`.
//!
//! Questions share the same input and answers retain question order. Native
//! backend extensions are preserved on every request object.
//!
//! Reference: <https://developers.openai.com/api/reference/resources/decisions/methods/create>

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use validator::{Validate, ValidationError, ValidationErrors};

use crate::{common::GenerationRequest, validated::Normalizable};

/// A non-streaming request to classify or score shared text/image evidence.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionsRequest {
    pub model: String,
    pub input: DecisionInput,
    pub questions: Vec<DecisionQuestion>,
    pub safety_identifier: Option<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for DecisionsRequest {}

impl GenerationRequest for DecisionsRequest {
    fn is_stream(&self) -> bool {
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        self.input.text()
    }
}

/// Evidence supplied as plain text or user messages.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum DecisionInput {
    Text(String),
    Messages(Vec<DecisionInputMessage>),
}

impl DecisionInput {
    /// Text in evidence order, excluding image data.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Messages(messages) => {
                let mut text = String::new();
                for message in messages {
                    match &message.content {
                        DecisionInputContent::Text(content) => append_text(&mut text, content),
                        DecisionInputContent::Parts(parts) => {
                            for part in parts {
                                if let DecisionInputPart::InputText { text: content, .. } = part {
                                    append_text(&mut text, content);
                                }
                            }
                        }
                    }
                }
                text
            }
        }
    }
}

fn append_text(buffer: &mut String, text: &str) {
    if !text.is_empty() {
        if !buffer.is_empty() {
            buffer.push(' ');
        }
        buffer.push_str(text);
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionInputMessage {
    pub role: DecisionInputRole,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    #[schemars(with = "DecisionMessageType")]
    pub r#type: Option<DecisionMessageType>,
    pub content: DecisionInputContent,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// Decisions only accepts user messages.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecisionInputRole {
    User,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMessageType {
    Message,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum DecisionInputContent {
    Text(String),
    Parts(Vec<DecisionInputPart>),
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionInputPart {
    InputText {
        text: String,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
    InputImage {
        /// Inline base64 image data URL. External URLs and file IDs are unsupported.
        image_url: String,
        detail: Option<DecisionImageDetail>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecisionImageDetail {
    Low,
    High,
    Auto,
    Original,
}

/// Typed choice values: boolean `true` differs from the string `"true"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum DecisionValue {
    String(String),
    Boolean(bool),
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionQuestion {
    Predicate {
        instructions: String,
        #[serde(default, deserialize_with = "deserialize_optional_non_null")]
        #[schemars(with = "String")]
        name: Option<String>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
    Choice {
        instructions: String,
        #[serde(default, deserialize_with = "deserialize_optional_non_null")]
        #[schemars(with = "String")]
        name: Option<String>,
        choices: Vec<DecisionChoice>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
    Score {
        instructions: String,
        #[serde(default, deserialize_with = "deserialize_optional_non_null")]
        #[schemars(with = "String")]
        name: Option<String>,
        levels: Vec<DecisionScoreLevel>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
}

impl DecisionQuestion {
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Predicate { name, .. } | Self::Choice { name, .. } | Self::Score { name, .. } => {
                name.as_deref()
            }
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Self::Predicate { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionChoice {
    pub value: DecisionValue,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    #[schemars(with = "String")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionScoreLevel {
    pub label: String,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    #[schemars(with = "String")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

// Optional non-null fields must reject explicit null while accepting omission.
fn deserialize_optional_non_null<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

impl Validate for DecisionsRequest {
    fn validate(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        if let Err(error) = check_length(&self.model, 1_048_576) {
            errors.add("model", error);
        }
        if let Some(identifier) = &self.safety_identifier {
            if let Err(error) = check_length(identifier, 128) {
                errors.add("safety_identifier", error);
            }
        }
        if let Err(error) = validate_input(&self.input) {
            errors.add("input", error);
        }
        if let Err(error) = validate_questions(&self.questions) {
            errors.add("questions", error);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn check_length(value: &str, max: usize) -> Result<(), ValidationError> {
    if value.len() > max && value.chars().take(max + 1).count() > max {
        let mut error = ValidationError::new("length");
        error.message = Some(format!("must be at most {max} characters").into());
        return Err(error);
    }
    Ok(())
}

fn validate_input(input: &DecisionInput) -> Result<(), ValidationError> {
    let DecisionInput::Messages(messages) = input else {
        return Ok(());
    };
    let mut image_count = 0;
    for message in messages {
        if let DecisionInputContent::Parts(parts) = &message.content {
            for part in parts {
                match part {
                    DecisionInputPart::InputText { text, .. } => {
                        check_length(text, 10_485_760)?;
                    }
                    DecisionInputPart::InputImage { image_url, .. } => {
                        image_count += 1;
                        if image_count > 128 {
                            let mut error = ValidationError::new("too_many_images");
                            error.message =
                                Some("at most 128 images are allowed per request".into());
                            return Err(error);
                        }
                        check_length(image_url, 1_073_741_824)?;
                        let valid = image_url.split_once(',').is_some_and(|(header, data)| {
                            !data.is_empty()
                                && header.strip_prefix("data:image/").is_some_and(|format| {
                                    format
                                        .strip_suffix(";base64")
                                        .is_some_and(|media_type| !media_type.is_empty())
                                })
                        });
                        if !valid {
                            let mut error = ValidationError::new("invalid_image_url");
                            error.message =
                                Some("images must be inline base64 image data URLs".into());
                            return Err(error);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_questions(questions: &[DecisionQuestion]) -> Result<(), ValidationError> {
    for question in questions {
        check_length(question.instructions(), 1_048_576)?;
        if let Some(name) = question.name() {
            check_length(name, 1_048_576)?;
        }
        match question {
            DecisionQuestion::Predicate { .. } => {}
            DecisionQuestion::Choice { choices, .. } => {
                for choice in choices {
                    if let Some(description) = &choice.description {
                        check_length(description, 1_048_576)?;
                    }
                }
            }
            DecisionQuestion::Score { levels, .. } => {
                for level in levels {
                    check_length(&level.label, 1_048_576)?;
                    if let Some(description) = &level.description {
                        check_length(description, 1_048_576)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Ordered answers and usage returned by the Decisions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionUsage,
}

/// An answer's name is always serialized, including `null` for unnamed questions.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    Predicate {
        #[serde(deserialize_with = "Option::deserialize")]
        #[schemars(schema_with = "nullable_name_schema")]
        name: Option<String>,
        probability: f64,
    },
    Choice {
        #[serde(deserialize_with = "Option::deserialize")]
        #[schemars(schema_with = "nullable_name_schema")]
        name: Option<String>,
        choice: DecisionValue,
        confidence: f64,
        probabilities: Vec<DecisionChoiceProbability>,
    },
    Score {
        #[serde(deserialize_with = "Option::deserialize")]
        #[schemars(schema_with = "nullable_name_schema")]
        name: Option<String>,
        score: f64,
        confidence: f64,
        probabilities: Vec<DecisionScoreProbability>,
    },
    Refusal {
        #[serde(deserialize_with = "Option::deserialize")]
        #[schemars(schema_with = "nullable_name_schema")]
        name: Option<String>,
    },
}

// A custom field schema retains nullability without treating the field as optional.
fn nullable_name_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    generator.subschema_for::<Option<String>>()
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionChoiceProbability {
    pub value: DecisionValue,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionScoreProbability {
    pub value: i64,
    pub label: String,
    pub probability: f64,
}

/// Decisions usage includes cache writes, unlike the Responses usage schema.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub input_tokens_details: DecisionInputTokensDetails,
    pub output_tokens_details: DecisionOutputTokensDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionInputTokensDetails {
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecisionOutputTokensDetails {
    pub reasoning_tokens: u64,
}
