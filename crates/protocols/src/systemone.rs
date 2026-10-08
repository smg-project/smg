//! System One HTTP API types for `/v1/systemone`.
//!
//! State is shared by named questions, and answers retain those names. Native
//! backend extensions and the insertion order of questions and criteria are
//! preserved when proxying requests.
//!
//! Reference: <https://api.typesafe.ai/openapi.json>

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::{Validate, ValidationError, ValidationErrors};

use crate::{common::GenerationRequest, validated::Normalizable};

/// Text or structured JSON used as state, instructions, or a criterion.
///
/// Objects and arrays may contain any JSON values. Top-level booleans, numbers,
/// and null are not content; nullable fields wrap this type in `Option`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum SystemOneContent {
    Text(String),
    Object(Map<String, Value>),
    Array(Vec<Value>),
}

/// A non-streaming request containing shared state and named questions.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SystemOneRequest {
    /// Required on the HTTP wire, even when a client SDK supplies a default.
    pub model: String,
    pub state: SystemOneContent,
    #[schemars(
        with = "std::collections::BTreeMap<String, SystemOneQuestion>",
        extend("minProperties" = 1)
    )]
    pub questions: IndexMap<String, SystemOneQuestion>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for SystemOneRequest {}

impl GenerationRequest for SystemOneRequest {
    fn is_stream(&self) -> bool {
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        match &self.state {
            SystemOneContent::Text(text) => text.clone(),
            state => serde_json::to_string(state).unwrap_or_default(),
        }
    }
}

/// The question key is its caller-selected name; it is not part of the question.
///
/// Optional nullable fields use two options to distinguish omission (`None`)
/// from explicit JSON null (`Some(None)`), preserving both on the wire.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemOneQuestion {
    Noul {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            with = "serde_with::rust::double_option"
        )]
        #[schemars(with = "Option<SystemOneContent>")]
        instructions: Option<Option<SystemOneContent>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            with = "serde_with::rust::double_option"
        )]
        #[schemars(with = "Option<SystemOneNoulCriteria>")]
        criteria: Option<Option<SystemOneNoulCriteria>>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
    Choice {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            with = "serde_with::rust::double_option"
        )]
        #[schemars(with = "Option<SystemOneContent>")]
        instructions: Option<Option<SystemOneContent>>,
        #[schemars(with = "std::collections::BTreeMap<String, Option<SystemOneContent>>")]
        criteria: IndexMap<String, Option<SystemOneContent>>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
    Score {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            with = "serde_with::rust::double_option"
        )]
        #[schemars(with = "Option<SystemOneContent>")]
        instructions: Option<Option<SystemOneContent>>,
        #[schemars(length(min = 1))]
        criteria: Vec<SystemOneContent>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
}

/// Optional descriptions of the true and false outcomes of a noul question.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SystemOneNoulCriteria {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with::rust::double_option"
    )]
    #[schemars(with = "Option<SystemOneContent>")]
    pub r#true: Option<Option<SystemOneContent>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with::rust::double_option"
    )]
    #[schemars(with = "Option<SystemOneContent>")]
    pub r#false: Option<Option<SystemOneContent>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Validate for SystemOneRequest {
    fn validate(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();
        if self.questions.is_empty() {
            let mut error = ValidationError::new("length");
            error.message = Some("at least one question is required".into());
            errors.add("questions", error);
        }
        for (name, question) in &self.questions {
            if let SystemOneQuestion::Score { criteria, .. } = question {
                if criteria.is_empty() {
                    let mut error = ValidationError::new("length");
                    error.message = Some(
                        format!("question {name:?} requires at least one score criterion").into(),
                    );
                    errors.add("questions", error);
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Answers keyed by the names in the request, with model and token usage.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SystemOneResponse {
    pub model: String,
    #[schemars(with = "std::collections::BTreeMap<String, SystemOneAnswer>")]
    pub answers: IndexMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemOneAnswer {
    Noul {
        /// Probability of a yes answer or a true statement.
        noul: f64,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
    Choice {
        choice: String,
        confidence: f64,
        #[schemars(with = "std::collections::BTreeMap<String, f64>")]
        probabilities: IndexMap<String, f64>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
    Score {
        score: f64,
        confidence: f64,
        #[schemars(with = "std::collections::BTreeMap<String, SystemOneContent>")]
        legend: IndexMap<String, SystemOneContent>,
        #[schemars(with = "std::collections::BTreeMap<String, f64>")]
        probabilities: IndexMap<String, f64>,
        #[serde(flatten)]
        other: Map<String, Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SystemOneUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}
