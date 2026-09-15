use serde::{Deserialize, Serialize};

/// Public, actionable response-template failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseTemplateError {
    #[error("invalid response template for {model_name} field {field} (limit {limit}): {reason}")]
    InvalidTemplate {
        model_name: String,
        field: String,
        limit: usize,
        reason: String,
    },
    #[error(
        "response-template parse failure for {model_name} field {field} (limit {limit}): {reason}"
    )]
    RuntimeFailure {
        model_name: String,
        field: String,
        limit: usize,
        reason: String,
    },
    #[error("response-template pending state overflow for {model_name} field {field}: limit {limit} bytes")]
    PendingOverflow {
        model_name: String,
        field: String,
        limit: usize,
    },
    #[error("response-template structured field overflow for {model_name} field {field}: limit {limit} bytes")]
    StructuredFieldOverflow {
        model_name: String,
        field: String,
        limit: usize,
    },
}

impl ResponseTemplateError {
    pub fn model_name(&self) -> &str {
        match self {
            Self::InvalidTemplate { model_name, .. }
            | Self::RuntimeFailure { model_name, .. }
            | Self::PendingOverflow { model_name, .. }
            | Self::StructuredFieldOverflow { model_name, .. } => model_name,
        }
    }

    pub fn field(&self) -> &str {
        match self {
            Self::InvalidTemplate { field, .. }
            | Self::RuntimeFailure { field, .. }
            | Self::PendingOverflow { field, .. }
            | Self::StructuredFieldOverflow { field, .. } => field,
        }
    }

    pub fn limit(&self) -> usize {
        match self {
            Self::InvalidTemplate { limit, .. }
            | Self::RuntimeFailure { limit, .. }
            | Self::PendingOverflow { limit, .. }
            | Self::StructuredFieldOverflow { limit, .. } => *limit,
        }
    }

    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::InvalidTemplate { .. } => "InvalidTemplate",
            Self::RuntimeFailure { .. } => "RuntimeFailure",
            Self::PendingOverflow { .. } => "PendingOverflow",
            Self::StructuredFieldOverflow { .. } => "StructuredFieldOverflow",
        }
    }
}
