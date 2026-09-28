use serde::{Deserialize, Serialize};

/// Public, actionable response-template failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseTemplateError {
    /// The template failed validation or compilation.
    #[error("invalid response template for {model_name} field {field} (limit {limit}): {reason}")]
    InvalidTemplate {
        /// Model name given to the parser.
        model_name: String,
        /// Template field or key that is invalid.
        field: String,
        /// Byte limit involved, or 0.
        limit: usize,
        /// Why the template is invalid.
        reason: String,
    },
    /// The output does not follow the template.
    #[error(
        "response-template parse failure for {model_name} field {field} (limit {limit}): {reason}"
    )]
    RuntimeFailure {
        /// Model name given to the parser.
        model_name: String,
        /// Field or check that failed.
        field: String,
        /// Byte limit involved, or 0.
        limit: usize,
        /// Why parsing failed.
        reason: String,
    },
    /// Pending delimiter state exceeded `max_pending_bytes`.
    #[error("response-template pending state overflow for {model_name} field {field}: limit {limit} bytes")]
    PendingOverflow {
        /// Model name given to the parser.
        model_name: String,
        /// Name of the exceeded limit.
        field: String,
        /// Value of the exceeded limit, in bytes.
        limit: usize,
    },
    /// A field body or structured value exceeded its byte limit.
    #[error("response-template structured field overflow for {model_name} field {field}: limit {limit} bytes")]
    StructuredFieldOverflow {
        /// Model name given to the parser.
        model_name: String,
        /// Name of the exceeded limit.
        field: String,
        /// Value of the exceeded limit, in bytes.
        limit: usize,
    },
}

impl ResponseTemplateError {
    /// Model name given to the parser.
    pub fn model_name(&self) -> &str {
        match self {
            Self::InvalidTemplate { model_name, .. }
            | Self::RuntimeFailure { model_name, .. }
            | Self::PendingOverflow { model_name, .. }
            | Self::StructuredFieldOverflow { model_name, .. } => model_name,
        }
    }

    /// Field, check or limit that the error refers to.
    pub fn field(&self) -> &str {
        match self {
            Self::InvalidTemplate { field, .. }
            | Self::RuntimeFailure { field, .. }
            | Self::PendingOverflow { field, .. }
            | Self::StructuredFieldOverflow { field, .. } => field,
        }
    }

    /// Byte limit reported with the error, or 0.
    pub fn limit(&self) -> usize {
        match self {
            Self::InvalidTemplate { limit, .. }
            | Self::RuntimeFailure { limit, .. }
            | Self::PendingOverflow { limit, .. }
            | Self::StructuredFieldOverflow { limit, .. } => *limit,
        }
    }

    /// Name of the error variant, such as `PendingOverflow`.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::InvalidTemplate { .. } => "InvalidTemplate",
            Self::RuntimeFailure { .. } => "RuntimeFailure",
            Self::PendingOverflow { .. } => "PendingOverflow",
            Self::StructuredFieldOverflow { .. } => "StructuredFieldOverflow",
        }
    }
}
