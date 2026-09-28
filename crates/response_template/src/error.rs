/// Why a `response_template` cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported response_template ({field}): {reason}")]
pub struct TemplateError {
    /// The template key or field at fault.
    pub field: String,
    /// What is wrong with it.
    pub reason: String,
}

pub(crate) fn invalid(field: &str, reason: impl Into<String>) -> TemplateError {
    TemplateError {
        field: field.to_string(),
        reason: reason.into(),
    }
}
