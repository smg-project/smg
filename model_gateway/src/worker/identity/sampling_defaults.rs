//! Backend-reported sampling defaults carried on worker labels.

use serde::{Deserialize, Serialize};

/// Worker label carrying backend-reported model sampling defaults as JSON.
pub const DEFAULT_SAMPLING_PARAMS_LABEL: &str = "default_sampling_params_json";

/// Model-author sampling defaults reported by a backend.
///
/// This intentionally covers only knobs that are true sampling defaults. Length
/// limits such as `max_new_tokens` are ignored even when present in a model's
/// `generation_config.json`.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub(crate) struct SamplingDefaults {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.min_p.is_none()
            && self.repetition_penalty.is_none()
    }

    pub(crate) fn from_json_str(json: &str) -> Result<Option<Self>, serde_json::Error> {
        let defaults: Self = serde_json::from_str(json)?;
        Ok((!defaults.is_empty()).then_some(defaults))
    }

    pub(crate) fn canonical_json_from_str(json: &str) -> Result<Option<String>, String> {
        let Some(defaults) = Self::from_json_str(json)
            .map_err(|e| format!("failed to parse sampling defaults JSON: {e}"))?
        else {
            return Ok(None);
        };

        serde_json::to_string(&defaults)
            .map(Some)
            .map_err(|e| format!("failed to serialize sampling defaults JSON: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::SamplingDefaults;

    #[test]
    fn canonical_json_filters_unknown_and_null_fields() {
        let canonical = SamplingDefaults::canonical_json_from_str(
            r#"{"temperature":0.6,"top_k":0,"min_p":null,"max_new_tokens":32}"#,
        )
        .unwrap()
        .unwrap();

        assert_eq!(canonical, r#"{"temperature":0.6,"top_k":0}"#);
    }

    #[test]
    fn canonical_json_removes_empty_defaults() {
        let canonical =
            SamplingDefaults::canonical_json_from_str(r#"{"min_p":null,"max_new_tokens":32}"#)
                .unwrap();

        assert_eq!(canonical, None);
    }

    #[test]
    fn canonical_json_rejects_invalid_default_types() {
        let err = SamplingDefaults::canonical_json_from_str(r#"{"top_k":1.5}"#).unwrap_err();

        assert!(err.contains("failed to parse sampling defaults JSON"));
    }
}
