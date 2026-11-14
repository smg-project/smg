//! OpenAI-compatible `/v1/models` response types.
//!
//! Provides [`ListModelsResponse`] built from in-memory [`ModelCard`] data,
//! ensuring every router returns the same format.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{model_card::ModelCard, worker::ProviderType};

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelObject {
    /// Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix timestamp of model creation (0 = unknown).
    pub created: i64,
    /// Who owns/hosts the model.
    pub owned_by: String,
    /// Engine-provided extension fields, preserved at the top level.
    #[serde(default, flatten)]
    pub extensions: Map<String, Value>,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ListModelsResponse {
    /// Always `"list"`.
    pub object: String,
    /// Available models.
    pub data: Vec<ModelObject>,
}

impl ListModelsResponse {
    /// Build a response from an iterator of [`ModelCard`]s.
    ///
    /// Deduplicates by model ID (first occurrence wins).
    pub fn from_model_cards(cards: impl IntoIterator<Item = ModelCard>) -> Self {
        let mut seen = HashSet::new();
        let data = cards
            .into_iter()
            .filter(|card| seen.insert(card.id.clone()))
            .map(ModelCard::into_model_object)
            .collect();
        Self {
            object: "list".to_owned(),
            data,
        }
    }

    /// Parse an upstream `/v1/models` JSON response into [`ModelCard`]s.
    ///
    /// Handles both OpenAI and Anthropic response schemas — both use
    /// `data[].id` for the model identifier. Returns an empty vec if
    /// the JSON does not contain a valid `data` array.
    pub fn parse_upstream(json: &Value, provider: Option<ProviderType>) -> Vec<ModelCard> {
        let Some(data) = json.get("data").and_then(|d| d.as_array()) else {
            return Vec::new();
        };
        data.iter()
            .filter_map(|m| m.get("id").and_then(|id| id.as_str()))
            .map(|id| {
                let mut card = ModelCard::new(id);
                card.provider.clone_from(&provider);
                card
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Representative engine object, including nullable and unknown extensions.
    fn engine_model() -> Value {
        json!({
            "id": "served-model", "object": "model", "created": 123,
            "owned_by": "vllm", "root": "org/checkpoint", "parent": null,
            "max_model_len": 32768,
            "permission": [{"id": "permission-1", "allow_sampling": true}],
            "future_field": {"nested": [false, null, 5]}
        })
    }

    /// List and single-model responses retain the complete public engine object.
    #[test]
    fn discovered_model_preserves_engine_fields_and_unknown_extensions() {
        let expected = engine_model();
        let mut card = ModelCard::new("served-model");
        card.upstream_model = Some(serde_json::from_value(expected.clone()).unwrap());
        let response = ListModelsResponse::from_model_cards([card.clone()]);
        assert_eq!(serde_json::to_value(&response.data[0]).unwrap(), expected);
        assert_eq!(
            serde_json::to_value(card.into_model_object()).unwrap(),
            expected
        );
    }

    /// Cards without discovery keep the old shape without leaking operator metadata.
    #[test]
    fn undiscovered_card_keeps_existing_shape_and_private_metadata_private() {
        let card = ModelCard::new("local").with_metadata(json!({"internal": "operator-only"}));
        assert_eq!(
            serde_json::to_value(card.into_model_object()).unwrap(),
            json!({
                "id": "local", "object": "model", "created": 0, "owned_by": "self_hosted"
            })
        );
    }

    /// Aliases and reserved keys cannot replace the registry identity.
    #[test]
    fn registry_identity_wins_for_aliases_and_reserved_extension_keys() {
        let mut card = ModelCard::new("friendly-alias");
        let mut model: ModelObject = serde_json::from_value(engine_model()).unwrap();
        model.extensions.insert("id".to_string(), json!("wrong-id"));
        model
            .extensions
            .insert("owned_by".to_string(), json!("wrong-owner"));
        card.upstream_model = Some(model);
        let raw = serde_json::to_string(&card.into_model_object()).unwrap();
        let body: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(body["id"], "friendly-alias");
        assert_eq!(body["owned_by"], "vllm");
        assert_eq!(raw.matches("\"id\":").count(), 2); // model plus permission id
    }

    /// Replica deduplication chooses a complete first card without mixing fields.
    #[test]
    fn replica_deduplication_keeps_one_complete_object_not_a_fieldwise_merge() {
        let mut first = ModelCard::new("served-model");
        first.upstream_model = Some(serde_json::from_value(engine_model()).unwrap());
        let mut second = first.clone();
        second
            .upstream_model
            .as_mut()
            .unwrap()
            .extensions
            .insert("max_model_len".into(), json!(8192));
        let response = ListModelsResponse::from_model_cards([first, second]);
        assert_eq!(response.data.len(), 1);
        assert_eq!(
            serde_json::to_value(&response.data[0]).unwrap(),
            engine_model()
        );
    }
}
