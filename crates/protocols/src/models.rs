//! OpenAI-compatible `/v1/models` response types.
//!
//! Provides [`ListModelsResponse`] built from in-memory [`ModelCard`] data,
//! ensuring every router returns the same format.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;
// The entry lives beside the model card in the worker crate; the path it had
// here is kept.
#[doc(inline)]
pub use smg_worker::models::ModelObject;

use super::{model_card::ModelCard, worker::ProviderType};

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
