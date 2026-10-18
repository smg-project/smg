//! External worker registration steps for OpenAI-compatible API endpoints.
//!
//! These steps handle the discovery and creation of workers that connect to
//! external API providers (OpenAI, Anthropic, etc.) via HTTP.

mod create_workers;
mod discover_models;

pub use create_workers::CreateExternalWorkersStep;
pub(crate) use discover_models::{fetch_models, resolve_discovery_api_key};
pub use discover_models::{
    group_models_into_cards, infer_model_type_from_id, DiscoverModelsStep, ModelInfo,
    ModelsResponse,
};
