//! Model card definitions for worker model configuration.
//!
//! Defines [`ModelCard`] which consolidates model-related configuration:
//! identity, capabilities, tokenization, and classification support.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{
    model_type::{Endpoint, ModelType},
    models::ModelObject,
    worker::ProviderType,
};

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes &T"
)]
fn is_zero(n: &u32) -> bool {
    *n == 0
}

fn default_model_type() -> ModelType {
    ModelType::LLM
}

/// Model card containing model configuration and capabilities.
///
/// # Example
///
/// ```
/// use openai_protocol::{model_type::ModelType, model_card::ModelCard, worker::ProviderType};
///
/// let card = ModelCard::new("meta-llama/Llama-3.1-8B-Instruct")
///     .with_display_name("Llama 3.1 8B Instruct")
///     .with_alias("llama-3.1-8b")
///     .with_model_type(ModelType::VISION_LLM)
///     .with_context_length(128_000)
///     .with_tokenizer_path("meta-llama/Llama-3.1-8B-Instruct");
///
/// assert!(card.matches("llama-3.1-8b"));
/// assert!(card.model_type.supports_vision());
/// assert!(card.provider.is_none()); // Local model, no external provider
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelCard {
    // === Identity ===
    /// Primary model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    pub id: String,

    /// Optional display name (e.g., "Llama 3.1 8B Instruct")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    /// Alternative names/aliases for this model
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,

    // === Capabilities ===
    /// Supported endpoint types (bitflags)
    #[serde(default = "default_model_type")]
    pub model_type: ModelType,

    /// HuggingFace model type string (e.g., "llama", "qwen2", "gpt-oss")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_model_type: Option<String>,

    /// Model architectures from HuggingFace config (e.g., ["LlamaForCausalLM"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub architectures: Vec<String>,

    /// Provider hint for API transformations.
    /// `None` means native/passthrough (no transformation needed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderType>,

    /// Maximum context length in tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,

    // === Tokenization & Parsing ===
    /// Path to tokenizer (e.g., HuggingFace model ID or local path)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,

    /// Chat template (Jinja2 template string or path)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    /// Reasoning parser type (e.g., "deepseek", "qwen")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,

    /// Tool/function calling parser type (e.g., "llama", "mistral")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,

    /// User-defined metadata (for fields not covered above)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    /// Public model object discovered from an HTTP engine, separate from
    /// operator metadata which must never be exposed through `/v1/models`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_model: Option<ModelObject>,

    // === Classification Support ===
    /// Classification label mapping (class index -> label name).
    /// Empty if not a classification model.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub id2label: HashMap<u32, String>,

    /// Number of classification labels (0 if not a classifier).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub num_labels: u32,
}

impl ModelCard {
    /// Create a new model card with minimal configuration.
    ///
    /// Defaults to `ModelType::LLM` and no provider (native/passthrough).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: None,
            aliases: Vec::new(),
            model_type: ModelType::LLM,
            hf_model_type: None,
            architectures: Vec::new(),
            provider: None,
            context_length: None,
            tokenizer_path: None,
            chat_template: None,
            reasoning_parser: None,
            tool_parser: None,
            metadata: None,
            upstream_model: None,
            id2label: HashMap::new(),
            num_labels: 0,
        }
    }

    // === Builder-style methods ===

    /// Set the display name
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Add a single alias
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Add multiple aliases
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases.extend(aliases.into_iter().map(|a| a.into()));
        self
    }

    /// Set the model type (capabilities)
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set the HuggingFace model type string
    pub fn with_hf_model_type(mut self, hf_model_type: impl Into<String>) -> Self {
        self.hf_model_type = Some(hf_model_type.into());
        self
    }

    /// Set the model architectures
    pub fn with_architectures(mut self, architectures: Vec<String>) -> Self {
        self.architectures = architectures;
        self
    }

    /// Set the provider type (for external API transformations)
    pub fn with_provider(mut self, provider: ProviderType) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the context length
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }

    /// Set the tokenizer path
    pub fn with_tokenizer_path(mut self, path: impl Into<String>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Set the chat template
    pub fn with_chat_template(mut self, template: impl Into<String>) -> Self {
        self.chat_template = Some(template.into());
        self
    }

    /// Set the reasoning parser type
    pub fn with_reasoning_parser(mut self, parser: impl Into<String>) -> Self {
        self.reasoning_parser = Some(parser.into());
        self
    }

    /// Set the tool parser type
    pub fn with_tool_parser(mut self, parser: impl Into<String>) -> Self {
        self.tool_parser = Some(parser.into());
        self
    }

    /// Set custom metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the id2label mapping for classification models
    pub fn with_id2label(mut self, id2label: HashMap<u32, String>) -> Self {
        self.num_labels = id2label.len() as u32;
        self.id2label = id2label;
        self
    }

    /// Set num_labels directly (alternative to with_id2label)
    pub fn with_num_labels(mut self, num_labels: u32) -> Self {
        self.num_labels = num_labels;
        self
    }

    // === Query methods ===

    /// Check if this model matches the given ID (including aliases)
    pub fn matches(&self, model_id: &str) -> bool {
        self.id == model_id || self.aliases.iter().any(|a| a == model_id)
    }

    /// Check if this model supports a given endpoint
    pub fn supports_endpoint(&self, endpoint: Endpoint) -> bool {
        self.model_type.supports_endpoint(endpoint)
    }

    /// Get the display name or fall back to ID
    pub fn name(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.id)
    }

    /// Check if this is a native/local model (no external provider)
    #[inline]
    pub fn is_native(&self) -> bool {
        self.provider.is_none()
    }

    /// Check if this model uses an external provider
    #[inline]
    pub fn has_external_provider(&self) -> bool {
        self.provider.is_some()
    }

    /// Check if this is an LLM (supports chat)
    #[inline]
    pub fn is_llm(&self) -> bool {
        self.model_type.is_llm()
    }

    /// Check if this is an embedding model
    #[inline]
    pub fn is_embedding_model(&self) -> bool {
        self.model_type.is_embedding_model()
    }

    /// Check if this model supports vision/multimodal
    #[inline]
    pub fn supports_vision(&self) -> bool {
        self.model_type.supports_vision()
    }

    /// Check if this model supports tools/function calling
    #[inline]
    pub fn supports_tools(&self) -> bool {
        self.model_type.supports_tools()
    }

    /// Check if this model supports reasoning
    #[inline]
    pub fn supports_reasoning(&self) -> bool {
        self.model_type.supports_reasoning()
    }

    /// Get the `owned_by` string for this model.
    ///
    /// Maps `None` → `"self_hosted"`, provider → `provider.as_str()`.
    pub fn owned_by(&self) -> &str {
        match &self.provider {
            Some(p) => p.as_str(),
            None => "self_hosted",
        }
    }

    /// Convert this model card into an OpenAI-compatible [`ModelObject`],
    /// consuming `self` to avoid cloning the model ID.
    pub fn into_model_object(self) -> ModelObject {
        let owned_by = self.owned_by().to_owned();
        if let Some(mut model) = self.upstream_model {
            // Registry identity wins, including entries listed under an alias.
            model.id = self.id;
            "model".clone_into(&mut model.object);
            for key in ["id", "object", "created", "owned_by"] {
                model.extensions.remove(key);
            }
            return model;
        }
        ModelObject {
            id: self.id,
            object: "model".to_owned(),
            created: 0,
            owned_by,
            extensions: Default::default(),
        }
    }

    /// Check if this is a classification model
    #[inline]
    pub fn is_classifier(&self) -> bool {
        self.num_labels > 0
    }

    /// Get label for a class index, with fallback to generic label (LABEL_N)
    pub fn get_label(&self, class_idx: u32) -> String {
        self.id2label
            .get(&class_idx)
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{class_idx}"))
    }
}

impl Default for ModelCard {
    fn default() -> Self {
        Self::new(super::UNKNOWN_MODEL_ID)
    }
}

impl std::fmt::Display for ModelCard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
