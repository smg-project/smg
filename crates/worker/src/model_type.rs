//! Model type definitions using bitflags for endpoint support.
//!
//! Defines [`ModelType`] using bitflags to represent which endpoints a model
//! can support, and [`Endpoint`] for routing decisions.

use std::borrow::Cow;

use bitflags::bitflags;
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

bitflags! {
    #[derive(Copy, Debug, Default, Clone, Eq, PartialEq, Hash)]
    pub struct ModelType: u16 {
        /// OpenAI Chat Completions API (/v1/chat/completions)
        const CHAT        = 1 << 0;
        /// OpenAI Completions API - legacy (/v1/completions)
        const COMPLETIONS = 1 << 1;
        /// OpenAI Responses API (/v1/responses)
        const RESPONSES   = 1 << 2;
        /// Embeddings API (/v1/embeddings)
        const EMBEDDINGS  = 1 << 3;
        /// Rerank API (/v1/rerank)
        const RERANK      = 1 << 4;
        /// The native Generate API (/generate)
        const GENERATE    = 1 << 5;
        /// Vision/multimodal support (images in input)
        const VISION      = 1 << 6;
        /// Tool/function calling support
        const TOOLS       = 1 << 7;
        /// Reasoning/thinking support (e.g., o1, DeepSeek-R1)
        const REASONING   = 1 << 8;
        /// Image generation (DALL-E, Sora, gpt-image)
        const IMAGE_GEN   = 1 << 9;
        /// Audio models (TTS, Whisper, realtime, transcribe)
        const AUDIO       = 1 << 10;
        /// Content moderation models
        const MODERATION  = 1 << 11;

        /// Standard LLM: chat + completions + responses + tools
        const LLM = Self::CHAT.bits() | Self::COMPLETIONS.bits()
                  | Self::RESPONSES.bits() | Self::TOOLS.bits();

        /// Vision-capable LLM: LLM + vision
        const VISION_LLM = Self::LLM.bits() | Self::VISION.bits();

        /// Reasoning LLM: LLM + reasoning (e.g., o1, o3, DeepSeek-R1)
        const REASONING_LLM = Self::LLM.bits() | Self::REASONING.bits();

        /// Full-featured LLM: all text generation capabilities
        const FULL_LLM = Self::VISION_LLM.bits() | Self::REASONING.bits();

        /// Embedding model only
        const EMBED_MODEL = Self::EMBEDDINGS.bits();

        /// Reranker model only
        const RERANK_MODEL = Self::RERANK.bits();

        /// Image generation model only (DALL-E, Sora, gpt-image)
        const IMAGE_MODEL = Self::IMAGE_GEN.bits();

        /// Audio model only (TTS, Whisper, realtime)
        const AUDIO_MODEL = Self::AUDIO.bits();

        /// Content moderation model only
        const MODERATION_MODEL = Self::MODERATION.bits();
    }
}

/// Mapping of individual capability flags to their names.
const CAPABILITY_NAMES: &[(ModelType, &str)] = &[
    (ModelType::CHAT, "chat"),
    (ModelType::COMPLETIONS, "completions"),
    (ModelType::RESPONSES, "responses"),
    (ModelType::EMBEDDINGS, "embeddings"),
    (ModelType::RERANK, "rerank"),
    (ModelType::GENERATE, "generate"),
    (ModelType::VISION, "vision"),
    (ModelType::TOOLS, "tools"),
    (ModelType::REASONING, "reasoning"),
    (ModelType::IMAGE_GEN, "image_gen"),
    (ModelType::AUDIO, "audio"),
    (ModelType::MODERATION, "moderation"),
];

impl ModelType {
    /// Check if this model type supports the chat completions endpoint
    #[inline]
    pub fn supports_chat(self) -> bool {
        self.contains(Self::CHAT)
    }

    /// Check if this model type supports the legacy completions endpoint
    #[inline]
    pub fn supports_completions(self) -> bool {
        self.contains(Self::COMPLETIONS)
    }

    /// Check if this model type supports the responses endpoint
    #[inline]
    pub fn supports_responses(self) -> bool {
        self.contains(Self::RESPONSES)
    }

    /// Check if this model type supports the embeddings endpoint
    #[inline]
    pub fn supports_embeddings(self) -> bool {
        self.contains(Self::EMBEDDINGS)
    }

    /// Check if this model type supports the rerank endpoint
    #[inline]
    pub fn supports_rerank(self) -> bool {
        self.contains(Self::RERANK)
    }

    /// Check if this model type supports the generate endpoint
    #[inline]
    pub fn supports_generate(self) -> bool {
        self.contains(Self::GENERATE)
    }

    /// Check if this model type supports vision/multimodal input
    #[inline]
    pub fn supports_vision(self) -> bool {
        self.contains(Self::VISION)
    }

    /// Check if this model type supports tool/function calling
    #[inline]
    pub fn supports_tools(self) -> bool {
        self.contains(Self::TOOLS)
    }

    /// Check if this model type supports reasoning/thinking
    #[inline]
    pub fn supports_reasoning(self) -> bool {
        self.contains(Self::REASONING)
    }

    /// Check if this model type supports image generation
    #[inline]
    pub fn supports_image_gen(self) -> bool {
        self.contains(Self::IMAGE_GEN)
    }

    /// Check if this model type supports audio (TTS, Whisper, etc.)
    #[inline]
    pub fn supports_audio(self) -> bool {
        self.contains(Self::AUDIO)
    }

    /// Check if this model type supports content moderation
    #[inline]
    pub fn supports_moderation(self) -> bool {
        self.contains(Self::MODERATION)
    }

    /// Check if this model type supports a given endpoint
    pub fn supports_endpoint(self, endpoint: Endpoint) -> bool {
        match endpoint {
            Endpoint::Chat => self.supports_chat(),
            Endpoint::Completions => self.supports_completions(),
            Endpoint::Responses => self.supports_responses(),
            Endpoint::Embeddings => self.supports_embeddings(),
            Endpoint::Rerank => self.supports_rerank(),
            Endpoint::Generate => self.supports_generate(),
            Endpoint::Models => true,
        }
    }

    /// Convert to a list of supported capability names
    pub fn as_capability_names(self) -> Vec<&'static str> {
        let mut result = Vec::with_capacity(CAPABILITY_NAMES.len());
        for &(flag, name) in CAPABILITY_NAMES {
            if self.contains(flag) {
                result.push(name);
            }
        }
        result
    }

    /// Check if this is an LLM (supports at least chat)
    #[inline]
    pub fn is_llm(self) -> bool {
        self.supports_chat()
    }

    /// Check if this is an embedding model
    #[inline]
    pub fn is_embedding_model(self) -> bool {
        self.supports_embeddings() && !self.supports_chat()
    }

    /// Check if this is a reranker model
    #[inline]
    pub fn is_reranker(self) -> bool {
        self.supports_rerank() && !self.supports_chat()
    }

    /// Check if this is an image generation model
    #[inline]
    pub fn is_image_model(self) -> bool {
        self.supports_image_gen() && !self.supports_chat()
    }

    /// Check if this is an audio model
    #[inline]
    pub fn is_audio_model(self) -> bool {
        self.supports_audio() && !self.supports_chat()
    }

    /// Check if this is a moderation model
    #[inline]
    pub fn is_moderation_model(self) -> bool {
        self.supports_moderation() && !self.supports_chat()
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = self.as_capability_names();
        if names.is_empty() {
            write!(f, "none")
        } else {
            write!(f, "{}", names.join(","))
        }
    }
}

impl Serialize for ModelType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let names = self.as_capability_names();
        let mut seq = serializer.serialize_seq(Some(names.len()))?;
        for name in names {
            seq.serialize_element(name)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for ModelType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;

        struct ModelTypeVisitor;

        impl<'de> de::Visitor<'de> for ModelTypeVisitor {
            type Value = ModelType;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an array of capability names or a u16 bitfield")
            }

            // Backward compat: accept numeric u16 bitfield
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<ModelType, E> {
                let bits = u16::try_from(v)
                    .map_err(|_| E::custom(format!("ModelType bits out of u16 range: {v}")))?;
                ModelType::from_bits(bits)
                    .ok_or_else(|| E::custom(format!("invalid ModelType bits: {bits}")))
            }

            // New format: array of capability name strings
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<ModelType, A::Error> {
                let mut model_type = ModelType::empty();
                while let Some(name) = seq.next_element::<String>()? {
                    let flag = CAPABILITY_NAMES
                        .iter()
                        .find(|(_, n)| *n == name.as_str())
                        .map(|(f, _)| *f)
                        .ok_or_else(|| {
                            de::Error::custom(format!("unknown ModelType capability: {name}"))
                        })?;
                    model_type |= flag;
                }
                Ok(model_type)
            }
        }

        deserializer.deserialize_any(ModelTypeVisitor)
    }
}

/// Manual JsonSchema impl for `ModelType` — serialized as an array of capability name strings.
impl JsonSchema for ModelType {
    fn schema_name() -> Cow<'static, str> {
        "ModelType".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "array",
            "description": "Bitflag capabilities serialized as an array of capability names",
            "items": {
                "type": "string",
                "enum": [
                    "chat",
                    "completions",
                    "responses",
                    "embeddings",
                    "rerank",
                    "generate",
                    "vision",
                    "tools",
                    "reasoning",
                    "image_gen",
                    "audio",
                    "moderation"
                ]
            }
        })
    }
}

/// Endpoint types for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Endpoint {
    /// Chat completions endpoint (/v1/chat/completions)
    Chat,
    /// Legacy completions endpoint (/v1/completions)
    Completions,
    /// Responses endpoint (/v1/responses)
    Responses,
    /// Embeddings endpoint (/v1/embeddings)
    Embeddings,
    /// Rerank endpoint (/v1/rerank)
    Rerank,
    /// The native generate endpoint (/generate)
    Generate,
    /// Models listing endpoint (/v1/models)
    Models,
}

impl Endpoint {
    /// Get the URL path for this endpoint
    pub fn path(self) -> &'static str {
        match self {
            Endpoint::Chat => "/v1/chat/completions",
            Endpoint::Completions => "/v1/completions",
            Endpoint::Responses => "/v1/responses",
            Endpoint::Embeddings => "/v1/embeddings",
            Endpoint::Rerank => "/v1/rerank",
            Endpoint::Generate => "/generate",
            Endpoint::Models => "/v1/models",
        }
    }

    /// Parse an endpoint from a URL path
    pub fn from_path(path: &str) -> Option<Self> {
        let path = path.trim_end_matches('/');
        match path {
            "/v1/chat/completions" => Some(Endpoint::Chat),
            "/v1/completions" => Some(Endpoint::Completions),
            "/v1/responses" => Some(Endpoint::Responses),
            "/v1/embeddings" => Some(Endpoint::Embeddings),
            "/v1/rerank" => Some(Endpoint::Rerank),
            "/generate" => Some(Endpoint::Generate),
            "/v1/models" => Some(Endpoint::Models),
            _ => None,
        }
    }

    /// Get the required ModelType flag for this endpoint
    pub fn required_capability(self) -> Option<ModelType> {
        match self {
            Endpoint::Chat => Some(ModelType::CHAT),
            Endpoint::Completions => Some(ModelType::COMPLETIONS),
            Endpoint::Responses => Some(ModelType::RESPONSES),
            Endpoint::Embeddings => Some(ModelType::EMBEDDINGS),
            Endpoint::Rerank => Some(ModelType::RERANK),
            Endpoint::Generate => Some(ModelType::GENERATE),
            Endpoint::Models => None,
        }
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Chat => write!(f, "chat"),
            Endpoint::Completions => write!(f, "completions"),
            Endpoint::Responses => write!(f, "responses"),
            Endpoint::Embeddings => write!(f, "embeddings"),
            Endpoint::Rerank => write!(f, "rerank"),
            Endpoint::Generate => write!(f, "generate"),
            Endpoint::Models => write!(f, "models"),
        }
    }
}
