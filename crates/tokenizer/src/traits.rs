use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use anyhow::Result;

use crate::chat_template::{
    ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
};

/// Type alias for token IDs
pub type TokenIdType = u32;

/// Core encoding trait - separate from decoding for modularity
pub trait Encoder: Send + Sync {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding>;
    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>>;
}

/// Core decoding trait - can be implemented independently
pub trait Decoder: Send + Sync {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String>;

    /// Incremental decode step — called once per generated token.
    ///
    /// Maintains mutable state (`ids`, `prefix`, `prefix_index`) across calls to
    /// produce incremental text output. The default implementation uses the
    /// double-decode algorithm (decode prefix, decode prefix+new, diff).
    ///
    /// HuggingFace overrides this with the native `step_decode_stream` from the
    /// `tokenizers` crate, which uses the same algorithm internally but avoids
    /// trait-method overhead for the two `decode()` calls.
    fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        // Recompute prefix if empty (first call or after incomplete UTF-8)
        if prefix.is_empty() && !ids.is_empty() {
            let new_prefix = self.decode(ids, skip_special_tokens)?;
            if !new_prefix.ends_with('�') {
                *prefix = new_prefix;
                *prefix_index = ids.len();
            }
        }

        ids.push(token_id);
        let string = self.decode(ids, skip_special_tokens)?;

        if string.len() > prefix.len() && !string.ends_with('�') {
            // Find char-safe split point
            let mut split_at = prefix.len();
            while !string.is_char_boundary(split_at) && split_at > 0 {
                split_at -= 1;
            }

            let new_text = string[split_at..].to_string();

            // Drain consumed tokens and cache new prefix for next call
            let new_prefix_len = ids.len() - *prefix_index;
            ids.drain(..*prefix_index);
            *prefix_index = new_prefix_len;
            *prefix = self.decode(ids, skip_special_tokens)?;

            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}

/// Combined tokenizer trait
pub trait Tokenizer: Encoder + Decoder {
    fn vocab_size(&self) -> usize;
    fn get_special_tokens(&self) -> &SpecialTokens;
    fn token_to_id(&self, token: &str) -> Option<TokenIdType>;
    fn id_to_token(&self, id: TokenIdType) -> Option<String>;

    /// Enable downcasting to concrete types
    fn as_any(&self) -> &dyn std::any::Any;

    /// Apply chat template to messages. Default returns an error for tokenizers without template support.
    fn apply_chat_template(
        &self,
        _messages: &[serde_json::Value],
        _params: ChatTemplateParams,
    ) -> Result<String> {
        Err(anyhow::anyhow!(
            "Chat template not supported by this tokenizer"
        ))
    }

    /// Get the content format expected by the chat template.
    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        ChatTemplateContentFormat::default()
    }

    /// Get the thinking toggle support for this template.
    fn thinking_toggle(&self) -> ThinkingToggle {
        ThinkingToggle::None
    }

    /// The variable name the template uses for the thinking toggle.
    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        None
    }

    /// `chat_template_kwargs.reasoning_effort` values that switch this
    /// tokenizer's renderer into thinking mode. Empty for renderers that
    /// don't interpret the kwarg natively.
    fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        &[]
    }

    /// Whether the template injects `<think>` in the generation prompt.
    fn think_in_prefill(&self) -> bool {
        false
    }

    /// Set or override the chat template.
    ///
    /// Returns an error if the template fails to parse or the tokenizer
    /// does not support chat templates.
    fn set_chat_template(&mut self, _template: String) -> Result<()> {
        Err(anyhow::anyhow!(
            "set_chat_template is not supported by this tokenizer"
        ))
    }

    /// EOS token IDs for stop detection.
    ///
    /// Merged from `config.json` and `generation_config.json` (eos_token_id, int or list).
    /// Models can have multiple EOS tokens (e.g., Llama 3: end_of_text + eom_id + eot_id).
    fn eos_token_ids(&self) -> &[TokenIdType] {
        &[]
    }
}

/// Contains the results of tokenizing text: token IDs, string tokens, and their spans
#[derive(Debug, Clone)]
pub enum Encoding {
    /// Hugging Face
    Hf(Box<tokenizers::tokenizer::Encoding>),
    /// Plain token ID vector
    Plain(Vec<TokenIdType>),
    /// Tiktoken (for GPT models) - now uses u32 in tiktoken-rs 0.7.0
    Tiktoken(Vec<TokenIdType>),
}

impl Encoding {
    /// Returns a reference to token IDs - zero-copy operation
    #[inline]
    pub fn token_ids(&self) -> &[TokenIdType] {
        match self {
            Encoding::Hf(inner) => inner.get_ids(),
            Encoding::Plain(inner) => inner,
            Encoding::Tiktoken(inner) => inner,
        }
    }

    /// Get a hash of the token IDs for caching purposes
    pub fn get_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Hash implementation for Encoding
impl Hash for Encoding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Encoding::Hf(inner) => inner.get_ids().hash(state),
            Encoding::Plain(inner) => inner.hash(state),
            Encoding::Tiktoken(inner) => inner.hash(state),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Vec<String>,
}
