use std::{collections::HashMap, path::Path};

use anyhow::{Error, Result};
use serde::Deserialize;
use tokenizers::{
    models::bpe::BPE,
    normalizers::unicode::NFC,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    processors::template::TemplateProcessing,
    tokenizer::{step_decode_stream, SplitDelimiterBehavior, Tokenizer as HfTokenizer},
    AddedToken,
};
use tracing::debug;

use crate::{
    chat_template::{
        load_chat_template_from_file, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateState, ThinkingKeyName, ThinkingToggle,
    },
    encoders::{deepseek_v32, deepseek_v4},
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait},
};

#[derive(Debug, Clone, Copy)]
enum Renderer {
    Jinja,
    DeepseekV32,
    DeepseekV4,
}

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    reverse_vocab: HashMap<TokenIdType, String>,
    chat_template: ChatTemplateState,
    /// EOS token IDs from config.json + generation_config.json
    eos_token_ids: Vec<TokenIdType>,
    /// Which renderer applies chat templates for this model.
    renderer: Renderer,
}

const QWEN2_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Deserialize)]
struct AddedTokenConfig {
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    normalized: Option<bool>,
    #[serde(default)]
    special: bool,
}

impl HuggingFaceTokenizer {
    /// Create a tokenizer from a HuggingFace tokenizer JSON file
    pub fn from_file(file_path: &str) -> Result<Self> {
        // Try to auto-discover chat template if not explicitly provided
        let path = Path::new(file_path);
        let chat_template_path = path
            .parent()
            .and_then(crate::factory::discover_chat_template_in_dir);
        Self::from_file_with_chat_template(file_path, chat_template_path.as_deref())
    }

    /// Create a tokenizer from a HuggingFace tokenizer JSON file with an optional chat template
    pub fn from_file_with_chat_template(
        file_path: &str,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(file_path)
            .map_err(|e| Error::msg(format!("Failed to load tokenizer: {e}")))?;
        Self::from_built_tokenizer(tokenizer, Path::new(file_path), chat_template_path)
    }

    /// Create a Qwen2-compatible byte-level BPE tokenizer from a Hugging Face
    /// directory containing `vocab.json`, `merges.txt`, and
    /// `tokenizer_config.json` but no `tokenizer.json`.
    pub fn from_vocab_and_merges_dir(dir: &Path) -> Result<Self> {
        let chat_template_path = crate::factory::discover_chat_template_in_dir(dir);
        Self::from_vocab_and_merges_dir_with_chat_template(dir, chat_template_path.as_deref())
    }

    /// Create a Qwen2-compatible byte-level BPE tokenizer with an optional
    /// explicit chat template.
    pub fn from_vocab_and_merges_dir_with_chat_template(
        dir: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let tokenizer = Self::build_qwen2_bpe_tokenizer(dir)?;
        // Shared initialization only needs this path to locate sibling config
        // files. The file itself intentionally does not exist in this layout.
        let logical_tokenizer_path = dir.join("tokenizer.json");
        Self::from_built_tokenizer(tokenizer, &logical_tokenizer_path, chat_template_path)
    }

    fn build_qwen2_bpe_tokenizer(dir: &Path) -> Result<HfTokenizer> {
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");
        let config_path = dir.join("tokenizer_config.json");

        let config_content = std::fs::read_to_string(&config_path).map_err(|error| {
            Error::msg(format!("Failed to read {}: {error}", config_path.display()))
        })?;
        let config: serde_json::Value = serde_json::from_str(&config_content).map_err(|error| {
            Error::msg(format!(
                "Failed to parse {}: {error}",
                config_path.display()
            ))
        })?;
        let tokenizer_class = config
            .get("tokenizer_class")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::msg(format!(
                    "{} is missing tokenizer_class; cannot infer vocab.json + merges.txt semantics",
                    config_path.display()
                ))
            })?;
        if !matches!(tokenizer_class, "Qwen2Tokenizer" | "Qwen2TokenizerFast") {
            return Err(Error::msg(format!(
                "Unsupported vocab.json + merges.txt tokenizer_class '{tokenizer_class}' in {}",
                config_path.display()
            )));
        }

        let vocab_path_str = vocab_path.to_str().ok_or_else(|| {
            Error::msg(format!("Tokenizer path is not valid UTF-8: {vocab_path:?}"))
        })?;
        let merges_path_str = merges_path.to_str().ok_or_else(|| {
            Error::msg(format!(
                "Tokenizer path is not valid UTF-8: {merges_path:?}"
            ))
        })?;
        let bpe = BPE::builder()
            .files(vocab_path_str.to_string(), merges_path_str.to_string())
            .build()
            .map_err(|error| Error::msg(format!("Failed to build Qwen2 BPE model: {error}")))?;
        let mut tokenizer = HfTokenizer::new(bpe);

        tokenizer
            .with_normalizer(Some(NFC))
            .map_err(|error| Error::msg(format!("Failed to configure NFC normalizer: {error}")))?;
        let split = Split::new(
            SplitPattern::Regex(QWEN2_PRETOKENIZE_REGEX.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|error| Error::msg(format!("Failed to build Qwen2 pre-tokenizer: {error}")))?;
        let add_prefix_space = config
            .get("add_prefix_space")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let byte_level = ByteLevel::default()
            .add_prefix_space(add_prefix_space)
            .use_regex(false);
        tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(byte_level),
        ])));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer.with_post_processor(Some(ByteLevel::default().trim_offsets(false)));

        let mut added_tokens = Vec::new();
        if let Some(decoder) = config
            .get("added_tokens_decoder")
            .and_then(serde_json::Value::as_object)
        {
            for (raw_id, raw_token) in decoder {
                let id = raw_id.parse::<u32>().map_err(|error| {
                    Error::msg(format!(
                        "Invalid added token ID '{raw_id}' in {}: {error}",
                        config_path.display()
                    ))
                })?;
                let entry: AddedTokenConfig =
                    serde_json::from_value(raw_token.clone()).map_err(|error| {
                        Error::msg(format!(
                            "Invalid added token {raw_id} in {}: {error}",
                            config_path.display()
                        ))
                    })?;
                let normalized = entry.normalized.unwrap_or(!entry.special);
                let token = AddedToken::from(entry.content, entry.special)
                    .single_word(entry.single_word)
                    .lstrip(entry.lstrip)
                    .rstrip(entry.rstrip)
                    .normalized(normalized);
                added_tokens.push((id, token));
            }
        }
        added_tokens.sort_unstable_by_key(|(id, _)| *id);

        tokenizer
            .add_tokens(added_tokens.iter().map(|(_, token)| token.clone()))
            .map_err(|error| Error::msg(format!("Failed to add configured tokens: {error}")))?;
        for (expected_id, token) in &added_tokens {
            let actual_id = tokenizer.token_to_id(&token.content);
            if actual_id != Some(*expected_id) {
                return Err(Error::msg(format!(
                    "Added token '{}' expected ID {expected_id}, got {actual_id:?}; non-contiguous explicit added-token IDs are unsupported",
                    token.content
                )));
            }
        }

        Ok(tokenizer)
    }

    fn from_built_tokenizer(
        mut tokenizer: HfTokenizer,
        tokenizer_path: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        // Build vocab mappings (include special tokens to get added_tokens like <|im_start|>)
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        // Load tokenizer_config.json once for chat template, add_bos/eos, and special tokens
        let config_result = Self::load_chat_template_and_config(&tokenizer_path.to_string_lossy());
        let mut chat_template_str = config_result.chat_template;
        let add_bos_token = config_result.add_bos_token;
        let add_eos_token = config_result.add_eos_token;

        // Extract special tokens — config values override vocab pattern matching
        let special_tokens = Self::extract_special_tokens(&tokenizer, &config_result.config_tokens);

        if let Some(template_path) = chat_template_path {
            chat_template_str = load_chat_template_from_file(template_path)?;
        }

        // Configure post_processor based on tokenizer_config.json (matches Python transformers)
        // Only modify when at least one setting is explicitly true
        let needs_eos = add_eos_token == Some(true);
        let needs_bos = match add_bos_token {
            Some(true) => true,
            Some(false) => false,
            // Not set: preserve existing behavior from tokenizer.json
            None => needs_eos && Self::tokenizer_adds_special_tokens(&tokenizer),
        };

        if needs_bos || needs_eos {
            if let Some(post_processor) =
                Self::build_post_processor(needs_bos, needs_eos, &special_tokens, &vocab)
            {
                debug!(needs_bos, needs_eos, "Configured post_processor");
                tokenizer.with_post_processor(Some(post_processor));
            }
        }

        // Load merged EOS token IDs from config.json + generation_config.json
        let eos_token_ids = tokenizer_path
            .parent()
            .map(crate::eos::load_eos_token_ids)
            .unwrap_or_default();

        // Detect a custom Python-encoder model from config.json::architectures.
        let renderer = tokenizer_path
            .parent()
            .map(detect_renderer_from_config)
            .unwrap_or(Renderer::Jinja);

        Ok(HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: ChatTemplateState::new(chat_template_str)?,
            eos_token_ids,
            renderer,
        })
    }

    /// Check if the tokenizer's post_processor adds special tokens (e.g., BOS)
    fn tokenizer_adds_special_tokens(tokenizer: &HfTokenizer) -> bool {
        tokenizer
            .encode("", true)
            .map(|enc| !enc.get_ids().is_empty())
            .unwrap_or(false)
    }

    /// Build a TemplateProcessing post_processor (matches Python transformers' update_post_processor)
    /// Template format: "{bos}:0 $A:0 {eos}:0" with optional BOS/EOS based on config
    fn build_post_processor(
        add_bos_token: bool,
        add_eos_token: bool,
        special_tokens: &SpecialTokens,
        vocab: &HashMap<String, TokenIdType>,
    ) -> Option<TemplateProcessing> {
        // Build template string exactly like Python:
        // single = f"{(bos + ':0 ') if add_bos_token else ''}$A:0{(' ' + eos + ':0') if add_eos_token else ''}"
        let mut template = String::with_capacity(32);
        let mut tokens = Vec::with_capacity(2);

        if add_bos_token {
            let bos = special_tokens.bos_token.as_ref()?;
            let bos_id = vocab.get(bos).copied()?;
            template.push_str(bos);
            template.push_str(":0 ");
            tokens.push((bos.clone(), bos_id));
        }

        template.push_str("$A:0");

        if add_eos_token {
            let eos = special_tokens.eos_token.as_ref()?;
            let eos_id = vocab.get(eos).copied()?;
            template.push(' ');
            template.push_str(eos);
            template.push_str(":0");
            tokens.push((eos.clone(), eos_id));
        }

        TemplateProcessing::builder()
            .try_single(template.as_str())
            .ok()?
            .special_tokens(tokens)
            .build()
            .ok()
    }

    /// Create from an existing HuggingFace tokenizer
    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Self {
        let special_tokens = Self::extract_special_tokens(&tokenizer, &ConfigTokens::default());
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        HuggingFaceTokenizer {
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: ChatTemplateState::empty(),
            eos_token_ids: Vec::new(), // No directory path in from_tokenizer
            renderer: Renderer::Jinja,
        }
    }

    /// Extract special tokens from the tokenizer, using config values when available.
    ///
    /// Prefers explicit values from `tokenizer_config.json` (e.g., `"bos_token": "<|begin_of_text|>"`)
    /// over pattern matching against the vocabulary, since models like Llama 4 use non-standard
    /// token names that aren't in the hardcoded pattern list.
    fn extract_special_tokens(
        tokenizer: &HfTokenizer,
        config_tokens: &ConfigTokens,
    ) -> SpecialTokens {
        // Get vocab with special tokens included (added_tokens like <|im_start|>)
        let vocab = tokenizer.get_vocab(true);

        let find_token = |patterns: &[&str]| -> Option<String> {
            for pattern in patterns {
                if vocab.contains_key(*pattern) {
                    return Some((*pattern).to_string());
                }
            }
            None
        };

        // Extract additional special tokens using the tokenizers library API
        let additional_special_tokens: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_id, token)| token.special)
            .map(|(_id, token)| token.content.clone())
            .collect();

        // Config values take priority over pattern matching
        SpecialTokens {
            bos_token: config_tokens
                .bos_token
                .clone()
                .or_else(|| find_token(&["<s>", "<|startoftext|>", "<BOS>", "[CLS]"])),
            eos_token: config_tokens
                .eos_token
                .clone()
                .or_else(|| find_token(&["</s>", "<|endoftext|>", "<EOS>", "[SEP]"])),
            unk_token: config_tokens
                .unk_token
                .clone()
                .or_else(|| find_token(&["<unk>", "<UNK>", "[UNK]"])),
            sep_token: find_token(&["[SEP]", "<sep>", "<SEP>"]),
            pad_token: config_tokens
                .pad_token
                .clone()
                .or_else(|| find_token(&["<pad>", "<PAD>", "[PAD]"])),
            cls_token: find_token(&["[CLS]", "<cls>", "<CLS>"]),
            mask_token: find_token(&["[MASK]", "<mask>", "<MASK>"]),
            additional_special_tokens,
        }
    }

    /// Load chat template, special token settings, and token strings from tokenizer_config.json.
    /// Reads the file once and extracts everything needed by the tokenizer constructor.
    fn load_chat_template_and_config(tokenizer_path: &str) -> TokenizerConfigResult {
        (|| {
            let path = Path::new(tokenizer_path);
            let config_path = path.parent()?.join("tokenizer_config.json");

            if !config_path.exists() {
                return None;
            }

            let content = std::fs::read_to_string(&config_path).ok()?;
            let config: serde_json::Value = serde_json::from_str(&content).ok()?;

            // Extract chat template directly from parsed config (avoid re-reading the file)
            let chat_template = config
                .get("chat_template")
                .and_then(|v| v.as_str())
                .map(String::from);

            let add_bos_token = config.get("add_bos_token").and_then(|v| v.as_bool());
            let add_eos_token = config.get("add_eos_token").and_then(|v| v.as_bool());

            // Extract special token strings (handles both "string" and {"content": "string"})
            let get_token = |key: &str| -> Option<String> {
                config.get(key).and_then(|v| {
                    v.as_str()
                        .map(String::from)
                        .or_else(|| v.get("content").and_then(|c| c.as_str()).map(String::from))
                })
            };

            let config_tokens = ConfigTokens {
                bos_token: get_token("bos_token"),
                eos_token: get_token("eos_token"),
                unk_token: get_token("unk_token"),
                pad_token: get_token("pad_token"),
            };

            Some(TokenizerConfigResult {
                chat_template,
                add_bos_token,
                add_eos_token,
                config_tokens,
            })
        })()
        .unwrap_or_default()
    }
}

/// Special token strings read from tokenizer_config.json.
#[derive(Default)]
struct ConfigTokens {
    bos_token: Option<String>,
    eos_token: Option<String>,
    unk_token: Option<String>,
    pad_token: Option<String>,
}

/// Result of parsing tokenizer_config.json.
#[derive(Default)]
struct TokenizerConfigResult {
    chat_template: Option<String>,
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    config_tokens: ConfigTokens,
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        self.tokenizer
            .encode(input, add_special_tokens)
            .map_err(|e| Error::msg(format!("Encoding failed: {e}")))
            .map(|encoding| Encoding::Hf(Box::new(encoding)))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.tokenizer
            .encode_batch(inputs.to_vec(), add_special_tokens)
            .map_err(|e| Error::msg(format!("Batch encoding failed: {e}")))
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|e| Encoding::Hf(Box::new(e)))
                    .collect()
            })
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| Error::msg(format!("Decoding failed: {e}")))
    }

    /// Native incremental decode using the HF `step_decode_stream`.
    ///
    /// This delegates to the same algorithm the default trait method uses, but
    /// the two internal `decode()` calls go directly through the concrete
    /// `TokenizerImpl` rather than through `dyn Decoder` vtable dispatch.
    fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        step_decode_stream(
            &self.tokenizer,
            vec![token_id],
            skip_special_tokens,
            ids,
            prefix,
            prefix_index,
        )
        .map_err(|e| Error::msg(format!("Decode stream error: {e}")))
    }
}

impl TokenizerTrait for HuggingFaceTokenizer {
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.reverse_vocab.get(&id).cloned()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn eos_token_ids(&self) -> &[TokenIdType] {
        &self.eos_token_ids
    }

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        match self.renderer {
            Renderer::Jinja => {
                // Inject special tokens if the caller didn't provide them.
                if params.special_tokens.is_some() {
                    return self.chat_template.apply(messages, params);
                }
                let params = ChatTemplateParams {
                    special_tokens: Some(&self.special_tokens),
                    ..params
                };
                self.chat_template.apply(messages, params)
            }
            Renderer::DeepseekV32 => apply_deepseek_v32(messages, &params),
            Renderer::DeepseekV4 => apply_deepseek_v4(messages, &params),
        }
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.chat_template.content_format()
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        match self.renderer {
            // Native encoders bypass Jinja detection, so report their official
            // defaults directly. V3.2 defaults off; V4 defaults on.
            Renderer::DeepseekV32 => ThinkingToggle::DefaultOff,
            Renderer::DeepseekV4 => ThinkingToggle::DefaultOn,
            Renderer::Jinja => self.chat_template.thinking_toggle(),
        }
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        match self.renderer {
            Renderer::DeepseekV32 | Renderer::DeepseekV4 => Some(ThinkingKeyName::Thinking),
            Renderer::Jinja => self.chat_template.thinking_key_name(),
        }
    }
    fn think_in_prefill(&self) -> bool {
        match self.renderer {
            // Both encoders emit `<｜Assistant｜><think>` at the end of the
            // prompt when thinking mode is on; the completion therefore starts
            // mid-reasoning and the parser must be told so.
            Renderer::DeepseekV32 | Renderer::DeepseekV4 => true,
            Renderer::Jinja => self.chat_template.think_in_prefill(),
        }
    }

    fn set_chat_template(&mut self, template: String) -> Result<()> {
        self.chat_template.set(template)
    }
}

// ---------------------------------------------------------------------------
// Renderer detection (config.json::architectures)
// ---------------------------------------------------------------------------
/// Inspect the sibling `config.json` to decide which chat-template renderer to
/// use. A missing or malformed file falls back to [`Renderer::Jinja`] without
/// erroring (debug-logged), preserving backward compatibility for every model
/// not in the architecture list.
fn detect_renderer_from_config(dir: &Path) -> Renderer {
    let path = dir.join("config.json");
    if !path.exists() {
        return Renderer::Jinja;
    }
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(err) => {
            debug!(?err, ?path, "config.json unreadable; using Jinja renderer");
            return Renderer::Jinja;
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(err) => {
            debug!(?err, ?path, "config.json malformed; using Jinja renderer");
            return Renderer::Jinja;
        }
    };
    let architectures = value.get("architectures").and_then(|v| v.as_array());
    let arch_strs: Vec<&str> = architectures
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if arch_strs.contains(&"DeepseekV32ForCausalLM") {
        debug!(?path, "selected DeepseekV32 chat-template renderer");
        return Renderer::DeepseekV32;
    }
    if arch_strs.contains(&"DeepseekV4ForCausalLM") {
        debug!(?path, "selected DeepseekV4 chat-template renderer");
        return Renderer::DeepseekV4;
    }
    Renderer::Jinja
}

// ---------------------------------------------------------------------------
// DeepSeek V3.2 / V4 dispatch shims
// ---------------------------------------------------------------------------
/// Derive a native DeepSeek renderer's thinking mode. These encoders bypass
/// `ChatTemplateState::apply`, so this is where the resolved thinking preference
/// is consumed. An explicit `template_kwargs["thinking"]` wins; otherwise fall
/// back to `params.thinking` (resolved from `reasoning_effort` / Anthropic
/// `ThinkingConfig`) and finally the renderer's official default.
fn derive_thinking_mode(
    params: &ChatTemplateParams,
    default_enabled: bool,
) -> deepseek_v32::ThinkingMode {
    let enabled = params
        .template_kwargs
        .and_then(|k| k.get("thinking"))
        .and_then(serde_json::Value::as_bool)
        .or(params.thinking)
        .unwrap_or(default_enabled);
    if enabled {
        deepseek_v32::ThinkingMode::Thinking
    } else {
        deepseek_v32::ThinkingMode::Chat
    }
}

/// Per DeepSeek's encoding README, preserve all reasoning when a system or
/// developer message declares `tools`; otherwise drop earlier reasoning.
fn resolve_drop_thinking(messages: &[serde_json::Value]) -> bool {
    !messages.iter().any(|m| {
        let role = m.get("role").and_then(|r| r.as_str());
        matches!(role, Some("system" | "developer"))
            && m.get("tools")
                .and_then(|t| t.as_array())
                .is_some_and(|arr| !arr.is_empty())
    })
}
/// Attach `tools` to a leading system/developer message so the V3.2/V4
/// encoder can render the tools block. Mirrors the wrapper step in
/// vllm's `vllm/tokenizers/deepseek_v32.py` and sglang's V4 serving path.
/// Returns `None` when no rewrite is needed so callers can pass the input
/// slice directly in the common path.
fn inject_tools_into_messages(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
) -> Option<Vec<serde_json::Value>> {
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }
    let mut owned: Vec<serde_json::Value> = messages.to_vec();
    let first_role = owned
        .first()
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str());
    if !matches!(first_role, Some("system" | "developer")) {
        owned.insert(0, serde_json::json!({ "role": "system", "content": "" }));
    }
    if let Some(obj) = owned[0].as_object_mut() {
        obj.insert("tools".into(), serde_json::Value::Array(tools.to_vec()));
    }
    Some(owned)
}

fn apply_deepseek_v32(
    messages: &[serde_json::Value],
    params: &ChatTemplateParams,
) -> Result<String> {
    let owned = inject_tools_into_messages(messages, params.tools);
    let msgs: &[serde_json::Value] = owned.as_deref().unwrap_or(messages);
    let thinking_mode = derive_thinking_mode(params, false);
    let encode_params = deepseek_v32::EncodeParams {
        add_default_bos_token: true,
        drop_thinking: resolve_drop_thinking(msgs),
    };
    deepseek_v32::encode_messages(msgs, thinking_mode, &encode_params)
        .map_err(|e| Error::msg(format!("DeepSeek V3.2 encode failed: {e}")))
}
fn apply_deepseek_v4(
    messages: &[serde_json::Value],
    params: &ChatTemplateParams,
) -> Result<String> {
    let owned = inject_tools_into_messages(messages, params.tools);
    let msgs: &[serde_json::Value] = owned.as_deref().unwrap_or(messages);
    let thinking_mode = derive_thinking_mode(params, true);
    let reasoning_effort = params
        .template_kwargs
        .and_then(|k| k.get("reasoning_effort"))
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "max" | "xhigh" => Some(deepseek_v4::ReasoningEffort::Max),
            "high" | "medium" | "low" => Some(deepseek_v4::ReasoningEffort::High),
            _ => None,
        });
    let encode_params = deepseek_v4::EncodeParams {
        add_default_bos_token: true,
        drop_thinking: resolve_drop_thinking(msgs),
        reasoning_effort,
    };
    deepseek_v4::encode_messages(msgs, thinking_mode, &encode_params)
        .map_err(|e| Error::msg(format!("DeepSeek V4 encode failed: {e}")))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::derive_thinking_mode;
    use crate::{chat_template::ChatTemplateParams, encoders::deepseek_v32::ThinkingMode};

    fn thinking_kwargs(value: bool) -> HashMap<String, serde_json::Value> {
        HashMap::from([("thinking".to_string(), serde_json::Value::Bool(value))])
    }

    #[test]
    fn derive_thinking_mode_honors_defaults_and_explicit_override() {
        assert!(matches!(
            derive_thinking_mode(&ChatTemplateParams::default(), false),
            ThinkingMode::Chat
        ));
        assert!(matches!(
            derive_thinking_mode(&ChatTemplateParams::default(), true),
            ThinkingMode::Thinking
        ));

        assert!(matches!(
            derive_thinking_mode(
                &ChatTemplateParams {
                    thinking: Some(true),
                    ..Default::default()
                },
                false
            ),
            ThinkingMode::Thinking
        ));
        assert!(matches!(
            derive_thinking_mode(
                &ChatTemplateParams {
                    thinking: Some(false),
                    ..Default::default()
                },
                true
            ),
            ThinkingMode::Chat
        ));

        let on = thinking_kwargs(true);
        assert!(matches!(
            derive_thinking_mode(
                &ChatTemplateParams {
                    thinking: Some(false),
                    template_kwargs: Some(&on),
                    ..Default::default()
                },
                false
            ),
            ThinkingMode::Thinking
        ));
        let off = thinking_kwargs(false);
        assert!(matches!(
            derive_thinking_mode(
                &ChatTemplateParams {
                    thinking: Some(true),
                    template_kwargs: Some(&off),
                    ..Default::default()
                },
                true
            ),
            ThinkingMode::Chat
        ));
    }
}
