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
    encoders::{deepseek_v32, deepseek_v4, deepseek_v41},
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait},
};

#[derive(Debug, Clone, Copy)]
enum Renderer {
    Jinja,
    DeepseekV32,
    DeepseekV4(deepseek_v4::EffortEncoding),
    DeepseekV41,
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
            Renderer::DeepseekV4(encoding) => apply_deepseek_v4(messages, &params, encoding),
            Renderer::DeepseekV41 => apply_deepseek_v41(messages, &params),
        }
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        match self.renderer {
            // V4.1 wants message parts preserved (OpenAI wire format) so the
            // gateway passes `image_url`/`image` parts through as a list
            // instead of flattening to a string; the encoder turns each part
            // into a placeholder at its position. Part *order* is still the
            // gateway's media-part order (the V4.1 multimodal spec declares
            // `Authored`; without a spec the gateway moves media first).
            // V3.2/V4 have no native opinion here and fall back to whatever
            // the (usually absent) Jinja template reports.
            Renderer::DeepseekV41 => ChatTemplateContentFormat::OpenAI,
            Renderer::Jinja | Renderer::DeepseekV32 | Renderer::DeepseekV4(_) => {
                self.chat_template.content_format()
            }
        }
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        match self.renderer {
            // DeepSeek V3.2 and V4 encoders gate thinking on the `thinking`
            // kwarg, default off. The Jinja processor has no knowledge of
            // the native encoder so we must report it directly.
            Renderer::DeepseekV32 | Renderer::DeepseekV4(_) => ThinkingToggle::DefaultOff,
            // V4.1 defaults thinking ON: `reasoning_effort: "none"` or an
            // explicit `thinking: false` (or vLLM's `enable_thinking` alias,
            // see `renderer_capabilities`) turns it off.
            Renderer::DeepseekV41 => ThinkingToggle::DefaultOn,
            Renderer::Jinja => self.chat_template.thinking_toggle(),
        }
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        match self.renderer {
            Renderer::DeepseekV32 | Renderer::DeepseekV4(_) | Renderer::DeepseekV41 => {
                Some(ThinkingKeyName::Thinking)
            }
            Renderer::Jinja => self.chat_template.thinking_key_name(),
        }
    }
    fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        match self.renderer {
            Renderer::DeepseekV4(encoding) => encoding.valid_native_values(),
            Renderer::DeepseekV41 => deepseek_v41::NATIVE_EFFORT_VALUES,
            Renderer::DeepseekV32 | Renderer::Jinja => &[],
        }
    }

    fn think_in_prefill(&self) -> bool {
        match self.renderer {
            // All three native encoders emit `<｜Assistant｜><think>` at the end
            // of the prompt when thinking mode is on; the completion therefore
            // starts mid-reasoning and the parser must be told so.
            Renderer::DeepseekV32 | Renderer::DeepseekV4(_) | Renderer::DeepseekV41 => true,
            Renderer::Jinja => self.chat_template.think_in_prefill(),
        }
    }

    fn renderer_capabilities(&self) -> crate::traits::RendererCapabilities {
        match self.renderer {
            // The V4.1 shim honours vLLM's `enable_thinking` alias, renders a
            // trailing assistant message itself when `add_generation_prompt`
            // is false, and parses tool-call `arguments` strings with the
            // reference's tolerance.
            Renderer::DeepseekV41 => crate::traits::RendererCapabilities {
                enable_thinking_alias: true,
                native_assistant_continuation: true,
                raw_tool_call_arguments: true,
            },
            Renderer::DeepseekV32 | Renderer::DeepseekV4(_) | Renderer::Jinja => {
                crate::traits::RendererCapabilities::default()
            }
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
    // Checked before the V4 arm below: V4.1 ships its own architecture name,
    // but some checkpoints identify themselves only via `model_type`.
    let model_type = value.get("model_type").and_then(|v| v.as_str());
    if arch_strs.contains(&"DeepseekV41ForCausalLM") || model_type == Some("deepseek_v41") {
        debug!(?path, "selected DeepseekV41 chat-template renderer");
        return Renderer::DeepseekV41;
    }
    if arch_strs.contains(&"DeepseekV4ForCausalLM") {
        let encoding = detect_dsv4_effort_encoding(dir);
        debug!(
            ?path,
            ?encoding,
            "selected DeepseekV4 chat-template renderer"
        );
        return Renderer::DeepseekV4(encoding);
    }
    Renderer::Jinja
}

/// Pick the effort-prompt revision for a V4 checkpoint. `config.json` and
/// `tokenizer_config.json` are identical across the V4 family, so the only
/// discriminator is the shipped `encoding/encoding_dsv4.py`; when it is
/// absent, fall back to a `0731` marker in the path (covers HF-hub cache
/// dirs like `models--deepseek-ai--DeepSeek-V4-Flash-0731/...`).
fn detect_dsv4_effort_encoding(dir: &Path) -> deepseek_v4::EffortEncoding {
    match std::fs::read_to_string(dir.join("encoding").join("encoding_dsv4.py")) {
        Ok(source) => deepseek_v4::EffortEncoding::detect_from_encoder_source(&source),
        Err(_) if dir.to_string_lossy().contains("0731") => deepseek_v4::EffortEncoding::V0731,
        Err(_) => deepseek_v4::EffortEncoding::Original,
    }
}

// ---------------------------------------------------------------------------
// DeepSeek V3.2 / V4 dispatch shims
// ---------------------------------------------------------------------------
/// Derive the V3.2 / V4 thinking mode. These native encoders bypass
/// `ChatTemplateState::apply`, so this is where the resolved thinking preference
/// is consumed. An explicit `template_kwargs["thinking"]` wins; otherwise fall
/// back to `params.thinking` (resolved from `reasoning_effort` / Anthropic
/// `ThinkingConfig`) — same precedence as the Jinja path. Default off, matching
/// the `ThinkingKeyName::Thinking` / `DefaultOff` contract reported here.
fn derive_thinking_mode(params: &ChatTemplateParams) -> deepseek_v32::ThinkingMode {
    if explicit_thinking(params).unwrap_or(false) {
        deepseek_v32::ThinkingMode::Thinking
    } else {
        deepseek_v32::ThinkingMode::Chat
    }
}

/// Explicit thinking preference: `template_kwargs["thinking"]` wins, else
/// `params.thinking`.
fn explicit_thinking(params: &ChatTemplateParams) -> Option<bool> {
    params
        .template_kwargs
        .and_then(|k| k.get("thinking"))
        .and_then(serde_json::Value::as_bool)
        .or(params.thinking)
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
    let thinking_mode = derive_thinking_mode(params);
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
    effort_encoding: deepseek_v4::EffortEncoding,
) -> Result<String> {
    let owned = inject_tools_into_messages(messages, params.tools);
    let msgs: &[serde_json::Value] = owned.as_deref().unwrap_or(messages);
    // Values the revision doesn't recognize are ignored, matching the
    // template-owns-interpretation contract for merged public efforts.
    let reasoning_effort = params
        .template_kwargs
        .and_then(|k| k.get("reasoning_effort"))
        .and_then(|v| v.as_str())
        .and_then(|s| effort_encoding.parse_native(s));
    // A recognized effort implies thinking; an explicit toggle still wins.
    let thinking_mode = if explicit_thinking(params).unwrap_or_else(|| reasoning_effort.is_some()) {
        deepseek_v32::ThinkingMode::Thinking
    } else {
        deepseek_v32::ThinkingMode::Chat
    };
    let encode_params = deepseek_v4::EncodeParams {
        add_default_bos_token: true,
        drop_thinking: resolve_drop_thinking(msgs),
        reasoning_effort,
        effort_encoding,
    };
    deepseek_v4::encode_messages(msgs, thinking_mode, &encode_params)
        .map_err(|e| Error::msg(format!("DeepSeek V4 encode failed: {e}")))
}

// ---------------------------------------------------------------------------
// DeepSeek V4.1 dispatch shim
// ---------------------------------------------------------------------------
/// Attach `tools` to the FIRST message whose role is `system`, wherever it
/// appears in the conversation — vLLM's V4.1 rule. This differs from V3.2/V4's
/// [`inject_tools_into_messages`], which only rewrites a *leading*
/// system/developer message. Synthesizes an empty leading system message when
/// none exists.
fn inject_tools_into_first_system_message(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
) -> Option<Vec<serde_json::Value>> {
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }
    let mut owned: Vec<serde_json::Value> = messages.to_vec();
    let system_index = owned
        .iter()
        .position(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"));
    let index = system_index.unwrap_or_else(|| {
        owned.insert(0, serde_json::json!({ "role": "system", "content": "" }));
        0
    });
    if let Some(obj) = owned[index].as_object_mut() {
        obj.insert("tools".into(), serde_json::Value::Array(tools.to_vec()));
    }
    Some(owned)
}

/// A V4.1 boolean template kwarg (`thinking`, `drop_thinking`): `None` when
/// absent or JSON `null`; a present value that isn't a JSON boolean is an
/// error naming the key and the value. Unlike V3.2/V4's [`explicit_thinking`],
/// which silently ignores a wrongly typed value.
fn boolean_kwarg_v41(params: &ChatTemplateParams, key: &str) -> Result<Option<bool>> {
    match params.template_kwargs.and_then(|k| k.get(key)) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(value)) => Ok(Some(*value)),
        Some(other) => Err(Error::msg(format!(
            "DeepSeek V4.1: template_kwargs[\"{key}\"] must be a boolean, got {other}"
        ))),
    }
}

/// V4.1's explicit thinking toggle: `template_kwargs["thinking"]` (the key
/// this tokenizer reports through `thinking_key_name()`) or vLLM's
/// `enable_thinking` alias, both read with the strict [`boolean_kwarg_v41`]
/// rule. The gateway reads the same two keys when it arms the reasoning
/// parser (`renderer_capabilities().enable_thinking_alias`), so the prompt
/// and the parser never disagree; both present and different is an error.
fn explicit_thinking_v41(params: &ChatTemplateParams) -> Result<Option<bool>> {
    let thinking = boolean_kwarg_v41(params, "thinking")?;
    let alias = boolean_kwarg_v41(params, "enable_thinking")?;
    match (thinking, alias) {
        (Some(a), Some(b)) if a != b => Err(Error::msg(format!(
            "DeepSeek V4.1: template_kwargs[\"thinking\"] = {a} and \
             template_kwargs[\"enable_thinking\"] = {b} disagree"
        ))),
        (Some(value), _) | (None, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

/// The gateway deserialises a top-level JSON number (`"reasoning_effort": 42`)
/// into the string `"42"` before forwarding it as a template kwarg. Restore
/// the number so [`deepseek_v41::parse_reasoning_effort`] sees the integer
/// budget the client sent. Only that exact form, a non-empty string of ASCII
/// digits, is restored: a signed or padded `"+42"` / `" 42 "` passes through
/// untouched and is rejected there as the string it is, and an out-of-range
/// `"101"` is restored and rejected there with the message a JSON number gets.
fn restore_integer_reasoning_effort(value: &serde_json::Value) -> Option<serde_json::Value> {
    let digits = value.as_str()?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok().map(serde_json::Value::from)
}

/// DeepSeek V4.1 chat-template shim. Order: attach tools to the first system
/// message (vLLM's rule) -> resolve `reasoning_effort` -> resolve the
/// thinking mode -> `drop_thinking` -> stamp `wo_eos` on a trailing assistant
/// message when `add_generation_prompt` is false -> encode.
///
/// `add_generation_prompt: false` with a trailing assistant message is the
/// encoder's `wo_eos` route (no EOS, no generation header). The gateway sends
/// it for `continue_final_message` with a trailing assistant turn because
/// `renderer_capabilities().native_assistant_continuation` is declared, and
/// does not arm the reasoning parser for that shape: the message is rendered
/// past its `</think>`, so the completion starts in content mode.
///
/// The shim reads the merged template kwargs, so an explicit
/// `chat_template_kwargs.reasoning_effort` wins over the projected top-level
/// `reasoning_effort` (SMG's global contract; vLLM prefers the top-level
/// field — a divergence only on contradictory requests).
///
/// The thinking mode mirrors the gateway's parser-arming precedence
/// (`resolve_thinking_pref` in `model_gateway/src/routers/grpc/utils/parsers.rs`)
/// so the rendered prompt and the arming decision agree on every request
/// shape:
/// 1. an explicit `template_kwargs["thinking"]` boolean decides;
/// 2. else the `reasoning_effort` kwarg: `"none"`/`"minimal"` (the gateway's
///    thinking switch, `thinking_from_reasoning_effort`) switch thinking off,
///    a native effort name (`low`/`high`/`xhigh`/`max`) switches it on, and
///    an integer budget has no opinion;
/// 3. else `params.thinking` (the gateway's projection of the top-level
///    `reasoning_effort`: `Some(false)` for `none`/`minimal`);
/// 4. else on ([`ThinkingToggle::DefaultOn`]).
///
/// Deliberate divergence from vLLM's Python: there `reasoning_effort: "none"`
/// forces chat mode even over an explicit `thinking: true`. Here the explicit
/// toggle wins, because the gateway arms the reasoning parser from the
/// explicit toggle first, and rendering chat mode for that contradictory input
/// would have the armed parser swallow the whole answer as reasoning. Neither
/// switch value reaches `parse_reasoning_effort` (which rejects both, as the
/// reference does) and both leave the effort unset, so a thinking-mode prompt
/// carries the default budget.
fn apply_deepseek_v41(
    messages: &[serde_json::Value],
    params: &ChatTemplateParams,
) -> Result<String> {
    // `template_kwargs["response_format"]` is deliberately not attached to a
    // message: vLLM Python never renders V4.1's `## Response Format:` block
    // (structured output is enforced by the constraint), so the gateway's
    // projected kwarg is ignored here too. The encoder still renders a
    // per-message `response_format` for direct callers.
    let owned = inject_tools_into_first_system_message(messages, params.tools);
    let mut msgs: Vec<serde_json::Value> = owned.unwrap_or_else(|| messages.to_vec());

    let effort_kwarg = params
        .template_kwargs
        .and_then(|k| k.get("reasoning_effort"));
    let effort_name = effort_kwarg.and_then(serde_json::Value::as_str);
    // `"none"`/`"minimal"` are the gateway's thinking switch (both project to
    // thinking off everywhere else in SMG), not effort levels.
    let effort_is_none = matches!(effort_name, Some("none") | Some("minimal"));
    let reasoning_effort = if effort_is_none {
        None
    } else {
        let restored = effort_kwarg.and_then(restore_integer_reasoning_effort);
        restored
            .as_ref()
            .or(effort_kwarg)
            .map(deepseek_v41::parse_reasoning_effort)
            .transpose()
            .map_err(|e| Error::msg(format!("DeepSeek V4.1 reasoning_effort invalid: {e}")))?
            .flatten()
    };
    // The effort kwarg's opinion on the mode (step 2 above). Only the names
    // advertised through `native_reasoning_effort_values()` switch thinking
    // on: exactly the set the gateway treats as arming the parser.
    let effort_mode = if effort_is_none {
        Some(false)
    } else {
        effort_name
            .is_some_and(|name| deepseek_v41::NATIVE_EFFORT_VALUES.contains(&name))
            .then_some(true)
    };
    let thinking_on = explicit_thinking_v41(params)?
        .or(effort_mode)
        .or(params.thinking)
        .unwrap_or(true);
    let thinking_mode = if thinking_on {
        deepseek_v32::ThinkingMode::Thinking
    } else {
        deepseek_v32::ThinkingMode::Chat
    };

    let drop_thinking = boolean_kwarg_v41(params, "drop_thinking")?.unwrap_or(true);

    // `add_generation_prompt: false` with a trailing assistant message reaches
    // the encoder as `wo_eos` on that message: no EOS, no generation header.
    // That is the only shape wired: with any other trailing role the flag is
    // ignored and the generation header is appended (V3.2/V4 ignore the flag
    // entirely).
    let continues_final_assistant_message = !params.add_generation_prompt
        && msgs
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("assistant");
    if continues_final_assistant_message {
        if let Some(obj) = msgs.last_mut().and_then(serde_json::Value::as_object_mut) {
            obj.insert("wo_eos".into(), serde_json::Value::Bool(true));
        }
    }

    let encode_params = deepseek_v41::EncodeParams {
        add_default_bos_token: true,
        drop_thinking,
        reasoning_effort,
    };
    deepseek_v41::encode_messages(&msgs, thinking_mode, &encode_params)
        .map_err(|e| Error::msg(format!("DeepSeek V4.1 encode failed: {e}")))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::derive_thinking_mode;
    use crate::{chat_template::ChatTemplateParams, encoders::deepseek_v32::ThinkingMode};

    fn thinking_kwargs(value: bool) -> HashMap<String, serde_json::Value> {
        HashMap::from([("thinking".to_string(), serde_json::Value::Bool(value))])
    }

    // Regression: DeepSeek V3.2/V4 bypass ChatTemplateState::apply, so the
    // resolved `params.thinking` (from reasoning_effort / Anthropic ThinkingConfig)
    // must be honored here — with an explicit `template_kwargs["thinking"]` still
    // winning. Same precedence as the Jinja path.
    #[test]
    fn derive_thinking_mode_honors_params_thinking_and_explicit_override() {
        // No signal at all -> Chat (default off).
        assert!(matches!(
            derive_thinking_mode(&ChatTemplateParams::default()),
            ThinkingMode::Chat
        ));

        // params.thinking is the fallback when there is no explicit kwarg.
        assert!(matches!(
            derive_thinking_mode(&ChatTemplateParams {
                thinking: Some(true),
                ..Default::default()
            }),
            ThinkingMode::Thinking
        ));
        assert!(matches!(
            derive_thinking_mode(&ChatTemplateParams {
                thinking: Some(false),
                ..Default::default()
            }),
            ThinkingMode::Chat
        ));

        // An explicit template_kwargs["thinking"] wins over params.thinking.
        let on = thinking_kwargs(true);
        assert!(matches!(
            derive_thinking_mode(&ChatTemplateParams {
                thinking: Some(false),
                template_kwargs: Some(&on),
                ..Default::default()
            }),
            ThinkingMode::Thinking
        ));
        let off = thinking_kwargs(false);
        assert!(matches!(
            derive_thinking_mode(&ChatTemplateParams {
                thinking: Some(true),
                template_kwargs: Some(&off),
                ..Default::default()
            }),
            ThinkingMode::Chat
        ));
    }
}
