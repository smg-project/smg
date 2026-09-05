use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{Error, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rustc_hash::FxHashMap;
use tiktoken_rs::{
    cl100k_base, o200k_base, p50k_base, p50k_edit, r50k_base,
    tokenizer::{get_tokenizer, Tokenizer},
    CoreBPE, DecodeKeyError,
};

use crate::{
    chat_template::{
        load_chat_template_from_file, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateState, ThinkingKeyName, ThinkingToggle,
    },
    encoders::{
        kimi_k25_tools::apply_kimi_k25_tools,
        kimi_k3_xtml::{
            apply_kimi_k3_xtml_with_effort_default,
            render_kimi_k3_xtml_segments_with_effort_default,
        },
    },
    factory::discover_chat_template_in_dir,
    kimi_k2_tokenizer,
    traits::{
        Decoder, Encoder, Encoding, PromptSegment, SpecialTokens, TokenIdType,
        Tokenizer as TokenizerTrait,
    },
};

#[derive(Debug, Clone, Copy)]
enum Renderer {
    Jinja,
    KimiK25Tools,
    /// Kimi-K3 renders its prompt entirely in the native XTML encoder (no Jinja
    /// chat template). See [`crate::encoders::kimi_k3_xtml`].
    KimiK3Xtml,
}

/// Regex pattern for cl100k_base tokenization.
///
/// This pattern is correct for OpenAI models and most open-source tiktoken models. Models
/// with a tokenizer-specific regex specialize the pattern inside `load_from_path`.
const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

type Rank = u32;

// ---------------------------------------------------------------------------
// Tiktoken-specific config parsing (from tokenizer_config.json)
// ---------------------------------------------------------------------------

/// Parsed `tokenizer_config.json` for tiktoken-based models.
#[derive(Default)]
struct TiktokenConfig {
    special_tokens: SpecialTokens,
    /// Token string -> ID mapping from `added_tokens_decoder`
    added_tokens: HashMap<String, TokenIdType>,
    /// Ids `skip_special_tokens` removes from decoded text.
    skip_token_ids: HashSet<TokenIdType>,
    chat_template: Option<String>,
}

/// Parse an already-loaded `tokenizer_config.json` value into a `TiktokenConfig`.
fn parse_tiktoken_config(value: &serde_json::Value) -> TiktokenConfig {
    TiktokenConfig {
        special_tokens: parse_special_tokens(value),
        added_tokens: parse_added_tokens_decoder(value),
        skip_token_ids: parse_skip_token_ids(value),
        chat_template: value
            .get("chat_template")
            .and_then(|v| v.as_str())
            .map(String::from),
    }
}

/// Ids of `added_tokens_decoder` entries flagged `"special": true`: the set
/// `skip_special_tokens` strips, as HuggingFace defines it. Added tokens
/// without the flag (Kimi-K3's `<|open|>` / `<|close|>` / `<|sep|>`) are
/// control tokens for encoding but stay in decoded text.
fn parse_skip_token_ids(config: &serde_json::Value) -> HashSet<TokenIdType> {
    let mut ids = HashSet::new();
    if let Some(added) = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
    {
        for (id_str, token_info) in added {
            let special = token_info
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if let (true, Ok(id)) = (special, id_str.parse::<TokenIdType>()) {
                ids.insert(id);
            }
        }
    }
    ids
}

/// Load `tokenizer_config.json` from `dir`, returning both the parsed
/// `TiktokenConfig` and the raw JSON value (so callers like Kimi detection
/// can inspect the same parse without re-reading the file).
fn load_tiktoken_config_from_dir(
    dir: &Path,
) -> Result<(TiktokenConfig, Option<serde_json::Value>)> {
    let config_path = dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok((TiktokenConfig::default(), None));
    }
    let content = std::fs::read_to_string(&config_path)?;
    let value: serde_json::Value = serde_json::from_str(&content)?;
    let config = parse_tiktoken_config(&value);
    Ok((config, Some(value)))
}

/// Parse `added_tokens_decoder` from config JSON.
///
/// Format: `{ "163584": { "content": "[BOS]", "special": true }, ... }`
fn parse_added_tokens_decoder(config: &serde_json::Value) -> HashMap<String, TokenIdType> {
    let mut tokens = HashMap::new();
    if let Some(added) = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
    {
        for (id_str, token_info) in added {
            if let (Ok(id), Some(content)) = (
                id_str.parse::<TokenIdType>(),
                token_info.get("content").and_then(|v| v.as_str()),
            ) {
                tokens.insert(content.to_string(), id);
            }
        }
    }
    tokens
}

/// Extract named special tokens (bos, eos, unk, etc.) from config JSON.
///
/// Handles both string-valued tokens (`"bos_token": "<s>"`) and object-valued tokens
/// (`"bos_token": {"content": "<s>", "lstrip": false, ...}`) found in some HuggingFace models.
fn parse_special_tokens(config: &serde_json::Value) -> SpecialTokens {
    let get_str = |key: &str| {
        config.get(key).and_then(|v| {
            v.as_str()
                .map(String::from)
                .or_else(|| v.get("content").and_then(|c| c.as_str()).map(String::from))
        })
    };

    let additional: Vec<String> = config
        .get("additional_special_tokens")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    v.as_str()
                        .map(String::from)
                        .or_else(|| v.get("content").and_then(|c| c.as_str()).map(String::from))
                })
                .collect()
        })
        .unwrap_or_default();

    SpecialTokens {
        bos_token: get_str("bos_token"),
        eos_token: get_str("eos_token"),
        unk_token: get_str("unk_token"),
        sep_token: get_str("sep_token"),
        pad_token: get_str("pad_token"),
        cls_token: get_str("cls_token"),
        mask_token: get_str("mask_token"),
        additional_special_tokens: additional,
    }
}

/// Tiktoken tokenizer wrapper — supports both built-in OpenAI encodings and hub-loaded models.
pub struct TiktokenTokenizer {
    tokenizer: CoreBPE,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    reverse_vocab: HashMap<TokenIdType, String>,
    vocab_size: usize,
    chat_template: ChatTemplateState,
    eos_token_ids: Vec<TokenIdType>,
    skip_token_ids: HashSet<TokenIdType>,
    renderer: Renderer,
}

/// Supported Tiktoken models
#[derive(Debug, Clone, Copy)]
pub enum TiktokenModel {
    /// GPT-4o, o1, o3, o4, GPT-4.5, GPT-5 — all 200k-vocab models
    O200kBase,
    /// GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    Cl100kBase,
    /// Codex models, text-davinci-002, text-davinci-003
    P50kBase,
    /// Use for edit models like text-davinci-edit-001, code-davinci-edit-001
    P50kEdit,
    /// GPT-3 models like davinci
    R50kBase,
}

impl TiktokenTokenizer {
    /// Create a new Tiktoken tokenizer for the specified built-in model
    pub fn new(model: TiktokenModel) -> Result<Self> {
        let tokenizer =
            match model {
                TiktokenModel::O200kBase => o200k_base()
                    .map_err(|e| Error::msg(format!("Failed to load o200k_base: {e}")))?,
                TiktokenModel::Cl100kBase => cl100k_base()
                    .map_err(|e| Error::msg(format!("Failed to load cl100k_base: {e}")))?,
                TiktokenModel::P50kBase => {
                    p50k_base().map_err(|e| Error::msg(format!("Failed to load p50k_base: {e}")))?
                }
                TiktokenModel::P50kEdit => {
                    p50k_edit().map_err(|e| Error::msg(format!("Failed to load p50k_edit: {e}")))?
                }
                TiktokenModel::R50kBase => {
                    r50k_base().map_err(|e| Error::msg(format!("Failed to load r50k_base: {e}")))?
                }
            };

        let special_tokens = Self::get_special_tokens_for_model(model);

        let vocab_size = match model {
            TiktokenModel::O200kBase => 200019,
            TiktokenModel::Cl100kBase => 100256,
            TiktokenModel::P50kBase | TiktokenModel::P50kEdit => 50281,
            TiktokenModel::R50kBase => 50257,
        };

        Ok(TiktokenTokenizer {
            tokenizer,
            special_tokens,
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            vocab_size,
            chat_template: ChatTemplateState::empty(),
            eos_token_ids: Vec::new(), // No directory path in from_model
            skip_token_ids: HashSet::new(),
            renderer: Renderer::Jinja,
        })
    }

    /// Create from a directory containing tiktoken.model + tokenizer_config.json
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_with_chat_template(dir, None)
    }

    /// Create from a directory with an optional chat template file path.
    /// Discovers the tiktoken model file automatically via `find_tiktoken_file`.
    pub fn from_dir_with_chat_template(
        dir: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let tiktoken_path = find_tiktoken_file(dir)?;
        Self::load_from_path(&tiktoken_path, chat_template_path)
    }

    /// Create from an exact tiktoken file path (`.tiktoken` or `tiktoken.model`).
    /// Looks for `tokenizer_config.json` in the same directory.
    pub fn from_file(tiktoken_path: &Path) -> Result<Self> {
        Self::from_file_with_chat_template(tiktoken_path, None)
    }

    /// Create from an exact tiktoken file path with an optional chat template.
    pub fn from_file_with_chat_template(
        tiktoken_path: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        Self::load_from_path(tiktoken_path, chat_template_path)
    }

    /// Core loading logic shared by `from_dir` and `from_file` constructors.
    fn load_from_path(tiktoken_path: &Path, chat_template_path: Option<&str>) -> Result<Self> {
        // 1. Load BPE encoder from the exact file
        let tiktoken_path_str = tiktoken_path
            .to_str()
            .ok_or_else(|| Error::msg("Tiktoken file path is not valid UTF-8"))?;
        let encoder = load_tiktoken_bpe(tiktoken_path_str)?;

        // 2. Parse tokenizer_config.json from the same directory
        let dir = tiktoken_path
            .parent()
            .ok_or_else(|| Error::msg("Cannot determine parent directory of tiktoken file"))?;
        let (mut config, tokenizer_config_value) = load_tiktoken_config_from_dir(dir)?;

        // Kimi-K2/K2.5/K2.6 specialize the regex and pre-fill 256 reserved
        // special-token slots starting at `len(mergeable_ranks)`; all other
        // tiktoken models use the cl100k pattern unchanged. Reuse the
        // already-parsed tokenizer_config.json so we don't re-read it.
        let pattern = if kimi_k2_tokenizer::matches(tokenizer_config_value.as_ref(), dir) {
            kimi_k2_tokenizer::apply_reserved_special_tokens(
                &mut config.added_tokens,
                encoder.len(),
            );
            kimi_k2_tokenizer::KIMI_K2_PATTERN
        } else {
            CL100K_BASE_PATTERN
        };

        // 3. Build special tokens encoder for CoreBPE (needs FxHashMap)
        let special_tokens_encoder: FxHashMap<String, Rank> = config
            .added_tokens
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        // 4. Calculate true vocab size from max token ID (handles sparse/reserved IDs),
        //    build string-based vocab maps (borrows encoder), then pass encoder by value to CoreBPE
        let vocab_size = encoder
            .values()
            .copied()
            .chain(special_tokens_encoder.values().copied())
            .max()
            .map(|id| id as usize + 1)
            .unwrap_or(0);
        let (vocab, reverse_vocab) = build_vocab_maps(&encoder, &config.added_tokens);
        let tokenizer = CoreBPE::new(encoder, special_tokens_encoder, pattern)?;

        // 5. Load chat template — propagate errors for explicit paths,
        //    silently fall back for auto-discovery
        let chat_template = if let Some(p) = chat_template_path {
            load_chat_template_from_file(p)?
        } else {
            config.chat_template.or_else(|| {
                discover_chat_template_in_dir(dir)
                    .and_then(|p| load_chat_template_from_file(&p).ok().flatten())
            })
        };

        // Load merged EOS token IDs from config.json + generation_config.json
        let eos_token_ids = crate::eos::load_eos_token_ids(dir);

        // Detect which chat-template renderer to use based on config.json::architectures
        let renderer = detect_renderer_from_config(dir);

        Ok(TiktokenTokenizer {
            tokenizer,
            special_tokens: config.special_tokens,
            vocab,
            reverse_vocab,
            vocab_size,
            chat_template: ChatTemplateState::new(chat_template)?,
            eos_token_ids,
            skip_token_ids: config.skip_token_ids,
            renderer,
        })
    }

    /// Create a tokenizer from a model string (e.g., "gpt-4", "gpt-3.5-turbo")
    pub fn from_model_name(model_name: &str) -> Result<Self> {
        let bare = model_name.rsplit('/').next().unwrap_or(model_name);
        let model = match get_tokenizer(bare) {
            Some(Tokenizer::O200kBase) => TiktokenModel::O200kBase,
            Some(Tokenizer::Cl100kBase) => TiktokenModel::Cl100kBase,
            Some(Tokenizer::P50kBase) => TiktokenModel::P50kBase,
            Some(Tokenizer::P50kEdit) => TiktokenModel::P50kEdit,
            Some(Tokenizer::R50kBase) => TiktokenModel::R50kBase,
            _ => return Err(anyhow::anyhow!(
                "Unrecognized OpenAI model name: '{model_name}'. Expected GPT-3, GPT-3.5, GPT-4, GPT-4o, GPT-4.5, GPT-5, o1, o3, o4, or related model names"
            )),
        };
        Self::new(model)
    }

    /// Get special tokens for a specific model
    fn get_special_tokens_for_model(model: TiktokenModel) -> SpecialTokens {
        match model {
            TiktokenModel::Cl100kBase => SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                sep_token: None,
                pad_token: Some("<|endoftext|>".to_string()),
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![
                    "<|fim_prefix|>".to_string(),
                    "<|fim_middle|>".to_string(),
                    "<|fim_suffix|>".to_string(),
                    "<|endofprompt|>".to_string(),
                ],
            },
            _ => SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                sep_token: None,
                pad_token: Some("<|endoftext|>".to_string()),
                cls_token: None,
                mask_token: None,
                additional_special_tokens: vec![],
            },
        }
    }
}

/// Parse a .tiktoken / tiktoken.model file into a BPE encoder.
///
/// Format: each line is `<base64-encoded-token-bytes> <rank>`
fn load_tiktoken_bpe(path: &str) -> Result<FxHashMap<Vec<u8>, Rank>> {
    let content = std::fs::read_to_string(path)?;
    let mut encoder =
        FxHashMap::with_capacity_and_hasher(content.lines().count(), Default::default());
    for line in content.lines() {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let token_b64 = parts
            .next()
            .ok_or_else(|| Error::msg("missing token in tiktoken file"))?;
        let rank_str = parts
            .next()
            .ok_or_else(|| Error::msg("missing rank in tiktoken file"))?;
        let token_bytes = STANDARD.decode(token_b64)?;
        let rank: Rank = rank_str.parse()?;
        encoder.insert(token_bytes, rank);
    }
    Ok(encoder)
}

/// Build string-level vocab from byte-level encoder + added tokens.
fn build_vocab_maps(
    encoder: &FxHashMap<Vec<u8>, Rank>,
    added_tokens: &HashMap<String, TokenIdType>,
) -> (HashMap<String, TokenIdType>, HashMap<TokenIdType, String>) {
    let capacity = encoder.len() + added_tokens.len();
    let mut vocab = HashMap::with_capacity(capacity);
    let mut reverse_vocab = HashMap::with_capacity(capacity);

    // BPE tokens (only valid UTF-8 sequences get string entries)
    for (token_bytes, &rank) in encoder {
        if let Ok(token_str) = std::str::from_utf8(token_bytes) {
            vocab.insert(token_str.to_string(), rank);
            reverse_vocab.insert(rank, token_str.to_string());
        }
    }

    // Special/added tokens (always valid UTF-8)
    for (token_str, &id) in added_tokens {
        vocab.insert(token_str.clone(), id);
        reverse_vocab.insert(id, token_str.clone());
    }

    (vocab, reverse_vocab)
}

/// Find a tiktoken model file in the given directory.
///
/// Looks for `tiktoken.model` first, then any `*.tiktoken` file.
fn find_tiktoken_file(dir: &Path) -> Result<PathBuf> {
    let tiktoken_model = dir.join("tiktoken.model");
    if tiktoken_model.exists() {
        return Ok(tiktoken_model);
    }

    // Look for *.tiktoken files
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".tiktoken") {
                    return Ok(entry.path());
                }
            }
        }
    }

    Err(Error::msg(format!(
        "No tiktoken model file found in '{}'",
        dir.display()
    )))
}

/// Check whether a directory contains a tiktoken model file.
pub fn has_tiktoken_file(dir: &Path) -> bool {
    if dir.join("tiktoken.model").exists() {
        return true;
    }
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with(".tiktoken"))
            })
        })
        .unwrap_or(false)
}

/// Check whether a single file is a tiktoken model file (by name).
pub fn is_tiktoken_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == "tiktoken.model" || name.ends_with(".tiktoken"))
}

impl Encoder for TiktokenTokenizer {
    fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
        // tiktoken ignores `add_special_tokens` (it means BOS/EOS prepend on HF
        // backends, which tiktoken has no concept of) and always recognizes
        // special-token patterns, so chat-template tokens like <|media_pad|> stay
        // atomic instead of splitting into BPE sub-tokens.
        let tokens = self.tokenizer.encode_with_special_tokens(input);
        Ok(Encoding::Tiktoken(tokens))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }

    fn encode_segments(&self, segments: &[PromptSegment]) -> Result<Encoding> {
        let mut ids = Vec::new();
        for segment in segments {
            if segment.allow_special {
                ids.extend(self.tokenizer.encode_with_special_tokens(&segment.text));
            } else {
                ids.extend(self.tokenizer.encode_ordinary(&segment.text));
            }
        }
        Ok(Encoding::Tiktoken(ids))
    }
}

impl Decoder for TiktokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        let kept: Vec<TokenIdType>;
        let token_ids: &[TokenIdType] = if skip_special_tokens && !self.skip_token_ids.is_empty() {
            kept = token_ids
                .iter()
                .copied()
                .filter(|id| !self.skip_token_ids.contains(id))
                .collect();
            &kept
        } else {
            token_ids
        };
        match self.tokenizer.decode(token_ids) {
            Ok(text) => Ok(text),
            Err(err) if is_unknown_tiktoken_decode_error(&err) => Err(Error::msg(format!(
                "tiktoken decode failed for unknown token id: {err}"
            ))),
            Err(err) => {
                // Fallback to lossy decoding for incomplete UTF-8 sequences
                let bytes: Vec<u8> = self
                    .tokenizer
                    ._decode_native_and_split(token_ids.to_vec())
                    .flatten()
                    .collect();
                tracing::warn!(
                    error = %err,
                    token_count = token_ids.len(),
                    "tiktoken decode failed; returning lossy UTF-8 fallback"
                );
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
        }
    }
}

/// Detect tiktoken's "unknown token id" error so we can surface a clean error
/// instead of letting the lossy-decode fallback panic on a missing key.
///
/// `CoreBPE::decode` surfaces a missing-token id as a `DecodeKeyError` (now
/// re-exported as of tiktoken-rs 0.12), while UTF-8 failures arrive as a plain
/// string error — so a typed `downcast_ref` cleanly separates the two cases.
fn is_unknown_tiktoken_decode_error(err: &Error) -> bool {
    err.downcast_ref::<DecodeKeyError>().is_some()
}

impl TokenizerTrait for TiktokenTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size
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

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        // Inject special tokens if the caller didn't provide them
        let params = if params.special_tokens.is_some() {
            params
        } else {
            ChatTemplateParams {
                special_tokens: Some(&self.special_tokens),
                ..params
            }
        };
        match self.renderer {
            Renderer::Jinja => self.chat_template.apply(messages, params),
            Renderer::KimiK25Tools => apply_kimi_k25_tools(&self.chat_template, messages, params),
            // This is the layer the checkpoint's own `apply_chat_template` sits
            // at, so it applies that wrapper's `thinking_effort` default.
            Renderer::KimiK3Xtml => apply_kimi_k3_xtml_with_effort_default(messages, &params),
        }
    }

    fn apply_chat_template_segments(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<Vec<PromptSegment>> {
        match self.renderer {
            Renderer::KimiK3Xtml => {
                render_kimi_k3_xtml_segments_with_effort_default(messages, &params)
            }
            _ => Ok(vec![PromptSegment::control(
                self.apply_chat_template(messages, params)?,
            )]),
        }
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.chat_template.content_format()
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        match self.renderer {
            // K3 has no Jinja template; its native renderer defaults thinking
            // ON (Python `build_chat_segments(thinking=True)`).
            Renderer::KimiK3Xtml => ThinkingToggle::DefaultOn,
            _ => self.chat_template.thinking_toggle(),
        }
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        match self.renderer {
            Renderer::KimiK3Xtml => Some(ThinkingKeyName::Thinking),
            _ => self.chat_template.thinking_key_name(),
        }
    }
    fn eos_token_ids(&self) -> &[TokenIdType] {
        &self.eos_token_ids
    }

    fn think_in_prefill(&self) -> bool {
        match self.renderer {
            // K3's generation-prompt tail opens `<think>` when thinking is on
            // (the default), so completions start mid-reasoning.
            Renderer::KimiK3Xtml => true,
            _ => self.chat_template.think_in_prefill(),
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
/// use. Missing / unreadable / malformed config falls back to `Renderer::Jinja`
/// silently with a debug log, mirroring `huggingface.rs::detect_renderer_from_config`.
fn detect_renderer_from_config(dir: &Path) -> Renderer {
    let path = dir.join("config.json");
    if !path.exists() {
        return Renderer::Jinja;
    }
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(err) => {
            tracing::debug!(?err, ?path, "config.json unreadable; using Jinja renderer");
            return Renderer::Jinja;
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(err) => {
            tracing::debug!(?err, ?path, "config.json malformed; using Jinja renderer");
            return Renderer::Jinja;
        }
    };
    let arch_strs: Vec<&str> = value
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if arch_strs.contains(&"KimiK3ForConditionalGeneration") {
        tracing::debug!(?path, "selected KimiK3Xtml chat-template renderer");
        return Renderer::KimiK3Xtml;
    }
    if arch_strs.contains(&"KimiK25ForConditionalGeneration") {
        tracing::debug!(?path, "selected KimiK25Tools chat-template renderer");
        return Renderer::KimiK25Tools;
    }
    // Fall back to `model_type` when `architectures` is absent or unrecognized.
    // Some Kimi checkpoints omit the architecture entry but still carry a
    // `model_type`, and `is_kimi_tokenizer` already keys off it — keep the
    // renderer selection consistent with that tokenizer detection.
    match value.get("model_type").and_then(|v| v.as_str()) {
        Some("kimi_k3") => {
            tracing::debug!(?path, "selected KimiK3Xtml renderer via model_type");
            Renderer::KimiK3Xtml
        }
        Some("kimi_k25") => {
            tracing::debug!(?path, "selected KimiK25Tools renderer via model_type");
            Renderer::KimiK25Tools
        }
        _ => Renderer::Jinja,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Decoder, Encoder, Tokenizer};

    const MINIMAL_TIKTOKEN_MODEL: &str = "YQ== 0\nYg== 1\n";

    fn write_minimal_tiktoken_dir(
        tokenizer_config: &str,
        model_config: Option<&str>,
    ) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), MINIMAL_TIKTOKEN_MODEL).unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), tokenizer_config).unwrap();
        if let Some(model_config) = model_config {
            std::fs::write(dir.path().join("config.json"), model_config).unwrap();
        }
        dir
    }

    #[test]
    fn test_detect_renderer_from_config() {
        let k3 = write_minimal_tiktoken_dir(
            "{}",
            Some(r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#),
        );
        assert!(matches!(
            detect_renderer_from_config(k3.path()),
            Renderer::KimiK3Xtml
        ));

        let k25 = write_minimal_tiktoken_dir(
            "{}",
            Some(r#"{"architectures": ["KimiK25ForConditionalGeneration"]}"#),
        );
        assert!(matches!(
            detect_renderer_from_config(k25.path()),
            Renderer::KimiK25Tools
        ));

        let other =
            write_minimal_tiktoken_dir("{}", Some(r#"{"architectures": ["LlamaForCausalLM"]}"#));
        assert!(matches!(
            detect_renderer_from_config(other.path()),
            Renderer::Jinja
        ));

        let missing = write_minimal_tiktoken_dir("{}", None);
        assert!(matches!(
            detect_renderer_from_config(missing.path()),
            Renderer::Jinja
        ));

        // `model_type` fallback: checkpoints without a recognized architecture
        // entry are still detected by their `model_type`, matching
        // `is_kimi_tokenizer`'s detection.
        let k3_model_type = write_minimal_tiktoken_dir("{}", Some(r#"{"model_type": "kimi_k3"}"#));
        assert!(matches!(
            detect_renderer_from_config(k3_model_type.path()),
            Renderer::KimiK3Xtml
        ));

        let k25_model_type =
            write_minimal_tiktoken_dir("{}", Some(r#"{"model_type": "kimi_k25"}"#));
        assert!(matches!(
            detect_renderer_from_config(k25_model_type.path()),
            Renderer::KimiK25Tools
        ));
    }

    #[test]
    fn test_tiktoken_creation() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        assert_eq!(tokenizer.vocab_size(), 100256);
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();

        let text = "Hello, world!";
        let encoding = tokenizer.encode(text, false).unwrap();

        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_batch_encode() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let encodings = tokenizer.encode_batch(&texts, false).unwrap();

        assert_eq!(encodings.len(), 3);
        for (i, encoding) in encodings.iter().enumerate() {
            let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
            assert_eq!(decoded, texts[i]);
        }
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        let special_tokens = tokenizer.get_special_tokens();

        assert!(special_tokens.eos_token.is_some());
        assert_eq!(special_tokens.eos_token.as_ref().unwrap(), "<|endoftext|>");
    }

    #[test]
    fn test_builtin_tokenizer_has_empty_vocab_maps() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        // Built-in path: vocab maps are empty, token_to_id returns None
        assert_eq!(tokenizer.token_to_id("hello"), None);
        assert_eq!(tokenizer.id_to_token(0), None);
    }

    #[test]
    fn test_load_tiktoken_bpe() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.tiktoken");
        let mut f = std::fs::File::create(&file_path).unwrap();
        // "IQ==" is base64 for byte 0x21 ('!'), rank 0
        // "Ig==" is base64 for byte 0x22 ('"'), rank 1
        writeln!(f, "IQ== 0").unwrap();
        writeln!(f, "Ig== 1").unwrap();

        let encoder = load_tiktoken_bpe(file_path.to_str().unwrap()).unwrap();
        assert_eq!(encoder.len(), 2);
        assert_eq!(encoder.get(&vec![0x21u8]), Some(&0));
        assert_eq!(encoder.get(&vec![0x22u8]), Some(&1));
    }

    #[test]
    fn test_build_vocab_maps() {
        let mut encoder = FxHashMap::default();
        encoder.insert(b"hello".to_vec(), 42u32);
        encoder.insert(vec![0xFF, 0xFE], 99u32); // invalid UTF-8

        let mut added = HashMap::new();
        added.insert("<|special|>".to_string(), 1000u32);

        let (vocab, reverse_vocab) = build_vocab_maps(&encoder, &added);

        // Valid UTF-8 token present
        assert_eq!(vocab.get("hello"), Some(&42));
        assert_eq!(reverse_vocab.get(&42), Some(&"hello".to_string()));

        // Invalid UTF-8 token excluded from vocab
        assert!(!vocab.contains_key("\u{FFFD}")); // not lossy-inserted

        // Added token present
        assert_eq!(vocab.get("<|special|>"), Some(&1000));
        assert_eq!(reverse_vocab.get(&1000), Some(&"<|special|>".to_string()));
    }

    #[test]
    fn test_has_tiktoken_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_tiktoken_file(dir.path()));

        std::fs::write(dir.path().join("tiktoken.model"), "test").unwrap();
        assert!(has_tiktoken_file(dir.path()));
    }

    #[test]
    fn test_find_tiktoken_file_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), "test").unwrap();
        let found = find_tiktoken_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "tiktoken.model");
    }

    #[test]
    fn test_find_tiktoken_file_extension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("vocab.tiktoken"), "test").unwrap();
        let found = find_tiktoken_file(dir.path()).unwrap();
        assert!(found
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".tiktoken"));
    }

    #[test]
    fn test_is_tiktoken_file() {
        assert!(is_tiktoken_file(Path::new("tiktoken.model")));
        assert!(is_tiktoken_file(Path::new("vocab.tiktoken")));
        assert!(!is_tiktoken_file(Path::new("tokenizer.json")));
        assert!(!is_tiktoken_file(Path::new("model.bin")));
    }

    #[test]
    fn test_parse_added_tokens_decoder() {
        let config: serde_json::Value = serde_json::json!({
            "added_tokens_decoder": {
                "163584": { "content": "[BOS]", "special": true },
                "163585": { "content": "[EOS]", "special": true },
                "163586": { "content": "<|im_end|>", "special": true }
            }
        });
        let tokens = parse_added_tokens_decoder(&config);
        assert_eq!(tokens.get("[BOS]"), Some(&163584));
        assert_eq!(tokens.get("[EOS]"), Some(&163585));
        assert_eq!(tokens.get("<|im_end|>"), Some(&163586));
    }

    #[test]
    fn test_tiktoken_unknown_token_decode_returns_error() {
        let dir = write_minimal_tiktoken_dir(
            r#"{
                "added_tokens_decoder": {
                    "2": { "content": "[BOS]", "special": true }
                }
            }"#,
            None,
        );
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();

        let err = tokenizer.decode(&[4], false).unwrap_err();
        assert!(
            err.to_string()
                .contains("tiktoken decode failed for unknown token id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_skip_special_tokens_drops_only_special_flagged_ids() {
        let dir = write_minimal_tiktoken_dir(
            r#"{
                "added_tokens_decoder": {
                    "2": { "content": "[EOS]", "special": true },
                    "3": { "content": "<|open|>", "special": false }
                }
            }"#,
            None,
        );
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();

        assert_eq!(
            tokenizer.decode(&[0, 2, 3, 1], false).unwrap(),
            "a[EOS]<|open|>b"
        );
        assert_eq!(tokenizer.decode(&[0, 2, 3, 1], true).unwrap(), "a<|open|>b");
    }

    #[test]
    fn test_skip_special_tokens_holds_text_in_incremental_decode() {
        let dir = write_minimal_tiktoken_dir(
            r#"{
                "added_tokens_decoder": {
                    "2": { "content": "[EOS]", "special": true }
                }
            }"#,
            None,
        );
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();

        let mut ids = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0;
        assert_eq!(
            tokenizer
                .decode_step(0, &mut ids, &mut prefix, &mut prefix_index, true)
                .unwrap(),
            Some("a".to_string())
        );
        assert_eq!(
            tokenizer
                .decode_step(2, &mut ids, &mut prefix, &mut prefix_index, true)
                .unwrap(),
            None
        );
        assert_eq!(
            tokenizer
                .decode_step(1, &mut ids, &mut prefix, &mut prefix_index, true)
                .unwrap(),
            Some("b".to_string())
        );
    }

    /// Every byte as its own rank, so any string is encodable as ordinary text.
    fn full_byte_tiktoken_model() -> String {
        (0u32..256)
            .map(|b| format!("{} {}\n", STANDARD.encode([b as u8]), b))
            .collect()
    }

    #[test]
    fn test_encode_segments_keeps_control_tokens_out_of_text_segments() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tiktoken.model"),
            full_byte_tiktoken_model(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {"300": {"content": "<|open|>", "special": false}}}"#,
        )
        .unwrap();
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();

        let control = tokenizer
            .encode_segments(&[PromptSegment::control("<|open|>")])
            .unwrap();
        assert_eq!(control.token_ids(), &[300]);

        let text = tokenizer
            .encode_segments(&[PromptSegment::text("<|open|>")])
            .unwrap();
        assert!(
            !text.token_ids().contains(&300),
            "marker in a text segment must not become a control id: {:?}",
            text.token_ids()
        );
        assert_eq!(
            tokenizer.decode(text.token_ids(), false).unwrap(),
            "<|open|>"
        );

        let mixed = tokenizer
            .encode_segments(&[
                PromptSegment::control("<|open|>"),
                PromptSegment::text("message"),
                PromptSegment::text("<|open|>"),
            ])
            .unwrap();
        assert_eq!(mixed.token_ids().iter().filter(|&&id| id == 300).count(), 1);
        assert_eq!(
            tokenizer.decode(mixed.token_ids(), false).unwrap(),
            "<|open|>message<|open|>"
        );

        // The flat encode still maps every marker string to its control id.
        let flat = tokenizer.encode("<|open|>message<|open|>", false).unwrap();
        assert_eq!(flat.token_ids().iter().filter(|&&id| id == 300).count(), 2);
    }

    #[test]
    fn test_parse_special_tokens() {
        let config: serde_json::Value = serde_json::json!({
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "additional_special_tokens": ["<|im_end|>", "<|im_user|>"]
        });
        let special = parse_special_tokens(&config);
        assert_eq!(special.bos_token.as_deref(), Some("[BOS]"));
        assert_eq!(special.eos_token.as_deref(), Some("[EOS]"));
        assert_eq!(special.unk_token.as_deref(), Some("[UNK]"));
        assert_eq!(special.pad_token.as_deref(), Some("[PAD]"));
        assert_eq!(special.additional_special_tokens.len(), 2);
    }

    #[test]
    fn test_parse_special_tokens_object_valued() {
        let config: serde_json::Value = serde_json::json!({
            "bos_token": {"content": "<s>", "lstrip": false, "rstrip": false, "single_word": false, "special": true},
            "eos_token": "</s>",
            "unk_token": {"content": "<unk>", "special": true}
        });
        let special = parse_special_tokens(&config);
        assert_eq!(special.bos_token.as_deref(), Some("<s>"));
        assert_eq!(special.eos_token.as_deref(), Some("</s>"));
        assert_eq!(special.unk_token.as_deref(), Some("<unk>"));
    }

    #[test]
    fn test_tiktoken_config_default() {
        let config = TiktokenConfig::default();
        assert!(config.special_tokens.bos_token.is_none());
        assert!(config.added_tokens.is_empty());
        assert!(config.chat_template.is_none());
    }

    #[test]
    fn test_load_tiktoken_config_from_dir_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let (config, value) = load_tiktoken_config_from_dir(dir.path()).unwrap();
        assert!(value.is_none());
        assert!(config.added_tokens.is_empty());
    }

    #[test]
    fn test_decode_lossy_fallback_for_invalid_utf8() {
        // cl100k_base maps individual bytes to token IDs via its byte-level BPE.
        // Encode a multi-byte UTF-8 character, then decode only a prefix of its
        // tokens so the raw bytes form an incomplete (invalid) UTF-8 sequence.
        // The old implementation would return an error; the new one should fall
        // back to lossy decoding and produce U+FFFD replacement characters.
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();

        // "😀" is U+1F600, encoded as 4 UTF-8 bytes: [0xF0, 0x9F, 0x98, 0x80].
        // With cl100k_base this encodes to multiple tokens. Taking a strict
        // subset of those tokens gives bytes that aren't valid UTF-8.
        let full_encoding = tokenizer.encode("😀", false).unwrap();
        let full_ids = full_encoding.token_ids();
        assert!(
            full_ids.len() > 1,
            "emoji should encode to multiple tokens in cl100k_base"
        );

        // Take only the first token — its raw bytes are an incomplete UTF-8 prefix.
        let partial_ids = &full_ids[..1];
        let result = tokenizer.decode(partial_ids, false);
        assert!(
            result.is_ok(),
            "decode of partial UTF-8 should succeed via lossy fallback"
        );
        let decoded = result.unwrap();
        assert!(
            decoded.contains('\u{FFFD}') || decoded.is_empty(),
            "lossy decode should contain replacement char or be empty, got: {decoded:?}"
        );
    }

    #[test]
    fn test_decode_valid_utf8_does_not_use_fallback() {
        // Ensure that valid UTF-8 round-trips through the happy path unchanged.
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        let text = "Hello, 世界!";
        let encoding = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_recognizes_special_tokens_in_input() {
        // encode_with_special_tokens must recognize special token strings
        // so that chat-template-rendered text (containing e.g. <|endoftext|>)
        // produces single token IDs, not BPE sub-tokens.
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        // <|endoftext|> is token 100257 in cl100k_base
        // Note: add_special_tokens is intentionally ignored for tiktoken
        // (see Encoder impl comment), so both true and false produce the same result.
        let encoding = tokenizer.encode("hello<|endoftext|>world", false).unwrap();
        let ids = encoding.token_ids();
        assert!(
            ids.contains(&100257),
            "Special token <|endoftext|> should be recognized as single token, got: {ids:?}"
        );
    }
}
