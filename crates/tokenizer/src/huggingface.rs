use std::{collections::HashMap, path::Path, sync::Arc};

use anyhow::{Error, Result};
use serde::Deserialize;
use tokenizers::{
    decoders::DecoderWrapper,
    models::{bpe::BPE, ModelWrapper},
    normalizers::unicode::NFC,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    processors::template::TemplateProcessing,
    tokenizer::{SplitDelimiterBehavior, Tokenizer as HfTokenizer},
    AddedToken,
};
use tracing::debug;

use crate::{
    byte_fallback::PieceTable,
    byte_level::{ByteLevelIncremental, ByteLevelTable, UnknownId},
    chat_template::{
        load_chat_template_from_file, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateState, ThinkingKeyName, ThinkingToggle,
    },
    encode_path::{self, EncodePath},
    encoders::{deepseek_v32, deepseek_v4, deepseek_v41, RendererProfile},
    native::NativeEncoder,
    traits::{
        DecodeStep, Decoder, Encoder, Encoding, IncrementalDecoder, SpecialTokens, TokenIdType,
        Tokenizer as TokenizerTrait,
    },
};

#[derive(Debug, Clone, Copy)]
enum Renderer {
    Jinja,
    DeepseekV32,
    DeepseekV4(deepseek_v4::EffortEncoding),
    DeepseekV41,
}

impl Renderer {
    /// The answers of a renderer with its own encoder (see
    /// [`RendererProfile`]); the Jinja renderer answers from the template.
    fn profile(self) -> Option<RendererProfile> {
        match self {
            Self::Jinja => None,
            Self::DeepseekV32 => Some(deepseek_v32::PROFILE),
            Self::DeepseekV4(encoding) => Some(encoding.profile()),
            Self::DeepseekV41 => Some(deepseek_v41::PROFILE),
        }
    }
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
    /// What that renderer answers around a render when it has an encoder of
    /// its own ([`Renderer::profile`]), built once with it.
    profile: Option<RendererProfile>,
    /// Bytes per id, when the decoder is a plain `ByteLevel` (see `byte_level`).
    byte_level: Option<Arc<ByteLevelTable>>,
    /// Text per id, when the decoder is the sentencepiece-style chain with
    /// byte fallback (see `byte_fallback`): the streaming step's window
    /// decodes read it instead of running the `tokenizers` decode.
    pieces: Option<Arc<PieceTable>>,
    /// The direct encode path for byte-level BPE tokenizers of the common
    /// shape (see `native`), when [`DirectEncoders::choose`] took it;
    /// `None` keeps every encode in `tokenizers`.
    native: Option<NativeEncoder>,
    /// The opt-in v1 pipeline, when the choice took it (then `native` is
    /// `None`).
    #[cfg(feature = "tokenizers-v1")]
    v1: Option<crate::v1::V1Encoder>,
    has_byte_fallback: bool,
    /// The byte each `<0xHH>` token stands for, indexed by id (see
    /// [`byte_fallback_values`]); empty unless the decoder has ByteFallback.
    byte_fallback_values: Vec<Option<u8>>,
}

fn has_byte_fallback(decoder: &DecoderWrapper) -> bool {
    match decoder {
        DecoderWrapper::ByteFallback(_) => true,
        DecoderWrapper::Sequence(sequence) => sequence.get_decoders().iter().any(has_byte_fallback),
        _ => false,
    }
}

/// The byte each `<0xHH>` token of a ByteFallback vocabulary stands for,
/// indexed by id: the streaming step reads it per token, so it is a table
/// rather than a vocabulary lookup and a parse. Empty for other vocabularies.
fn byte_fallback_values(
    reverse_vocab: &HashMap<TokenIdType, String>,
    has_byte_fallback: bool,
) -> Vec<Option<u8>> {
    if !has_byte_fallback {
        return Vec::new();
    }
    let len = reverse_vocab
        .keys()
        .filter_map(|&id| usize::try_from(id).ok())
        .map(|id| id + 1)
        .max()
        .unwrap_or(0);
    let mut values = vec![None; len];
    for (&id, token) in reverse_vocab {
        if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
            if let (Ok(index), Ok(byte)) =
                (usize::try_from(id), u8::from_str_radix(&token[3..5], 16))
            {
                values[index] = Some(byte);
            }
        }
    }
    values
}

pub(crate) const QWEN2_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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

/// The encoders a tokenizer may serve an encode with before `tokenizers`
/// itself, chosen once per load by [`DirectEncoders::choose`].
struct DirectEncoders {
    #[cfg(feature = "tokenizers-v1")]
    v1: Option<crate::v1::V1Encoder>,
    native: Option<NativeEncoder>,
}

impl DirectEncoders {
    /// The first path, in order of preference, that takes `tokenizer`: the
    /// opt-in v1 pipeline (behind its feature), the direct byte-level
    /// encoder, else none and `tokenizers` encodes. The choice is logged
    /// once, with the reasons the paths before it declined, and counted per
    /// path ([`encode_path::encode_path_loads`]).
    fn choose(tokenizer: &HfTokenizer) -> Self {
        #[cfg(feature = "tokenizers-v1")]
        let v1_declined = match crate::v1::V1Encoder::from_tokenizer(
            tokenizer,
            HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD,
        ) {
            Ok(v1) => {
                encode_path::note_load(EncodePath::V1);
                tracing::info!(path = %EncodePath::V1, "tokenizer encode path");
                return Self {
                    v1: Some(v1),
                    native: None,
                };
            }
            Err(declined) => declined,
        };
        let (native, path, native_declined) = match NativeEncoder::from_tokenizer(tokenizer) {
            Ok(native) => (Some(native), EncodePath::Native, None),
            Err(declined) => (None, EncodePath::Tokenizers, Some(declined.to_string())),
        };
        encode_path::note_load(path);
        // The opt-in pipeline is for BPE tokenizers: one of those it declines
        // is worth a warning (the opt-in did not take where it could have);
        // any other model kind was never its business.
        #[cfg(feature = "tokenizers-v1")]
        if matches!(tokenizer.get_model(), ModelWrapper::BPE(_)) {
            tracing::warn!(
                path = %path,
                v1_declined = %v1_declined,
                native_declined = native_declined.as_deref(),
                "the opt-in tokenizers v1 pipeline declined this BPE tokenizer; its encodes take the next path"
            );
        } else {
            tracing::info!(
                path = %path,
                v1_declined = %v1_declined,
                native_declined = native_declined.as_deref(),
                "tokenizer encode path (the opt-in v1 pipeline takes BPE models only)"
            );
        }
        #[cfg(not(feature = "tokenizers-v1"))]
        tracing::info!(
            path = %path,
            native_declined = native_declined.as_deref(),
            "tokenizer encode path"
        );
        Self {
            #[cfg(feature = "tokenizers-v1")]
            v1: None,
            native,
        }
    }

    /// Words the `tokenizers` model may cache per thread: none behind the
    /// direct encode path (`native`), whose own piece cache, owned by the
    /// encoder and freed with it, answers in front of the model for every
    /// piece of up to 256 bytes (the short ones by count, the long ones, a
    /// clause in a script without word spaces say, within a byte budget; a
    /// longer piece is rare and merged on every encode), else
    /// [`HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD`]. The model's
    /// cache outlives the model: it is a thread-local the crate never
    /// clears, so every thread that encoded with a tokenizer the gateway
    /// dropped (an unload, a re-registration, a reload) kept the words it
    /// had cached for it, half a MB per thread per dropped tokenizer. The
    /// piece cache is owned by the encoder and freed with it.
    fn bpe_words_cached(&self) -> usize {
        if self.native.is_some() {
            0
        } else {
            HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD
        }
    }
}

impl HuggingFaceTokenizer {
    /// The path this tokenizer's encodes take, chosen at load (see
    /// [`EncodePath`]).
    pub fn encode_path(&self) -> EncodePath {
        #[cfg(feature = "tokenizers-v1")]
        if self.v1.is_some() {
            return EncodePath::V1;
        }
        if self.native.is_some() {
            EncodePath::Native
        } else {
            EncodePath::Tokenizers
        }
    }

    /// Whether this tokenizer uses the opt-in tokenizers v1 encode pipeline.
    /// Unsupported configurations retain the existing encode path.
    pub fn uses_v1_encoder(&self) -> bool {
        self.encode_path() == EncodePath::V1
    }

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

        // Load merged EOS token IDs from config.json + generation_config.json,
        // plus the tokenizer's own eos_token (structured-output grammars end on it).
        let eos_token_ids = crate::eos::with_tokenizer_eos(
            tokenizer_path
                .parent()
                .map(crate::eos::load_eos_token_ids)
                .unwrap_or_default(),
            special_tokens
                .eos_token
                .as_deref()
                .and_then(|token| vocab.get(token).copied()),
        );

        // Detect a custom Python-encoder model from config.json::architectures.
        let renderer = tokenizer_path
            .parent()
            .map(detect_renderer_from_config)
            .unwrap_or(Renderer::Jinja);

        let encoders = DirectEncoders::choose(&tokenizer);
        Self::bound_bpe_cache(&mut tokenizer, encoders.bpe_words_cached());
        let has_byte_fallback = tokenizer.get_decoder().is_some_and(has_byte_fallback);
        let byte_fallback_values = byte_fallback_values(&reverse_vocab, has_byte_fallback);
        Ok(HuggingFaceTokenizer {
            byte_level: ByteLevelTable::build(&tokenizer),
            pieces: PieceTable::build(&tokenizer),
            native: encoders.native,
            #[cfg(feature = "tokenizers-v1")]
            v1: encoders.v1,
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: ChatTemplateState::new(chat_template_str)?,
            eos_token_ids,
            profile: renderer.profile(),
            renderer,
            has_byte_fallback,
            byte_fallback_values,
        })
    }

    /// The text of a window of the stream (a handful of ids): a walk of the
    /// piece table when the vocabulary has one, else the `tokenizers` decode.
    fn window_decode(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        match &self.pieces {
            Some(table) => Ok(table.render(ids, skip_special_tokens)),
            None => self.decode(ids, skip_special_tokens),
        }
    }

    /// The byte a ByteFallback byte token (`<0xHH>`) stands for.
    fn byte_fallback_value(&self, token_id: TokenIdType) -> Option<u8> {
        usize::try_from(token_id)
            .ok()
            .and_then(|index| self.byte_fallback_values.get(index).copied().flatten())
    }

    /// Whether the window is a lost byte run: every id a byte token rendered
    /// as its own replacement character (a valid run renders fewer characters
    /// than it has ids, a literal U+FFFD is three ids for one), nothing
    /// pending. The cheap conditions come first: on a valid run this is one
    /// `ends_with`.
    fn tainted_run(&self, ids: &[TokenIdType], prefix: &str, prefix_index: usize) -> bool {
        prefix.ends_with('\u{FFFD}')
            && prefix_index == ids.len()
            && prefix_index > 0
            && prefix.len() == prefix_index * '\u{FFFD}'.len_utf8()
            && prefix.chars().all(|c| c == '\u{FFFD}')
            && self.byte_fallback_value(ids[prefix_index - 1]).is_some()
    }

    fn byte_fallback_piece(&self, token_id: TokenIdType, skip_special_tokens: bool) -> bool {
        self.reverse_vocab.get(&token_id).is_some_and(|token| {
            let byte_token = token.len() == 6
                && token.starts_with("<0x")
                && token.ends_with('>')
                && token.as_bytes()[3..5].iter().all(u8::is_ascii_hexdigit);
            byte_token
                || (skip_special_tokens
                    && self
                        .tokenizer
                        .get_added_vocabulary()
                        .is_special_token(token))
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

    /// Words the `tokenizers` BPE model keeps in its per-thread cache when
    /// every encode runs in `tokenizers`. The crate's default is 10,000 for
    /// every thread that ever encodes, and the gateway encodes on each of its
    /// runtime and blocking threads, so the caches of one model grew with the
    /// thread count (a few MB per thread) instead of settling. The cache
    /// fills first-come and never evicts, so a thread keeps the first
    /// thousand distinct words it sees, which in practice catches the
    /// frequent ones.
    const BPE_WORDS_CACHED_PER_THREAD: usize = 1024;

    /// Sizes the per-thread word cache of a BPE model to `words`
    /// ([`DirectEncoders::bpe_words_cached`]); other model kinds have none.
    fn bound_bpe_cache(tokenizer: &mut HfTokenizer, words: usize) {
        let bounded = match tokenizer.get_model() {
            ModelWrapper::BPE(bpe) => {
                let mut bpe = bpe.clone();
                bpe.resize_cache(words);
                Some(bpe)
            }
            _ => None,
        };
        if let Some(bpe) = bounded {
            tokenizer.with_model(ModelWrapper::BPE(bpe));
        }
    }

    /// Create from an existing HuggingFace tokenizer
    pub fn from_tokenizer(mut tokenizer: HfTokenizer) -> Self {
        let encoders = DirectEncoders::choose(&tokenizer);
        Self::bound_bpe_cache(&mut tokenizer, encoders.bpe_words_cached());
        let special_tokens = Self::extract_special_tokens(&tokenizer, &ConfigTokens::default());
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        let has_byte_fallback = tokenizer.get_decoder().is_some_and(has_byte_fallback);
        let byte_fallback_values = byte_fallback_values(&reverse_vocab, has_byte_fallback);
        HuggingFaceTokenizer {
            byte_level: ByteLevelTable::build(&tokenizer),
            pieces: PieceTable::build(&tokenizer),
            native: encoders.native,
            #[cfg(feature = "tokenizers-v1")]
            v1: encoders.v1,
            tokenizer,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: ChatTemplateState::empty(),
            eos_token_ids: Vec::new(), // No directory path in from_tokenizer
            renderer: Renderer::Jinja,
            profile: None,
            has_byte_fallback,
            byte_fallback_values,
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
        #[cfg(feature = "tokenizers-v1")]
        if let Some(v1) = &self.v1 {
            return v1.encode(input, add_special_tokens);
        }
        // The direct path accepts no post-processor that adds tokens, so a
        // request for special tokens gets the same ids and is served too. An
        // input it gives up is counted and encoded by `tokenizers`.
        if let Some(native) = &self.native {
            match native.encode(self.tokenizer.get_model(), input) {
                Ok(ids) => return Ok(Encoding::Plain(ids)),
                Err(fallback) => encode_path::note_fallback(fallback),
            }
        }
        self.tokenizer
            .encode(input, add_special_tokens)
            .map_err(|e| Error::msg(format!("Encoding failed: {e}")))
            .map(|encoding| Encoding::Hf(Box::new(encoding)))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        #[cfg(feature = "tokenizers-v1")]
        if let Some(v1) = &self.v1 {
            return v1.encode_batch(inputs, add_special_tokens);
        }
        // A batch takes the path a single input takes: the direct encoder when
        // the vocabulary has one (an input it declines goes through the
        // `tokenizers` path on its own), so a batch never runs on the model
        // whose word cache the direct encoder replaced. The loop is serial on
        // the calling thread (the `tokenizers` batch fanned out with rayon):
        // the gateway reaches a batch through `CachedTokenizer::encode_batch`,
        // which fans out over `encode` itself.
        if self.native.is_some() {
            return inputs
                .iter()
                .map(|input| self.encode(input, add_special_tokens))
                .collect();
        }
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

impl HuggingFaceTokenizer {
    /// One step of the HF `step_decode_stream` algorithm over this tokenizer's
    /// `TokenizerImpl`: `ids` holds the tokens of the emitted `prefix` (the
    /// first `prefix_index`) and the tokens still pending; the step emits the
    /// decode's growth beyond the prefix once it no longer ends in an
    /// incomplete character. Where the native function fails because the
    /// decode no longer starts with the prefix (an invalid byte collapsed a
    /// ByteFallback run whose characters were already streamed, a `Replace`
    /// step spanned the boundary), this emits the remainder past the rewritten
    /// text instead, keeps streaming, and reports the recovery.
    /// `count` says whether a recovery goes into the process-wide totals:
    /// the sequence paths and `finish_decode` count, the raw `decode_step` /
    /// `decode_step_eager` calls do not (the gateway replays an output with
    /// them for token accounting, which no client sees).
    fn step_decode_lenient(
        &self,
        new_ids: Vec<TokenIdType>,
        skip_special_tokens: bool,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        count: bool,
    ) -> Result<DecodeStep> {
        if prefix.is_empty() && !ids.is_empty() {
            let new_prefix = self.window_decode(ids, skip_special_tokens)?;
            if !new_prefix.ends_with('\u{FFFD}') {
                *prefix = new_prefix;
                *prefix_index = ids.len();
            }
        }
        let token_id = new_ids.last().copied();
        ids.extend(new_ids);
        let string = self.window_decode(ids, skip_special_tokens)?;
        if string.ends_with('\u{FFFD}') {
            return Ok(DecodeStep::default());
        }
        let (new_text, prefix_mismatch) = match string.strip_prefix(prefix.as_str()) {
            Some("") => return Ok(DecodeStep::default()),
            Some(rest) => (rest.to_string(), false),
            None => {
                let run_start =
                    self.streamed_run_start(ids, *prefix_index, skip_special_tokens, prefix)?;
                let (rest, byte_run) = remainder_after_rewrite(&string, prefix, run_start);
                debug!(
                    ?token_id,
                    emitted = prefix.len(),
                    decoded = string.len(),
                    byte_run,
                    "decode no longer starts with the emitted text; streaming the remainder"
                );
                if count {
                    crate::decode_activity::note_prefix_mismatch(byte_run);
                }
                (rest, true)
            }
        };
        let new_prefix_index = ids.len() - *prefix_index;
        ids.drain(..*prefix_index);
        *prefix = self.window_decode(ids, skip_special_tokens)?;
        *prefix_index = new_prefix_index;
        Ok(DecodeStep {
            text: Some(new_text),
            prefix_mismatch,
        })
    }
}

/// The part of `decoded` that follows what `emitted` already covers, when the
/// decoder chain rewrote `emitted`. The common character prefix is skipped, up
/// to `run_start`, where the text streamed from the open ByteFallback run
/// begins in `emitted`; beyond it, a run collapsed by an invalid byte decodes
/// to one replacement character per byte, so one such character is skipped
/// for every byte of the emitted characters (the second value says that
/// applied). The run's own characters are never matched by equality: a
/// literal U+FFFD streamed from the run stands for its three bytes, and the
/// collapsed run holds three replacement characters for it, not one. Nothing
/// already emitted is retracted: a reader sees the emitted text, then the
/// remainder as the batch decode would end.
///
/// Cold: it runs only when the decode no longer starts with the emitted text
/// (an invalid byte collapsed a run), never on the step's regular path, and
/// it is kept out of that path's text.
#[cold]
#[inline(never)]
fn remainder_after_rewrite(decoded: &str, emitted: &str, run_start: usize) -> (String, bool) {
    let common = decoded
        .char_indices()
        .zip(emitted.chars())
        .find(|((_, a), b)| a != b)
        .map_or_else(|| decoded.len().min(emitted.len()), |((index, _), _)| index)
        .min(run_start);
    let rewritten_bytes = emitted.len() - common;
    let rest = &decoded[common..];
    let replacement_run = rest.chars().take_while(|&c| c == '\u{FFFD}').count();
    if rewritten_bytes > 0 && replacement_run >= rewritten_bytes {
        let skip = '\u{FFFD}'.len_utf8() * rewritten_bytes;
        (rest[skip..].to_string(), true)
    } else {
        (rest.to_string(), false)
    }
}

impl HuggingFaceTokenizer {
    /// Where the text streamed from the open ByteFallback run begins in
    /// `prefix`, the emitted text of `ids[..prefix_index]`: the length of the
    /// text the ids before the run's first byte piece decode to; `prefix.len()`
    /// when the emitted ids end in no byte piece, or the vocabulary has none.
    /// Cold, like [`remainder_after_rewrite`], which both of its callers (the
    /// step and the end-of-input path) call right after it, on the rewrite path only.
    #[cold]
    #[inline(never)]
    fn streamed_run_start(
        &self,
        ids: &[TokenIdType],
        prefix_index: usize,
        skip_special_tokens: bool,
        prefix: &str,
    ) -> Result<usize> {
        if !self.has_byte_fallback {
            return Ok(prefix.len());
        }
        let emitted = &ids[..prefix_index.min(ids.len())];
        let stable_end = emitted
            .iter()
            .rposition(|&id| !self.byte_fallback_piece(id, skip_special_tokens))
            .map_or(0, |index| index + 1);
        if stable_end == emitted.len() {
            return Ok(prefix.len());
        }
        // The start is used as a byte index into the emitted text and into the
        // decode it is aligned with: it has to be a boundary of both, which
        // holds when the stable text is a prefix of the emitted text. A decoder
        // step that spans the boundary (a `Replace` after the fuse) breaks
        // that, and the alignment then covers the whole emitted text, as it
        // did before the run's start was known.
        let stable = self.window_decode(&emitted[..stable_end], skip_special_tokens)?;
        Ok(if prefix.starts_with(stable.as_str()) {
            stable.len()
        } else {
            prefix.len()
        })
    }

    /// The client-visible step (see [`Decoder::decode_step`]): a seeded
    /// sequence's prompt is decode context, never emitted; a ByteFallback run
    /// streams each complete character as it closes; `eager` is the
    /// engine-side observer's path. `count` says whether a recovery goes into
    /// the process-wide totals (the reporting step counts, the raw step does
    /// not).
    #[expect(clippy::too_many_arguments)]
    fn client_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
        eager: bool,
        count: bool,
    ) -> Result<DecodeStep> {
        // Pending byte tokens from which a ByteFallback run can be lost: a
        // valid character has at most four bytes, a complete one streams at once.
        const LOST_RUN_BYTES: usize = 4;
        // Ids a lost run's window keeps: at most four and the latest byte.
        const TAINTED_WINDOW: usize = LOST_RUN_BYTES + 1;
        // A seeded sequence (every id so far is seed, nothing pending): its prompt
        // text is context, never emitted. With pending ids behind the prefix
        // (a skipped special token followed by an open byte run) the step
        // algorithm's own prefix bookkeeping applies instead.
        if self.has_byte_fallback
            && !eager
            && prefix.is_empty()
            && *prefix_index > 0
            && *prefix_index == ids.len()
        {
            let prefill = self.window_decode(&ids[..*prefix_index], skip_special_tokens)?;
            let stable_seed_end = ids[..*prefix_index]
                .iter()
                .rposition(|&id| !self.byte_fallback_piece(id, skip_special_tokens))
                .map_or(0, |index| index + 1);
            if prefill.ends_with('\u{FFFD}') {
                // Keep an unfinished seed byte run as decode context, but
                // suppress the stable prompt text before it.
                *prefix_index = stable_seed_end;
                *prefix = self.window_decode(&ids[..*prefix_index], skip_special_tokens)?;
            } else if stable_seed_end < *prefix_index {
                // A completed seed run is prompt text. Start new bytes at
                // its boundary so later invalid bytes cannot rewrite it.
                ids.truncate(stable_seed_end);
                *prefix_index = stable_seed_end;
                *prefix = self.window_decode(ids, skip_special_tokens)?;
            } else {
                *prefix = prefill;
            }
        }
        if self.has_byte_fallback
            && !eager
            && self.tainted_run(ids, prefix, *prefix_index)
            && self.byte_fallback_value(token_id).is_some()
        {
            // A run that cannot become valid any more renders as one
            // replacement character per byte whatever follows: stream that
            // character now, without a decode. The window keeps the bytes that
            // made the run lost (its first ids) and the latest byte: its decode
            // stays one replacement character per id whatever follows, and
            // the window stays a handful of ids.
            if ids.len() >= TAINTED_WINDOW {
                ids.pop();
                prefix.pop();
            }
            ids.push(token_id);
            prefix.push('\u{FFFD}');
            *prefix_index = ids.len();
            return Ok(DecodeStep {
                text: Some('\u{FFFD}'.to_string()),
                prefix_mismatch: false,
            });
        }
        if let Some(table) = &self.pieces {
            // Nothing pending and a plain piece: the window's text grows by
            // exactly this piece's text (the leading-space strip touched the
            // window's first piece, which this one is not), so it streams at
            // once and becomes the window on its own, where the strip applies
            // to it: the prefix the full step would record.
            if *prefix_index == ids.len() && !prefix.is_empty() && !prefix.ends_with('\u{FFFD}') {
                if let Some(text) = table.plain_text(token_id, skip_special_tokens) {
                    let text = text.to_owned();
                    ids.clear();
                    ids.push(token_id);
                    *prefix = table.render(ids, skip_special_tokens);
                    *prefix_index = 1;
                    return Ok(DecodeStep {
                        text: Some(text),
                        prefix_mismatch: false,
                    });
                }
            }
        }
        let step = self.step_decode_lenient(
            vec![token_id],
            skip_special_tokens,
            ids,
            prefix,
            prefix_index,
            count,
        )?;
        if step.text.is_none() && self.has_byte_fallback && !eager {
            // Held. A valid character has at most four bytes and a complete
            // one streams at once, so four pending byte tokens are a run that
            // cannot become valid any more, unless they hold a literal U+FFFD
            // (`EF BF BD`: a valid character whose decode ends in the
            // replacement character, so the step holds it): the pending bytes
            // must also fail UTF-8 at a definite position. A valid run reaches
            // four pending bytes only through such a literal, so on a valid
            // run this is one comparison per held step. A lost run's decoder
            // renders every byte of it as a replacement character: stream
            // those for the pending bytes now (the characters already sent
            // stay as sent).
            let pending = ids.len() - *prefix_index;
            if pending >= LOST_RUN_BYTES {
                let bytes: Option<Vec<u8>> = ids[*prefix_index..]
                    .iter()
                    .map(|&id| self.byte_fallback_value(id))
                    .collect();
                let lost = bytes.as_deref().and_then(|bytes| {
                    let error = std::str::from_utf8(bytes).err()?;
                    Some((error.valid_up_to(), error.error_len()?))
                });
                if let Some((valid_up_to, error_len)) = lost {
                    // The batch decode rewrites characters already sent only
                    // when they belong to this run (the emitted text ends in a
                    // byte token): that rewrite is the recovery the totals
                    // count, and only the reporting step counts it.
                    let rewritten = *prefix_index > 0
                        && self.byte_fallback_value(ids[*prefix_index - 1]).is_some();
                    if rewritten {
                        debug!(
                            token_id,
                            emitted = prefix.len(),
                            pending,
                            "an invalid byte collapsed a byte run already streamed; streaming its replacement characters"
                        );
                    }
                    let replacements: String = "\u{FFFD}".repeat(pending);
                    // The window keeps the invalid sequence and the byte that
                    // proved it so (at most four ids): their decode is one
                    // replacement character per id whatever bytes follow.
                    ids.drain(..*prefix_index + valid_up_to);
                    ids.truncate(error_len + 1);
                    *prefix = "\u{FFFD}".repeat(ids.len());
                    *prefix_index = ids.len();
                    if rewritten && count {
                        crate::decode_activity::note_prefix_mismatch(true);
                    }
                    return Ok(DecodeStep {
                        text: Some(replacements),
                        prefix_mismatch: rewritten,
                    });
                }
            }
        }
        Ok(step)
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn uses_byte_fallback(&self) -> bool {
        self.has_byte_fallback
    }

    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        #[cfg(feature = "tokenizers-v1")]
        if let Some(table) = &self.byte_level {
            return Ok(table.decode(token_ids, skip_special_tokens));
        }
        // A sentencepiece-style vocabulary with byte fallback decodes through
        // its piece table, as its streaming windows do: one pass over the ids
        // in place of the `tokenizers` decoder chain over every piece.
        if let Some(table) = &self.pieces {
            return Ok(table.render(token_ids, skip_special_tokens));
        }
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| Error::msg(format!("Decoding failed: {e}")))
    }

    /// Native incremental decode (the raw step: a recovery here is not counted,
    /// see [`Self::step_decode_lenient`]): the algorithm of the HF `step_decode_stream`
    /// (hold while the decode ends in an incomplete character, emit what grew
    /// beyond the emitted prefix), run through the concrete `TokenizerImpl`
    /// rather than `dyn Decoder` dispatch, and lenient when the decoder chain
    /// rewrites text already emitted (see [`Self::step_decode_lenient`]).
    ///
    /// A ByteFallback vocabulary streams the complete characters of a byte run
    /// as they close, like the serving engines' own detokenizers; only an
    /// incomplete trailing sequence waits. Batch decoding would turn a run
    /// with an invalid byte into one replacement character per byte, including
    /// the characters already streamed: those stay as streamed and the rest of
    /// the run follows the batch text (reported as a `byte_run` recovery by
    /// `decode_step_reporting`; this raw step does not count it).
    fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        self.client_step(
            token_id,
            ids,
            prefix,
            prefix_index,
            skip_special_tokens,
            false,
            false,
        )
        .map(|step| step.text)
    }

    fn decode_step_eager(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        Ok(self
            .step_decode_lenient(
                vec![token_id],
                skip_special_tokens,
                ids,
                prefix,
                prefix_index,
                false,
            )?
            .text)
    }

    fn decode_step_reporting(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
        eager: bool,
    ) -> Result<DecodeStep> {
        self.client_step(
            token_id,
            ids,
            prefix,
            prefix_index,
            skip_special_tokens,
            eager,
            true,
        )
    }

    /// Byte-level vocabularies decode each id to fixed bytes, so one stream
    /// needs only a pending-bytes buffer instead of two decodes per token.
    fn incremental_decoder(
        &self,
        skip_special_tokens: bool,
    ) -> Option<Box<dyn IncrementalDecoder>> {
        let table = self.byte_level.as_ref()?;
        Some(Box::new(ByteLevelIncremental::new(
            Arc::clone(table),
            skip_special_tokens,
            UnknownId::Drop,
        )))
    }

    fn finish_decode(
        &self,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<DecodeStep> {
        if !self.has_byte_fallback {
            return Ok(DecodeStep::default());
        }
        if prefix.is_empty() && ids.len() == *prefix_index {
            // A seeded sequence that generated nothing: the seed is prompt text.
            ids.clear();
            *prefix_index = 0;
            return Ok(DecodeStep::default());
        }
        let decoded = self.window_decode(ids, skip_special_tokens)?;
        let (pending, prefix_mismatch) = match decoded.strip_prefix(prefix.as_str()) {
            Some(pending) => (pending.to_string(), false),
            None => {
                let run_start =
                    self.streamed_run_start(ids, *prefix_index, skip_special_tokens, prefix)?;
                let (pending, byte_run) = remainder_after_rewrite(&decoded, prefix, run_start);
                debug!(
                    emitted = prefix.len(),
                    decoded = decoded.len(),
                    byte_run,
                    "end-of-input decode no longer starts with the emitted text; streaming the remainder"
                );
                crate::decode_activity::note_prefix_mismatch(byte_run);
                (pending, true)
            }
        };
        ids.clear();
        prefix.clear();
        *prefix_index = 0;
        Ok(DecodeStep {
            text: (!pending.is_empty()).then_some(pending),
            prefix_mismatch,
        })
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

    fn added_token_strings(&self) -> Vec<String> {
        self.tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|token| !token.special && !token.normalized)
            .map(|token| token.content.clone())
            .collect()
    }

    fn added_tokens_after_normalization(&self) -> Vec<String> {
        self.tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|token| token.normalized)
            .map(|token| token.content.clone())
            .collect()
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
        self.profile
            .and_then(|profile| profile.content_format)
            .unwrap_or_else(|| self.chat_template.content_format())
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        self.profile.map_or_else(
            || self.chat_template.thinking_toggle(),
            |profile| profile.thinking_toggle,
        )
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        self.profile.map_or_else(
            || self.chat_template.thinking_key_name(),
            |profile| Some(profile.thinking_key_name),
        )
    }

    fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        self.profile.map_or_else(
            || self.chat_template.native_reasoning_effort_values(),
            |profile| profile.native_effort_values,
        )
    }

    fn native_reasoning_effort_off_values(&self) -> &'static [&'static str] {
        self.profile.map_or_else(
            || self.chat_template.native_reasoning_effort_off_values(),
            |profile| profile.native_effort_off_values,
        )
    }

    fn think_in_prefill(&self) -> bool {
        self.profile.map_or_else(
            || self.chat_template.think_in_prefill(),
            |profile| profile.think_in_prefill,
        )
    }

    fn prefill_opens_think_block(&self) -> bool {
        self.profile.map_or_else(
            || self.chat_template.prefill_opens_think_block(),
            |profile| profile.think_in_prefill,
        )
    }

    fn renderer_capabilities(&self) -> crate::traits::RendererCapabilities {
        self.profile
            .map_or_else(Default::default, |profile| profile.capabilities)
    }

    fn native_thinking_mode(
        &self,
        template_kwargs: Option<&HashMap<String, serde_json::Value>>,
        thinking: Option<bool>,
    ) -> Option<bool> {
        self.profile
            .and_then(|profile| profile.thinking_mode)
            .map(|resolve| resolve(template_kwargs, thinking))
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

/// Pick the effort-prompt revision for a V4 checkpoint. `config.json`,
/// `tokenizer_config.json` and `generation_config.json` do not tell the
/// revisions apart (the DSpark checkpoints share the 0731 config with the
/// original effort table), so the only discriminator is the shipped
/// `encoding/encoding_dsv4.py`. Without it (a tokenizer directory streamed
/// from a worker carries no `encoding/`), render the table the engine's own
/// server renders: its port of the encoder carries the refreshed table for
/// every V4 checkpoint.
fn detect_dsv4_effort_encoding(dir: &Path) -> deepseek_v4::EffortEncoding {
    match std::fs::read_to_string(dir.join("encoding").join("encoding_dsv4.py")) {
        Ok(source) => deepseek_v4::EffortEncoding::detect_from_encoder_source(&source),
        Err(_) => deepseek_v4::EffortEncoding::V0731,
    }
}

// ---------------------------------------------------------------------------
// DeepSeek V3.2 / V4 dispatch shims
// ---------------------------------------------------------------------------
/// Derive the V3.2 thinking mode. The native encoder bypasses
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
/// one engine's `tokenizers/deepseek_v32.py` and another engine's V4 serving path.
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
/// V4's explicit thinking toggle: `template_kwargs["thinking"]`, else the engine's
/// `enable_thinking` alias (the gateway reads the two keys in the same order
/// when it arms the reasoning parser, see `renderer_capabilities`).
fn explicit_thinking_v4(params: &ChatTemplateParams) -> Option<bool> {
    let kwargs = params.template_kwargs?;
    kwargs
        .get("thinking")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            kwargs
                .get("enable_thinking")
                .and_then(serde_json::Value::as_bool)
        })
}

/// DeepSeek V4 chat-template shim, rendering what the engine's own server
/// renders (its `tokenizers/deepseek_v4.py`): thinking is on unless the
/// request says otherwise, and a thinking prompt carries the `high` effort
/// unless `reasoning_effort` names another level.
///
/// The thinking mode mirrors the gateway's parser-arming precedence
/// (`resolve_thinking_pref` in `model_gateway/src/routers/grpc/utils/parsers.rs`),
/// so the rendered prompt and the arming decision agree on every request:
/// 1. an explicit `thinking` / `enable_thinking` kwarg decides;
/// 2. else the `reasoning_effort` kwarg: `"none"` switches thinking off (the
///    renderer's off word), any other string keeps it on;
/// 3. else `params.thinking` (the gateway's projection of the typed
///    `thinking.type` toggle, else `Some(false)` for a `none`/`minimal`
///    effective `reasoning_effort`);
/// 4. else on ([`ThinkingToggle::DefaultOn`]).
///
/// `reasoning_effort` maps as the engine maps it: `max` is the top level, `low`,
/// `minimal` and `medium` the bottom one (which renders no prefix), every
/// other string and an absent value `high`. Which text a level renders is
/// the detected revision's table (see `detect_dsv4_effort_encoding`).
///
/// Deliberate divergence from the engine's Python renderer: there `reasoning_effort: "none"`
/// forces chat mode even over an explicit `thinking: true`. Here the explicit
/// toggle wins, because the gateway arms the reasoning parser from the
/// explicit toggle first, and rendering chat mode for that contradictory
/// input would have the armed parser swallow the whole answer as reasoning.
fn apply_deepseek_v4(
    messages: &[serde_json::Value],
    params: &ChatTemplateParams,
    effort_encoding: deepseek_v4::EffortEncoding,
) -> Result<String> {
    let owned = inject_tools_into_messages(messages, params.tools);
    let msgs: &[serde_json::Value] = owned.as_deref().unwrap_or(messages);
    let effort_kwarg = params
        .template_kwargs
        .and_then(|k| k.get("reasoning_effort"))
        .and_then(serde_json::Value::as_str);
    let (effort_mode, requested_effort) = match effort_kwarg {
        None => (None, None),
        Some("none") => (Some(false), None),
        Some("max") => (Some(true), Some(deepseek_v4::ReasoningEffort::Max)),
        Some("low" | "minimal" | "medium") => (Some(true), Some(deepseek_v4::ReasoningEffort::Low)),
        Some(_) => (Some(true), Some(deepseek_v4::ReasoningEffort::High)),
    };
    let thinking_on = explicit_thinking_v4(params)
        .or(effort_mode)
        .or(params.thinking)
        .unwrap_or(true);
    let (thinking_mode, reasoning_effort) = if thinking_on {
        (
            deepseek_v32::ThinkingMode::Thinking,
            requested_effort.or(Some(deepseek_v4::ReasoningEffort::High)),
        )
    } else {
        (deepseek_v32::ThinkingMode::Chat, None)
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
/// appears in the conversation — the engine's V4.1 rule. This differs from V3.2/V4's
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
/// this tokenizer reports through `thinking_key_name()`) or the engine's
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
/// message (the engine's rule) -> resolve `reasoning_effort` -> resolve the
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
/// `reasoning_effort` (SMG's global contract; the engine prefers the top-level
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
/// 3. else `params.thinking` (the gateway's projection of the typed
///    `thinking.type` toggle, else `Some(false)` for a `none`/`minimal`
///    effective `reasoning_effort`);
/// 4. else on ([`ThinkingToggle::DefaultOn`]).
///
/// Deliberate divergence from the engine's Python renderer: there `reasoning_effort: "none"`
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
    // message: the engine's Python renderer never renders V4.1's `## Response Format:` block
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

    /// The added tokens a prefix cache may match are those of the raw pass
    /// (`normalized: false`); the tokens matched after normalization, special
    /// or not, are reported apart, so the cache does not split at any of them.
    #[test]
    fn the_cache_sees_the_raw_pass_added_tokens_and_the_normalized_ones_apart() {
        use tokenizers::{
            models::bpe::{Merges, Vocab, BPE},
            AddedToken,
        };

        use super::HuggingFaceTokenizer;
        use crate::traits::Tokenizer as TokenizerTrait;
        let mut vocab = Vocab::default();
        for (id, piece) in ["<unk>", "a", "b", "<", ">"].iter().enumerate() {
            vocab.insert((*piece).to_string(), id as u32);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Merges::new())
            .unk_token("<unk>".into())
            .build()
            .expect("model");
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer
            .add_special_tokens([
                AddedToken::from("<|s|>", true),
                AddedToken::from("<|n|>", true).normalized(true),
            ])
            .expect("special tokens added");
        tokenizer
            .add_tokens([
                AddedToken::from("</think>", false).normalized(false),
                AddedToken::from("a<", false),
            ])
            .expect("added tokens added");
        let wrapped = HuggingFaceTokenizer::from_tokenizer(tokenizer);
        assert_eq!(wrapped.added_token_strings(), vec!["</think>".to_owned()]);
        let mut after_normalization = wrapped.added_tokens_after_normalization();
        after_normalization.sort();
        assert_eq!(
            after_normalization,
            vec!["<|n|>".to_owned(), "a<".to_owned()]
        );
    }

    /// The one-shot decode through the piece table is byte for byte the
    /// `tokenizers` decode: random id sequences of word pieces, byte runs
    /// (valid, incomplete, lost), specials, a lone marker and a piece ending
    /// in U+FFFD, with and without the trailing strip, under both
    /// special-token settings, long and short.
    #[test]
    fn the_piece_table_decodes_what_the_tokenizers_decode_decodes() {
        use super::HuggingFaceTokenizer;
        use crate::{
            byte_fallback::tests::{random_ids, synthetic as synthetic_for_decode},
            traits::Decoder as _,
        };
        for strip in [false, true] {
            for skip in [false, true] {
                let with = HuggingFaceTokenizer::from_tokenizer(synthetic_for_decode(strip));
                assert!(with.pieces.is_some(), "the synthetic chain has a table");
                let mut without = HuggingFaceTokenizer::from_tokenizer(synthetic_for_decode(strip));
                without.pieces = None;
                let mut state =
                    0x2545_F491_4F6C_DD1Du64 ^ u64::from(strip) ^ (u64::from(skip) << 1);
                for round in 0..2_000usize {
                    let ids = random_ids(&mut state, 1 + round % 40);
                    assert_eq!(
                        with.decode(&ids, skip).unwrap(),
                        without.decode(&ids, skip).unwrap(),
                        "ids {ids:?} strip {strip} skip {skip}"
                    );
                }
                // A long sequence, as a one-shot decode of a whole output is.
                let long: Vec<u32> = (0..400)
                    .flat_map(|r| random_ids(&mut state, 1 + r % 40))
                    .collect();
                assert_eq!(
                    with.decode(&long, skip).unwrap(),
                    without.decode(&long, skip).unwrap(),
                    "{} ids, strip {strip} skip {skip}",
                    long.len()
                );
            }
        }
    }

    /// With the piece table the step algorithm streams byte for byte what it
    /// streams through the `tokenizers` window decode. Random streams of word
    /// pieces, byte runs (valid, incomplete, lost), specials, a lone marker
    /// and a piece ending in U+FFFD go through the raw client step, the eager
    /// step and `finish_decode`, fresh and seeded, with and without the
    /// trailing strip, under both special-token settings; the raw step keeps
    /// the process-wide recovery counters still, which other tests of this
    /// binary read. Streams without a lost run also go through `Sequence`,
    /// the gateway's path.
    #[test]
    fn the_piece_table_streams_what_the_tokenizers_window_decode_streams() {
        use std::sync::Arc;

        use super::HuggingFaceTokenizer;
        use crate::{
            byte_fallback::tests::{random_ids, random_ids_without_lost_runs, synthetic},
            sequence::Sequence,
            traits::{Decoder, Tokenizer as TokenizerTrait},
        };
        type Window = (Vec<u32>, String, usize);
        for strip in [false, true] {
            for skip in [false, true] {
                let with = HuggingFaceTokenizer::from_tokenizer(synthetic(strip));
                assert!(with.pieces.is_some(), "the synthetic chain has a table");
                let mut without = HuggingFaceTokenizer::from_tokenizer(synthetic(strip));
                without.pieces = None;
                let mut state =
                    0x9E37_79B9_7F4A_7C15u64 ^ u64::from(strip) ^ (u64::from(skip) << 1);
                for round in 0..1_200usize {
                    let seed = random_ids(&mut state, round % 5);
                    let ids = random_ids(&mut state, 1 + round % 24);
                    let eager = round % 3 == 2;
                    // The window a seeded sequence starts from: the seed as
                    // context, nothing emitted.
                    let mut a: Window = (seed.clone(), String::new(), seed.len());
                    let mut b: Window = (seed.clone(), String::new(), seed.len());
                    let step = |tok: &HuggingFaceTokenizer, w: &mut Window, id: u32| {
                        if eager {
                            tok.decode_step_eager(id, &mut w.0, &mut w.1, &mut w.2, skip)
                        } else {
                            tok.decode_step(id, &mut w.0, &mut w.1, &mut w.2, skip)
                        }
                    };
                    for &id in &ids {
                        assert_eq!(
                            step(&with, &mut a, id).unwrap(),
                            step(&without, &mut b, id).unwrap(),
                            "seed {seed:?} ids {ids:?} at {id} strip {strip} skip {skip} eager {eager}"
                        );
                    }
                    // The end of input, where nothing is pending that a lost
                    // run would make the counters record: the last id closes
                    // any run (a skipped special does not).
                    if ids.last().is_some_and(|&id| {
                        (3..=7).contains(&id) || (!skip && (1..=2).contains(&id))
                    }) {
                        let fa = with
                            .finish_decode(&mut a.0, &mut a.1, &mut a.2, skip)
                            .unwrap();
                        let fb = without
                            .finish_decode(&mut b.0, &mut b.1, &mut b.2, skip)
                            .unwrap();
                        assert_eq!(
                            (fa.text, fa.prefix_mismatch),
                            (fb.text, fb.prefix_mismatch),
                            "finish of {seed:?} + {ids:?}"
                        );
                    }
                }
                let with: Arc<dyn TokenizerTrait> = Arc::new(with);
                let without: Arc<dyn TokenizerTrait> = Arc::new(without);
                for round in 0..300usize {
                    let ids = random_ids_without_lost_runs(&mut state, 1 + round % 30);
                    let mut a = Sequence::new_with_options(Arc::clone(&with), skip);
                    let mut b = Sequence::new_with_options(Arc::clone(&without), skip);
                    for &id in &ids {
                        assert_eq!(
                            a.append_token(id).unwrap(),
                            b.append_token(id).unwrap(),
                            "{ids:?} at {id} strip {strip} skip {skip}"
                        );
                    }
                    assert_eq!(
                        a.finish().unwrap(),
                        b.finish().unwrap(),
                        "finish of {ids:?}"
                    );
                    assert_eq!(
                        a.prefix_mismatches(),
                        0,
                        "{ids:?}: no lost run, no recovery"
                    );
                }
            }
        }
    }

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
