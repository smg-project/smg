use std::{
    collections::HashMap,
    ops::Range,
    path::Path,
    sync::{Arc, OnceLock},
};

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use tokenizers::{
    decoders::{
        byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence,
        strip::Strip, DecoderWrapper,
    },
    models::{bpe::BPE, ModelWrapper},
    normalizers::{unicode::NFC, NormalizerWrapper, Replace},
    pre_tokenizers::{
        byte_level::ByteLevel,
        metaspace::{Metaspace, PrependScheme},
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    processors::template::TemplateProcessing,
    tokenizer::{SplitDelimiterBehavior, Tokenizer as HfTokenizer},
    AddedToken, AddedVocabulary, Model,
};
use tracing::{debug, info, warn};

use crate::{
    byte_fallback::PieceTable,
    byte_level::{cut_start, push_token_bytes, ByteLevelIncremental, ByteLevelTable, UnknownId},
    chat_template::{
        load_chat_template_from_file, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateState, ConfigChatTemplates, ThinkingKeyName, ThinkingToggle,
    },
    encode_path::{self, EncodePath},
    encoders::{
        deepseek_v32, deepseek_v4, deepseek_v41, ends_with_generation_header, glm53, marks,
        PromptSegment, RendererProfile, TemplateAnswers,
    },
    native::{AddedTokens, NativeEncoder},
    traits::{
        append_step_text, AppendedStep, ChatTemplateOutput, DecodeStep, Decoder, EncodeJob,
        Encoder, Encoding, IncrementalDecoder, PromptEncoding, SpecialTokens, TokenIdType,
        Tokenizer as TokenizerTrait,
    },
    word_cache::{LateBoundModel, LoadingTokenizer, ModelView, TokenizerOf},
};

mod deepseek_v31;
mod glm;

#[derive(Debug, Clone, Copy)]
enum Renderer {
    Jinja,
    DeepseekV32,
    DeepseekV4,
    DeepseekV41(deepseek_v41::EffortTable),
    Glm53(glm53::Template),
}

impl Renderer {
    /// The answers of a renderer with its own encoder (see
    /// [`RendererProfile`]); the Jinja renderer answers from the template.
    fn profile(self) -> Option<RendererProfile> {
        match self {
            // GLM keeps the active template's switchless thinking contract.
            Self::Jinja | Self::Glm53(_) => None,
            Self::DeepseekV32 => Some(deepseek_v32::PROFILE),
            Self::DeepseekV4 => Some(deepseek_v4::PROFILE),
            Self::DeepseekV41(_) => Some(deepseek_v41::PROFILE),
        }
    }
}

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    /// The `tokenizers` pipeline and the direct encode path, shared with the
    /// deferred encode of a prompt rendered by segments.
    core: Arc<EncoderCore>,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    reverse_vocab: HashMap<TokenIdType, String>,
    chat_template: ChatTemplateState,
    /// Raw `response_template` from the sibling tokenizer_config.json.
    response_template: Option<serde_json::Value>,
    /// EOS token IDs from config.json + generation_config.json
    eos_token_ids: Vec<TokenIdType>,
    /// Which renderer applies chat templates for this model.
    renderer: Renderer,
    /// Checkpoint identity and media policy resolved when binding the template.
    glm: Option<glm::Checkpoint>,
    media_handling: crate::traits::MediaHandling,
    /// Checkpoint identity of the V3 architecture family, resolved when the
    /// template is bound (see `deepseek_v31`).
    deepseek_v31: Option<deepseek_v31::Checkpoint>,
    /// Whether the active template gets the request's tools as a system
    /// message block it does not write itself (the V3.1 template, see
    /// `deepseek_v31`); resolved with the template and refreshed when the
    /// template is replaced.
    tools_block: bool,
    /// What that renderer answers around a render when it has an encoder of
    /// its own ([`Renderer::profile`]), built once with it.
    profile: Option<RendererProfile>,
    /// Bytes per id, when the decoder is a plain `ByteLevel` (see `byte_level`).
    byte_level: Option<Arc<ByteLevelTable>>,
    /// Text per id, when the decoder is the sentencepiece-style chain with
    /// byte fallback (see `byte_fallback`): the streaming step's window
    /// decodes read it instead of running the `tokenizers` decode.
    pieces: Option<Arc<PieceTable>>,
    /// The opt-in v1 pipeline, when the choice took it (then the core's
    /// `native` is `None`).
    #[cfg(feature = "tokenizers-v1")]
    v1: Option<crate::v1::V1Encoder>,
    has_byte_fallback: bool,
    /// The byte each `<0xHH>` token stands for, indexed by id (see
    /// [`byte_fallback_values`]); empty unless the decoder has ByteFallback.
    byte_fallback_values: Vec<Option<u8>>,
    /// The decoder is a sequence with a byte-level member, which gets no
    /// per-id table: the generic window then holds the bytes of an
    /// unfinished character, and the stream's end judges a cut from them.
    byte_level_chain: bool,
    /// The checkpoint ships the family's own vocabulary file beside the
    /// conversion (`tekken.json`): the file's own reader, the family's
    /// reference library, writes the control tokens of a request
    /// structurally and BPE-encodes every request string as plain text, so a
    /// control spelling typed inside a message stays text there. The
    /// conversion's added-token matcher reads the same spelling as the
    /// control token wherever it stands, so the render tells the two apart
    /// for such a checkpoint (see `apply_chat_template_with_encoding`).
    structural_controls: bool,
    /// The special tokens between the marks, for the marked template's
    /// context (`Some` with `structural_controls`, read while the template
    /// is marked).
    special_tokens_marked: Option<SpecialTokens>,
}

/// Logs, once per load, which path a tokenizer's full decode takes: the
/// byte-level table (a `ByteLevel` decoder), the piece table (byte fallback),
/// or `tokenizers` itself, with the decoder kind that kept the tables out.
fn log_decode_path(tokenizer: &HfTokenizer, byte_level: bool, pieces: bool) {
    let path = if byte_level {
        "byte-level table"
    } else if pieces {
        "piece table"
    } else {
        "tokenizers"
    };
    if byte_level || pieces {
        tracing::info!(path, "tokenizer decode path");
    } else {
        tracing::info!(
            path,
            declined = %crate::byte_level::decoder_kind(tokenizer.get_decoder()),
            "tokenizer decode path (the tables take a ByteLevel decoder, or the chain \
             [Replace U+2581 -> space] ByteFallback Fuse [Strip one leading space])"
        );
    }
}

fn has_byte_fallback(decoder: &DecoderWrapper) -> bool {
    match decoder {
        DecoderWrapper::ByteFallback(_) => true,
        DecoderWrapper::Sequence(sequence) => sequence.get_decoders().iter().any(has_byte_fallback),
        _ => false,
    }
}

/// Whether a `Sequence` decoder has a byte-level member (a plain `ByteLevel`
/// decoder gets the per-id table instead, see `byte_level`).
fn has_byte_level_member(decoder: &DecoderWrapper) -> bool {
    match decoder {
        DecoderWrapper::Sequence(sequence) => sequence.get_decoders().iter().any(|member| {
            matches!(member, DecoderWrapper::ByteLevel(_)) || has_byte_level_member(member)
        }),
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
    fn choose<M: Model + ModelView + Serialize>(tokenizer: &TokenizerOf<M>) -> Self {
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
        if matches!(tokenizer.get_model().wrapper(), ModelWrapper::BPE(_)) {
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
    /// piece of up to 16 KiB (`LONG_PIECE_MAX_BYTES`: the short ones by
    /// count, the long ones, a paragraph in a script without word spaces
    /// say, within a byte budget; a longer piece is merged on every encode),
    /// else
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

    /// Reads the vocabulary into the direct encoder's seed, through the model
    /// once its word cache is bounded: the words the seed merges then leave
    /// nothing in the model's per-thread caches.
    fn seed_native(&mut self, model: &ModelWrapper) {
        if let Some(native) = &mut self.native {
            native.seed_vocabulary(model);
        }
    }
}

/// What an encode runs on, shared between the wrapper and the deferred
/// encode of a prompt rendered by segments (an [`EncodeJob`] outlives the
/// borrow of the wrapper that prepared it).
struct EncoderCore {
    tokenizer: HfTokenizer,
    /// The direct encode path for byte-level BPE tokenizers of the common
    /// shape (see `native`), when [`DirectEncoders::choose`] took it;
    /// `None` keeps every encode in `tokenizers`.
    native: Option<NativeEncoder>,
    /// Every added token, special or not, as one leftmost-longest matcher
    /// over the raw text, built on first use (only a renderer with an
    /// encoder of its own asks; a vocabulary can carry tens of thousands of
    /// added tokens, which a Jinja-rendered model never pays for); `None`
    /// for a vocabulary without added tokens. It tells whether the flat
    /// encode of a prompt rendered by segments would cut out exactly the
    /// renderer's control tokens.
    added: OnceLock<Option<AddedTokens>>,
    /// `tokenizer` without its added vocabulary, built on first use: the
    /// plain segments the direct path declines are encoded by it, every
    /// spelling in them as text.
    plain: OnceLock<HfTokenizer>,
    /// [`Self::plain`] for a plain segment after a control, under a
    /// `Metaspace` pre-tokenizer that prepends the word marker at the start
    /// of the text only: the same pipeline with the scheme `never`, built on
    /// first use (see [`Self::plain_tokenizer`]).
    plain_after_control: OnceLock<HfTokenizer>,
}

impl EncoderCore {
    fn new(tokenizer: HfTokenizer, native: Option<NativeEncoder>) -> Self {
        Self {
            tokenizer,
            native,
            added: OnceLock::new(),
            plain: OnceLock::new(),
            plain_after_control: OnceLock::new(),
        }
    }

    /// The matcher over the added vocabulary, built on first use.
    fn added(&self) -> Option<&AddedTokens> {
        self.added
            .get_or_init(|| {
                let added: Vec<(String, u32)> = self
                    .tokenizer
                    .get_added_vocabulary()
                    .get_vocab()
                    .iter()
                    .map(|(content, &id)| (content.clone(), id))
                    .collect();
                AddedTokens::build(&added).unwrap_or_else(|declined| {
                    debug!(%declined, "no matcher over the added tokens; prompts rendered by segments are encoded by segments");
                    None
                })
            })
            .as_ref()
    }

    /// Whether the flat encode of `text` cuts out exactly the added tokens at
    /// `controls` (the renderer's control tokens, in order): then it is the
    /// segment encode and the text can be encoded as text, through every
    /// path and cache a flat encode takes. False when another added-token
    /// spelling occurs in the text (one a client typed), or a control is no
    /// added token of this vocabulary.
    fn flat_encode_is_segment_encode(&self, text: &str, controls: &[Range<usize>]) -> bool {
        match self.added() {
            Some(added) => added.ranges(text).eq(controls.iter().cloned()),
            None => controls.is_empty(),
        }
    }

    /// The ids of a prompt rendered by segments: each control segment as
    /// the id of its spelling, the plain text between two controls through
    /// the pre-tokenizer and the model as one text with no added token cut
    /// out of it (the family's reference library encodes the plain regions
    /// of a prompt the same way). A control spelling the vocabulary has no
    /// token for is text, joined with the plain text around it so that its
    /// pieces merge with their neighbours as the flat encode merged them.
    fn encode_segments(&self, segments: &[PromptSegment]) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(segments.iter().map(|s| s.text.len() / 3 + 1).sum());
        let mut plain = String::new();
        for segment in segments {
            let control = segment
                .allow_special
                .then(|| self.tokenizer.token_to_id(&segment.text))
                .flatten();
            match control {
                Some(id) => {
                    if !plain.is_empty() {
                        self.encode_plain(&plain, ids.is_empty(), &mut ids)?;
                        plain.clear();
                    }
                    ids.push(id);
                }
                None => {
                    if segment.allow_special {
                        debug!(token = %segment.text, "a control token the vocabulary lacks is encoded as text");
                    }
                    plain.push_str(&segment.text);
                }
            }
        }
        if !plain.is_empty() {
            self.encode_plain(&plain, ids.is_empty(), &mut ids)?;
        }
        Ok(ids)
    }

    /// Appends the ids of `text` with every spelling in it encoded as text:
    /// the direct path without its added-token cut, else (an input it
    /// declines, or no direct path) `tokenizers` without the added
    /// vocabulary, as the region at the start of the prompt (`at_start`) or
    /// after a control. The opt-in v1 pipeline has no such mode and is
    /// passed by.
    fn encode_plain(&self, text: &str, at_start: bool, ids: &mut Vec<u32>) -> Result<()> {
        if let Some(native) = &self.native {
            match native.encode_plain(Some(self.tokenizer.get_model()), text, at_start) {
                Ok(plain) => {
                    ids.extend(plain);
                    return Ok(());
                }
                Err(fallback) => encode_path::note_fallback(fallback),
            }
        }
        let encoding = self
            .plain_tokenizer(at_start)
            .encode(text, false)
            .map_err(|e| Error::msg(format!("Encoding failed: {e}")))?;
        ids.extend_from_slice(encoding.get_ids());
        Ok(())
    }

    /// The pipeline without its added vocabulary, cloned once on first use:
    /// as loaded for a region at the start of the prompt; for a region after
    /// a control, under a `Metaspace` pre-tokenizer that prepends the word
    /// marker at the start of the text only (the scheme `first`), the same
    /// with the scheme `never`, which is how the flat encode treats a split
    /// that does not start the text.
    fn plain_tokenizer(&self, at_start: bool) -> &HfTokenizer {
        let plain_of = || {
            let mut plain = self.tokenizer.clone();
            plain.with_added_vocabulary(AddedVocabulary::new());
            plain
        };
        let after_control = match self.tokenizer.get_pre_tokenizer() {
            Some(PreTokenizerWrapper::Metaspace(metaspace))
                if !at_start && metaspace.get_prepend_scheme() == PrependScheme::First =>
            {
                Some(Metaspace::new(
                    metaspace.get_replacement(),
                    PrependScheme::Never,
                    metaspace.get_split(),
                ))
            }
            _ => None,
        };
        match after_control {
            None => self.plain.get_or_init(plain_of),
            Some(never) => self.plain_after_control.get_or_init(|| {
                let mut plain = plain_of();
                plain.with_pre_tokenizer(Some(never));
                plain
            }),
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
        if self.core.native.is_some() {
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

    /// The `tokenizers` pipeline behind this tokenizer, as loaded: the
    /// vocabulary and stages of `tokenizer.json`, with the post-processor
    /// the configuration prescribes when `tokenizer_config.json` sets
    /// `add_bos_token` or `add_eos_token`, and with the file's `truncation`
    /// and `padding` blocks cleared (both rules are on [`Self::from_file`]).
    /// A cross-check of the crate's ids against the `tokenizers` crate
    /// compares with this pipeline, which adds the same special tokens on
    /// request, or with `add_special_tokens` off, where it and
    /// `tokenizer.json` read alone agree once that file's blocks are cleared
    /// too.
    pub fn tokenizers_pipeline(&self) -> &HfTokenizer {
        &self.core.tokenizer
    }

    /// Create a tokenizer from a HuggingFace tokenizer JSON file.
    ///
    /// Special tokens on request: an encode with `add_special_tokens` set
    /// adds what the checkpoint's configuration prescribes, not only what
    /// `tokenizer.json` holds. When the sibling `tokenizer_config.json` sets
    /// `add_bos_token` or `add_eos_token` to `true`, the loader replaces the
    /// file's post-processor with the template `{bos} $A {eos}` (each end as
    /// configured, when its token is in the vocabulary), the one the
    /// `transformers` Llama-style fast tokenizer classes install at load
    /// (`update_post_processor`), so the ids are those the serving engine's
    /// tokenizer produces. A checkpoint whose `tokenizer.json` carries a
    /// plain `ByteLevel` post-processor while its configuration says
    /// `add_bos_token: true` therefore gets the configured BOS prepended on
    /// request, where the `tokenizers` crate reading `tokenizer.json` alone
    /// adds nothing; without the flag the ids are the file's in both.
    /// Nothing changes unless one of the two is an explicit `true` (with
    /// `add_bos_token` unset and `add_eos_token: true`, the BOS stays when
    /// the file's post-processor already adds tokens). A cross-check against
    /// the `tokenizers` crate compares with [`Self::tokenizers_pipeline`],
    /// the same post-processor, or with the flag off.
    ///
    /// The file's `truncation` and `padding` blocks do not apply: they are
    /// the defaults the serving engine's tokenizer keeps for a call that asks
    /// to truncate or pad, and a plain encode runs with both off, so a prompt
    /// longer than the block's `max_length` encodes to its whole id list here
    /// as there.
    pub fn from_file(file_path: &str) -> Result<Self> {
        // Try to auto-discover chat template if not explicitly provided
        let path = Path::new(file_path);
        let chat_template_path = path
            .parent()
            .and_then(crate::factory::discover_chat_template_in_dir);
        Self::from_file_with_chat_template(file_path, chat_template_path.as_deref())
    }

    /// Create a tokenizer from a HuggingFace tokenizer JSON file with an
    /// optional chat template; the special-token rule of [`Self::from_file`]
    /// applies.
    pub fn from_file_with_chat_template(
        file_path: &str,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let tokenizer = LoadingTokenizer::from_file(file_path)
            .map_err(|e| Error::msg(format!("Failed to load tokenizer: {e}")))?;
        Self::from_built_tokenizer(tokenizer, Path::new(file_path), chat_template_path)
    }

    /// Create a Qwen2-compatible byte-level BPE tokenizer from a Hugging Face
    /// directory containing `vocab.json`, `merges.txt`, and
    /// `tokenizer_config.json` but no `tokenizer.json`, with an optional
    /// explicit chat template. The special-token rule of [`Self::from_file`]
    /// applies to the configuration here too.
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

    fn build_qwen2_bpe_tokenizer(dir: &Path) -> Result<LoadingTokenizer> {
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
        let mut tokenizer = LoadingTokenizer::new(LateBoundModel::from(ModelWrapper::BPE(bpe)));

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

    /// Settles the added tokens matched after normalization whose normalized
    /// spellings are equal: a letter and its compatibility form under an
    /// NFKC normalizer, for instance. `tokenizers` matches those tokens with
    /// an automaton of their normalized spellings built from a hash map it
    /// iterates in a per-process order, and between equal spellings the
    /// first wins, so the id such a text encodes to changes from one
    /// process to the next. Here the member spelled as the normalized form
    /// itself (the letter) keeps the normalized match, which every text
    /// whose normalized form holds the spelling then encodes to; the lowest
    /// id keeps it when no member spells the form. Each other member (a
    /// compatibility form) is re-added matched on the raw text
    /// (`normalized: false`), where its own spelling alone reaches it. The
    /// raw match comes before normalization, so such a spelling in a text is
    /// cut ahead of any longer normalized token that would have spanned it:
    /// the one reading that changes beyond the choice of id, and it is
    /// confined to the compatibility forms, never to the letter plain text
    /// holds. Decoding is untouched: every id keeps its token. Returns how
    /// many tokens were re-added; a vocabulary without such a group is left
    /// as it is.
    fn settle_normalized_added_token_ties<M: Model>(
        tokenizer: &mut TokenizerOf<M>,
    ) -> tokenizers::Result<usize> {
        if tokenizer.get_normalizer().is_none() {
            return Ok(0);
        }
        let added = tokenizer.get_added_vocabulary();
        let tokens = added.get_added_tokens_decoder();
        // The ids of every normalized spelling, `simple_id_to_token` being
        // the spelling the automaton holds: the normalized form where it
        // differs from the content.
        let mut ids_of_spelling: HashMap<String, Vec<u32>> = HashMap::new();
        for (&id, token) in tokens {
            if token.normalized {
                let spelling = added
                    .simple_id_to_token(id)
                    .unwrap_or_else(|| token.content.clone());
                ids_of_spelling.entry(spelling).or_default().push(id);
            }
        }
        let mut demoted: Vec<AddedToken> = Vec::new();
        for (spelling, mut ids) in ids_of_spelling {
            if ids.len() < 2 {
                continue;
            }
            ids.sort_unstable();
            let kept = ids
                .iter()
                .copied()
                .find(|id| {
                    tokens
                        .get(id)
                        .is_some_and(|token| token.content == spelling)
                })
                .unwrap_or(ids[0]);
            demoted.extend(ids.iter().filter(|&&id| id != kept).filter_map(|id| {
                let mut token = tokens.get(id)?.clone();
                token.normalized = false;
                Some(token)
            }));
        }
        if demoted.is_empty() {
            return Ok(0);
        }
        // A token re-added under a content the vocabulary holds keeps its id;
        // the two automatons are rebuilt once, after the whole batch.
        let settled = demoted.len();
        tokenizer.add_tokens(demoted)?;
        info!(
            settled,
            "added tokens with one normalized spelling: the letter keeps the normalized match, the \
             compatibility forms match their own spelling raw"
        );
        Ok(settled)
    }

    /// Bind checkpoint metadata, the active template, and its cached renderer policy.
    fn from_built_tokenizer(
        mut tokenizer: LoadingTokenizer,
        tokenizer_path: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        // The file's `truncation` and `padding` blocks are the defaults the
        // engine's tokenizer keeps for a call that asks to truncate or pad; a
        // plain encode runs with both off (the rule is on `from_file`).
        // Cleared before the encode path is chosen, so the direct encoders
        // decline nothing for a block that no longer applies. `None` passes
        // the stride check `with_truncation` runs on a block.
        tokenizer.with_truncation(None).map_err(|error| {
            Error::msg(format!(
                "Failed to load tokenizer: the truncation block did not clear: {error}"
            ))
        })?;
        tokenizer.with_padding(None);
        // Build vocab mappings (include special tokens to get added_tokens like <|im_start|>)
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        // Load tokenizer_config.json once for chat template, add_bos/eos, and special tokens
        let config_result = Self::load_chat_template_and_config(&tokenizer_path.to_string_lossy());
        // The class the configuration names decides the pipeline of a
        // Llama-class checkpoint (the rule is on `llama_class_pipeline`).
        Self::llama_class_pipeline(&mut tokenizer, tokenizer_path.parent(), &config_result);
        // Two added tokens matched after normalization can share one
        // normalized spelling, and which of them `tokenizers` matches changes
        // from process to process. Settled before the encode path is chosen
        // (the rule is on `settle_normalized_added_token_ties`).
        Self::settle_normalized_added_token_ties(&mut tokenizer).map_err(|error| {
            Error::msg(format!(
                "Failed to load tokenizer: the added tokens with one normalized spelling did not settle: {error}"
            ))
        })?;

        let mut chat_template_str = config_result.chat_template;
        let mut tool_use_template = config_result.tool_use_chat_template;
        let response_template = config_result.response_template;
        let add_bos_token = config_result.add_bos_token;
        let add_eos_token = config_result.add_eos_token;

        // Special tokens: the config's (a null is none), else a guess among the added special tokens
        let special_tokens = Self::extract_special_tokens(&tokenizer, &config_result.config_tokens);

        if let Some(template_path) = chat_template_path {
            chat_template_str = load_chat_template_from_file(template_path)?;
            // A template file stands alone: transformers' `chat_template.jinja`
            // replaces the configuration's templates, named ones included.
            tool_use_template = None;
        }

        // The processor's legacy template file beside the tokenizer, for the
        // requests without tools (the rule is on `processor_chat_template`).
        let processor_template =
            crate::factory::processor_chat_template(tokenizer_path.parent(), chat_template_path);

        // The configuration's special tokens on request (the rule is on
        // `from_file`): an explicit `add_bos_token` / `add_eos_token` of
        // true replaces the file's post-processor with the template the
        // `transformers` Llama-style fast classes install at load
        // (`update_post_processor`); nothing changes unless one is true.
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
            .map(|dir| detect_renderer_from_config(dir, chat_template_str.is_some()))
            .unwrap_or(Renderer::Jinja);

        let mut encoders = DirectEncoders::choose(&tokenizer);
        // The bound rides on the file's model into the built tokenizer: no
        // copy of the model (see `word_cache`).
        tokenizer.get_model().bound(encoders.bpe_words_cached());
        let tokenizer = HfTokenizer::from(tokenizer);
        encoders.seed_native(tokenizer.get_model());
        let has_byte_fallback = tokenizer.get_decoder().is_some_and(has_byte_fallback);
        let byte_fallback_values = byte_fallback_values(&reverse_vocab, has_byte_fallback);
        let glm = tokenizer_path.parent().and_then(glm::Checkpoint::load);
        let renderer = glm
            .as_ref()
            .and_then(|checkpoint| checkpoint.renderer(chat_template_str.as_deref()))
            .map(Renderer::Glm53)
            .unwrap_or(renderer);
        let mut chat_template = ChatTemplateState::from_checkpoint(
            ConfigChatTemplates {
                default: chat_template_str,
                tool_use: tool_use_template,
            },
            processor_template,
        )?;
        let media_handling = glm.as_ref().map_or_else(Default::default, |checkpoint| {
            checkpoint.media_handling(&chat_template)
        });
        let deepseek_v31 = tokenizer_path
            .parent()
            .and_then(deepseek_v31::Checkpoint::load);
        let tools_block =
            deepseek_v31.is_some() && deepseek_v31::Checkpoint::renders_tools_block(&chat_template);
        let byte_level = ByteLevelTable::build(&tokenizer);
        let byte_level_chain =
            byte_level.is_none() && tokenizer.get_decoder().is_some_and(has_byte_level_member);
        let pieces = PieceTable::build(&tokenizer);
        log_decode_path(&tokenizer, byte_level.is_some(), pieces.is_some());
        let core = Arc::new(EncoderCore::new(tokenizer, encoders.native));
        let structural_controls = tokenizer_path
            .parent()
            .is_some_and(|dir| dir.join("tekken.json").is_file());
        let special_tokens_marked = if structural_controls {
            if let Some(controls) = core.added() {
                chat_template.mark_controls(controls)?;
            }
            Some(marks::marked_special_tokens(&special_tokens))
        } else {
            None
        };
        Ok(HuggingFaceTokenizer {
            byte_level,
            pieces,
            #[cfg(feature = "tokenizers-v1")]
            v1: encoders.v1,
            core,
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template,
            glm,
            media_handling,
            deepseek_v31,
            tools_block,
            response_template,
            eos_token_ids,
            profile: renderer.profile(),
            renderer,
            has_byte_fallback,
            byte_fallback_values,
            byte_level_chain,
            structural_controls,
            special_tokens_marked,
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

    /// How many ids at the end of `held` (the window's ids past the emitted
    /// text) are a character the stream's end cut short: the byte tokens of
    /// a trailing sequence that is the valid start of a character and nothing
    /// more (`cut_start`), with a skipped special token among or after them,
    /// which renders nothing. Zero when the held run ends on a complete
    /// character or on an invalid sequence, which the full decode's
    /// replacement characters stand for.
    fn cut_tail(&self, held: &[TokenIdType], skip_special_tokens: bool) -> usize {
        let run_start = held
            .iter()
            .rposition(|&id| !self.byte_fallback_piece(id, skip_special_tokens))
            .map_or(0, |index| index + 1);
        let run = &held[run_start..];
        let bytes: Vec<u8> = run
            .iter()
            .filter_map(|&id| self.byte_fallback_value(id))
            .collect();
        let mut cut_bytes = bytes.len() - cut_start(&bytes);
        let mut cut = 0;
        for &id in run.iter().rev() {
            if cut_bytes == 0 {
                break;
            }
            if self.byte_fallback_value(id).is_some() {
                cut_bytes -= 1;
            }
            cut += 1;
        }
        cut
    }

    /// Whether the held ids of a window decoded by a sequence with a
    /// byte-level member end in a character the stream's end cut short: their
    /// bytes (the byte-level characters mapped back, a skipped special token
    /// left out) end in the start of a character and nothing more.
    fn byte_level_window_ends_cut(&self, held: &[TokenIdType], skip_special_tokens: bool) -> bool {
        let added = self.core.tokenizer.get_added_vocabulary();
        let mut bytes = Vec::new();
        for &id in held {
            if let Some(token) = self.reverse_vocab.get(&id) {
                if skip_special_tokens && added.is_special_token(token) {
                    continue;
                }
                push_token_bytes(token, &mut bytes);
            }
        }
        cut_start(&bytes) < bytes.len()
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
                        .core
                        .tokenizer
                        .get_added_vocabulary()
                        .is_special_token(token))
        })
    }

    /// Encodes a Llama-class checkpoint as its reference class does.
    ///
    /// `transformers` builds the tokenizer of a checkpoint whose
    /// `tokenizer_config.json` names `LlamaTokenizer` (or the `Fast`
    /// spelling) from the file's vocabulary and merges with a pipeline of
    /// its own: no normalizer, and a `Metaspace` pre-tokenizer whose prepend
    /// scheme the configuration selects (`add_prefix_space: false`: the word
    /// marker is prepended nowhere; `legacy: true`: before every split;
    /// otherwise at the start of the text only). The serving engines load
    /// their tokenizer through that class, so those are the ids a prompt
    /// gets. A `tokenizer.json` converted the legacy way keeps instead a
    /// `Prepend` normalizer, which puts the word marker before every split
    /// (after every special token too) and into the pattern of every added
    /// token matched after normalization, so that a marker such as
    /// `</think>` is found only behind a space, and takes the space with it.
    /// For a file of that legacy shape the loader installs the class's
    /// pipeline (no normalizer, the `Metaspace` pre-tokenizer, and the
    /// class's decoder, which strips the one leading space of the decoded
    /// text only where the marker is prepended) over the file's vocabulary,
    /// added tokens and post-processor; any other file stays as it is.
    ///
    /// The class is read from `tokenizer_config.json`, as the reference reads
    /// it, before any `config.json`: a checkpoint shipped as the tokenizer's
    /// files alone, with no `config.json` or none that says a `model_type`,
    /// is the class's, and so is one whose `config.json` names a model type
    /// the reference leaves to the named class (`llama`, `minicpm`, `zamba2`,
    /// and every type it has no tokenizer class of its own for). A
    /// `config.json` that names a model type the reference reads as its file
    /// ([`REFERENCE_READS_AS_FILE`]: `phi3`, `llava`, `mistral`, ...) keeps
    /// the file.
    fn llama_class_pipeline(
        tokenizer: &mut LoadingTokenizer,
        dir: Option<&Path>,
        config: &TokenizerConfigResult,
    ) {
        if !matches!(
            config.tokenizer_class.as_deref(),
            Some("LlamaTokenizer" | "LlamaTokenizerFast")
        ) {
            return;
        }
        if dir
            .and_then(model_type_from_config)
            .is_some_and(|model_type| reference_reads_as_file(&model_type))
        {
            return;
        }
        if tokenizer.get_pre_tokenizer().is_some()
            || !is_legacy_prepend_normalizer(tokenizer.get_normalizer())
        {
            return;
        }
        let add_prefix_space = config.add_prefix_space.unwrap_or(true);
        let scheme = match (add_prefix_space, config.legacy.unwrap_or(false)) {
            (false, _) => PrependScheme::Never,
            (true, true) => PrependScheme::Always,
            (true, false) => PrependScheme::First,
        };
        let class = config.tokenizer_class.as_deref().unwrap_or_default();
        // The class's decoder: the word marker as a space, the byte pieces,
        // the pieces fused, and the one leading space of the whole text off
        // only where the pre-tokenizer prepended the marker. A legacy file's
        // chain always ends in that `Strip`, which with `add_prefix_space:
        // false` would take the space a text starts with.
        let marker = match Replace::new("\u{2581}", " ") {
            Ok(marker) => marker,
            Err(error) => {
                warn!(class, %error, "the file's pipeline stays: the class's decoder did not build");
                return;
            }
        };
        let mut decoder: Vec<DecoderWrapper> = vec![
            marker.into(),
            ByteFallback::new().into(),
            Fuse::new().into(),
        ];
        if add_prefix_space {
            decoder.push(Strip::new(' ', 1, 0).into());
        }
        // A failure here leaves the checkpoint on the legacy file pipeline,
        // which this function exists to replace: said at `warn!`.
        if let Err(error) = tokenizer.with_normalizer(None::<NormalizerWrapper>) {
            warn!(class, %error, "the file's pipeline stays: the added vocabulary did not refresh without the normalizer");
            return;
        }
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', scheme, false)));
        tokenizer.with_decoder(Some(DecoderSequence::new(decoder)));
        debug!(
            ?scheme,
            add_prefix_space,
            "Llama-class pipeline: no normalizer, a Metaspace pre-tokenizer, the class's decoder"
        );
    }

    /// Check if the tokenizer's post_processor adds special tokens (e.g., BOS)
    fn tokenizer_adds_special_tokens<M: Model>(tokenizer: &TokenizerOf<M>) -> bool {
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
    pub(crate) const BPE_WORDS_CACHED_PER_THREAD: usize = 1024;

    /// Sizes the per-thread word cache of a built tokenizer's BPE model to
    /// `words` ([`DirectEncoders::bpe_words_cached`]); other model kinds have
    /// none. A built tokenizer hands out its model by shared reference only,
    /// so this installs a copy of the model with the bound: the price of a
    /// tokenizer handed over built ([`Self::from_tokenizer`]). The loaders
    /// read a file into a [`LoadingTokenizer`] and bound its model in place
    /// instead (see `word_cache`).
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
        let mut encoders = DirectEncoders::choose(&tokenizer);
        Self::bound_bpe_cache(&mut tokenizer, encoders.bpe_words_cached());
        encoders.seed_native(tokenizer.get_model());
        let special_tokens = Self::extract_special_tokens(&tokenizer, &ConfigTokens::default());
        let vocab = tokenizer.get_vocab(true); // true = include special tokens and added_tokens
        let reverse_vocab: HashMap<TokenIdType, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        let has_byte_fallback = tokenizer.get_decoder().is_some_and(has_byte_fallback);
        let byte_fallback_values = byte_fallback_values(&reverse_vocab, has_byte_fallback);
        let byte_level = ByteLevelTable::build(&tokenizer);
        let byte_level_chain =
            byte_level.is_none() && tokenizer.get_decoder().is_some_and(has_byte_level_member);
        let pieces = PieceTable::build(&tokenizer);
        log_decode_path(&tokenizer, byte_level.is_some(), pieces.is_some());
        HuggingFaceTokenizer {
            byte_level,
            pieces,
            #[cfg(feature = "tokenizers-v1")]
            v1: encoders.v1,
            core: Arc::new(EncoderCore::new(tokenizer, encoders.native)),
            special_tokens,
            vocab,
            reverse_vocab,
            chat_template: ChatTemplateState::empty(),
            response_template: None,
            eos_token_ids: Vec::new(), // No directory path in from_tokenizer
            renderer: Renderer::Jinja,
            glm: None,
            media_handling: Default::default(),
            deepseek_v31: None,
            tools_block: false,
            profile: None,
            has_byte_fallback,
            byte_fallback_values,
            byte_level_chain,
            structural_controls: false,
            special_tokens_marked: None,
        }
    }

    /// The special tokens: the configuration's where it names them, with a
    /// guess among the file's added special tokens standing in only for a
    /// token the configuration does not mention.
    ///
    /// The rule is `transformers`': a special token is what
    /// `tokenizer_config.json` sets (`"bos_token": "<|begin_of_text|>"`, a
    /// string or the added-token object), an explicit `null` is the
    /// checkpoint saying it has no such token (the attribute stays unset,
    /// whatever the tokenizer class's default; a template's `{{ bos_token }}`
    /// then renders nothing), and only a key the file leaves out falls back
    /// to the class's own default (`<s>` for the Llama classes, none for the
    /// Qwen2 class). The loader has no class table: the guess that stands in
    /// for the default looks the usual spellings up among the added tokens
    /// the file marks special, where a declared special token lives, never in
    /// the plain vocabulary, where a byte-level BPE may hold the text `<s>`
    /// as an ordinary piece.
    fn extract_special_tokens<M: Model>(
        tokenizer: &TokenizerOf<M>,
        config_tokens: &ConfigTokens,
    ) -> SpecialTokens {
        // The added tokens the file marks special (`<|im_start|>` and the
        // like): the only tokens a guess may name.
        let additional_special_tokens: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_id, token)| token.special)
            .map(|(_id, token)| token.content.clone())
            .collect();

        let find_token = |patterns: &[&str]| -> Option<String> {
            patterns
                .iter()
                .find(|pattern| {
                    additional_special_tokens
                        .iter()
                        .any(|token| token == *pattern)
                })
                .map(|pattern| (*pattern).to_string())
        };

        SpecialTokens {
            bos_token: config_tokens
                .bos_token
                .or_guess(|| find_token(&["<s>", "<|startoftext|>", "<BOS>", "[CLS]"])),
            eos_token: config_tokens
                .eos_token
                .or_guess(|| find_token(&["</s>", "<|endoftext|>", "<EOS>", "[SEP]"])),
            unk_token: config_tokens
                .unk_token
                .or_guess(|| find_token(&["<unk>", "<UNK>", "[UNK]"])),
            sep_token: find_token(&["[SEP]", "<sep>", "<SEP>"]),
            pad_token: config_tokens
                .pad_token
                .or_guess(|| find_token(&["<pad>", "<PAD>", "[PAD]"])),
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

            // The chat template(s) from the parsed config (avoid re-reading the file)
            let templates = ConfigChatTemplates::read(config.get("chat_template"));
            // Kept raw; parser selection validates it.
            let response_template = config.get("response_template").cloned();

            let add_bos_token = config.get("add_bos_token").and_then(|v| v.as_bool());
            let add_eos_token = config.get("add_eos_token").and_then(|v| v.as_bool());
            let tokenizer_class = config
                .get("tokenizer_class")
                .and_then(|v| v.as_str())
                .map(String::from);
            let legacy = config.get("legacy").and_then(|v| v.as_bool());
            let add_prefix_space = config.get("add_prefix_space").and_then(|v| v.as_bool());

            let config_tokens = ConfigTokens {
                bos_token: ConfigToken::read(&config, "bos_token"),
                eos_token: ConfigToken::read(&config, "eos_token"),
                unk_token: ConfigToken::read(&config, "unk_token"),
                pad_token: ConfigToken::read(&config, "pad_token"),
            };

            Some(TokenizerConfigResult {
                chat_template: templates.default,
                tool_use_chat_template: templates.tool_use,
                response_template,
                add_bos_token,
                add_eos_token,
                tokenizer_class,
                legacy,
                add_prefix_space,
                config_tokens,
            })
        })()
        .unwrap_or_default()
    }

    /// The Jinja renderer, with this tokenizer's special tokens unless the
    /// caller passed its own. The flag says whether the prompt continues the
    /// final message (see `ChatTemplateState::apply_with_continuation`).
    fn apply_jinja(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<(String, bool)> {
        // A marked template (see `structural_controls`) writes its control
        // tokens between marks, the special tokens it reads included, and
        // reads every request string (messages, tools, documents, template
        // kwargs) with its marks removed, so no request can forge one.
        let inputs = marks::TemplateInputs::new(
            messages,
            &params,
            self.special_tokens_marked
                .as_ref()
                .filter(|_| self.chat_template.marked()),
            &self.special_tokens,
        );
        let (messages, params) = inputs.split(params);
        // A template that writes nothing for the request's tools gets them
        // as a system message block (the V3.1 template, see `deepseek_v31`).
        let with_block = self
            .tools_block
            .then(|| deepseek_v31::messages_with_tools_block(messages, params.tools))
            .flatten();
        let messages = with_block.as_deref().unwrap_or(messages);
        self.chat_template.apply_with_continuation(messages, params)
    }

    fn apply_glm53(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        template: glm53::Template,
    ) -> Result<(String, bool)> {
        // MiniJinja permits kwargs to replace the base context. Preserve that
        // uncommon API escape hatch rather than silently ignoring an override.
        if params.template_kwargs.is_some_and(|kwargs| {
            ["messages", "tools", "add_generation_prompt"]
                .iter()
                .any(|key| kwargs.contains_key(*key))
        }) {
            self.apply_jinja(messages, params)
        } else {
            glm53::render(messages, params, template)
        }
    }
}

/// The special tokens as `tokenizer_config.json` gives them.
#[derive(Default)]
struct ConfigTokens {
    bos_token: ConfigToken,
    eos_token: ConfigToken,
    unk_token: ConfigToken,
    pad_token: ConfigToken,
}

/// One special token of `tokenizer_config.json`: the key may be left out,
/// set to `null`, or carry the token (a string, or the added-token object
/// whose `content` is the string). The three read apart because they mean
/// three things to `transformers` (see
/// [`HuggingFaceTokenizer::extract_special_tokens`]).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
enum ConfigToken {
    /// The key is not in the file: the tokenizer class's own default
    /// applies, which the loader's guess stands in for.
    #[default]
    Unmentioned,
    /// The key is `null`: the checkpoint has no such token.
    Null,
    /// The configured token.
    Set(String),
}

impl ConfigToken {
    /// `key` as the parsed `tokenizer_config.json` holds it. A value of
    /// another shape (neither a string, an object with a string `content`,
    /// nor `null`) reads as unmentioned, as it did before the three were
    /// told apart.
    fn read(config: &serde_json::Value, key: &str) -> Self {
        match config.get(key) {
            None => Self::Unmentioned,
            Some(serde_json::Value::Null) => Self::Null,
            Some(serde_json::Value::String(token)) => Self::Set(token.clone()),
            Some(value) => value
                .get("content")
                .and_then(serde_json::Value::as_str)
                .map_or(Self::Unmentioned, |token| Self::Set(token.to_string())),
        }
    }

    /// The token, with `guess` standing in only where the configuration
    /// does not mention it: a configured token is itself, a `null` is none.
    fn or_guess(&self, guess: impl FnOnce() -> Option<String>) -> Option<String> {
        match self {
            Self::Set(token) => Some(token.clone()),
            Self::Null => None,
            Self::Unmentioned => guess(),
        }
    }
}

/// Result of parsing tokenizer_config.json.
#[derive(Default)]
struct TokenizerConfigResult {
    chat_template: Option<String>,
    /// The `tool_use` entry of a named-template list (see `ConfigChatTemplates`).
    tool_use_chat_template: Option<String>,
    response_template: Option<serde_json::Value>,
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    /// The `transformers` class the checkpoint names for its tokenizer.
    tokenizer_class: Option<String>,
    /// The Llama classes' `legacy` flag (the prepend scheme of their
    /// pre-tokenizer).
    legacy: Option<bool>,
    /// `add_prefix_space`, when the configuration sets it.
    add_prefix_space: Option<bool>,
    config_tokens: ConfigTokens,
}

/// `model_type` of the checkpoint's `config.json` in `dir`, if the file is
/// there and says one.
fn model_type_from_config(dir: &Path) -> Option<String> {
    let content = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&content).ok()?;
    value.get("model_type")?.as_str().map(String::from)
}

/// The model types whose checkpoints `transformers` reads as their
/// `tokenizer.json` although `tokenizer_config.json` names the Llama class:
/// the types it lists as carrying a wrong class on the Hub, and the ones it
/// registers to a plain or Mistral backend (its 5.22.1 tables, sorted). For
/// every other type, registered to a class of its own or to none, the named
/// class builds the pipeline, so a type missing here reads as the reference
/// reads a type it does not know.
const REFERENCE_READS_AS_FILE: &[&str] = &[
    "EvollaModel",
    "arctic",
    "aria",
    "beam",
    "bigbird_pegasus",
    "camembertv2-base",
    "canary",
    "chatlm",
    "cohere_asr",
    "ernie4_5",
    "ernie4_5_moe",
    "flex_olmo",
    "fuyu",
    "glm",
    "glm4",
    "glm4_moe",
    "glm4_moe_lite",
    "glm4v",
    "glm4v_moe",
    "glm_image",
    "glmasr",
    "got_ocr2",
    "gpt_bigcode",
    "gradium_phonon_tts",
    "granite",
    "granitemoe",
    "granitemoehybrid",
    "granitemoeshared",
    "h2ovl_chat",
    "hyperclovax",
    "hyperclovax_vision_v2",
    "hyperclovax_vlm",
    "internlm2",
    "jamba",
    "kimi_k25",
    "kimi_linear",
    "kosmos-2",
    "llava",
    "llava_next",
    "mimo_v2_flash",
    "minicpm3",
    "minicpm_ocr",
    "minicpmv",
    "minicpmv4_6",
    "minicpmv4_7",
    "minimax_m2",
    "ministral",
    "ministral3",
    "mistral",
    "mistral3",
    "mistral_large4",
    "mixtral",
    "modernbert",
    "modernbert-decoder",
    "molmo",
    "molmo2",
    "nemo_speechlm",
    "nemotron",
    "nvfp4",
    "nystromformer",
    "olmo2",
    "olmo3",
    "olmo_hybrid",
    "opencua",
    "openvla",
    "paddleocr_vl",
    "pegasus_x",
    "persimmon",
    "phi3",
    "phi3_v",
    "phimoe",
    "pix2struct",
    "pixtral",
    "pocket_tts",
    "qwen2",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen4_exp",
    "siglip2",
    "smolvlm",
    "stablelm",
    "step3_vl",
    "umt5",
    "vipllava",
    "voxtral",
    "voxtral_realtime",
    "xlm-roberta-xl",
];

/// Whether the reference reads a Llama-class checkpoint of `model_type` as
/// its file ([`REFERENCE_READS_AS_FILE`]).
fn reference_reads_as_file(model_type: &str) -> bool {
    REFERENCE_READS_AS_FILE.binary_search(&model_type).is_ok()
}

/// Whether `normalizer` is the one a legacy conversion of a sentencepiece
/// vocabulary writes: a sequence that prepends the word marker U+2581 and
/// then only replaces the space by that marker. A file that replaces
/// anything else is not that conversion, and keeps its own pipeline.
fn is_legacy_prepend_normalizer(normalizer: Option<&NormalizerWrapper>) -> bool {
    let Some(NormalizerWrapper::Sequence(sequence)) = normalizer else {
        return false;
    };
    let mut steps = sequence.as_ref().iter();
    matches!(steps.next(), Some(NormalizerWrapper::Prepend(prepend)) if prepend.prepend == "\u{2581}")
        && steps.all(is_legacy_space_replace)
}

/// Whether `step` is the legacy conversion's `Replace`: the one space by the
/// word marker. `tokenizers` keeps the pattern private, so it is read through
/// the step's own serialized form (`{"String": " "}`), once, at load time.
fn is_legacy_space_replace(step: &NormalizerWrapper) -> bool {
    let NormalizerWrapper::Replace(replace) = step else {
        return false;
    };
    replace.content == "\u{2581}"
        && serde_json::to_value(replace)
            .ok()
            .and_then(|value| value.get("pattern").cloned())
            .is_some_and(|pattern| pattern == serde_json::json!({"String": " "}))
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        #[cfg(feature = "tokenizers-v1")]
        if let Some(v1) = &self.v1 {
            return v1.encode(input, add_special_tokens);
        }
        // The direct path frames the ids itself on a request for special
        // tokens (what the post-processor it read at load adds). An input it
        // gives up is counted and encoded by `tokenizers`.
        if let Some(native) = &self.core.native {
            match native.encode(
                Some(self.core.tokenizer.get_model()),
                input,
                add_special_tokens,
            ) {
                Ok(ids) => return Ok(Encoding::Plain(ids)),
                Err(fallback) => encode_path::note_fallback(fallback),
            }
        }
        self.core
            .tokenizer
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
        if self.core.native.is_some() {
            return inputs
                .iter()
                .map(|input| self.encode(input, add_special_tokens))
                .collect();
        }
        self.core
            .tokenizer
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
        new_ids: &[TokenIdType],
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
        ids.extend_from_slice(new_ids);
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
        Ok(
            match self.stepped(
                token_id,
                ids,
                prefix,
                prefix_index,
                skip_special_tokens,
                eager,
                count,
            )? {
                Stepped::Piece(text) => DecodeStep {
                    text: Some(text.to_owned()),
                    prefix_mismatch: false,
                },
                Stepped::Step(step) => step,
            },
        )
    }

    /// One token through the step algorithm, with its text where the step
    /// found it: a plain piece borrowed from the piece table, or the step as
    /// the window decode made it. The one algorithm behind
    /// [`Self::client_step`] and `Decoder::decode_step_into`.
    #[expect(clippy::too_many_arguments)]
    fn stepped(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
        eager: bool,
        count: bool,
    ) -> Result<Stepped<'_>> {
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
            return Ok(Stepped::Step(DecodeStep {
                text: Some('\u{FFFD}'.to_string()),
                prefix_mismatch: false,
            }));
        }
        if let Some(table) = &self.pieces {
            // Nothing pending and a plain piece: the window's text grows by
            // exactly this piece's text (the leading-space strip touched the
            // window's first piece, which this one is not), so it streams at
            // once, borrowed from the table, and becomes the window on its
            // own, where the strip applies to it: the prefix the full step
            // would record, written without a render.
            if *prefix_index == ids.len() && !prefix.is_empty() && !prefix.ends_with('\u{FFFD}') {
                if let Some(text) = table.plain_text(token_id, skip_special_tokens) {
                    ids.clear();
                    ids.push(token_id);
                    prefix.clear();
                    prefix.push_str(table.at_window_start(text));
                    *prefix_index = 1;
                    return Ok(Stepped::Piece(text));
                }
            }
        }
        let step = self.step_decode_lenient(
            &[token_id],
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
                    return Ok(Stepped::Step(DecodeStep {
                        text: Some(replacements),
                        prefix_mismatch: rewritten,
                    }));
                }
            }
        }
        Ok(Stepped::Step(step))
    }
}

/// What a step gave ([`HuggingFaceTokenizer::stepped`]): a plain piece's text,
/// borrowed from the piece table, or the step as the full algorithm made it.
enum Stepped<'t> {
    Piece(&'t str),
    Step(DecodeStep),
}

impl Decoder for HuggingFaceTokenizer {
    fn uses_byte_fallback(&self) -> bool {
        self.has_byte_fallback
    }

    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        // A byte-level vocabulary decodes through the table its streams use:
        // every id's bytes concatenated and one UTF-8 conversion, in place of
        // the `tokenizers` decoder chain over every piece.
        if let Some(table) = &self.byte_level {
            return Ok(table.decode(token_ids, skip_special_tokens));
        }
        // A sentencepiece-style vocabulary with byte fallback decodes through
        // its piece table, as its streaming windows do: one pass over the ids
        // in place of the `tokenizers` decoder chain over every piece.
        if let Some(table) = &self.pieces {
            return Ok(table.render(token_ids, skip_special_tokens));
        }
        self.core
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| Error::msg(format!("Decoding failed: {e}")))
    }

    /// Native incremental decode (the raw step: a recovery here is not counted,
    /// see `Self::step_decode_lenient`): the algorithm of the HF `step_decode_stream`
    /// (hold while the decode ends in an incomplete character, emit what grew
    /// beyond the emitted prefix), run through the concrete `TokenizerImpl`
    /// rather than `dyn Decoder` dispatch, and lenient when the decoder chain
    /// rewrites text already emitted (see `Self::step_decode_lenient`).
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
                &[token_id],
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

    /// The reporting step with its text appended to `out`: a plain piece on a
    /// vocabulary with a piece table goes from the table into the buffer, so a
    /// stream of such pieces allocates nothing per token.
    fn decode_step_into(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
        eager: bool,
        out: &mut String,
    ) -> Result<AppendedStep> {
        Ok(
            match self.stepped(
                token_id,
                ids,
                prefix,
                prefix_index,
                skip_special_tokens,
                eager,
                true,
            )? {
                Stepped::Piece(text) => {
                    out.push_str(text);
                    AppendedStep {
                        appended: true,
                        prefix_mismatch: false,
                    }
                }
                Stepped::Step(step) => AppendedStep {
                    appended: append_step_text(step.text, out),
                    prefix_mismatch: step.prefix_mismatch,
                },
            },
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
        if !self.has_byte_fallback && ids.len() <= *prefix_index {
            // Nothing held past the emitted text: a window without byte runs
            // holds only while its decode ends in an incomplete character.
            ids.clear();
            prefix.clear();
            *prefix_index = 0;
            return Ok(DecodeStep::default());
        }
        if prefix.is_empty() && ids.len() == *prefix_index {
            // A seeded sequence that generated nothing: the seed is prompt text.
            ids.clear();
            *prefix_index = 0;
            return Ok(DecodeStep::default());
        }
        if self.has_byte_fallback {
            // The byte tokens of a character the end cut short come off the
            // window: the end releases the text before them and nothing for
            // the character, where the window's decode would render one
            // replacement character per byte of it.
            let held = (*prefix_index).min(ids.len());
            let cut = self.cut_tail(&ids[held..], skip_special_tokens);
            ids.truncate(ids.len() - cut);
        }
        let decoded = self.window_decode(ids, skip_special_tokens)?;
        let (mut pending, prefix_mismatch) = match decoded.strip_prefix(prefix.as_str()) {
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
        if self.byte_level_chain
            && pending.ends_with(char::REPLACEMENT_CHARACTER)
            && self.byte_level_window_ends_cut(
                &ids[(*prefix_index).min(ids.len())..],
                skip_special_tokens,
            )
        {
            // The chain's byte-level decoder renders a sequence the end cut
            // short as one replacement character at the very end: the start
            // of a character the model never finished, released as nothing.
            pending.pop();
        }
        ids.clear();
        prefix.clear();
        *prefix_index = 0;
        Ok(DecodeStep {
            text: (!pending.is_empty()).then_some(pending),
            prefix_mismatch,
        })
    }
}

impl HuggingFaceTokenizer {
    /// The prompt of a renderer with an encoder of its own, its control
    /// tokens marked (see `encoders::marks`); the Jinja renderer has none.
    fn render_marked(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<(String, bool)> {
        match self.renderer {
            // The template compiled marked: whether it continued the final
            // message comes with the render.
            Renderer::Jinja if self.chat_template.marked() => self.apply_jinja(messages, params),
            Renderer::Jinja | Renderer::Glm53(_) => Err(Error::msg(
                "the Jinja and GLM renderers render no marked prompt",
            )),
            Renderer::DeepseekV32 => {
                apply_deepseek_v32(messages, &params).map(|marked| (marked, false))
            }
            Renderer::DeepseekV4 => {
                apply_deepseek_v4(messages, &params).map(|marked| (marked, false))
            }
            Renderer::DeepseekV41(table) => {
                apply_deepseek_v41(messages, &params, table).map(|marked| (marked, false))
            }
        }
    }

    /// The prompt of a renderer with an encoder of its own, encoded by
    /// segments (see `apply_chat_template_with_encoding`).
    fn apply_marked_with_encoding(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        // The native encoders read neither flag, as the engines' own encoders
        // do, and the flags report what the renderer wrote: a continuation by
        // the V4.1 encoder alone, on a trailing assistant message, when the
        // request asks (`wo_eos`; V4 closes it with EOS); the header read from
        // the marked render for V4 and V4.1 (their encoders' final transition,
        // after their own preprocessing), and from the final message's role
        // for V3.2, which writes the header inside every user, developer and
        // tool-result render, before any trailing system text.
        let generation_prompt_unless_continued = params.generation_prompt_unless_continued();
        let continued_by_encoder = matches!(self.renderer, Renderer::DeepseekV41(_))
            && continues_final_assistant_message(messages, &params);
        let (mut marked, continued_by_template) = self.render_marked(messages, params)?;
        let continued_final_message = continued_by_encoder || continued_by_template;
        let generation_prompt = match self.renderer {
            Renderer::DeepseekV32 => !final_message_is_assistant(messages),
            Renderer::DeepseekV4 | Renderer::DeepseekV41(_) => ends_with_generation_header(&marked),
            // The template's header is the flag it rendered under.
            Renderer::Jinja => generation_prompt_unless_continued && !continued_final_message,
            // No marked prompt: `render_marked` refused above.
            Renderer::Glm53(_) => false,
        };
        if let Some(prefix) = assistant_prefix {
            // Request text: a mark a client wrote into it is no mark.
            marked.push_str(&marks::strip(prefix));
        }
        let (text, controls) = marks::flatten(&marked)?;
        let encoding = if self.core.flat_encode_is_segment_encode(&text, &controls) {
            PromptEncoding::FromText
        } else {
            let segments = marks::segments(&marked)?;
            let core = Arc::clone(&self.core);
            PromptEncoding::Deferred(EncodeJob::new(move || {
                core.encode_segments(&segments).map(Encoding::Plain)
            }))
        };
        Ok(ChatTemplateOutput {
            text,
            encoding,
            unbilled_prompt_tokens: 0,
            continued_final_message,
            generation_prompt,
        })
    }
}

impl HuggingFaceTokenizer {
    /// The template questions, answered by the profile or by the template.
    fn template_answers(&self) -> TemplateAnswers<'_> {
        TemplateAnswers {
            profile: self.profile.as_ref(),
            template: &self.chat_template,
        }
    }
}

impl TokenizerTrait for HuggingFaceTokenizer {
    fn vocab_size(&self) -> usize {
        self.core.tokenizer.get_vocab_size(false)
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

    fn response_template(&self) -> Option<&serde_json::Value> {
        self.response_template.as_ref()
    }

    fn eos_token_ids(&self) -> &[TokenIdType] {
        &self.eos_token_ids
    }

    fn added_token_strings(&self) -> Vec<String> {
        self.core
            .tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|token| !token.special && !token.normalized)
            .map(|token| token.content.clone())
            .collect()
    }

    fn added_tokens_after_normalization(&self) -> Vec<String> {
        self.core
            .tokenizer
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
            Renderer::Jinja if !self.chat_template.marked() => {
                self.apply_jinja(messages, params).map(|(text, _)| text)
            }
            Renderer::Glm53(template) => self
                .apply_glm53(messages, params, template)
                .map(|(text, _)| text),
            _ => self
                .render_marked(messages, params)
                .map(|(marked, _)| marks::strip(&marked).into_owned()),
        }
    }

    /// The Jinja renderer of a plain checkpoint and the GLM renderer encode
    /// from their text. A renderer with an encoder of its own, and the Jinja
    /// renderer of a checkpoint with `structural_controls`, render their
    /// control tokens marked (see `encoders::marks`) and the prompt is
    /// encoded by segments, so that a control-token spelling typed in a
    /// message stays text: as text all the same when the flat encode would
    /// cut out exactly the renderer's control tokens (the ids are then the
    /// same, and the text takes every path and cache a flat encode takes),
    /// else as a deferred encode of the segments.
    fn apply_chat_template_with_encoding(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        // The Jinja and GLM renderers say whether they continued the final
        // message or appended its text after the generation prompt; their
        // header is the flag they rendered under (the template's
        // `add_generation_prompt` variable is that flag: the render context
        // keeps the field's value over a `template_kwargs` entry of that name).
        let generation_prompt_unless_continued = params.generation_prompt_unless_continued();
        let (mut text, continued_final_message) = match self.renderer {
            Renderer::Jinja if !self.chat_template.marked() => {
                self.apply_jinja(messages, params)?
            }
            Renderer::Glm53(template) => self.apply_glm53(messages, params, template)?,
            _ => return self.apply_marked_with_encoding(messages, params, assistant_prefix),
        };
        if let Some(prefix) = assistant_prefix {
            text.push_str(prefix);
        }
        Ok(ChatTemplateOutput {
            text,
            encoding: PromptEncoding::FromText,
            unbilled_prompt_tokens: 0,
            continued_final_message,
            generation_prompt: generation_prompt_unless_continued && !continued_final_message,
        })
    }

    fn apply_chat_template_with_encoding_owned(
        &self,
        mut messages: Vec<serde_json::Value>,
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        // The shaping the Jinja render does for an OpenAI-format template
        // (`ChatTemplateState::render_with_continuation`: every string content
        // as a one-item text part list), done in place on the caller's
        // vector first; the render then finds no string content to wrap and
        // copies no message.
        if matches!(self.renderer, Renderer::Jinja)
            && !self.chat_template.marked()
            && self
                .chat_template
                .content_format_for_tools(params.tools.is_some())
                == ChatTemplateContentFormat::OpenAI
        {
            crate::chat_template::wrap_string_content_in_place(&mut messages);
        }
        self.apply_chat_template_with_encoding(&messages, params, assistant_prefix)
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.template_answers().content_format()
    }

    fn chat_template_content_format_for_request(
        &self,
        has_tools: bool,
    ) -> ChatTemplateContentFormat {
        self.template_answers().content_format_for_tools(has_tools)
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        self.template_answers().thinking_toggle()
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        self.template_answers().thinking_key_name()
    }

    fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        self.template_answers().native_reasoning_effort_values()
    }

    fn native_reasoning_effort_off_values(&self) -> &'static [&'static str] {
        self.template_answers().native_reasoning_effort_off_values()
    }

    fn think_in_prefill(&self) -> bool {
        self.template_answers().think_in_prefill()
    }

    fn prefill_opens_think_block(&self) -> bool {
        self.template_answers().prefill_opens_think_block()
    }

    /// The profile's, or the Jinja template's with the media handling resolved
    /// for this checkpoint.
    fn renderer_capabilities(&self) -> crate::traits::RendererCapabilities {
        self.template_answers()
            .renderer_capabilities(self.media_handling)
    }

    fn native_thinking_mode(
        &self,
        template_kwargs: Option<&HashMap<String, serde_json::Value>>,
        thinking: Option<bool>,
    ) -> Option<bool> {
        self.template_answers()
            .native_thinking_mode(template_kwargs, thinking)
    }

    /// Refresh media handling and the tools block only after a replacement
    /// template compiles successfully.
    fn set_chat_template(&mut self, template: String) -> Result<()> {
        let glm_renderer = self
            .glm
            .as_ref()
            .and_then(|checkpoint| checkpoint.renderer(Some(&template)));
        let controls = self
            .structural_controls
            .then(|| self.core.added())
            .flatten();
        self.chat_template.replace_marked(template, controls)?;
        if self.glm.is_some() {
            self.renderer = glm_renderer.map_or(Renderer::Jinja, Renderer::Glm53);
            self.profile = self.renderer.profile();
        }
        self.media_handling = self
            .glm
            .as_ref()
            .map_or_else(Default::default, |checkpoint| {
                checkpoint.media_handling(&self.chat_template)
            });
        self.tools_block = self.deepseek_v31.is_some()
            && deepseek_v31::Checkpoint::renders_tools_block(&self.chat_template);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Renderer detection (config.json::architectures)
// ---------------------------------------------------------------------------
/// Inspect the sibling `config.json` to decide which chat-template renderer to
/// use. A missing or malformed file falls back to [`Renderer::Jinja`] without
/// erroring (debug-logged), preserving backward compatibility for every model
/// not in the architecture list. `has_chat_template` says whether the
/// checkpoint (its configuration or a template file) or the operator gave a
/// chat template: the DeepSeek V4 port stands in for the encoder module those
/// checkpoints ship in place of one, so a checkpoint of that architecture
/// with a template renders through the template, as the engine's own
/// renderer does.
fn detect_renderer_from_config(dir: &Path, has_chat_template: bool) -> Renderer {
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
        let table = detect_dsv41_effort_table(dir);
        debug!(?path, ?table, "selected DeepseekV41 chat-template renderer");
        return Renderer::DeepseekV41(table);
    }
    if arch_strs.contains(&"DeepseekV4ForCausalLM") {
        if has_chat_template {
            debug!(
                ?path,
                "DeepseekV4 architecture with a chat template of its own; using Jinja renderer"
            );
            return Renderer::Jinja;
        }
        debug!(?path, "selected DeepseekV4 chat-template renderer");
        return Renderer::DeepseekV4;
    }
    Renderer::Jinja
}

/// Pick the effort table of a V4.1 checkpoint from its shipped
/// `encoding/encoding.py` (`config.json` and `tokenizer_config.json` are the
/// same across the revisions). Without it (a tokenizer directory streamed
/// from a worker carries no `encoding/`), the rescaled table the checkpoint
/// carries today, which its chat template writes too.
fn detect_dsv41_effort_table(dir: &Path) -> deepseek_v41::EffortTable {
    match std::fs::read_to_string(dir.join("encoding").join("encoding.py")) {
        Ok(source) => deepseek_v41::EffortTable::detect_from_encoder_source(&source),
        Err(_) => deepseek_v41::EffortTable::Rescaled,
    }
}

// ---------------------------------------------------------------------------
// DeepSeek V3.2 / V4 dispatch shims
// ---------------------------------------------------------------------------
/// Derive the V3.2 thinking mode. The native encoder bypasses
/// `ChatTemplateState::apply`, so this is where the resolved thinking preference
/// is consumed. An explicit `template_kwargs["thinking"]` wins, then the
/// engine's `enable_thinking` alias; otherwise fall back to `params.thinking`
/// (resolved from `reasoning_effort` / Anthropic `ThinkingConfig`) — same
/// precedence as the Jinja path and as the gateway's parser arming. The
/// engine's tokenizer wrapper ORs the two keys; `thinking` wins here when both
/// are present, as it does in the parser arming and in the V4 shim, so the
/// prompt and the parser agree (a contradictory pair renders chat mode here
/// where the engine would think). Default off, matching the
/// `ThinkingKeyName::Thinking` / `DefaultOff` contract reported here.
fn derive_thinking_mode(params: &ChatTemplateParams) -> deepseek_v32::ThinkingMode {
    if explicit_thinking(params).unwrap_or(false) {
        deepseek_v32::ThinkingMode::Thinking
    } else {
        deepseek_v32::ThinkingMode::Chat
    }
}

/// Explicit thinking preference: the `thinking` kwarg, else its
/// `enable_thinking` alias ([`explicit_thinking_kwarg`]), else
/// `params.thinking`.
fn explicit_thinking(params: &ChatTemplateParams) -> Option<bool> {
    explicit_thinking_kwarg(params).or(params.thinking)
}

/// Per DeepSeek's encoding README, preserve all reasoning when a system
/// message declares `tools`; otherwise drop earlier reasoning. (The shims read
/// the list with the developer role as written, as the engine's renderers
/// hand it to their tokenizer wrappers, and a request's tools ride on the
/// system message [`inject_tools_into_messages`] attaches them to.)
fn resolve_drop_thinking(messages: &[serde_json::Value]) -> bool {
    !messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
            && m.get("tools")
                .and_then(|t| t.as_array())
                .is_some_and(|arr| !arr.is_empty())
    })
}
/// Attach `tools` to a leading system message so the V3.2/V4
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
    if first_role != Some("system") {
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
    // A developer message reaches the encoder as written, as the engine's
    // renderer hands it to its tokenizer wrapper (no developer-to-system
    // rewrite on this path): the encoder renders it as a user turn under a
    // heading. The request's tools ride on the system message the wrapper
    // step attaches them to.
    let owned = inject_tools_into_messages(messages, params.tools);
    let msgs: &[serde_json::Value] = owned.as_deref().unwrap_or(messages);
    let thinking_mode = derive_thinking_mode(params);
    let encode_params = deepseek_v32::EncodeParams {
        add_default_bos_token: true,
        drop_thinking: resolve_drop_thinking(msgs),
    };
    deepseek_v32::render_messages(msgs, thinking_mode, &encode_params)
        .map_err(|e| Error::msg(format!("DeepSeek V3.2 encode failed: {e}")))
}
/// The V3.2 and V4 shims' explicit thinking toggle: `template_kwargs["thinking"]`,
/// else the engine's `enable_thinking` alias (the gateway reads the two keys
/// in the same order when it arms the reasoning parser, see
/// `renderer_capabilities`).
fn explicit_thinking_kwarg(params: &ChatTemplateParams) -> Option<bool> {
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
/// the table the engine's copy of the encoder carries for every V4
/// checkpoint, whatever the checkpoint's own `encoding/encoding_dsv4.py`
/// says: the engine never reads it, so neither does this renderer.
///
/// Deliberate divergence from the engine's Python renderer: there `reasoning_effort: "none"`
/// forces chat mode even over an explicit `thinking: true`. Here the explicit
/// toggle wins, because the gateway arms the reasoning parser from the
/// explicit toggle first, and rendering chat mode for that contradictory
/// input would have the armed parser swallow the whole answer as reasoning.
fn apply_deepseek_v4(
    messages: &[serde_json::Value],
    params: &ChatTemplateParams,
) -> Result<String> {
    // A developer message reaches the encoder as written, as the engine's
    // renderer hands it to its tokenizer wrapper (no developer-to-system
    // rewrite on this path): the encoder drops one before the last user turn
    // with the thinking and renders it as a user turn otherwise. The
    // request's tools ride on the system message the wrapper step attaches
    // them to. (The gateway's Responses path is unlike the chat path here: it
    // folds a developer item into the system prompt before any renderer sees
    // the conversation, so the drop is the chat path's alone.)
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
    let thinking_on = explicit_thinking_kwarg(params)
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
    };
    deepseek_v4::render_messages(msgs, thinking_mode, &encode_params)
        .map_err(|e| Error::msg(format!("DeepSeek V4 encode failed: {e}")))
}

// ---------------------------------------------------------------------------
// DeepSeek V4.1 dispatch shim
// ---------------------------------------------------------------------------
/// Attach `tools` to the FIRST message whose role is `system`, wherever it
/// appears in the conversation — the engine's V4.1 rule. This differs from V3.2/V4's
/// [`inject_tools_into_messages`], which only rewrites a *leading* system
/// message. Synthesizes an empty leading system message when none exists.
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

/// Whether the conversation ends on an assistant message: the one shape the
/// native encoders write no generation header for (closed with EOS, or
/// continued).
fn final_message_is_assistant(messages: &[serde_json::Value]) -> bool {
    messages
        .last()
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        == Some("assistant")
}

/// Whether a render continues its final message natively: asked to, and the
/// final message is an assistant one (the only shape the V4.1 encoder
/// continues, as `wo_eos` on that message).
fn continues_final_assistant_message(
    messages: &[serde_json::Value],
    params: &ChatTemplateParams,
) -> bool {
    params.continue_final_message && final_message_is_assistant(messages)
}

/// DeepSeek V4.1 chat-template shim. Order: attach tools to the first system
/// message (the engine's rule) -> resolve `reasoning_effort` -> resolve the
/// thinking mode -> `drop_thinking` -> stamp `wo_eos` on a trailing assistant
/// message when `continue_final_message` asks -> encode.
///
/// `continue_final_message` with a trailing assistant message is the
/// encoder's `wo_eos` route (no EOS, no generation header). The gateway sends
/// it for a trailing assistant turn because
/// `renderer_capabilities().native_assistant_continuation` is declared, and
/// does not arm the reasoning parser for that shape: the message is rendered
/// past its `</think>`, so the completion starts in content mode.
/// `add_generation_prompt` is not read: the engine's own encoder reads neither
/// flag and writes the generation header whatever the request asked, so the
/// render reports the header it wrote (`ChatTemplateOutput::generation_prompt`)
/// and the gateway's reasoning decisions follow that rather than the request.
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
///    a native effort name (`low`/`medium`/`high`/`xhigh`/`max`) switches it
///    on, and an integer budget has no opinion;
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
    effort_table: deepseek_v41::EffortTable,
) -> Result<String> {
    // `template_kwargs["response_format"]` is deliberately not attached to a
    // message: the engine's Python renderer never renders V4.1's `## Response Format:` block
    // (structured output is enforced by the constraint), so the gateway's
    // projected kwarg is ignored here too. The encoder still renders a
    // per-message `response_format` for direct callers.
    // A developer message reaches the encoder as written, as the engine's
    // renderer hands it to its tokenizer wrapper (no developer-to-system
    // rewrite on this path, see `apply_deepseek_v4`); the request's tools
    // ride on the system message the wrapper step attaches them to.
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

    // `continue_final_message` with a trailing assistant message reaches the
    // encoder as `wo_eos` on that message: no EOS, no generation header. That
    // is the only shape wired; `add_generation_prompt` alone is not read, as
    // the engine's own encoder reads neither flag: with any other shape the
    // generation header is written where the conversation takes one (V3.2/V4
    // read neither flag either).
    let continues_final_assistant_message = continues_final_assistant_message(&msgs, params);
    if continues_final_assistant_message {
        if let Some(obj) = msgs.last_mut().and_then(serde_json::Value::as_object_mut) {
            obj.insert("wo_eos".into(), serde_json::Value::Bool(true));
        }
    }

    let encode_params = deepseek_v41::EncodeParams {
        add_default_bos_token: true,
        drop_thinking,
        reasoning_effort,
        effort_table,
    };
    deepseek_v41::render_messages(&msgs, thinking_mode, &encode_params)
        .map_err(|e| Error::msg(format!("DeepSeek V4.1 encode failed: {e}")))
}

#[cfg(test)]
mod tests;
