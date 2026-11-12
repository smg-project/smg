use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
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
    byte_level::{ByteLevelIncremental, ByteLevelTable, ByteLevelTableBuilder, UnknownId},
    chat_template::{
        load_chat_template_from_file, python_isspace, ChatTemplateContentFormat,
        ChatTemplateParams, ChatTemplateState, ConfigChatTemplates, PreparedTools, ThinkingKeyName,
        ThinkingToggle,
    },
    encode_path::{self, EncodePath},
    encoders::{
        join_segments,
        kimi_k25_tools::apply_kimi_k25_tools,
        kimi_k3_xtml::{
            apply_kimi_k3_xtml_with_effort_default, render_kimi_k3_xtml_prompt, KimiXtml,
            RenderedXtml,
        },
        marks, PromptSegment, RendererProfile, TemplateAnswers,
    },
    factory::discover_chat_template_in_dir,
    kimi_k2_tokenizer,
    native::{AddedTokens, NativeEncoder},
    tekken,
    traits::{
        ChatTemplateOutput, Decoder, EncodeJob, Encoder, Encoding, IncrementalDecoder,
        PromptEncoding, RendererCapabilities, SpecialTokens, TokenIdType,
        Tokenizer as TokenizerTrait,
    },
};

#[derive(Debug, Clone, Copy)]
enum Renderer {
    Jinja,
    KimiK25Tools,
    /// Kimi-K3 and Kimi-K3.1 render their prompt entirely in the native XTML
    /// encoder (no Jinja chat template). See [`crate::encoders::kimi_k3_xtml`].
    KimiXtml(KimiXtml),
}

impl Renderer {
    /// The answers of a renderer with its own encoder (see
    /// [`RendererProfile`]); the others answer from the template.
    fn profile(self) -> Option<RendererProfile> {
        match self {
            Self::Jinja | Self::KimiK25Tools => None,
            Self::KimiXtml(model) => Some(model.profile()),
        }
    }
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
    /// The config's chat template(s), named ones included.
    chat_template: ConfigChatTemplates,
}

/// Parse an already-loaded `tokenizer_config.json` value into a `TiktokenConfig`.
fn parse_tiktoken_config(value: &serde_json::Value) -> TiktokenConfig {
    TiktokenConfig {
        special_tokens: parse_special_tokens(value),
        added_tokens: parse_added_tokens_decoder(value),
        skip_token_ids: parse_skip_token_ids(value),
        chat_template: ConfigChatTemplates::read(value.get("chat_template")),
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

/// Tiktoken tokenizer wrapper — supports the built-in OpenAI encodings, hub-loaded
/// tiktoken models and `tekken.json` vocabularies (see `tekken.rs`).
pub struct TiktokenTokenizer {
    /// Shared so a deferred chat encode can own a handle and run on another
    /// thread after the render call returned.
    tokenizer: Arc<CoreBPE>,
    /// The crate's direct byte-level encoder over the file's ranks, when the
    /// file fits it (see [`NativeEncoder::from_ranks`]): every encode of
    /// text goes through it, decode stays with `tokenizer`. `None` keeps
    /// every encode in `tokenizer`. Shared as `tokenizer` is.
    native: Option<Arc<NativeEncoder>>,
    special_tokens: SpecialTokens,
    vocab: HashMap<String, TokenIdType>,
    /// The added tokens' texts by id (`added_tokens_decoder`), read before
    /// the encoder by `id_to_token`: an added token shadows a rank sharing
    /// its id, as `vocab` and the other backend order them. Empty for a
    /// built-in encoding.
    added_texts: HashMap<TokenIdType, String>,
    vocab_size: usize,
    chat_template: ChatTemplateState,
    eos_token_ids: Vec<TokenIdType>,
    skip_token_ids: HashSet<TokenIdType>,
    renderer: Renderer,
    /// What that renderer answers around a render when it has an encoder of
    /// its own ([`Renderer::profile`]), built once with it.
    profile: Option<RendererProfile>,
    /// Bytes per id, for the per-stream decoder (see `byte_level`); `None`
    /// for a vocabulary whose ids lie too far apart for a dense table, which
    /// streams through the generic decoder.
    byte_level: Option<Arc<ByteLevelTable>>,
    /// Encode a text in the pieces the checkpoint's own tokenizer cuts it
    /// into (see [`checkpoint_pieces`]): the Kimi family's `tokenization_kimi.py`.
    checkpoint_pieces: bool,
    /// Render with the keys of every map in the tools sorted, at every
    /// depth, as the checkpoint's own tokenizer class does before its
    /// template or tool encoder reads them: the Kimi family's
    /// `tokenization_kimi.py` sorts them in its `apply_chat_template`
    /// (Kimi K3's class, which renders its own format, does not).
    tools_key_sorted: bool,
    /// The id `add_special_tokens` puts first: a tekken.json vocabulary's
    /// own encoder prepends BOS on request and frames nothing else; a
    /// tiktoken file's has no frame, so `None`, and the flag adds nothing.
    bos: Option<TokenIdType>,
    /// The matcher over the special spellings of a vocabulary whose own
    /// reader writes a request's control tokens structurally and encodes
    /// every request string as plain text (a `tekken.json` file, see
    /// `tekken.rs`): the templates are compiled with the spellings between
    /// marks (see `encoders::marks`), so the render tells the template's
    /// control tokens from the same spelling typed into a message, which
    /// stays text; the matcher says whether a flat encode of the rendered
    /// text would cut out exactly the template's controls (see
    /// `apply_chat_template_with_encoding`). `None` reads every spelling in
    /// a rendered prompt as its token, as the flat encode does.
    controls: Option<AddedTokens>,
    /// The special tokens between the marks, for the marked templates'
    /// context (`Some` with `controls`).
    special_tokens_marked: Option<SpecialTokens>,
}

/// What a vocabulary file gives the tokenizer before the directory beside it
/// is read: the engines are built over these; the chat template, the EOS ids
/// and the renderer come from the directory (see
/// [`TiktokenTokenizer::from_vocabulary`]).
struct FileVocabulary {
    /// Token bytes to id.
    ranks: FxHashMap<Vec<u8>, Rank>,
    /// The split pattern the ranks were trained under.
    pattern: Cow<'static, str>,
    /// Special token spellings to id: recognized in text, decoded by spelling.
    specials: HashMap<String, TokenIdType>,
    /// The ids `skip_special_tokens` strips.
    skip_token_ids: HashSet<TokenIdType>,
    /// The named special tokens (BOS, EOS, ...).
    special_tokens: SpecialTokens,
    /// The id `add_special_tokens` puts first, when the file's own encoder
    /// has that frame.
    bos: Option<TokenIdType>,
    /// Encode a text in the checkpoint's pieces (see [`checkpoint_pieces`]).
    checkpoint_pieces: bool,
    /// The file's own reader writes a request's control tokens structurally
    /// and encodes every request string as plain text: the chat templates
    /// render under the backend's structural form (see
    /// [`TiktokenTokenizer::controls`]).
    structural_controls: bool,
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
            Arc::new(match model {
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
            });

        let special_tokens = Self::get_special_tokens_for_model(model);
        // Built-in encodings have no tokenizer_config.json; every token the
        // encoding treats as special is one `skip_special_tokens` strips.
        let skip_token_ids = tokenizer
            .special_tokens()
            .into_iter()
            .flat_map(|token| tokenizer.encode_with_special_tokens(token))
            .collect();

        let vocab_size = match model {
            TiktokenModel::O200kBase => 200019,
            TiktokenModel::Cl100kBase => 100256,
            TiktokenModel::P50kBase | TiktokenModel::P50kEdit => 50281,
            TiktokenModel::R50kBase => 50257,
        };
        let byte_level = byte_level_table(&tokenizer, vocab_size, &skip_token_ids);

        Ok(TiktokenTokenizer {
            tokenizer,
            // A built-in encoding keeps its ranks to itself.
            native: None,
            special_tokens,
            vocab: HashMap::new(),
            added_texts: HashMap::new(),
            vocab_size,
            chat_template: ChatTemplateState::empty(),
            eos_token_ids: Vec::new(), // No directory path in from_model
            skip_token_ids,
            renderer: Renderer::Jinja,
            profile: None,
            byte_level,
            checkpoint_pieces: false,
            tools_key_sorted: false,
            bos: None,
            controls: None,
            special_tokens_marked: None,
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
        let ranks = load_tiktoken_bpe(tiktoken_path_str)?;

        // 2. Parse tokenizer_config.json from the same directory
        let dir = tiktoken_path
            .parent()
            .ok_or_else(|| Error::msg("Cannot determine parent directory of tiktoken file"))?;
        let (mut config, tokenizer_config_value) = load_tiktoken_config_from_dir(dir)?;

        // Kimi-K2/K2.5/K2.6 specialize the regex and pre-fill 256 reserved
        // special-token slots starting at `len(mergeable_ranks)`; all other
        // tiktoken models use the cl100k pattern unchanged. Reuse the
        // already-parsed tokenizer_config.json so we don't re-read it.
        let kimi = kimi_k2_tokenizer::matches(tokenizer_config_value.as_ref(), dir);
        let pattern = if kimi {
            kimi_k2_tokenizer::apply_reserved_special_tokens(&mut config.added_tokens, ranks.len());
            kimi_k2_tokenizer::KIMI_K2_PATTERN
        } else {
            CL100K_BASE_PATTERN
        };
        Self::from_vocabulary(
            FileVocabulary {
                ranks,
                pattern: Cow::Borrowed(pattern),
                specials: config.added_tokens,
                skip_token_ids: config.skip_token_ids,
                special_tokens: config.special_tokens,
                // The file's encoder frames nothing: the flag adds nothing.
                bos: None,
                // The same file gives the pattern and the piece rule.
                checkpoint_pieces: kimi,
                structural_controls: false,
            },
            dir,
            chat_template_path,
            config.chat_template,
        )
    }

    /// Create from a `tekken.json` vocabulary file (see `tekken.rs`): the
    /// ranks, the split pattern and the special tokens are the file's, laid
    /// out as its reference reader lays them out. `tokenizer_config.json`,
    /// the chat template and the EOS ids are read beside the file as for a
    /// tiktoken file; the file's own sentence markers stand in where the
    /// configuration names none. `add_special_tokens` prepends the file's
    /// BOS. The chat templates render under the structural form (see
    /// `controls`): a control spelling typed into a request stays text, as
    /// the file's own reader encodes it, while the flat `encode` reads the
    /// spelling as its token, as the checkpoint's conversion does.
    pub fn from_tekken_file(path: &Path) -> Result<Self> {
        Self::from_tekken_file_with_chat_template(path, None)
    }

    /// [`Self::from_tekken_file`] with an optional chat template file path.
    pub fn from_tekken_file_with_chat_template(
        path: &Path,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let file = tekken::read(path)?;
        let dir = path
            .parent()
            .ok_or_else(|| Error::msg("Cannot determine parent directory of the tekken file"))?;
        let (config, _) = load_tiktoken_config_from_dir(dir)?;
        let mut special_tokens = config.special_tokens;
        let named = |spelling: &str| file.special(spelling).map(|_| spelling.to_string());
        special_tokens.bos_token = special_tokens.bos_token.or_else(|| named(tekken::BOS));
        special_tokens.eos_token = special_tokens.eos_token.or_else(|| named(tekken::EOS));
        special_tokens.unk_token = special_tokens.unk_token.or_else(|| named(tekken::UNK));
        special_tokens.pad_token = special_tokens.pad_token.or_else(|| named(tekken::PAD));
        // The file's decode rule: a skip drops every id of the special
        // block, fillers included.
        let skip_token_ids = file.specials.values().copied().collect();
        // The file's own encoder prepends BOS on request.
        let bos = file.special(tekken::BOS);
        Self::from_vocabulary(
            FileVocabulary {
                ranks: file.ranks,
                pattern: Cow::Owned(file.pattern),
                specials: file.specials,
                skip_token_ids,
                special_tokens,
                bos,
                checkpoint_pieces: false,
                // The file's own reader writes control tokens structurally.
                structural_controls: true,
            },
            dir,
            chat_template_path,
            config.chat_template,
        )
    }

    /// The tokenizer over a file's vocabulary: the engines over its ranks, the
    /// text-to-id map and the added tokens' texts, then what the directory
    /// beside it gives (the chat
    /// template from an explicit path, else the configuration's, else a file
    /// beside the vocabulary; the EOS ids; the renderer).
    fn from_vocabulary(
        vocabulary: FileVocabulary,
        dir: &Path,
        chat_template_path: Option<&str>,
        config_chat_template: ConfigChatTemplates,
    ) -> Result<Self> {
        let FileVocabulary {
            ranks,
            pattern,
            specials,
            skip_token_ids,
            special_tokens,
            bos,
            checkpoint_pieces,
            structural_controls,
        } = vocabulary;

        // Build special tokens encoder for CoreBPE (needs FxHashMap)
        let special_tokens_encoder: FxHashMap<String, Rank> =
            specials.iter().map(|(k, &v)| (k.clone(), v)).collect();

        // Calculate true vocab size from max token ID (handles sparse/reserved IDs),
        // build the text-to-id map (borrowing the ranks) and the added tokens' texts
        // by id, then pass the ranks by value to CoreBPE
        let vocab_size = ranks
            .values()
            .copied()
            .chain(special_tokens_encoder.values().copied())
            .max()
            .map(|id| id as usize + 1)
            .unwrap_or(0);
        let vocab = build_vocab_map(&ranks, &specials);
        let added_texts: HashMap<TokenIdType, String> = specials
            .iter()
            .map(|(text, &id)| (id, text.clone()))
            .collect();
        let native = Self::native_encoder(&ranks, &pattern, &specials);
        let tokenizer = Arc::new(CoreBPE::new(ranks, special_tokens_encoder, &pattern)?);
        // Counted and announced once the file's own encoder has loaded too,
        // the point the Hugging Face path counts at as well (before the chat
        // template is read).
        if native.is_some() {
            encode_path::note_load(EncodePath::Native);
            tracing::info!(path = %EncodePath::Native, "tiktoken encode path");
        }

        // Load chat template — propagate errors for explicit paths,
        // silently fall back for auto-discovery
        // A template file stands alone (no `tool_use` beside it).
        let chat_template = if let Some(p) = chat_template_path {
            ConfigChatTemplates {
                default: load_chat_template_from_file(p)?,
                tool_use: None,
            }
        } else if config_chat_template.is_some() {
            config_chat_template
        } else {
            ConfigChatTemplates {
                default: discover_chat_template_in_dir(dir)
                    .and_then(|p| load_chat_template_from_file(&p).ok().flatten()),
                tool_use: None,
            }
        };

        // Load merged EOS token IDs from config.json + generation_config.json,
        // plus the tokenizer's own eos_token (structured-output grammars end on it).
        let eos_token_ids = crate::eos::with_tokenizer_eos(
            crate::eos::load_eos_token_ids(dir),
            special_tokens
                .eos_token
                .as_deref()
                .and_then(|token| vocab.get(token).copied()),
        );

        // Detect which chat-template renderer to use based on config.json::architectures
        let renderer = detect_renderer_from_config(dir);
        let byte_level = byte_level_table(&tokenizer, vocab_size, &skip_token_ids);

        // The templates of a vocabulary whose own reader writes control
        // tokens structurally are compiled marked over the special spellings
        // (see `controls`); a special block the matcher declines keeps the
        // flat reading, as the conversion of such a checkpoint does.
        // The processor's legacy template beside the files, for the requests
        // without tools, read as the conversion of the checkpoint reads it
        // (`factory::processor_chat_template`, one read for both backends).
        let processor_template =
            crate::factory::processor_chat_template(Some(dir), chat_template_path);
        let mut chat_template =
            ChatTemplateState::from_checkpoint(chat_template, processor_template)?;
        let controls = structural_controls
            .then(|| {
                let spellings: Vec<(String, TokenIdType)> = specials
                    .iter()
                    .map(|(spelling, &id)| (spelling.clone(), id))
                    .collect();
                AddedTokens::build(&spellings).unwrap_or_else(|declined| {
                    tracing::debug!(%declined, "no matcher over the special tokens; a control spelling typed into a request is the control");
                    None
                })
            })
            .flatten();
        let special_tokens_marked = match &controls {
            Some(controls) => {
                chat_template.mark_controls(controls)?;
                Some(marks::marked_special_tokens(&special_tokens))
            }
            None => None,
        };

        Ok(TiktokenTokenizer {
            tokenizer,
            native,
            special_tokens,
            vocab,
            added_texts,
            vocab_size,
            chat_template,
            eos_token_ids,
            skip_token_ids,
            profile: renderer.profile(),
            renderer,
            byte_level,
            checkpoint_pieces,
            tools_key_sorted: match renderer {
                Renderer::KimiK25Tools => true,
                Renderer::KimiXtml(_) => false,
                Renderer::Jinja => checkpoint_pieces,
            },
            bos,
            controls,
            special_tokens_marked,
        })
    }

    /// The crate's direct encoder over the file's ranks, when the file fits
    /// it (see [`NativeEncoder::from_ranks`]); the choice is logged once and
    /// counted with the Hugging Face loads ([`encode_path::encode_path_loads`]).
    /// A file it declines keeps its own encoder for every encode.
    fn native_encoder(
        ranks: &FxHashMap<Vec<u8>, Rank>,
        pattern: &str,
        added_tokens: &HashMap<String, TokenIdType>,
    ) -> Option<Arc<NativeEncoder>> {
        let mut specials: Vec<(String, TokenIdType)> = added_tokens
            .iter()
            .map(|(token, &id)| (token.clone(), id))
            .collect();
        specials.sort();
        match NativeEncoder::from_ranks(ranks, pattern, &specials) {
            Ok(native) => Some(Arc::new(native)),
            Err(declined) => {
                tracing::info!(
                    native_declined = %declined,
                    "tiktoken encode path: the file's own encoder"
                );
                None
            }
        }
    }

    /// Whether this vocabulary's encodes take the crate's direct byte-level
    /// encoder ([`EncodePath::Native`]); otherwise the file's own encoder
    /// encodes.
    pub fn uses_native_encoder(&self) -> bool {
        self.native.is_some()
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

    /// `apply_chat_template`, also saying whether the prompt continues the
    /// final message (see `ChatTemplateState::apply_with_continuation`).
    fn apply_with_continuation(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<(String, bool)> {
        let (text, continued_final_message) = self.render(messages, params)?;
        // A marked render (see `controls`) is the flat prompt once its marks
        // are gone.
        let text = if self.chat_template.marked() {
            marks::strip(&text).into_owned()
        } else {
            text
        };
        Ok((text, continued_final_message))
    }

    /// The render of `messages` under `params`, marked when the templates
    /// are (see `controls`), and whether the prompt continues the final
    /// message.
    fn render(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<(String, bool)> {
        // A marked template writes its control tokens between marks, the
        // special tokens it reads included, and reads every request string
        // (messages, tools, documents, template kwargs) with its marks
        // removed, so no request can forge one; else the tokenizer's special
        // tokens stand in when the caller passed none.
        let inputs = marks::TemplateInputs::new(
            messages,
            &params,
            self.special_tokens_marked
                .as_ref()
                .filter(|_| self.chat_template.marked()),
            &self.special_tokens,
        );
        let (messages, params) = inputs.split(params);
        // The checkpoint's own class sorts the tools' keys before anything
        // reads them: so here, from the kept sorted form of the prepared set
        // when the caller hands one (made once per set), else sorted now.
        let sorted;
        let params = match params.tools.filter(|_| self.tools_key_sorted) {
            Some(tools) => {
                let prepared = match params
                    .prepared_tools
                    .filter(|prepared| prepared.is_of(tools))
                {
                    Some(prepared) => prepared.key_sorted(),
                    None => {
                        sorted = PreparedTools::key_sorted_of(tools);
                        &sorted
                    }
                };
                ChatTemplateParams {
                    tools: Some(prepared.tools()),
                    prepared_tools: Some(prepared),
                    ..params
                }
            }
            None => params,
        };
        match self.renderer {
            Renderer::Jinja => self.chat_template.apply_with_continuation(messages, params),
            Renderer::KimiK25Tools => apply_kimi_k25_tools(&self.chat_template, messages, params),
            // This is the layer the checkpoint's own `apply_chat_template` sits
            // at, so it applies that wrapper's `thinking_effort` default. K3 does
            // not continue a final message (its prefill is the assistant prefix).
            Renderer::KimiXtml(model) => Ok((
                apply_kimi_k3_xtml_with_effort_default(model, messages, &params)?,
                false,
            )),
        }
    }

    /// The prompt of a marked template (see `controls`), encoded by
    /// segments so that a control spelling typed in a message stays text:
    /// as text all the same when the flat encode would cut out exactly the
    /// template's control tokens (the ids are then the same, and the text
    /// takes every path and cache a flat encode takes), else as a deferred
    /// encode of the segments, the text between two controls as plain BPE.
    fn apply_marked_with_encoding(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        let generation_prompt_unless_continued = params.generation_prompt_unless_continued();
        let (mut marked, continued_final_message) = self.render(messages, params)?;
        if let Some(prefix) = assistant_prefix {
            // Request text: a mark a client wrote into it is no mark.
            marked.push_str(&marks::strip(prefix));
        }
        let (text, controls) = marks::flatten(&marked)?;
        let flat_encode_is_segment_encode = match &self.controls {
            Some(matcher) => matcher.ranges(&text).eq(controls.iter().cloned()),
            None => controls.is_empty(),
        };
        let encoding = if flat_encode_is_segment_encode {
            PromptEncoding::FromText
        } else {
            let segments = marks::segments(&marked)?;
            let bpe = Arc::clone(&self.tokenizer);
            let native = self.native.clone();
            let checkpoint_pieces = self.checkpoint_pieces;
            PromptEncoding::Deferred(EncodeJob::new(move || {
                encode_segments(&bpe, native.as_deref(), &segments, checkpoint_pieces)
                    .map(Encoding::Tiktoken)
            }))
        };
        Ok(ChatTemplateOutput {
            text,
            encoding,
            unbilled_prompt_tokens: 0,
            continued_final_message,
            generation_prompt: generation_prompt_unless_continued && !continued_final_message,
        })
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
/// The text tokens by their text (a rank whose bytes are not UTF-8 has no
/// entry), the added tokens with them. The other direction, an id's text,
/// is read from the file's own encoder (`id_to_token`), which holds every
/// rank's bytes and the special tokens already.
fn build_vocab_map(
    encoder: &FxHashMap<Vec<u8>, Rank>,
    added_tokens: &HashMap<String, TokenIdType>,
) -> HashMap<String, TokenIdType> {
    let mut vocab = HashMap::with_capacity(encoder.len() + added_tokens.len());
    for (token_bytes, &rank) in encoder {
        if let Ok(token_str) = std::str::from_utf8(token_bytes) {
            vocab.insert(token_str.to_string(), rank);
        }
    }
    for (token_str, &id) in added_tokens {
        vocab.insert(token_str.clone(), id);
    }
    vocab
}

/// The largest id range a dense per-id byte table is built for. Every real
/// vocabulary ends far below it (o200k at 200,019, Kimi K3 at its 163,840
/// ranks plus the specials); a `tokenizer_config.json` whose added-token ids
/// lie far beyond the ranks would otherwise cost about nine bytes and one
/// BPE lookup per id up to the largest one, so such a vocabulary keeps the
/// generic per-stream decoder instead. The direct encoder's table of rank
/// ids over such a file is bounded the same way.
pub(crate) const MAX_BYTE_LEVEL_TABLE_IDS: usize = 1 << 22;

/// The bytes every id decodes to, read back from the BPE: a rank's byte
/// string, a special token's text. Every id up to the largest rank or special
/// id gets an entry; `skip_token_ids` marks the ones `skip_special_tokens`
/// strips, as `decode` strips them. `None` when the ids reach past
/// [`MAX_BYTE_LEVEL_TABLE_IDS`].
fn byte_level_table(
    bpe: &CoreBPE,
    vocab_size: usize,
    skip_token_ids: &HashSet<TokenIdType>,
) -> Option<Arc<ByteLevelTable>> {
    let ids_end = skip_token_ids
        .iter()
        .map(|&id| id as usize + 1)
        .max()
        .unwrap_or(0)
        .max(vocab_size);
    if ids_end > MAX_BYTE_LEVEL_TABLE_IDS {
        tracing::debug!(
            ids_end,
            "tiktoken ids too sparse for a per-id byte table; streams use the generic decoder"
        );
        return None;
    }
    let mut table = ByteLevelTableBuilder::with_capacity(ids_end);
    for id in 0..ids_end {
        let Ok(id) = TokenIdType::try_from(id) else {
            break;
        };
        match bpe.decode_bytes(&[id]) {
            Ok(bytes) => table.push(&bytes, skip_token_ids.contains(&id)),
            Err(_) => table.push_absent(),
        }
    }
    Some(table.finish())
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

/// The Kimi family's `tokenization_kimi.py` (`_encode_text_piece`) hands
/// tiktoken at most this many characters at a time ...
const CHECKPOINT_WINDOW_CHARS: usize = 400_000;
/// ... and, inside a window, cuts a run of more than this many consecutive
/// whitespace or consecutive non-whitespace characters
/// (`_split_whitespaces_or_nonwhitespaces`).
const CHECKPOINT_RUN_CHARS: usize = 25_000;

/// The pieces the Kimi checkpoint tokenizer encodes one by one and
/// concatenates: windows of [`CHECKPOINT_WINDOW_CHARS`] characters, each cut
/// again where a run of whitespace or of non-whitespace continues past its
/// [`CHECKPOINT_RUN_CHARS`]-th character (that character starts the next
/// piece; a change of kind restarts the count). A BPE merge never crosses a
/// cut, so the ids of a text longer than a window, or holding a longer run,
/// are not the ids of the whole text, and the engine's ids are the pieces'.
/// Any other text is one piece.
fn checkpoint_pieces(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut window_start = 0;
    let mut window_chars = 0;
    for (at, _) in text.char_indices() {
        if window_chars == CHECKPOINT_WINDOW_CHARS {
            cut_runs(&text[window_start..at], &mut pieces);
            window_start = at;
            window_chars = 0;
        }
        window_chars += 1;
    }
    if !text.is_empty() {
        cut_runs(&text[window_start..], &mut pieces);
    }
    pieces
}

/// One window of [`checkpoint_pieces`], cut at its long runs.
fn cut_runs<'a>(window: &'a str, pieces: &mut Vec<&'a str>) {
    let mut run_is_space = window.chars().next().is_some_and(python_isspace);
    let mut run_chars = 0;
    let mut piece_start = 0;
    for (at, c) in window.char_indices() {
        let is_space = python_isspace(c);
        if run_is_space == is_space {
            run_chars += 1;
            if run_chars > CHECKPOINT_RUN_CHARS {
                pieces.push(&window[piece_start..at]);
                piece_start = at;
                run_chars = 1;
            }
        } else {
            run_is_space = is_space;
            run_chars = 1;
        }
    }
    pieces.push(&window[piece_start..]);
}

/// Piecewise encode for a segmented prompt: control pieces with special
/// tokens recognized, text pieces as ordinary BPE, each segment in the
/// checkpoint's pieces when `checkpoint_pieces` is set. The allowed-special
/// set is built once per prompt; `encode_with_special_tokens` would rebuild
/// it for every piece, and a Kimi prompt has hundreds of control pieces.
/// The ids are the segments' own encodes, one after another: a prompt's ids
/// are the ids of any split of its segments, encoded apart and joined.
fn encode_segments(
    bpe: &CoreBPE,
    native: Option<&NativeEncoder>,
    segments: &[PromptSegment],
    checkpoint_pieces: bool,
) -> Result<Vec<TokenIdType>> {
    let mut ids = Vec::with_capacity(ids_capacity(segments));
    encode_segments_into(
        bpe,
        native,
        &bpe.special_tokens(),
        segments,
        checkpoint_pieces,
        &mut ids,
    )?;
    Ok(ids)
}

/// Room for the ids of `segments` before any is encoded: one id per three
/// bytes of text, the direct encoder's rule for prose and code. A prompt
/// denser than that grows the buffer once; nothing grows it per segment.
fn ids_capacity(segments: &[PromptSegment]) -> usize {
    segments
        .iter()
        .map(|segment| segment.text.len())
        .sum::<usize>()
        / 3
        + 8
}

/// [`encode_segments`], appending to `ids`, with the allowed-special set the
/// caller built once for the prompt (`bpe.special_tokens()`): a caller that
/// holds some of the segments' ids already writes the rest around them into
/// one buffer.
fn encode_segments_into(
    bpe: &CoreBPE,
    native: Option<&NativeEncoder>,
    allowed: &HashSet<&str>,
    segments: &[PromptSegment],
    checkpoint_pieces: bool,
    ids: &mut Vec<TokenIdType>,
) -> Result<()> {
    for segment in segments {
        // The checkpoint cuts at a window of characters and at a run of them, both longer than
        // this bound in bytes: a shorter segment is one piece, encoded without the cut's walk
        // over its characters and the list of its pieces.
        if checkpoint_pieces && segment.text.len() > CHECKPOINT_RUN_CHARS {
            for piece in self::checkpoint_pieces(&segment.text) {
                encode_piece(bpe, native, allowed, piece, segment.allow_special, ids)?;
            }
        } else {
            encode_piece(
                bpe,
                native,
                allowed,
                &segment.text,
                segment.allow_special,
                ids,
            )?;
        }
    }
    Ok(())
}

/// One piece's ids, appended: a control piece with the special tokens of
/// `allowed` recognized, a text piece as ordinary BPE; on the direct
/// encoder when the vocabulary has one, else through the file's.
#[inline(always)]
fn encode_piece(
    bpe: &CoreBPE,
    native: Option<&NativeEncoder>,
    allowed: &HashSet<&str>,
    piece: &str,
    allow_special: bool,
    ids: &mut Vec<TokenIdType>,
) -> Result<()> {
    if let Some(native) = native {
        let encoded = if allow_special {
            native.encode(None, piece, false)
        } else {
            // The start flag is the sentencepiece front's; no front on this path.
            native.encode_plain(None, piece, false)
        };
        match encoded {
            Ok(ids_of_piece) => {
                ids.extend(ids_of_piece);
                return Ok(());
            }
            Err(fallback) => encode_path::note_fallback(fallback),
        }
    }
    if allow_special {
        let (ids_of_piece, _) = bpe
            .encode(piece, allowed)
            .map_err(|e| Error::msg(format!("tiktoken encode failed: {e}")))?;
        ids.extend(ids_of_piece);
    } else {
        ids.extend(bpe.encode_ordinary(piece));
    }
    Ok(())
}

/// The ids of one piece with the special tokens recognized: on the direct
/// encoder when the vocabulary has one, else through the file's.
fn encode_with_special_tokens(
    bpe: &CoreBPE,
    native: Option<&NativeEncoder>,
    piece: &str,
) -> Vec<TokenIdType> {
    if let Some(native) = native {
        match native.encode(None, piece, false) {
            Ok(ids) => return ids,
            Err(fallback) => encode_path::note_fallback(fallback),
        }
    }
    bpe.encode_with_special_tokens(piece)
}

impl Encoder for TiktokenTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        // The special tokens' spellings are recognized whatever the flag asks:
        // a rendered chat prompt carries its control tokens as text
        // (`<|media_pad|>`, `[INST]`), so they stay atomic instead of
        // splitting into BPE sub-tokens. The flag adds the frame the
        // vocabulary's own encoder adds: BOS for a tekken.json file, nothing
        // for a tiktoken file (no concept of one).
        let native = self.native.as_deref();
        let mut tokens: Vec<TokenIdType> = if self.checkpoint_pieces {
            checkpoint_pieces(input)
                .into_iter()
                .flat_map(|piece| encode_with_special_tokens(&self.tokenizer, native, piece))
                .collect()
        } else {
            encode_with_special_tokens(&self.tokenizer, native, input)
        };
        if let Some(bos) = self.bos.filter(|_| add_special_tokens) {
            tokens.insert(0, bos);
        }
        Ok(Encoding::Tiktoken(tokens))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl TiktokenTokenizer {
    /// `token_ids` without the special tokens a skip leaves out: the ids
    /// `decode` renders.
    fn kept_ids<'a>(
        &self,
        token_ids: &'a [TokenIdType],
        skip_special_tokens: bool,
    ) -> Cow<'a, [TokenIdType]> {
        if skip_special_tokens && !self.skip_token_ids.is_empty() {
            Cow::Owned(
                token_ids
                    .iter()
                    .copied()
                    .filter(|id| !self.skip_token_ids.contains(id))
                    .collect(),
            )
        } else {
            Cow::Borrowed(token_ids)
        }
    }

    /// The bytes known ids stand for, in order: what `decode` renders before
    /// the UTF-8 validation.
    fn bytes_of(&self, token_ids: &[TokenIdType]) -> Vec<u8> {
        self.tokenizer
            ._decode_native_and_split(token_ids.to_vec())
            .flatten()
            .collect()
    }
}

impl Decoder for TiktokenTokenizer {
    /// A vocabulary with the per-id table decodes through it, as its streams
    /// do: every id's bytes concatenated, specials dropped by the table's
    /// per-id marks when asked, one UTF-8 conversion, in place of the crate's
    /// lookup per id; bytes that are not UTF-8 render lossily and are logged
    /// as the crate path logs its fallback. An id the table lacks, or a
    /// vocabulary too sparse for one, takes the crate's decode below, with
    /// its unknown-id error and its lossy fallback.
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        if let Some((text, lossy)) = self
            .byte_level
            .as_ref()
            .and_then(|table| table.decode_known(token_ids, skip_special_tokens))
        {
            if lossy {
                tracing::warn!(
                    token_count = token_ids.len(),
                    "tiktoken decode failed; returning lossy UTF-8 fallback"
                );
            }
            return Ok(text);
        }
        let token_ids = self.kept_ids(token_ids, skip_special_tokens);
        match self.tokenizer.decode(&token_ids) {
            Ok(text) => Ok(text),
            Err(err) if is_unknown_tiktoken_decode_error(&err) => Err(Error::msg(format!(
                "tiktoken decode failed for unknown token id: {err}"
            ))),
            Err(err) => {
                // Fallback to lossy decoding for incomplete UTF-8 sequences
                let bytes = self.bytes_of(&token_ids);
                tracing::warn!(
                    error = %err,
                    token_count = token_ids.len(),
                    "tiktoken decode failed; returning lossy UTF-8 fallback"
                );
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
        }
    }

    /// Every id stands for fixed bytes, so one stream needs only a
    /// pending-bytes buffer: no window re-decoded per token, and a character
    /// that merely spans tokens is held back, not a decode failure. An id the
    /// vocabulary lacks fails the stream, as `decode` fails on it. A
    /// vocabulary without the table (see `MAX_BYTE_LEVEL_TABLE_IDS`) streams
    /// through the generic decoder.
    fn incremental_decoder(
        &self,
        skip_special_tokens: bool,
    ) -> Option<Box<dyn IncrementalDecoder>> {
        let table = self.byte_level.as_ref()?;
        Some(Box::new(ByteLevelIncremental::new(
            Arc::clone(table),
            skip_special_tokens,
            UnknownId::Reject,
        )))
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

impl TiktokenTokenizer {
    /// The template questions, answered by the profile or by the template.
    fn template_answers(&self) -> TemplateAnswers<'_> {
        TemplateAnswers {
            profile: self.profile.as_ref(),
            template: &self.chat_template,
        }
    }
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
        // An added token first (it shadows a rank sharing its id, as the
        // text-to-id map ordered them), then the file's encoder, which holds
        // every rank's bytes and the special tokens; an id whose bytes are
        // not text has no token string, as before.
        if let Some(text) = self.added_texts.get(&id) {
            return Some(text.clone());
        }
        self.tokenizer
            .decode_bytes(&[id])
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        self.apply_with_continuation(messages, params)
            .map(|(text, _)| text)
    }

    fn apply_chat_template_with_encoding(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        let Renderer::KimiXtml(model) = self.renderer else {
            // A marked template (see `controls`): the prompt is encoded by
            // segments, so that a control spelling typed in a message stays
            // text.
            if self.chat_template.marked() {
                return self.apply_marked_with_encoding(messages, params, assistant_prefix);
            }
            let generation_prompt_unless_continued = params.generation_prompt_unless_continued();
            let (mut text, continued_final_message) =
                self.apply_with_continuation(messages, params)?;
            if let Some(prefix) = assistant_prefix {
                text.push_str(prefix);
            }
            return Ok(ChatTemplateOutput {
                text,
                encoding: PromptEncoding::FromText,
                unbilled_prompt_tokens: 0,
                continued_final_message,
                generation_prompt: generation_prompt_unless_continued && !continued_final_message,
            });
        };
        // K3's ids are not a function of the flat text: render the reference's
        // segments and hand the caller the piecewise encode to run where it
        // would have encoded the text. The renderer reads none of
        // `params.special_tokens`, so the injection `apply_chat_template`
        // performs is a no-op here.
        let RenderedXtml { segments, pending } =
            render_kimi_k3_xtml_prompt(model, messages, &params, assistant_prefix)?;
        let text = join_segments(&segments);
        // Piecewise encoding makes the stub's own count its share of the full
        // encode: its ids, encoded once here for the count, are kept for the
        // job, which writes the segments before and after them around them.
        let checkpoint_pieces = self.checkpoint_pieces;
        let pending_ids = encode_segments(
            &self.tokenizer,
            self.native.as_deref(),
            &segments[pending.clone()],
            checkpoint_pieces,
        )?;
        let unbilled_prompt_tokens = pending_ids.len() as u32;
        let bpe = Arc::clone(&self.tokenizer);
        let native = self.native.clone();
        let job = EncodeJob::new(move || {
            let native = native.as_deref();
            let allowed = bpe.special_tokens();
            let mut ids = Vec::with_capacity(ids_capacity(&segments));
            let (before, after) = (&segments[..pending.start], &segments[pending.end..]);
            encode_segments_into(&bpe, native, &allowed, before, checkpoint_pieces, &mut ids)?;
            ids.extend_from_slice(&pending_ids);
            encode_segments_into(&bpe, native, &allowed, after, checkpoint_pieces, &mut ids)?;
            Ok(Encoding::Tiktoken(ids))
        });
        Ok(ChatTemplateOutput {
            text,
            encoding: PromptEncoding::Deferred(job),
            unbilled_prompt_tokens,
            continued_final_message: false,
            // K3 writes the generation header under the flag and never
            // continues a message (its prefill is the assistant prefix).
            generation_prompt: params.add_generation_prompt,
        })
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

    fn eos_token_ids(&self) -> &[TokenIdType] {
        &self.eos_token_ids
    }

    fn think_in_prefill(&self) -> bool {
        self.template_answers().think_in_prefill()
    }

    fn prefill_opens_think_block(&self) -> bool {
        self.template_answers().prefill_opens_think_block()
    }

    /// The profile's, or the Jinja template's; this tokenizer resolves no
    /// media handling of its own.
    fn renderer_capabilities(&self) -> RendererCapabilities {
        self.template_answers()
            .renderer_capabilities(RendererProfile::TEMPLATE_CAPABILITIES.media_handling)
    }

    fn native_thinking_mode(
        &self,
        template_kwargs: Option<&HashMap<String, serde_json::Value>>,
        thinking: Option<bool>,
    ) -> Option<bool> {
        self.template_answers()
            .native_thinking_mode(template_kwargs, thinking)
    }

    fn set_chat_template(&mut self, template: String) -> Result<()> {
        // Replaced and marked as one step: a template that fails to mark
        // leaves the previous one in force (see `replace_marked`).
        self.chat_template
            .replace_marked(template, self.controls.as_ref())
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
        return Renderer::KimiXtml(KimiXtml::K3);
    }
    if arch_strs.contains(&"KimiK31ForConditionalGeneration") {
        tracing::debug!(?path, "selected KimiK31Xtml chat-template renderer");
        return Renderer::KimiXtml(KimiXtml::K31);
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
            Renderer::KimiXtml(KimiXtml::K3)
        }
        Some("kimi_k31") => {
            tracing::debug!(?path, "selected KimiK31Xtml renderer via model_type");
            Renderer::KimiXtml(KimiXtml::K31)
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
    use std::sync::Mutex;

    use super::*;
    use crate::{
        sequence::Sequence,
        traits::{Decoder, Encoder, Tokenizer},
    };

    const MINIMAL_TIKTOKEN_MODEL: &str = "YQ== 0\nYg== 1\n";

    /// Counts the WARN events of the thread it is the default subscriber on.
    struct WarnCount(Arc<Mutex<usize>>);

    impl tracing::Subscriber for WarnCount {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() == tracing::Level::WARN {
                *self.0.lock().unwrap() += 1;
            }
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// Run `f` and count the warnings it logs.
    fn warnings_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
        let count = Arc::new(Mutex::new(0));
        let guard = tracing::subscriber::set_default(WarnCount(Arc::clone(&count)));
        let out = f();
        drop(guard);
        let warnings = *count.lock().unwrap();
        (out, warnings)
    }

    /// The pieces the gateway's per-stream decoder emits for `ids`, one per id.
    fn stream(tokenizer: &Arc<dyn Tokenizer>, ids: &[TokenIdType], skip: bool) -> Vec<String> {
        let mut sequence = Sequence::new_with_options(Arc::clone(tokenizer), skip);
        ids.iter()
            .map(|&id| sequence.append_token(id).unwrap())
            .collect()
    }

    /// The pieces the gateway's per-stream decoder emits for `ids`, then what
    /// its finish releases.
    fn stream_then_finish(
        tokenizer: &Arc<dyn Tokenizer>,
        ids: &[TokenIdType],
        skip: bool,
    ) -> (Vec<String>, String) {
        let mut sequence = Sequence::new_with_options(Arc::clone(tokenizer), skip);
        let pieces = ids
            .iter()
            .map(|&id| sequence.append_token(id).unwrap())
            .collect();
        (pieces, sequence.finish().unwrap())
    }

    /// Ranks `a`, `b`, the lead byte C3, the continuation byte A9, the three
    /// bytes of a literal U+FFFD and a space with the lead byte C3 after it:
    /// a character can end a stream unfinished, alone or behind text.
    const PARTIAL_BYTE_TIKTOKEN_MODEL: &str = "YQ== 0\nYg== 1\nww== 2\nqQ== 3\n77+9 4\nIMM= 5\n";

    /// The stop decoder's reading of `ids`: as a stream (the deltas joined,
    /// the finish last) and as one buffer (every token, then the finish,
    /// appended into it: the non-streaming path).
    fn stop_decoded(
        tokenizer: &Arc<dyn Tokenizer>,
        ids: &[TokenIdType],
        skip: bool,
    ) -> (String, String) {
        use crate::stop::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
        let mut decoder =
            StopSequenceDecoder::new(Arc::clone(tokenizer), StopSequenceConfig::default(), skip);
        let mut streamed = String::new();
        for &id in ids {
            if let SequenceDecoderOutput::Text(text) = decoder.process_token(id).unwrap() {
                streamed.push_str(&text);
            }
        }
        if let SequenceDecoderOutput::Text(text) = decoder.finish().unwrap() {
            streamed.push_str(&text);
        }
        let mut decoder =
            StopSequenceDecoder::new(Arc::clone(tokenizer), StopSequenceConfig::default(), skip);
        let mut whole = String::new();
        for &id in ids {
            decoder.process_token_into(id, &mut whole).unwrap();
        }
        decoder.finish_into(&mut whole).unwrap();
        (streamed, whole)
    }

    fn write_minimal_tiktoken_dir(
        tokenizer_config: &str,
        model_config: Option<&str>,
    ) -> tempfile::TempDir {
        write_tiktoken_dir(MINIMAL_TIKTOKEN_MODEL, tokenizer_config, model_config)
    }

    fn write_tiktoken_dir(
        model: &str,
        tokenizer_config: &str,
        model_config: Option<&str>,
    ) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), model).unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), tokenizer_config).unwrap();
        if let Some(model_config) = model_config {
            std::fs::write(dir.path().join("config.json"), model_config).unwrap();
        }
        dir
    }

    /// The tiktoken loader reads a configuration's named templates the same
    /// way: `default` for a plain request, `tool_use` for one that declares
    /// tools; a template file stands alone for both.
    #[test]
    fn the_loader_reads_a_named_template_list_and_a_template_file_stands_alone() {
        let dir = write_minimal_tiktoken_dir(
            r#"{"chat_template": [{"name": "default", "template": "d"}, {"name": "tool_use", "template": "t"}]}"#,
            None,
        );
        let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
        let tools: [serde_json::Value; 0] = [];
        let with_tools = || ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };

        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert_eq!(
            tokenizer
                .apply_chat_template(&messages, ChatTemplateParams::default())
                .unwrap(),
            "d"
        );
        assert_eq!(
            tokenizer
                .apply_chat_template(&messages, with_tools())
                .unwrap(),
            "t"
        );

        let file = dir.path().join("own.jinja");
        std::fs::write(&file, "f").unwrap();
        let tokenizer =
            TiktokenTokenizer::from_dir_with_chat_template(dir.path(), file.to_str()).unwrap();
        assert_eq!(
            tokenizer
                .apply_chat_template(&messages, ChatTemplateParams::default())
                .unwrap(),
            "f"
        );
        assert_eq!(
            tokenizer
                .apply_chat_template(&messages, with_tools())
                .unwrap(),
            "f"
        );
    }

    /// The content format a request's messages are shaped into follows the
    /// template the request renders through: the `tool_use` template's for
    /// a request that declares tools, the `default` one's otherwise, as the
    /// Hugging Face tokenizer answers it; the plain question keeps answering
    /// for the default template.
    #[test]
    fn the_content_format_of_a_tools_request_is_the_tool_use_template_s() {
        let dir = write_minimal_tiktoken_dir(
            r#"{"chat_template": [
                {"name": "default", "template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"},
                {"name": "tool_use", "template": "{% for message in messages %}{% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endfor %}"}]}"#,
            None,
        );
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert_eq!(
            tokenizer.chat_template_content_format(),
            ChatTemplateContentFormat::String
        );
        assert_eq!(
            tokenizer.chat_template_content_format_for_request(false),
            ChatTemplateContentFormat::String
        );
        assert_eq!(
            tokenizer.chat_template_content_format_for_request(true),
            ChatTemplateContentFormat::OpenAI
        );
    }

    #[test]
    fn test_detect_renderer_from_config() {
        let k3 = write_minimal_tiktoken_dir(
            "{}",
            Some(r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#),
        );
        assert!(matches!(
            detect_renderer_from_config(k3.path()),
            Renderer::KimiXtml(KimiXtml::K3)
        ));

        let k31 = write_minimal_tiktoken_dir(
            "{}",
            Some(r#"{"architectures": ["KimiK31ForConditionalGeneration"]}"#),
        );
        assert!(matches!(
            detect_renderer_from_config(k31.path()),
            Renderer::KimiXtml(KimiXtml::K31)
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
            Renderer::KimiXtml(KimiXtml::K3)
        ));

        let k31_model_type =
            write_minimal_tiktoken_dir("{}", Some(r#"{"model_type": "kimi_k31"}"#));
        assert!(matches!(
            detect_renderer_from_config(k31_model_type.path()),
            Renderer::KimiXtml(KimiXtml::K31)
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

    /// A built-in encoding keeps no text map, so `token_to_id` answers
    /// nothing there; `id_to_token` reads the encoder and answers anyway.
    #[test]
    fn a_built_in_encoding_answers_an_id_from_its_encoder_and_no_text() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        assert_eq!(tokenizer.token_to_id("hello"), None);
        assert_eq!(tokenizer.id_to_token(0).as_deref(), Some("!"));
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
    fn test_build_vocab_map() {
        let mut encoder = FxHashMap::default();
        encoder.insert(b"hello".to_vec(), 42u32);
        encoder.insert(vec![0xFF, 0xFE], 99u32); // invalid UTF-8

        let mut added = HashMap::new();
        added.insert("<|special|>".to_string(), 1000u32);

        let vocab = build_vocab_map(&encoder, &added);

        // Valid UTF-8 token present
        assert_eq!(vocab.get("hello"), Some(&42));

        // Invalid UTF-8 token excluded from vocab
        assert!(!vocab.contains_key("\u{FFFD}")); // not lossy-inserted

        // Added token present
        assert_eq!(vocab.get("<|special|>"), Some(&1000));
    }

    /// `id_to_token` answers what the text map used to hold: a rank whose
    /// bytes are text, an added token's text, and nothing for a rank whose
    /// bytes are not text or an id the file does not have.
    #[test]
    fn an_id_reads_its_text_from_the_encoder_as_the_map_held_it() {
        let dir = split_vocabulary_dir();
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        for byte in 0u32..256 {
            let expected = std::str::from_utf8(&[byte as u8]).ok().map(str::to_owned);
            assert_eq!(tokenizer.id_to_token(byte), expected, "rank {byte}");
        }
        assert_eq!(
            tokenizer.id_to_token(256),
            None,
            "a rank whose bytes are not text"
        );
        assert_eq!(tokenizer.id_to_token(300).as_deref(), Some("[EOS]"));
        assert_eq!(tokenizer.id_to_token(301).as_deref(), Some("<|open|>"));
        assert_eq!(
            tokenizer.id_to_token(400),
            None,
            "an id the file does not have"
        );
    }

    /// An id that is both a rank and an added token answers the added token,
    /// as the text map did (it inserted the added tokens last) and as the
    /// other backend orders them; the text-to-id map keeps both texts.
    #[test]
    fn an_added_token_shadows_a_rank_sharing_its_id() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tiktoken.model"),
            full_byte_tiktoken_model(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {
                "65": {"content": "<|shadow|>", "special": false}
            }}"#,
        )
        .unwrap();
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert_eq!(tokenizer.id_to_token(65).as_deref(), Some("<|shadow|>"));
        assert_eq!(tokenizer.id_to_token(66).as_deref(), Some("B"));
        assert_eq!(tokenizer.token_to_id("<|shadow|>"), Some(65));
        assert_eq!(tokenizer.token_to_id("A"), Some(65));
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
    fn test_builtin_encoding_skips_its_special_tokens_on_decode() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        let ids = tokenizer.encode("hello<|endoftext|>world", false).unwrap();
        assert!(ids.token_ids().contains(&100257));
        assert_eq!(
            tokenizer.decode(ids.token_ids(), false).unwrap(),
            "hello<|endoftext|>world"
        );
        assert_eq!(
            tokenizer.decode(ids.token_ids(), true).unwrap(),
            "helloworld"
        );
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

    /// The byte vocabulary plus two merges that split characters across
    /// tokens, the first two bytes of '中' (E4 B8) and of '😀' (F0 9F), a
    /// special `[EOS]` and a kept `<|open|>`.
    fn split_vocabulary_dir() -> tempfile::TempDir {
        let mut model = full_byte_tiktoken_model();
        model.push_str(&format!("{} 256\n", STANDARD.encode([0xE4, 0xB8])));
        model.push_str(&format!("{} 257\n", STANDARD.encode([0xF0, 0x9F])));
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), model).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {
                "300": {"content": "[EOS]", "special": true},
                "301": {"content": "<|open|>", "special": false}
            }}"#,
        )
        .unwrap();
        dir
    }

    /// A tiktoken-file checkpoint reads the processor's legacy
    /// `chat_template.json` beside it as the conversion of a checkpoint
    /// does: a request without tools renders through the processor's
    /// template, a request with tools through the configuration's.
    #[test]
    fn a_tiktoken_file_checkpoint_reads_the_processor_template_for_tool_less_requests() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tiktoken.model"),
            full_byte_tiktoken_model(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "config"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("chat_template.json"),
            r#"{"chat_template": "processor"}"#,
        )
        .unwrap();
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        let messages = [serde_json::json!({"role": "user", "content": "hello"})];
        let tools =
            [serde_json::json!({"type": "function", "function": {"name": "f", "parameters": {}}})];
        let without_tools = tokenizer
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert_eq!(without_tools, "processor");
        let with_tools = ChatTemplateParams {
            tools: Some(tools.as_slice()),
            ..ChatTemplateParams::default()
        };
        assert_eq!(
            tokenizer
                .apply_chat_template(&messages, with_tools)
                .unwrap(),
            "config"
        );
    }

    #[test]
    fn test_incremental_decode_holds_a_split_character_without_a_lossy_fallback() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::from_dir(split_vocabulary_dir().path()).unwrap());
        let ids = tokenizer
            .encode("a中😀b", false)
            .unwrap()
            .token_ids()
            .to_vec();
        // '中' over two tokens, '😀' over three.
        assert_eq!(ids, [0x61, 256, 0xAD, 257, 0x98, 0x80, 0x62]);

        let (pieces, warnings) = warnings_during(|| stream(&tokenizer, &ids, false));
        assert_eq!(pieces, ["a", "", "中", "", "", "😀", "b"]);
        assert_eq!(pieces.concat(), tokenizer.decode(&ids, false).unwrap());
        assert_eq!(
            warnings, 0,
            "a character spanning tokens is not a decode failure"
        );
    }

    #[test]
    fn test_incremental_decode_reports_invalid_bytes_once_per_stream() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::from_dir(split_vocabulary_dir().path()).unwrap());
        // A lone continuation byte, text, a lone lead byte, text.
        let ids = [0xAD, 0x61, 0xE4, 0x62];
        let (pieces, warnings) = warnings_during(|| stream(&tokenizer, &ids, false));
        assert_eq!(pieces, ["", "\u{FFFD}a", "", "\u{FFFD}b"]);
        assert_eq!(pieces.concat(), tokenizer.decode(&ids, false).unwrap());
        assert_eq!(warnings, 1, "invalid bytes are reported once per stream");
    }

    #[test]
    fn test_incremental_decode_follows_the_skip_flag() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::from_dir(split_vocabulary_dir().path()).unwrap());
        let ids = [0x61, 300, 301, 0x62];
        assert_eq!(
            stream(&tokenizer, &ids, false),
            ["a", "[EOS]", "<|open|>", "b"]
        );
        assert_eq!(stream(&tokenizer, &ids, true), ["a", "", "<|open|>", "b"]);
    }

    #[test]
    fn test_incremental_decode_rejects_unknown_ids_like_decode() {
        let dir = write_minimal_tiktoken_dir(
            r#"{"added_tokens_decoder": {"2": {"content": "[BOS]", "special": true}}}"#,
            None,
        );
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::from_dir(dir.path()).unwrap());
        let mut sequence = Sequence::new_with_options(tokenizer, false);
        assert_eq!(sequence.append_token(0).unwrap(), "a");
        assert_eq!(sequence.append_token(2).unwrap(), "[BOS]");
        let err = sequence.append_token(4).unwrap_err();
        assert!(err.to_string().contains("unknown token id"), "{err}");
    }

    /// The streams of `PARTIAL_BYTE_TIKTOKEN_MODEL` (with `[EOS]` as id 6) that
    /// end while the decoder holds bytes: (ids, skip special tokens, the text
    /// the full decode and the stream both read). A lone continuation byte, a
    /// literal replacement character, a byte before a skipped special token
    /// and a lead byte the end cut (C3 alone, after `a`, before a skipped
    /// special token, or behind the space of its own rank) are released as
    /// the full decode renders them.
    const STREAM_END_CASES: [(&[TokenIdType], bool, &str); 9] = [
        (&[2], false, "\u{FFFD}"),
        (&[0, 2], false, "a\u{FFFD}"),
        (&[2, 6], true, "\u{FFFD}"),
        (&[5], false, " \u{FFFD}"),
        (&[0, 1, 3], false, "ab\u{FFFD}"),
        (&[4], false, "\u{FFFD}"),
        (&[2, 3], false, "é"),
        (&[3, 6], true, "\u{FFFD}"),
        (&[3, 6], false, "\u{FFFD}[EOS]"),
    ];

    /// `STREAM_END_CASES` on `tokenizer`, through `Sequence`, through the
    /// stop decoder as a stream and as one buffer (the non-streaming path).
    fn assert_stream_end_cases(tokenizer: &Arc<dyn Tokenizer>) {
        for (ids, skip, text) in STREAM_END_CASES {
            assert_eq!(tokenizer.decode(ids, skip).unwrap(), text, "ids={ids:?}");
            let (pieces, finish) = stream_then_finish(tokenizer, ids, skip);
            assert_eq!(
                pieces.concat() + &finish,
                text,
                "ids={ids:?} skip={skip}: pieces {pieces:?}, finish {finish:?}"
            );
            let (stop_streamed, whole) = stop_decoded(tokenizer, ids, skip);
            assert_eq!(stop_streamed, text, "stop decoder: ids={ids:?} skip={skip}");
            assert_eq!(whole, text, "whole: ids={ids:?} skip={skip}");
        }
    }

    /// A stream that ends while the per-stream decoder holds bytes: its finish
    /// releases the held bytes as the full decode renders them, a character
    /// the end cut as its replacement character, and the pieces before it
    /// hold as they did.
    #[test]
    fn stream_end_renders_held_bytes_as_decode_does() {
        let dir = write_tiktoken_dir(
            PARTIAL_BYTE_TIKTOKEN_MODEL,
            r#"{"added_tokens_decoder": {"6": {"content": "[EOS]", "special": true}}}"#,
            None,
        );
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::from_dir(dir.path()).unwrap());
        assert!(tokenizer.incremental_decoder(false).is_some());
        assert_stream_end_cases(&tokenizer);
        let (pieces, finish) = stream_then_finish(&tokenizer, &[0, 1, 3], false);
        assert_eq!(pieces, ["a", "b", ""]);
        assert_eq!(finish, "\u{FFFD}");
        let (pieces, finish) = stream_then_finish(&tokenizer, &[3, 6], true);
        assert_eq!(pieces, ["", ""]);
        assert_eq!(finish, "\u{FFFD}");
        let (pieces, finish) = stream_then_finish(&tokenizer, &[0, 2], false);
        assert_eq!(pieces, ["a", ""]);
        assert_eq!(finish, "\u{FFFD}");
        let (pieces, finish) = stream_then_finish(&tokenizer, &[5], false);
        assert_eq!(pieces, [""]);
        assert_eq!(finish, " \u{FFFD}");
    }

    /// The same end through the generic decoder (no per-id table because an
    /// added token sits far beyond the ranks): the window's held bytes are
    /// released as the full decode renders them, a cut character included.
    #[test]
    fn stream_end_renders_held_bytes_as_decode_does_on_the_generic_path() {
        let dir = write_tiktoken_dir(
            PARTIAL_BYTE_TIKTOKEN_MODEL,
            r#"{"added_tokens_decoder": {"6": {"content": "[EOS]", "special": true}, "100000000": {"content": "[FAR]", "special": true}}}"#,
            None,
        );
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::from_dir(dir.path()).unwrap());
        assert!(tokenizer.incremental_decoder(false).is_none());
        assert_stream_end_cases(&tokenizer);
        let (pieces, finish) = stream_then_finish(&tokenizer, &[0, 1, 3], false);
        assert_eq!(pieces, ["a", "b", ""]);
        assert_eq!(finish, "\u{FFFD}");
        let (pieces, finish) = stream_then_finish(&tokenizer, &[0, 2], false);
        assert_eq!(pieces, ["a", ""]);
        assert_eq!(finish, "\u{FFFD}");
    }

    #[test]
    fn test_incremental_decode_keeps_the_generic_path_for_distant_ids() {
        // An added token far beyond the ranks: no per-id table is built for
        // it, the stream goes through the generic decoder and is still right.
        let dir = write_minimal_tiktoken_dir(
            r#"{"added_tokens_decoder": {"100000000": {"content": "[FAR]", "special": true}}}"#,
            None,
        );
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::from_dir(dir.path()).unwrap());
        assert!(tokenizer.incremental_decoder(false).is_none());
        assert_eq!(stream(&tokenizer, &[0, 1, 0], false), ["a", "b", "a"]);
        assert_eq!(
            stream(&tokenizer, &[0, 100_000_000, 1], false).concat(),
            tokenizer.decode(&[0, 100_000_000, 1], false).unwrap()
        );
    }

    #[test]
    fn test_incremental_decode_is_exact_on_a_real_vocabulary() {
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap());
        let texts = [
            "Hello, 世界! Café naïve 😀👍🏽 ✨ 日本語のテキスト 한국어 Ünïcödé ½ € –",
            "अनुच्छेद 🇯🇵 🤷\u{200d}♂\u{fe0f} Привет мир<|endoftext|>",
        ];
        for text in texts {
            let ids = tokenizer.encode(text, false).unwrap().token_ids().to_vec();
            for skip in [false, true] {
                let (pieces, warnings) = warnings_during(|| stream(&tokenizer, &ids, skip));
                let joined = pieces.concat();
                assert_eq!(
                    joined,
                    tokenizer.decode(&ids, skip).unwrap(),
                    "{text:?} skip={skip}"
                );
                assert!(!joined.contains('\u{FFFD}'), "{text:?} skip={skip}");
                assert_eq!(warnings, 0, "{text:?} skip={skip}");
            }
        }
        // The emoji is two tokens here: nothing, then the whole character.
        let ids = tokenizer.encode("😀", false).unwrap().token_ids().to_vec();
        assert_eq!(ids.len(), 2);
        assert_eq!(stream(&tokenizer, &ids, false), ["", "😀"]);
    }

    /// A tiktoken directory whose `config.json` selects the K3 renderer and
    /// whose vocabulary is every byte plus the four structural markers.
    fn k3_byte_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tiktoken.model"),
            full_byte_tiktoken_model(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {
                "300": {"content": "<|open|>", "special": false},
                "301": {"content": "<|close|>", "special": false},
                "302": {"content": "<|sep|>", "special": false},
                "303": {"content": "<|end_of_msg|>", "special": true}
            }}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#,
        )
        .unwrap();
        dir
    }

    #[test]
    fn test_encode_segments_keeps_control_tokens_out_of_text_segments() {
        let tokenizer = TiktokenTokenizer::from_dir(k3_byte_dir().path()).unwrap();
        let bpe = &tokenizer.tokenizer;
        let native = tokenizer.native.as_deref();

        let control =
            encode_segments(bpe, native, &[PromptSegment::control("<|open|>")], false).unwrap();
        assert_eq!(control, vec![300]);

        let text = encode_segments(bpe, native, &[PromptSegment::text("<|open|>")], false).unwrap();
        assert!(
            !text.contains(&300),
            "marker in a text segment must not become a control id: {text:?}"
        );
        assert_eq!(tokenizer.decode(&text, false).unwrap(), "<|open|>");

        let mixed = encode_segments(
            bpe,
            native,
            &[
                PromptSegment::control("<|open|>"),
                PromptSegment::text("message"),
                PromptSegment::text("<|open|>"),
            ],
            false,
        )
        .unwrap();
        assert_eq!(mixed.iter().filter(|&&id| id == 300).count(), 1);
        assert_eq!(
            tokenizer.decode(&mixed, false).unwrap(),
            "<|open|>message<|open|>"
        );

        // The flat encode still maps every marker string to its control id.
        let flat = tokenizer.encode("<|open|>message<|open|>", false).unwrap();
        assert_eq!(flat.token_ids().iter().filter(|&&id| id == 300).count(), 2);
    }

    /// The segment encode skips the checkpoint's cut for a segment of at most
    /// `CHECKPOINT_RUN_CHARS` bytes: such a text holds at most that many
    /// characters, so neither cut can fall inside it and it is one piece.
    #[test]
    fn a_segment_within_the_run_bound_in_bytes_is_one_checkpoint_piece() {
        let at_the_bound = " ".repeat(CHECKPOINT_RUN_CHARS);
        assert_eq!(
            checkpoint_pieces(&at_the_bound),
            vec![at_the_bound.as_str()]
        );
        let past_it = " ".repeat(CHECKPOINT_RUN_CHARS + 1);
        assert_eq!(checkpoint_pieces(&past_it).len(), 2);
        let two_byte_characters = "\u{e9}".repeat(CHECKPOINT_RUN_CHARS / 2);
        assert_eq!(
            checkpoint_pieces(&two_byte_characters),
            vec![two_byte_characters.as_str()]
        );
    }

    #[test]
    fn checkpoint_pieces_cut_at_the_window_and_at_long_runs() {
        assert_eq!(checkpoint_pieces(""), Vec::<&str>::new());
        // Up to one window of characters (not bytes) is one piece when no run
        // is long; the next character starts the second window.
        let window = "\u{e9} ".repeat(CHECKPOINT_WINDOW_CHARS / 2);
        assert_eq!(checkpoint_pieces(&window), vec![window.as_str()]);
        let longer = format!("{window}x");
        assert_eq!(checkpoint_pieces(&longer), vec![window.as_str(), "x"]);
        // A run of 25,000 is whole; the run's next character starts a new
        // piece, and the piece before it carries the window's earlier runs.
        let run = "x".repeat(CHECKPOINT_RUN_CHARS);
        assert_eq!(checkpoint_pieces(&run), vec![run.as_str()]);
        let over = format!("ab {run}x");
        assert_eq!(checkpoint_pieces(&over), vec![&over[..over.len() - 1], "x"]);
        // A change of kind restarts the count.
        let two = format!("{run} {run}");
        assert_eq!(checkpoint_pieces(&two), vec![two.as_str()]);
        // Whitespace runs are cut the same way; U+001C..U+001F are whitespace
        // to Python's `str.isspace`, which the checkpoint uses.
        let seps = format!("a{}b", "\u{1c}".repeat(CHECKPOINT_RUN_CHARS + 1));
        assert_eq!(
            checkpoint_pieces(&seps),
            vec![&seps[..seps.len() - 2], "\u{1c}b"]
        );
    }

    /// A `tokenization_kimi` checkpoint encodes in the checkpoint's pieces,
    /// so the merge the one-piece encode makes across a cut does not happen;
    /// any other tiktoken vocabulary encodes the whole text as before.
    #[test]
    fn kimi_checkpoints_encode_in_the_checkpoint_pieces() {
        // 'a' 0, 'b' 1, 'ab' 2
        const MODEL: &str = "YQ== 0\nYg== 1\nYWI= 2\n";
        let kimi = tempfile::tempdir().unwrap();
        std::fs::write(kimi.path().join("tiktoken.model"), MODEL).unwrap();
        std::fs::write(
            kimi.path().join("tokenizer_config.json"),
            r#"{"tokenizer_class": "TikTokenTokenizer",
                "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", null]},
                "added_tokens_decoder": {}}"#,
        )
        .unwrap();
        let other = tempfile::tempdir().unwrap();
        std::fs::write(other.path().join("tiktoken.model"), MODEL).unwrap();
        std::fs::write(
            other.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {}}"#,
        )
        .unwrap();
        let kimi = TiktokenTokenizer::from_dir(kimi.path()).unwrap();
        let other = TiktokenTokenizer::from_dir(other.path()).unwrap();
        assert!(kimi.checkpoint_pieces);
        assert!(!other.checkpoint_pieces);

        // 25,001 letters: the checkpoint cuts before the last one, so it
        // cannot merge with its neighbour.
        let text = format!("{}b", "a".repeat(CHECKPOINT_RUN_CHARS));
        let whole: Vec<TokenIdType> = std::iter::repeat_n(0, CHECKPOINT_RUN_CHARS - 1)
            .chain([2])
            .collect();
        let pieces: Vec<TokenIdType> = std::iter::repeat_n(0, CHECKPOINT_RUN_CHARS)
            .chain([1])
            .collect();
        assert_eq!(kimi.encode(&text, false).unwrap().token_ids(), &pieces[..]);
        assert_eq!(other.encode(&text, false).unwrap().token_ids(), &whole[..]);
        let segment = [PromptSegment::text(&text)];
        assert_eq!(
            encode_segments(
                &kimi.tokenizer,
                kimi.native.as_deref(),
                &segment,
                kimi.checkpoint_pieces
            )
            .unwrap(),
            pieces
        );
        assert_eq!(
            encode_segments(&kimi.tokenizer, kimi.native.as_deref(), &segment, false).unwrap(),
            whole
        );
        // Below the cut nothing changes.
        let short = format!("{}b", "a".repeat(CHECKPOINT_RUN_CHARS - 1));
        assert_eq!(
            kimi.encode(&short, false).unwrap().token_ids(),
            other.encode(&short, false).unwrap().token_ids()
        );
    }

    /// A file whose ranks give a complete merge table and whose pattern
    /// the engine reads encodes on the crate's direct path; the ids are the
    /// file's own encoder's, special tokens included.
    #[test]
    fn a_file_whose_ranks_fit_encodes_on_the_direct_path() {
        let dir = split_vocabulary_dir();
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert!(tokenizer.uses_native_encoder());
        for text in [
            "ab",
            "中文 😀!",
            "[EOS]a<|open|>b",
            "a b\n\n c",
            "the quick's 123456 brown.\r\n",
        ] {
            assert_eq!(
                tokenizer.encode(text, false).unwrap().token_ids(),
                tokenizer.tokenizer.encode_with_special_tokens(text),
                "{text:?}"
            );
        }
        assert_eq!(
            tokenizer.encode("ab", false).unwrap().token_ids(),
            &[97, 98]
        );
        assert_eq!(
            tokenizer.encode("中", false).unwrap().token_ids(),
            &[256, 0xAD]
        );
        assert_eq!(
            tokenizer.encode("[EOS]", false).unwrap().token_ids(),
            &[300]
        );
    }

    /// A file whose ranks miss a byte, or hold a token its own bytes do not
    /// merge back into, keeps its own encoder (which looks a whole piece up
    /// before it merges, so that token is one id there).
    #[test]
    fn a_file_whose_ranks_do_not_fit_keeps_its_own_encoder() {
        let dir = write_minimal_tiktoken_dir("{}", None);
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert!(!tokenizer.uses_native_encoder());
        assert_eq!(tokenizer.encode("ab", false).unwrap().token_ids(), &[0, 1]);

        let mut model = full_byte_tiktoken_model();
        for (token, rank) in [("ab", 256), ("bcd", 257), ("bc", 258), ("abcd", 259)] {
            model.push_str(&format!("{} {rank}\n", STANDARD.encode(token)));
        }
        let dir = write_tiktoken_dir(&model, "{}", None);
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert!(!tokenizer.uses_native_encoder());
        assert_eq!(tokenizer.encode("abcd", false).unwrap().token_ids(), &[259]);
    }

    /// The Kimi checkpoint's pieces hold on the direct path: a text with a
    /// run past the bound encodes as the pieces, each as the file's encoder
    /// encodes it, and the same file without the checkpoint rule as one.
    #[test]
    fn kimi_checkpoint_pieces_hold_on_the_direct_path() {
        let mut model = full_byte_tiktoken_model();
        model.push_str(&format!("{} 256\n", STANDARD.encode("ab")));
        let kimi_dir = write_tiktoken_dir(&model, "{}", Some(r#"{"model_type": "kimi_k3"}"#));
        let kimi = TiktokenTokenizer::from_dir(kimi_dir.path()).unwrap();
        assert!(kimi.checkpoint_pieces && kimi.uses_native_encoder());
        let other_dir = write_tiktoken_dir(&model, "{}", None);
        let other = TiktokenTokenizer::from_dir(other_dir.path()).unwrap();
        assert!(!other.checkpoint_pieces && other.uses_native_encoder());
        let text = format!("{}b", "a".repeat(CHECKPOINT_RUN_CHARS));
        let mut pieces = vec![97; CHECKPOINT_RUN_CHARS];
        pieces.push(98);
        let mut whole = vec![97; CHECKPOINT_RUN_CHARS - 1];
        whole.push(256);
        assert_eq!(kimi.encode(&text, false).unwrap().token_ids(), &pieces[..]);
        assert_eq!(other.encode(&text, false).unwrap().token_ids(), &whole[..]);
        let expected: Vec<TokenIdType> = checkpoint_pieces(&text)
            .into_iter()
            .flat_map(|piece| kimi.tokenizer.encode_with_special_tokens(piece))
            .collect();
        assert_eq!(expected, pieces);
    }

    #[test]
    fn test_k3_chat_template_defers_its_encode() {
        let tokenizer = TiktokenTokenizer::from_dir(k3_byte_dir().path()).unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "<|open|>x"})];
        let params = || ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        };

        let rendered = tokenizer
            .apply_chat_template_with_encoding(&messages, params(), None)
            .unwrap();
        assert_eq!(
            rendered.text,
            tokenizer.apply_chat_template(&messages, params()).unwrap(),
            "the flat text is the served rendering"
        );
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("K3 must defer its encode");
        };
        let ids = job.run().unwrap();
        let expected = encode_segments(
            &tokenizer.tokenizer,
            tokenizer.native.as_deref(),
            &render_kimi_k3_xtml_prompt(KimiXtml::K3, &messages, &params(), None)
                .unwrap()
                .segments,
            tokenizer.checkpoint_pieces,
        )
        .unwrap();
        assert_eq!(ids.token_ids(), &expected[..]);
        // The user-typed marker is text (four structural <|open|> per prompt:
        // thinking-effort message, user message, assistant tail, think tail).
        assert_eq!(ids.token_ids().iter().filter(|&&id| id == 300).count(), 4);
        let flat = tokenizer.encode(&rendered.text, false).unwrap();
        assert_eq!(flat.token_ids().iter().filter(|&&id| id == 300).count(), 5);
    }

    #[test]
    fn test_k3_prefill_keeps_its_control_tokens() {
        let tokenizer = TiktokenTokenizer::from_dir(k3_byte_dir().path()).unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        };
        let prefix = "<|close|>think<|sep|>Sure";
        let rendered = tokenizer
            .apply_chat_template_with_encoding(&messages, params, Some(prefix))
            .unwrap();
        assert!(rendered.text.ends_with(prefix));
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("K3 must defer its encode");
        };
        let ids = job.run().unwrap();
        let tail: Vec<u32> = std::iter::once(301)
            .chain("think".bytes().map(u32::from))
            .chain(std::iter::once(302))
            .chain("Sure".bytes().map(u32::from))
            .collect();
        assert!(
            ids.token_ids().ends_with(&tail),
            "prefill markers must stay control ids: {:?}",
            ids.token_ids()
        );
    }

    #[test]
    fn test_k3_renderer_reports_openai_content_format() {
        let k3 = write_minimal_tiktoken_dir(
            "{}",
            Some(r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#),
        );
        let k3 = TiktokenTokenizer::from_dir(k3.path()).unwrap();
        assert_eq!(
            k3.chat_template_content_format(),
            ChatTemplateContentFormat::OpenAI
        );

        let llama =
            write_minimal_tiktoken_dir("{}", Some(r#"{"architectures": ["LlamaForCausalLM"]}"#));
        let llama = TiktokenTokenizer::from_dir(llama.path()).unwrap();
        assert_eq!(
            llama.chat_template_content_format(),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn test_k3_reports_raw_tool_call_arguments() {
        let k3 = TiktokenTokenizer::from_dir(k3_byte_dir().path()).unwrap();
        let caps = k3.renderer_capabilities();
        assert!(caps.raw_tool_call_arguments, "{caps:?}");
        // The engine's K3 renderer honours `enable_thinking` as an alias of the
        // encoder's `thinking`; the parser must be armed from it as well.
        assert!(
            caps.enable_thinking_alias && !caps.native_assistant_continuation,
            "{caps:?}"
        );
        assert_eq!(k3.native_reasoning_effort_values(), &["low", "high", "max"]);
        assert!(k3.native_reasoning_effort_off_values().is_empty());

        let jinja = write_minimal_tiktoken_dir("{}", None);
        let jinja = TiktokenTokenizer::from_dir(jinja.path()).unwrap();
        assert_eq!(
            jinja.renderer_capabilities(),
            RendererCapabilities {
                native_assistant_continuation: true,
                ..RendererCapabilities::default()
            }
        );
    }

    #[test]
    fn test_k3_reports_the_pending_stub_length() {
        let tokenizer = TiktokenTokenizer::from_dir(k3_byte_dir().path()).unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let params = || ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        };
        // `<|open|>` + b"think" + `<|sep|>` under the byte vocabulary: 7 ids.
        let stub: Vec<u32> = std::iter::once(300)
            .chain("think".bytes().map(u32::from))
            .chain(std::iter::once(302))
            .collect();

        let rendered = tokenizer
            .apply_chat_template_with_encoding(&messages, params(), None)
            .unwrap();
        assert_eq!(rendered.unbilled_prompt_tokens, stub.len() as u32);
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("K3 must defer its encode");
        };
        let ids = job.run().unwrap();
        assert!(
            ids.token_ids().ends_with(&stub),
            "the stub is the prompt's tail: {:?}",
            ids.token_ids()
        );

        // A prefill after the stub is billed.
        let prefilled = tokenizer
            .apply_chat_template_with_encoding(&messages, params(), Some("Sure"))
            .unwrap();
        assert_eq!(prefilled.unbilled_prompt_tokens, stub.len() as u32);
    }

    #[test]
    fn test_flat_renderer_reports_from_text_with_prefill_joined() {
        let dir = write_minimal_tiktoken_dir("{}", None);
        let mut tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        tokenizer
            .set_chat_template(
                "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}A:{% endif %}"
                    .to_string(),
            )
            .unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let params = || ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        };
        let rendered = tokenizer
            .apply_chat_template_with_encoding(&messages, params(), Some(" there"))
            .unwrap();
        assert_eq!(
            rendered.text,
            tokenizer.apply_chat_template(&messages, params()).unwrap() + " there"
        );
        assert!(matches!(rendered.encoding, PromptEncoding::FromText));
        assert_eq!(rendered.unbilled_prompt_tokens, 0);
    }

    /// A continuation the template cannot render falls back to the header
    /// with the message's text after it, and the render reports the header
    /// it fell back to; a continuation the template renders reports none.
    #[test]
    fn a_continuation_the_template_cannot_render_reports_the_header_it_fell_back_to() {
        let dir = write_minimal_tiktoken_dir("{}", None);
        let mut tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        let continuing = || ChatTemplateParams {
            add_generation_prompt: false,
            continue_final_message: true,
            ..Default::default()
        };
        let messages = vec![
            serde_json::json!({"role": "user", "content": "hello"}),
            serde_json::json!({"role": "assistant", "content": " there"}),
        ];
        // No assistant turns: the final message's text never reaches the
        // prompt, so the continuation falls back to the header plus the text.
        tokenizer
            .set_chat_template(
                "{% for m in messages %}{% if m.role != 'assistant' %}{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}A:{% endif %}"
                    .to_string(),
            )
            .unwrap();
        let rendered = tokenizer
            .apply_chat_template_with_encoding(&messages, continuing(), None)
            .unwrap();
        assert_eq!(rendered.text, "helloA: there");
        assert!(!rendered.continued_final_message);
        assert!(rendered.generation_prompt, "the fallback wrote the header");

        // A continuation the template renders has no header.
        tokenizer
            .set_chat_template(
                "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}A:{% endif %}"
                    .to_string(),
            )
            .unwrap();
        let rendered = tokenizer
            .apply_chat_template_with_encoding(&messages, continuing(), None)
            .unwrap();
        assert_eq!(rendered.text, "hello there");
        assert!(rendered.continued_final_message);
        assert!(!rendered.generation_prompt);
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

    /// The whole-sequence decode runs through the per-id table the streams
    /// read and renders what the crate's decode renders: the same bytes,
    /// specials included, the skip flag read from the table's per-id marks.
    #[test]
    fn decode_through_the_table_renders_what_the_crate_decode_renders() {
        let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).unwrap();
        assert!(
            tokenizer.byte_level.is_some(),
            "the builtin vocabulary has a table"
        );
        let ids = tokenizer
            .encode(
                "hello<|endoftext|>w\u{f6}rld \u{4f60}\u{597d} \u{1f642} <|endoftext|>",
                false,
            )
            .unwrap();
        let ids = ids.token_ids().to_vec();
        assert!(ids.contains(&100257));
        assert_eq!(
            tokenizer.decode(&ids, false).unwrap(),
            tokenizer.tokenizer.decode(&ids).unwrap()
        );
        let kept: Vec<TokenIdType> = ids.iter().copied().filter(|id| *id != 100257).collect();
        assert_eq!(
            tokenizer.decode(&ids, true).unwrap(),
            tokenizer.tokenizer.decode(&kept).unwrap()
        );
    }

    /// Bytes that are not UTF-8 render lossily on the table path, as the
    /// crate path's fallback rendered them; an id the table lacks sends the
    /// sequence to the crate path, whose unknown-id error stays.
    #[test]
    fn decode_through_the_table_keeps_the_lossy_text_and_the_unknown_id_error() {
        // `a`, `b`, and the lone lead byte 0xC3 of a two-byte character.
        let dir = write_tiktoken_dir("YQ== 0\nYg== 1\nww== 2\n", "{}", None);
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert!(tokenizer.byte_level.is_some());
        let (text, warnings) = warnings_during(|| tokenizer.decode(&[0, 2, 1], false));
        assert_eq!(text.unwrap(), "a\u{FFFD}b");
        assert_eq!(
            warnings, 1,
            "the lossy conversion is logged once, as the crate path logs it"
        );
        let (text, warnings) = warnings_during(|| tokenizer.decode(&[0, 1], false));
        assert_eq!(text.unwrap(), "ab");
        assert_eq!(warnings, 0);
        assert_eq!(tokenizer.decode(&[2], true).unwrap(), "\u{FFFD}");
        let err = tokenizer.decode(&[0, 2, 1, 9], false).unwrap_err();
        assert!(
            err.to_string()
                .contains("tiktoken decode failed for unknown token id"),
            "unexpected error: {err}"
        );
    }
}
