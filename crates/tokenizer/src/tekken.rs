//! The `tekken.json` vocabulary file: a byte-level BPE in the tiktoken form
//! (base64 token bytes with their ranks, the merges implied by the rank
//! order), its own split pattern, and its special tokens in a block of ids
//! below the ranks. The reader hands the tiktoken backend what a
//! `tiktoken.model` hands it, laid out as the file's reference reader lays
//! it out: an id is a rank plus `default_num_special_tokens`, the vocabulary
//! is cut at `default_vocab_size - default_num_special_tokens`, and a
//! special id the file does not name up to that count is a `<SPECIAL_{n}>`
//! filler. The file's own encoder prepends `<s>` on request and frames
//! nothing else, which the backend applies under `add_special_tokens`.
//!
//! The file's own reader never matches a special token's spelling in text:
//! it writes a request's control tokens structurally and BPE-encodes every
//! request string as plain text. The backend renders a chat template to
//! text and recognizes the spellings in it, so the loader asks for the
//! backend's structural form (see `TiktokenTokenizer::controls`): the
//! templates are compiled with the spellings marked, and the render tells
//! the template's control tokens from the same spelling typed into a
//! message, which stays text, as the reference reads it. The flat `encode`
//! reads a spelling as its token, as the checkpoint's conversion does; the
//! tests `a_control_spelling_typed_into_a_message_stays_text` and
//! `encode_reads_a_special_spelling_as_its_token` pin the two readings.

use std::{collections::HashMap, path::Path};

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::traits::TokenIdType;

/// The file name of the layout: a checkpoint directory ships the vocabulary
/// under it.
pub(crate) const FILE_NAME: &str = "tekken.json";

/// The spellings the reference reader looks its sentence markers up by.
pub(crate) const BOS: &str = "<s>";
pub(crate) const EOS: &str = "</s>";
pub(crate) const UNK: &str = "<unk>";
pub(crate) const PAD: &str = "<pad>";

/// The special tokens of a file that lists none: the reference reader's
/// list for the versions up to v7, which predate the `special_tokens` block.
const DEPRECATED_SPECIAL_TOKENS: [&str; 20] = [
    UNK,
    BOS,
    EOS,
    "[INST]",
    "[/INST]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[TOOL_CALLS]",
    "[IMG]",
    PAD,
    "[IMG_BREAK]",
    "[IMG_END]",
    "[PREFIX]",
    "[MIDDLE]",
    "[SUFFIX]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[TOOL_CONTENT]",
];

/// Whether `path` names a file of the layout, by the reference reader's
/// rule: `tekken` in the name and the `.json` suffix.
pub(crate) fn is_tekken_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("tekken") && name.ends_with(".json"))
}

/// Whether `dir` ships the layout's file.
pub(crate) fn has_tekken_file(dir: &Path) -> bool {
    dir.join(FILE_NAME).is_file()
}

/// The file as written: the configuration block, the ranked tokens and the
/// special tokens; the image, audio and model-settings blocks are not read.
#[derive(Deserialize)]
struct File {
    config: Config,
    vocab: Vec<Token>,
    #[serde(default)]
    special_tokens: Option<Vec<SpecialToken>>,
}

#[derive(Deserialize)]
struct Config {
    pattern: String,
    default_vocab_size: usize,
    default_num_special_tokens: usize,
    #[serde(default)]
    version: Option<String>,
}

/// A ranked token; its `token_str` is a rendering of the bytes, null where
/// they are not UTF-8, and is not read.
#[derive(Deserialize)]
struct Token {
    rank: u32,
    token_bytes: String,
}

/// A special token; its `is_control` flag reaches neither the encoder nor
/// the decoder of the reference reader and is not read.
#[derive(Deserialize)]
struct SpecialToken {
    rank: u32,
    token_str: String,
}

/// What the file gives the tokenizer.
pub(crate) struct Vocabulary {
    /// Token bytes to id: the rank plus the special-token count.
    pub(crate) ranks: FxHashMap<Vec<u8>, u32>,
    /// The split pattern the ranks were trained under.
    pub(crate) pattern: String,
    /// Special token spellings to id: the file's list, and a filler for
    /// every id of the block it leaves unnamed.
    pub(crate) specials: HashMap<String, TokenIdType>,
}

impl Vocabulary {
    /// The id of the special token spelled `spelling`, if the file has it.
    pub(crate) fn special(&self, spelling: &str) -> Option<TokenIdType> {
        self.specials.get(spelling).copied()
    }
}

/// Reads the file at `path`.
pub(crate) fn read(path: &Path) -> Result<Vocabulary> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let file: File = serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not a tekken vocabulary file", path.display()))?;
    vocabulary(file).with_context(|| format!("{} is not a valid tekken vocabulary", path.display()))
}

/// The vocabulary of a parsed file, checked as the reference reader checks
/// it: the ranks in order, the first 256 tokens the single bytes, no token
/// twice, the special tokens distinct and inside their block.
fn vocabulary(file: File) -> Result<Vocabulary> {
    let File {
        config,
        vocab,
        special_tokens,
    } = file;
    let num_special = config.default_num_special_tokens;
    let Some(inner) = config.default_vocab_size.checked_sub(num_special) else {
        bail!(
            "default_vocab_size {} is below default_num_special_tokens {num_special}",
            config.default_vocab_size
        );
    };
    if vocab.len() < inner {
        bail!(
            "the vocabulary lists {} tokens, the configuration needs {inner}",
            vocab.len()
        );
    }
    if config.pattern.is_empty() {
        bail!("the configuration has no split pattern");
    }
    let offset = TokenIdType::try_from(num_special)
        .context("default_num_special_tokens does not fit an id")?;
    let mut ranks = FxHashMap::with_capacity_and_hasher(inner, Default::default());
    for (index, token) in vocab.into_iter().take(inner).enumerate() {
        let rank = u32::try_from(index).context("more ranks than ids")?;
        if token.rank != rank {
            bail!(
                "token {index} carries rank {}: the ranks are not in order",
                token.rank
            );
        }
        let bytes = STANDARD
            .decode(&token.token_bytes)
            .with_context(|| format!("token {index}: token_bytes is not base64"))?;
        if u8::try_from(index).is_ok_and(|byte| bytes.as_slice() != [byte]) {
            bail!("token {index} is not the byte {index:#04x}");
        }
        let id = rank
            .checked_add(offset)
            .context("an id past the id range")?;
        if ranks.insert(bytes, id).is_some() {
            bail!("token {index} repeats an earlier token's bytes");
        }
    }
    let listed: Vec<SpecialToken> = match special_tokens {
        Some(listed) => listed,
        None if version_at_most_v7(config.version.as_deref()) => DEPRECATED_SPECIAL_TOKENS
            .iter()
            .zip(0u32..)
            .map(|(spelling, rank)| SpecialToken {
                rank,
                token_str: (*spelling).to_owned(),
            })
            .collect(),
        None => bail!(
            "the file lists no special tokens and its version {:?} ships them in the file",
            config.version
        ),
    };
    let mut specials: HashMap<String, TokenIdType> = HashMap::with_capacity(num_special);
    let mut named = vec![false; num_special];
    for special in listed {
        if special.token_str.is_empty() {
            bail!("special id {} has an empty spelling", special.rank);
        }
        let Some(slot) = named.get_mut(special.rank as usize) else {
            bail!(
                "special token {:?} has rank {} past the {num_special} special ids",
                special.token_str,
                special.rank
            );
        };
        if std::mem::replace(slot, true) {
            bail!("special id {} is named twice", special.rank);
        }
        if specials
            .insert(special.token_str.clone(), special.rank)
            .is_some()
        {
            bail!("special token {:?} is listed twice", special.token_str);
        }
    }
    for id in named
        .iter()
        .zip(0u32..)
        .filter_map(|(&named, id)| (!named).then_some(id))
    {
        if specials.insert(format!("<SPECIAL_{id}>"), id).is_some() {
            bail!("<SPECIAL_{id}> is listed under another id and is id {id}'s filler");
        }
    }
    Ok(Vocabulary {
        ranks,
        pattern: config.pattern,
        specials,
    })
}

/// Whether a file version predates the `special_tokens` block (`v1` to
/// `v7`); an unreadable version does not.
fn version_at_most_v7(version: Option<&str>) -> bool {
    version
        .and_then(|version| version.strip_prefix('v'))
        .and_then(|number| number.parse::<u32>().ok())
        .is_some_and(|number| number <= 7)
}

/// A small vocabulary of the layout for the loader tests: a special block,
/// the 256 bytes, then the merges `he`, `ll`, `hell`, `hello` (`hello` is id
/// 259 + the block), so `hello` is one id and ` hello` the space byte's id
/// and that one.
#[cfg(test)]
pub(crate) mod testing {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde_json::{json, Value};

    /// The split pattern of the real files.
    pub(crate) const PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    pub(crate) const MERGES: [&str; 4] = ["he", "ll", "hell", "hello"];

    /// The file text: the 256 bytes and `merges` ranked in order, `specials`
    /// as `(rank, spelling)`, with the configured sizes.
    pub(crate) fn tekken_json(
        merges: &[&str],
        specials: Option<&[(u32, &str)]>,
        num_special: usize,
        vocab_size: usize,
        version: &str,
    ) -> String {
        let bytes = (0u8..=255).map(|byte| vec![byte]);
        let vocab: Vec<Value> = bytes
            .chain(merges.iter().map(|merge| merge.as_bytes().to_vec()))
            .enumerate()
            .map(|(rank, token)| {
                json!({
                    "rank": rank,
                    "token_bytes": STANDARD.encode(&token),
                    "token_str": String::from_utf8(token).ok(),
                })
            })
            .collect();
        let mut file = json!({
            "config": {
                "pattern": PATTERN,
                "num_vocab_tokens": vocab.len(),
                "default_vocab_size": vocab_size,
                "default_num_special_tokens": num_special,
                "version": version,
            },
            "vocab": vocab,
        });
        if let Some(specials) = specials {
            file["special_tokens"] = specials
                .iter()
                .map(|(rank, spelling)| {
                    json!({"rank": rank, "token_str": spelling, "is_control": true})
                })
                .collect();
        }
        file.to_string()
    }

    /// `<unk>`, `<s>`, `</s>` and the four merges: ids 0..=2 the specials,
    /// 3..=258 the bytes, 259..=262 the merges, `vocab_size` 263.
    pub(crate) fn minimal() -> String {
        tekken_json(
            &MERGES,
            Some(&[(0, "<unk>"), (1, "<s>"), (2, "</s>")]),
            3,
            263,
            "v13",
        )
    }

    /// A directory holding `tekken.json` with `text`.
    pub(crate) fn dir_with(text: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(super::FILE_NAME), text).unwrap();
        dir
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::{testing, *};
    use crate::{
        chat_template::ChatTemplateParams,
        sequence::Sequence,
        tiktoken::TiktokenTokenizer,
        traits::{Decoder, Encoder, PromptEncoding, Tokenizer},
    };

    fn load(dir: &Path) -> TiktokenTokenizer {
        TiktokenTokenizer::from_tekken_file(&dir.join(FILE_NAME))
            .unwrap_or_else(|e| panic!("the file does not load: {e:#}"))
    }

    fn ids(tokenizer: &TiktokenTokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
        tokenizer
            .encode(text, add_special_tokens)
            .unwrap()
            .token_ids()
            .to_vec()
    }

    #[test]
    fn the_ids_are_the_ranks_offset_by_the_special_block() {
        let dir = testing::dir_with(&testing::minimal());
        let tokenizer = load(dir.path());
        assert_eq!(tokenizer.vocab_size(), 263);
        assert_eq!(ids(&tokenizer, "hello", false), [262]);
        // ` hello` is one piece of the pattern: the space byte, then the merge.
        assert_eq!(ids(&tokenizer, "hello hello", false), [262, 35, 262]);
        assert_eq!(tokenizer.token_to_id("hello"), Some(262));
        assert_eq!(tokenizer.id_to_token(262).as_deref(), Some("hello"));
        assert_eq!(tokenizer.token_to_id("<s>"), Some(1));
        assert_eq!(tokenizer.token_to_id("</s>"), Some(2));
        assert_eq!(
            tokenizer.get_special_tokens().bos_token.as_deref(),
            Some("<s>")
        );
        assert_eq!(
            tokenizer.get_special_tokens().eos_token.as_deref(),
            Some("</s>")
        );
        assert_eq!(
            tokenizer.get_special_tokens().unk_token.as_deref(),
            Some("<unk>")
        );
        assert_eq!(tokenizer.get_special_tokens().pad_token, None);
        assert_eq!(tokenizer.eos_token_ids(), [2]);
    }

    /// `add_special_tokens` is the file's own frame: BOS first, nothing after.
    #[test]
    fn the_flag_prepends_bos_and_nothing_else() {
        let dir = testing::dir_with(&testing::minimal());
        let tokenizer = load(dir.path());
        assert_eq!(ids(&tokenizer, "hello", true), [1, 262]);
        assert_eq!(ids(&tokenizer, "", true), [1]);
        assert!(ids(&tokenizer, "", false).is_empty());
        let batch = tokenizer
            .encode_batch(&["hello", "hello hello"], true)
            .unwrap();
        assert_eq!(batch[0].token_ids(), [1, 262]);
        assert_eq!(batch[1].token_ids(), [1, 262, 35, 262]);
    }

    /// The flat `encode` reads a special token's spelling as its token, as
    /// the checkpoint's conversion does; the render is where a typed
    /// spelling stays text (the structural tests at the end).
    #[test]
    fn encode_reads_a_special_spelling_as_its_token() {
        let dir = testing::dir_with(&testing::minimal());
        let tokenizer = load(dir.path());
        assert_eq!(ids(&tokenizer, "<s>hello</s>", false), [1, 262, 2]);
        assert_eq!(ids(&tokenizer, "<s>hello</s>", true), [1, 1, 262, 2]);
    }

    /// `skip_special_tokens` drops every id of the block, a filler included;
    /// without it the spelling is rendered, as the reference's KEEP policy.
    #[test]
    fn decode_keeps_or_drops_the_whole_special_block() {
        let two_named = testing::tekken_json(
            &testing::MERGES,
            Some(&[(0, "<unk>"), (1, "<s>")]),
            3,
            263,
            "v13",
        );
        let dir = testing::dir_with(&two_named);
        let tokenizer = load(dir.path());
        assert_eq!(tokenizer.id_to_token(2).as_deref(), Some("<SPECIAL_2>"));
        assert_eq!(tokenizer.token_to_id("<SPECIAL_2>"), Some(2));
        assert_eq!(
            tokenizer.decode(&[1, 262, 35, 262, 2], false).unwrap(),
            "<s>hello hello<SPECIAL_2>"
        );
        assert_eq!(
            tokenizer.decode(&[1, 262, 35, 262, 2], true).unwrap(),
            "hello hello"
        );
        // The per-stream decoder follows the same flag.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
        for (skip, expected) in [(false, "<s>hello hello<SPECIAL_2>"), (true, "hello hello")] {
            let mut sequence = Sequence::new_with_options(Arc::clone(&tokenizer), skip);
            let mut streamed = String::new();
            for id in [1, 262, 35, 262, 2] {
                streamed.push_str(&sequence.append_token(id).unwrap());
            }
            streamed.push_str(&sequence.finish().unwrap());
            assert_eq!(streamed, expected, "skip {skip}");
        }
    }

    /// Bytes that are no token of their own round-trip through the byte
    /// ranks, a character split across ids included.
    #[test]
    fn text_outside_the_merges_round_trips_through_the_bytes() {
        let dir = testing::dir_with(&testing::minimal());
        let tokenizer = load(dir.path());
        let text = "héllo, wörld 🙂";
        let encoded = ids(&tokenizer, text, false);
        // Byte ids throughout, except the `ll` merge (rank 257, id 260).
        assert!(
            encoded
                .iter()
                .all(|&id| (3..259).contains(&id) || id == 260),
            "{encoded:?}"
        );
        assert!(encoded.contains(&260), "{encoded:?}");
        assert_eq!(tokenizer.decode(&encoded, false).unwrap(), text);
        // A stream releases a character only once its bytes are all in.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
        let mut sequence = Sequence::new_with_options(Arc::clone(&tokenizer), false);
        let pieces: Vec<String> = encoded
            .iter()
            .map(|&id| sequence.append_token(id).unwrap())
            .collect();
        assert_eq!(pieces.concat() + &sequence.finish().unwrap(), text);
        assert!(
            pieces
                .iter()
                .all(|piece| piece.is_empty()
                    || piece.chars().all(|c| c != char::REPLACEMENT_CHARACTER)),
            "{pieces:?}"
        );
    }

    /// The vocabulary is cut at `default_vocab_size - default_num_special_tokens`.
    #[test]
    fn the_vocabulary_is_cut_at_the_configured_size() {
        let longer = testing::tekken_json(
            &["he", "ll", "hell", "hello", "hellox"],
            Some(&[(0, "<unk>"), (1, "<s>"), (2, "</s>")]),
            3,
            263,
            "v13",
        );
        let dir = testing::dir_with(&longer);
        let tokenizer = load(dir.path());
        assert_eq!(tokenizer.vocab_size(), 263);
        assert_eq!(tokenizer.token_to_id("hellox"), None);
        assert_eq!(ids(&tokenizer, "hellox", false), [262, 123]);
    }

    /// A file without the block (the layout's early versions) takes the
    /// reference reader's list; a later version without it is refused.
    #[test]
    fn a_file_without_the_special_block_takes_the_reference_list_up_to_v7() {
        let early = testing::tekken_json(&testing::MERGES, None, 20, 280, "v3");
        let dir = testing::dir_with(&early);
        let tokenizer = load(dir.path());
        assert_eq!(tokenizer.token_to_id("<s>"), Some(1));
        assert_eq!(tokenizer.token_to_id("[INST]"), Some(3));
        assert_eq!(tokenizer.token_to_id("[TOOL_CONTENT]"), Some(19));
        assert_eq!(ids(&tokenizer, "hello", true), [1, 279]);

        let later = testing::tekken_json(&testing::MERGES, None, 20, 280, "v13");
        let dir = testing::dir_with(&later);
        let Err(error) = read(&dir.path().join(FILE_NAME)) else {
            panic!("a later version without the block loaded");
        };
        assert!(
            format!("{error:#}").contains("lists no special tokens"),
            "{error:#}"
        );
    }

    #[test]
    fn the_reader_refuses_a_file_whose_layout_breaks_the_rules() {
        let refused = |text: String, reason: &str| {
            let dir = testing::dir_with(&text);
            let Err(error) = read(&dir.path().join(FILE_NAME)) else {
                panic!("a file that breaks the rules loaded ({reason})");
            };
            assert!(format!("{error:#}").contains(reason), "{error:#}");
        };
        let minimal = testing::minimal();
        refused(
            minimal.replacen("\"rank\":256", "\"rank\":257", 1),
            "not in order",
        );
        // Byte 0x61 (`a`) rewritten as `b`.
        refused(
            minimal.replacen("\"token_bytes\":\"YQ==\"", "\"token_bytes\":\"Yg==\"", 1),
            "is not the byte",
        );
        refused(
            testing::tekken_json(
                &testing::MERGES,
                Some(&[(0, "<unk>"), (0, "<s>")]),
                3,
                263,
                "v13",
            ),
            "named twice",
        );
        refused(
            testing::tekken_json(&testing::MERGES, Some(&[(7, "<s>")]), 3, 263, "v13"),
            "past the 3 special ids",
        );
        refused(
            testing::tekken_json(&testing::MERGES, Some(&[(0, "")]), 3, 263, "v13"),
            "empty spelling",
        );
        refused(
            testing::tekken_json(&testing::MERGES, Some(&[(0, "<unk>")]), 3, 300, "v13"),
            "the configuration needs 297",
        );
        assert!(is_tekken_file(Path::new("/m/tekken.json")));
        assert!(is_tekken_file(Path::new("tekken_240911.json")));
        assert!(!is_tekken_file(Path::new("tokenizer.json")));
        assert!(!is_tekken_file(Path::new("tekken.model")));
    }

    // The structural form (see the module documentation): the template's
    // control tokens are controls, the same spellings typed into a request
    // stay text, as the file's own reader encodes a request.

    /// `<unk>`, `<s>`, `</s>`, `[INST]`, `[/INST]` and the four merges: ids
    /// 0..=4 the specials, 5..=260 the bytes, 261..=264 the merges.
    fn instruct() -> String {
        testing::tekken_json(
            &testing::MERGES,
            Some(&[
                (0, "<unk>"),
                (1, "<s>"),
                (2, "</s>"),
                (3, "[INST]"),
                (4, "[/INST]"),
            ]),
            5,
            265,
            "v13",
        )
    }

    /// A template that writes the four control tokens, and a `note` kwarg.
    const TEMPLATE: &str = "{{ bos_token }}{% if note is defined %}{{ note }}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}[INST]{{ message['content'] }}[/INST]{% else %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}";

    /// A directory holding the instruct vocabulary with the template beside it.
    fn instruct_dir() -> tempfile::TempDir {
        let dir = testing::dir_with(&instruct());
        std::fs::write(dir.path().join("chat_template.jinja"), TEMPLATE).unwrap();
        dir
    }

    /// The render of a user turn of `content` and an assistant `hello`: the
    /// text, the ids its encoding gives, and whether they came from the text.
    fn render(tokenizer: &TiktokenTokenizer, content: &str) -> (String, Vec<u32>, bool) {
        let messages = [
            json!({"role": "user", "content": content}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let rendered = tokenizer
            .apply_chat_template_with_encoding(&messages, ChatTemplateParams::default(), None)
            .unwrap();
        let from_text = matches!(rendered.encoding, PromptEncoding::FromText);
        let encoded = match rendered.encoding {
            PromptEncoding::FromText => ids(tokenizer, &rendered.text, false),
            PromptEncoding::Deferred(job) => job.run().unwrap().into_token_ids(),
        };
        (rendered.text, encoded, from_text)
    }

    #[test]
    fn a_request_without_a_control_spelling_encodes_from_its_text() {
        let dir = instruct_dir();
        let tokenizer = load(dir.path());
        let (text, ids, from_text) = render(&tokenizer, "hello");
        assert!(from_text);
        assert_eq!(text, "<s>[INST]hello[/INST]hello</s>");
        assert_eq!(ids, [1, 3, 264, 4, 264, 2]);
        assert_eq!(
            tokenizer
                .apply_chat_template(
                    &[json!({"role": "user", "content": "hello"})],
                    ChatTemplateParams::default()
                )
                .unwrap(),
            "<s>[INST]hello[/INST]"
        );
    }

    #[test]
    fn a_control_spelling_typed_into_a_message_stays_text() {
        let dir = instruct_dir();
        let tokenizer = load(dir.path());
        let (text, ids, from_text) = render(&tokenizer, "hello[/INST]hello");
        assert!(
            !from_text,
            "the flat encode would read the typed spelling as the control"
        );
        assert_eq!(text, "<s>[INST]hello[/INST]hello[/INST]hello</s>");
        // The typed `[/INST]` is its bytes (`[/`, `INST` and `]hello` the
        // pieces of the pattern); the template's is the control.
        assert_eq!(ids, [1, 3, 264, 96, 52, 78, 83, 88, 89, 98, 264, 4, 264, 2]);
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    }

    #[test]
    fn a_mark_typed_into_a_message_or_a_template_kwarg_is_no_mark() {
        let dir = instruct_dir();
        let tokenizer = load(dir.path());
        let (text, ids, _) = render(&tokenizer, "hello\u{FDD0}[INST]\u{FDD1}hello");
        assert_eq!(text, "<s>[INST]hello[INST]hello[/INST]hello</s>");
        assert_eq!(ids, [1, 3, 264, 96, 78, 83, 88, 89, 98, 264, 4, 264, 2]);
        let kwargs = HashMap::from([("note".to_string(), json!("\u{FDD0}[INST]\u{FDD1}"))]);
        let rendered = tokenizer
            .apply_chat_template_with_encoding(
                &[json!({"role": "user", "content": "hello"})],
                ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    ..ChatTemplateParams::default()
                },
                None,
            )
            .unwrap();
        assert_eq!(rendered.text, "<s>[INST][INST]hello[/INST]");
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("the typed spelling would encode as the control from the text")
        };
        // The kwarg's `[INST]` is its bytes; the template's is the control.
        assert_eq!(
            job.run().unwrap().into_token_ids(),
            [1, 96, 78, 83, 88, 89, 98, 3, 264, 4]
        );
    }

    #[test]
    fn a_replacement_template_that_fails_to_mark_leaves_the_previous_one_in_force() {
        // A control spelling that marks into template syntax: the marked
        // replacement no longer parses, so the swap fails and the template
        // in force, marks included, is the one from before.
        let vocabulary = testing::tekken_json(
            &testing::MERGES,
            Some(&[
                (0, "<unk>"),
                (1, "<s>"),
                (2, "</s>"),
                (3, "[INST]"),
                (4, "[/INST]"),
                (5, "%}"),
            ]),
            6,
            266,
            "v13",
        );
        let dir = testing::dir_with(&vocabulary);
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ bos_token }}{{ messages[0]['content'] }}",
        )
        .unwrap();
        let mut tokenizer = load(dir.path());
        let before = render(&tokenizer, "hello[/INST]hello");
        assert!(!before.2);
        assert_eq!(before.1.iter().filter(|&&id| id == 4).count(), 0);
        let replacement = "{% if true %}{{ bos_token }}{{ messages[0]['content'] }}{% endif %}";
        assert!(tokenizer
            .set_chat_template(replacement.to_string())
            .is_err());
        assert_eq!(render(&tokenizer, "hello[/INST]hello"), before);
    }

    #[test]
    fn a_replaced_template_renders_under_the_structural_form_too() {
        let dir = instruct_dir();
        let mut tokenizer = load(dir.path());
        tokenizer
            .set_chat_template(TEMPLATE.replace("[INST]", "[INST] "))
            .unwrap();
        let (text, ids, from_text) = render(&tokenizer, "hello[/INST]hello");
        assert!(!from_text);
        assert_eq!(text, "<s>[INST] hello[/INST]hello[/INST]hello</s>");
        assert_eq!(ids.iter().filter(|&&id| id == 4).count(), 1);
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    }
}
