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

use std::{borrow::Cow, collections::HashMap, path::Path};

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::{tiktoken::RankFile, traits::TokenIdType};

/// The file name of the layout: a checkpoint directory ships the vocabulary
/// under it.
pub(crate) const FILE_NAME: &str = "tekken.json";

/// The spellings the reference reader looks its sentence markers up by.
pub(crate) const BOS: &str = "<s>";
pub(crate) const EOS: &str = "</s>";
pub(crate) const UNK: &str = "<unk>";
pub(crate) const PAD: &str = "<pad>";

/// The split pattern of the real files. A test proves it against the own
/// encoder, so a load with it skips that check (see `SHIPPED_PATTERNS` in
/// `tiktoken.rs`).
pub(crate) const PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The most special ids a configuration may name: a bound of this reader,
/// beyond the checks of the reference reader. A load keeps a slot for every
/// special id the configuration names. The files in use name 1,000.
const MAX_SPECIAL_TOKENS: usize = 1 << 16;

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
struct File<'a> {
    config: Config,
    #[serde(borrow)]
    vocab: Vec<Token<'a>>,
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
struct Token<'a> {
    rank: u32,
    // Borrowed from the file's text; owned only where the writer escaped a
    // character.
    #[serde(borrow)]
    token_bytes: Cow<'a, str>,
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
    /// The tokens in rank order, each with its id: the rank plus the
    /// special-token count.
    pub(crate) ranks: RankFile,
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
    // One UTF-8 check of the whole file in place of one a string; a file
    // that is not UTF-8 in a place the reader skips is read as before.
    let file: File = match std::str::from_utf8(&bytes) {
        Ok(text) => match written_layout(text) {
            Some(file) => Ok(file),
            None => serde_json::from_str(text),
        },
        Err(_) => serde_json::from_slice(&bytes),
    }
    .with_context(|| format!("{} is not a tekken vocabulary file", path.display()))?;
    vocabulary(file).with_context(|| format!("{} is not a valid tekken vocabulary", path.display()))
}

/// The file whose ranked tokens are written as the reference writer writes
/// them: objects of `rank` (a plain integer), `token_bytes` (a string
/// without an escape) and `token_str` (a string or null), in that order.
/// The array is walked here, at a fraction of the JSON reader's pass over
/// it; the text around it goes through the JSON reader with an empty array
/// in its place. `None` for any other shape and for a text the JSON reader
/// would refuse: the caller hands it the whole file, so every file reads as
/// before and every error keeps its text.
fn written_layout(text: &str) -> Option<File<'_>> {
    let mut walk = Walk { text, at: 0 };
    walk.take(b"{")?;
    loop {
        walk.take(b"\"")?;
        let key = walk.plain_string()?;
        walk.take(b":")?;
        walk.space();
        if key == "vocab" {
            break;
        }
        walk.skip_value()?;
        walk.take(b",")?;
    }
    let start = walk.at;
    walk.take(b"[")?;
    let mut vocab = Vec::new();
    if walk.take(b"]").is_none() {
        // A writer repeats the text between two values byte for byte: it is
        // walked token by token once and compared after that.
        let mut seams = [None; 3];
        walk.take(b"{")?;
        walk.take(b"\"rank\"")?;
        walk.take(b":")?;
        loop {
            let rank = walk.integer()?;
            walk.seam(&mut seams[0], |walk| {
                walk.take(b",")?;
                walk.take(b"\"token_bytes\"")?;
                walk.take(b":")?;
                walk.take(b"\"")
            })?;
            let token_bytes = Cow::Borrowed(walk.plain_string()?);
            walk.seam(&mut seams[1], |walk| {
                walk.take(b",")?;
                walk.take(b"\"token_str\"")?;
                walk.take(b":")
            })?;
            if walk.take(b"\"").is_some() {
                walk.skipped_string()?;
            } else {
                walk.take(b"null")?;
            }
            vocab.push(Token { rank, token_bytes });
            let next = walk.seam(&mut seams[2], |walk| {
                walk.take(b"}")?;
                walk.take(b",")?;
                walk.take(b"{")?;
                walk.take(b"\"rank\"")?;
                walk.take(b":")
            });
            if next.is_none() {
                break;
            }
        }
        walk.take(b"}")?;
        walk.take(b"]")?;
    }
    let around = [text.get(..start)?, "[]", text.get(walk.at..)?].concat();
    let File {
        config,
        special_tokens,
        ..
    } = serde_json::from_str(&around).ok()?;
    Some(File {
        config,
        vocab,
        special_tokens,
    })
}

/// A position in the file's text for [`written_layout`]. Every method
/// answers `None` where the JSON reader would refuse the text, or might.
struct Walk<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Walk<'a> {
    fn rest(&self) -> &'a [u8] {
        self.text.as_bytes().get(self.at..).unwrap_or_default()
    }

    fn space(&mut self) {
        let space = |byte: &&u8| matches!(byte, b' ' | b'\n' | b'\r' | b'\t');
        self.at += self.rest().iter().take_while(space).count();
    }

    /// Passes `token` and the white space in front of it.
    fn take(&mut self, token: &[u8]) -> Option<()> {
        self.space();
        self.rest()
            .starts_with(token)
            .then(|| self.at += token.len())
    }

    /// Passes the text between two values: by a compare where `seam`, the
    /// text `tokens` passed at this place last, stands here again, else by
    /// `tokens`.
    fn seam(
        &mut self,
        seam: &mut Option<&'a [u8]>,
        tokens: impl Fn(&mut Self) -> Option<()>,
    ) -> Option<()> {
        let (from, rest) = (self.at, self.rest());
        match *seam {
            Some(seam) if rest.starts_with(seam) => self.at += seam.len(),
            _ => {
                if tokens(self).is_none() {
                    self.at = from;
                    return None;
                }
                *seam = rest.get(..self.at - from);
            }
        }
        Some(())
    }

    /// The rest of a string that has no escape; ends past its quote.
    fn plain_string(&mut self) -> Option<&'a str> {
        let rest = self.rest();
        let len = rest
            .iter()
            .position(|byte| matches!(byte, b'"' | b'\\' | 0..=0x1f))?;
        let string = self.text.get(self.at..self.at + len)?;
        (rest.get(len) == Some(&b'"')).then(|| {
            self.at += len + 1;
            string
        })
    }

    /// Passes the rest of a string that is not read, its escapes checked as
    /// the JSON reader checks them.
    fn skipped_string(&mut self) -> Option<()> {
        let bytes = self.text.as_bytes();
        let mut at = self.at;
        loop {
            match *bytes.get(at)? {
                b'"' => break,
                b'\\' => match *bytes.get(at + 1)? {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => at += 2,
                    b'u' => {
                        // The code unit of the four hex digits at `at`.
                        let unit = |at: usize| {
                            let hex = bytes.get(at..at + 4)?;
                            hex.iter().try_fold(0u32, |unit, &digit| {
                                Some((unit << 4) | char::from(digit).to_digit(16)?)
                            })
                        };
                        at += match unit(at + 2)? {
                            // A high half right before a low half is one
                            // character to every version of the JSON
                            // reader; whether any other half is an error
                            // is its rule and changed between its versions.
                            0xD800..=0xDBFF
                                if matches!(bytes.get(at + 6..at + 8), Some(b"\\u"))
                                    && matches!(unit(at + 8), Some(0xDC00..=0xDFFF)) =>
                            {
                                12
                            }
                            0xD800..=0xDFFF => return None,
                            _ => 6,
                        };
                    }
                    _ => return None,
                },
                0..=0x1f => return None,
                _ => at += 1,
            }
        }
        self.at = at + 1;
        Some(())
    }

    /// A number of digits alone, without a leading zero, that fits a rank.
    fn integer(&mut self) -> Option<u32> {
        self.space();
        let rest = self.rest();
        let digits = rest.iter().take_while(|byte| byte.is_ascii_digit()).count();
        let number = rest.get(..digits)?;
        if number.is_empty() || (digits > 1 && number.first() == Some(&b'0')) {
            return None;
        }
        self.at += digits;
        number.iter().try_fold(0u32, |value, digit| {
            value.checked_mul(10)?.checked_add(u32::from(digit - b'0'))
        })
    }

    /// Passes a value in front of the array; the JSON reader checks it
    /// when it reads the text around the array.
    fn skip_value(&mut self) -> Option<()> {
        let mut depth = 0usize;
        loop {
            match *self.rest().first()? {
                b',' | b'}' | b']' if depth == 0 => return Some(()),
                b'"' => {
                    self.at += 1;
                    self.skipped_string()?;
                    continue;
                }
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                _ => {}
            }
            self.at += 1;
        }
    }
}

/// The ids by their token. A token the file lists twice is refused here,
/// where the map is built, as the reference reader refuses it.
pub(crate) fn by_token(tokens: &RankFile) -> Result<FxHashMap<&[u8], TokenIdType>> {
    let mut ranks = FxHashMap::with_capacity_and_hasher(tokens.ranks.len(), Default::default());
    for (index, (token, id)) in tokens.tokens().enumerate() {
        if ranks.insert(token, id).is_some() {
            bail!("token {index} repeats an earlier token's bytes");
        }
    }
    Ok(ranks)
}

/// The vocabulary of a parsed file, checked as the reference reader checks
/// it: the ranks in order, the first 256 tokens the single bytes, the
/// special tokens distinct and inside their block ([`by_token`] refuses a
/// token listed twice).
fn vocabulary(file: File<'_>) -> Result<Vocabulary> {
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
    if num_special > MAX_SPECIAL_TOKENS {
        bail!(
            "default_num_special_tokens {num_special} is above the bound of {MAX_SPECIAL_TOKENS}"
        );
    }
    // Four characters of base64 decode to three bytes at the most.
    let bytes = vocab
        .iter()
        .take(inner)
        .map(|token| token.token_bytes.len().div_ceil(4) * 3)
        .sum();
    let mut ranks = RankFile::with_capacity(inner, bytes);
    for (index, token) in vocab.into_iter().take(inner).enumerate() {
        let rank = u32::try_from(index).context("more ranks than ids")?;
        if token.rank != rank {
            bail!(
                "token {index} carries rank {}: the ranks are not in order",
                token.rank
            );
        }
        let start = ranks.bytes.len();
        STANDARD
            .decode_vec(token.token_bytes.as_bytes(), &mut ranks.bytes)
            .with_context(|| format!("token {index}: token_bytes is not base64"))?;
        if u8::try_from(index).is_ok_and(|byte| ranks.bytes[start..] != [byte]) {
            bail!("token {index} is not the byte {index:#04x}");
        }
        let id = rank
            .checked_add(offset)
            .context("an id past the id range")?;
        ranks.end_token(id)?;
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
                "pattern": super::PATTERN,
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

    /// The reader skips a ranked token's `token_str`: a byte that is not
    /// UTF-8 there does not stop the load; in a field it reads, the file is
    /// refused.
    #[test]
    fn a_file_that_is_not_utf8_where_the_reader_skips_still_loads() {
        let text = testing::minimal();
        let dir = testing::dir_with(&text);
        let path = dir.path().join(FILE_NAME);
        let whole = read(&path).unwrap().ranks;

        let mut bytes = text.clone().into_bytes();
        bytes[text.find("\"token_str\":\"A\"").unwrap() + 13] = 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(read(&path).unwrap().ranks, whole);

        let mut bytes = text.clone().into_bytes();
        bytes[text.find("\"token_bytes\":\"QQ==\"").unwrap() + 15] = 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let Err(error) = read(&path) else {
            panic!("a token whose bytes are not UTF-8 loaded");
        };
        assert!(
            format!("{error:#}").contains("is not a tekken vocabulary file"),
            "{error:#}"
        );
    }

    /// The walk over the ranked tokens reads the vocabulary the JSON reader
    /// reads; a token written any other way sends the file to the JSON
    /// reader, which reads or refuses it as before.
    #[test]
    fn the_walk_reads_the_ranked_tokens_as_the_json_reader_does() {
        let compact = testing::minimal();
        let edit = |from: &str, to: &str| {
            let text = compact.replacen(from, to, 1);
            assert_ne!(text, compact, "{from} is not in the file");
            text
        };
        let value: serde_json::Value = serde_json::from_str(&compact).unwrap();
        // A file, and whether the walk takes it.
        let files = [
            (compact.clone(), true),
            (serde_json::to_string_pretty(&value).unwrap(), true),
            (edit(r#""\u0007""#, r#""\ud83d\ude00""#), true),
            (edit(r#""\u0007""#, r#""\ud83d""#), false),
            (edit(r#""\u0007""#, "\"\u{7}\""), false),
            (
                edit(
                    r#""rank":7,"token_bytes":"Bw==""#,
                    r#""token_bytes":"Bw==","rank":7"#,
                ),
                false,
            ),
            (edit(r#""rank":7,"#, r#""rank":07,"#), false),
            (edit(r#""rank":7,"#, r#""rank":7.0,"#), false),
            (edit(r#""rank":7,"#, r#""rank":4294967296,"#), false),
            (edit(r#""Bw==""#, r#""Bw=\u003d""#), false),
            (edit(r#""hello"}]"#, r#""hello"},]"#), false),
        ];
        for (row, (text, walked)) in files.into_iter().enumerate() {
            assert_eq!(written_layout(&text).is_some(), walked, "row {row}");
            let loaded = read(&testing::dir_with(&text).path().join(FILE_NAME));
            match (loaded, serde_json::from_str::<File>(&text)) {
                (Ok(loaded), Ok(file)) => {
                    let parsed = vocabulary(file).unwrap();
                    assert_eq!(loaded.ranks, parsed.ranks, "row {row}");
                    assert_eq!(loaded.specials, parsed.specials, "row {row}");
                    assert_eq!(loaded.pattern, parsed.pattern, "row {row}");
                }
                (Err(loaded), Err(parsed)) => {
                    let loaded = format!("{loaded:#}");
                    assert!(loaded.ends_with(&parsed.to_string()), "row {row}: {loaded}");
                }
                (loaded, _) => panic!("row {row}: one reader refuses (read: {})", loaded.is_ok()),
            }
        }
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
        refused(
            testing::tekken_json(
                &testing::MERGES,
                Some(&[(0, "<unk>")]),
                MAX_SPECIAL_TOKENS + 1,
                MAX_SPECIAL_TOKENS + 261,
                "v13",
            ),
            "default_num_special_tokens 65537 is above the bound of 65536",
        );
        // `hello` rewritten as `he`, a token the file lists already: the
        // reader passes it; the load refuses it and names the file as the
        // reader names it.
        let dir = testing::dir_with(&minimal.replacen("aGVsbG8=", "aGU=", 1));
        let path = dir.path().join(FILE_NAME);
        let Err(error) = TiktokenTokenizer::from_tekken_file(&path) else {
            panic!("a file that lists a token twice loaded");
        };
        let error = format!("{error:#}");
        assert!(error.contains(&path.display().to_string()), "{error}");
        assert!(
            error.contains("repeats an earlier token's bytes"),
            "{error}"
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
