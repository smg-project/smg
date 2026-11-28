//! The crate's direct encoder for WordPiece tokenizers of the common shape.
//!
//! `tokenizers` encodes such a tokenizer in passes over owned strings: its
//! normalizer rewrites the whole text once per rule (clean, CJK spacing,
//! decomposition, mark strip, lowercase), each pass with an alignment per
//! byte; its pre-tokenizer cuts the result into words that own their text;
//! its model builds a `String` for every candidate piece it looks up. This
//! encoder reads the text once. A byte of ASCII is answered by one table; a
//! character beyond ASCII runs the same rules in the same order on its own
//! run of characters; a finished word is matched greedily, longest piece
//! first, in the model's own vocabulary. An added token is cut in the same
//! pass: a byte that starts one is a class of the table, and the tokens are
//! tried where it stands. The encoder holds no table of the pieces and
//! builds no matcher, so a tokenizer loads as fast and as small as before,
//! and no input leaves it for `tokenizers`.
//!
//! It takes the configuration it reproduces (a `BertNormalizer` with any
//! flags, a `BertPreTokenizer`, a `WordPiece` model, a template
//! post-processor or none, at most `MAX_ADDED` added tokens, each cut from
//! the raw text as it stands and starting with a byte of ASCII the
//! normalizer keeps) and declines every other with a reason; a declined
//! tokenizer encodes in `tokenizers` as before. The character classes are
//! the library's own (see `classes`), and the decomposition is the crate
//! the library calls.

mod classes;

use std::cmp::Ordering;

use serde_json::Value;
use tokenizers::{
    models::{wordpiece::WordPiece, ModelWrapper},
    normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper,
    tokenizer::{Encoding, Model, Token},
};
use unicode_normalization_alignments::UnicodeNormalization;

use crate::{
    encode_path::Declined,
    word_cache::{ModelView, TokenizerOf},
};

/// The normalizer's rules, as `BertNormalizer` carries them.
#[derive(Clone, Copy)]
struct Rules {
    /// Control characters go and every whitespace is a space.
    clean_text: bool,
    /// A CJK ideograph is a word of its own.
    handle_chinese_chars: bool,
    /// The text is decomposed and its nonspacing marks go.
    strip_accents: bool,
    /// Every character is lowercased.
    lowercase: bool,
}

/// What a byte of ASCII does to the word under construction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ascii {
    /// It joins the word as this byte (its lowercase where the rules say so).
    Letter(u8),
    /// It ends the word and leaves nothing.
    Space,
    /// It ends the word and is a word of its own.
    Punctuation,
    /// The normalizer removes it.
    Removed,
    /// An added token starts with it: the tokens are tried where it stands,
    /// and where none does the byte is what the rules alone make of it.
    Added,
}

/// Whether `c` is in `class`, a sorted list of inclusive ranges.
fn in_class(class: &[(u32, u32)], c: char) -> bool {
    let c = u32::from(c);
    class
        .binary_search_by(|&(first, last)| {
            if last < c {
                Ordering::Less
            } else if first > c {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// What `clean_text` removes: NUL, the replacement character, and the
/// library's control characters (the three it counts as whitespace
/// apart).
fn removed(c: char) -> bool {
    in_class(classes::REMOVED, c)
}

/// The ideographs `handle_chinese_chars` sets apart (the library's ranges).
fn is_cjk(c: char) -> bool {
    matches!(
        u32::from(c),
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// What the pre-tokenizer sets apart as a word of its own.
fn is_punctuation(c: char) -> bool {
    in_class(classes::PUNCTUATION, c)
}

/// What `rules` make of one byte of ASCII: the rules a character beyond
/// ASCII runs through.
fn ascii_class(rules: Rules, byte: u8) -> Ascii {
    if rules.clean_text && removed(char::from(byte)) {
        Ascii::Removed
    } else {
        kept_class(rules, byte)
    }
}

/// What `rules` make of a byte of ASCII the normalizer keeps.
fn kept_class(rules: Rules, byte: u8) -> Ascii {
    let c = char::from(byte);
    if c.is_whitespace() {
        Ascii::Space
    } else if c.is_ascii_punctuation() {
        Ascii::Punctuation
    } else if rules.lowercase {
        Ascii::Letter(byte.to_ascii_lowercase())
    } else {
        Ascii::Letter(byte)
    }
}

/// The table of the ASCII bytes under `rules`, made once, at load: each
/// byte by its class, and the bytes that start a token of `added` set
/// apart.
fn ascii_table(rules: Rules, added: &[(Box<str>, u32)]) -> [Ascii; 128] {
    let mut table = [Ascii::Removed; 128];
    for (byte, entry) in (0u8..).zip(table.iter_mut()) {
        *entry = ascii_class(rules, byte);
    }
    for (token, _) in added {
        let first = token.as_bytes().first().map(|&first| usize::from(first));
        if let Some(entry) = first.and_then(|first| table.get_mut(first)) {
            *entry = Ascii::Added;
        }
    }
    table
}

/// The most added tokens the encoder takes. They are tried in turn where a
/// byte that starts one stands, so the set is kept to what a tokenizer of
/// this shape carries (its special tokens); a larger set stays with
/// `tokenizers`, whose matcher is built for any number.
const MAX_ADDED: usize = 32;

/// The word under construction: its normalized text and its length in
/// characters. Past the model's bound only the count grows (such a word is
/// the unknown token whatever it holds), so the text stays bounded.
struct Word {
    text: String,
    chars: usize,
    bound: usize,
}

impl Word {
    fn new(bound: usize) -> Self {
        Self {
            text: String::with_capacity(bound.min(64) * 4),
            chars: 0,
            bound,
        }
    }

    fn push(&mut self, c: char) {
        self.chars += 1;
        if self.chars <= self.bound {
            self.text.push(c);
        }
    }

    fn clear(&mut self) {
        self.text.clear();
        self.chars = 0;
    }
}

/// What one encode works on: the model whose vocabulary answers the pieces,
/// the word in hand and the ids so far.
struct Run<'a> {
    model: &'a WordPiece,
    word: Word,
    /// The model's continuation prefix, and behind it the rest of the word
    /// in hand: what a piece after the first is looked up in.
    probe: String,
    ids: Vec<u32>,
}

/// The direct encoder of one WordPiece tokenizer.
pub(crate) struct WordPieceEncoder {
    rules: Rules,
    ascii: [Ascii; 128],
    /// The model's continuation prefix.
    marker: Box<str>,
    unknown: u32,
    max_chars: usize,
    /// The added tokens with their ids, longest first: the first of them
    /// that stands at a place is the longest there.
    added: Vec<(Box<str>, u32)>,
    /// The ids the post-processor puts before and after one sequence.
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

impl WordPieceEncoder {
    /// The encoder of `tokenizer`, or why it is not one this path reproduces.
    pub(crate) fn from_tokenizer<M: Model + ModelView>(
        tokenizer: &TokenizerOf<M>,
    ) -> Result<Self, Declined> {
        let ModelWrapper::WordPiece(model) = tokenizer.get_model().wrapper() else {
            return Err(Declined::from("the model is not WordPiece"));
        };
        if tokenizer.get_truncation().is_some() {
            return Err(Declined::from("truncation is configured"));
        }
        if tokenizer.get_padding().is_some() {
            return Err(Declined::from("padding is configured"));
        }
        let Some(NormalizerWrapper::BertNormalizer(normalizer)) = tokenizer.get_normalizer() else {
            return Err(Declined::from("the normalizer is not a BertNormalizer"));
        };
        let Some(PreTokenizerWrapper::BertPreTokenizer(_)) = tokenizer.get_pre_tokenizer() else {
            return Err(Declined::from(
                "the pre-tokenizer is not a BertPreTokenizer",
            ));
        };
        let rules = Rules {
            clean_text: normalizer.clean_text,
            handle_chinese_chars: normalizer.handle_chinese_chars,
            strip_accents: normalizer.strip_accents.unwrap_or(normalizer.lowercase),
            lowercase: normalizer.lowercase,
        };
        let (prefix, suffix) = Self::frame(tokenizer)?;
        let added = Self::added_tokens(tokenizer, rules)?;
        let Some(unknown) = model.token_to_id(&model.unk_token) else {
            return Err(Declined::from("the unknown token is not in the vocabulary"));
        };
        Ok(Self {
            rules,
            ascii: ascii_table(rules, &added),
            marker: model.continuing_subword_prefix.as_str().into(),
            unknown,
            max_chars: model.max_input_chars_per_word,
            added,
            prefix,
            suffix,
        })
    }

    /// The ids the post-processor puts before and after one sequence: the
    /// library frames one token no vocabulary holds, and what stands around
    /// it is the frame. The direct Unigram encoder reads its frame here
    /// too.
    pub(crate) fn frame<M: Model>(
        tokenizer: &TokenizerOf<M>,
    ) -> Result<(Vec<u32>, Vec<u32>), Declined> {
        const PROBE: u32 = u32::MAX;
        let Some(post_processor) = tokenizer.get_post_processor() else {
            return Ok((Vec::new(), Vec::new()));
        };
        let kind = serde_json::to_value(post_processor)
            .map_err(|e| Declined::from(format!("the post-processor does not serialize: {e}")))?;
        if !matches!(
            kind.get("type").and_then(Value::as_str),
            Some("TemplateProcessing" | "BertProcessing")
        ) {
            return Err(Declined::from(
                "the post-processor is neither a TemplateProcessing nor a BertProcessing",
            ));
        }
        let probe = Encoding::from_tokens(vec![Token::new(PROBE, String::new(), (0, 0))], 0);
        let framed = tokenizer
            .post_process(probe, None, true)
            .map_err(|e| Declined::from(format!("the post-processor frames no sequence: {e}")))?;
        let ids = framed.get_ids();
        let mut probes = ids.iter().enumerate().filter(|&(_, &id)| id == PROBE);
        match (probes.next(), probes.next()) {
            (Some((at, _)), None) => Ok((ids[..at].to_vec(), ids[at + 1..].to_vec())),
            _ => Err(Declined::from(
                "the post-processor does not frame one sequence exactly once",
            )),
        }
    }

    /// The added tokens with their ids, longest first. A token the library
    /// does not cut from the raw text as it stands (one it matches after
    /// normalization, at word edges only, or with the whitespace beside it)
    /// declines. So does a set the pass does not cut: a token that starts
    /// beyond ASCII or with a byte the normalizer removes (the pass tries
    /// the tokens at the bytes of ASCII it keeps), or more than
    /// [`MAX_ADDED`] tokens.
    fn added_tokens<M: Model>(
        tokenizer: &TokenizerOf<M>,
        rules: Rules,
    ) -> Result<Vec<(Box<str>, u32)>, Declined> {
        let vocabulary = tokenizer.get_added_vocabulary();
        if vocabulary.get_encode_special_tokens() {
            return Err(Declined::from("encode_special_tokens is set"));
        }
        // The library builds its matcher over the tokens of this map, each
        // with its key as the id.
        let tokens = vocabulary.get_added_tokens_decoder();
        if tokens.len() > MAX_ADDED {
            return Err(Declined::from(format!(
                "there are more than {MAX_ADDED} added tokens"
            )));
        }
        let mut added: Vec<(Box<str>, u32)> = Vec::with_capacity(tokens.len());
        for (&id, token) in tokens {
            if token.single_word || token.lstrip || token.rstrip || token.normalized {
                return Err(Declined::from(format!(
                    "added token {:?} is not cut from the raw text as it stands",
                    token.content
                )));
            }
            let kept = token.content.bytes().next().is_some_and(|first| {
                first.is_ascii() && ascii_class(rules, first) != Ascii::Removed
            });
            if !kept {
                return Err(Declined::from(format!(
                    "added token {:?} does not start with a byte of ASCII the normalizer keeps",
                    token.content
                )));
            }
            added.push((token.content.as_str().into(), id));
        }
        // Longest first. At most one of two tokens of one length stands at
        // a place, so their order is free; their text makes it fixed.
        added.sort_unstable_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        Ok(added)
    }

    /// The ids of `text`, framed by the post-processor's tokens on request.
    /// `model` is the model of the tokenizer this encoder was made from: its
    /// vocabulary answers the pieces.
    pub(crate) fn encode(
        &self,
        model: &WordPiece,
        text: &str,
        add_special_tokens: bool,
    ) -> Vec<u32> {
        // A third of the bytes: text of words runs near one piece to four
        // bytes, and a list that grows once is copied and twice as large.
        let mut ids =
            Vec::with_capacity(self.prefix.len() + text.len() / 3 + self.suffix.len() + 4);
        if add_special_tokens {
            ids.extend_from_slice(&self.prefix);
        }
        let mut probe = String::with_capacity(self.marker.len() + 64);
        probe.push_str(&self.marker);
        let mut run = Run {
            model,
            word: Word::new(self.max_chars),
            probe,
            ids,
        };
        self.pass(text, &mut run);
        if add_special_tokens {
            run.ids.extend_from_slice(&self.suffix);
        }
        run.ids
    }

    /// The one pass over `text`: a byte of ASCII by its class, a run beyond
    /// ASCII by the rules, an added token as its id. A word ends at an added
    /// token and with the text: the library cuts the text at the tokens
    /// before it reads what stands between them.
    fn pass(&self, text: &str, run: &mut Run<'_>) {
        // A run of characters beyond ASCII, as the decomposition reads it.
        let mut segment = String::new();
        let bytes = text.as_bytes();
        let mut at = 0;
        while let Some(&byte) = bytes.get(at) {
            if byte >= 0x80 {
                at = self.beyond_ascii(text, at, &mut segment, run);
                continue;
            }
            match self.ascii[usize::from(byte)] {
                Ascii::Letter(letter) => run.word.push(char::from(letter)),
                Ascii::Space => self.end_word(run),
                Ascii::Punctuation => self.punctuation(byte, run),
                Ascii::Removed => {}
                Ascii::Added => {
                    at = self.added_or_byte(bytes, at, run);
                    continue;
                }
            }
            at += 1;
        }
        self.end_word(run);
    }

    /// A punctuation character of ASCII: a word of its own.
    fn punctuation(&self, byte: u8, run: &mut Run<'_>) {
        self.end_word(run);
        run.word.push(char::from(byte));
        self.end_word(run);
    }

    /// At `at`, a byte that starts an added token: the token that stands
    /// there as its id, a word ending before it, or else the byte as what
    /// the rules alone make of it. Returns where the next read starts. Kept
    /// out of the pass: such a byte is rare in text.
    #[cold]
    fn added_or_byte(&self, bytes: &[u8], at: usize, run: &mut Run<'_>) -> usize {
        let rest = bytes.get(at..).unwrap_or_default();
        if let Some((len, id)) = self.added_at(rest) {
            self.end_word(run);
            run.ids.push(id);
            return at + len;
        }
        if let Some(&byte) = rest.first() {
            // A byte that starts a token is one the normalizer keeps (the
            // load declined any other).
            match kept_class(self.rules, byte) {
                Ascii::Letter(letter) => run.word.push(char::from(letter)),
                Ascii::Space => self.end_word(run),
                Ascii::Punctuation => self.punctuation(byte, run),
                Ascii::Removed | Ascii::Added => {}
            }
        }
        at + 1
    }

    /// The added token that stands at the start of `rest`, the longest one:
    /// its length and its id. The library matches the tokens leftmost first
    /// and longest first; the pass asks at every byte that starts one, from
    /// the left, and the list is longest first.
    fn added_at(&self, rest: &[u8]) -> Option<(usize, u32)> {
        // Compared byte by byte: a start that no token finishes differs at
        // its second byte, sooner than a call to compare would return.
        self.added.iter().find_map(|(token, id)| {
            let token = token.as_bytes();
            let stands = token.len() <= rest.len() && token.iter().zip(rest).all(|(a, b)| a == b);
            stands.then_some((token.len(), *id))
        })
    }

    /// Reads from `at`, where a character beyond ASCII starts, what the
    /// rules read as one: a removed character, a whitespace, an ideograph,
    /// or a run of other characters, which the decomposition must see whole
    /// (it sorts the marks behind a letter; a removed character between two
    /// marks does not part them, a kept ASCII byte does). Returns where the
    /// next read starts.
    fn beyond_ascii(
        &self,
        text: &str,
        at: usize,
        segment: &mut String,
        run: &mut Run<'_>,
    ) -> usize {
        let Some(c) = text[at..].chars().next() else {
            return text.len();
        };
        let next = at + c.len_utf8();
        if self.rules.clean_text {
            if removed(c) {
                return next;
            }
            if c.is_whitespace() {
                self.end_word(run);
                return next;
            }
        }
        if self.rules.handle_chinese_chars && is_cjk(c) {
            self.end_word(run);
            let mut utf8 = [0; 4];
            self.normalized(c.encode_utf8(&mut utf8), run);
            self.end_word(run);
            return next;
        }
        segment.clear();
        segment.push(c);
        let bytes = text.as_bytes();
        let mut end = next;
        while let Some(&byte) = bytes.get(end) {
            if byte < 0x80 {
                if self.ascii[usize::from(byte)] == Ascii::Removed {
                    end += 1;
                    continue;
                }
                break;
            }
            let Some(c) = text[end..].chars().next() else {
                break;
            };
            if self.rules.clean_text && removed(c) {
                end += c.len_utf8();
                continue;
            }
            if (self.rules.clean_text && c.is_whitespace())
                || (self.rules.handle_chinese_chars && is_cjk(c))
            {
                break;
            }
            segment.push(c);
            end += c.len_utf8();
        }
        self.normalized(segment, run);
        end
    }

    /// The rest of the normalizer on a cleaned run (decomposition and mark
    /// strip, then lowercase), each character it leaves handed to the cut.
    fn normalized(&self, segment: &str, run: &mut Run<'_>) {
        if self.rules.strip_accents {
            for (c, _) in segment.nfd() {
                if !in_class(classes::NONSPACING_MARK, c) {
                    self.lowercased(c, run);
                }
            }
        } else {
            for c in segment.chars() {
                self.lowercased(c, run);
            }
        }
    }

    fn lowercased(&self, c: char, run: &mut Run<'_>) {
        if self.rules.lowercase {
            for lower in c.to_lowercase() {
                self.cut(lower, run);
            }
        } else {
            self.cut(c, run);
        }
    }

    /// The pre-tokenizer on one normalized character: whitespace ends the
    /// word, punctuation is a word of its own, anything else joins the word.
    fn cut(&self, c: char, run: &mut Run<'_>) {
        if c.is_whitespace() {
            self.end_word(run);
        } else if is_punctuation(c) {
            self.end_word(run);
            run.word.push(c);
            self.end_word(run);
        } else {
            run.word.push(c);
        }
    }

    /// Ends the word in hand: its pieces, or the unknown token for a word
    /// over the model's bound.
    fn end_word(&self, run: &mut Run<'_>) {
        if run.word.chars == 0 {
            return;
        }
        if run.word.chars > self.max_chars {
            run.ids.push(self.unknown);
        } else {
            self.pieces(run);
        }
        run.word.clear();
    }

    /// The model on the word in hand: the longest piece that starts the
    /// word, then the longest piece that continues it (looked up behind the
    /// continuation prefix), to the word's end; a rest no piece starts makes
    /// the whole word the unknown token. Every lookup is the library's own.
    fn pieces(&self, run: &mut Run<'_>) {
        let Run {
            model,
            word,
            probe,
            ids,
        } = run;
        let word = word.text.as_str();
        let whole = ids.len();
        let mut start = 0;
        while start < word.len() {
            let rest = &word[start..];
            let (text, lead) = if start == 0 {
                (rest, 0)
            } else {
                probe.truncate(self.marker.len());
                probe.push_str(rest);
                (probe.as_str(), self.marker.len())
            };
            let mut len = rest.len();
            let found = loop {
                if let Some(id) = model.token_to_id(&text[..lead + len]) {
                    break Some(id);
                }
                len -= 1;
                while !rest.is_char_boundary(len) {
                    len -= 1;
                }
                if len == 0 {
                    break None;
                }
            };
            let Some(id) = found else {
                ids.truncate(whole);
                ids.push(self.unknown);
                return;
            };
            ids.push(id);
            start += len;
        }
    }
}

#[cfg(test)]
mod tests;
