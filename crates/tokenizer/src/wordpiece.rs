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
//! first, in the model's own vocabulary. The encoder holds no table of the
//! pieces, so a tokenizer loads as fast and as small as before.
//!
//! It takes the configuration it reproduces (a `BertNormalizer` with any
//! flags, a `BertPreTokenizer`, a `WordPiece` model, a template
//! post-processor or none, added tokens cut from the raw text as they stand)
//! and declines every other with a reason; a declined tokenizer encodes in
//! `tokenizers` as before. The character classes are the library's own (see
//! `classes`), and the decomposition is the crate the library calls.

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
    native::AddedTokens,
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
/// library's control characters (its `is_other`, the three it counts as
/// whitespace apart).
fn removed(c: char) -> bool {
    c == '\0'
        || c == '\u{fffd}'
        || (!matches!(c, '\t' | '\n' | '\r') && in_class(classes::OTHER, c))
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
    c.is_ascii_punctuation() || in_class(classes::PUNCTUATION, c)
}

/// The table of the ASCII bytes under `rules`: each byte through the rules
/// a character beyond ASCII runs through, once, at load.
fn ascii_table(rules: Rules) -> [Ascii; 128] {
    let mut table = [Ascii::Removed; 128];
    for (byte, entry) in (0u8..).zip(table.iter_mut()) {
        let c = char::from(byte);
        *entry = if rules.clean_text && removed(c) {
            Ascii::Removed
        } else if c.is_whitespace() {
            Ascii::Space
        } else if c.is_ascii_punctuation() {
            Ascii::Punctuation
        } else if rules.lowercase {
            Ascii::Letter(byte.to_ascii_lowercase())
        } else {
            Ascii::Letter(byte)
        };
    }
    table
}

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
    /// Whether the tokenizer has added tokens: an encode then needs their
    /// matcher.
    has_added: bool,
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
        let has_added = Self::added_tokens(tokenizer)?;
        let Some(unknown) = model.token_to_id(&model.unk_token) else {
            return Err(Declined::from("the unknown token is not in the vocabulary"));
        };
        Ok(Self {
            rules,
            ascii: ascii_table(rules),
            marker: model.continuing_subword_prefix.as_str().into(),
            unknown,
            max_chars: model.max_input_chars_per_word,
            has_added,
            prefix,
            suffix,
        })
    }

    /// The ids the post-processor puts before and after one sequence: the
    /// library frames one token no vocabulary holds, and what stands around
    /// it is the frame.
    fn frame<M: Model>(tokenizer: &TokenizerOf<M>) -> Result<(Vec<u32>, Vec<u32>), Declined> {
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

    /// Whether the tokenizer has added tokens. A token the library does not
    /// cut from the raw text as it stands (one it matches after
    /// normalization, at word edges only, or with the whitespace beside it)
    /// declines. The matcher over them is the wrapper's, built at its first
    /// use.
    fn added_tokens<M: Model>(tokenizer: &TokenizerOf<M>) -> Result<bool, Declined> {
        let vocabulary = tokenizer.get_added_vocabulary();
        if vocabulary.get_encode_special_tokens() {
            return Err(Declined::from("encode_special_tokens is set"));
        }
        let tokens = vocabulary.get_added_tokens_decoder();
        for token in tokens.values() {
            if token.single_word
                || token.lstrip
                || token.rstrip
                || token.normalized
                || token.content.is_empty()
            {
                return Err(Declined::from(format!(
                    "added token {:?} is not cut from the raw text as it stands",
                    token.content
                )));
            }
        }
        Ok(!tokens.is_empty())
    }

    /// The ids of `text`, framed by the post-processor's tokens on request.
    /// `added` is the wrapper's matcher over the added tokens of `tokenizer`
    /// (built at its first use, one for every caller). `None` when
    /// `tokenizer` is not the one this encoder was made from or the matcher
    /// it needs is missing: the caller then encodes in `tokenizers`.
    pub(crate) fn encode<M: Model + ModelView>(
        &self,
        tokenizer: &TokenizerOf<M>,
        added: Option<&AddedTokens>,
        text: &str,
        add_special_tokens: bool,
    ) -> Option<Vec<u32>> {
        let ModelWrapper::WordPiece(model) = tokenizer.get_model().wrapper() else {
            return None;
        };
        if self.has_added && added.is_none() {
            return None;
        }
        let mut ids =
            Vec::with_capacity(self.prefix.len() + text.len() / 4 + self.suffix.len() + 4);
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
        // A run of characters beyond ASCII, as the decomposition reads it.
        let mut segment = String::new();
        match added {
            None => self.span(text, &mut segment, &mut run),
            Some(added) => {
                let mut from = 0;
                for range in added.ranges(text) {
                    self.span(&text[from..range.start], &mut segment, &mut run);
                    run.ids.push(tokenizer.token_to_id(&text[range.clone()])?);
                    from = range.end;
                }
                self.span(&text[from..], &mut segment, &mut run);
            }
        }
        if add_special_tokens {
            run.ids.extend_from_slice(&self.suffix);
        }
        Some(run.ids)
    }

    /// The ids of a span between added tokens. A word ends with the span.
    fn span(&self, text: &str, segment: &mut String, run: &mut Run<'_>) {
        let bytes = text.as_bytes();
        let mut at = 0;
        while let Some(&byte) = bytes.get(at) {
            if byte < 0x80 {
                match self.ascii[usize::from(byte)] {
                    Ascii::Letter(letter) => run.word.push(char::from(letter)),
                    Ascii::Space => self.end_word(run),
                    Ascii::Punctuation => {
                        self.end_word(run);
                        run.word.push(char::from(byte));
                        self.end_word(run);
                    }
                    Ascii::Removed => {}
                }
                at += 1;
            } else {
                at = self.beyond_ascii(text, at, segment, run);
            }
        }
        self.end_word(run);
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
