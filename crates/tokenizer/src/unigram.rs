//! The crate's direct encoder for Unigram tokenizers of the common shape.
//!
//! `tokenizers` runs such a tokenizer in passes over owned strings with an
//! alignment per byte (the precompiled character map, the whitespace split,
//! the word marker), and its model walks a trie of hash maps that hands
//! back an owned string for every piece it finds. This encoder does the
//! same steps on plain strings and finds the best path of each word in one
//! pass over its bytes: it asks the model's own map for every text that may
//! be a piece, and keeps the library's rule for paths of equal score and
//! for characters no piece covers. The character map is applied by the
//! library's own string function, to the stretches of a text it can change.
//!
//! It takes a `Precompiled` normalizer alone, a `WhitespaceSplit` followed
//! by a `Metaspace` that marks every word and splits at the marker, a
//! `Unigram` model with an unknown piece, without sampling and without the
//! byte fallback, a template post-processor or none, and added tokens cut
//! from the raw text as they stand; it declines every other with a reason.

use tokenizers::{
    models::{unigram::Unigram, ModelWrapper},
    normalizers::{NormalizerWrapper, Precompiled},
    pre_tokenizers::{metaspace::PrependScheme, PreTokenizerWrapper},
    tokenizer::Model,
};

use crate::{
    encode_path::Declined,
    native::AddedTokens,
    word_cache::{ModelView, TokenizerOf},
    wordpiece::WordPieceEncoder,
};

/// No piece, no path.
const NONE: u32 = u32::MAX;
/// What the library takes from the lowest score for a character no piece
/// covers.
const UNKNOWN_PENALTY: f64 = 10.0;

/// The best path that ends at one byte of a word: its score, where its
/// last piece starts ([`NONE`]: no path ends here) and that piece's id.
#[derive(Clone, Copy)]
struct Best {
    score: f64,
    start: u32,
    id: u32,
}

/// The direct encoder of one Unigram tokenizer.
pub(crate) struct UnigramEncoder {
    /// The word marker of the `Metaspace` pre-tokenizer.
    marker: char,
    /// Whether the tokenizer has added tokens: an encode then needs their
    /// matcher.
    has_added: bool,
    /// The ids the post-processor puts before and after one sequence.
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    /// The score of each piece, by id.
    score: Vec<f64>,
    /// The longest piece, in bytes: no text longer than it is one.
    longest: usize,
    /// The id and the score of a character no piece covers.
    unknown: u32,
    unknown_score: f64,
    /// Which bytes of ASCII the character map leaves as they are.
    kept: [bool; 128],
}

impl UnigramEncoder {
    /// The encoder of `tokenizer`, or why it is not one this path
    /// reproduces.
    pub(crate) fn from_tokenizer<M: Model + ModelView>(
        tokenizer: &TokenizerOf<M>,
    ) -> Result<Self, Declined> {
        let ModelWrapper::Unigram(model) = tokenizer.get_model().wrapper() else {
            return Err(Declined::from("the model is not Unigram"));
        };
        if tokenizer.get_truncation().is_some() {
            return Err(Declined::from("truncation is configured"));
        }
        if tokenizer.get_padding().is_some() {
            return Err(Declined::from("padding is configured"));
        }
        let Some(NormalizerWrapper::Precompiled(normalizer)) = tokenizer.get_normalizer() else {
            return Err(Declined::from(
                "the normalizer is not a precompiled character map alone",
            ));
        };
        if model.byte_fallback() {
            return Err(Declined::from("the model falls back to bytes"));
        }
        if model.alpha.is_some_and(|alpha| alpha != 0.0) {
            return Err(Declined::from("the model samples its paths"));
        }
        let marker = Self::marker(tokenizer)?;
        let (prefix, suffix) = WordPieceEncoder::frame(tokenizer)?;
        let has_added = Self::added_tokens(tokenizer)?;
        let mut score = Vec::with_capacity(model.get_vocab_size());
        let mut longest = 0;
        for (piece, piece_score) in model.iter() {
            score.push(*piece_score);
            longest = longest.max(piece.len());
        }
        Ok(Self {
            marker,
            has_added,
            prefix,
            suffix,
            unknown: Self::unknown(model)?,
            unknown_score: model.min_score - UNKNOWN_PENALTY,
            score,
            longest,
            kept: Self::kept(normalizer),
        })
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

    /// Which bytes of ASCII the character map leaves as they are: the map
    /// is asked for each. A carriage return is never one: with a line feed
    /// after it, it is one cluster.
    fn kept(map: &Precompiled) -> [bool; 128] {
        let mut kept = [false; 128];
        for (byte, kept) in (0u8..).zip(kept.iter_mut()) {
            let c = char::from(byte);
            let mut buffer = [0; 4];
            let text: &str = c.encode_utf8(&mut buffer);
            *kept = c != '\r' && map.transform(text).is_none_or(|to| to == text);
        }
        kept
    }

    /// `text` as the character map leaves it. The map works one cluster at
    /// a time, and between two bytes of ASCII there is always a cluster
    /// boundary (a carriage return before a line feed apart). So a run of
    /// bytes the map keeps is copied as it is, and the map reads the rest;
    /// a kept byte beside a character beyond ASCII goes to the map with
    /// that character, for the two may be one cluster.
    fn normalized<'a>(
        &self,
        map: impl Fn(&str) -> String,
        text: &'a str,
        out: &'a mut String,
    ) -> &'a str {
        let bytes = text.as_bytes();
        let fast = |at: usize| bytes[at] < 0x80 && self.kept[usize::from(bytes[at])];
        if (0..bytes.len()).all(fast) {
            return text;
        }
        out.clear();
        let mut done = 0;
        let mut at = 0;
        while at < bytes.len() {
            if !fast(at) {
                at += 1;
                continue;
            }
            let mut start = at;
            while at < bytes.len() && fast(at) {
                at += 1;
            }
            let mut end = at;
            if start > 0 && (bytes[start - 1] >= 0x80 || bytes[start - 1] == b'\r') {
                start += 1;
            }
            if end < bytes.len() && bytes[end] >= 0x80 {
                end -= 1;
            }
            if start < end {
                if done < start {
                    out.push_str(&map(&text[done..start]));
                }
                out.push_str(&text[start..end]);
                done = end;
            }
        }
        if done < bytes.len() {
            out.push_str(&map(&text[done..]));
        }
        out
    }

    /// The id the model gives a character no piece covers: the model is
    /// asked for a character that is not a piece.
    fn unknown(model: &Unigram) -> Result<u32, Declined> {
        const PROBE: &str = "\u{10FFFE}";
        if model.token_to_id(PROBE).is_none() {
            if let Ok([token]) = model.tokenize(PROBE).as_deref() {
                return Ok(token.id);
            }
        }
        Err(Declined::from("the model names no unknown piece"))
    }

    /// The word marker, when the pre-tokenizer is a whitespace split and
    /// then a `Metaspace` that marks every word and splits at the marker.
    fn marker<M: Model>(tokenizer: &TokenizerOf<M>) -> Result<char, Declined> {
        let declined = || {
            Declined::from("the pre-tokenizer is not a whitespace split and then a Metaspace that marks every word")
        };
        let Some(PreTokenizerWrapper::Sequence(steps)) = tokenizer.get_pre_tokenizer() else {
            return Err(declined());
        };
        let [PreTokenizerWrapper::WhitespaceSplit(_), PreTokenizerWrapper::Metaspace(metaspace)] =
            steps.as_ref()
        else {
            return Err(declined());
        };
        if !matches!(metaspace.get_prepend_scheme(), PrependScheme::Always)
            || !metaspace.get_split()
        {
            return Err(declined());
        }
        Ok(metaspace.get_replacement())
    }

    /// The ids of `text`, framed by the post-processor's tokens on request.
    /// `added` is the wrapper's matcher over the added tokens of `tokenizer`.
    /// `None` when `tokenizer` is not the one this encoder was made from or
    /// the matcher it needs is missing: the caller then encodes in
    /// `tokenizers`.
    pub(crate) fn encode<M: Model + ModelView>(
        &self,
        tokenizer: &TokenizerOf<M>,
        added: Option<&AddedTokens>,
        text: &str,
        add_special_tokens: bool,
    ) -> Option<Vec<u32>> {
        let Some(NormalizerWrapper::Precompiled(normalizer)) = tokenizer.get_normalizer() else {
            return None;
        };
        if self.has_added && added.is_none() {
            return None;
        }
        let ModelWrapper::Unigram(model) = tokenizer.get_model().wrapper() else {
            return None;
        };
        let mut ids =
            Vec::with_capacity(self.prefix.len() + text.len() / 3 + self.suffix.len() + 4);
        if add_special_tokens {
            ids.extend_from_slice(&self.prefix);
        }
        let mut piece = String::new();
        let mut best = Vec::new();
        let mut mapped = String::new();
        let mut span = |text: &str, ids: &mut Vec<u32>| {
            let normalized =
                self.normalized(|part| normalizer.normalize_string(part), text, &mut mapped);
            self.words(model, normalized, &mut piece, &mut best, ids);
        };
        let mut from = 0;
        if let Some(added) = added {
            for range in added.ranges(text) {
                span(&text[from..range.start], &mut ids);
                ids.push(tokenizer.token_to_id(&text[range.clone()])?);
                from = range.end;
            }
        }
        span(&text[from..], &mut ids);
        if add_special_tokens {
            ids.extend_from_slice(&self.suffix);
        }
        Some(ids)
    }

    /// The pre-tokenizer and the model on a normalized span: a word ends at
    /// whitespace; the marker goes before a word that does not start with
    /// it; a piece starts at every marker; each piece gets its best path.
    fn words(
        &self,
        model: &Unigram,
        normalized: &str,
        piece: &mut String,
        best: &mut Vec<Best>,
        ids: &mut Vec<u32>,
    ) {
        for word in normalized.split(char::is_whitespace) {
            let mut rest = word;
            let mut marked = word.starts_with(self.marker);
            while !rest.is_empty() {
                piece.clear();
                piece.push(self.marker);
                if marked {
                    rest = &rest[self.marker.len_utf8()..];
                }
                marked = true;
                let end = rest.find(self.marker).unwrap_or(rest.len());
                piece.push_str(&rest[..end]);
                rest = &rest[end..];
                self.best_path(model, piece, best, ids);
            }
        }
    }

    /// The ids of the best path over `word`, as the library's model finds
    /// it: from every character on, each piece that starts there offers the
    /// byte after it a path, and a byte keeps the first path of the highest
    /// score; a character that no piece of its own length covers is offered
    /// as unknown. Unknown pieces in a row are one piece: the piece of that
    /// text when there is one, the unknown piece otherwise.
    fn best_path(&self, model: &Unigram, word: &str, best: &mut Vec<Best>, ids: &mut Vec<u32>) {
        let bytes = word.as_bytes();
        best.clear();
        best.resize(
            bytes.len() + 1,
            Best {
                score: 0.0,
                start: NONE,
                id: NONE,
            },
        );
        let mut start = 0;
        while start < bytes.len() {
            let so_far = best[start].score;
            let char_len = match bytes[start] {
                0x00..=0x7F => 1,
                0x80..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xFF => 4,
            };
            let mut offer = |end: usize, id: u32, score: f64| {
                let target = &mut best[end];
                if target.start == NONE || score > target.score {
                    *target = Best {
                        score,
                        start: start as u32,
                        id,
                    };
                }
            };
            let mut covered = false;
            // Every piece that starts here: the model's own map is asked
            // for each text that ends between two characters and is no
            // longer than the longest piece.
            for end in start + char_len..=bytes.len().min(start + self.longest) {
                if !word.is_char_boundary(end) {
                    continue;
                }
                if let Some(id) = model.token_to_id(&word[start..end]) {
                    offer(end, id, self.score[id as usize] + so_far);
                    covered |= end == start + char_len;
                }
            }
            if !covered {
                offer(start + char_len, self.unknown, self.unknown_score + so_far);
            }
            start += char_len;
        }
        // The path is read from its end, so the ids come out last first.
        let from = ids.len();
        let mut end = bytes.len();
        let mut unknown_end = None;
        while end > 0 {
            let last = best[end];
            if last.id == self.unknown {
                unknown_end.get_or_insert(end);
            } else {
                if let Some(run_end) = unknown_end.take() {
                    ids.push(
                        model
                            .token_to_id(&word[end..run_end])
                            .unwrap_or(self.unknown),
                    );
                }
                ids.push(last.id);
            }
            end = last.start as usize;
        }
        if let Some(run_end) = unknown_end {
            ids.push(model.token_to_id(&word[..run_end]).unwrap_or(self.unknown));
        }
        ids[from..].reverse();
    }
}

#[cfg(test)]
mod tests;
