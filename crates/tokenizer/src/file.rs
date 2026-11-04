//! What the wrapper reads from a checkpoint's `tokenizer.json` itself, beside
//! the backend that encodes and decodes with it: the vocabulary, the added
//! tokens with the flags that place them in the pipeline, and the decoder's
//! shape. The special-token lists and the per-stream tables are built from
//! these, so that a tokenizer the backend serves needs no second pipeline.

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::traits::TokenIdType;

/// One entry of the file's `added_tokens`: a token cut out of the text before
/// the model sees it, on the raw text (`normalized: false`) or after
/// normalization, and special (stripped by `skip_special_tokens`) or not.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AddedToken {
    pub(crate) id: TokenIdType,
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) special: bool,
    #[serde(default)]
    pub(crate) normalized: bool,
}

#[derive(Deserialize)]
struct FileShape {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<Value>,
    #[serde(default)]
    decoder: Option<Value>,
    model: Value,
}

/// The parts of `tokenizer.json` the wrapper keeps.
pub(crate) struct TokenizerFile {
    pub(crate) added_tokens: Vec<AddedToken>,
    /// The pre-tokenizer as the file writes it; `None` for a file without
    /// one.
    pub(crate) pre_tokenizer: Option<Value>,
    /// The decoder as the file writes it; `None` for a file without one.
    pub(crate) decoder: Option<Value>,
    /// The model's vocabulary (`model.vocab`): the token strings of a BPE,
    /// WordPiece or WordLevel model and their ids. A Unigram model lists
    /// pieces with scores instead; the reader declines such a file, and the
    /// pipeline describes it.
    pub(crate) model_vocab: HashMap<String, TokenIdType>,
}

impl TokenizerFile {
    pub(crate) fn read(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let shape: FileShape =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let Some(vocab) = shape.model.get("vocab").and_then(Value::as_object) else {
            anyhow::bail!(
                "the model's vocabulary in {} is not an object of token to id (a Unigram model \
                 lists pieces with scores)",
                path.display()
            );
        };
        let model_vocab = vocab
            .iter()
            .filter_map(|(token, id)| {
                id.as_u64()
                    .and_then(|id| TokenIdType::try_from(id).ok())
                    .map(|id| (token.clone(), id))
            })
            .collect();
        Ok(Self {
            added_tokens: shape.added_tokens,
            pre_tokenizer: shape.pre_tokenizer,
            decoder: shape.decoder,
            model_vocab,
        })
    }

    /// The first `Split` pattern of the pre-tokenizer (through a
    /// `Sequence`) anchored to the piece it runs on: `\A`, `\z`, `\Z`,
    /// `\G`, or `^` and `$` outside a class. `tokenizers` runs each `Split`
    /// on each piece the steps before it cut, where such an anchor holds at
    /// the piece's ends; the backend splits a run of seven or more digits
    /// otherwise under one (a checkpoint grouping digits by threes with
    /// `\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)`), so the load keeps the pipeline for
    /// such a file.
    pub(crate) fn anchored_split_pattern(&self) -> Option<&str> {
        fn walk(step: &Value) -> Option<&str> {
            match step.get("type").and_then(Value::as_str) {
                Some("Sequence") => step
                    .get("pretokenizers")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .find_map(walk),
                Some("Split") => step
                    .get("pattern")
                    .and_then(|pattern| pattern.get("Regex"))
                    .and_then(Value::as_str)
                    .filter(|pattern| pattern_has_piece_anchor(pattern)),
                _ => None,
            }
        }
        self.pre_tokenizer.as_ref().and_then(walk)
    }

    /// Every token and its id: the model's vocabulary with the added tokens
    /// over it, as `tokenizers` reports the vocabulary with added tokens.
    pub(crate) fn vocab(&self) -> HashMap<String, TokenIdType> {
        let mut vocab = self.model_vocab.clone();
        for token in &self.added_tokens {
            vocab.insert(token.content.clone(), token.id);
        }
        vocab
    }

    /// The added tokens the file marks special, in the file's order.
    pub(crate) fn special_tokens(&self) -> Vec<String> {
        self.added_tokens
            .iter()
            .filter(|token| token.special)
            .map(|token| token.content.clone())
            .collect()
    }

    /// Whether the decoder is exactly a plain `ByteLevel` one.
    pub(crate) fn decoder_is_byte_level(&self) -> bool {
        self.decoder
            .as_ref()
            .and_then(|decoder| decoder.get("type"))
            .and_then(Value::as_str)
            == Some("ByteLevel")
    }

    /// Whether the decoder chain holds a `ByteFallback` step.
    pub(crate) fn has_byte_fallback(&self) -> bool {
        fn holds(decoder: &Value) -> bool {
            match decoder.get("type").and_then(Value::as_str) {
                Some("ByteFallback") => true,
                Some("Sequence") => decoder
                    .get("decoders")
                    .and_then(Value::as_array)
                    .is_some_and(|steps| steps.iter().any(holds)),
                _ => false,
            }
        }
        self.decoder.as_ref().is_some_and(holds)
    }
}

/// Whether `pattern` anchors to the start or the end of its subject: the
/// escapes `\A`, `\z`, `\Z` and `\G`, or `^` and `$` outside a character
/// class (`[^...]` negates a class, so it is not an anchor).
pub(crate) fn pattern_has_piece_anchor(pattern: &str) -> bool {
    let mut chars = pattern.chars();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('A' | 'z' | 'Z' | 'G') if !in_class => return true,
                _ => {}
            },
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '^' | '$' if !in_class => return true,
            _ => {}
        }
    }
    false
}

/// The added tokens matched on the raw text that are not special: strings
/// the tokenizer takes whole before pre-tokenization (see
/// `Tokenizer::added_token_strings`).
pub(crate) fn raw_pass_added_tokens(added_tokens: &[AddedToken]) -> Vec<String> {
    added_tokens
        .iter()
        .filter(|token| !token.special && !token.normalized)
        .map(|token| token.content.clone())
        .collect()
}

/// The added tokens matched after normalization, special or not (see
/// `Tokenizer::added_tokens_after_normalization`).
pub(crate) fn normalized_pass_added_tokens(added_tokens: &[AddedToken]) -> Vec<String> {
    added_tokens
        .iter()
        .filter(|token| token.normalized)
        .map(|token| token.content.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"{
        "version": "1.0",
        "added_tokens": [
            {"id": 3, "content": "<s>", "special": true, "normalized": false},
            {"id": 4, "content": "<tool>", "special": false, "normalized": false},
            {"id": 5, "content": "<think>", "special": false, "normalized": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<s>": 2}, "merges": []}
    }"#;

    #[test]
    fn the_file_s_vocabulary_added_tokens_and_decoder_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, FILE).unwrap();
        let file = TokenizerFile::read(&path).unwrap();
        let vocab = file.vocab();
        // The added token's id replaces the model's for the same spelling.
        assert_eq!(vocab["<s>"], 3);
        assert_eq!(vocab.len(), 5);
        assert_eq!(file.special_tokens(), ["<s>"]);
        assert_eq!(raw_pass_added_tokens(&file.added_tokens), ["<tool>"]);
        assert_eq!(
            normalized_pass_added_tokens(&file.added_tokens),
            ["<think>"]
        );
        assert!(!file.decoder_is_byte_level());
        assert!(file.has_byte_fallback());
    }

    /// The GPT-2 and Qwen patterns (a lookahead, a negated class) carry no
    /// anchor; a pattern anchored to the piece does, escaped or bare.
    #[test]
    fn a_pattern_anchored_to_the_piece_is_told_from_one_that_is_not() {
        for pattern in [
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            r"\\A[\^$]",
        ] {
            assert!(!pattern_has_piece_anchor(pattern), "{pattern}");
        }
        for pattern in [
            r"\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)",
            r"\A\p{Nd}{3}|\G\p{Nd}{3}",
            r"\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))",
            r"^\s+",
        ] {
            assert!(pattern_has_piece_anchor(pattern), "{pattern}");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"added_tokens": [], "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Digits", "individual_digits": false},
                {"type": "Split", "pattern": {"Regex": "\\A\\p{Nd}{3}|\\G\\p{Nd}{3}"}, "behavior": "Isolated", "invert": false}]},
                "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}"#,
        )
        .unwrap();
        let file = TokenizerFile::read(&path).unwrap();
        assert_eq!(
            file.anchored_split_pattern(),
            Some(r"\A\p{Nd}{3}|\G\p{Nd}{3}")
        );
    }

    /// A Unigram model's vocabulary is a list of pieces with scores: the
    /// reader declines the file rather than reading an empty vocabulary,
    /// so the pipeline describes the tokenizer.
    #[test]
    fn a_unigram_model_is_not_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"added_tokens": [], "decoder": {"type": "ByteLevel"},
                "model": {"type": "Unigram", "vocab": [["a", -1.0], ["b", -2.0]]}}"#,
        )
        .unwrap();
        let Err(error) = TokenizerFile::read(&path) else {
            panic!("a Unigram file is not read");
        };
        assert!(error.to_string().contains("not an object"), "{error}");
    }
}
