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
    decoder: Option<Value>,
    model: Value,
}

/// The parts of `tokenizer.json` the wrapper keeps.
pub(crate) struct TokenizerFile {
    pub(crate) added_tokens: Vec<AddedToken>,
    /// The decoder as the file writes it; `None` for a file without one.
    pub(crate) decoder: Option<Value>,
    /// The model's vocabulary (`model.vocab`): the token strings of a BPE,
    /// WordPiece or WordLevel model and their ids. A Unigram model lists
    /// pieces with scores instead; such a vocabulary stays empty here, and
    /// no backend reads it.
    pub(crate) model_vocab: HashMap<String, TokenIdType>,
}

impl TokenizerFile {
    pub(crate) fn read(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let shape: FileShape =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let model_vocab = shape
            .model
            .get("vocab")
            .and_then(Value::as_object)
            .map(|vocab| {
                vocab
                    .iter()
                    .filter_map(|(token, id)| {
                        id.as_u64()
                            .and_then(|id| TokenIdType::try_from(id).ok())
                            .map(|id| (token.clone(), id))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            added_tokens: shape.added_tokens,
            decoder: shape.decoder,
            model_vocab,
        })
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

    #[test]
    fn a_unigram_model_leaves_the_vocabulary_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"added_tokens": [], "decoder": {"type": "ByteLevel"},
                "model": {"type": "Unigram", "vocab": [["a", -1.0], ["b", -2.0]]}}"#,
        )
        .unwrap();
        let file = TokenizerFile::read(&path).unwrap();
        assert!(file.model_vocab.is_empty());
        assert!(file.decoder_is_byte_level());
        assert!(!file.has_byte_fallback());
    }
}
