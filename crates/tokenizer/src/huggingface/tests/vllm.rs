//! The `vllm-tokenizer` backend: the files it takes, the loads it
//! declines, and that its ids and text are the `tokenizers` pipeline's.

use std::{fs, sync::Arc};

use serde_json::json;
use tempfile::TempDir;

use super::{
    llama_style_checkpoint, BYTE_FALLBACK_TOKENIZER_JSON, BYTE_LEVEL_TOKENIZER_JSON,
    NORMALIZED_TIE_TOKENIZER_JSON, WORD_LEVEL_TOKENIZER_JSON,
};
use crate::{
    encode_path::EncodePath,
    traits::{Decoder as _, Encoder as _, Tokenizer as _},
    HuggingFaceTokenizer, Sequence,
};

/// A checkpoint of `tokenizer.json` alone, loaded through the file's path.
fn checkpoint(tokenizer_json: &str) -> (TempDir, HuggingFaceTokenizer) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tokenizer.json");
    fs::write(&path, tokenizer_json).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    (dir, tokenizer)
}

/// Texts within the test vocabularies: a byte the vocabulary has no token
/// for is an error in `fastokens` where `tokenizers` drops it, which no
/// byte-level vocabulary of a checkpoint (all 256 byte tokens) reaches.
const TEXTS: &[&str] = &["", "a", "b a", "ab ba", " a  b ", "a b b a"];

/// Texts within the byte-fallback fixture's vocabulary (its pieces are the
/// text `> <s> </s> <|`'s).
const BYTE_FALLBACK_TEXTS: &[&str] = &["", "<", "> <", "<|", "> <s> </s> <|", " < > "];

#[test]
fn a_file_s_own_pipeline_takes_the_backend_and_agrees_with_tokenizers() {
    // The byte-fallback fixture renumbered to contiguous ids, as
    // `fastokens` reads a vocabulary (see `contiguous_ids`).
    let byte_fallback = contiguous_ids(BYTE_FALLBACK_TOKENIZER_JSON);
    for (name, json, texts) in [
        ("byte-level", BYTE_LEVEL_TOKENIZER_JSON, TEXTS),
        ("byte-fallback", byte_fallback.as_str(), BYTE_FALLBACK_TEXTS),
    ] {
        let (dir, ours) = checkpoint(json);
        assert_eq!(ours.encode_path(), EncodePath::Vllm, "{name}");
        assert!(
            ours.tokenizers_pipeline().is_none(),
            "{name}: the pipeline is not kept"
        );
        let path = dir.path().join("tokenizer.json");
        let on_tokenizers =
            HuggingFaceTokenizer::from_file_on_tokenizers(path.to_str().unwrap()).unwrap();
        assert_eq!(
            on_tokenizers.encode_path(),
            EncodePath::Tokenizers,
            "{name}"
        );
        let reference = on_tokenizers.tokenizers_pipeline().unwrap();
        for &text in texts {
            for specials in [false, true] {
                let expected = reference.encode(text, specials).unwrap();
                let ids = ours.encode(text, specials).unwrap();
                assert_eq!(
                    ids.token_ids(),
                    expected.get_ids(),
                    "{name}: {text:?}, specials={specials}"
                );
                let batch = ours.encode_batch(&[text, text], specials).unwrap();
                assert_eq!(batch.len(), 2, "{name}: batch of {text:?}");
                assert!(
                    batch
                        .iter()
                        .all(|encoding| encoding.token_ids() == expected.get_ids()),
                    "{name}: batch of {text:?}"
                );
                for skip in [false, true] {
                    assert_eq!(
                        ours.decode(expected.get_ids(), skip).unwrap(),
                        reference.decode(expected.get_ids(), skip).unwrap(),
                        "{name}: decode of {text:?}, skip={skip}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_tokenizer_built_in_memory_keeps_the_established_paths() {
    let tokenizer = tokenizers::Tokenizer::from_bytes(BYTE_LEVEL_TOKENIZER_JSON).unwrap();
    let ours = HuggingFaceTokenizer::from_tokenizer(tokenizer);
    assert_ne!(ours.encode_path(), EncodePath::Vllm);
}

/// A file `fastokens` does not read (a word-level model behind a
/// `Whitespace` pre-tokenizer) keeps the established paths: the frontend's
/// own fallback would be `tokenizers` with its default word cache.
#[test]
fn a_file_fastokens_does_not_read_keeps_the_established_paths() {
    let (_dir, ours) = checkpoint(WORD_LEVEL_TOKENIZER_JSON);
    assert_eq!(ours.encode_path(), EncodePath::Tokenizers);
    assert_eq!(ours.encode("a b", false).unwrap().token_ids(), &[0, 1]);
}

/// The file alone would give other ids than the wrapper: the Llama class's
/// pipeline over a legacy file, a tie among normalized added tokens settled.
#[test]
fn a_pipeline_the_load_changed_beyond_the_file_declines_the_backend() {
    let dir = TempDir::new().unwrap();
    let path = llama_style_checkpoint(dir.path(), "LlamaTokenizerFast", None, None, "llama");
    let ours = HuggingFaceTokenizer::from_file(&path).unwrap();
    assert_ne!(
        ours.encode_path(),
        EncodePath::Vllm,
        "the Llama class's pipeline"
    );

    let (_dir, ours) = checkpoint(NORMALIZED_TIE_TOKENIZER_JSON);
    assert_ne!(ours.encode_path(), EncodePath::Vllm, "a settled tie");
}

/// `tokenizer_config.json` can name added tokens beyond the file's
/// (`added_tokens_decoder`), as the Qwen3.5 checkpoints name their audio
/// markers: `transformers` and the backend add them with the ids after the
/// file's, and so does the wrapper, on the backend and on the pipeline
/// alike. Ids that do not follow on are left out, and the backend, which
/// would read them, is declined.
#[test]
fn the_added_tokens_the_configuration_names_beyond_the_file_are_added_on_both_paths() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tokenizer.json");
    fs::write(&path, BYTE_LEVEL_TOKENIZER_JSON).unwrap();
    let config = dir.path().join("tokenizer_config.json");
    // The file's vocabulary ends at 4: the configuration's token takes 5.
    fs::write(
        &config,
        r#"{"added_tokens_decoder": {"5": {"content": "<|extra|>", "special": true, "normalized": false}}}"#,
    )
    .unwrap();
    let path = path.to_str().unwrap();
    let ours = HuggingFaceTokenizer::from_file(path).unwrap();
    assert_eq!(ours.encode_path(), EncodePath::Vllm);
    let on_tokenizers = HuggingFaceTokenizer::from_file_on_tokenizers(path).unwrap();
    for (name, tokenizer) in [("backend", &ours), ("pipeline", &on_tokenizers)] {
        assert_eq!(tokenizer.token_to_id("<|extra|>"), Some(5), "{name}");
        assert_eq!(
            tokenizer.id_to_token(5).as_deref(),
            Some("<|extra|>"),
            "{name}"
        );
        assert_eq!(
            tokenizer.encode("a<|extra|>b", false).unwrap().token_ids(),
            &[0, 5, 1],
            "{name}"
        );
        assert_eq!(tokenizer.decode(&[0, 5, 1], true).unwrap(), "ab", "{name}");
        assert_eq!(
            tokenizer.decode(&[0, 5, 1], false).unwrap(),
            "a<|extra|>b",
            "{name}"
        );
    }

    // An id that does not follow the vocabulary's end: left out, and the
    // backend declined.
    fs::write(
        &config,
        r#"{"added_tokens_decoder": {"7": {"content": "<|extra|>", "special": true, "normalized": false}}}"#,
    )
    .unwrap();
    let ours = HuggingFaceTokenizer::from_file(path).unwrap();
    assert_eq!(ours.encode_path(), EncodePath::Tokenizers);
    assert_eq!(ours.token_to_id("<|extra|>"), None);
}

/// A pre-tokenizer pattern anchored to the piece (a checkpoint grouping
/// digits by threes, `\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)`) splits a run of seven
/// or more digits otherwise on the backend than in `tokenizers`: the load
/// keeps the pipeline for such a file.
#[test]
fn a_pre_tokenizer_pattern_anchored_to_the_piece_declines_the_backend() {
    let mut definition: serde_json::Value =
        serde_json::from_str(BYTE_LEVEL_TOKENIZER_JSON).unwrap();
    definition["pre_tokenizer"] = json!({"type": "Sequence", "pretokenizers": [
        {"type": "Digits", "individual_digits": false},
        {"type": "Split", "pattern": {"Regex": "\\A\\p{Nd}{1,2}(?=\\p{Nd}{3}+\\z)"}, "behavior": "Isolated", "invert": false},
        {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]});
    let (_dir, ours) = checkpoint(&definition.to_string());
    assert_eq!(ours.encode_path(), EncodePath::Tokenizers);
}

/// A plain ByteLevel file whose configuration prescribes the BOS on request:
/// the backend takes the file and puts the configured frame around its ids,
/// where the file's post-processor adds nothing.
#[test]
fn the_configured_frame_the_file_lacks_goes_around_the_backend_s_ids() {
    let dir = TempDir::new().unwrap();
    let mut definition: serde_json::Value =
        serde_json::from_str(BYTE_LEVEL_TOKENIZER_JSON).unwrap();
    definition["added_tokens"] = json!([{
        "id": 5, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false,
        "normalized": false, "special": true
    }]);
    let tokenizer_path = dir.path().join("tokenizer.json");
    fs::write(&tokenizer_path, definition.to_string()).unwrap();
    fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"bos_token": "<s>", "add_bos_token": true}).to_string(),
    )
    .unwrap();
    let ours = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    assert_eq!(ours.encode_path(), EncodePath::Vllm);
    assert_eq!(ours.encode("a", true).unwrap().token_ids(), &[5, 0]);
    assert_eq!(ours.encode("a", false).unwrap().token_ids(), &[0]);
    assert_eq!(
        ours.encode_batch(&["a", "b"], true).unwrap()[1].token_ids(),
        &[5, 1]
    );
    assert_eq!(ours.decode(&[5, 0], true).unwrap(), "a");
}

/// A tokenizer the backend serves builds its vocabulary, its added tokens
/// and its per-stream tables from the pipeline the load read the file
/// into, then lets the pipeline go; one built in memory keeps it. Over
/// every id, with and without special tokens, the two stream and decode
/// the same text and answer the same lookups.
#[test]
fn a_tokenizer_the_backend_serves_streams_what_one_built_in_memory_streams() {
    // `fastokens` reads a vocabulary whose ids are contiguous, as every
    // checkpoint's are; the byte-fallback fixture's stand in for a 32k
    // vocabulary's, so they are renumbered in rank order here.
    let byte_fallback = contiguous_ids(BYTE_FALLBACK_TOKENIZER_JSON);
    for (name, json) in [
        ("byte-level", BYTE_LEVEL_TOKENIZER_JSON),
        ("byte-fallback", byte_fallback.as_str()),
    ] {
        let (_dir, from_file) = checkpoint(json);
        let from_pipeline =
            HuggingFaceTokenizer::from_tokenizer(tokenizers::Tokenizer::from_bytes(json).unwrap());
        assert_eq!(from_file.encode_path(), EncodePath::Vllm, "{name}");
        assert_ne!(from_pipeline.encode_path(), EncodePath::Vllm, "{name}");
        assert_eq!(
            from_file.vocab, from_pipeline.vocab,
            "{name}: the vocabulary"
        );
        assert_eq!(
            from_file.added_token_strings(),
            from_pipeline.added_token_strings(),
            "{name}"
        );
        assert_eq!(
            from_file.added_tokens_after_normalization(),
            from_pipeline.added_tokens_after_normalization(),
            "{name}"
        );
        assert_eq!(
            from_file.get_special_tokens().eos_token,
            from_pipeline.get_special_tokens().eos_token,
            "{name}"
        );
        assert_eq!(
            from_file.pieces.is_some(),
            from_pipeline.pieces.is_some(),
            "{name}"
        );
        assert_eq!(
            from_file.byte_level.is_some(),
            from_pipeline.byte_level.is_some(),
            "{name}"
        );
        let mut ids: Vec<u32> = from_file.reverse_vocab.keys().copied().collect();
        ids.sort_unstable();
        ids.push(ids.len() as u32 + 1_000); // an id the vocabulary lacks
        let from_file: Arc<dyn crate::traits::Tokenizer> = Arc::new(from_file);
        let from_pipeline: Arc<dyn crate::traits::Tokenizer> = Arc::new(from_pipeline);
        for skip in [false, true] {
            let stream = |tokenizer: &Arc<dyn crate::traits::Tokenizer>| {
                let mut sequence = Sequence::new_with_options(Arc::clone(tokenizer), skip);
                let mut text = String::new();
                for &id in &ids {
                    if let Ok(piece) = sequence.append_token(id) {
                        text.push_str(&piece);
                    }
                }
                text.push_str(&sequence.finish().unwrap_or_default());
                text
            };
            assert_eq!(
                stream(&from_file),
                stream(&from_pipeline),
                "{name}: stream, skip={skip}"
            );
        }
    }
}

/// The tokenizer definition with its vocabulary renumbered to contiguous
/// ids in the order of the original ones, the added tokens and the
/// post-processor's special tokens following.
fn contiguous_ids(tokenizer_json: &str) -> String {
    let mut definition: serde_json::Value = serde_json::from_str(tokenizer_json).unwrap();
    let vocab = definition["model"]["vocab"].as_object().unwrap().clone();
    let mut by_id: Vec<(&str, u64)> = vocab
        .iter()
        .map(|(token, id)| (token.as_str(), id.as_u64().unwrap()))
        .collect();
    by_id.sort_unstable_by_key(|(_, id)| *id);
    let renumbered: std::collections::HashMap<&str, u64> = by_id
        .iter()
        .enumerate()
        .map(|(new_id, (token, _))| (*token, new_id as u64))
        .collect();
    definition["model"]["vocab"] = renumbered
        .iter()
        .map(|(token, id)| ((*token).to_string(), json!(id)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    for token in definition["added_tokens"].as_array_mut().unwrap() {
        let content = token["content"].as_str().unwrap().to_string();
        token["id"] = json!(renumbered[content.as_str()]);
    }
    if let Some(specials) = definition["post_processor"]["special_tokens"].as_object_mut() {
        for (token, entry) in specials.iter_mut() {
            entry["ids"] = json!([renumbered[token.as_str()]]);
        }
    }
    definition.to_string()
}
