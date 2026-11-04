//! The opt-in `vllm-tokenizer` backend: the files it takes, the loads it
//! declines, and that its ids and text are the `tokenizers` pipeline's.

use std::fs;

use serde_json::json;
use tempfile::TempDir;

use super::{
    llama_style_checkpoint, BYTE_LEVEL_TOKENIZER_JSON, NORMALIZED_TIE_TOKENIZER_JSON,
    WORD_LEVEL_TOKENIZER_JSON,
};
use crate::{
    encode_path::EncodePath,
    traits::{Decoder as _, Encoder as _},
    HuggingFaceTokenizer,
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

#[test]
fn a_file_s_own_pipeline_takes_the_backend_and_agrees_with_tokenizers() {
    for (name, json) in [("byte-level", BYTE_LEVEL_TOKENIZER_JSON)] {
        let (_dir, ours) = checkpoint(json);
        assert_eq!(ours.encode_path(), EncodePath::Vllm, "{name}");
        let reference = ours.tokenizers_pipeline();
        for &text in TEXTS {
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
/// pipeline over a legacy file, a tie among normalized added tokens
/// settled, a post-processor the configuration prescribes.
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

    // A plain ByteLevel file whose configuration prescribes the BOS on request.
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
    assert_ne!(
        ours.encode_path(),
        EncodePath::Vllm,
        "a configured post-processor"
    );
    assert_eq!(ours.encode("a", true).unwrap().token_ids(), &[5, 0]);
}
