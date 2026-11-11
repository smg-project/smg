//! The added tokens `tokenizer_config.json` declares beyond `tokenizer.json`.
//! `transformers` and the engines' tokenizers read the configuration's
//! `added_tokens_decoder` on top of the file's added tokens, so a token only
//! the configuration names is a token of the model: it encodes to its id,
//! its id decodes to its text, and the id is known to the gateway's bound.
//! A token both files name is held once; one the model's table holds as a
//! plain piece becomes an added token on the table's id; a declared id the
//! pipeline cannot place leaves that token out with a warning and the rest
//! load (the vocab.json + merges.txt path, where the configuration is the
//! only source of the ids, refuses it instead).

#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]

use llm_tokenizer::{
    traits::{Decoder as _, Encoder as _, Tokenizer as _},
    HuggingFaceTokenizer,
};
use serde_json::{json, Map, Value};
use tempfile::TempDir;

/// A `tokenizer.json` whose model vocabulary holds `plain` as ordinary pieces
/// and whose added tokens mark `special` as special tokens, in order after
/// the vocabulary.
fn tokenizer_json(plain: &[&str], special: &[&str]) -> String {
    let mut vocab = Map::new();
    for (id, piece) in ["hello", "Hi"].iter().chain(plain).enumerate() {
        vocab.insert((*piece).to_string(), json!(id));
    }
    let added_tokens: Vec<Value> = special
        .iter()
        .enumerate()
        .map(|(offset, token)| {
            json!({
                "id": vocab.len() + offset,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    })
    .to_string()
}

/// A `tokenizer_config.json` whose `added_tokens_decoder` declares `tokens`
/// at the ids given.
fn config(tokens: &[(u32, &str)]) -> Value {
    let mut decoder = Map::new();
    for (id, token) in tokens {
        decoder.insert(
            id.to_string(),
            json!({
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            }),
        );
    }
    json!({"added_tokens_decoder": decoder})
}

/// The directory holding `tokenizer_json` and `config`, loaded.
fn load(
    dir: &TempDir,
    tokenizer_json: &str,
    config: Value,
) -> anyhow::Result<HuggingFaceTokenizer> {
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, tokenizer_json).expect("tokenizer.json written");
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string())
        .expect("tokenizer_config.json written");
    HuggingFaceTokenizer::from_file(path.to_str().expect("a UTF-8 path"))
}

/// Two plain pieces (ids 0, 1), one special token in the file (id 2).
const FILE_TOKENS: [&str; 1] = ["<|im_start|>"];

#[test]
fn a_token_only_the_configuration_declares_is_a_token_of_the_model() {
    let dir = TempDir::new().expect("a temp dir");
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &FILE_TOKENS),
        config(&[(2, "<|im_start|>"), (3, "<|audio_start|>")]),
    )
    .expect("the tokenizer loads");

    assert_eq!(tokenizer.token_to_id("<|audio_start|>"), Some(3));
    assert_eq!(tokenizer.id_to_token(3).as_deref(), Some("<|audio_start|>"));
    let encoding = tokenizer
        .encode("<|audio_start|>", false)
        .expect("the control encodes");
    assert_eq!(encoding.token_ids(), &[3]);
    assert_eq!(
        tokenizer.decode(&[3], false).expect("the id decodes"),
        "<|audio_start|>"
    );
    // The model's table is unchanged; the count with the added tokens grew by the one token.
    assert_eq!(tokenizer.vocab_size(), 2);
    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 4);
}

#[test]
fn a_token_both_files_declare_is_held_once_on_its_id() {
    let dir = TempDir::new().expect("a temp dir");
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &FILE_TOKENS),
        config(&[(2, "<|im_start|>")]),
    )
    .expect("the tokenizer loads");

    assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(2));
    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 3);
    assert_eq!(
        tokenizer
            .encode("<|im_start|>", false)
            .expect("the token encodes")
            .token_ids(),
        &[2]
    );
}

#[test]
fn a_declared_token_the_table_holds_as_a_piece_becomes_an_added_token() {
    let dir = TempDir::new().expect("a temp dir");
    // The table holds the control as a plain piece (id 2) that the file's
    // added tokens leave out, as a GPT-2 layout holds `<|endoftext|>`.
    let tokenizer = load(
        &dir,
        &tokenizer_json(&["<|endoftext|>"], &FILE_TOKENS),
        config(&[(2, "<|endoftext|>"), (3, "<|im_start|>")]),
    )
    .expect("the tokenizer loads");

    let added = tokenizer.tokenizers_pipeline().get_added_vocabulary();
    assert_eq!(added.get_vocab().get("<|endoftext|>"), Some(&2));
    assert!(added.is_special_token("<|endoftext|>"));
    // Matched whole and dropped by a skip-special decode, as a control is.
    assert_eq!(
        tokenizer
            .encode("<|endoftext|>", false)
            .expect("the control encodes")
            .token_ids(),
        &[2]
    );
    assert_eq!(tokenizer.decode(&[2], true).expect("the id decodes"), "");
    assert_eq!(
        tokenizer.decode(&[2], false).expect("the id decodes"),
        "<|endoftext|>"
    );
    // The table keeps its three pieces, and the count with the added tokens
    // counts the control once.
    assert_eq!(tokenizer.vocab_size(), 3);
    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 4);
}

#[test]
fn a_declared_id_the_pipeline_cannot_place_leaves_that_token_out() {
    let dir = TempDir::new().expect("a temp dir");
    // Id 0 is the table's `hello`, id 7 is past the next free id (3); id 3 fits.
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &FILE_TOKENS),
        config(&[
            (0, "<|pad|>"),
            (2, "<|im_start|>"),
            (3, "<|audio_end|>"),
            (7, "<|audio_start|>"),
        ]),
    )
    .expect("the checkpoint loads as it did without the configuration's tokens");

    assert_eq!(tokenizer.token_to_id("hello"), Some(0));
    assert_eq!(tokenizer.token_to_id("<|audio_end|>"), Some(3));
    assert_eq!(tokenizer.token_to_id("<|pad|>"), None);
    assert_eq!(tokenizer.token_to_id("<|audio_start|>"), None);
    // The left-out tokens are not in the pipeline's added vocabulary: text,
    // as tokenizer.json alone reads them; the one that fits is.
    let added = tokenizer.tokenizers_pipeline().get_added_vocabulary();
    assert!(added.get_vocab().get("<|pad|>").is_none());
    assert!(added.get_vocab().get("<|audio_start|>").is_none());
    assert_eq!(added.get_vocab().get("<|audio_end|>"), Some(&3));
    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 4);
}

#[test]
fn a_content_declared_on_two_ids_is_held_on_the_first() {
    let dir = TempDir::new().expect("a temp dir");
    // The pipeline's add gives the second declaration the first one's id, so
    // the second is a declared id the pipeline cannot place: left out here.
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &FILE_TOKENS),
        config(&[
            (2, "<|im_start|>"),
            (3, "<|audio_start|>"),
            (4, "<|audio_start|>"),
        ]),
    )
    .expect("the checkpoint loads");

    assert_eq!(tokenizer.token_to_id("<|audio_start|>"), Some(3));
    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 4);
}

#[test]
fn an_entry_without_text_takes_no_id() {
    let dir = TempDir::new().expect("a temp dir");
    // The empty entry sits on the next free id (3), where a placement that
    // counted it would queue it and the check after the add would find no
    // token there. It is no token and takes no id, so `<|audio_end|>` on 4
    // would land on 3: left out, and the load goes through.
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &FILE_TOKENS),
        config(&[(2, "<|im_start|>"), (3, ""), (4, "<|audio_end|>")]),
    )
    .expect("the checkpoint loads");

    assert_eq!(tokenizer.token_to_id("<|audio_end|>"), None);
    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 3);
}

#[test]
fn a_configuration_without_the_block_changes_nothing() {
    let dir = TempDir::new().expect("a temp dir");
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &FILE_TOKENS),
        json!({"add_bos_token": false}),
    )
    .expect("the tokenizer loads");

    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 3);
    assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(2));
}
