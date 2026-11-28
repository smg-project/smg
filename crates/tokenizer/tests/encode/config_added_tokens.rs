//! The special tokens `tokenizer_config.json` declares beyond `tokenizer.json`.
//! `transformers` and the engines' tokenizers read the configuration's
//! `added_tokens_decoder` on top of the file's added tokens, so a special
//! token only the configuration names is a token of the model: it encodes to
//! its id, its id decodes to its text, and the id is known to the gateway's
//! bound. A token both files name is held once; one the model's table holds
//! as a plain piece becomes an added token on the table's id; a declared id
//! the pipeline cannot place, or an entry that does not read, leaves that
//! token out with a warning and the rest load (the vocab.json + merges.txt
//! path, where the configuration is the only source of the ids, refuses it
//! instead). A token declared `special: false` that the file lacks stays
//! text, and a special token declared past it is left out with a warning
//! that names it.

#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]

use std::sync::{Arc, Mutex, PoisonError};

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

/// The WARN events of the thread it is the default subscriber on, each as
/// its fields in `name=value` form.
struct Warnings(Arc<Mutex<Vec<Vec<String>>>>);

impl tracing::Subscriber for Warnings {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        /// The fields of one event.
        struct Fields(Vec<String>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.push(format!("{}={value:?}", field.name()));
            }
        }
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        let mut fields = Fields(Vec::new());
        event.record(&mut fields);
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(fields.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Runs `f` and returns its value with the warnings it logged.
fn warnings_during<T>(f: impl FnOnce() -> T) -> (T, Vec<Vec<String>>) {
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let guard = tracing::subscriber::set_default(Warnings(Arc::clone(&warnings)));
    let out = f();
    drop(guard);
    let warnings = std::mem::take(&mut *warnings.lock().unwrap_or_else(PoisonError::into_inner));
    (out, warnings)
}

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

    // The same without a content key, and with a null content: the reference
    // reads both as the empty text, and so does the loader.
    for entry in [
        json!({"special": true}),
        json!({"content": null, "special": true}),
    ] {
        let dir = TempDir::new().expect("a temp dir");
        let mut config = config(&[(2, "<|im_start|>"), (4, "<|audio_end|>")]);
        config["added_tokens_decoder"]["3"] = entry;
        let tokenizer =
            load(&dir, &tokenizer_json(&[], &FILE_TOKENS), config).expect("the checkpoint loads");
        assert_eq!(tokenizer.token_to_id("<|audio_end|>"), None);
        assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 3);
    }
}

#[test]
fn an_entry_that_does_not_read_is_left_out_and_the_rest_load() {
    let well_formed = || config(&[(2, "<|im_start|>"), (3, "<|audio_start|>")]);

    // A key that is not an id: tokenizer.json holds the model's tokens, so
    // the file loads as it did before this path read the block; the entry
    // is left out and its well-placed neighbour loads on its id.
    let dir = TempDir::new().expect("a temp dir");
    let mut config = well_formed();
    config["added_tokens_decoder"]["notanumber"] = json!({"content": "<x>", "special": true});
    let tokenizer = load(&dir, &tokenizer_json(&[], &FILE_TOKENS), config)
        .expect("a key that is not an id does not refuse the load");
    assert_eq!(tokenizer.token_to_id("<|audio_start|>"), Some(3));
    assert_eq!(tokenizer.token_to_id("<x>"), None);

    // An entry whose content is not text is left out the same way.
    let dir = TempDir::new().expect("a temp dir");
    let mut config = well_formed();
    config["added_tokens_decoder"]["4"] = json!({"content": 4, "special": true});
    let tokenizer = load(&dir, &tokenizer_json(&[], &FILE_TOKENS), config)
        .expect("an entry that does not read does not refuse the load");
    assert_eq!(tokenizer.token_to_id("<|audio_start|>"), Some(3));
    assert_eq!(tokenizer.id_to_token(4), None);
}

#[test]
fn a_declared_token_that_is_not_special_stays_text() {
    // A token the configuration declares `special: false` and the file
    // lacks is not added on this path: its text encodes as it did before
    // this path read the block. The special token declared before it loads.
    let dir = TempDir::new().expect("a temp dir");
    let mut config = config(&[(2, "<|im_start|>"), (3, "<|audio_start|>"), (4, "<think>")]);
    config["added_tokens_decoder"]["4"]["special"] = json!(false);
    let tokenizer =
        load(&dir, &tokenizer_json(&[], &FILE_TOKENS), config).expect("the configuration loads");
    assert_eq!(tokenizer.token_to_id("<|audio_start|>"), Some(3));
    assert_eq!(tokenizer.token_to_id("<think>"), None);
}

#[test]
fn a_special_token_after_a_text_token_the_file_lacks_is_left_out() {
    // `<tool_call>` is declared `special: false` on the next free id (3) and
    // the file lacks it: it stays text and takes no id, so the pipeline's
    // next id stays 3 and neither special token declared past it (4, 5) can
    // be placed. Both are left out, as tokenizer.json alone reads them, and
    // the warning for each names the text token before it.
    let dir = TempDir::new().expect("a temp dir");
    let mut config = config(&[
        (2, "<|im_start|>"),
        (3, "<tool_call>"),
        (4, "<|audio_start|>"),
        (5, "<|audio_end|>"),
    ]);
    config["added_tokens_decoder"]["3"]["special"] = json!(false);
    let (tokenizer, warnings) =
        warnings_during(|| load(&dir, &tokenizer_json(&[], &FILE_TOKENS), config));
    let tokenizer = tokenizer.expect("the configuration loads");

    assert_eq!(tokenizer.token_to_id("<tool_call>"), None);
    assert_eq!(tokenizer.token_to_id("<|audio_start|>"), None);
    assert_eq!(tokenizer.token_to_id("<|audio_end|>"), None);
    assert_eq!(tokenizer.tokenizers_pipeline().get_vocab_size(true), 3);
    for token in ["<|audio_start|>", "<|audio_end|>"] {
        let left_out: Vec<&Vec<String>> = warnings
            .iter()
            .filter(|fields| fields.contains(&format!("token={token}")))
            .collect();
        assert_eq!(left_out.len(), 1, "one warning for {token}: {warnings:?}");
        assert!(
            left_out
                .iter()
                .all(|fields| fields.contains(&"text_token=<tool_call>".to_string())),
            "the warning for {token} names the text token before it: {warnings:?}"
        );
    }
}

/// A `tokenizer.json` whose model table holds `pieces` on the ids given and
/// no added token: a table with holes where the ids skip.
fn table_json(pieces: &[(&str, u32)]) -> String {
    let vocab: Map<String, Value> = pieces
        .iter()
        .map(|(piece, id)| ((*piece).to_string(), json!(id)))
        .collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    })
    .to_string()
}

#[test]
fn a_special_token_declared_on_an_id_the_table_holds_is_left_out() {
    // The table has a hole: two entries, `b` on id 2. The pipeline's next
    // free id is its entry count, 2, so a special token declared on 2 would
    // be added on the id of `b` and take its decode. It is left out, as
    // tokenizer.json alone reads the checkpoint, and the warning names both.
    let dir = TempDir::new().expect("a temp dir");
    let (tokenizer, warnings) = warnings_during(|| {
        load(
            &dir,
            &table_json(&[("a", 0), ("b", 2)]),
            config(&[(2, "<x>")]),
        )
    });
    let tokenizer = tokenizer.expect("the checkpoint loads");

    assert_eq!(tokenizer.token_to_id("<x>"), None);
    assert_eq!(tokenizer.id_to_token(2).as_deref(), Some("b"));
    let encoded = tokenizer.encode("b", false).expect("the piece encodes");
    assert_eq!(encoded.token_ids(), &[2]);
    assert_eq!(tokenizer.decode(&[2], true).expect("the id decodes"), "b");
    assert!(
        warnings.iter().any(|fields| {
            fields.contains(&"token=<x>".to_string())
                && fields.contains(&"table_piece=b".to_string())
        }),
        "the warning names the token and the piece: {warnings:?}"
    );
}

#[test]
fn a_special_token_declared_on_a_free_id_of_a_table_with_holes_is_added() {
    // The same hole, but the table holds nothing on its entry count: the
    // token is added there, as on a table without holes.
    let dir = TempDir::new().expect("a temp dir");
    let tokenizer = load(
        &dir,
        &table_json(&[("a", 0), ("b", 5)]),
        config(&[(2, "<x>")]),
    )
    .expect("the checkpoint loads");

    assert_eq!(tokenizer.token_to_id("<x>"), Some(2));
    assert_eq!(tokenizer.id_to_token(2).as_deref(), Some("<x>"));
    assert_eq!(tokenizer.decode(&[5], true).expect("the id decodes"), "b");
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
