//! The special tokens as `tokenizer_config.json` gives them. A token the
//! configuration sets is itself; one it sets to `null` is none, whatever the
//! vocabulary holds: the checkpoint says it has no such token, and a
//! template's `{{ bos_token }}` renders nothing, as the reference renderer
//! writes it. Only a token the configuration leaves out is guessed, and only
//! among the added tokens the file marks special: a plain vocabulary entry
//! spelled `<s>` is an ordinary piece, not a beginning-of-sequence token.

#![expect(clippy::expect_used, reason = "a test that stops on a broken setup")]

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    traits::{SpecialTokens, Tokenizer as _},
    HuggingFaceTokenizer,
};
use serde_json::{json, Map, Value};
use tempfile::TempDir;

/// A template that writes the two tokens most templates write.
const TEMPLATE: &str =
    "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}";

/// A `tokenizer.json` whose model vocabulary holds `plain` as ordinary
/// pieces and whose added tokens mark `special` as special tokens.
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

/// The tokenizer of a directory holding `tokenizer_json` and `config` as its
/// `tokenizer_config.json`.
fn load(dir: &TempDir, tokenizer_json: &str, config: Value) -> HuggingFaceTokenizer {
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, tokenizer_json).expect("tokenizer.json written");
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string())
        .expect("tokenizer_config.json written");
    HuggingFaceTokenizer::from_file(path.to_str().expect("a UTF-8 path"))
        .expect("the tokenizer loads")
}

fn render(tokenizer: &HuggingFaceTokenizer) -> String {
    let messages = [json!({"role": "user", "content": "Hi"})];
    tokenizer
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .expect("the template renders")
}

/// One of the four tokens the configuration names: its key, the spelling
/// the guess looks up first, and the field that holds it.
struct Named {
    key: &'static str,
    spelling: &'static str,
    read: fn(&SpecialTokens) -> Option<&str>,
}

fn bos(special: &SpecialTokens) -> Option<&str> {
    special.bos_token.as_deref()
}

fn eos(special: &SpecialTokens) -> Option<&str> {
    special.eos_token.as_deref()
}

fn unk(special: &SpecialTokens) -> Option<&str> {
    special.unk_token.as_deref()
}

fn pad(special: &SpecialTokens) -> Option<&str> {
    special.pad_token.as_deref()
}

const CONFIGURED: [Named; 4] = [
    Named {
        key: "bos_token",
        spelling: "<s>",
        read: bos,
    },
    Named {
        key: "eos_token",
        spelling: "</s>",
        read: eos,
    },
    Named {
        key: "unk_token",
        spelling: "<unk>",
        read: unk,
    },
    Named {
        key: "pad_token",
        spelling: "<pad>",
        read: pad,
    },
];

#[test]
fn an_explicit_null_is_no_token_whatever_the_vocabulary_holds() {
    let spellings: Vec<&str> = CONFIGURED.iter().map(|named| named.spelling).collect();
    let nulls: Map<String, Value> = CONFIGURED
        .iter()
        .map(|named| (named.key.to_string(), Value::Null))
        .collect();
    let mut config = Value::Object(nulls);
    config["chat_template"] = json!(TEMPLATE);

    // The spellings as plain pieces, then as added special tokens.
    for file in [
        tokenizer_json(&spellings, &[]),
        tokenizer_json(&[], &spellings),
    ] {
        let dir = TempDir::new().expect("a temporary directory");
        let tokenizer = load(&dir, &file, config.clone());
        for named in &CONFIGURED {
            assert_eq!(
                (named.read)(tokenizer.get_special_tokens()),
                None,
                "{} set to null",
                named.key
            );
        }
        assert_eq!(render(&tokenizer), "Hi");
    }
}

#[test]
fn a_token_the_configuration_leaves_out_is_the_added_special_token_of_its_usual_spelling() {
    let spellings: Vec<&str> = CONFIGURED.iter().map(|named| named.spelling).collect();
    let dir = TempDir::new().expect("a temporary directory");
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &spellings),
        json!({"chat_template": TEMPLATE}),
    );
    for named in &CONFIGURED {
        assert_eq!(
            (named.read)(tokenizer.get_special_tokens()),
            Some(named.spelling),
            "{} left out",
            named.key
        );
    }
    assert_eq!(render(&tokenizer), "<s>Hi</s>");

    // The later spellings of the list, with the first ones absent.
    let dir = TempDir::new().expect("a temporary directory");
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &["<|startoftext|>", "<|endoftext|>", "[UNK]", "[PAD]"]),
        json!({"chat_template": TEMPLATE}),
    );
    let special = tokenizer.get_special_tokens();
    assert_eq!(special.bos_token.as_deref(), Some("<|startoftext|>"));
    assert_eq!(special.eos_token.as_deref(), Some("<|endoftext|>"));
    assert_eq!(special.unk_token.as_deref(), Some("[UNK]"));
    assert_eq!(special.pad_token.as_deref(), Some("[PAD]"));
    assert_eq!(render(&tokenizer), "<|startoftext|>Hi<|endoftext|>");
}

#[test]
fn a_plain_vocabulary_entry_is_never_guessed() {
    let dir = TempDir::new().expect("a temporary directory");
    let tokenizer = load(
        &dir,
        &tokenizer_json(
            &["<s>", "</s>", "<unk>", "<pad>", "[CLS]", "[SEP]", "[MASK]"],
            &[],
        ),
        json!({"chat_template": TEMPLATE}),
    );
    let special = tokenizer.get_special_tokens();
    assert_eq!(special.bos_token, None);
    assert_eq!(special.eos_token, None);
    assert_eq!(special.unk_token, None);
    assert_eq!(special.pad_token, None);
    assert_eq!(special.sep_token, None);
    assert_eq!(special.cls_token, None);
    assert_eq!(special.mask_token, None);
    assert_eq!(render(&tokenizer), "Hi");
}

#[test]
fn a_configured_token_is_read_as_given_over_any_guess() {
    // As a string and as the added-token object; `<s>` and `</s>` are in the
    // file as added special tokens and do not stand in for them.
    let dir = TempDir::new().expect("a temporary directory");
    let tokenizer = load(
        &dir,
        &tokenizer_json(&[], &["<s>", "</s>", "<|begin|>", "<|end|>"]),
        json!({
            "bos_token": "<|begin|>",
            "eos_token": {"content": "<|end|>", "lstrip": false, "normalized": false,
                          "rstrip": false, "single_word": false, "special": true},
            "chat_template": TEMPLATE,
        }),
    );
    let special = tokenizer.get_special_tokens();
    assert_eq!(special.bos_token.as_deref(), Some("<|begin|>"));
    assert_eq!(special.eos_token.as_deref(), Some("<|end|>"));
    assert_eq!(render(&tokenizer), "<|begin|>Hi<|end|>");
}

/// The shape of a byte-level BPE checkpoint whose vocabulary holds the text
/// `<s>` as an ordinary piece, whose configuration sets `bos_token` and
/// `unk_token` to `null`, and whose template opens with `{{ bos_token }}`:
/// the reference renders the first message's own opening marker, nothing
/// before it.
#[test]
fn a_null_bos_token_renders_nothing_before_the_first_marker() {
    let dir = TempDir::new().expect("a temporary directory");
    let tokenizer = load(
        &dir,
        &tokenizer_json(
            &["<s>", "</s>", "<unk>"],
            &["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        ),
        json!({
            "add_bos_token": false,
            "bos_token": null,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "unk_token": null,
            "chat_template": "{{ bos_token }}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}",
        }),
    );
    let special = tokenizer.get_special_tokens();
    assert_eq!(special.bos_token, None);
    assert_eq!(special.unk_token, None);
    assert_eq!(special.eos_token.as_deref(), Some("<|im_end|>"));
    assert_eq!(special.pad_token.as_deref(), Some("<|endoftext|>"));
    assert_eq!(render(&tokenizer), "<|im_start|>user\nHi<|im_end|>\n");
}
