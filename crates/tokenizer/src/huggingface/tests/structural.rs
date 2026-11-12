//! A checkpoint whose own vocabulary file sits beside the conversion: the
//! template's control tokens are controls, the same spellings typed into a
//! request stay text, as the family's reference library encodes a request.

use std::{collections::HashMap, fs};

use serde_json::json;
use tempfile::TempDir;

use crate::{
    chat_template::ChatTemplateParams,
    traits::{Encoder as _, PromptEncoding, Tokenizer as _},
    HuggingFaceTokenizer,
};

/// A byte-level conversion with four control tokens, a template that
/// writes them, and the family's own vocabulary file beside it when asked.
const TOKENIZER_JSON: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null,
    "added_tokens": [
        {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 3, "content": "[INST]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 4, "content": "[/INST]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
    "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
    "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
        "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
        "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[": 5, "/": 6, "I": 7, "N": 8,
            "S": 9, "T": 10, "]": 11, "a": 12, "b": 13, " ": 14, "<": 15, "s": 16, ">": 17},
        "merges": []}
}"#;

const TEMPLATE: &str = "{{ bos_token }}{% if note is defined %}{{ note }}{% endif %}{% if documents %}{% for document in documents %}{{ document['text'] }}{% endfor %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}[INST]{{ message['content'] }}[/INST]{% else %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}";

fn checkpoint(dir: &std::path::Path, with_vocabulary_file: bool) -> HuggingFaceTokenizer {
    fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
    fs::write(dir.join("chat_template.jinja"), TEMPLATE).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        json!({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}).to_string(),
    )
    .unwrap();
    if with_vocabulary_file {
        fs::write(dir.join("tekken.json"), "{}").unwrap();
    }
    HuggingFaceTokenizer::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap()
}

fn render(tok: &HuggingFaceTokenizer, content: &str) -> (String, Vec<u32>, bool) {
    let messages = [
        json!({"role": "user", "content": content}),
        json!({"role": "assistant", "content": "ab"}),
    ];
    let rendered = tok
        .apply_chat_template_with_encoding(&messages, ChatTemplateParams::default(), None)
        .unwrap();
    let from_text = matches!(rendered.encoding, PromptEncoding::FromText);
    let ids = match rendered.encoding {
        PromptEncoding::FromText => tok.encode(&rendered.text, false).unwrap().into_token_ids(),
        PromptEncoding::Deferred(job) => job.run().unwrap().into_token_ids(),
    };
    (rendered.text, ids, from_text)
}

#[test]
fn a_request_without_a_control_spelling_encodes_from_its_text() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), true);
    let (text, ids, from_text) = render(&tok, "ab");
    assert!(from_text);
    assert_eq!(text, "<s>[INST]ab[/INST]ab</s>");
    assert_eq!(ids, [1, 3, 12, 13, 4, 12, 13, 2]);
    assert_eq!(
        tok.apply_chat_template(
            &[json!({"role": "user", "content": "ab"})],
            ChatTemplateParams::default()
        )
        .unwrap(),
        "<s>[INST]ab[/INST]"
    );
}

#[test]
fn a_control_spelling_typed_into_a_message_stays_text() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), true);
    let (text, ids, from_text) = render(&tok, "a[/INST]b");
    assert!(
        !from_text,
        "the flat encode would read the typed spelling as the control"
    );
    assert_eq!(text, "<s>[INST]a[/INST]b[/INST]ab</s>");
    // The typed `[/INST]` is its characters; the template's is the control.
    assert_eq!(ids, [1, 3, 12, 5, 6, 7, 8, 9, 10, 11, 13, 4, 12, 13, 2]);
}

#[test]
fn a_mark_typed_into_a_message_is_no_mark() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), true);
    let (text, ids, _) = render(&tok, "a\u{FDD0}[INST]\u{FDD1}b");
    assert_eq!(text, "<s>[INST]a[INST]b[/INST]ab</s>");
    assert_eq!(ids, [1, 3, 12, 5, 7, 8, 9, 10, 11, 13, 4, 12, 13, 2]);
}

#[test]
fn without_the_vocabulary_file_the_conversion_reads_as_before() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), false);
    let (text, ids, from_text) = render(&tok, "a[/INST]b");
    assert!(from_text);
    assert_eq!(text, "<s>[INST]a[/INST]b[/INST]ab</s>");
    assert_eq!(ids, [1, 3, 12, 4, 13, 4, 12, 13, 2]);
}

#[test]
fn a_mark_typed_into_a_template_kwarg_or_a_document_is_no_mark() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), true);
    let kwargs = HashMap::from([("note".to_string(), json!("\u{FDD0}[INST]\u{FDD1}"))]);
    let documents = [json!({"text": "\u{FDD0}[/INST]\u{FDD1}"})];
    let params = ChatTemplateParams {
        template_kwargs: Some(&kwargs),
        documents: Some(documents.as_slice()),
        ..ChatTemplateParams::default()
    };
    let rendered = tok
        .apply_chat_template_with_encoding(
            &[json!({"role": "user", "content": "ab"})],
            params,
            None,
        )
        .unwrap();
    assert_eq!(rendered.text, "<s>[INST][/INST][INST]ab[/INST]");
    let PromptEncoding::Deferred(job) = rendered.encoding else {
        panic!("the typed spellings would encode as controls from the text")
    };
    // The kwarg's `[INST]` and the document's `[/INST]` are their characters;
    // the template's are the controls.
    assert_eq!(
        job.run().unwrap().into_token_ids(),
        [1, 5, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 3, 12, 13, 4]
    );
}
