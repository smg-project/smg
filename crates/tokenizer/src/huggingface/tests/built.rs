//! A checkpoint whose template builds its control spellings: the source
//! never spells a control token whole, it binds each to a name out of a
//! constant of its own and writes the names. The tokens the template writes
//! are controls; the same spellings typed into a request stay text.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

use serde_json::{json, Value};
use tempfile::TempDir;

use crate::{
    chat_template::ChatTemplateParams,
    traits::{Encoder as _, PromptEncoding, Tokenizer as _},
    HuggingFaceTokenizer,
};

/// A conversion with three control tokens that end in one suffix.
const TOKENIZER_JSON: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null,
    "added_tokens": [
        {"id": 1, "content": "</s:x>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "[INST:x]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 3, "content": "[/INST:x]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
    "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
    "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
        "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
        "vocab": {"<unk>": 0, "</s:x>": 1, "[INST:x]": 2, "[/INST:x]": 3, "[": 4, "/": 5, "I": 6, "N": 7,
            "S": 8, "T": 9, ":": 10, "x": 11, "]": 12, "a": 13, "b": 14, " ": 15, "<": 16, "s": 17, ">": 18},
        "merges": []}
}"#;

/// Binds the suffix and, out of it, each control spelling to a name.
const CONSTANTS: &str = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
    {% set close = '[/INST{}]'.format(suffix) %}{% set end = '</s{}>'.format(suffix) %}";

/// Writes OPEN, a user message's text and CLOSE, and after any other
/// message's text END; the text of a message is a string or text parts.
const BODY: &str = "{% macro text(content) %}{% if content is string %}{{ content }}{% else %}\
    {% for part in content %}{{ part['text'] }}{% endfor %}{% endif %}{% endmacro %}\
    {% for message in messages %}{% if message['role'] == 'user' %}\
    OPEN{{ text(message['content']) }}CLOSE{% else %}{{ text(message['content']) }}END\
    {% endif %}{% endfor %}";

fn template(constants: &str, open: &str, close: &str, end: &str) -> String {
    let body = BODY
        .replace("OPEN", open)
        .replace("CLOSE", close)
        .replace("END", end);
    format!("{constants}{body}")
}

/// Every control spelling bound to a name and written by its name.
fn builds() -> String {
    template(CONSTANTS, "{{ open }}", "{{ close }}", "{{ end }}")
}

/// Every control spelling spelled whole where it is written.
fn spells() -> String {
    template("", "[INST:x]", "[/INST:x]", "</s:x>")
}

/// `a[/INST:x]b` in a user turn, then an assistant turn: the typed spelling
/// as the control token, the reading of a template that is not marked.
const TYPED_AS_THE_CONTROL: [u32; 8] = [2, 13, 3, 14, 3, 13, 14, 1];
/// The same prompt with the typed spelling as its characters.
const TYPED_AS_TEXT: [u32; 16] = [2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 3, 13, 14, 1];
/// [`TYPED_AS_THE_CONTROL`] with one more closing token behind the prompt.
const TYPED_AS_THE_CONTROL_AND_A_CLOSE: [u32; 9] = [2, 13, 3, 14, 3, 13, 14, 1, 3];

/// The closing token put together where it is written, behind the messages,
/// under CONDITION: no name carries it.
const CLOSE_IN_PLACE: &str = "{% if CONDITION %}{{ '[/INST' ~ suffix ~ ']' }}{% endif %}";

/// The messages of [`BODY`] with any message but the user's written through
/// SEARCH, an expression over the message's text.
const SEARCHES: &str = "{% for message in messages %}{% if message['role'] == 'user' %}\
    {{ open }}{{ message['content'] }}{{ close }}{% else %}{{ SEARCH }}{{ end }}\
    {% endif %}{% endfor %}";

fn checkpoint(dir: &Path, template: &str) -> HuggingFaceTokenizer {
    checkpoint_of(dir, TOKENIZER_JSON, template)
}

/// [`TOKENIZER_JSON`] with one more added token, `content`, that is not
/// special.
fn with_added_token(content: &str) -> String {
    let flags = r#""single_word": false, "lstrip": false, "rstrip": false, "normalized": false"#;
    let conversion = TOKENIZER_JSON
        .replace(
            r#""special": true}],"#,
            &format!(
                r#""special": true}},
        {{"id": 19, "content": "{content}", {flags}, "special": false}}],"#
            ),
        )
        .replace(r#"">": 18}"#, &format!(r#"">": 18, "{content}": 19}}"#));
    assert!(conversion.contains(r#""id": 19"#) && conversion.contains(r#"": 19}"#));
    conversion
}

/// `conversion` with `flag` set on the entry of its added token `content`.
fn with_flag(conversion: &str, content: &str, flag: &str) -> String {
    let entry = format!(
        r#""content": "{content}", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false"#
    );
    let flagged = entry.replace(
        &format!(r#""{flag}": false"#),
        &format!(r#""{flag}": true"#),
    );
    assert!(conversion.contains(&entry) && flagged != entry);
    conversion.replace(&entry, &flagged)
}

/// [`checkpoint`] over `conversion`, a tokenizer file.
fn checkpoint_of(dir: &Path, conversion: &str, template: &str) -> HuggingFaceTokenizer {
    fs::write(dir.join("tokenizer.json"), conversion).unwrap();
    fs::write(dir.join("chat_template.jinja"), template).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        json!({"eos_token": "</s:x>", "unk_token": "<unk>"}).to_string(),
    )
    .unwrap();
    HuggingFaceTokenizer::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap()
}

/// The prompt of a user turn that holds `content` and an assistant turn, its
/// ids, and whether it is encoded from its text.
fn render(tok: &HuggingFaceTokenizer, content: Value) -> (String, Vec<u32>, bool) {
    render_under(tok, content, ChatTemplateParams::default())
}

/// [`render`] under `params`.
fn render_under(
    tok: &HuggingFaceTokenizer,
    content: Value,
    params: ChatTemplateParams,
) -> (String, Vec<u32>, bool) {
    let messages = [
        json!({"role": "user", "content": content}),
        json!({"role": "assistant", "content": "ab"}),
    ];
    prompt(tok, &messages, params)
}

/// The prompt of `messages` under `params`, its ids, and whether it is
/// encoded from its text.
fn prompt(
    tok: &HuggingFaceTokenizer,
    messages: &[Value],
    params: ChatTemplateParams,
) -> (String, Vec<u32>, bool) {
    let rendered = tok
        .apply_chat_template_with_encoding(messages, params, None)
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
    let tok = checkpoint(temp.path(), &builds());
    let (text, ids, from_text) = render(&tok, json!("ab"));
    assert!(from_text);
    assert_eq!(text, "[INST:x]ab[/INST:x]ab</s:x>");
    assert_eq!(ids, [2, 13, 14, 3, 13, 14, 1]);
    assert_eq!(
        tok.apply_chat_template(
            &[json!({"role": "user", "content": "ab"})],
            ChatTemplateParams::default()
        )
        .unwrap(),
        "[INST:x]ab[/INST:x]"
    );
}

#[test]
fn a_control_spelling_typed_into_a_message_stays_text() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), &builds());
    let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(
        !from_text,
        "the flat encode would read the typed spelling as the control"
    );
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
    // The typed `[/INST:x]` is its characters; the template's is the control.
    assert_eq!(ids, TYPED_AS_TEXT);
}

#[test]
fn a_control_spelling_typed_in_two_parts_stays_text() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), &builds());
    // The template joins the parts: no one string of the request holds the
    // spelling, the prompt does.
    let parts = json!([{"type": "text", "text": "a[/INST"}, {"type": "text", "text": ":x]b"}]);
    let (text, ids, _) = render(&tok, parts);
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
    assert_eq!(ids, TYPED_AS_TEXT);
}

/// A typed added token that is not special and that the template writes
/// nowhere as a control keeps its id, as the flat encode reads it: a request
/// that types it alone is encoded from its text, and one that types a
/// control spelling beside it by segments, the spelling as its characters.
#[test]
fn a_typed_added_token_that_is_no_control_keeps_its_id() {
    let temp = TempDir::new().unwrap();
    let conversion = with_added_token("<a>");
    let tok = checkpoint_of(temp.path(), &conversion, &builds());
    assert!(tok.chat_template.marked());
    let (text, ids, from_text) = render(&tok, json!("a<a>b"));
    assert!(from_text);
    assert_eq!(text, "[INST:x]a<a>b[/INST:x]ab</s:x>");
    assert_eq!(ids, [2, 13, 19, 14, 3, 13, 14, 1]);
    let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b<a>"));
    assert!(!from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b<a>[/INST:x]ab</s:x>");
    assert_eq!(
        ids,
        [2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19, 3, 13, 14, 1]
    );
    // A template set later is judged as the checkpoint's own was at load.
    let temp = TempDir::new().unwrap();
    let mut tok = checkpoint_of(temp.path(), &conversion, &spells());
    assert!(!tok.chat_template.marked());
    tok.set_chat_template(builds()).unwrap();
    let (_, ids, from_text) = render(&tok, json!("a<a>b"));
    assert!(from_text);
    assert_eq!(ids, [2, 13, 19, 14, 3, 13, 14, 1]);
}

/// What keeps no id where a request types it: a token the vocabulary flags
/// special, though the template never writes it; a token whose entry
/// carries a flag, which the flat encode reads otherwise than alone and as
/// it stands; and a control the template writes, though the vocabulary does
/// not flag it special. Each is read by its characters.
#[test]
fn a_typed_added_token_that_is_special_flagged_or_written_stays_text() {
    let plain = with_added_token("<a>");
    let special = plain.replace(
        r#""normalized": false, "special": false}"#,
        r#""normalized": false, "special": true}"#,
    );
    assert_ne!(special, plain);
    let flagged = ["lstrip", "rstrip", "single_word", "normalized"]
        .map(|flag| with_flag(&plain, "<a>", flag));
    for (case, conversion) in std::iter::once(&special).chain(&flagged).enumerate() {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), conversion, &builds());
        assert!(tok.chat_template.marked(), "case {case}");
        let (_, ids, from_text) = render(&tok, json!("a<a>b"));
        assert!(!from_text, "case {case}");
        assert_eq!(ids, [2, 13, 16, 13, 18, 14, 3, 13, 14, 1], "case {case}");
    }
    let entry = r#""content": "[/INST:x]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": "#;
    let written = TOKENIZER_JSON.replace(&format!("{entry}true"), &format!("{entry}false"));
    assert_ne!(written, TOKENIZER_JSON);
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &written, &builds());
    assert!(tok.chat_template.marked());
    let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(!from_text);
    assert_eq!(ids, TYPED_AS_TEXT);
}

/// The tokens that keep their id are the template's: under a renderer with
/// an encoder of its own, which writes its controls itself, a typed added
/// token is read by its characters, also once the template is replaced.
#[test]
fn a_typed_added_token_is_text_under_a_renderer_with_its_own_encoder() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("config.json"),
        json!({"architectures": ["DeepseekV32ForCausalLM"]}).to_string(),
    )
    .unwrap();
    let mut tok = checkpoint_of(temp.path(), &with_added_token("<a>"), &builds());
    assert!(tok.chat_template.marked());
    for replaced in [false, true] {
        if replaced {
            tok.set_chat_template(builds()).unwrap();
        }
        let (_, ids, from_text) = render(&tok, json!("a<a>b"));
        assert!(!from_text, "replaced: {replaced}");
        assert!(
            ids.windows(5).any(|typed| typed == [13, 16, 13, 18, 14]),
            "replaced: {replaced}"
        );
        assert!(!ids.contains(&19), "replaced: {replaced}");
    }
}

/// A pipeline that matches an added token after normalization reads as
/// that token a text its normalizer turns into the spelling. Under it the
/// template is left as it was: a prompt is encoded from its text, and such
/// a text is the entry's token there, alone and beside a typed token, as
/// the flat encode reads it. With no normalizer, or an empty sequence of
/// them, the entry matches as it is spelled, and the template is marked;
/// so is the template of a vocabulary whose entries all match as they are
/// spelled, under any normalizer.
#[test]
fn a_pipeline_that_matches_an_added_token_after_normalization_leaves_the_template_as_it_was() {
    let conversion = with_added_token("<a>")
        .replace(
            r#""special": false}],"#,
            r#""special": false},
        {"id": 20, "content": "<b>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true}],"#,
        )
        .replace(r#""<a>": 19}"#, r#""<a>": 19, "<b>": 20}"#);
    assert!(conversion.contains(r#""id": 20"#) && conversion.contains(r#"": 20}"#));
    let under = |normalizer: &str| {
        let conversion = conversion.replace(
            r#""normalizer": null"#,
            &format!(r#""normalizer": {normalizer}"#),
        );
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &conversion, &builds());
        (temp, tok)
    };
    let (_temp, tok) = under(r#"{"type": "Lowercase"}"#);
    assert!(!tok.chat_template.marked());
    let (_, ids, from_text) = render(&tok, json!("<B>"));
    assert!(from_text);
    assert_eq!(ids, [2, 20, 3, 13, 14, 1]);
    let (_, ids, from_text) = render(&tok, json!("<a><B>"));
    assert!(from_text);
    assert_eq!(ids, [2, 19, 20, 3, 13, 14, 1]);
    for normalizer in ["null", r#"{"type": "Sequence", "normalizers": []}"#] {
        let (_temp, tok) = under(normalizer);
        assert!(tok.chat_template.marked(), "{normalizer}");
        let (_, ids, from_text) = render(&tok, json!("a<a>b"));
        assert!(from_text, "{normalizer}");
        assert_eq!(ids, [2, 13, 19, 14, 3, 13, 14, 1], "{normalizer}");
    }
    let lowered = with_added_token("<a>").replace(
        r#""normalizer": null"#,
        r#""normalizer": {"type": "Lowercase"}"#,
    );
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &lowered, &builds());
    assert!(tok.chat_template.marked());
    let (_, ids, from_text) = render(&tok, json!("a<a>b"));
    assert!(from_text);
    assert_eq!(ids, [2, 13, 19, 14, 3, 13, 14, 1]);
}

/// Under a pipeline that lowercases, an added token matched after
/// normalization and only as a whole word stands in the second of two
/// pieces side by side (`TAG`) and not in the run (`xTAG`). The template is
/// left as it is whatever it writes: that piece alone, the run, or neither.
#[test]
fn a_whole_word_token_in_one_of_two_joined_pieces_leaves_the_template_as_it_was() {
    let conversion = with_added_token("<a>")
        .replace(
            r#""special": false}],"#,
            r#""special": false},
        {"id": 20, "content": "tag", "single_word": true, "lstrip": false, "rstrip": false, "normalized": true, "special": true}],"#,
        )
        .replace(r#""<a>": 19}"#, r#""<a>": 19, "tag": 20}"#)
        .replace(
            r#""normalizer": null"#,
            r#""normalizer": {"type": "Lowercase"}"#,
        );
    assert!(conversion.contains(r#""id": 20"#) && conversion.contains("Lowercase"));
    let marked = |written: &str| {
        let template = format!("{}{{% if tagged %}}{written}{{% endif %}}", builds());
        let temp = TempDir::new().unwrap();
        checkpoint_of(temp.path(), &conversion, &template)
            .chat_template
            .marked()
    };
    assert!(!marked("{{ 'x' }}"));
    assert!(!marked("{{ 'TAG' }}"));
    assert!(!marked("{{ ('x' ~ 'TAG')[1:] }}"));
}

/// An added token matched after normalization and only as a whole word
/// ends a piece: the template is left as it is, whether that piece is the
/// last, a run of two pieces stands before it, or a piece follows it. So
/// is, under that pipeline, a template that writes no such piece.
#[test]
fn a_whole_word_token_at_the_end_of_a_piece_leaves_the_template_as_it_was() {
    let conversion = with_a_whole_word_entry();
    let marked = |before: &str, after: &str| {
        let template = [before, &builds(), after].concat();
        let temp = TempDir::new().unwrap();
        checkpoint_of(temp.path(), &conversion, &template)
            .chat_template
            .marked()
    };
    let last = "{% if tagged %}{{ ' TAG' }}{% endif %}";
    assert!(!marked("{{ 'x' ~ 'y' }}", ""));
    assert!(!marked("", last));
    assert!(!marked("{{ 'x' ~ 'y' }}", last));
    assert!(!marked("", &[last, "{{ 'x' }}"].concat()));
}

/// [`TOKENIZER_JSON`] with `<a>` and one more added token, `tag`, special,
/// matched after normalization and only as a whole word, under a
/// normalizer that lowercases.
fn with_a_whole_word_entry() -> String {
    let conversion = with_added_token("<a>")
        .replace(
            r#""special": false}],"#,
            r#""special": false},
        {"id": 20, "content": "tag", "single_word": true, "lstrip": false, "rstrip": false, "normalized": true, "special": true}],"#,
        )
        .replace(r#""<a>": 19}"#, r#""<a>": 19, "tag": 20}"#)
        .replace(
            r#""normalizer": null"#,
            r#""normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": false, "strip_accents": null, "lowercase": true}"#,
        );
    assert!(conversion.contains(r#""id": 20"#) && conversion.contains("BertNormalizer"));
    conversion
}

/// Whether the loader marks [`builds`] with `written` behind it under the
/// argument `tagged`, which no probe conversation passes, over
/// `conversion`. A template that is left as it is reads what it writes
/// there as the flat encode does: with the argument its prompt ends with
/// `text`, is encoded from its text and holds the added token `id`.
fn marked_with_a_tagged_write(conversion: &str, written: &str, text: &str, id: u32) -> bool {
    let template = format!("{}{{% if tagged %}}{written}{{% endif %}}", builds());
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), conversion, &template);
    let marked = tok.chat_template.marked();
    if !marked {
        let kwargs = HashMap::from([("tagged".to_owned(), json!(true))]);
        let tagged = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        let (prompt, ids, from_text) = render_under(&tok, json!("a"), tagged);
        assert!(from_text, "{written}");
        assert!(prompt.ends_with(text), "{written}: {prompt}");
        assert!(ids.contains(&id), "{written}: {ids:?}");
    }
    marked
}

/// Every piece a template writes is asked once more glued to the others,
/// in the order of the source with nothing between. Two pieces, each in
/// parentheses of its own, are no run, and they spell together a text an
/// added token matches: the template is left as it is, and one whose
/// pieces spell no token there is marked. Under a pipeline that matches
/// the token after normalization the template is left as it is too, and
/// the text its pieces spell is the entry's token.
#[test]
fn a_token_two_pieces_spell_across_parentheses_leaves_the_template_as_it_was() {
    let as_spelled = with_added_token("<a>");
    let marked =
        |written: &str, text: &str| marked_with_a_tagged_write(&as_spelled, written, text, 19);
    assert!(marked("{{ ('<') ~ ('b>') }}", "<b>"));
    assert!(!marked("{{ ('<') ~ ('a>') }}", "<a>"));
    let conversion = with_a_whole_word_entry();
    let marked =
        |written: &str, text: &str| marked_with_a_tagged_write(&conversion, written, text, 20);
    assert!(!marked("{{ (' T') ~ ('AG ') }}", " TAG "));
}

/// A block tag stands between two pieces, and a render that takes the
/// block writes the second right after the first: glued, they hold the
/// token they spell together, and the template is left as it is. Under a
/// pipeline that matches the token after normalization the template is
/// left as it is too, and the text its pieces spell is the entry's token.
#[test]
fn a_token_two_pieces_spell_across_a_block_tag_leaves_the_template_as_it_was() {
    let conversion = with_a_whole_word_entry();
    let marked =
        |written: &str, text: &str| marked_with_a_tagged_write(&conversion, written, text, 20);
    assert!(!marked(" T{% if tagged %}AG {% endif %}", " TAG "));
    let as_spelled = with_added_token("<a>");
    let marked =
        |written: &str, text: &str| marked_with_a_tagged_write(&as_spelled, written, text, 19);
    assert!(marked("<{% if tagged %}b>{% endif %}", "<b>"));
    assert!(!marked("<{% if tagged %}a>{% endif %}", "<a>"));
}

/// An expression that comes out empty stands between two pieces, and the
/// render writes the second right after the first: glued, they hold the
/// token they spell together, and the template is left as it is. Under a
/// pipeline that matches the token after normalization the template is
/// left as it is too, and the text its pieces spell is the entry's token.
#[test]
fn a_token_two_pieces_spell_around_an_empty_expression_leaves_the_template_as_it_was() {
    let as_spelled = with_added_token("<a>");
    let marked =
        |written: &str, text: &str| marked_with_a_tagged_write(&as_spelled, written, text, 19);
    assert!(marked("<{{ note }}b>", "<b>"));
    assert!(!marked("<{{ note }}a>", "<a>"));
    let conversion = with_a_whole_word_entry();
    let marked =
        |written: &str, text: &str| marked_with_a_tagged_write(&conversion, written, text, 20);
    assert!(!marked(" T{{ note }}AG ", " TAG "));
}

/// A render writes the pieces of a template in an order of its own: it
/// leaves a piece out, takes one arm of a branch and not the other, writes
/// the body of a loop again, or writes a macro's text where the macro is
/// called. Two pieces that stand apart in the source then spell together,
/// with whole pieces between them or none, a text an added token matches,
/// and the template is left as it is: the token it writes there is read as
/// the flat encode reads it, whether its entry is special or not. A
/// template whose pieces spell no token in any order is marked.
#[test]
fn a_token_pieces_spell_in_an_order_of_the_render_leaves_the_template_as_it_was() {
    let plain = with_added_token("<a>");
    let special = plain.replace(
        r#""normalized": false, "special": false}"#,
        r#""normalized": false, "special": true}"#,
    );
    assert!(special != plain);
    for conversion in [&plain, &special] {
        let marked =
            |written: &str, text: &str| marked_with_a_tagged_write(conversion, written, text, 19);
        // A piece between the two is left out.
        assert!(marked("<{% if extra %}x{% endif %}b>", "<b>"));
        assert!(!marked("<{% if extra %}x{% endif %}a>", "<a>"));
        // One arm of a branch is taken, and the piece of the other is not
        // written.
        assert!(!marked(
            "{% if not extra %}<{% else %}x{% endif %}a>",
            "<a>"
        ));
        // A whole piece stands between the two.
        assert!(!marked(
            "<{% if extra %}x{% endif %}{{ 'a' }}{% if extra %}y{% endif %}>",
            "<a>"
        ));
        // The body of a loop is written again: its end, then its start.
        assert!(marked("{% for n in [1, 2] %}b>a<{% endfor %}", "b>a<b>a<"));
        assert!(!marked("{% for n in [1, 2] %}a>b<{% endfor %}", "a>b<a>b<"));
        // A macro's text is written where the macro is called.
        assert!(!marked(
            "{% macro tail() %}a>{% endmacro %}<{{ tail() }}",
            "<a>"
        ));
    }
}

/// Under a normalizer that lowercases, an added token is matched after
/// normalization. A template that writes two pieces around a branch a
/// request leaves out is left as it is: with the argument its prompt is
/// encoded from its text, and the text the pieces write there, which the
/// normalizer turns into the entry's spelling, is the entry's token.
#[test]
fn pieces_around_a_branch_under_a_normalizer_that_lowercases_leave_the_template_as_it_was() {
    let conversion = with_a_normalized_entry().replace(
        r#""normalizer": null"#,
        r#""normalizer": {"type": "Lowercase"}"#,
    );
    assert!(conversion.contains("Lowercase"));
    assert!(!marked_with_a_tagged_write(
        &conversion,
        "<{% if extra %}x{% endif %}B>",
        "<B>",
        19
    ));
}

/// The same under a pipeline that matches the added token after
/// normalization and only as a whole word: the pieces stand around a
/// branch a request leaves out, the template is left as it is, and the
/// word they write there is the entry's token.
#[test]
fn pieces_of_a_whole_word_around_a_branch_leave_the_template_as_it_was() {
    let conversion = with_a_whole_word_entry();
    assert!(!marked_with_a_tagged_write(
        &conversion,
        " T{% if extra %}x{% endif %}AG ",
        " TAG ",
        20
    ));
}

/// Under a normalizer that strips the edges of a text, an added token
/// whose spelling begins with a blank is matched after normalization. A
/// template is left as it is under it, whether it writes such a text or
/// not. One that writes, behind a block tag, a text the normalizer turns
/// into that spelling: with the argument its prompt is encoded from its
/// text, and the text is the entry's token.
#[test]
fn a_token_with_a_blank_edge_under_a_normalizer_that_strips_leaves_the_template_as_it_was() {
    let conversion = with_added_token("<a>")
        .replace(
            r#""special": false}],"#,
            r#""special": false},
        {"id": 20, "content": " tag", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true}],"#,
        )
        .replace(r#""<a>": 19}"#, r#""<a>": 19, " tag": 20}"#)
        .replace(
            r#""normalizer": null"#,
            r#""normalizer": {"type": "Sequence", "normalizers": [{"type": "Strip", "strip_left": true, "strip_right": true}, {"type": "Lowercase"}]}"#,
        );
    assert!(conversion.contains(r#""id": 20"#) && conversion.contains("Strip"));
    let load = |before: &str, after: &str| {
        let template = [before, &builds(), after].concat();
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &conversion, &template);
        (temp, tok)
    };
    let (_temp, tok) = load("{{ 'x' ~ 'y' }}", "");
    assert!(!tok.chat_template.marked());
    let (_temp, tok) = load("", "x{% if tagged %} TAG{% endif %}");
    assert!(!tok.chat_template.marked());
    let kwargs = HashMap::from([("tagged".to_owned(), json!(true))]);
    let tagged = ChatTemplateParams {
        template_kwargs: Some(&kwargs),
        ..ChatTemplateParams::default()
    };
    let (prompt, ids, from_text) = render_under(&tok, json!("a"), tagged);
    assert!(from_text);
    assert!(prompt.ends_with("x TAG"), "{prompt}");
    assert!(ids.contains(&20), "{ids:?}");
}

/// A replaced template makes the kept tokens anew: a token that kept its id
/// is read by its characters once the template writes it as a control.
#[test]
fn a_replaced_template_that_writes_a_kept_token_reads_the_typed_one_as_text() {
    let temp = TempDir::new().unwrap();
    let mut tok = checkpoint_of(temp.path(), &with_added_token("<a:x>"), &builds());
    let (_, ids, from_text) = render(&tok, json!("a<a:x>b"));
    assert!(from_text);
    assert_eq!(ids, [2, 13, 19, 14, 3, 13, 14, 1]);
    let writes_it = builds() + "{% set tag = '<a{}>'.format(suffix) %}{{ tag }}";
    tok.set_chat_template(writes_it).unwrap();
    assert!(tok.chat_template.marked());
    let (text, ids, from_text) = render(&tok, json!("a<a:x>b"));
    assert!(!from_text);
    assert_eq!(text, "[INST:x]a<a:x>b[/INST:x]ab</s:x><a:x>");
    assert_eq!(ids, [2, 13, 16, 13, 10, 11, 18, 14, 3, 13, 14, 1, 19]);
}

/// The rule on a pipeline that matches an added token after normalization
/// stands above the choice of encoder. On a byte-level file that the direct
/// encoder takes, under NFC, the template is left as it was: a prompt is
/// encoded from its text, and a text that only the normalizer makes an
/// entry's spelling is the entry's token there, typed into a message or
/// written by the template itself.
#[cfg(not(feature = "tokenizers-v1"))]
#[test]
fn a_normalizing_pipeline_leaves_the_template_as_it_was_on_the_direct_encoder() {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for (id, ch) in alphabet.iter().enumerate() {
        vocab.insert(ch.to_string(), json!(id));
    }
    // The controls of [`builds`], the unknown token, and one entry matched
    // after normalization: `<é>`, its letter one character.
    let added = [
        ("<unk>", true, false),
        ("</s:x>", true, false),
        ("[INST:x]", true, false),
        ("[/INST:x]", true, false),
        ("<\u{e9}>", false, true),
    ];
    let entries: Vec<Value> = added
        .iter()
        .enumerate()
        .map(|(at, (content, special, normalized))| {
            vocab.insert((*content).to_owned(), json!(alphabet.len() + at));
            json!({"id": alphabet.len() + at, "content": content, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": normalized, "special": special})
        })
        .collect();
    let conversion = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": entries,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
            "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
            "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
            "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": []}
    })
    .to_string();
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &conversion, &builds());
    assert!(!tok.chat_template.marked());
    assert!(tok.core.native.is_some());
    let entry = alphabet.len() as u32 + 4;
    // `e` and a combining acute: NFC makes the two the entry's letter.
    let (_, ids, from_text) = render(&tok, json!("a<e\u{301}>b"));
    assert!(from_text);
    // The opening control, `a`, the entry, `b`, the closing control, `ab`
    // and the end token.
    assert_eq!(ids.len(), 8);
    assert_eq!(ids[2], entry);
    // The same text written by the template itself: the flat encode reads
    // the entry there too.
    let writes_it = template(
        CONSTANTS,
        "{{ open }}",
        "{{ close }}<e\u{301}>",
        "{{ end }}",
    );
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &conversion, &writes_it);
    assert!(!tok.chat_template.marked());
    let (_, ids, from_text) = render(&tok, json!("ab"));
    assert!(from_text);
    assert!(ids.contains(&entry));
    // A text the normalizer shortens into no entry, written before a
    // control: the template is left as it was all the same.
    let shortened = template(CONSTANTS, "e\u{301}{{ open }}", "{{ close }}", "{{ end }}");
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &conversion, &shortened);
    assert!(!tok.chat_template.marked());
}

#[test]
fn a_mark_typed_into_a_message_is_no_mark() {
    let temp = TempDir::new().unwrap();
    let tok = checkpoint(temp.path(), &builds());
    let (text, ids, _) = render(&tok, json!("a\u{FDD0}[/INST:x]\u{FDD1}b"));
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
    assert_eq!(ids, TYPED_AS_TEXT);
}

#[test]
fn a_control_built_where_it_is_written_leaves_the_template_as_it_was() {
    let temp = TempDir::new().unwrap();
    // The closing token is put together in the output expression: no name
    // carries it, so the marks would miss a control the template writes.
    let in_place = template(
        CONSTANTS,
        "{{ open }}",
        "{{ '[/INST' ~ suffix ~ ']' }}",
        "{{ end }}",
    );
    let tok = checkpoint(temp.path(), &in_place);
    let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
    assert_eq!(ids, TYPED_AS_THE_CONTROL);
}

#[test]
fn a_template_that_spells_a_control_whole_is_left_as_it_was() {
    // The end token spelled whole where it is written: beside a name that
    // carries it, then with no such name.
    let without_end = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
        {% set close = '[/INST{}]'.format(suffix) %}";
    for (constants, names_it) in [(CONSTANTS, true), (without_end, false)] {
        let temp = TempDir::new().unwrap();
        let spells_one = template(constants, "{{ open }}", "{{ close }}", "</s:x>");
        let tok = checkpoint(temp.path(), &spells_one);
        // A name whose spelling the source holds says by itself that the
        // template is of the other kind: no matcher is built to find it.
        assert!(!names_it || tok.core.added.get().is_none());
        let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(from_text);
        assert_eq!(ids, TYPED_AS_THE_CONTROL);
    }
}

/// [`CONSTANTS`] with the closing token spelled whole in a statement of its
/// own.
const CLOSE_SPELLED: &str = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
    {% set close = '[/INST:x]' %}{% set end = '</s{}>'.format(suffix) %}";

#[test]
fn a_control_spelled_whole_in_its_own_statement_is_bound_beside_built_ones() {
    // One statement spells its control whole in a literal of its own, the
    // others build theirs, and the template writes each by its name, alone
    // or in a join. The template writes that literal only as the value of
    // its statement: the spelled control is bound like the built ones.
    let named = template(CLOSE_SPELLED, "{{ open }}", "{{ close }}", "{{ end }}");
    let joined = shaped(
        "{% set shut = '[/INST:x]' %}",
        "",
        "{{ open ~ message['content'] ~ shut }}",
    );
    for template in [named, joined] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint(temp.path(), &template);
        assert!(tok.chat_template.marked(), "{template}");
        let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(!from_text, "{template}");
        assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>", "{template}");
        assert_eq!(ids, TYPED_AS_TEXT, "{template}");
        // A request that types no spelling is encoded from its text.
        let (_, ids, from_text) = render(&tok, json!("a"));
        assert!(from_text, "{template}");
        assert_eq!(ids, [2, 13, 3, 13, 14, 1], "{template}");
    }
}

#[test]
fn a_spelled_control_is_bound_where_its_name_is_only_written() {
    // The name of a statement that spells its control whole is watched like
    // the name of a built value: a search, a comparison, a filter, a
    // subscript or a loop over it, wherever it stands, leaves the template
    // as it was. So does the same spelling written whole anywhere else, in
    // another literal or in the template's text, and so does its name
    // written where a built value is also made from it: the first read of
    // the syntax takes it for a part of that value.
    let spelled = "{% set shut = '[/INST:x]' %}";
    let on = switched_on(&[
        (
            "a search",
            spelled,
            "",
            "{% if shut in messages[0]['content'] %}a{% endif %}",
        ),
        (
            "a comparison",
            spelled,
            "",
            "{% if messages[0]['content'] == shut %}a{% endif %}",
        ),
        ("a filter", spelled, "", "{{ shut | length }}"),
        ("a subscript", spelled, "", "{{ shut[0] }}"),
        (
            "a loop over it",
            spelled,
            "",
            "{% for c in shut %}{{ c }}{% endfor %}",
        ),
        (
            "the spelling in another literal",
            spelled,
            "",
            "{{ shut ~ '[/INST:x]' }}",
        ),
        (
            "the spelling in the text",
            spelled,
            "",
            "{{ shut }}[/INST:x]",
        ),
        (
            "a built value made from it beside its name written",
            "{% set shut = '[/INST:x]' %}{% set turn = shut ~ end %}",
            "",
            "{{ shut }}{{ turn }}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_spelled_control_is_bound_in_a_template_that_builds_a_value() {
    // Every control spelled whole in a statement of its own and written by
    // its name: no statement builds a value, so the load reads no statement
    // of this template, and it is left as it was.
    let constants =
        "{% set open = '[INST:x]' %}{% set close = '[/INST:x]' %}{% set end = '</s:x>' %}";
    let temp = TempDir::new().unwrap();
    let spells_all = template(constants, "{{ open }}", "{{ close }}", "{{ end }}");
    let tok = checkpoint(temp.path(), &spells_all);
    assert!(!tok.chat_template.marked());
    assert!(tok.core.added.get().is_none());
    let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(ids, TYPED_AS_THE_CONTROL);
}

#[test]
fn a_control_that_a_longer_added_token_holds_leaves_the_template_as_it_was() {
    // The flat encode takes the added tokens leftmost and longest. Where the
    // text beside a control the template writes completes a longer added
    // token, that token is read; between marks the control would be. So a
    // token that begins with one of the template's controls and goes on,
    // that ends with one, that holds one, or that ends inside one leaves the
    // template as it was, and no matcher is built to tell: the prompt reads
    // as the flat encode reads it, the longer token (19) where the text
    // completes it.
    for (content, flat) in [
        ("[/INST:x]a", vec![2, 13, 3, 14, 19, 14, 1]),
        ("b[/INST:x]", vec![2, 13, 3, 19, 13, 14, 1]),
        ("b[/INST:x]a", vec![2, 13, 3, 19, 14, 1]),
        ("b[/INST", vec![2, 13, 3, 19, 10, 11, 12, 13, 14, 1]),
    ] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &with_added_token(content), &builds());
        assert!(!tok.chat_template.marked(), "{content}");
        assert!(tok.core.added.get().is_none(), "{content}");
        let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
        assert!(from_text, "{content}");
        assert_eq!(ids, flat, "{content}");
    }
    // A token that stands inside a control, that is its beginning, or that
    // begins inside it never takes its place: the control starts first, or
    // is the longer. Those templates are marked.
    for content in ["T:x", "[/INST:", ":x]a"] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &with_added_token(content), &builds());
        assert!(tok.chat_template.marked(), "{content}");
        let (_, ids, from_text) = render(&tok, json!("a[INST:x]b"));
        assert!(!from_text, "{content}");
        assert_eq!(
            ids,
            [2, 13, 4, 6, 7, 8, 9, 10, 11, 12, 14, 3, 13, 14, 1],
            "{content}"
        );
    }
    // The same for a special token the template writes by the name a render
    // gives it: no statement binds the end token here.
    let constants = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
        {% set close = '[/INST{}]'.format(suffix) %}";
    let by_name = template(constants, "{{ open }}", "{{ close }}", "{{ eos_token }}");
    for (content, marked) in [("</s:x>a", false), ("s:x", true)] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &with_added_token(content), &by_name);
        assert_eq!(tok.chat_template.marked(), marked, "{content}");
    }
}

#[test]
fn a_control_whose_entry_carries_a_flag_leaves_the_template_as_it_was() {
    // The closing token takes the whitespace behind it or before it, matches
    // only as a whole word, or is matched after normalization: the flat
    // encode reads more than its characters there, or none of them, or
    // reads it in a later pass, and between marks it would be read as
    // itself. The template is left as it was.
    for flag in ["rstrip", "lstrip", "single_word", "normalized"] {
        let temp = TempDir::new().unwrap();
        let conversion = with_flag(TOKENIZER_JSON, "[/INST:x]", flag);
        let tok = checkpoint_of(temp.path(), &conversion, &builds());
        assert!(!tok.chat_template.marked(), "{flag}");
        assert!(tok.core.added.get().is_none(), "{flag}");
        let (text, ids, from_text) = render(&tok, json!("a[INST:x]b"));
        assert!(from_text, "{flag}");
        let flat = tok.encode(&text, false).unwrap().into_token_ids();
        assert_eq!(ids, flat, "{flag}");
    }
}

#[test]
fn a_control_matched_after_a_lowering_normalizer_leaves_the_template_as_it_was() {
    // The file lowers the text, and the opening token is matched after
    // normalization: the flat encode reads it wherever the lowered text
    // holds its lowered spelling, and between marks only the spelling as it
    // stands would be read. The template is left as it was, and the prompt
    // reads as the flat encode reads it.
    let lowers = TOKENIZER_JSON.replace(
        r#""normalizer": null"#,
        r#""normalizer": {"type": "Lowercase"}"#,
    );
    assert_ne!(lowers, TOKENIZER_JSON);
    let temp = TempDir::new().unwrap();
    let conversion = with_flag(&lowers, "[INST:x]", "normalized");
    let tok = checkpoint_of(temp.path(), &conversion, &builds());
    assert!(!tok.chat_template.marked());
    assert!(tok.core.added.get().is_none());
    let (text, ids, from_text) = render(&tok, json!("a[inst:x]b"));
    assert_eq!(text, "[INST:x]a[inst:x]b[/INST:x]ab</s:x>");
    assert!(from_text);
    assert_eq!(ids, [2, 13, 2, 14, 3, 13, 14, 1]);
}

#[test]
fn a_token_that_begins_inside_a_normalized_control_leaves_the_template_as_it_was() {
    // The closing token is matched after normalization, and an added token
    // that is matched as it stands begins inside it. The flat encode takes
    // the tokens matched as they stand first, so where the text behind the
    // closing token completes that one (19), it is read; between marks the
    // closing token would be. The template is left as it was, and what it
    // writes reads the same whatever the request holds.
    let conversion = with_flag(&with_added_token("x]b"), "[/INST:x]", "normalized");
    let source = format!(
        "{}{{% if flat %}}{{{{ close ~ 'b' }}}}{{% endif %}}",
        builds()
    );
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &conversion, &source);
    assert!(!tok.chat_template.marked());
    let flat = HashMap::from([("flat".to_owned(), json!(true))]);
    for content in ["a[INST:x]b", "ab"] {
        let params = ChatTemplateParams {
            template_kwargs: Some(&flat),
            ..ChatTemplateParams::default()
        };
        let (text, ids, from_text) = render_under(&tok, json!(content), params);
        assert!(from_text, "{content}");
        assert!(text.ends_with("</s:x>[/INST:x]b"), "{content}");
        assert!(ids.ends_with(&[1, 4, 5, 6, 7, 8, 9, 10, 19]), "{content}");
    }
}

#[test]
fn a_control_spelled_in_a_comment_is_nothing_the_template_writes() {
    // A comment names a control: at the top, in the loop's body, beside a
    // second name for that control. Or the pattern of a value that is built
    // holds one whole. The template writes none of these, so it is marked
    // all the same. A string literal outside the statements that build is a
    // write, whatever reads it: such a template is left as it was.
    for (at_top, in_loop, marked) in [
        ("{# the closing token is [/INST:x] #}", "", true),
        ("", "{# [INST:x] opens a turn and </s:x> ends one #}", true),
        ("{# [/INST:x] #}{% set shut = close %}", "", true),
        ("{% set shut = '[/INST:x]' if suffix else '' %}", "", true),
        ("{% if flat %}{{ '</s:x>' }}{% endif %}", "", false),
        (
            "",
            "{% if message['content'] == '[/INST:x]' %}{% endif %}",
            false,
        ),
    ] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint(temp.path(), &shaped(at_top, in_loop, WRITTEN));
        assert_eq!(tok.chat_template.marked(), marked, "{at_top}{in_loop}");
        let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
        assert_eq!(from_text, !marked, "{at_top}{in_loop}");
        let typed: &[u32] = if marked {
            &TYPED_AS_TEXT
        } else {
            &TYPED_AS_THE_CONTROL
        };
        assert_eq!(ids, typed, "{at_top}{in_loop}");
    }
}

#[test]
fn a_control_that_begins_with_another_added_token_is_marked() {
    // Of two added tokens that begin at one place the flat encode takes the
    // longer, so a token that is the beginning of a control the template
    // writes never takes its place. The template is marked, and the control
    // reads as itself from the text of a request that types no spelling and
    // between the marks of one that does, on every load of the same files.
    for content in ["[/INST:", "</s:"] {
        let temp = TempDir::new().unwrap();
        let conversion = with_added_token(content);
        for load in 0..32 {
            let tok = checkpoint_of(temp.path(), &conversion, &builds());
            assert!(tok.chat_template.marked(), "{content}, load {load}");
            let (_, ids, from_text) = render(&tok, json!("ab"));
            assert!(from_text, "{content}, load {load}");
            assert_eq!(ids, [2, 13, 14, 3, 13, 14, 1], "{content}, load {load}");
            let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
            assert!(!from_text, "{content}, load {load}");
            assert_eq!(ids, TYPED_AS_TEXT, "{content}, load {load}");
        }
    }
}

#[test]
fn an_added_token_in_the_pattern_of_a_built_control_is_nothing_the_template_writes() {
    // `/INST` is an added token of its own, and the pattern the closing
    // token is built from holds it whole. The template writes the closing
    // token there and never that one: it is marked, and a prompt reads as
    // the flat encode reads it, which takes the token that starts first.
    let conversion = with_added_token("/INST");
    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &conversion, &builds());
    assert!(tok.chat_template.marked());
    let (text, ids, _) = render(&tok, json!("ab"));
    assert_eq!(ids, [2, 13, 14, 3, 13, 14, 1]);
    assert_eq!(ids, tok.encode(&text, false).unwrap().into_token_ids());
    // A typed opening token is its characters.
    let (text, ids, from_text) = render(&tok, json!("a[INST:x]b"));
    assert!(!from_text);
    assert_eq!(text, "[INST:x]a[INST:x]b[/INST:x]ab</s:x>");
    assert_eq!(ids, [2, 13, 4, 6, 7, 8, 9, 10, 11, 12, 14, 3, 13, 14, 1]);
    // Where the template writes that added token whole by itself, it is of
    // the other kind and is left as it was.
    let temp = TempDir::new().unwrap();
    let writes_it = builds() + "{% if flat %}/INST{% endif %}";
    let tok = checkpoint_of(temp.path(), &conversion, &writes_it);
    assert!(!tok.chat_template.marked());
    let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(ids, TYPED_AS_THE_CONTROL);
}

#[test]
fn a_replaced_template_is_judged_anew() {
    let temp = TempDir::new().unwrap();
    let mut tok = checkpoint(temp.path(), &spells());
    assert_eq!(render(&tok, json!("a[/INST:x]b")).1, TYPED_AS_THE_CONTROL);
    // A template without a control constant never asks for the matcher over
    // the added tokens: not at load, not for a prompt encoded from its text.
    assert!(tok.core.added.get().is_none());
    tok.set_chat_template(builds()).unwrap();
    assert_eq!(render(&tok, json!("a[/INST:x]b")).1, TYPED_AS_TEXT);
    tok.set_chat_template(spells()).unwrap();
    assert_eq!(render(&tok, json!("a[/INST:x]b")).1, TYPED_AS_THE_CONTROL);
}

/// A checkpoint over `conversion`, a tokenizer file, with a named pair of
/// templates: the default one builds its controls, `tool_use` is the other.
fn named_pair(dir: &Path, conversion: &str, tool_use: &str) -> HuggingFaceTokenizer {
    fs::write(dir.join("tokenizer.json"), conversion).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        json!({"eos_token": "</s:x>", "unk_token": "<unk>", "chat_template": [
            {"name": "default", "template": builds()},
            {"name": "tool_use", "template": tool_use},
        ]})
        .to_string(),
    )
    .unwrap();
    HuggingFaceTokenizer::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap()
}

#[test]
fn a_named_pair_of_templates_is_marked_together_or_not_at_all() {
    let pair = |tool_use: String| {
        let temp = TempDir::new().unwrap();
        let tok = named_pair(temp.path(), TOKENIZER_JSON, &tool_use);
        let tools = [json!({"type": "function", "function": {"name": "f"}})];
        let with_tools = ChatTemplateParams {
            tools: Some(tools.as_slice()),
            ..ChatTemplateParams::default()
        };
        let messages = [json!({"role": "user", "content": "a[/INST:x]b"})];
        let ids = |params| {
            let rendered = tok
                .apply_chat_template_with_encoding(&messages, params, None)
                .unwrap();
            match rendered.encoding {
                PromptEncoding::FromText => {
                    tok.encode(&rendered.text, false).unwrap().into_token_ids()
                }
                PromptEncoding::Deferred(job) => job.run().unwrap().into_token_ids(),
            }
        };
        (ids(ChatTemplateParams::default()), ids(with_tools))
    };
    // `[INST:x]a[/INST:x]b[/INST:x]`, through either template of the pair.
    let (typed_as_the_control, typed_as_text) = (
        vec![2, 13, 3, 14, 3],
        vec![2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 3],
    );
    assert_eq!(pair(builds()), (typed_as_text.clone(), typed_as_text));
    assert_eq!(
        pair(spells()),
        (typed_as_the_control.clone(), typed_as_the_control)
    );
}

/// A control that only the second template of a named pair builds is read
/// against the vocabulary with the default template's: where a longer added
/// token begins with it, neither template is marked and no matcher is
/// built; without that token both are marked.
#[test]
fn a_control_of_a_second_named_template_is_read_against_the_vocabulary() {
    let tool_use =
        builds() + "{% set call = '[CALL{}]'.format(suffix) %}{% if tools %}{{ call }}{% endif %}";
    let call = with_added_token("[CALL:x]");
    let longer = call
        .replace(
            r#""special": false}],"#,
            r#""special": false},
        {"id": 20, "content": "[CALL:x]a", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}],"#,
        )
        .replace(r#""[CALL:x]": 19}"#, r#""[CALL:x]": 19, "[CALL:x]a": 20}"#);
    assert!(longer.contains(r#""id": 20"#) && longer.contains(r#"": 20}"#));
    for (conversion, marked) in [(&call, true), (&longer, false)] {
        let temp = TempDir::new().unwrap();
        let tok = named_pair(temp.path(), conversion, &tool_use);
        assert_eq!(tok.chat_template.marked(), marked);
        assert!(marked || tok.core.added.get().is_none());
    }
}

#[test]
fn a_constant_bound_under_a_condition_is_marked() {
    let temp = TempDir::new().unwrap();
    // One spelling is bound under a top-level `if`, one under an `elif`.
    let constants = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
        {% if suffix %}{% set close = '[/INST{}]'.format(suffix) %}{% endif %}\
        {% if not suffix %}{% elif suffix %}{% set end = '</s{}>'.format(suffix) %}{% endif %}";
    let conditional = template(constants, "{{ open }}", "{{ close }}", "{{ end }}");
    let tok = checkpoint(temp.path(), &conditional);
    assert_eq!(render(&tok, json!("a[/INST:x]b")).1, TYPED_AS_TEXT);
}

#[test]
fn a_control_built_in_place_without_the_generation_prompt_leaves_the_template_as_it_was() {
    let temp = TempDir::new().unwrap();
    // Written only where no generation prompt is asked for, as in this
    // render: the marks would miss a control the template writes there.
    let in_place = builds() + &CLOSE_IN_PLACE.replace("CONDITION", "not add_generation_prompt");
    let tok = checkpoint(temp.path(), &in_place);
    let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>[/INST:x]");
    assert_eq!(ids, TYPED_AS_THE_CONTROL_AND_A_CLOSE);
}

#[test]
fn a_control_built_in_place_under_the_thinking_switch_leaves_the_template_as_it_was() {
    let temp = TempDir::new().unwrap();
    // Written only with thinking switched off, a value the gateway passes
    // by itself.
    let in_place = builds()
        + &CLOSE_IN_PLACE.replace(
            "CONDITION",
            "enable_thinking is defined and not enable_thinking",
        );
    let tok = checkpoint(temp.path(), &in_place);
    let off = ChatTemplateParams {
        thinking: Some(false),
        ..ChatTemplateParams::default()
    };
    let (text, ids, from_text) = render_under(&tok, json!("a[/INST:x]b"), off);
    assert!(from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>[/INST:x]");
    assert_eq!(ids, TYPED_AS_THE_CONTROL_AND_A_CLOSE);
}

#[test]
fn a_template_that_searches_request_text_for_a_constant_is_left_as_it_was() {
    // What a template does with a message that replays its own markup: cut
    // it at a spelling, drop the spelling, test for it. Between marks the
    // constant would no longer be found in the text a request sends, and the
    // prompt would keep what the template drops. Each search with what it
    // leaves of a history message `a[/INST:x]b`.
    for (search, left) in [
        ("message['content'].split(close)[-1]", "b"),
        ("message['content'] | replace(close, '')", "ab"),
        (
            "'' if message['content'].startswith(close) else message['content']",
            "a[/INST:x]b",
        ),
        (
            "'' if message['content'].endswith(close) else message['content']",
            "a[/INST:x]b",
        ),
        (
            "'' if message['content'].startswith(open) and message['content'].endswith(close) \
             else message['content']",
            "a[/INST:x]b",
        ),
        (
            "'' if close in message['content'] else message['content']",
            "",
        ),
    ] {
        let temp = TempDir::new().unwrap();
        let searches = format!("{CONSTANTS}{}", SEARCHES.replace("SEARCH", search));
        let tok = checkpoint(temp.path(), &searches);
        let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(from_text, "{search}");
        assert_eq!(ids, TYPED_AS_THE_CONTROL, "{search}");
        let history = [
            json!({"role": "user", "content": "ab"}),
            json!({"role": "assistant", "content": "a[/INST:x]b"}),
        ];
        let (text, _, _) = prompt(&tok, &history, ChatTemplateParams::default());
        assert_eq!(text, format!("[INST:x]ab[/INST:x]{left}</s:x>"), "{search}");
    }
}

#[test]
fn a_processor_s_template_is_marked_with_the_default_one_or_not_at_all() {
    let beside = |processor: String| {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("chat_template.json"),
            json!({"chat_template": processor}).to_string(),
        )
        .unwrap();
        let tok = checkpoint(temp.path(), &builds());
        let tools = [json!({"type": "function", "function": {"name": "f"}})];
        let with_tools = ChatTemplateParams {
            tools: Some(tools.as_slice()),
            ..ChatTemplateParams::default()
        };
        let messages = [json!({"role": "user", "content": "a[/INST:x]b"})];
        let ids = |params| {
            let rendered = tok
                .apply_chat_template_with_encoding(&messages, params, None)
                .unwrap();
            match rendered.encoding {
                PromptEncoding::FromText => {
                    tok.encode(&rendered.text, false).unwrap().into_token_ids()
                }
                PromptEncoding::Deferred(job) => job.run().unwrap().into_token_ids(),
            }
        };
        // Without tools the processor's template renders, with tools the
        // default one.
        (ids(ChatTemplateParams::default()), ids(with_tools))
    };
    // `[INST:x]a[/INST:x]b[/INST:x]`, through either template.
    let (typed_as_the_control, typed_as_text) = (
        vec![2, 13, 3, 14, 3],
        vec![2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 3],
    );
    assert_eq!(
        beside(builds() + "{# the processor's #}"),
        (typed_as_text.clone(), typed_as_text)
    );
    assert_eq!(
        beside(spells()),
        (typed_as_the_control.clone(), typed_as_the_control)
    );
}

#[test]
fn a_constant_that_holds_another_control_under_a_request_argument_is_marked() {
    let temp = TempDir::new().unwrap();
    // `close` is bound to the end token when a request passes `flat`, an
    // argument the load never passes, and no other name holds that token:
    // the template writes a value the load never saw as a control too. Both
    // values are built from the template's own text.
    let constants = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
        {% if flat %}{% set close = '</s{}>'.format(suffix) %}{% else %}\
        {% set close = '[/INST{}]'.format(suffix) %}{% endif %}";
    let varies = template(constants, "{{ open }}", "{{ close }}", "{{ close }}");
    let tok = checkpoint(temp.path(), &varies);
    let kwargs = HashMap::from([("flat".to_owned(), json!(true))]);
    let flat = ChatTemplateParams {
        template_kwargs: Some(&kwargs),
        ..ChatTemplateParams::default()
    };
    let (text, ids, from_text) = render_under(&tok, json!("a[/INST:x]b"), flat);
    assert!(!from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b</s:x>ab</s:x>");
    // The typed `[/INST:x]` is its characters; each `</s:x>` is the control.
    assert_eq!(ids, [2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 1, 13, 14, 1]);
}

#[test]
fn a_constant_bound_from_a_request_argument_leaves_the_template_as_it_was() {
    // A spelling a request supplies must not become a control token through
    // the marks, and the template that takes one keeps writing it as it did:
    // a name that may take a request argument is no constant. The argument
    // may be read in the binding itself, through another name, through a
    // name that is bound only when the request left it out, through a name a
    // block, a `set` of several names or a `set` inside a `filter` block
    // binds again, through a name bound again inside the body of a block
    // `set` or in the `else` of a loop, or by a second binding of the
    // constant's own name: a `set` under a request argument, a macro.
    let direct = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
        {% set close = custom if custom is defined else '[/INST{}]'.format(suffix) %}";
    let through = "{% set suffix = custom if custom is defined else ':x' %}\
        {% set open = '[INST{}]'.format(suffix) %}{% set close = '[/INST{}]'.format(suffix) %}";
    let defaulted = "{% if suffix is not defined %}{% set suffix = ':x' %}{% endif %}\
        {% set open = '[INST{}]'.format(suffix) %}{% set close = '[/INST{}]'.format(suffix) %}";
    let block = "{% set suffix = ':x' %}{% if custom is defined %}{% set suffix %}{{ custom }}\
        {% endset %}{% endif %}{% set open = '[INST{}]'.format(suffix) %}\
        {% set close = '[/INST{}]'.format(suffix) %}";
    let several = "{% set suffix, other = (custom if custom is defined else ':x'), 1 %}\
        {% set open = '[INST{}]'.format(suffix) %}{% set close = '[/INST{}]'.format(suffix) %}";
    let filtered = "{% set suffix = ':x' %}{% filter trim %}{% if custom is defined %}\
        {% set suffix = custom %}{% endif %}{% endfilter %}\
        {% set open = '[INST{}]'.format(suffix) %}{% set close = '[/INST{}]'.format(suffix) %}";
    let rebound = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
        {% set close = '[/INST{}]'.format(suffix) %}\
        {% if custom is defined %}{% set close = custom %}{% endif %}";
    let nested = "{% set suffix = ':x' %}{% set note %}{% if custom is defined %}\
        {% set suffix = custom %}{% endif %}{% endset %}\
        {% set open = '[INST{}]'.format(suffix) %}{% set close = '[/INST{}]'.format(suffix) %}";
    let unlooped = "{% set suffix = ':x' %}{% for item in [] %}{% else %}\
        {% if custom is defined %}{% set suffix = custom %}{% endif %}{% endfor %}\
        {% set open = '[INST{}]'.format(suffix) %}{% set close = '[/INST{}]'.format(suffix) %}";
    let declared = "{% macro close() %}{% endmacro %}{% set suffix = ':x' %}\
        {% set open = '[INST{}]'.format(suffix) %}{% set close = '[/INST{}]'.format(suffix) %}";
    for constants in [
        direct, through, defaulted, block, several, filtered, rebound, nested, unlooped, declared,
    ] {
        let temp = TempDir::new().unwrap();
        let takes = template(constants, "{{ open }}", "{{ close }}", "{{ close }}");
        let tok = checkpoint(temp.path(), &takes);
        let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(from_text, "{constants}");
        assert_eq!(ids, [2, 13, 3, 14, 3, 13, 14, 3], "{constants}");
    }

    // The body of a block `set` and the `else` of a loop without an item
    // share the top level's names: the suffix a request passes there is the
    // one the constants are built from.
    for constants in [nested, unlooped] {
        let temp = TempDir::new().unwrap();
        let takes = template(constants, "{{ open }}", "{{ close }}", "{{ close }}");
        let tok = checkpoint(temp.path(), &takes);
        let kwargs = HashMap::from([("custom".to_owned(), json!(":y"))]);
        let supplied = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        let (text, _, from_text) = render_under(&tok, json!("ab"), supplied);
        assert!(from_text, "{constants}");
        assert_eq!(text, "[INST:y]ab[/INST:y]ab[/INST:y]", "{constants}");
    }

    // The end token as a request's argument is written as the argument
    // always was: by the template left as it was, read from the text.
    let temp = TempDir::new().unwrap();
    let takes = template(direct, "{{ open }}", "{{ close }}", "{{ close }}");
    let tok = checkpoint(temp.path(), &takes);
    let kwargs = HashMap::from([("custom".to_owned(), json!("</s:x>"))]);
    let supplied = ChatTemplateParams {
        template_kwargs: Some(&kwargs),
        ..ChatTemplateParams::default()
    };
    let (text, ids, from_text) = render_under(&tok, json!("ab"), supplied);
    assert!(from_text);
    assert_eq!(text, "[INST:x]ab</s:x>ab</s:x>");
    assert_eq!(ids, [2, 13, 14, 1, 13, 14, 1]);
}

#[test]
fn a_constant_passed_to_a_format_leaves_the_template_as_it_was() {
    // A `format` may write its arguments or build a text the template
    // compares, and nothing tells the two apart: compared, a marked constant
    // would no longer equal the text a request sends.
    let compares = "'' if message['content'] == '{}'.format(close) else message['content']";
    let temp = TempDir::new().unwrap();
    let searches = format!("{CONSTANTS}{}", SEARCHES.replace("SEARCH", compares));
    let tok = checkpoint(temp.path(), &searches);
    let history = [
        json!({"role": "user", "content": "ab"}),
        json!({"role": "assistant", "content": "[/INST:x]"}),
    ];
    let (text, _, from_text) = prompt(&tok, &history, ChatTemplateParams::default());
    assert!(from_text);
    assert_eq!(text, "[INST:x]ab[/INST:x]</s:x>");

    let temp = TempDir::new().unwrap();
    let writes = template(
        CONSTANTS,
        "{{ open }}",
        "{{ '{}'.format(close) }}",
        "{{ end }}",
    );
    let tok = checkpoint(temp.path(), &writes);
    let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(ids, TYPED_AS_THE_CONTROL);
}

#[test]
fn a_template_that_refuses_a_typed_spelling_is_left_as_it_was() {
    let temp = TempDir::new().unwrap();
    // The refusal looks for a constant in request text: marked, the template
    // would render what it refuses.
    let refuses = "{% for message in messages %}{% if close in message['content'] %}\
        {{ raise_exception('no control spellings') }}{% endif %}{% endfor %}";
    let guarded = format!(
        "{CONSTANTS}{refuses}{}",
        SEARCHES.replace("SEARCH", "message['content']")
    );
    let tok = checkpoint(temp.path(), &guarded);
    // Another spelling typed is still read as its token.
    let (_, ids, from_text) = render(&tok, json!("a[INST:x]b"));
    assert!(from_text);
    assert_eq!(ids, [2, 13, 2, 14, 3, 13, 14, 1]);
    let typed = [json!({"role": "user", "content": "a[/INST:x]b"})];
    assert!(tok
        .apply_chat_template(&typed, ChatTemplateParams::default())
        .is_err());
}

#[test]
fn a_name_bound_inside_a_block_is_the_block_s_own() {
    let temp = TempDir::new().unwrap();
    // A `block` runs among names of its own, as a loop's body does: the
    // `set` inside binds nothing the statements after it read, so the
    // constant of the same name keeps its one value.
    let scoped = format!(
        "{CONSTANTS}{{% block note %}}{{% set close = custom %}}{{% endblock %}}{}",
        SEARCHES.replace("SEARCH", "message['content']")
    );
    let tok = checkpoint(temp.path(), &scoped);
    assert_eq!(render(&tok, json!("a[/INST:x]b")).1, TYPED_AS_TEXT);
    let kwargs = HashMap::from([("custom".to_owned(), json!("</s:x>"))]);
    let supplied = ChatTemplateParams {
        template_kwargs: Some(&kwargs),
        ..ChatTemplateParams::default()
    };
    let (text, ids, _) = render_under(&tok, json!("a[/INST:x]b"), supplied);
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
    assert_eq!(ids, TYPED_AS_TEXT);
}

#[test]
fn a_template_that_takes_a_history_only_with_a_spelling_in_it_is_left_as_it_was() {
    let temp = TempDir::new().unwrap();
    // Any message but the user's is refused unless it holds the closing
    // spelling: the refusal looks for a constant in request text. Marked,
    // the template would refuse the history it takes.
    let requires = "{% for message in messages %}{% if message['role'] != 'user' \
        and close not in message['content'] %}{{ raise_exception('no closing spelling') }}\
        {% endif %}{% endfor %}";
    let guarded = format!(
        "{CONSTANTS}{requires}{}",
        SEARCHES.replace("SEARCH", "message['content']")
    );
    let tok = checkpoint(temp.path(), &guarded);
    let history = [
        json!({"role": "user", "content": "ab"}),
        json!({"role": "assistant", "content": "a[/INST:x]b"}),
    ];
    let (text, ids, from_text) = prompt(&tok, &history, ChatTemplateParams::default());
    assert!(from_text);
    assert_eq!(text, "[INST:x]ab[/INST:x]a[/INST:x]b</s:x>");
    assert_eq!(ids, [2, 13, 14, 3, 13, 3, 14, 1]);
}

#[test]
fn a_template_whose_refusal_the_marks_would_lift_is_left_as_it_was() {
    let temp = TempDir::new().unwrap();
    // A second message is refused while the opening constant has its own
    // length: between marks it is longer, and the marked template would
    // render the conversations this one refuses.
    let one_turn = "{% if messages | length > 1 and open | length == 8 %}\
        {{ raise_exception('one turn only') }}{% endif %}";
    let guarded = format!(
        "{CONSTANTS}{one_turn}{}",
        SEARCHES.replace("SEARCH", "message['content']")
    );
    let tok = checkpoint(temp.path(), &guarded);
    let turn = [json!({"role": "user", "content": "a[/INST:x]b"})];
    let (text, ids, from_text) = prompt(&tok, &turn, ChatTemplateParams::default());
    assert!(from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]");
    assert_eq!(ids, [2, 13, 3, 14, 3]);
    let history = [
        json!({"role": "user", "content": "ab"}),
        json!({"role": "assistant", "content": "ab"}),
    ];
    assert!(tok
        .apply_chat_template(&history, ChatTemplateParams::default())
        .is_err());
}

#[test]
fn a_constant_drawn_by_a_filter_leaves_the_template_as_it_was() {
    let temp = TempDir::new().unwrap();
    // Under `flat` the closing token is drawn afresh at each render. A
    // filter is no name: nothing says that what it gives is the template's
    // text alone, and one draw read at load would stand for all.
    let constants = "{% set suffix = ':x' %}{% set open = '[INST{}]'.format(suffix) %}\
        {% if flat %}{% set close = (['</s{}>', '[/INST{}]'] | random).format(suffix) %}\
        {% else %}{% set close = '[/INST{}]'.format(suffix) %}{% endif %}";
    let draws = template(constants, "{{ open }}", "{{ close }}", "{{ close }}");
    let tok = checkpoint(temp.path(), &draws);
    let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(ids, [2, 13, 3, 14, 3, 13, 14, 3]);
    // Each render draws by the seed its request passes: both tokens come.
    // The seeds are spread over the whole range (the generator draws the
    // first item from a small one).
    let drawn: HashSet<String> = (1..=32u64)
        .map(|seed| {
            let kwargs = HashMap::from([
                ("flat".to_owned(), json!(true)),
                (
                    "RAND_SEED".to_owned(),
                    json!(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                ),
            ]);
            let flat = ChatTemplateParams {
                template_kwargs: Some(&kwargs),
                ..ChatTemplateParams::default()
            };
            render_under(&tok, json!("ab"), flat).0
        })
        .collect();
    let both = ["[INST:x]ab</s:x>ab</s:x>", "[INST:x]ab[/INST:x]ab[/INST:x]"];
    assert_eq!(drawn, both.map(str::to_owned).into());
}

#[test]
fn a_control_the_load_cannot_write_as_a_literal_leaves_the_template_as_it_was() {
    // A statement may bind a control spelling in a way the load computes
    // and cannot keep: a filter takes part in it, it reads a name bound
    // under a condition, or a request argument may bind its name again.
    // Written only under `flat`, an argument no probe conversation passes,
    // that control would leave the marked template without marks and be
    // read as text. The template stays as it was instead, and the control
    // it writes under `flat` is the token it always was.
    let filtered = "{% if flat %}{% set extra = ('[/INST' ~ suffix ~ ']') | trim %}\
        {{ extra }}{% endif %}";
    let chained = "{% if flat %}{% set part = '[/INST' %}{% set extra = part ~ suffix ~ ']' %}\
        {{ extra }}{% endif %}";
    let rebound = "{% set extra = '[/INST' ~ suffix ~ ']' %}{% if custom is defined %}\
        {% set extra = custom %}{% endif %}{% if flat %}{{ extra }}{% endif %}";
    for tail in [filtered, chained, rebound] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint(temp.path(), &format!("{}{tail}", builds()));
        let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(from_text, "{tail}");
        assert_eq!(ids, TYPED_AS_THE_CONTROL, "{tail}");
        let kwargs = HashMap::from([("flat".to_owned(), json!(true))]);
        let flat = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        let (text, ids, from_text) = render_under(&tok, json!("ab"), flat);
        assert!(from_text, "{tail}");
        assert_eq!(text, "[INST:x]ab[/INST:x]ab</s:x>[/INST:x]", "{tail}");
        assert_eq!(ids, [2, 13, 14, 3, 13, 14, 1, 3], "{tail}");
    }

    // A filter over the template's own constants gives a value the load
    // reads once, and nothing in the syntax says that every render gives
    // the same: the statement leaves the template as it was, whatever the
    // one value is.
    let temp = TempDir::new().unwrap();
    let counts = format!("{}{{% set width = suffix | length %}}", builds());
    let tok = checkpoint(temp.path(), &counts);
    let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(ids, TYPED_AS_THE_CONTROL);
}

#[test]
fn a_search_made_only_under_the_thinking_switch_leaves_the_template_as_it_was() {
    let temp = TempDir::new().unwrap();
    // A filter takes the constant as an argument, and only with thinking
    // switched off: the syntax shows it, whatever a render passes.
    let search = "(message['content'] | replace(close, '')) \
        if enable_thinking is defined and not enable_thinking else message['content']";
    let searches = format!("{CONSTANTS}{}", SEARCHES.replace("SEARCH", search));
    let tok = checkpoint(temp.path(), &searches);
    let off = ChatTemplateParams {
        thinking: Some(false),
        ..ChatTemplateParams::default()
    };
    let history = [
        json!({"role": "user", "content": "ab"}),
        json!({"role": "assistant", "content": "a[/INST:x]b"}),
    ];
    let (text, _, from_text) = prompt(&tok, &history, off);
    assert!(from_text);
    assert_eq!(text, "[INST:x]ab[/INST:x]ab</s:x>");
}

#[test]
fn a_conversation_the_template_refuses_is_not_judged_but_the_bare_turn_is() {
    // Refuses a system message: the conversations without one decide.
    let temp = TempDir::new().unwrap();
    let no_system = "{% for message in messages %}{% if message['role'] == 'system' %}\
        {{ raise_exception('no system message') }}{% endif %}{% endfor %}";
    let tok = checkpoint(temp.path(), &(no_system.to_owned() + &builds()));
    assert_eq!(render(&tok, json!("a[/INST:x]b")).1, TYPED_AS_TEXT);

    // Refuses a conversation that does not open with a system message, the
    // bare turn among them: nothing says what the template writes, and it
    // stays as it was.
    let temp = TempDir::new().unwrap();
    let system_first = "{% if messages[0]['role'] != 'system' %}\
        {{ raise_exception('a system message first') }}{% endif %}";
    let tok = checkpoint(temp.path(), &(system_first.to_owned() + &builds()));
    let messages = [
        json!({"role": "system", "content": "ab"}),
        json!({"role": "user", "content": "a[/INST:x]b"}),
    ];
    let (text, ids, from_text) = prompt(&tok, &messages, ChatTemplateParams::default());
    assert!(from_text);
    assert_eq!(text, "ab</s:x>[INST:x]a[/INST:x]b[/INST:x]");
    assert_eq!(ids, [13, 14, 1, 2, 13, 3, 14, 3]);
}

/// The messages with room for a shape: AT_TOP stands behind the constants,
/// IN_LOOP first in the body of the message loop, and USER writes a user
/// message; any other message is its text and the end token.
const SHAPED: &str = "AT_TOP{% for message in messages %}IN_LOOP\
    {% if message['role'] == 'user' %}USER{% else %}{{ message['content'] }}{{ end }}\
    {% endif %}{% endfor %}";

/// A user message between the opening and the closing token, each written by
/// its name alone.
const WRITTEN: &str = "{{ open }}{{ message['content'] }}{{ close }}";

/// [`CONSTANTS`] and [`SHAPED`] with its three places filled.
fn shaped(at_top: &str, in_loop: &str, user: &str) -> String {
    let body = SHAPED
        .replace("AT_TOP", at_top)
        .replace("IN_LOOP", in_loop)
        .replace("USER", user);
    format!("{CONSTANTS}{body}")
}

/// The names of those `shapes` whose template is switched on. A shape is its
/// name and three pieces of a template that writes its constants by their
/// names ([`shaped`], [`WRITTEN`]): one behind the constants, one in the
/// loop's body, and one behind the messages that runs only under `flat`. No
/// conversation of the load's check passes `flat`, so only the syntax tells
/// what the template does under it. A template left as it was encodes its
/// prompt from its text, a typed spelling as the control, with the argument
/// and without.
fn switched_on<'a>(shapes: &[(&'a str, &str, &str, &str)]) -> Vec<&'a str> {
    let kwargs = HashMap::from([("flat".to_owned(), json!(true))]);
    let mut on = Vec::new();
    for (name, at_top, in_loop, under_flat) in shapes {
        let temp = TempDir::new().unwrap();
        let template = format!(
            "{}{{% if flat %}}{under_flat}{{% endif %}}",
            shaped(at_top, in_loop, WRITTEN)
        );
        let tok = checkpoint(temp.path(), &template);
        let flat = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        if tok.chat_template.marked()
            || render(&tok, json!("a[/INST:x]b")).1 != TYPED_AS_THE_CONTROL
            || !render_under(&tok, json!("a[/INST:x]b"), flat).2
        {
            on.push(*name);
        }
    }
    on
}

#[test]
fn a_write_that_is_no_plain_write_leaves_the_template_as_it_was() {
    // What a block around a write does with the text, and what reads the
    // text a block or a macro kept, is not the write of a name.
    let on = switched_on(&[
        (
            "a filter block",
            "",
            "",
            "{% filter upper %}{{ close }}{% endfilter %}",
        ),
        (
            "an autoescape block",
            "",
            "",
            "{% autoescape true %}{{ end }}{% endautoescape %}",
        ),
        (
            "a block set read later",
            "",
            "",
            "{% set kept %}{{ close }}{% endset %}{{ kept | length }}",
        ),
        (
            "a macro's text read later",
            "{% macro closing() %}{{ close }}{% endmacro %}",
            "",
            "{{ closing() | length }}",
        ),
        (
            "a macro's text bound and read later",
            "{% macro closing() %}{{ close }}{% endmacro %}",
            "",
            "{% set kept = closing() %}{{ kept[1:] }}",
        ),
        (
            "a macro under another name",
            "{% macro closing() %}{{ close }}{% endmacro %}",
            "",
            "{% set other = closing %}{{ other() | length }}",
        ),
        (
            "a call block",
            "{% macro counted() %}{{ caller() | length }}{% endmacro %}",
            "",
            "{% call counted() %}{{ close }}{% endcall %}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_join_that_is_no_plain_join_leaves_the_template_as_it_was() {
    let on = switched_on(&[
        ("under a filter", "", "", "{{ (open ~ close) | length }}"),
        ("under a test", "", "", "{{ (open ~ close) is string }}"),
        ("under a subscript", "", "", "{{ (open ~ close)[0] }}"),
        ("under a slice", "", "", "{{ (open ~ close)[1:] }}"),
        (
            "under a comparison",
            "",
            "",
            "{% if (open ~ close) == messages[0]['content'] %}{{ end }}{% endif %}",
        ),
        ("under a method", "", "", "{{ (open ~ close).upper() }}"),
        (
            "as a method's argument",
            "",
            "",
            "{{ ''.join([open ~ close]) }}",
        ),
        (
            "by `+`, under a filter",
            "",
            "",
            "{{ (open + close) | length }}",
        ),
        (
            "by `+`, under a comparison",
            "",
            "",
            "{% if (open + close) == messages[0]['content'] %}{{ end }}{% endif %}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_constant_carried_on_leaves_the_template_as_it_was() {
    // The value goes on under another name, or inside a value, and is read
    // there in a way that is no write.
    let on = switched_on(&[
        (
            "an alias at the top",
            "",
            "",
            "{% set alias = close %}{{ alias | length }}",
        ),
        (
            "an alias in a loop's body",
            "",
            "{% set alias = close %}{% if flat %}{{ alias | length }}{% endif %}",
            "",
        ),
        (
            "a with",
            "",
            "",
            "{% with held = close %}{{ held | length }}{% endwith %}",
        ),
        (
            "a list",
            "",
            "",
            "{% set held = [close] %}{{ held[0] | length }}",
        ),
        (
            "a map",
            "",
            "",
            "{% set held = {'k': close} %}{{ held['k'] | length }}",
        ),
        (
            "a namespace's attribute",
            "{% set ns = namespace(held='') %}",
            "",
            "{% set ns.held = close %}{{ ns.held | length }}",
        ),
        (
            "a loop's sequence",
            "",
            "",
            "{% for letter in close %}{{ letter }}.{% endfor %}",
        ),
        (
            "a conditional expression bound",
            "",
            "",
            "{% set held = close if flat else '' %}{{ held | length }}",
        ),
        (
            "a conditional expression written",
            "",
            "",
            "{{ close if flat else '' }}",
        ),
        (
            "a macro's argument",
            "{% macro counted(text) %}{{ text | length }}{% endmacro %}",
            "",
            "{{ counted(close) }}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_joined_name_read_otherwise_than_by_a_write_leaves_the_template_as_it_was() {
    // `whole` is bound in the loop's body from a join that holds two
    // constants: it carries them, and the rule for a constant is its rule.
    const BOUND: &str = "{% set whole = open ~ message['content'] ~ close %}";
    let read = |shape: &str| format!("{BOUND}{{% if flat %}}{shape}{{% endif %}}");
    let shapes = [
        ("a filter", read("{{ whole | length }}")),
        ("a method", read("{{ whole.upper() }}")),
        ("a subscript", read("{{ whole[0] }}")),
        ("a slice", read("{{ whole[1:] }}")),
        (
            "a search",
            read("{{ message['content'].split(whole)[-1] }}"),
        ),
        // The one test a carrying name may stand under is whether it is
        // empty, in the condition of an `if`: nothing else there.
        (
            "its length in a condition",
            read("{% if whole | length %}{{ end }}{% endif %}"),
        ),
        (
            "a comparison in a condition",
            read("{% if whole == '' %}{{ end }}{% endif %}"),
        ),
        (
            "a test word in a condition",
            read("{% if whole is string %}{{ end }}{% endif %}"),
        ),
        (
            "in, in a condition",
            read("{% if whole in message['content'] %}{{ end }}{% endif %}"),
        ),
        (
            "a conditional expression's condition",
            read("{{ 'y' if whole else 'n' }}"),
        ),
    ];
    let shapes: Vec<_> = shapes
        .iter()
        .map(|(name, in_loop)| (*name, "", in_loop.as_str(), ""))
        .collect();
    let on = switched_on(&shapes);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_read_of_a_constant_that_is_no_write_leaves_the_template_as_it_was() {
    let on = switched_on(&[
        ("its length", "", "", "{{ close | length }}"),
        ("a slice of it", "", "", "{{ close[1:] }}"),
        (
            "a comparison of it",
            "",
            "",
            "{% if close == messages[0]['content'] %}{{ end }}{% endif %}",
        ),
        (
            "in",
            "",
            "",
            "{% if close in messages[0]['content'] %}{{ end }}{% endif %}",
        ),
        (
            "a search for it",
            "",
            "",
            "{{ messages[0]['content'].split(close)[-1] }}",
        ),
        ("a method of it", "", "", "{{ close.strip('[') }}"),
        ("a test word", "", "", "{{ close is string }}"),
        // A macro's own `set` of the name does not hide the constant from
        // what the macro reads before it.
        (
            "a macro that reads it before its own set",
            "{% macro peek(text) %}{{ close | length }}{% set close = text %}{% endmacro %}",
            "",
            "{{ peek('') }}",
        ),
        (
            "a special token's name sliced",
            "",
            "",
            "{{ eos_token[1:] }}",
        ),
        (
            "a special token's name compared",
            "",
            "",
            "{% if messages[0]['content'] == eos_token %}{{ end }}{% endif %}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_search_only_another_spelling_opens_leaves_the_template_as_it_was() {
    let temp = TempDir::new().unwrap();
    // The search for the closing token runs only on a text that holds a
    // spelling no conversation of the load's check types.
    let hidden = format!(
        "{CONSTANTS}{}",
        SEARCHES.replace(
            "SEARCH",
            "message['content'].split(close)[-1] if ('<hi' ~ 'dden>') in message['content'] \
             else message['content']"
        )
    );
    let tok = checkpoint(temp.path(), &hidden);
    assert!(!tok.chat_template.marked());
    let (_, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(from_text);
    assert_eq!(ids, TYPED_AS_THE_CONTROL);
}

#[test]
fn a_part_of_a_constant_read_elsewhere_leaves_the_template_as_it_was() {
    // `suffix` is what the constants are built from. Read anywhere but in
    // the top-level statements the load computes, it may build a control
    // the load never saw: in a statement that also reads a request
    // argument, where a value is written, in a loop's body, beside literal
    // text, under another name.
    let on = switched_on(&[
        (
            "a statement that reads a request argument",
            "{% set extra = custom if custom is defined else '[/INST' ~ suffix ~ ']' %}",
            "",
            "{{ extra }}",
        ),
        (
            "a default",
            "{% set extra = custom | default('[/INST' ~ suffix ~ ']') %}",
            "",
            "{{ extra }}",
        ),
        (
            "built where it is written",
            "",
            "",
            "{{ '[/INST' ~ suffix ~ ']' }}",
        ),
        (
            "bound in a loop's body",
            "",
            "{% set built = '[/INST' ~ suffix ~ ']' %}{% if flat %}{{ built }}{% endif %}",
            "",
        ),
        ("beside literal text", "", "", "[/INST{{ suffix }}]"),
        (
            "under another name",
            "{% set tail = suffix %}",
            "",
            "{{ '[/INST' ~ tail ~ ']' }}",
        ),
        (
            "in a macro",
            "{% macro closing() %}{{ '[/INST' ~ suffix ~ ']' }}{% endmacro %}",
            "",
            "{{ closing() }}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_value_the_load_cannot_know_leaves_the_template_as_it_was() {
    // The load computes a top-level statement once. One that builds its
    // value from a name with two bindings, or through a filter, may build
    // another value in a render: here the closing token, under `flat`.
    let on = switched_on(&[
        (
            "a name bound again under a condition",
            "{% set tail = ':x' %}{% if not flat %}{% set tail = ':y' %}{% endif %}\
             {% set extra = '[/INST{}]'.format(tail) %}",
            "",
            "{{ extra }}",
        ),
        (
            "a filter",
            "{% set extra = ['[/INST:y]', '[/INST' ~ suffix ~ ']'] | last %}",
            "",
            "{{ extra }}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_built_piece_of_a_control_leaves_the_template_as_it_was() {
    // A top-level statement builds a value that is only a piece of the
    // closing token. No mark stands around a piece, and the template
    // completes the token where it writes the piece: behind the messages,
    // under `flat`. Each shape with the token's head as the piece, then
    // turned, with its tail.
    let shapes = [
        (
            "a piece and a literal joined",
            "{% set half = '[/INST' ~ suffix %}",
            "{{ half ~ ']' }}",
        ),
        (
            "a literal and a piece joined",
            "{% set half = suffix ~ ']' %}",
            "{{ '[/INST' ~ half }}",
        ),
        (
            "two pieces joined",
            "{% set left = '[/INST' ~ suffix %}{% set right = ']' ~ '' %}",
            "{{ left ~ right }}",
        ),
        (
            "two pieces joined, the short one first",
            "{% set left = '[' ~ '' %}{% set right = '/INST' ~ suffix ~ ']' %}",
            "{{ left ~ right }}",
        ),
        (
            "a piece before literal text",
            "{% set half = '[/INST' ~ suffix %}",
            "{{ half }}]",
        ),
        (
            "a piece behind literal text",
            "{% set half = suffix ~ ']' %}",
            "[/INST{{ half }}",
        ),
        (
            "a piece built from literals alone",
            "{% set half = '[/INST' ~ ':' %}",
            "{{ half ~ 'x]' }}",
        ),
        (
            "a tail built from literals alone",
            "{% set half = ':' ~ 'x]' %}",
            "{{ '[/INST' ~ half }}",
        ),
        (
            "a piece a macro writes",
            "{% set half = '[/INST' ~ suffix %}{% macro piece() %}{{ half }}{% endmacro %}",
            "{{ piece() ~ ']' }}",
        ),
        (
            "a tail a macro writes",
            "{% set half = suffix ~ ']' %}{% macro piece() %}{{ half }}{% endmacro %}",
            "{{ '[/INST' ~ piece() }}",
        ),
        (
            "a piece joined into a name",
            "{% set half = '[/INST' ~ suffix %}",
            "{% set whole = half ~ ']' %}{{ whole }}",
        ),
        (
            "a tail joined into a name",
            "{% set half = suffix ~ ']' %}",
            "{% set whole = '[/INST' ~ half %}{{ whole }}",
        ),
    ];
    let kwargs = HashMap::from([("flat".to_owned(), json!(true))]);
    for (name, at_top, under_flat) in shapes {
        let temp = TempDir::new().unwrap();
        let template = format!(
            "{}{{% if flat %}}{under_flat}{{% endif %}}",
            shaped(at_top, "", WRITTEN)
        );
        let tok = checkpoint(temp.path(), &template);
        assert!(!tok.chat_template.marked(), "{name}");
        let flat = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        // What the plain template gives: the token the template completes
        // is the control, and so is the typed spelling.
        let (text, ids, from_text) = render_under(&tok, json!("a[/INST:x]b"), flat);
        assert!(from_text, "{name}");
        assert_eq!(
            text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>[/INST:x]",
            "{name}"
        );
        assert_eq!(ids, TYPED_AS_THE_CONTROL_AND_A_CLOSE, "{name}");
    }
}

#[test]
fn a_built_value_that_is_no_control_leaves_the_template_as_it_was() {
    // The load does not ask what a built value could be a piece of: a
    // top-level statement that builds anything but the empty string, one
    // control, or a string with a control at each end leaves the template
    // as it was. Text that no control holds, blanks, a control built in two
    // steps, also under one name. A value that is no string, which a join
    // would still write: a sum, a negative number, a truth value, none, a
    // list and a map, even an empty one. And a literal whose escape the
    // source does not hold as a character, which the load takes for built.
    let on = switched_on(&[
        (
            "text no control holds",
            "{% set sep = '-' ~ '-' %}",
            "",
            "{{ sep ~ 'a' }}",
        ),
        (
            "blanks",
            "{% set gap = ' ' ~ ' ' %}",
            "",
            "{{ close ~ gap }}",
        ),
        (
            "a control built in two steps",
            "{% set half = '[/INST' ~ suffix %}{% set extra = half ~ ']' %}",
            "",
            "{{ extra }}",
        ),
        (
            "a piece bound again as the whole control",
            "{% set extra = '[/INST' ~ suffix %}{% set extra = extra ~ ']' %}",
            "",
            "{{ extra }}",
        ),
        ("a sum", "{% set count = 1 + 1 %}", "", "{{ open ~ count }}"),
        ("a negative number", "{% set low = -1 %}", "", ""),
        ("a truth value", "{% set yes = not false %}", "", ""),
        ("none", "{% set unset = none if suffix else '' %}", "", ""),
        ("a list", "{% set roles = ['a', 'b'] %}", "", ""),
        ("an empty list", "{% set seen = [] %}", "", ""),
        ("a map", "{% set held = {'k': 1} %}", "", ""),
        ("a literal with an escape", "{% set tab = '\\t' %}", "", ""),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn a_built_value_that_holds_several_controls_is_marked_around_each() {
    // One name for several controls, side by side or with text between
    // them. The value begins with a control and ends with one, so nothing
    // written beside it completes a control, and the marks stand around
    // each: the typed spelling stays text, and what stands between two
    // controls is encoded as the text it is.
    for (name, at_top, value, ids) in [
        (
            "two controls side by side",
            "{% set both = close ~ end %}",
            "[/INST:x]</s:x>",
            &[3, 1][..],
        ),
        (
            "text between two controls",
            "{% set both = close ~ 'ab' ~ end %}",
            "[/INST:x]ab</s:x>",
            &[3, 13, 14, 1][..],
        ),
        (
            "three controls out of literals alone",
            "{% set both = '[/INST' ~ ':x]' ~ '[INST' ~ ':x]a' ~ '</s' ~ ':x>' %}",
            "[/INST:x][INST:x]a</s:x>",
            &[3, 2, 13, 1][..],
        ),
        (
            "one control twice",
            "{% set both = close ~ close %}",
            "[/INST:x][/INST:x]",
            &[3, 3][..],
        ),
    ] {
        let temp = TempDir::new().unwrap();
        let user = "{{ open }}{{ message['content'] }}{{ both }}";
        let tok = checkpoint(temp.path(), &shaped(at_top, "", user));
        assert!(tok.chat_template.marked(), "{name}");
        let (text, typed, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(!from_text, "{name}");
        assert_eq!(
            text,
            format!("[INST:x]a[/INST:x]b{value}ab</s:x>"),
            "{name}"
        );
        let mut expected = vec![2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14];
        expected.extend_from_slice(ids);
        expected.extend_from_slice(&[13, 14, 1]);
        assert_eq!(typed, expected, "{name}");
        // A request that types no spelling is encoded from its text, the
        // value read the same way there.
        let (text, plain, from_text) = render(&tok, json!("a"));
        assert!(from_text, "{name}");
        assert_eq!(text, format!("[INST:x]a{value}ab</s:x>"), "{name}");
        assert_eq!(plain[..2], [2, 13], "{name}");
        assert_eq!(plain[2..], expected[12..], "{name}");
    }
}

#[test]
fn a_built_value_is_bound_from_its_first_control_to_its_last() {
    // A built value takes the marks only when it begins with a control and
    // ends with one. Text at an edge may be a piece of a control, which the
    // template would complete where it writes the value: text behind one
    // control or before it, behind two or before two, the head of a control
    // behind a whole one. And a line break between two controls, which no
    // plain literal holds.
    let on = switched_on(&[
        (
            "text behind a control",
            "{% set held = close ~ 'a' %}",
            "",
            "{{ held }}",
        ),
        (
            "text before a control",
            "{% set held = 'a' ~ close %}",
            "",
            "{{ held }}",
        ),
        (
            "text behind two controls",
            "{% set held = close ~ end ~ 'a' %}",
            "",
            "{{ held }}",
        ),
        (
            "text before two controls",
            "{% set held = 'a' ~ close ~ end %}",
            "",
            "{{ held }}",
        ),
        (
            "the head of a control behind a whole one",
            "{% set held = close ~ '[/INST' %}",
            "",
            "{{ held ~ ':x]' }}",
        ),
        (
            "a line break between two controls",
            "{% set held = close ~ '\\n' ~ end %}",
            "",
            "{{ held }}",
        ),
    ]);
    assert!(on.is_empty(), "switched on: {on:?}");
}

#[test]
fn the_matcher_is_made_for_a_built_value_a_literal_holds() {
    // A default prompt with a line break in it is a value the load takes
    // for built, and it is no control. No plain literal holds it, so it
    // could take no marks: the template is left as it was before the
    // matcher over the added tokens is asked for. So is a value that is no
    // string.
    for at_top in ["{% set note = 'ab\\nab' %}", "{% set held = ['a'] %}"] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint(temp.path(), &shaped(at_top, "", WRITTEN));
        assert!(!tok.chat_template.marked(), "{at_top}");
        assert!(tok.core.added.get().is_none(), "{at_top}");
    }
}

#[test]
fn every_control_of_a_built_value_is_read_against_the_vocabulary() {
    // A value of two controls, the second an added token that no other
    // statement binds. The marks stand around both. Where that token's
    // entry takes the whitespace behind it, the flat encode may read it
    // otherwise than as it stands, and the template is left as it was.
    let user = "{{ open }}{{ message['content'] }}{{ both }}";
    let template = shaped("{% set both = close ~ '<' ~ 'x>' %}", "", user);
    let plain = with_added_token("<x>");

    let temp = TempDir::new().unwrap();
    let tok = checkpoint_of(temp.path(), &plain, &template);
    assert!(tok.chat_template.marked());
    let (text, ids, from_text) = render(&tok, json!("a<x>b"));
    assert!(!from_text);
    assert_eq!(text, "[INST:x]a<x>b[/INST:x]<x>ab</s:x>");
    assert_eq!(ids, [2, 13, 16, 11, 18, 14, 3, 19, 13, 14, 1]);

    // With the flag: the token at the end of the value, and with a blank
    // behind it in the value, which the token takes.
    let flagged = with_flag(&plain, "<x>", "rstrip");
    let with_a_blank = shaped("{% set both = close ~ '<' ~ 'x> ' ~ end %}", "", user);
    for (template, ids) in [
        (&template, &[2, 13, 19, 14, 3, 19, 13, 14, 1][..]),
        (&with_a_blank, &[2, 13, 19, 14, 3, 19, 1, 13, 14, 1][..]),
    ] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &flagged, template);
        assert!(!tok.chat_template.marked(), "{template}");
        let (_, typed, from_text) = render(&tok, json!("a<x>b"));
        assert!(from_text, "{template}");
        assert_eq!(typed, ids, "{template}");
    }
}

/// [`TOKENIZER_JSON`] with one more added token, `<b>`, special and matched
/// after normalization.
fn with_a_normalized_entry() -> String {
    let conversion = TOKENIZER_JSON
        .replace(
            r#""special": true}],"#,
            r#""special": true},
        {"id": 19, "content": "<b>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true}],"#,
        )
        .replace(r#"">": 18}"#, r#"">": 18, "<b>": 19}"#);
    assert!(conversion.contains(r#""id": 19"#) && conversion.contains(r#"": 19}"#));
    conversion
}

/// A built value of several controls holds text between them, and a bound
/// value's text is read by its characters. The flat encode reads it so
/// where it matches every added token as it is spelled: there the value is
/// bound. Under a pipeline that matches an added token after normalization
/// the value stays unbound: its template reads as before, a prompt is
/// encoded from its text, and a text of the template that the normalizer
/// turns into an entry's spelling is that entry's token.
#[test]
fn a_built_value_of_several_controls_is_bound_where_added_tokens_match_as_spelled() {
    let conversion = with_a_normalized_entry();
    let several = template(
        &format!("{CONSTANTS}{{% set both = close ~ '<B>' ~ end %}}"),
        "{{ open }}",
        "{{ both }}",
        "{{ end }}",
    );
    let under = |normalizer: &str| {
        let conversion = conversion.replace(
            r#""normalizer": null"#,
            &format!(r#""normalizer": {normalizer}"#),
        );
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &conversion, &several);
        (temp, tok)
    };
    // The entry matches as it is spelled: the value is bound.
    let (_temp, tok) = under("null");
    assert!(tok.chat_template.marked());
    // The entry matches after normalization: the value stays unbound, and
    // the `<B>` the template writes is the entry's token.
    let (_temp, tok) = under(r#"{"type": "Lowercase"}"#);
    assert!(!tok.chat_template.marked());
    let (text, ids, from_text) = render(&tok, json!("a"));
    assert!(from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]<B></s:x>ab</s:x>");
    assert_eq!(ids, [2, 13, 3, 19, 1, 13, 14, 1]);
}

/// Under a pipeline that matches an added token after normalization the
/// flat encode reads as that token a text the normalizer turns into its
/// spelling, a text of the template as well. A marked prompt would read
/// that text by its characters, so a template that writes it stays as it
/// is: a prompt is encoded from its text, and the text is the entry's
/// token. The template may write the text whole, in two pieces side by
/// side, or only under an argument no probe conversation passes. Where the
/// entry matches as it is spelled the same text is no added token: the
/// template is marked.
#[test]
fn a_template_text_the_normalizer_turns_into_an_added_token_leaves_the_template_as_it_was() {
    let conversion = with_a_normalized_entry();
    let lowered = conversion.replace(
        r#""normalizer": null"#,
        r#""normalizer": {"type": "Lowercase"}"#,
    );
    assert_ne!(lowered, conversion);
    let kwargs = HashMap::from([("tag".to_owned(), json!(true))]);
    for close in [
        "{{ close }}<B>",
        "{{ close }}{{ '<' }}B>",
        "{{ close }}{% if tag %}<B>{% endif %}",
    ] {
        let writes_it = template(CONSTANTS, "{{ open }}", close, "{{ end }}");
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &lowered, &writes_it);
        assert!(!tok.chat_template.marked(), "{close}");
        let tagged = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        let (text, ids, from_text) = render_under(&tok, json!("a"), tagged);
        assert!(from_text, "{close}");
        assert_eq!(text, "[INST:x]a[/INST:x]<B>ab</s:x>", "{close}");
        assert_eq!(ids, [2, 13, 3, 19, 13, 14, 1], "{close}");
        let temp = TempDir::new().unwrap();
        let tok = checkpoint_of(temp.path(), &conversion, &writes_it);
        assert!(tok.chat_template.marked(), "{close}");
    }
}

#[test]
fn a_join_of_constants_is_marked() {
    // A constant is written as a whole operand of a join, and a name a
    // plain `set` binds from such a join is written: the marks go with the
    // value, in a loop's body, through a macro.
    let joined = "{{ open ~ message['content'] ~ close }}";
    let bound = "{% set whole = open ~ message['content'] ~ close %}";
    let wraps = "{% macro wrap(text) %}{{ open ~ text ~ close }}{% endmacro %}";
    for (at_top, in_loop, user) in [
        ("", "", joined),
        ("", bound, "{{ whole }}"),
        ("", bound, "{% set again = whole %}{{ '' ~ again }}"),
        (wraps, "", "{{ wrap(message['content']) }}"),
        (
            wraps,
            "",
            "{% set whole = '' ~ wrap(message['content']) %}{{ whole }}",
        ),
        // `+` joins two strings as `~` does.
        ("", "", "{{ open + message['content'] + close }}"),
        (
            "",
            "{% set whole = open + message['content'] + close %}",
            "{{ whole + '' }}",
        ),
    ] {
        let temp = TempDir::new().unwrap();
        let tok = checkpoint(temp.path(), &shaped(at_top, in_loop, user));
        assert!(tok.chat_template.marked(), "{at_top}{in_loop}{user}");
        let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(!from_text);
        assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
        assert_eq!(ids, TYPED_AS_TEXT, "{at_top}{in_loop}{user}");
    }
}

#[test]
fn a_join_by_plus_fails_with_the_marks_as_without_them() {
    // `+` joins two strings and reads no character of either; on a string
    // and a number it fails. A constant between marks is a string still, so
    // the marked template refuses the request that the plain one refuses.
    let adds = "{% if flat %}{{ close + 1 }}{% endif %}";
    let kwargs = HashMap::from([("flat".to_owned(), json!(true))]);
    for (spelled_whole, marked) in [("", true), ("{% if flat %}</s:x>{% endif %}", false)] {
        let temp = TempDir::new().unwrap();
        let template = shaped("", "", WRITTEN) + adds + spelled_whole;
        let tok = checkpoint(temp.path(), &template);
        assert_eq!(tok.chat_template.marked(), marked);
        let flat = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        let messages = [json!({"role": "user", "content": "ab"})];
        assert!(tok.apply_chat_template(&messages, flat).is_err());
    }
}

#[test]
fn a_carrying_name_under_a_truth_test_is_marked() {
    let temp = TempDir::new().unwrap();
    // `content` is bound from a join of two constants in one branch of the
    // loop's body, and from a macro's text in the other, where an `if` asks
    // whether it is empty before it is written. Two macros have a name of
    // the same spelling, a parameter and a `set` of their own, and read it
    // by a test word, a loop, a method and a subscript: those are the
    // macros' names.
    let template = format!(
        "{CONSTANTS}{}",
        "{% macro text(content) %}{% if content is string %}{{ content }}{% else %}\
         {% for part in content %}{{ part['text'] }}{% endfor %}{% endif %}{% endmacro %}\
         {% macro first(message) %}{% set content = message['content'] %}\
         {% if content is string and content | length %}{{ content.split(' ')[0] }}{% endif %}\
         {% endmacro %}\
         {% for message in messages %}{% if message['role'] == 'user' %}\
         {% set content = open ~ text(message['content']) ~ close %}{{ content }}\
         {% else %}{% set content = text(message['content']) %}\
         {% if not content %}{% elif messages and content %}{{ content }}{{ end }}{% endif %}\
         {% if flat %}{{ first(message) }}{% endif %}{% endif %}{% endfor %}"
    );
    let tok = checkpoint(temp.path(), &template);
    assert!(tok.chat_template.marked());
    let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
    assert!(!from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
    assert_eq!(ids, TYPED_AS_TEXT);
    // An empty text is empty in both renders: nothing is written for it.
    let messages = [
        json!({"role": "user", "content": "ab"}),
        json!({"role": "assistant", "content": ""}),
    ];
    let (text, ids, _) = prompt(&tok, &messages, ChatTemplateParams::default());
    assert_eq!(text, "[INST:x]ab[/INST:x]");
    assert_eq!(ids, [2, 13, 14, 3]);
}

#[test]
fn marking_never_changes_whether_a_value_is_empty() {
    let temp = TempDir::new().unwrap();
    // A constant that is the empty string is no control and takes no marks:
    // it stays empty. A constant between marks is never empty. So a test
    // for emptiness gives one answer with the marks and without them.
    let template = shaped(
        "{% set nothing = '{}'.format('') %}",
        "",
        "{% if nothing %}{{ end }}{% endif %}{% if open and not nothing %}{{ open }}{% endif %}\
         {{ message['content'] }}{% if not close or nothing %}{{ end }}{% else %}{{ close }}\
         {% endif %}",
    );
    let tok = checkpoint(temp.path(), &template);
    assert!(tok.chat_template.marked());
    let (text, ids, _) = render(&tok, json!("a[/INST:x]b"));
    assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
    assert_eq!(ids, TYPED_AS_TEXT);
}

#[test]
fn a_special_token_written_by_its_name_is_marked() {
    let temp = TempDir::new().unwrap();
    // The end token by the name a render gives it, alone, in a join and
    // under a test for emptiness.
    let template = format!(
        "{CONSTANTS}{}",
        "{% for message in messages %}{% if message['role'] == 'user' %}\
         {{ open }}{{ message['content'] }}{{ close }}{% elif eos_token and not bos_token %}\
         {{ message['content'] ~ eos_token }}{% endif %}{% endfor %}"
    );
    let tok = checkpoint(temp.path(), &template);
    assert!(tok.chat_template.marked());
    let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b</s:x>"));
    assert!(!from_text);
    assert_eq!(text, "[INST:x]a[/INST:x]b</s:x>[/INST:x]ab</s:x>");
    assert_eq!(
        ids,
        [2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 5, 17, 10, 11, 18, 3, 13, 14, 1]
    );
}

/// A control kept in an attribute of a namespace: the top level makes the
/// namespace once and binds the attribute from a join that holds a control
/// the template builds.
mod holder {
    use super::*;

    /// A namespace one of whose attributes a top-level statement binds from
    /// a join that holds a control is a holder: the attribute is written,
    /// bound again inside the message loop and asked whether it is empty,
    /// and the template is marked. The control it holds is the control
    /// token, and a spelling a message brought into it stays text.
    #[test]
    fn a_control_kept_in_an_attribute_of_a_namespace_is_marked() {
        let temp = TempDir::new().unwrap();
        let holds = "{% set ns = namespace(held='') %}\
            {% if flat %}{% set ns.held = ns.held + close + 'b' %}{% endif %}";
        let appends = "{% set ns.held = ns.held + message['content'] %}";
        let writes = "{% if ns.held != '' and ns.held %}{{ ns.held }}{% endif %}";
        let template = format!(
            "{}{{% if flat %}}{writes}{{% endif %}}",
            shaped(holds, appends, WRITTEN)
        );
        let tok = checkpoint(temp.path(), &template);
        assert!(tok.chat_template.marked());
        let kwargs = HashMap::from([("flat".to_owned(), json!(true))]);
        let flat = || ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..ChatTemplateParams::default()
        };
        let (text, ids, from_text) = render_under(&tok, json!("a[/INST:x]b"), flat());
        assert!(!from_text);
        assert_eq!(
            text,
            "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>[/INST:x]ba[/INST:x]bab"
        );
        // The closing token the template put into the attribute is the
        // control; the one the message brought is its characters, there as
        // in the turn.
        assert_eq!(
            ids,
            [
                2, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 3, 13, 14, 1, 3, 14, 13, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 14, 13, 14
            ]
        );
        let (text, ids, from_text) = render_under(&tok, json!("ab"), flat());
        assert!(from_text);
        assert_eq!(text, "[INST:x]ab[/INST:x]ab</s:x>[/INST:x]babab");
        assert_eq!(ids, [2, 13, 14, 3, 13, 14, 1, 3, 14, 13, 14, 13, 14]);
    }

    /// A control that one statement spells whole in a literal of its own is
    /// kept in an attribute the same way: the second read of the syntax
    /// takes the namespace as the first does.
    #[test]
    fn a_spelled_control_kept_in_an_attribute_of_a_namespace_is_marked() {
        let temp = TempDir::new().unwrap();
        let built = "{% set close = '[/INST{}]'.format(suffix) %}";
        let template = shaped(
            "{% set ns = namespace(shut='') %}{% set ns.shut = close %}",
            "",
            "{{ open }}{{ message['content'] }}{{ ns.shut }}",
        )
        .replace(built, "{% set close = '[/INST:x]' %}");
        assert!(!template.contains(built));
        let tok = checkpoint(temp.path(), &template);
        assert!(tok.chat_template.marked());
        let (text, ids, from_text) = render(&tok, json!("a[/INST:x]b"));
        assert!(!from_text);
        assert_eq!(text, "[INST:x]a[/INST:x]b[/INST:x]ab</s:x>");
        assert_eq!(ids, TYPED_AS_TEXT);
    }

    /// Where a namespace holds a control: the top level binds its name once
    /// and always, by a call of `namespace` that reads no control; a
    /// top-level statement binds the attribute from the control; every read
    /// of the namespace is one of its attributes in the place of a name
    /// that carries a control; and its name bound again names a new
    /// namespace. Each shape below steps outside one of these, wherever the
    /// statement that makes the holder stands, and its template reads as
    /// before.
    #[test]
    fn a_namespace_holds_a_control_where_its_attributes_are_only_written() {
        const NS: &str = "{% set ns = namespace(held='') %}";
        let on = switched_on(&[
            (
                "a filter on the attribute",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{{ ns.held | length }}",
            ),
            (
                "a slice of the attribute",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{{ ns.held[:3] }}",
            ),
            (
                "a method of the attribute",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{{ ns.held.strip() }}",
            ),
            (
                "a search in the attribute",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{% if 'b' in ns.held %}.{% endif %}",
            ),
            (
                "the attribute against a text",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{% if ns.held == 'b' %}.{% endif %}",
            ),
            (
                "an attribute bound from a filter",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{% set ns.held = ns.held | trim %}{{ ns.held }}",
            ),
            (
                "another name for the holder",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{% set other = ns %}{{ other.held }}",
            ),
            (
                "the holder as an argument",
                "{% set ns = namespace(held='') %}\
                 {% macro shown(space) %}{{ space.held }}{% endmacro %}",
                "",
                "{% set ns.held = close ~ 'b' %}{{ shown(ns) }}",
            ),
            (
                "a subscript of the holder",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{% if false %}{{ ns['held'] }}{% endif %}",
            ),
            (
                "a loop over the holder",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}\
                 {% if false %}{% for key in ns %}{{ key }}{% endfor %}{% endif %}",
            ),
            (
                "the holder written whole",
                NS,
                "",
                "{% set ns.held = close ~ 'b' %}{% if false %}{{ ns }}{% endif %}",
            ),
            (
                "a holder bound twice",
                "{% set ns = namespace(held='') %}{% set ns = namespace(held='') %}",
                "",
                "{% set ns.held = close ~ 'b' %}{{ ns.held }}",
            ),
            (
                "a holder bound under a condition",
                "{% if true %}{% set ns = namespace(held='') %}{% endif %}",
                "",
                "{% set ns.held = close ~ 'b' %}{{ ns.held }}",
            ),
            (
                "a keyword of the call that reads a constant",
                "{% set ns = namespace(held=close) %}",
                "",
                "{{ ns.held }}",
            ),
            (
                "an attribute bound from a control inside the loop",
                NS,
                "{% set ns.held = close ~ 'b' %}",
                "{{ ns.held | length }}",
            ),
            (
                "an attribute bound from a control inside a macro",
                "{% set ns = namespace(held='') %}\
                 {% macro hold() %}{% set ns.held = close ~ 'b' ~ ns.held %}{% endmacro %}",
                "",
                "{{ hold() }}{{ ns.held | length }}",
            ),
            (
                "an attribute bound from a control inside a block",
                "{% set ns = namespace(held='') %}\
                 {% block hold %}{% set ns.held = close ~ 'b' %}{% endblock %}",
                "",
                "{{ ns.held | length }}",
            ),
            (
                "an attribute of an attribute bound from a control",
                "{% set inner = namespace(held='') %}{% set ns = namespace(child=inner) %}",
                "",
                "{% set ns.child.held = close ~ 'b' %}{{ inner.held | length }}",
            ),
            (
                "a name bound from the attribute before the holder is made",
                NS,
                "{% set ns.held = close %}",
                "{% set shown = ns.held %}{{ shown | length }}{% set ns.held = close ~ 'b' %}",
            ),
            (
                "the holder's name bound to another namespace inside the loop",
                "{% set ns = namespace(held='') %}{% set other = namespace(held='') %}",
                "{% set ns = other %}{% set ns.held = close %}",
                "{% set ns.held = close ~ 'b' %}{{ other.held | length }}",
            ),
        ]);
        assert!(on.is_empty(), "switched on: {on:?}");
    }

    /// A loop's target or a macro's parameter with a holder's name is the
    /// loop's or the macro's own inside, as the engine's scopes have it:
    /// read there as a template likes, while the holder outside is read by
    /// its attributes. A new namespace bound to the name inside the loop is
    /// read by its attributes as well.
    #[test]
    fn a_name_that_shadows_a_holder_in_a_loop_or_a_macro_is_its_own_there() {
        let on = switched_on(&[
            (
                "a loop's target",
                "{% set ns = namespace(held='') %}",
                "",
                "{% set ns.held = close ~ 'b' %}\
                 {% for ns in [1] %}{{ ns | string | length }}{% endfor %}{{ ns.held }}",
            ),
            (
                "a macro's parameter",
                "{% set ns = namespace(held='') %}\
                 {% macro counted(ns) %}{{ ns | length }}{% endmacro %}",
                "",
                "{% set ns.held = close ~ 'b' %}{{ counted('ab') }}{{ ns.held }}",
            ),
            (
                "a new namespace under the name",
                "{% set ns = namespace(held='') %}",
                "{% set ns = namespace(held='') %}{% set ns.held = close ~ 'b' %}",
                "{% set ns.held = close ~ 'b' %}{{ ns.held }}",
            ),
        ]);
        assert_eq!(
            on,
            [
                "a loop's target",
                "a macro's parameter",
                "a new namespace under the name"
            ]
        );
    }

    /// A namespace that holds no control is read as the template likes: a
    /// counter under a filter, request text kept and searched, the
    /// namespace written whole, under another name or as an argument, and
    /// so beside a namespace that holds a control. Each template is marked.
    #[test]
    fn a_namespace_that_holds_no_control_is_read_as_the_template_likes() {
        const COUNTS: &str = "{% set ns = namespace(count=0) %}";
        let on = switched_on(&[
            (
                "a counter under a filter",
                COUNTS,
                "{% set ns.count = ns.count + 1 %}",
                "{{ ns.count | string | length }}",
            ),
            (
                "request text kept and searched",
                "{% set ns = namespace(seen='') %}",
                "{% set ns.seen = ns.seen ~ message['content'] %}",
                "{% if 'a' in ns.seen %}.{% endif %}",
            ),
            (
                "the namespace written whole",
                COUNTS,
                "",
                "{% if false %}{{ ns }}{% endif %}",
            ),
            (
                "another name for the namespace",
                COUNTS,
                "",
                "{% set other = ns %}{{ other.count }}",
            ),
            (
                "the namespace as an argument",
                "{% set ns = namespace(count=0) %}\
                 {% macro shown(space) %}{{ space.count }}{% endmacro %}",
                "",
                "{{ shown(ns) }}",
            ),
            (
                "beside a namespace that holds a control",
                "{% set ns = namespace(held='') %}{% set free = namespace(count=0) %}",
                "{% set free.count = free.count + 1 %}",
                "{% set ns.held = close ~ 'b' %}{{ ns.held }}{{ free.count | string | length }}",
            ),
        ]);
        assert_eq!(
            on,
            [
                "a counter under a filter",
                "request text kept and searched",
                "the namespace written whole",
                "another name for the namespace",
                "the namespace as an argument",
                "beside a namespace that holds a control"
            ]
        );
    }
}
