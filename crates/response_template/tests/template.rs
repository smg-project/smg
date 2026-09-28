use serde_json::{json, Value};
use smg_response_template::{
    consume, find_close, partial_close_len, CompiledTemplate, Field, Opener,
};

const ALL: [Field; 3] = Field::ALL;

fn template() -> Value {
    json!({
        "start_anchor_pattern": "<\\|turn\\|>bot",
        "fields": {
            "thinking": {
                "open_pattern": "(?:<\\|turn\\|>bot)?<\\|think\\|>",
                "close": "<|done|>",
                "content": "text"
            },
            "content": {
                "open_pattern": "(?:<\\|turn\\|>bot)?<\\|say\\|>",
                "close": ["<|done|>", "<|eos|>"],
                "content": "text"
            },
            "tool_calls": {
                "open_pattern": "(?:<\\|turn\\|>bot)?<\\|tool\\|>(?P<name>[A-Za-z_][A-Za-z0-9_.]*)<\\|args\\|>",
                "close": ["<|done|>", "<|eos|>"],
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": "<arg name=\"(?P<key>[^\"]+)\"> *(?P<value>.*?)</arg>",
                    "value_parser": {"name": "text"}
                },
                "transform": {"name": "{name}", "arguments": "{content}"}
            }
        }
    })
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
fn compiled(value: &Value) -> CompiledTemplate {
    CompiledTemplate::from_value(value).unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
fn with(pointer: &str, replacement: Value) -> Value {
    let mut value = template();
    let (parent, key) = pointer.rsplit_once('/').unwrap();
    value.pointer_mut(parent).unwrap()[key] = replacement;
    value
}

#[test]
fn parser_name_follows_contents_not_key_order() {
    let a = compiled(&template());
    let reordered: Value = serde_json::from_str(
        &serde_json::to_string(&json!({
            "fields": template()["fields"],
            "start_anchor_pattern": "<\\|turn\\|>bot",
        }))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(a.parser_name(), compiled(&reordered).parser_name());
    assert!(a.parser_name().starts_with("response_template_"));

    let changed = with("/fields/content/close", json!("<|stop|>"));
    assert_ne!(a.parser_name(), compiled(&changed).parser_name());
    assert_eq!(a.tag().close, "</arg>");
    // `.` matches a newline in template patterns.
    let tag = a
        .tag()
        .regex
        .captures("<arg name=\"k\">a\nb</arg>")
        .unwrap();
    assert_eq!(&tag["value"], "a\nb");
    assert_eq!(a.closes(Field::Content), ["<|done|>", "<|eos|>"]);
}

#[test]
fn rejects_templates_outside_the_supported_subset() {
    let mut extra = template();
    extra["fields"]["summary"] = extra["fields"]["content"].clone();
    let mut unknown_key = template();
    unknown_key["implicit"] = json!("content");
    let mut missing = template();
    missing["fields"]
        .as_object_mut()
        .unwrap()
        .remove("thinking");

    let cases = [
        ("extra field", extra),
        ("unknown key", unknown_key),
        ("missing field", missing),
        (
            "json content",
            with("/fields/content/content", json!("json")),
        ),
        (
            "repeating text",
            with("/fields/thinking/repeats", json!(true)),
        ),
        (
            "bad regex",
            with("/fields/content/open_pattern", json!("(<")),
        ),
        (
            "empty match",
            with("/fields/content/open_pattern", json!("x*")),
        ),
        ("empty close", with("/fields/content/close", json!([]))),
        (
            "no name capture",
            with("/fields/tool_calls/open_pattern", json!("<\\|tool\\|>")),
        ),
        (
            "no value capture",
            with(
                "/fields/tool_calls/content_args/tag_pattern",
                json!("<arg name=\"(?P<key>[^\"]+)\">.*?</arg>"),
            ),
        ),
        (
            "tag without a closing literal",
            with(
                "/fields/tool_calls/content_args/tag_pattern",
                json!("<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*)"),
            ),
        ),
        (
            "value parser",
            with(
                "/fields/tool_calls/content_args/value_parser/name",
                json!("json"),
            ),
        ),
        (
            "transform",
            with("/fields/tool_calls/transform", json!({"call": "{name}"})),
        ),
        (
            "unicode word boundary",
            with("/fields/content/open_pattern", json!("\\bsay\\b")),
        ),
    ];
    for (name, value) in cases {
        assert!(CompiledTemplate::from_value(&value).is_err(), "{name}");
    }
    let wrapped = with(
        "/fields/tool_calls/transform",
        json!({"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}),
    );
    assert!(CompiledTemplate::from_value(&wrapped).is_ok());
}

#[test]
fn complete_openers_commit_at_the_buffer_edge() {
    let t = compiled(&template());
    let hay = "x<|tool|>get_weather<|args|>";
    assert_eq!(
        t.scan(hay, 0, &ALL, false),
        Some(Opener::Complete {
            field: Field::ToolCalls,
            start: 1,
            end: hay.len()
        })
    );
    let captures = t.opener_captures(Field::ToolCalls, hay, 1).unwrap();
    assert_eq!(&captures["name"], "get_weather");
    // Only the requested fields are considered.
    assert_eq!(t.scan(hay, 0, &[Field::Content], false), None);
}

#[test]
fn partial_and_ambiguous_openers_are_pending() {
    let t = compiled(&template());
    for hay in [
        "ab<|to",
        "ab<|turn|>bot",
        "ab<|tool|>get_wea",
        "ab<|tool|>get<|ar",
    ] {
        assert_eq!(
            t.scan(hay, 0, &ALL, false),
            Some(Opener::Pending { start: 2 }),
            "{hay}"
        );
        // Nothing is pending at end of input.
        assert_eq!(t.scan(hay, 0, &ALL, true), None, "{hay}");
    }
    assert_eq!(
        t.scan("<|turn|>bot<|say|>", 0, &ALL, false),
        Some(Opener::Complete {
            field: Field::Content,
            start: 0,
            end: 18
        })
    );
    assert_eq!(t.scan("plain < text", 0, &ALL, false), None);
}

#[test]
fn an_opener_that_can_still_grow_waits_for_more_input() {
    let t = compiled(&with(
        "/fields/content/open_pattern",
        json!("<\\|say\\|> *"),
    ));
    assert_eq!(
        t.scan("<|say|> ", 0, &ALL, false),
        Some(Opener::Pending { start: 0 })
    );
    assert_eq!(
        t.scan("<|say|> x", 0, &ALL, false),
        Some(Opener::Complete {
            field: Field::Content,
            start: 0,
            end: 8
        })
    );
    assert_eq!(
        t.scan("<|say|> ", 0, &ALL, true),
        Some(Opener::Complete {
            field: Field::Content,
            start: 0,
            end: 8
        })
    );
}

#[test]
fn start_anchors_see_the_context_before_from() {
    let t = compiled(&with(
        "/fields/thinking/open_pattern",
        json!("^<\\|think\\|>"),
    ));
    let opener = Some(Opener::Complete {
        field: Field::Thinking,
        start: 0,
        end: 9,
    });
    assert_eq!(t.scan("<|think|>", 0, &ALL, false), opener);
    // `x` is look-behind context, so the text no longer starts at `^`.
    assert_eq!(t.scan("x<|think|>", 1, &ALL, false), None);
}

#[test]
fn close_helpers() {
    let closes = ["<|done|>".to_string(), "<|do|>".to_string()];
    assert_eq!(find_close("ab<|do|>c<|done|>", &closes), Some((2, 8)));
    assert_eq!(find_close("ab", &closes), None);
    assert_eq!(partial_close_len("text<|do", &closes), 4);
    assert_eq!(partial_close_len("text<|done|", &closes), 7);
    assert_eq!(partial_close_len("text", &closes), 0);

    let mut buffer = "héllo".to_string();
    let context = consume(&mut buffer, 3);
    assert_eq!((buffer.as_str(), context), ("éllo", 2));
    assert_eq!(consume(&mut buffer, 0), 0);
}
