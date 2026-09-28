use std::sync::Arc;

use openai_protocol::common::Tool;
use serde_json::{json, Value};
use smg_response_template::CompiledTemplate;
use tool_parser::{types::ToolCallItem, ParserFactory, TemplateToolParser, ToolParser};

const TAG: &str = "<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>";

#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
fn template(tag_pattern: &str, strip: bool) -> Arc<CompiledTemplate> {
    let value = json!({
        "start_anchor_pattern": "<\\|turn\\|>bot",
        "fields": {
            "thinking": {"open_pattern": "<\\|think\\|>", "close": "<|done|>", "content": "text"},
            "content": {"open_pattern": "<\\|say\\|>", "close": "<|done|>", "content": "text"},
            "tool_calls": {
                "open_pattern": "(?:<\\|turn\\|>bot)?<\\|tool\\|>(?P<name>[A-Za-z_][A-Za-z0-9_.]*)<\\|args\\|>",
                "close": ["<|done|>", "<|eos|>"],
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": tag_pattern,
                    "value_parser": {"name": "text", "args": {"strip": strip}}
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
            }
        }
    });
    Arc::new(CompiledTemplate::from_value(&value).unwrap())
}

fn parser() -> TemplateToolParser {
    TemplateToolParser::new(template(TAG, false))
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
fn tools() -> Vec<Tool> {
    serde_json::from_value(json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}, "code": {"type": "string"},
                "days": {"type": "integer"}, "ratio": {"type": "number"},
                "metric": {"type": "boolean"}, "filters": {"type": "object"},
                "hours": {"type": "array"}
            }
        }}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}
    ]))
    .unwrap()
}

fn call(name: &str, args: &[(&str, &str)]) -> String {
    let tags: String = args
        .iter()
        .map(|(key, value)| format!("<arg name=\"{key}\">{value}</arg>"))
        .collect();
    format!("<|turn|>bot<|tool|>{name}<|args|>{tags}<|done|>")
}

type Calls = Vec<(String, Value)>;

#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
async fn unary(parser: &TemplateToolParser, text: &str, tools: &[Tool]) -> (String, Calls) {
    let (normal, calls) = parser.parse_complete_with_tools(text, tools).await.unwrap();
    let calls = calls
        .into_iter()
        .map(|call| {
            let arguments = serde_json::from_str(&call.function.arguments).unwrap();
            (call.function.name, arguments)
        })
        .collect();
    (normal, calls)
}

/// Stream `chunks` and finish the way the gateway does at end of stream.
#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
async fn stream(chunks: &[&str], tools: &[Tool]) -> (String, Calls, Vec<Vec<ToolCallItem>>) {
    let mut parser = parser();
    let mut normal = String::new();
    let mut steps = Vec::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, tools).await.unwrap();
        normal.push_str(&result.normal_text);
        steps.push(result.calls);
    }
    normal.push_str(&parser.take_unstreamed_normal_text());
    steps.push(parser.get_unstreamed_tool_args().unwrap_or_default());
    let mut calls: Vec<(String, String)> = Vec::new();
    for item in steps.iter().flatten() {
        if let Some(name) = &item.name {
            assert_eq!(item.tool_index, calls.len(), "name first, once per call");
            calls.push((name.clone(), String::new()));
        } else {
            calls[item.tool_index].1.push_str(&item.parameters);
        }
    }
    let calls = calls
        .into_iter()
        .map(|(name, args)| (name, serde_json::from_str(&args).unwrap()))
        .collect();
    (normal, calls, steps)
}

#[tokio::test]
async fn converts_arguments_by_declared_type() {
    let text = format!(
        "Checking. {}",
        call(
            "get_weather",
            &[
                ("city", "Paris"),
                ("code", "007"),
                ("days", " 3 "),
                ("ratio", "0.5"),
                ("metric", "true"),
                ("filters", "{\"rain\": 1}"),
                ("hours", "[1, 2]"),
                ("unit", "C"),
                ("days", "4"),
            ]
        )
    );
    let expected = json!({
        "city": "Paris", "code": "007", "days": 3, "ratio": 0.5, "metric": true,
        "filters": {"rain": 1}, "hours": [1, 2], "unit": "C"
    });
    let (normal, calls) = unary(&parser(), &text, &tools()).await;
    assert_eq!(normal, "Checking. ");
    assert_eq!(calls, [("get_weather".to_string(), expected)]);

    // A value that does not fit its type stays a string.
    let text = call("get_weather", &[("days", "soon")]);
    let (_, calls) = unary(&parser(), &text, &tools()).await;
    assert_eq!(calls[0].1, json!({"days": "soon"}));
}

#[tokio::test]
async fn streaming_matches_unary_at_every_split() {
    let full = format!(
        "Sure.{}{}Done.",
        call("get_weather", &[("city", "Paris"), ("days", "2")]),
        call("get_time", &[]),
    );
    // The last close is often an EOS token that the parser never sees.
    let eos_hidden = full.replace("Done.", "").replace("<|done|>", "<|eos|>");
    let eos_hidden = eos_hidden.strip_suffix("<|eos|>").unwrap();
    for text in [full.as_str(), eos_hidden] {
        let expected = unary(&parser(), text, &tools()).await;
        assert_eq!(expected.1.len(), 2);
        for (at, _) in text.char_indices() {
            let (normal, calls, _) = stream(&[&text[..at], &text[at..]], &tools()).await;
            assert_eq!((normal, calls), expected, "split at {at}");
        }
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        let (normal, calls, _) = stream(&chars, &tools()).await;
        assert_eq!((normal, calls), expected);
    }
}

#[tokio::test]
async fn streams_the_name_first_then_each_argument() {
    let chunks = [
        "<|tool|>get_weather",
        "<|args|>",
        "<arg name=\"city\">Pa",
        "ris</arg>",
        "<arg name=\"days\">2</arg><|do",
        "ne|>",
    ];
    let (_, calls, steps) = stream(&chunks, &tools()).await;
    let fragments: Vec<Vec<String>> = steps
        .iter()
        .map(|items| {
            items
                .iter()
                .map(|item| item.name.clone().unwrap_or_else(|| item.parameters.clone()))
                .collect()
        })
        .collect();
    assert_eq!(
        fragments,
        [
            vec![],
            vec!["get_weather".to_string()],
            vec![],
            vec!["{\"city\": \"Paris\"".to_string()],
            vec![", \"days\": 2".to_string()],
            vec!["}".to_string()],
            vec![],
        ]
    );
    assert_eq!(calls[0].1, json!({"city": "Paris", "days": 2}));
}

#[tokio::test]
async fn a_call_closed_by_eos_is_completed_at_end_of_stream() {
    // No arguments: the name is streamed as soon as the opener is complete.
    let (normal, calls, steps) = stream(&["<|tool|>get_time<|args|>"], &tools()).await;
    assert_eq!(normal, "");
    assert_eq!(calls, [("get_time".to_string(), json!({}))]);
    assert_eq!(steps[0].len(), 1);

    // An unfinished tag is dropped; complete arguments are kept.
    let text = "<|tool|>get_weather<|args|><arg name=\"city\">Paris</arg><arg name=\"da";
    let (normal, calls, _) = stream(&[text], &tools()).await;
    assert_eq!(normal, "");
    assert_eq!(
        calls,
        [("get_weather".to_string(), json!({"city": "Paris"}))]
    );
}

#[tokio::test]
async fn calls_to_undeclared_functions_are_dropped() {
    let text = format!(
        "a{}b{}",
        call("delete_all", &[("x", "1")]),
        call("get_time", &[])
    );
    let expected = ("ab".to_string(), vec![("get_time".to_string(), json!({}))]);
    assert_eq!(unary(&parser(), &text, &tools()).await, expected);
    let (normal, calls, _) = stream(&[&text], &tools()).await;
    assert_eq!((normal, calls), expected);

    // Without tool schemas nothing is checked and values stay strings.
    let (_, calls) = parser().parse_complete(&text).await.unwrap();
    assert_eq!(calls[0].function.name, "delete_all");
    assert_eq!(calls[0].function.arguments, r#"{"x":"1"}"#);
}

#[tokio::test]
async fn values_may_span_lines_and_malformed_tags_are_skipped() {
    let text = concat!(
        "<|tool|>get_weather<|args|><arg name=city>x</arg>",
        "<arg name=\"city\">Pa\nris</arg><arg name=\"code\"> 42 </arg><|done|>"
    );
    let (_, calls) = unary(&parser(), text, &tools()).await;
    assert_eq!(calls[0].1, json!({"city": "Pa\nris", "code": " 42 "}));
    let (_, streamed, _) = stream(&[text], &tools()).await;
    assert_eq!(streamed, calls);

    let strip = TemplateToolParser::new(template(TAG, true));
    let (_, calls) = unary(&strip, text, &tools()).await;
    assert_eq!(calls[0].1, json!({"city": "Pa\nris", "code": "42"}));
}

#[tokio::test]
async fn held_text_that_never_opens_a_call_is_content() {
    let (normal, calls, _) = stream(&["Almost <|to"], &tools()).await;
    assert_eq!(normal, "Almost <|to");
    assert!(calls.is_empty());
    assert!(parser().has_tool_markers(&call("get_time", &[])));
    assert!(!parser().has_tool_markers("<|tool|>get_time"));
}

#[test]
fn factory_registers_the_template_once() {
    let factory = ParserFactory::new();
    let template = template(TAG, false);
    let name = factory.register_response_template(template.clone());
    assert_eq!(name, template.parser_name());
    assert_eq!(factory.register_response_template(template), name);
    assert!(factory.registry().create_parser(&name).is_some());
}
