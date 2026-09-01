//! MiniMax M3 Parser Integration Tests
mod common;

use common::create_test_tools;
use openai_protocol::common::{Function, Tool};
use serde_json::json;
use tool_parser::{MinimaxM3Parser, ToolParser};

const NS: &str = "]<]minimax[>[";

/// Build a leaf or nested parameter element: `]<]minimax[>[<name>body]<]minimax[>[</name>`.
fn element(name: &str, body: &str) -> String {
    format!("{NS}<{name}>{body}{NS}</{name}>")
}

/// Build a single `<invoke>` block.
fn invoke(function_name: &str, body: &str) -> String {
    format!("{NS}<invoke name=\"{function_name}\">{body}{NS}</invoke>")
}

/// Build a complete tool-call wrapper around one or more invokes.
fn tool_block(invokes: &[(&str, String)]) -> String {
    let inner = invokes
        .iter()
        .map(|(name, body)| invoke(name, body))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{NS}<tool_call>\n{inner}\n{NS}</tool_call>")
}

/// Tools matching the M2 `update_task(taskId: string, subject: string)` example.
fn update_task_tools() -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "update_task".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "taskId":  {"type": "string"},
                    "subject": {"type": "string"}
                },
                "required": ["taskId", "subject"]
            }),
            strict: None,
        },
    }]
}

#[tokio::test]
async fn test_m3_complete_single_call() {
    let parser = MinimaxM3Parser::new();
    let input = format!(
        "Let me check. {}",
        tool_block(&[(
            "get_weather",
            format!(
                "{}{}",
                element("city", "Seattle"),
                element("date", "2024-12-25")
            )
        )])
    );

    let (normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(normal, "Let me check. ");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Seattle");
    assert_eq!(args["date"], "2024-12-25");
}

#[tokio::test]
async fn test_m3_multiple_invokes_in_one_block() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[
        ("get_weather", element("city", "Seattle")),
        ("search", element("query", "rust")),
    ]);

    let (normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(normal, "");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[1].function.name, "search");
}

#[tokio::test]
async fn test_m3_parallel_tool_blocks() {
    let parser = MinimaxM3Parser::new();
    let input = format!(
        "{}{}",
        tool_block(&[("get_weather", element("city", "Tokyo"))]),
        tool_block(&[("search", element("query", "weather"))]),
    );

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[1].function.name, "search");
}

#[tokio::test]
async fn test_m3_empty_arguments() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[("ping", String::new())]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "ping");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, json!({}));
}

#[tokio::test]
async fn test_m3_no_markers_passthrough() {
    let parser = MinimaxM3Parser::new();
    let input = "This is plain text mentioning get_weather but not calling it.";

    let (normal, calls) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal, input);
    assert!(calls.is_empty());
}

#[tokio::test]
async fn test_m3_schema_string_kept_as_string() {
    let parser = MinimaxM3Parser::new();
    let tools = update_task_tools();
    let input = tool_block(&[(
        "update_task",
        format!("{}{}", element("taskId", "3"), element("subject", "X")),
    )]);

    let (_normal, calls) = parser
        .parse_complete_with_tools(&input, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["taskId"],
        json!("3"),
        "string schema must stay a string"
    );
    assert_eq!(args["subject"], json!("X"));
}

#[tokio::test]
async fn test_m3_no_schema_infers_number() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[("update_task", element("taskId", "3"))]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["taskId"], json!(3), "no schema -> inferred number");
}

#[tokio::test]
async fn test_m3_nested_schema_types_respected() {
    let parser = MinimaxM3Parser::new();
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "create_task".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "meta": {
                        "type": "object",
                        "properties": { "code": {"type": "string"} }
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }),
            strict: None,
        },
    }];
    let params = format!(
        "{}{}",
        element("meta", &element("code", "007")),
        element(
            "labels",
            &format!("{}{}", element("item", "1"), element("item", "2"))
        )
    );
    let input = tool_block(&[("create_task", params)]);

    let (_normal, calls) = parser
        .parse_complete_with_tools(&input, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["meta"]["code"],
        json!("007"),
        "nested string schema must stay a string"
    );
    assert_eq!(
        args["labels"],
        json!(["1", "2"]),
        "array items schema must apply to every element"
    );
}

#[tokio::test]
async fn test_m3_mixed_text_in_object_element_preserved() {
    let parser = MinimaxM3Parser::new();
    let body = format!("note{}", element("code", "007"));
    let input = tool_block(&[("create_task", element("meta", &body))]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["meta"]["code"], json!(7), "no schema -> inferred");
    assert_eq!(
        args["meta"]["$text"],
        json!("note"),
        "stray text must be preserved under the reserved field"
    );
}

/// A parameter element that starts but never parses must reject the whole
/// invoke: partial arguments would run the tool with silently missing
/// parameters. Other invokes in the block are unaffected.
#[tokio::test]
async fn test_m3_truncated_parameter_rejects_only_that_invoke() {
    let parser = MinimaxM3Parser::new();
    let broken = format!("{}{NS}<broken>oops", element("city", "Miami"));
    let input = tool_block(&[
        ("get_weather", broken),
        ("get_weather", element("city", "Chicago")),
    ]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1, "only the intact invoke may emit a call");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], json!("Chicago"));
}

#[tokio::test]
async fn test_m3_streaming_malformed_block_returned_as_text() {
    let mut parser = MinimaxM3Parser::new();
    // A complete wrapper whose body has no parsable invoke.
    let block = format!("{NS}<tool_call>\ngarbage\n{NS}</tool_call>");

    let result = parser.parse_incremental(&block, &[]).await.unwrap();
    assert!(result.calls.is_empty());
    assert_eq!(result.normal_text, block, "malformed block must not vanish");
}

#[tokio::test]
async fn test_m3_nested_object_and_array_arguments() {
    let parser = MinimaxM3Parser::new();
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "create_order".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "user_id": { "type": "integer" },
                    "shipping": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "zip": { "type": "integer" }
                        }
                    },
                    "items": {
                        "type": "array",
                        "items": { "type": "object" }
                    }
                }
            }),
            strict: None,
        },
    }];

    let shipping = element(
        "shipping",
        &format!(
            "{}{}",
            element("city", "Singapore"),
            element("zip", "18956")
        ),
    );
    let items = element(
        "items",
        &format!(
            "{}{}",
            element(
                "item",
                &format!("{}{}", element("sku", "book-001"), element("qty", "2"))
            ),
            element(
                "item",
                &format!("{}{}", element("sku", "pen-007"), element("qty", "5"))
            ),
        ),
    );
    let body = format!("{}{}{}", element("user_id", "42"), shipping, items);
    let input = tool_block(&[("create_order", body)]);

    let (_normal, calls) = parser
        .parse_complete_with_tools(&input, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["user_id"], json!(42));
    assert_eq!(args["shipping"]["city"], json!("Singapore"));
    assert_eq!(args["shipping"]["zip"], json!(18956));
    assert_eq!(args["items"].as_array().unwrap().len(), 2);
    assert_eq!(args["items"][0]["sku"], json!("book-001"));
    assert_eq!(args["items"][1]["qty"], json!(5));
}

#[tokio::test]
async fn test_m3_streaming_single_call() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();
    let full = tool_block(&[("get_weather", element("city", "Seattle"))]);

    let chunks = [&full[..10], &full[10..25], &full[25..40], &full[40..]];

    let mut name = None;
    let mut args_json = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(n) = call.name {
                name = Some(n);
            } else {
                args_json.push_str(&call.parameters);
            }
        }
    }

    assert_eq!(name.as_deref(), Some("get_weather"));
    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
    assert_eq!(args["city"], "Seattle");
}

#[tokio::test]
async fn test_m3_streaming_char_by_char() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();
    let full = format!(
        "Hi. {} Done.",
        tool_block(&[("search", element("query", "rust programming"))])
    );

    let mut normal = String::new();
    let mut name = None;
    let mut args_json = String::new();

    let bytes: Vec<usize> = full
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(full.len()))
        .collect();
    for w in bytes.windows(2) {
        let chunk = &full[w[0]..w[1]];
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal.push_str(&result.normal_text);
        for call in result.calls {
            if let Some(n) = call.name {
                name = Some(n);
            } else {
                args_json.push_str(&call.parameters);
            }
        }
    }

    assert_eq!(name.as_deref(), Some("search"));
    assert!(normal.contains("Hi. "));
    assert!(normal.contains("Done."));
    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
    assert_eq!(args["query"], "rust programming");
}

#[tokio::test]
async fn test_m3_streaming_multiple_invokes() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();
    let full = tool_block(&[
        ("get_weather", element("city", "Seattle")),
        ("search", element("query", "x")),
    ]);
    let chunks = [&full[..15], &full[15..50], &full[50..]];

    let mut names = Vec::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(n) = call.name {
                names.push(n);
            }
        }
    }
    assert_eq!(names, vec!["get_weather".to_string(), "search".to_string()]);
}

#[tokio::test]
async fn test_m3_streaming_no_markers_passthrough() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();

    let mut normal = String::new();
    for chunk in ["Hello, ", "world!"] {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal.push_str(&result.normal_text);
        assert!(result.calls.is_empty());
    }
    assert_eq!(normal, "Hello, world!");
}

#[tokio::test]
async fn test_m3_reset_between_requests() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();

    let first = tool_block(&[("get_weather", element("city", "London"))]);
    parser.parse_incremental(&first, &tools).await.unwrap();

    parser.reset();

    let second = tool_block(&[("search", element("query", "rust"))]);
    let result = parser.parse_incremental(&second, &tools).await.unwrap();
    let name = result
        .calls
        .into_iter()
        .find_map(|c| c.name)
        .expect("second request should yield a tool name");
    assert_eq!(name, "search");
}

#[tokio::test]
async fn test_m3_xml_entities_decoded() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[("process", element("text", "if (a &amp;&amp; b) &lt;ok&gt;"))]);
    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["text"], "if (a && b) <ok>");
}

#[test]
fn test_m3_format_detection() {
    let parser = MinimaxM3Parser::new();
    assert!(parser.has_tool_markers("]<]minimax[>[<tool_call>"));
    assert!(!parser.has_tool_markers("<minimax:tool_call>"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

/// Content both before and after a single tool-call block. `parse_complete`
/// returns only the text preceding the first block (trailing text is dropped),
/// mirroring `test_minimax_content_before_and_after_tool_calls`.
#[tokio::test]
async fn test_m3_content_before_and_after_tool_calls() {
    let parser = MinimaxM3Parser::new();
    let input =
        format!(
        "I'll analyze the weather for you now.\n{}\nBased on the analysis, here's what I found.",
        tool_block(&[(
            "get_weather",
            format!("{}{}", element("city", "Boston"), element("date", "2024-12-25"))
        )])
    );

    let (normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert!(normal.contains("I'll analyze the weather for you now."));
    // Text after the tool call is not included by parse_complete.
    assert!(!normal.contains("Based on the analysis"));

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Boston");
    assert_eq!(args["date"], "2024-12-25");
}

/// Truncated tool call (start marker, invoke, one param, but no closing
/// `</tool_call>`). `parse_tool_calls` requires the end marker, so no call is
/// emitted and the raw text is returned unchanged.
#[tokio::test]
async fn test_m3_incomplete_tool_call() {
    let parser = MinimaxM3Parser::new();
    let input = format!(
        "{NS}<tool_call>\n{NS}<invoke name=\"get_weather\">{}",
        element("city", "Chicago")
    );

    let (normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 0, "incomplete block must not emit a call");
    assert_eq!(normal, input, "unterminated block returned as normal text");
}

/// Malformed invoke tag missing the `name=` attribute. `parse_invoke_name`
/// returns `None`, so `parse_invoke` yields no call and the whole input is
/// returned as normal text (mirrors `test_minimax_malformed_invoke_tag`).
#[tokio::test]
async fn test_m3_malformed_invoke_tag() {
    let parser = MinimaxM3Parser::new();
    // A bare `<invoke>` with no `name=` attribute.
    let body = element("city", "Miami");
    let input = format!("{NS}<tool_call>\n{NS}<invoke>{body}{NS}</invoke>\n{NS}</tool_call>");

    let (normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 0, "malformed invoke must not emit a call");
    assert_eq!(normal, input);
}

/// Special characters (symbols, quotes, unicode/emoji) survive as-is in leaf
/// values. Note `&` is only consumed when part of a recognized XML entity.
#[tokio::test]
async fn test_m3_special_characters_in_values() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[(
        "process",
        format!(
            "{}{}{}",
            element("text", "Special chars: @#$%^*()"),
            element("emoji", "🦀 Rust 🚀"),
            element("quotes", "\"double\" and 'single' quotes"),
        ),
    )]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Special chars: @#$%^*()");
    assert_eq!(args["emoji"], "🦀 Rust 🚀");
    assert_eq!(args["quotes"], "\"double\" and 'single' quotes");
}

/// Whitespace inside leaf values is preserved verbatim (the parser does not trim
/// leaf bodies); tabs and internal newlines are kept.
#[tokio::test]
async fn test_m3_whitespace_handling() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[(
        "process",
        format!(
            "{}{}",
            element("tabs", "\ttab\tseparated\t"),
            element(
                "newlines",
                "\n            Line 1\n            Line 2\n        "
            ),
        ),
    )]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["tabs"], "\ttab\tseparated\t");
    assert!(args["newlines"].as_str().unwrap().contains("Line 1"));
    assert!(args["newlines"].as_str().unwrap().contains("Line 2"));
}

/// A single invoke with many (12) parameters; all are parsed and inferred.
#[tokio::test]
async fn test_m3_many_parameters() {
    let parser = MinimaxM3Parser::new();
    let mut body = String::new();
    for i in 1..=12 {
        body.push_str(&element(&format!("param{i}"), &format!("value{i}")));
    }
    let input = tool_block(&[("complex_func", body)]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "complex_func");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    for i in 1..=12 {
        assert_eq!(args[format!("param{i}")], format!("value{i}"));
    }
}

/// Unknown/hallucinated function names are FORWARDED as tool calls. `parse_invoke`
/// in `minimax_m3.rs` never checks the name against `tools`; it forwards any
/// name parsed by `parse_invoke_name`. This matches M2's
/// `test_minimax_unknown_function_name_is_forwarded`.
#[tokio::test]
async fn test_m3_unknown_function_name_is_forwarded() {
    let parser = MinimaxM3Parser::new();
    let tools = create_test_tools();
    let input = tool_block(&[("invalid_function", element("param", "value"))]);

    let (normal, calls) = parser
        .parse_complete_with_tools(&input, &tools)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1, "unknown name forwarded as a tool call");
    assert_eq!(calls[0].function.name, "invalid_function");
    assert!(
        !normal.contains("<invoke"),
        "tool-call markup must not leak into normal text"
    );
}

/// With an empty `tools` slice (no schema), numeric-looking values are inferred
/// as numbers and booleans/null literals are inferred too.
#[tokio::test]
async fn test_m3_empty_tools_infers_values() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[(
        "process",
        format!(
            "{}{}{}{}",
            element("count", "42"),
            element("rate", "1.5"),
            element("enabled", "true"),
            element("data", "null"),
        ),
    )]);

    // Explicitly pass an empty tools slice: values must be inferred from text.
    let (_normal, calls) = parser.parse_complete_with_tools(&input, &[]).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["count"], json!(42));
    assert_eq!(args["rate"], json!(1.5));
    assert_eq!(args["enabled"], json!(true));
    assert_eq!(args["data"], serde_json::Value::Null);
}

/// Invalid / unparsable JSON-ish content in a leaf parameter is preserved as a
/// string (no schema -> inference falls through to string for non-numeric text).
#[tokio::test]
async fn test_m3_invalid_json_in_parameters() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[(
        "process",
        format!(
            "{}{}{}{}",
            element("valid", "{\"key\": \"value\"}"),
            element("invalid", "{invalid json: no quotes}"),
            element("broken", "[1, 2, unclosed"),
            element("mixed", "Some text {\"partial\": json} more text"),
        ),
    )]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    // No schema: leaf text is kept as a string, even when JSON-ish.
    assert_eq!(args["valid"], "{\"key\": \"value\"}");
    assert_eq!(args["invalid"], "{invalid json: no quotes}");
    assert_eq!(args["broken"], "[1, 2, unclosed");
    assert_eq!(args["mixed"], "Some text {\"partial\": json} more text");
}

/// Python-style literals (`True`/`False`/`None`) are inferred when no schema is
/// present, mirroring `test_minimax_python_literals`.
#[tokio::test]
async fn test_m3_python_literals_inferred() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[(
        "test_func",
        format!(
            "{}{}{}",
            element("bool_true", "True"),
            element("bool_false", "False"),
            element("none_val", "None"),
        ),
    )]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["bool_true"], json!(true));
    assert_eq!(args["bool_false"], json!(false));
    assert_eq!(args["none_val"], serde_json::Value::Null);
}

/// Multiline and unicode leaf values are preserved verbatim.
#[tokio::test]
async fn test_m3_multiline_parameter_values() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[(
        "process",
        format!(
            "{}{}",
            element("multiline", "line1\nline2\nline3"),
            element("unicode", "你好世界 🌍"),
        ),
    )]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["multiline"], "line1\nline2\nline3");
    assert_eq!(args["unicode"], "你好世界 🌍");
}

/// Nested XML-like text inside a leaf value: because raw `<...>` text carries no
/// `]<]minimax[>[` namespace marker, the parser treats it as plain leaf text and
/// keeps it intact.
#[tokio::test]
async fn test_m3_nested_xml_like_content_in_value() {
    let parser = MinimaxM3Parser::new();
    let input = tool_block(&[(
        "process",
        format!(
            "{}{}",
            element("template", "<html><body>Hello</body></html>"),
            element("config", "{\"key\": \"<value>nested</value>\"}"),
        ),
    )]);

    let (_normal, calls) = parser.parse_complete(&input).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["template"], "<html><body>Hello</body></html>");
    assert_eq!(args["config"], "{\"key\": \"<value>nested</value>\"}");
}

/// Multiple tool blocks fed back-to-back across streaming chunks, with the block
/// boundary aligned to the chunk boundary (mirrors
/// `test_minimax_multiple_tools_boundary`).
#[tokio::test]
async fn test_m3_streaming_multiple_blocks_boundary() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();
    let chunks = [
        tool_block(&[("get_weather", element("city", "Tokyo"))]),
        tool_block(&[("search", element("query", "weather forecast"))]),
    ];

    let mut names = Vec::new();
    for chunk in &chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(n) = call.name {
                names.push(n);
            }
        }
    }
    assert_eq!(names, vec!["get_weather".to_string(), "search".to_string()]);
}

/// Rapid streaming where the value arrives split across several mid-tag bursts
/// (distinct from the existing char-by-char test: here boundaries fall inside
/// markers and inside the value). The full call is still assembled.
#[tokio::test]
async fn test_m3_streaming_rapid_bursts() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();
    let full = tool_block(&[("search", element("query", "rust programming"))]);

    // Three uneven bursts that cut through markers and the parameter value.
    let split_a = full.len() / 3;
    let split_b = full.len() * 2 / 3;
    // Ensure splits fall on char boundaries.
    let split_a = (0..=split_a)
        .rev()
        .find(|&i| full.is_char_boundary(i))
        .unwrap();
    let split_b = (0..=split_b)
        .rev()
        .find(|&i| full.is_char_boundary(i))
        .unwrap();
    let chunks = [&full[..split_a], &full[split_a..split_b], &full[split_b..]];

    let mut name = None;
    let mut args_json = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(n) = call.name {
                name = Some(n);
            } else {
                args_json.push_str(&call.parameters);
            }
        }
    }

    assert_eq!(name.as_deref(), Some("search"));
    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
    assert_eq!(args["query"], "rust programming");
}

/// Streaming a truncated tool call: the start marker and an open invoke arrive,
/// but the closing `</tool_call>` never does. The parser waits for the end token
/// (see the `self.buffer.find(TOOL_CALL_END)` guard in `parse_incremental`) and
/// emits no call and no leaked markup.
#[tokio::test]
async fn test_m3_streaming_incomplete_tool_call_emits_nothing() {
    let mut parser = MinimaxM3Parser::new();
    let tools = create_test_tools();
    let chunks = [
        format!("{NS}<tool_call>"),
        format!("{NS}<invoke name=\"get_weather\">"),
        element("city", "Chicago"),
    ];

    let mut normal = String::new();
    let mut emitted = 0;
    for chunk in &chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal.push_str(&result.normal_text);
        emitted += result.calls.len();
    }

    assert_eq!(emitted, 0, "incomplete streamed block must not emit a call");
    assert!(
        !normal.contains("<invoke"),
        "tool-call markup must not leak into normal text"
    );
}

// ---------------------------------------------------------------------------
// Empty containers
//
// The template renders an empty sequence or mapping as an element with no
// children and no text, which is indistinguishable at the leaf from an empty
// string. Without the declared type it infers as `""` and fails the tool's
// schema — observed against MiniMax's Provider-Verifier as
// "'' is not of type 'array'" on a declared `array` parameter.
// ---------------------------------------------------------------------------

fn container_tools() -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "plan_route".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "avoid_aisles": {"type": "array", "items": {"type": "string"}},
                    "options":      {"type": "object"},
                    "note":         {"type": "string"}
                }
            }),
            strict: None,
        },
    }]
}

#[tokio::test]
async fn test_empty_leaf_under_array_schema_is_empty_array() {
    let parser = MinimaxM3Parser::new();
    let tools = container_tools();
    let text = tool_block(&[("plan_route", element("avoid_aisles", ""))]);

    let (_, calls) = parser
        .parse_complete_with_tools(&text, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["avoid_aisles"], json!([]));
}

#[tokio::test]
async fn test_empty_leaf_under_object_schema_is_empty_object() {
    let parser = MinimaxM3Parser::new();
    let tools = container_tools();
    let text = tool_block(&[("plan_route", element("options", ""))]);

    let (_, calls) = parser
        .parse_complete_with_tools(&text, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["options"], json!({}));
}

#[tokio::test]
async fn test_whitespace_only_leaf_under_array_schema_is_empty_array() {
    let parser = MinimaxM3Parser::new();
    let tools = container_tools();
    let text = tool_block(&[("plan_route", element("avoid_aisles", "\n  "))]);

    let (_, calls) = parser
        .parse_complete_with_tools(&text, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["avoid_aisles"], json!([]));
}

#[tokio::test]
async fn test_empty_leaf_under_string_schema_stays_empty_string() {
    let parser = MinimaxM3Parser::new();
    let tools = container_tools();
    let text = tool_block(&[("plan_route", element("note", ""))]);

    let (_, calls) = parser
        .parse_complete_with_tools(&text, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    // Only container types collapse; a declared string keeps the empty string.
    assert_eq!(args["note"], json!(""));
}

#[tokio::test]
async fn test_empty_leaf_without_schema_is_unchanged() {
    let parser = MinimaxM3Parser::new();
    let text = tool_block(&[("plan_route", element("avoid_aisles", ""))]);

    // With no tool schema there is nothing to say the value is a container,
    // so the previous inference behaviour is preserved.
    let (_, calls) = parser.parse_complete(&text).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["avoid_aisles"], json!(""));
}

#[tokio::test]
async fn test_populated_array_still_parses_after_empty_container_fix() {
    let parser = MinimaxM3Parser::new();
    let tools = container_tools();
    let items = format!("{}{}", element("item", "dairy"), element("item", "frozen"));
    let text = tool_block(&[("plan_route", element("avoid_aisles", &items))]);

    let (_, calls) = parser
        .parse_complete_with_tools(&text, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["avoid_aisles"], json!(["dairy", "frozen"]));
}
