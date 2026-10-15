//! MiniMax M3 Parser Integration Tests
mod common;

use common::create_test_tools;
use openai_protocol::common::{Function, Tool, ToolChoice, ToolChoiceValue};
use serde_json::json;
use tool_parser::{GrammarStart, MinimaxM3Parser, ParserFactory, ToolConstraint, ToolParser};

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
            extra: Default::default(),
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
            extra: Default::default(),
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
            extra: Default::default(),
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
async fn test_m3_streaming_wrapper_and_invoke_across_every_chunk_boundary() {
    let tools = create_test_tools();
    let full = tool_block(&[("get_weather", element("city", "Seattle"))]);
    let invoke_header = "name=\"get_weather\">";
    let prefix_end = full.find(invoke_header).unwrap() + invoke_header.len();

    for split in 1..prefix_end {
        let mut parser = MinimaxM3Parser::new();

        let first = parser
            .parse_incremental(&full[..split], &tools)
            .await
            .unwrap();
        assert!(first.normal_text.is_empty(), "split {split}");
        assert!(first.calls.is_empty(), "split {split}");

        let second = parser
            .parse_incremental(&full[split..], &tools)
            .await
            .unwrap();
        assert!(second.normal_text.is_empty(), "split {split}");
        assert_eq!(
            second.calls.iter().find_map(|call| call.name.as_deref()),
            Some("get_weather"),
            "split {split}"
        );
    }
}

#[tokio::test]
async fn test_m3_streaming_false_invoke_prefix_recovers_at_every_divergence() {
    let tools = create_test_tools();
    let wrapper = format!("{NS}<tool_call>");
    let possible_invoke = format!("\n\t{NS}<invoke");

    // Every prefix through the complete marker remains viable, including
    // whitespace after the wrapper. The first divergent byte must release the
    // entire candidate, including when it follows a complete invoke marker.
    for split in 0..=possible_invoke.len() {
        let mut parser = MinimaxM3Parser::new();
        let held = format!("{wrapper}{}", &possible_invoke[..split]);

        let first = parser.parse_incremental(&held, &tools).await.unwrap();
        assert!(first.normal_text.is_empty(), "split {split}");
        assert!(first.calls.is_empty(), "split {split}");

        let recovered = parser.parse_incremental("X", &tools).await.unwrap();
        assert_eq!(recovered.normal_text, format!("{held}X"), "split {split}");
        assert!(recovered.calls.is_empty(), "split {split}");

        let tail = parser
            .parse_incremental(" ordinary tail", &tools)
            .await
            .unwrap();
        assert_eq!(tail.normal_text, " ordinary tail", "split {split}");
        assert!(tail.calls.is_empty(), "split {split}");
    }
}

#[tokio::test]
async fn test_m3_streaming_eof_returns_incomplete_candidates_and_resets_state() {
    let tools = create_test_tools();
    let wrapper = format!("{NS}<tool_call>");
    let candidates = [
        wrapper[..wrapper.len() - 1].to_string(),
        format!("{wrapper}\n  {NS}<invoke name=\"get_weather\">"),
    ];

    for candidate in candidates {
        let mut parser = MinimaxM3Parser::new();
        let result = parser.parse_incremental(&candidate, &tools).await.unwrap();
        assert!(result.normal_text.is_empty(), "candidate {candidate:?}");
        assert!(result.calls.is_empty(), "candidate {candidate:?}");

        assert_eq!(parser.take_unstreamed_normal_text(), candidate);
        assert_eq!(parser.take_unstreamed_normal_text(), "");

        let next = parser
            .parse_incremental("ordinary text", &tools)
            .await
            .unwrap();
        assert_eq!(next.normal_text, "ordinary text");
        assert!(next.calls.is_empty());
    }
}

#[tokio::test]
async fn test_m3_streaming_eof_returns_new_candidate_after_completed_call() {
    let tools = create_test_tools();
    let mut parser = MinimaxM3Parser::new();
    let call = tool_block(&[("get_weather", element("city", "Seattle"))]);

    let complete = parser.parse_incremental(&call, &tools).await.unwrap();
    assert_eq!(
        complete.calls.iter().find_map(|item| item.name.as_deref()),
        Some("get_weather")
    );

    let incomplete = format!("{NS}<tool_call>\n{NS}<invoke name=\"search\">");
    let pending = parser.parse_incremental(&incomplete, &tools).await.unwrap();
    assert!(pending.normal_text.is_empty());
    assert!(pending.calls.is_empty());
    assert_eq!(parser.take_unstreamed_normal_text(), incomplete);
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
            extra: Default::default(),
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

#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
fn composite_schema_tools(combinator: &str) -> Vec<Tool> {
    let branches = json!([
        {
            "type": "object",
            "properties": {
                "string_list": {"type": "array", "items": {"type": "string"}}
            }
        },
        {
            "type": "object",
            "properties": {
                "number_list": {"type": "array", "items": {"type": "number"}}
            }
        }
    ]);
    let mut nested = json!({"type": "object"});
    nested
        .as_object_mut()
        .unwrap()
        .insert(combinator.to_string(), branches.clone());
    let mut parameters = json!({
        "type": "object",
        "properties": {
            "plain": {"type": "string"},
            "nested": nested
        }
    });
    parameters
        .as_object_mut()
        .unwrap()
        .insert(combinator.to_string(), branches);

    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "composite_schema".to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    }]
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
async fn assert_composite_property_schemas(combinator: &str) {
    let parser = MinimaxM3Parser::new();
    let tools = composite_schema_tools(combinator);
    let string_items = format!("{}{}", element("item", "12"), element("item", "34"));
    let number_items = format!("{}{}", element("item", "12"), element("item", "34"));
    let input = tool_block(&[
        ("composite_schema", element("plain", "007")),
        ("composite_schema", element("string_list", &string_items)),
        ("composite_schema", element("number_list", &number_items)),
    ]);

    let (_, calls) = parser
        .parse_complete_with_tools(&input, &tools)
        .await
        .unwrap();
    let arguments = calls
        .iter()
        .map(|call| serde_json::from_str::<serde_json::Value>(&call.function.arguments).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(arguments[0], json!({"plain": "007"}));
    assert_eq!(arguments[1], json!({"string_list": ["12", "34"]}));
    assert_eq!(arguments[2], json!({"number_list": [12, 34]}));
}

#[tokio::test]
async fn test_m3_resolves_property_schemas_in_top_level_one_of() {
    assert_composite_property_schemas("oneOf").await;
}

#[tokio::test]
async fn test_m3_resolves_property_schemas_in_top_level_any_of() {
    assert_composite_property_schemas("anyOf").await;
}

#[tokio::test]
async fn test_m3_resolves_property_schemas_in_top_level_all_of() {
    assert_composite_property_schemas("allOf").await;
}

#[tokio::test]
async fn test_m3_resolves_nested_composite_property_schemas() {
    for combinator in ["oneOf", "anyOf", "allOf"] {
        let parser = MinimaxM3Parser::new();
        let tools = composite_schema_tools(combinator);
        let string_items = format!("{}{}", element("item", "12"), element("item", "34"));
        let number_items = format!("{}{}", element("item", "12"), element("item", "34"));
        let input = tool_block(&[
            (
                "composite_schema",
                element("nested", &element("string_list", &string_items)),
            ),
            (
                "composite_schema",
                element("nested", &element("number_list", &number_items)),
            ),
        ]);

        let (_, calls) = parser
            .parse_complete_with_tools(&input, &tools)
            .await
            .unwrap();
        let string_args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let number_args: serde_json::Value =
            serde_json::from_str(&calls[1].function.arguments).unwrap();

        assert_eq!(
            string_args,
            json!({"nested": {"string_list": ["12", "34"]}}),
            "{combinator} must preserve nested string array items"
        );
        assert_eq!(
            number_args,
            json!({"nested": {"number_list": [12, 34]}}),
            "{combinator} must coerce nested number array items"
        );
    }
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

/// Reduced from M3 output where Sunday's empty time_blocks omitted its close
/// before </item>. Keep nested array schema traversal and later parameters.
#[tokio::test]
async fn test_m3_missing_empty_container_close_in_nested_array() {
    let tools = vec![Tool {
        tool_type: "function".into(),
        function: Function {
            name: "plan".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "days": {"type": "array", "items": {
                    "type": "object", "properties": {
                        "name": {"type": "string"},
                        "time_blocks": {"type": "array", "items": {"type": "string"}}
                    }
                }},
                "timezone": {"type": "string"}
            }}),
            strict: None,
            extra: Default::default(),
        },
    }];
    let days = format!(
        "{}{}{}",
        element(
            "item",
            &format!(
                "{}{}",
                element("name", "Saturday"),
                element("time_blocks", &element("item", "10:00–13:00"))
            )
        ),
        element(
            "item",
            &format!("{}{NS}<time_blocks>", element("name", "Sunday"))
        ),
        element(
            "item",
            &format!(
                "{}{}",
                element("name", "Monday"),
                element("time_blocks", &element("item", "18:30–21:00"))
            )
        )
    );
    let text = tool_block(&[(
        "plan",
        format!(
            "{}{}",
            element("days", &days),
            element("timezone", "Asia/Shanghai")
        ),
    )]);
    let expected = json!({
        "days": [
            {"name": "Saturday", "time_blocks": ["10:00–13:00"]},
            {"name": "Sunday", "time_blocks": []},
            {"name": "Monday", "time_blocks": ["18:30–21:00"]}
        ],
        "timezone": "Asia/Shanghai"
    });
    let (normal, calls) = MinimaxM3Parser::new()
        .parse_complete_with_tools(&text, &tools)
        .await
        .unwrap();
    assert!(normal.is_empty());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "plan");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, expected);

    let chars: Vec<char> = text.chars().collect();
    for size in [1, 2, 7, 29, chars.len()] {
        let mut parser = MinimaxM3Parser::new();
        let mut arguments = String::new();
        let mut names = Vec::new();
        for chunk in chars.chunks(size) {
            let result = parser
                .parse_incremental(&chunk.iter().collect::<String>(), &tools)
                .await
                .unwrap();
            assert!(result.normal_text.is_empty(), "chunk size {size}");
            for call in result.calls {
                assert_eq!(call.tool_index, 0);
                if let Some(name) = call.name {
                    names.push(name);
                }
                arguments.push_str(&call.parameters);
            }
        }
        assert!(parser.take_unstreamed_normal_text().is_empty());
        assert_eq!(names, ["plan"]);
        let args: serde_json::Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(args, expected, "chunk size {size}");
    }
}

#[tokio::test]
async fn test_m3_empty_container_recovery_requires_schema_and_parent_close() {
    for (kind, content, close, recover) in [
        ("array", "", "parent", true),
        ("object", " \n", "parent", true),
        ("string", "", "parent", false),
        ("array", "lost value", "parent", false),
        ("array", "", "wrong", false),
        ("array", &element("item", "value"), "parent", false),
    ] {
        let tools = vec![Tool {
            tool_type: "function".into(),
            function: Function {
                name: "test".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {
                    "parent": {"type": "object", "properties": {
                        "empty": {"type": kind}
                    }}
                }}),
                strict: None,
                extra: Default::default(),
            },
        }];
        let text = tool_block(&[(
            "test",
            format!("{NS}<parent>{NS}<empty>{content}{NS}</{close}>"),
        )]);
        let (_, calls) = MinimaxM3Parser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), usize::from(recover));
        if recover {
            let args: serde_json::Value =
                serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(
                args["parent"]["empty"],
                if kind == "array" {
                    json!([])
                } else {
                    json!({})
                }
            );
        }
        let (_, no_schema_calls) = MinimaxM3Parser::new().parse_complete(&text).await.unwrap();
        assert!(no_schema_calls.is_empty());

        for supplied_tools in [&tools[..], &[][..]] {
            let mut parser = MinimaxM3Parser::new();
            let mut arguments = String::new();
            let mut normal = String::new();
            for ch in text.chars() {
                let result = parser
                    .parse_incremental(&ch.to_string(), supplied_tools)
                    .await
                    .unwrap();
                normal.push_str(&result.normal_text);
                for call in result.calls {
                    arguments.push_str(&call.parameters);
                }
            }
            normal.push_str(&parser.take_unstreamed_normal_text());
            assert_eq!(
                normal,
                if recover && !supplied_tools.is_empty() {
                    ""
                } else {
                    &text
                }
            );
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&arguments).is_ok(),
                recover && !supplied_tools.is_empty()
            );
        }
    }
}

#[tokio::test]
async fn test_m3_composed_container_recovery() {
    for keyword in ["oneOf", "anyOf", "allOf"] {
        let wrap = |schema: serde_json::Value| {
            let branches = if keyword == "allOf" {
                vec![schema]
            } else {
                vec![json!({"type": "null"}), schema]
            };
            json!({keyword: branches})
        };
        let item_schema = wrap(json!({"type": "object", "properties": {
            "values": wrap(json!({"type": "array", "items": {"type": "string"}})),
            "options": wrap(json!({"type": "object"}))
        }}));
        let tools = vec![Tool {
            tool_type: "function".into(),
            function: Function {
                name: "test".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {
                    "rows": wrap(json!({"type": "array", "items": item_schema}))
                }}),
                strict: None,
                extra: Default::default(),
            },
        }];
        let rows = format!(
            "{}{}{}",
            element("item", &format!("{NS}<values>")),
            element("item", &format!("{NS}<options> \n")),
            element("item", &element("values", &element("item", "001")))
        );
        let text = tool_block(&[("test", element("rows", &rows))]);
        let expected = json!({"rows": [
            {"values": []}, {"options": {}}, {"values": ["001"]}
        ]});
        let (normal, calls) = MinimaxM3Parser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        assert!(normal.is_empty(), "{keyword}");
        assert_eq!(calls.len(), 1, "{keyword}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            expected,
            "{keyword}"
        );

        let mut parser = MinimaxM3Parser::new();
        let mut arguments = String::new();
        for ch in text.chars() {
            let result = parser
                .parse_incremental(&ch.to_string(), &tools)
                .await
                .unwrap();
            assert!(result.normal_text.is_empty(), "{keyword}");
            for call in result.calls {
                arguments.push_str(&call.parameters);
            }
        }
        assert!(parser.take_unstreamed_normal_text().is_empty());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
            expected
        );
    }
}

#[tokio::test]
async fn test_m3_composed_container_recovery_rejects_ambiguous_schemas() {
    for schema in [
        json!({"anyOf": [{"type": "array"}, {"type": "object"}]}),
        json!({"oneOf": [{"type": "array"}, {"type": "array", "items": {"type": "string"}}]}),
        json!({"anyOf": [{"type": "array"}, {"type": "string"}]}),
        json!({"anyOf": [{"type": "array"}, {}]}),
        json!({"allOf": [{"type": "array"}, {"type": "null"}]}),
        json!({"allOf": [{"type": "array"}, {"items": {"type": "string"}}]}),
        json!({"anyOf": [{"type": "array"}, {"$ref": "#/$defs/other"}]}),
        json!({"anyOf": [{"type": "array"}], "not": {"type": "array"}}),
    ] {
        let tools = vec![Tool {
            tool_type: "function".into(),
            function: Function {
                name: "test".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {
                    "parent": {"type": "object", "properties": {"empty": schema}}
                }}),
                strict: None,
                extra: Default::default(),
            },
        }];
        let text = tool_block(&[("test", format!("{NS}<parent>{NS}<empty>{NS}</parent>"))]);
        let (normal, calls) = MinimaxM3Parser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        assert_eq!(normal, text, "{schema}");
        assert!(calls.is_empty(), "{schema}");
        let mut parser = MinimaxM3Parser::new();
        let mut normal = String::new();
        for ch in text.chars() {
            let result = parser
                .parse_incremental(&ch.to_string(), &tools)
                .await
                .unwrap();
            assert!(result.calls.is_empty(), "{schema}");
            normal.push_str(&result.normal_text);
        }
        normal.push_str(&parser.take_unstreamed_normal_text());
        assert_eq!(normal, text, "{schema}");
    }
}

/// A call to a tool the request never declared (the conversation established
/// it) still parses, complete and streaming, with no schema to coerce by.
#[tokio::test]
async fn test_m3_undeclared_tool_parses_without_a_tool_inventory() {
    let text = tool_block(&[("list_skills", String::new())]);
    let (normal, calls) = MinimaxM3Parser::new()
        .parse_complete_with_tools(&text, &[])
        .await
        .unwrap();
    assert_eq!(normal, "");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "list_skills");
    assert_eq!(calls[0].function.arguments, "{}");

    let mut parser = MinimaxM3Parser::new();
    let mut streamed = Vec::new();
    let mut normal = String::new();
    for ch in text.chars() {
        let result = parser
            .parse_incremental(&ch.to_string(), &[])
            .await
            .unwrap();
        streamed.extend(result.calls);
        normal.push_str(&result.normal_text);
    }
    normal.push_str(&parser.take_unstreamed_normal_text());
    assert_eq!(normal, "");
    let names: Vec<&str> = streamed
        .iter()
        .filter_map(|call| call.name.as_deref())
        .collect();
    assert_eq!(names, ["list_skills"]);
    // The name arrives on its own and the arguments follow, so the pieces have
    // to add back up to what the one-shot parse returned. Checking the name
    // alone would also pass on a call that streamed no arguments at all.
    let arguments: String = streamed
        .iter()
        .map(|call| call.parameters.as_str())
        .collect();
    assert_eq!(arguments, "{}");
}

// ---------------------------------------------------------------------------
// Forced tool choices: the structural tag and the reasoning prefix
// ---------------------------------------------------------------------------

/// One tool with a shaped schema and one without parameters.
fn constraint_tools() -> Vec<Tool> {
    let tool = |name: &str, parameters: serde_json::Value| Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    };
    vec![
        tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1},
                    "units": {"type": "string", "enum": ["c", "f"]},
                    "verbose": {"type": "boolean"},
                    "where": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "label": {"type": "string"}},
                        "required": ["x"],
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "nothing": {"type": "null"},
                    "any": {},
                },
                "required": ["city", "days", "verbose", "where", "tags", "nothing", "any"],
            }),
        ),
        tool("ping", json!({"type": "object"})),
    ]
}

/// The markers that end a call, which no value may hold.
fn exits() -> Vec<&'static str> {
    vec!["</invoke>", "<invoke name=\"", "</tool_call>"]
}

/// The expected element: the separator and the key's tag, the value, the separator and the
/// closing tag.
fn expected_element(key: &str, value: serde_json::Value) -> serde_json::Value {
    json!({"type": "sequence", "elements": [
        {"type": "const_string", "value": format!("{NS}<{key}>")},
        value,
        {"type": "const_string", "value": format!("{NS}</{key}>")},
    ]})
}

/// The expected text of a string inside the element `key`.
fn expected_text(key: &str) -> serde_json::Value {
    let close = format!("</{key}>");
    let mut excludes = vec![NS, close.as_str()];
    excludes.extend(exits());
    json!({"type": "any_text", "excludes": excludes})
}

/// What the tag admits before the block: the thought, the bare close of an empty one, or nothing.
fn expected_before_the_block() -> serde_json::Value {
    json!({"type": "optional", "content": {"type": "or", "elements": [
        {
            "type": "tag",
            "begin": "<mm:think>",
            "content": {"type": "any_text", "excludes": ["<mm:think>", "</mm:think>", NS]},
            "end": "</mm:think>",
        },
        {"type": "const_string", "value": "</mm:think>"},
    ]}})
}

#[test]
fn m3_structural_tag_spells_the_block_in_the_models_syntax() {
    let mut tools = constraint_tools();
    let mut nameless = tools[0].clone();
    nameless.function.name.clear();
    tools.push(nameless);

    for at_least_one in [false, true] {
        let tag = MinimaxM3Parser::build_structural_tag(&tools, at_least_one);
        let format = &tag["format"];
        assert_eq!(format["type"], "sequence");
        let elements = format["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 2, "what precedes the block, then the block");
        assert_eq!(elements[0], expected_before_the_block());

        let calls = &elements[1];
        assert_eq!(calls["type"], "triggered_tags");
        assert_eq!(calls["triggers"], json!([format!("{NS}<tool_call>")]));
        assert_eq!(calls["at_least_one"], at_least_one);
        let blocks = calls["tags"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tag");
        assert_eq!(blocks[0]["begin"], format!("{NS}<tool_call>\n"));
        assert_eq!(blocks[0]["end"], format!("{NS}</tool_call>"));
        let invokes = &blocks[0]["content"];
        assert_eq!(invokes["type"], "tags_with_separator");
        assert_eq!(invokes["separator"], "");
        assert_eq!(
            invokes["at_least_one"], true,
            "a block holds at least one invoke"
        );
        let invokes = invokes["tags"].as_array().unwrap();
        assert_eq!(invokes.len(), 2, "the nameless tool is excluded");

        let weather = &invokes[0];
        assert_eq!(weather["type"], "tag");
        assert_eq!(
            weather["begin"],
            format!("{NS}<invoke name=\"get_weather\">")
        );
        assert_eq!(weather["end"], format!("{NS}</invoke>\n"));
        assert_eq!(
            weather["content"],
            json!({"type": "sequence", "elements": [
                expected_element("city", expected_text("city")),
                expected_element(
                    "days",
                    json!({"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}}),
                ),
                {"type": "optional", "content": expected_element(
                    "units",
                    json!({"type": "or", "elements": [
                        {"type": "const_string", "value": "c"},
                        {"type": "const_string", "value": "f"},
                    ]}),
                )},
                expected_element(
                    "verbose",
                    json!({"type": "or", "elements": [
                        {"type": "const_string", "value": "true"},
                        {"type": "const_string", "value": "false"},
                        {"type": "const_string", "value": "True"},
                        {"type": "const_string", "value": "False"},
                    ]}),
                ),
                expected_element(
                    "where",
                    json!({"type": "sequence", "elements": [
                        expected_element(
                            "x",
                            json!({"type": "json_schema", "json_schema": {"type": "number"}}),
                        ),
                        {"type": "optional", "content": expected_element("label", expected_text("label"))},
                    ]}),
                ),
                expected_element(
                    "tags",
                    json!({"type": "star", "content": expected_element("item", expected_text("item"))}),
                ),
                expected_element(
                    "any",
                    json!({"type": "any_text", "excludes": ["</any>", "</invoke>", "<invoke name=\"", "</tool_call>"]}),
                ),
            ]}),
            "one element per property in the schema's order, the null left out"
        );

        let ping = &invokes[1];
        assert_eq!(ping["begin"], format!("{NS}<invoke name=\"ping\">"));
        assert_eq!(
            ping["content"],
            json!({"type": "any_text", "excludes": exits()}),
            "a schema without properties takes any text"
        );
    }

    let unnamed = vec![tools.pop().unwrap()];
    let tag = MinimaxM3Parser::build_structural_tag(&unnamed, true);
    assert_eq!(
        tag["format"]["elements"][1]["tags"][0]["content"],
        json!({"type": "const_string", "value": ""}),
        "with no named tool the block is empty"
    );
}

#[test]
fn m3_references_and_nulls_in_lists_follow_the_schema() {
    let tool = Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "t".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "$defs": {
                    "Point": {"type": "object", "properties": {"x": {"type": "integer"}}},
                    "Count": {"type": "integer"},
                },
                "properties": {
                    "at": {"$ref": "#/$defs/Point"},
                    "count": {"$ref": "#/$defs/Count"},
                    "holes": {"type": "array", "items": {"type": "null"}},
                },
                "required": ["at", "count", "holes"],
            }),
            strict: None,
            extra: Default::default(),
        },
    };
    let tag = MinimaxM3Parser::build_structural_tag(&[tool], true);
    let elements = tag["format"]["elements"][1]["tags"][0]["content"]["tags"][0]["content"]
        ["elements"]
        .clone();
    assert_eq!(
        elements[0],
        expected_element(
            "at",
            json!({"type": "sequence", "elements": [
                {"type": "optional", "content": expected_element(
                    "x",
                    json!({"type": "json_schema", "json_schema": {"type": "integer"}}),
                )},
            ]}),
        ),
        "a reference lends its target's properties"
    );
    assert_eq!(
        elements[1]["elements"][1],
        json!({"type": "json_schema", "json_schema": {"type": "integer"}}),
        "a referenced number is its target's schema"
    );
    assert_eq!(
        elements[2],
        expected_element(
            "holes",
            json!({"type": "star", "content": {
                "type": "const_string",
                "value": format!("{NS}<item>{NS}</item>"),
            }}),
        ),
        "a list of nulls holds empty items"
    );
}

#[test]
fn m3_constrains_only_forced_tool_choices() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    let parser = Some("minimax_m3");
    let tools = constraint_tools();
    assert!(registry.has_structural_tag("minimax_m3"));
    assert!(registry.has_reasoning_prefix(parser));
    let structural_tag = |constraint: Option<ToolConstraint>| -> serde_json::Value {
        match constraint {
            Some(ToolConstraint::StructuralTag(tag)) => serde_json::from_str(&tag).unwrap(),
            other => panic!("expected a structural tag, got {other:?}"),
        }
    };

    for choice in [
        json!("auto"),
        json!("none"),
        json!({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "get_weather"}]}),
    ] {
        let choice: ToolChoice = serde_json::from_value(choice).unwrap();
        for reasoning in [false, true] {
            assert!(registry
                .generate_tool_constraint(parser, &tools, &choice, reasoning)
                .unwrap()
                .is_none());
        }
    }
    let required = ToolChoice::Value(ToolChoiceValue::Required);
    assert!(registry
        .generate_tool_constraint(parser, &[], &required, false)
        .unwrap()
        .is_none());

    for (choice, selected) in [
        (json!("required"), tools.as_slice()),
        (
            json!({"type": "function", "function": {"name": "get_weather"}}),
            &tools[..1],
        ),
        (
            json!({"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}),
            &tools[..1],
        ),
    ] {
        // The gateway filters named/allowed tools before invoking the registry.
        let choice: ToolChoice = serde_json::from_value(choice).unwrap();
        let plain = MinimaxM3Parser::build_structural_tag(selected, true);
        let thinking = json!({
            "format": {
                "type": "sequence",
                "elements": [
                    MinimaxM3Parser::reasoning_prefix(GrammarStart::Unknown).unwrap(),
                    plain["format"]
                ],
            },
        });
        for (reasoning, expected) in [(false, &plain), (true, &thinking)] {
            let constraint = registry
                .generate_tool_constraint(parser, selected, &choice, reasoning)
                .unwrap();
            assert!(
                !matches!(constraint, Some(ToolConstraint::JsonSchema(_))),
                "no JSON-schema fallback for a parser with a structural tag"
            );
            assert_eq!(&structural_tag(constraint), expected);
        }
    }
}

/// On a prompt that ends inside the thought the forced call follows free text
/// with nothing owed at its end; with the worker known, an engine that starts
/// the grammar after its reasoning parser's gate gets the bare tag, one that
/// starts at the first token the `</mm:think>`-closed thought in front of it.
/// A tag built for a prompt outside the thought is never re-shaped: what it
/// admits before the block is its own, not the prefix.
#[test]
fn m3_forced_call_grammar_follows_where_the_engine_starts_it() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    let parser = Some("minimax_m3");
    let tools = constraint_tools();
    let required = ToolChoice::Value(ToolChoiceValue::Required);
    let tag = |reasoning: bool| -> String {
        match registry
            .generate_tool_constraint(parser, &tools, &required, reasoning)
            .unwrap()
        {
            Some(ToolConstraint::StructuralTag(tag)) => tag,
            other => panic!("expected a structural tag, got {other:?}"),
        }
    };
    let plain = MinimaxM3Parser::build_structural_tag(&tools, true);
    let thinking = tag(true);
    let thinking_value: serde_json::Value = serde_json::from_str(&thinking).unwrap();
    let prefix = &thinking_value["format"]["elements"][0];
    assert_eq!(prefix["type"], "any_text");
    assert!(
        prefix.get("end").is_none(),
        "nothing owed at the text's end"
    );
    assert_eq!(prefix["excludes"], json!([NS]));

    assert!(registry
        .structural_tag_for_grammar_start(parser, &thinking, GrammarStart::Unknown)
        .unwrap()
        .is_none());

    let deferred = registry
        .structural_tag_for_grammar_start(parser, &thinking, GrammarStart::AfterReasoning)
        .unwrap()
        .expect("re-shaped");
    let deferred: serde_json::Value = serde_json::from_str(&deferred).unwrap();
    assert_eq!(deferred, plain, "nothing precedes the tag");

    let first_token = registry
        .structural_tag_for_grammar_start(parser, &thinking, GrammarStart::FirstToken)
        .unwrap()
        .expect("re-shaped");
    let first_token: serde_json::Value = serde_json::from_str(&first_token).unwrap();
    assert_eq!(
        first_token,
        json!({
            "format": {
                "type": "sequence",
                "elements": [
                    MinimaxM3Parser::reasoning_prefix(GrammarStart::FirstToken).unwrap(),
                    plain["format"]
                ],
            },
        })
    );
    let thought = &first_token["format"]["elements"][0];
    assert_eq!(thought["begin"], "");
    assert_eq!(thought["end"], "</mm:think>");
    assert_eq!(
        thought["content"]["excludes"],
        json!(["<mm:think>", "</mm:think>", NS]),
        "neither a think marker nor a call marker inside the thought"
    );

    for start in [GrammarStart::AfterReasoning, GrammarStart::FirstToken] {
        assert!(
            registry
                .structural_tag_for_grammar_start(parser, &tag(false), start)
                .unwrap()
                .is_none(),
            "a tag for a prompt outside the thought carries no prefix"
        );
    }
}

/// What the tag admits is what this parser reads: the thought and the block,
/// the bare close and the block, and the block alone, with every value kind
/// the elements spell.
#[tokio::test]
async fn m3_outputs_the_grammar_admits_parse_back() {
    let tools = constraint_tools();
    let block = tool_block(&[(
        "get_weather",
        format!(
            "{}{}{}{}{}{}{}",
            element("city", "Beijing"),
            element("days", "3"),
            element("units", "c"),
            element("verbose", "True"),
            element(
                "where",
                &format!("{}{}", element("x", "1.5"), element("label", "home"))
            ),
            element(
                "tags",
                &format!("{}{}", element("item", "a"), element("item", "b"))
            ),
            element("any", "free text"),
        ),
    )]);
    for before in ["<mm:think>Beijing it is.</mm:think>", "</mm:think>", ""] {
        let parser = MinimaxM3Parser::new();
        let (_, calls) = parser
            .parse_complete_with_tools(&format!("{before}{block}"), &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1, "{before:?}");
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args,
            json!({
                "city": "Beijing",
                "days": 3,
                "units": "c",
                "verbose": true,
                "where": {"x": 1.5, "label": "home"},
                "tags": ["a", "b"],
                "any": "free text",
            }),
            "{before:?}"
        );
    }
}

/// The first invoke's content of a tag.
fn first_invoke(tag: &serde_json::Value) -> serde_json::Value {
    tag["format"]["elements"][1]["tags"][0]["content"]["tags"][0]["content"].clone()
}

/// Free text up to the element's closing tag: what an untyped or recurring value is written as.
fn expected_free_text(key: &str) -> serde_json::Value {
    let close = format!("</{key}>");
    let mut excludes = vec![close.as_str()];
    excludes.extend(exits());
    json!({"type": "any_text", "excludes": excludes})
}

fn schema_tool(parameters: serde_json::Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "t".to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    }
}

/// A reference back into a schema being written is written as free text where it recurs, so
/// the tag stays bounded: the self-referencing root (a bare `#` is refused, as it would name
/// the fragment's own document), a mutually recursive pair of definitions, a filter whose lists
/// and negation hold filters, and a definition two sibling properties share, written twice.
#[test]
fn m3_references_that_recur_are_written_once_then_as_free_text() {
    let size = |tag: &serde_json::Value| serde_json::to_string(tag).unwrap().len();

    let root = schema_tool(json!({
        "type": "object",
        "properties": {"a": {"$ref": "#"}, "b": {"$ref": "#"}, "c": {"$ref": "#"}},
        "required": ["a", "b", "c"],
    }));
    let tag = MinimaxM3Parser::build_structural_tag(&[root], true);
    assert!(size(&tag) < 4096, "{} bytes", size(&tag));
    let elements = first_invoke(&tag)["elements"].clone();
    assert_eq!(elements.as_array().unwrap().len(), 3);
    for key in ["a", "b", "c"] {
        let element = elements
            .as_array()
            .unwrap()
            .iter()
            .find(|element| element["elements"][0]["value"] == format!("{NS}<{key}>"))
            .unwrap();
        assert_eq!(
            element["elements"][1],
            expected_free_text(key),
            "a bare `#` is refused"
        );
    }

    let pair = schema_tool(json!({
        "type": "object",
        "$defs": {
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}, "x": {"type": "string"}}, "required": ["b", "x"]},
            "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}, "y": {"type": "integer"}}, "required": ["a", "y"]},
        },
        "properties": {"start": {"$ref": "#/$defs/A"}},
        "required": ["start"],
    }));
    let tag = MinimaxM3Parser::build_structural_tag(&[pair], true);
    assert!(size(&tag) < 4096, "{} bytes", size(&tag));
    let start = first_invoke(&tag)["elements"][0].clone();
    let a_members = &start["elements"][1]["elements"];
    assert_eq!(a_members[1], expected_element("x", expected_text("x")));
    let b_members = &a_members[0]["elements"][1]["elements"];
    assert_eq!(
        b_members[0],
        expected_element("a", expected_free_text("a")),
        "A re-entered through B is free text"
    );
    assert_eq!(
        b_members[1],
        expected_element(
            "y",
            json!({"type": "json_schema", "json_schema": {"type": "integer"}})
        )
    );

    let filter = schema_tool(json!({
        "type": "object",
        "$defs": {"Filter": {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "and": {"type": "array", "items": {"$ref": "#/$defs/Filter"}},
                "or": {"type": "array", "items": {"$ref": "#/$defs/Filter"}},
                "not": {"$ref": "#/$defs/Filter"},
            },
        }},
        "properties": {"filter": {"$ref": "#/$defs/Filter"}},
        "required": ["filter"],
    }));
    let tag = MinimaxM3Parser::build_structural_tag(&[filter], true);
    assert!(size(&tag) < 8192, "{} bytes", size(&tag));
    let members = first_invoke(&tag)["elements"][0]["elements"][1]["elements"].clone();
    assert_eq!(
        members[1]["content"]["elements"][1],
        json!({"type": "star", "content": expected_element("item", expected_free_text("item"))}),
        "a list of filters inside a filter holds free-text items"
    );
    assert_eq!(
        members[3]["content"],
        expected_element("not", expected_free_text("not")),
        "a filter inside a filter is free text"
    );

    let twice = schema_tool(json!({
        "type": "object",
        "$defs": {"Point": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}},
        "properties": {"home": {"$ref": "#/$defs/Point"}, "work": {"$ref": "#/$defs/Point"}},
        "required": ["home", "work"],
    }));
    let tag = MinimaxM3Parser::build_structural_tag(&[twice], true);
    let point = json!({"type": "sequence", "elements": [
        expected_element("x", json!({"type": "json_schema", "json_schema": {"type": "number"}})),
    ]});
    let members = first_invoke(&tag)["elements"].clone();
    assert_eq!(members[0], expected_element("home", point.clone()));
    assert_eq!(
        members[1],
        expected_element("work", point),
        "a shared definition is no cycle"
    );
}

/// A number's schema that still holds a local reference travels with the root's definitions, so
/// the engine, which compiles the fragment alone, can resolve the pointer.
#[test]
fn m3_number_fragments_carry_the_definitions_they_reference() {
    let tool = schema_tool(json!({
        "type": "object",
        "$defs": {"Count": {"type": "integer", "minimum": 1}},
        "properties": {
            "n": {"type": "integer", "anyOf": [{"$ref": "#/$defs/Count"}, {"type": "integer", "maximum": 0}]},
            "m": {"$ref": "#/$defs/Count"},
        },
        "required": ["n", "m"],
    }));
    let tag = MinimaxM3Parser::build_structural_tag(&[tool], true);
    let members = first_invoke(&tag)["elements"].clone();
    assert_eq!(
        members[0],
        expected_element(
            "n",
            json!({"type": "json_schema", "json_schema": {
                "type": "integer",
                "anyOf": [{"$ref": "#/$defs/Count"}, {"type": "integer", "maximum": 0}],
                "$defs": {"Count": {"type": "integer", "minimum": 1}},
            }})
        )
    );
    assert_eq!(
        members[1],
        expected_element(
            "m",
            json!({"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}})
        ),
        "a reference followed to its target needs no definitions"
    );
}
