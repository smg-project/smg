//! DeepSeek DSML Parser Integration Tests (V3.2 + V4)
mod common;

use common::create_test_tools;
use tool_parser::{DeepSeekDsmlParser, ToolParser};

#[tokio::test]
async fn test_deepseek32_complete_single_tool() {
    let parser = DeepSeekDsmlParser::v32();

    let input = concat!(
        "Let me check that.\n\n",
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"units\" string=\"true\">celsius</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "Let me check that.");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_deepseek32_complete_multiple_tools() {
    let parser = DeepSeekDsmlParser::v32();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">rust programming</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"translate\">\n",
        "<｜DSML｜parameter name=\"text\" string=\"true\">Hello World</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"to\" string=\"true\">ja</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_deepseek32_complete_direct_json() {
    let parser = DeepSeekDsmlParser::v32();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "{\"location\": \"Beijing\", \"date\": \"2024-01-16\"}\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Beijing");
    assert_eq!(args["date"], "2024-01-16");
}

#[tokio::test]
async fn test_deepseek32_complete_mixed_types() {
    let parser = DeepSeekDsmlParser::v32();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"process\">\n",
        "<｜DSML｜parameter name=\"text\" string=\"true\">hello</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"count\" string=\"false\">42</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"enabled\" string=\"false\">true</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "hello");
    assert_eq!(args["count"], 42);
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_deepseek32_complete_nested_json_param() {
    let parser = DeepSeekDsmlParser::v32();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"process\">\n",
        "<｜DSML｜parameter name=\"data\" string=\"false\">{\"nested\": [1, 2, 3]}</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"].is_array());
}

#[tokio::test]
async fn test_deepseek32_complete_malformed_skips() {
    let parser = DeepSeekDsmlParser::v32();

    let input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "not valid at all\n",
        "</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"translate\">\n",
        "<｜DSML｜parameter name=\"text\" string=\"true\">hello</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"to\" string=\"true\">ja</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert!(!tools.is_empty());
    assert!(tools.iter().any(|t| t.function.name == "translate"));
}

#[test]
fn test_deepseek32_format_detection() {
    let parser = DeepSeekDsmlParser::v32();

    assert!(parser.has_tool_markers("<｜DSML｜function_calls>"));
    assert!(parser.has_tool_markers("text with <｜DSML｜function_calls> marker"));

    assert!(!parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_deepseek32_no_tool_calls() {
    let parser = DeepSeekDsmlParser::v32();

    let input = "Just a normal response.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, input);
    assert!(tools.is_empty());
}

#[tokio::test]
async fn test_deepseek32_streaming_single_tool() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v32();

    let chunks = vec![
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">",
        "Beijing",
        "</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"units\" string=\"true\">",
        "celsius",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    ];

    let mut found_name = false;
    let mut collected_args = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(!collected_args.is_empty(), "Should have streamed arguments");
}

#[tokio::test]
async fn test_deepseek32_streaming_multiple_tools() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v32();

    let chunks = vec![
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">rust</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    ];

    let mut tool_names: Vec<String> = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
        }
    }

    assert_eq!(tool_names, vec!["search", "get_weather"]);
}

#[tokio::test]
async fn test_deepseek32_streaming_text_before_tools() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v32();

    let chunks = vec![
        "Here is ",
        "the result\n\n",
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">test</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    ];

    let mut normal_text = String::new();
    let mut found_tool = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal_text.push_str(&result.normal_text);
        for call in result.calls {
            if call.name.is_some() {
                found_tool = true;
            }
        }
    }

    assert!(normal_text.contains("Here is the result"));
    assert!(found_tool);
}

#[tokio::test]
async fn test_deepseek32_streaming_end_tokens_stripped() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v32();

    let result = parser
        .parse_incremental("</｜DSML｜function_calls>", &tools)
        .await
        .unwrap();
    assert!(!result.normal_text.contains("</｜DSML｜function_calls>"));
}

use tool_parser::ParserFactory;

#[tokio::test]
async fn test_deepseek32_factory_registration() {
    let factory = ParserFactory::new();

    assert!(factory.has_parser("deepseek32"));

    // V3.2 DSML models resolve to deepseek32 parser
    let dsml_input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">test</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );
    for model in ["deepseek-v3.2", "deepseek-ai/DeepSeek-V3.2"] {
        let parser = factory
            .registry()
            .create_for_model(model)
            .expect("parser should exist");
        let (_text, calls) = parser.parse_complete(dsml_input).await.unwrap();
        assert_eq!(calls.len(), 1, "model {model} should parse DSML format");
        assert_eq!(calls[0].function.name, "search");
    }

    // V3.2-Exp resolves to deepseek31 parser (V3.1 format)
    let v31_input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        r#"{"query": "test"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );
    for model in ["deepseek-v3.2-exp", "deepseek-ai/DeepSeek-V3.2-Exp"] {
        let parser = factory
            .registry()
            .create_for_model(model)
            .expect("parser should exist");
        let (_text, calls) = parser.parse_complete(v31_input).await.unwrap();
        assert_eq!(calls.len(), 1, "model {model} should parse V3.1 format");
        assert_eq!(calls[0].function.name, "search");
    }

    // Existing V3 and V3.1 mappings still work
    assert!(factory.registry().has_parser_for_model("deepseek-v3"));
    assert!(factory.registry().has_parser_for_model("deepseek-v3.1"));
}

// ---------------------------------------------------------------------------
// DeepSeek V4 coverage
//
// V4 shares the entire DSML invoke/parameter grammar with V3.2; the only
// structural difference is the outer block-name token (`tool_calls` instead
// of `function_calls`). These tests lock in that variance and verify that
// each constructor's parser rejects the other variant's block.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_deepseek_v4_complete_single_tool() {
    let parser = DeepSeekDsmlParser::v4();

    let input = concat!(
        "Let me check that.\n\n",
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>",
    );

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "Let me check that.");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
}

#[tokio::test]
async fn test_deepseek_v4_complete_mixed_types() {
    let parser = DeepSeekDsmlParser::v4();

    let input = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"process\">\n",
        "<｜DSML｜parameter name=\"text\" string=\"true\">hello</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"count\" string=\"false\">42</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"enabled\" string=\"false\">true</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "hello");
    assert_eq!(args["count"], 42);
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_deepseek_v4_complete_with_tools_omits_invalid_calls() {
    let parser = DeepSeekDsmlParser::v4();
    let tools = create_test_tools();
    let input = concat!(
        "Visible answer.\n\n",
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"not_supplied\">{\"value\":1}</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"get_weather\">{not-json}</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>",
        "Ordinary trailing answer.",
    );

    let (normal_text, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(normal_text, "Visible answer.Ordinary trailing answer.");
    assert!(calls.is_empty());
}

#[test]
fn test_deepseek_v4_format_detection() {
    let parser = DeepSeekDsmlParser::v4();

    assert!(parser.has_tool_markers("<｜DSML｜tool_calls>"));
    assert!(parser.has_tool_markers("text <｜DSML｜tool_calls> marker"));

    // Any DSML sentinel is structural and must not leak as ordinary content,
    // even when the block name is malformed for the selected model family.
    assert!(parser.has_tool_markers("<｜DSML｜function_calls>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[test]
fn test_deepseek_v32_does_not_match_v4_block() {
    // A cross-variant sentinel is structural even though it cannot form a call.
    let parser = DeepSeekDsmlParser::v32();
    assert!(parser.has_tool_markers("<｜DSML｜tool_calls>"));
}

#[tokio::test]
async fn test_deepseek_v4_cross_variant_payload_is_not_a_call() {
    // A V4 parser given a V3.2-shaped payload must not parse calls or expose
    // the foreign DSML framing as normal content.
    let parser = DeepSeekDsmlParser::v4();
    let v32_input = concat!(
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">test</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    );
    let (normal_text, tools) = parser.parse_complete(v32_input).await.unwrap();
    assert!(tools.is_empty(), "V4 parser must not parse V3.2 block");
    assert!(normal_text.is_empty());
}

#[tokio::test]
async fn test_deepseek_v4_streaming_single_tool() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v4();

    let chunks = vec![
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">",
        "Beijing",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>",
    ];

    let mut found_name = false;
    let mut collected_args = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(!collected_args.is_empty(), "Should have streamed arguments");
    assert_eq!(parser.completed_tool_call_count(), Some(1));
}

#[tokio::test]
async fn test_deepseek_v4_factory_registration() {
    let factory = ParserFactory::new();

    assert!(factory.has_parser("deepseek_v4"));

    let dsml_input = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">test</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>",
    );
    for model in ["deepseek-v4", "deepseek-ai/DeepSeek-V4-Flash"] {
        let parser = factory
            .registry()
            .create_for_model(model)
            .expect("parser should exist");
        let (_text, calls) = parser.parse_complete(dsml_input).await.unwrap();
        assert_eq!(calls.len(), 1, "model {model} should parse V4 DSML");
        assert_eq!(calls[0].function.name, "search");
    }
}

// ---------------------------------------------------------------------------
// Robustness regressions
// ---------------------------------------------------------------------------

/// When the engine emits `<｜end▁of▁sentence｜>` mid-parameter (e.g. a turn
/// cut off at max_tokens), the EOS marker must not leak into streamed
/// argument bytes. Previously only `</｜DSML｜parameter>` prefixes were
/// stripped from partial values, so EOS bled through as raw arg bytes.
#[tokio::test]
async fn test_deepseek_dsml_streaming_strips_eos_from_partial_parameter() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v32();

    let chunks = [
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo",
        // Engine truncated mid-parameter and emitted EOS as raw text.
        "<｜end▁of▁sentence｜>",
    ];

    let mut collected_args = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert!(
        !collected_args.contains("<｜end▁of▁sentence｜>"),
        "EOS must not leak into streamed argument bytes, got: {collected_args:?}"
    );
}

/// Same test against the V4 variant (different outer block name, same
/// parameter-level behaviour). Locks in that Fix 1 applies to both.
#[tokio::test]
async fn test_deepseek_dsml_v4_streaming_strips_eos_from_partial_parameter() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v4();

    let chunks = [
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing",
        "<｜end▁of▁sentence｜>",
    ];

    let mut collected_args = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert!(
        !collected_args.contains("<｜end▁of▁sentence｜>"),
        "EOS must not leak into V4 streamed argument bytes, got: {collected_args:?}"
    );
    assert_eq!(parser.completed_tool_call_count(), Some(0));
}

#[tokio::test]
async fn test_deepseek_dsml_v4_terminal_invalid_call_is_not_complete() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v4();
    let result = parser
        .parse_incremental(
            concat!(
                "Visible answer.\n\n",
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"get_weather\">{not-json}</｜DSML｜invoke>",
                "</｜DSML｜tool_calls>",
                "Ordinary trailing answer.",
            ),
            &tools,
        )
        .await
        .unwrap();

    assert_eq!(
        result.normal_text,
        "Visible answer.Ordinary trailing answer."
    );
    assert!(result.calls.is_empty());
    assert_eq!(parser.completed_tool_call_count(), Some(0));
}

#[tokio::test]
async fn test_deepseek_dsml_v4_terminal_partial_preserves_only_prefix_text() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v4();
    let result = parser
        .parse_incremental(
            concat!(
                "Visible answer.\n\n",
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"get_weather\">",
                "<｜DSML｜parameter name=\"city\" string=\"true\">Ber",
            ),
            &tools,
        )
        .await
        .unwrap();
    assert_eq!(result.normal_text, "Visible answer.");
    assert_eq!(parser.completed_tool_call_count(), Some(0));

    let terminal = parser.finish_incremental();
    assert!(terminal.normal_text.is_empty());
    assert!(terminal.calls.is_empty());
}

/// A malformed complete invoke with `name=""` must not stall the buffer.
/// Previously the streaming `invoke_regex` required `[^"]+` so `name=""`
/// never matched, leaving the bad block stuck at the head of the buffer —
/// `has_dsml` then stayed true forever and every subsequent delta was
/// suppressed. After the fix, the regex allows empty names and the
/// invalid-name guard advances the buffer past the bad invoke, so the
/// next valid invoke is emitted normally.
#[tokio::test]
async fn test_deepseek_dsml_streaming_malformed_empty_name_does_not_trap_buffer() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v32();

    let chunks = [
        "<｜DSML｜function_calls>\n",
        // Malformed complete invoke (name=""). Must be advanced past, not stuck.
        "<｜DSML｜invoke name=\"\">junk</｜DSML｜invoke>\n",
        // Valid invoke after it — parser must still emit this one.
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">foo</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    ];

    let mut tool_names: Vec<String> = Vec::new();
    let mut collected_args = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert_eq!(
        tool_names,
        vec!["search"],
        "valid invoke after malformed name=\"\" must still be emitted"
    );
    assert!(
        collected_args.contains("foo"),
        "argument bytes from the valid invoke must be emitted, got: {collected_args:?}"
    );
}

/// Regression for Catherine's e2e failure (PR #1030 comment 4314866618).
///
/// Real DeepSeek-V4 streams deliver chunks at BPE-token granularity: the
/// `<｜DSML｜` sentinel is a single token (id 128793), and the surrounding
/// text (`tool_calls`, `invoke`, parameter names, values) arrives as tiny
/// sub-word pieces. The exact chunk sequence below was captured from a live
/// `tool_choice=auto` completion.
///
/// Pre-fix bug: `has_dsml` only fired on a complete outer opener
/// (`<｜DSML｜tool_calls>`) or `<｜DSML｜invoke` substring. When chunk 2
/// arrived and the buffer held only `\n\n<｜DSML｜`, neither matched, and
/// `has_partial_prefix` missed `<｜DSML｜` (it only tracked `<`, `<｜`, `</`,
/// `</｜`). The passthrough branch fired, `std::mem::take` flushed the
/// buffer, and the sentinel was lost. Every subsequent chunk was then
/// treated as plain text and the tool call never emitted.
#[tokio::test]
async fn test_deepseek_dsml_v4_streaming_bpe_chunked_opener() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v4();

    // BPE-sized chunks modelled on live backend output. The `<｜DSML｜`
    // sentinel is its own atomic token; the surrounding characters arrive as
    // small sub-word pieces. The exact sub-word split is not load-bearing —
    // what matters is that the outer opener is fragmented across chunks.
    let chunks = [
        "\n\n",
        "<｜DSML｜",
        "tool",
        "_c",
        "all",
        "s",
        ">\n",
        "<｜DSML｜",
        "inv",
        "oke",
        " name",
        "=\"",
        "get",
        "_",
        "weather",
        "\">\n",
        "<｜DSML｜",
        "parameter",
        " name",
        "=\"",
        "location",
        "\"",
        " string",
        "=\"",
        "true",
        "\">",
        "Tokyo",
        "</｜DSML｜",
        "parameter",
        ">\n",
        "</｜DSML｜",
        "inv",
        "oke",
        ">\n",
        "</｜DSML｜",
        "tool",
        "_c",
        "all",
        "s",
        ">",
    ];

    let mut tool_names: Vec<String> = Vec::new();
    let mut collected_args = String::new();
    let mut normal_text = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal_text.push_str(&result.normal_text);
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert_eq!(
        tool_names,
        vec!["get_weather"],
        "BPE-chunked DSML opener must be recognized, not passed through as text"
    );
    assert!(
        collected_args.contains("Tokyo"),
        "argument bytes must be emitted, got: {collected_args:?}"
    );
    assert!(
        normal_text.is_empty(),
        "DSML framing and its leading whitespace must not leak into normal_text, got: {normal_text:?}"
    );
}

/// Same BPE-chunking scenario for V3.2 (block name `function_calls`).
/// The bug is identical across variants — the `has_dsml` check is the same
/// code path for both.
#[tokio::test]
async fn test_deepseek_dsml_v32_streaming_bpe_chunked_opener() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v32();

    let chunks = [
        "<｜DSML｜",
        "function",
        "_c",
        "all",
        "s",
        ">\n",
        "<｜DSML｜",
        "invoke",
        " name=\"",
        "get_weather",
        "\">\n",
        "<｜DSML｜",
        "parameter",
        " name=\"location\" string=\"true\">",
        "Beijing",
        "</｜DSML｜",
        "parameter>\n",
        "</｜DSML｜",
        "invoke>\n",
        "</｜DSML｜",
        "function_calls>",
    ];

    let mut tool_names: Vec<String> = Vec::new();
    let mut collected_args = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert_eq!(tool_names, vec!["get_weather"]);
    assert!(collected_args.contains("Beijing"));
}

/// Same scenario against V4 — proves the fix applies to both block names.
#[tokio::test]
async fn test_deepseek_dsml_v4_streaming_malformed_empty_name_does_not_trap_buffer() {
    let tools = create_test_tools();
    let mut parser = DeepSeekDsmlParser::v4();

    let chunks = [
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"\">junk</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"search\">\n",
        "<｜DSML｜parameter name=\"query\" string=\"true\">bar</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>",
    ];

    let mut tool_names: Vec<String> = Vec::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
        }
    }

    assert_eq!(tool_names, vec!["search"]);
}
