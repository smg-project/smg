//! DeepSeek-V4.1 DSML tool-call parser: spaced tags, reference parsing rules.
//!
//! Parity cases ported from vLLM `tests/parser/engine/test_deepseek_v41.py`
//! (parallel calls at chunk sizes 1/7/10000) and SGLang
//! `test_deepseekv41_detector.py` (nested JSON string, list values, JSON-body
//! invoke, chunk sizes 1..1000).

use openai_protocol::common::{Function, Tool};
use serde_json::json;
use tool_parser::{
    parsers::{DeepSeekDsmlParser, DsmlDialect},
    ParserFactory, ToolParser,
};

const TWO_CALLS: &str = "Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">杭州</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"count\" string=\"false\">42</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"x\" string=\"false\">1.5</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"y\" string=\"false\">2.25</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

const SGLANG_CALL: &str = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">{\"a\": 1}</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"flags\" string=\"false\">[1, true, null]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

/// Stream `text` in chunks of `chunk_chars` characters through a fresh V4.1
/// parser with no tool list; returns the content and per-tool argument bytes.
#[expect(clippy::unwrap_used, reason = "test helper — panics are intentional")]
async fn stream(text: &str, chunk_chars: usize) -> (String, Vec<(Option<String>, String)>) {
    let chars: Vec<char> = text.chars().collect();
    let mut parser = DeepSeekDsmlParser::v41();
    let mut content = String::new();
    let mut calls: Vec<(Option<String>, String)> = Vec::new();
    for chunk in chars.chunks(chunk_chars) {
        let piece: String = chunk.iter().collect();
        let result = parser.parse_incremental(&piece, &[]).await.unwrap();
        content.push_str(&result.normal_text);
        for item in result.calls {
            if calls.len() <= item.tool_index {
                calls.resize(item.tool_index + 1, (None, String::new()));
            }
            if item.name.is_some() {
                calls[item.tool_index].0 = item.name;
            }
            calls[item.tool_index].1.push_str(&item.parameters);
        }
    }
    (content, calls)
}

#[tokio::test]
async fn complete_parse_strips_the_separator_and_formats_json_python_style() {
    let (text, calls) = DeepSeekDsmlParser::v41()
        .parse_complete(TWO_CALLS)
        .await
        .unwrap();
    assert_eq!(text, "Checking.");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(
        calls[0].function.arguments,
        r#"{"city": "杭州", "count": 42}"#
    );
    assert_eq!(calls[1].function.arguments, r#"{"x": 1.5, "y": 2.25}"#);
}

#[tokio::test]
async fn only_the_exact_blank_line_separator_is_stripped() {
    let text = "Done. \n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let (content, calls) = DeepSeekDsmlParser::v41()
        .parse_complete(text)
        .await
        .unwrap();
    assert_eq!(content, "Done. ");
    assert_eq!(calls[0].function.arguments, "{}");
    let text = "Done.\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let (content, _) = DeepSeekDsmlParser::v41()
        .parse_complete(text)
        .await
        .unwrap();
    assert_eq!(content, "Done.\n");
}

#[tokio::test]
async fn sglang_nested_json_string_and_list_values() {
    let (_, calls) = DeepSeekDsmlParser::v41()
        .parse_complete(SGLANG_CALL)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(
        calls[0].function.arguments,
        r#"{"query": "{\"a\": 1}", "limit": 2, "flags": [1, true, null]}"#
    );
}

#[tokio::test]
async fn json_body_invoke_from_sglang_grammar_is_accepted() {
    let text = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">{\"a\": 1}</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let (content, calls) = DeepSeekDsmlParser::v41()
        .parse_complete(text)
        .await
        .unwrap();
    assert_eq!(content, "");
    assert_eq!(calls[0].function.name, "f");
    assert_eq!(calls[0].function.arguments, r#"{"a": 1}"#);
}

#[tokio::test]
async fn string_true_is_never_coerced_and_bad_false_json_stays_raw() {
    let text = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"n\" string=\"true\">42</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"bad\" string=\"false\">[broken</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let (_, calls) = DeepSeekDsmlParser::v41()
        .parse_complete(text)
        .await
        .unwrap();
    assert_eq!(
        calls[0].function.arguments,
        r#"{"n": "42", "bad": "[broken"}"#
    );
}

#[tokio::test]
async fn v4_unspaced_tags_are_plain_text_for_v41_and_vice_versa() {
    let v4 = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    let (text, calls) = DeepSeekDsmlParser::v41().parse_complete(v4).await.unwrap();
    assert!(calls.is_empty());
    assert_eq!(text, v4);
    let (text, calls) = DeepSeekDsmlParser::v4()
        .parse_complete(TWO_CALLS)
        .await
        .unwrap();
    assert!(calls.is_empty());
    assert_eq!(text, TWO_CALLS);
}

#[tokio::test]
async fn orphan_invoke_without_a_block_is_a_call_and_text_after_the_section_is_dropped() {
    let text = "Sure.\n\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"k\" string=\"true\">v</｜DSML｜ parameter>\n</｜DSML｜ invoke>\ntrailing text";
    let (content, calls) = DeepSeekDsmlParser::v41()
        .parse_complete(text)
        .await
        .unwrap();
    assert_eq!(content, "Sure.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, r#"{"k": "v"}"#);

    let (content, calls) = stream(text, 5).await;
    assert_eq!(content, "Sure.");
    assert_eq!(
        calls,
        vec![(Some("f".to_string()), r#"{"k": "v"}"#.to_string())]
    );
}

#[tokio::test]
async fn streaming_yields_identical_calls_at_every_chunk_size() {
    for chunk_chars in [1usize, 2, 3, 5, 7, 11, 13, 17, 23, 31, 47, 10_000] {
        let (content, calls) = stream(TWO_CALLS, chunk_chars).await;
        assert_eq!(content, "Checking.", "chunk {chunk_chars}");
        assert_eq!(
            calls,
            vec![
                (
                    Some("get_weather".to_string()),
                    r#"{"city": "杭州", "count": 42}"#.to_string()
                ),
                (
                    Some("get_weather".to_string()),
                    r#"{"x": 1.5, "y": 2.25}"#.to_string()
                ),
            ],
            "chunk {chunk_chars}"
        );
    }
}

#[tokio::test]
async fn streaming_sglang_case_at_many_chunk_sizes() {
    for chunk_chars in [1usize, 2, 3, 5, 11, 37, 1000] {
        let (content, calls) = stream(SGLANG_CALL, chunk_chars).await;
        assert_eq!(content, "", "chunk {chunk_chars}");
        assert_eq!(
            calls,
            vec![(
                Some("search".to_string()),
                r#"{"query": "{\"a\": 1}", "limit": 2, "flags": [1, true, null]}"#.to_string()
            )],
            "chunk {chunk_chars}"
        );
    }
}

#[tokio::test]
async fn streaming_flushes_plain_content_and_holds_only_a_possible_separator() {
    let mut parser = DeepSeekDsmlParser::v41();
    let first = parser
        .parse_incremental("Hello\nWorld\n", &[])
        .await
        .unwrap();
    assert_eq!(first.normal_text, "Hello\nWorld");
    let second = parser.parse_incremental("!", &[]).await.unwrap();
    assert_eq!(second.normal_text, "\n!");
    let third = parser
        .parse_incremental(" <｜DSML｜ ca", &[])
        .await
        .unwrap();
    assert_eq!(third.normal_text, " ");
    let fourth = parser.parse_incremental("lm", &[]).await.unwrap();
    assert_eq!(fourth.normal_text, "<｜DSML｜ calm");
    assert!(fourth.calls.is_empty());
}

#[tokio::test]
async fn streaming_forwards_unknown_tool_names() {
    let known = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "other".to_string(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: None,
        },
    }];
    let mut parser = DeepSeekDsmlParser::v41();
    let result = parser.parse_incremental(TWO_CALLS, &known).await.unwrap();
    let names: Vec<String> = result.calls.iter().filter_map(|c| c.name.clone()).collect();
    assert_eq!(names, vec!["get_weather", "get_weather"]);
}

#[tokio::test]
async fn streaming_keeps_leading_whitespace_in_string_values_and_never_slices_mid_char() {
    // A `string="true"` value that starts with a newline and continues with
    // multi-byte text: the partial value must not be trimmed, or the delta
    // offsets land inside a character once the value completes.
    let text = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"c\" string=\"true\">\n杭州市</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let (_, calls) = DeepSeekDsmlParser::v41()
        .parse_complete(text)
        .await
        .unwrap();
    assert_eq!(calls[0].function.arguments, "{\"c\": \"\\n杭州市\"}");
    for chunk_chars in [1usize, 2, 5, 1000] {
        let (content, streamed) = stream(text, chunk_chars).await;
        assert_eq!(content, "", "chunk {chunk_chars}");
        assert_eq!(
            streamed,
            vec![(Some("f".to_string()), "{\"c\": \"\\n杭州市\"}".to_string())],
            "chunk {chunk_chars}"
        );
    }
}

/// A `string="false"` number parses at every length, so a chunk boundary
/// inside it must not stream the shorter number as final: the first partial
/// snapshot is diffed against the empty object like every later one.
#[tokio::test]
async fn a_number_split_across_chunks_is_streamed_once() {
    let text = concat!(
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
        "<｜DSML｜ parameter name=\"n\" string=\"false\">123</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    );
    let split = text.find("123").unwrap() + 2;
    let mut parser = DeepSeekDsmlParser::v41();
    let mut args = String::new();
    for chunk in [&text[..split], &text[split..]] {
        for item in parser.parse_incremental(chunk, &[]).await.unwrap().calls {
            args.push_str(&item.parameters);
        }
    }
    assert_eq!(args, r#"{"n": 123}"#);
}

/// A turn truncated inside an invoke: the streamed prefix plus the
/// end-of-stream remainder from `get_unstreamed_tool_args` is the parser's
/// last snapshot, so the client's arguments close.
#[tokio::test]
async fn truncated_invoke_remainder_closes_the_streamed_arguments() {
    let mut parser = DeepSeekDsmlParser::v41();
    let mut args = String::new();
    for chunk in [
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
        "<｜DSML｜ parameter name=\"city\" string=\"true\">杭",
        "州",
    ] {
        for item in parser.parse_incremental(chunk, &[]).await.unwrap().calls {
            args.push_str(&item.parameters);
        }
    }
    for item in parser.get_unstreamed_tool_args().unwrap_or_default() {
        args.push_str(&item.parameters);
    }
    assert_eq!(args, r#"{"city": "杭州"}"#);
    assert_eq!(parser.take_unstreamed_normal_text(), "");
}

#[tokio::test]
async fn end_of_stream_flushes_held_back_content_but_not_tool_syntax() {
    let mut parser = DeepSeekDsmlParser::v41();
    let first = parser.parse_incremental("Hi\n", &[]).await.unwrap();
    assert_eq!(first.normal_text, "Hi");
    assert_eq!(parser.take_unstreamed_normal_text(), "\n");
    assert_eq!(parser.take_unstreamed_normal_text(), "");

    let mut parser = DeepSeekDsmlParser::v41();
    parser.parse_incremental(TWO_CALLS, &[]).await.unwrap();
    parser.parse_incremental("\ntrailing", &[]).await.unwrap();
    assert_eq!(parser.take_unstreamed_normal_text(), "");
}

#[tokio::test]
async fn reset_clears_the_tool_section_state() {
    let mut parser = DeepSeekDsmlParser::v41();
    parser.parse_incremental(TWO_CALLS, &[]).await.unwrap();
    parser.reset();
    let result = parser.parse_incremental("plain", &[]).await.unwrap();
    assert_eq!(result.normal_text, "plain");
    assert!(result.calls.is_empty());
}

#[test]
fn factory_maps_v41_names_ahead_of_v4_and_registers_a_structural_tag() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    assert!(registry.has_parser("deepseek_v41"));
    assert!(registry.has_structural_tag("deepseek_v41"));
    assert!(!registry.has_structural_tag("deepseek_v4"));
    for model in [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "deepseek-ai/DeepSeek-V41-Flash",
        "deepseek-v4.1-flash",
        "DeepSeek_V41",
        "deepseek-v41-flash",
    ] {
        let parser = registry.create_for_model(model).unwrap();
        let (_, calls) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(parser.parse_complete(TWO_CALLS))
            .unwrap();
        assert_eq!(calls.len(), 2, "{model}");
    }
    let v4 = registry
        .create_for_model("deepseek-ai/DeepSeek-V4-Flash")
        .unwrap();
    let (_, calls) = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(v4.parse_complete(TWO_CALLS))
        .unwrap();
    assert!(calls.is_empty(), "V4 must not parse spaced tags");
    assert_eq!(DeepSeekDsmlParser::v41().dialect(), DsmlDialect::V41);
    assert_eq!(DeepSeekDsmlParser::v4().dialect(), DsmlDialect::V4);
}

#[test]
fn structural_tag_mirrors_vllm_forced_grammar() {
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            strict: None,
        },
    }];
    let tag = DeepSeekDsmlParser::build_v41_structural_tag(&tools, true);
    let format = &tag["format"];
    assert_eq!(format["type"], "sequence");
    assert_eq!(format["elements"][0]["value"], "\n\n<｜DSML｜ calls>\n");
    assert_eq!(format["elements"][2]["value"], "</｜DSML｜ calls>");
    let calls = &format["elements"][1];
    assert_eq!(calls["type"], "tags_with_separator");
    assert_eq!(calls["at_least_one"], true);
    assert_eq!(
        calls["tags"][0]["begin"],
        "<｜DSML｜ invoke name=\"get_weather\">\n"
    );
    assert_eq!(calls["tags"][0]["end"], "</｜DSML｜ invoke>\n");
    let parameter = &calls["tags"][0]["content"]["content"];
    assert_eq!(parameter["begin"], "<｜DSML｜ parameter name=\"");
    assert_eq!(parameter["end"], "</｜DSML｜ parameter>\n");
    let branches = &parameter["content"]["elements"][2]["elements"];
    assert_eq!(branches[0]["elements"][0]["value"], "true\">");
    assert_eq!(
        branches[0]["elements"][1]["excludes"],
        json!([
            "</｜DSML｜ parameter>",
            "</｜DSML｜ invoke>",
            "</｜DSML｜ calls>"
        ])
    );
    assert_eq!(branches[1]["elements"][0]["value"], "false\">");
    assert_eq!(branches[1]["elements"][1]["type"], "json_schema");
    let constraint = factory_constraint(&tools);
    assert!(constraint.contains("<｜DSML｜ invoke name=\\\"get_weather\\\">"));
}

#[expect(clippy::unwrap_used, reason = "test helper — panics are intentional")]
fn factory_constraint(tools: &[Tool]) -> String {
    use openai_protocol::common::{ToolChoice, ToolChoiceValue};
    let factory = ParserFactory::new();
    let constraint = factory
        .registry()
        .generate_tool_constraint(
            Some("deepseek_v41"),
            tools,
            &ToolChoice::Value(ToolChoiceValue::Required),
        )
        .unwrap()
        .unwrap();
    constraint.to_tuple().1
}
