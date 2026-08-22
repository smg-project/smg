//! Undeclared tool names must not reach the client.
//!
//! Every parser that validates names while streaming rejects a call naming a
//! function the request never declared: `parse_incremental` receives the tool
//! list, and since the streaming-flush fix the text surfaces as content
//! instead. The non-streaming path had no such guard - `parse_complete` is not
//! given the tool list at all - so identical model output produced a
//! `tool_calls` entry naming an undeclared function when `stream=false` and
//! plain content when `stream=true`.
//!
//! That is not hypothetical: asked "What is 2+2?" with tools declared,
//! Llama-3.2-1B emits a tool call literally named `2+2`, which a client
//! dispatching on `function.name` would be handed despite it never appearing
//! in its schema.
mod common;

use common::create_test_tools;
use tool_parser::{
    parsers::QwenXmlParser, CohereParser, Glm4MoeParser, JsonParser, LlamaParser, MinimaxM2Parser,
    MistralParser, QwenParser, ToolParser,
};

/// `(label, parser, text naming an UNDECLARED tool, text naming a DECLARED one)`
type Case = (
    &'static str,
    fn() -> Box<dyn ToolParser>,
    &'static str,
    &'static str,
);

fn llama() -> Box<dyn ToolParser> {
    Box::new(LlamaParser::new())
}
fn json() -> Box<dyn ToolParser> {
    Box::new(JsonParser::new())
}
fn mistral() -> Box<dyn ToolParser> {
    Box::new(MistralParser::new())
}
fn qwen() -> Box<dyn ToolParser> {
    Box::new(QwenParser::new())
}
fn cohere() -> Box<dyn ToolParser> {
    Box::new(CohereParser::new())
}

const CASES: &[Case] = &[
    (
        "llama",
        llama,
        r#"{"name": "bogus_tool", "parameters": {"x": 1}}"#,
        r#"{"name": "get_weather", "parameters": {"city": "Tokyo"}}"#,
    ),
    (
        "json",
        json,
        r#"{"name": "bogus_tool", "arguments": {"x": 1}}"#,
        r#"{"name": "get_weather", "arguments": {"city": "Tokyo"}}"#,
    ),
    (
        "mistral",
        mistral,
        r#"[TOOL_CALLS] [{"name": "bogus_tool", "arguments": {"x": 1}}]"#,
        r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#,
    ),
    (
        "qwen",
        qwen,
        "<tool_call>\n{\"name\": \"bogus_tool\", \"arguments\": {\"x\": 1}}\n</tool_call>",
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>",
    ),
    (
        "cohere",
        cohere,
        r#"<|START_ACTION|>{"tool_name": "bogus_tool", "parameters": {"x": 1}}<|END_ACTION|>"#,
        r#"<|START_ACTION|>{"tool_name": "search", "parameters": {"query": "x"}}<|END_ACTION|>"#,
    ),
];

#[tokio::test]
async fn undeclared_name_never_becomes_a_tool_call() {
    let tools = create_test_tools();
    for (label, make, undeclared_text, _) in CASES {
        let parser = make();
        let (normal_text, calls) = parser
            .parse_complete_with_tools(undeclared_text, &tools)
            .await
            .expect("parse must not error");

        assert!(
            calls.is_empty(),
            "[{label}] undeclared tool name must not be forwarded, got {:?}",
            calls.iter().map(|c| &c.function.name).collect::<Vec<_>>()
        );
        // And it must not vanish either: the text is content, as streaming does.
        assert!(
            !normal_text.is_empty(),
            "[{label}] the undeclared call's text must surface as content"
        );
    }
}

#[tokio::test]
async fn declared_names_are_untouched() {
    let tools = create_test_tools();
    for (label, make, _, declared_text) in CASES {
        let parser = make();
        let (_, calls) = parser
            .parse_complete_with_tools(declared_text, &tools)
            .await
            .expect("parse must not error");

        assert_eq!(calls.len(), 1, "[{label}] declared call must survive");
        let name = &calls[0].function.name;
        assert!(
            tools.iter().any(|t| &t.function.name == name),
            "[{label}] surviving call must name a declared tool, got {name:?}"
        );
    }
}

#[tokio::test]
async fn the_llama_2_plus_2_regression() {
    // Verbatim shape of the CI failure this fix addresses.
    let tools = create_test_tools();
    let parser = LlamaParser::new();
    let (normal_text, calls) = parser
        .parse_complete_with_tools(r#"{"name": "2+2", "parameters": {}}"#, &tools)
        .await
        .unwrap();

    assert!(calls.is_empty(), "hallucinated tool name must be rejected");
    assert!(
        normal_text.contains("2+2"),
        "the text must still reach the client"
    );
}

#[tokio::test]
async fn a_declared_call_survives_alongside_an_undeclared_one() {
    let tools = create_test_tools();
    let parser = MistralParser::new();
    let (normal_text, calls) = parser
        .parse_complete_with_tools(
            r#"[TOOL_CALLS] [{"name": "bogus_tool", "arguments": {}}, {"name": "get_weather", "arguments": {"city": "Paris"}}]"#,
            &tools,
        )
        .await
        .unwrap();

    let names: Vec<&str> = calls.iter().map(|c| c.function.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["get_weather"],
        "only the declared call may survive"
    );
    // ...and the dropped one must not vanish. Streaming emits an undeclared
    // call's text as content; losing it here would be the swallow bug again,
    // just narrower.
    assert!(
        normal_text.contains("bogus_tool"),
        "the dropped call's content must survive, got {normal_text:?}"
    );
}

#[tokio::test]
async fn both_paths_agree_on_a_mixed_batch() {
    // The contract that matters: neither path forwards an undeclared name,
    // and neither path loses it silently.
    let tools = create_test_tools();
    let input = r#"[TOOL_CALLS] [{"name": "bogus_tool", "arguments": {"x": 1}}, {"name": "get_weather", "arguments": {"city": "Paris"}}]"#;

    let complete = MistralParser::new();
    let (batch_text, batch_calls) = complete
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();

    let mut stream = MistralParser::new();
    let result = stream.parse_incremental(input, &tools).await.unwrap();
    let mut streamed_text = result.normal_text.clone();
    let streamed_names: Vec<String> = result.calls.into_iter().filter_map(|c| c.name).collect();
    streamed_text.push_str(&stream.take_unstreamed_normal_text());

    let batch_names: Vec<&str> = batch_calls
        .iter()
        .map(|c| c.function.name.as_str())
        .collect();
    assert_eq!(batch_names, vec!["get_weather"], "non-streaming");
    assert_eq!(streamed_names, vec!["get_weather".to_string()], "streaming");
    assert!(
        batch_text.contains("bogus_tool"),
        "non-streaming must preserve the dropped call, got {batch_text:?}"
    );
    assert!(
        streamed_text.contains("bogus_tool"),
        "streaming must preserve the dropped call, got {streamed_text:?}"
    );
}

#[tokio::test]
async fn no_declared_tools_means_no_filtering() {
    // The parse endpoint and the Go FFI both allow an empty tool list; with no
    // declared set there is nothing to validate against, so nothing is dropped.
    let parser = LlamaParser::new();
    let (_, calls) = parser
        .parse_complete_with_tools(r#"{"name": "bogus_tool", "parameters": {}}"#, &[])
        .await
        .unwrap();

    assert_eq!(
        calls.len(),
        1,
        "an empty tool list must not filter anything"
    );
}

#[tokio::test]
async fn parsers_that_deliberately_forward_unknown_names_are_unaffected() {
    // MiniMax-M2 overrides parse_complete_with_tools and forwards unknown
    // names on purpose, so it does not leak <invoke> markup into assistant
    // text. Overriding is how a parser opts out of the default's validation.
    let tools = create_test_tools();
    let parser = MinimaxM2Parser::new();
    let (_, calls) = parser
        .parse_complete_with_tools(
            "<minimax:tool_call>\n<invoke name=\"bogus_tool\">\n<parameter name=\"x\">1</parameter>\n</invoke>\n</minimax:tool_call>",
            &tools,
        )
        .await
        .unwrap();

    let names: Vec<&str> = calls.iter().map(|c| c.function.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["bogus_tool"],
        "deliberate forwarding must be preserved"
    );
}

#[tokio::test]
async fn overriding_parsers_that_validate_while_streaming_validate_here_too() {
    // QwenXmlParser and Glm4MoeParser override parse_complete_with_tools for
    // schema-aware coercion, which bypasses the default's check. Both reject
    // undeclared names in parse_incremental, so they must reject them here as
    // well or they keep the exact asymmetry this fix removes.
    let tools = create_test_tools();

    let qwen_xml = QwenXmlParser::new();
    let (text, calls) = qwen_xml
        .parse_complete_with_tools(
            "<tool_call>\n<function=bogus_tool>\n<parameter=city>Beijing</parameter>\n</function>\n</tool_call>",
            &tools,
        )
        .await
        .unwrap();
    assert!(
        calls.is_empty(),
        "[qwen_xml] undeclared name must not be forwarded, got {:?}",
        calls.iter().map(|c| &c.function.name).collect::<Vec<_>>()
    );
    assert!(!text.is_empty(), "[qwen_xml] the text must survive");

    let glm4 = Glm4MoeParser::glm45();
    let (text, calls) = glm4
        .parse_complete_with_tools(
            "<tool_call>bogus_tool\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>",
            &tools,
        )
        .await
        .unwrap();
    assert!(
        calls.is_empty(),
        "[glm4_moe] undeclared name must not be forwarded, got {:?}",
        calls.iter().map(|c| &c.function.name).collect::<Vec<_>>()
    );
    assert!(!text.is_empty(), "[glm4_moe] the text must survive");
}

#[tokio::test]
async fn overriding_parsers_still_accept_declared_names() {
    let tools = create_test_tools();

    let qwen_xml = QwenXmlParser::new();
    let (_, calls) = qwen_xml
        .parse_complete_with_tools(
            "<tool_call>\n<function=get_weather>\n<parameter=city>Beijing</parameter>\n</function>\n</tool_call>",
            &tools,
        )
        .await
        .unwrap();
    assert_eq!(calls.len(), 1, "[qwen_xml] declared call must survive");
    assert_eq!(calls[0].function.name, "get_weather");

    let glm4 = Glm4MoeParser::glm45();
    let (_, calls) = glm4
        .parse_complete_with_tools(
            "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>",
            &tools,
        )
        .await
        .unwrap();
    assert_eq!(calls.len(), 1, "[glm4_moe] declared call must survive");
    assert_eq!(calls[0].function.name, "get_weather");
}
