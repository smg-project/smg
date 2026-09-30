//! The response-template tool parser, with the `transformers serve` templates
//! for qwen3_5 (XML arguments) and qwen2 (JSON calls).

#![expect(clippy::unwrap_used, reason = "test helpers")]

mod common;

use common::create_test_tools;
use serde_json::{json, Value};
use smg_response_template::{
    adapter::ResponseParserState, load_response_template, ResponseTemplate,
};
use tool_parser::{traits::ToolParser, TemplateToolParser};

fn qwen3_5() -> ResponseTemplate {
    load_response_template(&json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
            "tool_calls": {
                "open_pattern": "\\s*<tool_call>\\s*<function=(?P<name>[^>\\n]+)>",
                "close_pattern": "</function>\\s*</tool_call>",
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": "<parameter=(?P<key>[^>\\n]+)>\\s*(?P<value>.*?)\\s*</parameter>"
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
            },
            "content": {"close_pattern": "\\s*(?:<\\|im_end\\|>|<\\|endoftext\\|>)", "content": "text"}
        }
    }))
    .unwrap()
}

fn qwen2() -> ResponseTemplate {
    load_response_template(&json!({
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "tool_calls": {
                "open_pattern": "\\s*<tool_call>", "close": "</tool_call>", "repeats": true,
                "content": "json", "transform": {"type": "function", "function": "{content}"}
            },
            "content": {"close_pattern": "\\s*(?:<\\|im_end\\|>|<\\|endoftext\\|>|<\\|eot_id\\|>)"}
        }
    }))
    .unwrap()
}

const CALL: &str = "\n<tool_call>\n<function=calculate>\n<parameter=x>\n2\n</parameter>\n<parameter=y>\n1.5\n</parameter>\n</function>\n</tool_call>";

fn chunks(text: &str, size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(size).map(|c| c.iter().collect()).collect()
}

/// Normal text and `(index, name, arguments)` of each call, over a stream
/// ended as the gateway ends it.
async fn stream(
    parser: &mut TemplateToolParser,
    text: &str,
) -> (String, Vec<(usize, String, Value)>) {
    let tools = create_test_tools();
    let (mut normal, mut items) = (String::new(), Vec::new());
    for chunk in chunks(text, 3) {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        normal.push_str(&result.normal_text);
        items.extend(result.calls);
    }
    normal.push_str(&parser.take_unstreamed_normal_text());
    items.extend(parser.get_unstreamed_tool_args().unwrap_or_default());
    let calls = items
        .into_iter()
        .map(|item| {
            let arguments = serde_json::from_str(&item.parameters).unwrap();
            (item.tool_index, item.name.unwrap(), arguments)
        })
        .collect();
    (normal, calls)
}

#[tokio::test]
async fn without_a_reasoning_parser_it_feeds_the_parser_state_and_keeps_reasoning_as_content() {
    let template = qwen3_5();
    let mut parser = TemplateToolParser::new(template.clone());
    let tools: Vec<Value> = create_test_tools()
        .iter()
        .map(|tool| serde_json::to_value(tool).unwrap())
        .collect();
    parser.attach_response_parser_state(ResponseParserState::new(
        &template,
        "<think>\n",
        &tools,
        false,
    ));
    let output = format!("plan\n</think>\n\nSure.{CALL}");
    let (normal, calls) = stream(&mut parser, &output).await;
    assert_eq!(normal, "plan\n\n\nSure.");
    // The close pattern can grow at the edge of the output, so the call only
    // closes when the output ends; arguments keep the declared types.
    let arguments = json!({"x": 2, "y": 1.5});
    assert_eq!(calls, [(0, "calculate".to_owned(), arguments)]);
}

#[tokio::test]
async fn with_a_reasoning_parser_it_takes_the_calls_that_parser_closed() {
    let template = qwen3_5();
    let state = ResponseParserState::new(&template, "", &[], false);
    let mut parser = TemplateToolParser::new(template);
    parser.attach_response_parser_state(state.clone());
    let tools = create_test_tools();

    let output = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>\n<think>again";
    let (reasoning, content) = state.reasoning(Some(output)).unwrap();
    assert_eq!((reasoning.as_str(), content.as_str()), ("again", "\n"));
    // Reasoning goes on, but a call waits: the tool parser must run now.
    assert!(!state.in_reasoning());
    let result = parser.parse_incremental(&content, &tools).await.unwrap();
    assert_eq!(result.normal_text, "\n");
    assert_eq!(result.calls.len(), 1);
    assert_eq!(result.calls[0].name.as_deref(), Some("get_time"));
    assert_eq!(result.calls[0].parameters, "{}");
    assert!(state.in_reasoning());
}

#[tokio::test]
async fn a_complete_output_casts_declared_arguments_and_keeps_undeclared_calls() {
    let parser = TemplateToolParser::new(qwen3_5());
    let output = format!(
        "Sure.{CALL}\n<tool_call>\n<function=unknown>\n</function>\n</tool_call><|im_end|>"
    );
    let (normal, calls) = parser
        .parse_complete_with_tools(&output, &create_test_tools())
        .await
        .unwrap();
    assert_eq!(normal, "Sure.");
    let calls: Vec<(&str, &str)> = calls
        .iter()
        .map(|c| (c.function.name.as_str(), c.function.arguments.as_str()))
        .collect();
    assert_eq!(
        calls,
        [("calculate", r#"{"x": 2, "y": 1.5}"#), ("unknown", "{}"),]
    );
    // Without the tool schemas the values stay strings.
    let (_, calls) = parser.parse_complete(&output).await.unwrap();
    assert_eq!(calls[0].function.arguments, r#"{"x": "2", "y": "1.5"}"#);
    assert!(parser.has_tool_markers(&output));
    assert!(!parser.has_tool_markers("Sure."));
}

#[tokio::test]
async fn after_an_error_the_output_passes_through() {
    let mut parser = TemplateToolParser::new(qwen2());
    let tools = create_test_tools();
    let result = parser
        .parse_incremental("<tool_call>{\"name\": </tool_call>", &tools)
        .await
        .unwrap();
    assert_eq!(result.normal_text, "<tool_call>{\"name\": </tool_call>");
    let result = parser
        .parse_incremental("<tool_call>", &tools)
        .await
        .unwrap();
    assert_eq!(result.normal_text, "<tool_call>");
    assert!(result.calls.is_empty());
    assert_eq!(parser.take_unstreamed_normal_text(), "");
    assert!(parser.get_unstreamed_tool_args().is_none());
}
