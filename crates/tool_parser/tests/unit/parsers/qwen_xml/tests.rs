use openai_protocol::common::Function;

use super::*;

#[test]
fn test_safe_val_json() {
    assert_eq!(safe_val("42"), Value::Number(42.into()));
    assert_eq!(safe_val("1.5"), serde_json::json!(1.5));
    assert_eq!(safe_val("true"), Value::Bool(true));
    assert_eq!(safe_val("false"), Value::Bool(false));
    assert_eq!(safe_val("null"), Value::Null);
    assert_eq!(
        safe_val(r#"{"key": "value"}"#),
        serde_json::json!({"key": "value"})
    );
    assert_eq!(safe_val(r"[1, 2, 3]"), serde_json::json!([1, 2, 3]));
}

#[test]
fn test_safe_val_python_literals() {
    assert_eq!(safe_val("True"), Value::Bool(true));
    assert_eq!(safe_val("False"), Value::Bool(false));
    assert_eq!(safe_val("None"), Value::Null);
}

#[test]
fn test_safe_val_string_fallback() {
    assert_eq!(
        safe_val("hello world"),
        Value::String("hello world".to_string())
    );
    assert_eq!(safe_val("  spaces  "), Value::String("spaces".to_string()));
}

// Values are treated literally: entity-like substrings must NOT be decoded
// (parity with Qwen's official API, which returns them intact).
#[test]
fn test_safe_val_preserves_html_entities() {
    assert_eq!(
        safe_val("&lt;div&gt;"),
        Value::String("&lt;div&gt;".to_string())
    );
    assert_eq!(
        safe_val("Tom &amp; Jerry"),
        Value::String("Tom &amp; Jerry".to_string())
    );
    // Numeric/hex entities are likewise left untouched.
    assert_eq!(safe_val("it&#39;s"), Value::String("it&#39;s".to_string()));
    assert_eq!(safe_val("&#x3C;"), Value::String("&#x3C;".to_string()));
}

fn tool_with_props(props: Value) -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "f".to_string(),
            description: None,
            parameters: Some(serde_json::json!({"type": "object", "properties": props})),
            strict: None,
            extra: Default::default(),
        },
    }]
}

// String-typed params stay strings even when they look numeric/bool/array/object.
#[tokio::test]
async fn test_schema_aware_coercion_keeps_strings() {
    let tools = tool_with_props(serde_json::json!({
        "limit": {"type": "string"},
        "flag": {"type": "string"},
        "coords": {"type": "string"},
        "cfg": {"type": "string"},
        "count": {"type": "integer"},
    }));
    let text = "<tool_call>\n<function=f>\n\
            <parameter=limit>4</parameter>\n\
            <parameter=flag>true</parameter>\n\
            <parameter=coords>[60,30]</parameter>\n\
            <parameter=cfg>{\"a\": 1}</parameter>\n\
            <parameter=count>5</parameter>\n\
            </function>\n</tool_call>";
    let (_, calls) = QwenXmlParser::new()
        .parse_complete_with_tools(text, &tools)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], Value::String("4".to_string()));
    assert_eq!(args["flag"], Value::String("true".to_string()));
    assert_eq!(args["coords"], Value::String("[60,30]".to_string()));
    assert_eq!(args["cfg"], Value::String("{\"a\": 1}".to_string()));
    assert_eq!(args["count"], Value::Number(5.into()));
}

// The streaming path threads `tools` separately, so cover it too.
#[tokio::test]
async fn test_streaming_schema_aware_coercion() {
    let tools = tool_with_props(serde_json::json!({
        "limit": {"type": "string"},
        "count": {"type": "integer"},
    }));
    let text = "<tool_call>\n<function=f>\n\
            <parameter=limit>4</parameter>\n\
            <parameter=count>5</parameter>\n\
            </function>\n</tool_call>";
    let result = QwenXmlParser::new()
        .parse_incremental(text, &tools)
        .await
        .unwrap();
    let args: String = result.calls.iter().map(|c| c.parameters.as_str()).collect();
    assert!(
        args.contains(r#""limit": "4""#),
        "string param must stay string: {args}"
    );
    assert!(
        args.contains(r#""count": 5"#),
        "int param must coerce: {args}"
    );
}

// Golden conformance test (regression guard for #1888): a tool argument
// whose value contains HTML entities must round-trip UNCHANGED, matching
// Qwen's official API (verified on Qwen3-Coder and Qwen3.5). Covers both the
// schema-typed `string` path and the schema-less inference fallback.
#[tokio::test]
async fn test_arg_values_with_entities_roundtrip_unchanged() {
    let literal = "<a>Tom &amp; Jerry</a> &lt;x&gt; it&#39;s";

    // Function name matches `tool_with_props` (`f`) so the schema-typed
    // branch below actually resolves the `content` param's type.
    let text = format!(
        "<tool_call>\n<function=f>\n\
             <parameter=content>\n{literal}\n</parameter>\n\
             </function>\n</tool_call>"
    );

    // Schema-less inference path (`safe_val`): not valid JSON -> stays a
    // literal string with entities intact.
    let (_, calls) = QwenXmlParser::new().parse_complete(&text).await.unwrap();
    assert_eq!(calls.len(), 1);
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["content"], Value::String(literal.to_string()));

    // Schema-typed `string` path (`coerce_value` -> `coerce_by_schema_type`).
    let tools = tool_with_props(serde_json::json!({
        "content": {"type": "string"},
    }));
    let (_, calls) = QwenXmlParser::new()
        .parse_complete_with_tools(&text, &tools)
        .await
        .unwrap();
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["content"], Value::String(literal.to_string()));
}

fn bash_tool() -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "bash".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                }
            })),
            strict: None,
            extra: Default::default(),
        },
    }]
}

/// Feed `chunks` and collect the argument deltas (name-less items with
/// parameters) together with the index of the chunk that produced each.
async fn argument_deltas(
    parser: &mut QwenXmlParser,
    chunks: &[&str],
    tools: &[Tool],
) -> Vec<(usize, String)> {
    let mut deltas = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        for call in parser.parse_incremental(chunk, tools).await.unwrap().calls {
            if call.name.is_none() && !call.parameters.is_empty() {
                deltas.push((index, call.parameters));
            }
        }
    }
    deltas
}

// The value of a string parameter goes out as it is generated (smg-lab#27):
// the first argument bytes follow the name within a few tokens instead of
// the whole value arriving once `</parameter>` closes it.
#[tokio::test]
async fn test_string_values_stream_before_the_parameter_closes() {
    let tools = bash_tool();
    let chunks = [
        "<tool_call>",
        "\n",
        "<function=",
        "bash",
        ">\n",
        "<parameter=",
        "command",
        ">\n",
        "ls",
        " -",
        "la",
        " /",
        "tmp",
        "\n",
        "</parameter>",
        "\n</function>",
        "\n</tool_call>",
    ];
    let close_at = chunks.iter().position(|c| *c == "</parameter>").unwrap();
    let mut parser = QwenXmlParser::new();
    let deltas = argument_deltas(&mut parser, &chunks, &tools).await;

    assert!(deltas.len() > 2, "arguments arrived as {deltas:?}");
    assert_eq!(deltas[0], (8, "{\"command\": \"ls".to_string()));
    let before_close = deltas.iter().filter(|(at, _)| *at < close_at).count();
    assert_eq!(before_close, 5, "{deltas:?}");
    let joined: String = deltas.iter().map(|(_, d)| d.as_str()).collect();
    assert_eq!(joined, "{\"command\": \"ls -la /tmp\"}");
    assert!(parser.get_unstreamed_tool_args().is_none());
    assert!(parser.take_unstreamed_normal_text().is_empty());
}

// Whatever the chunking, the streamed arguments re-assemble to exactly
// the complete parse: escaped like JSON, trimmed like the XML value.
#[tokio::test]
async fn test_streamed_string_values_match_the_complete_parse_at_every_split() {
    let tools = bash_tool();
    let command = "<parameter=command>\n  echo \"a\\b\"\t<tag> \n\n</parameter>\n";
    let timeout = "<parameter=timeout>30</parameter>\n";
    // Both orders: a chunk may carry a complete parameter and the start of
    // the string one.
    for text in [
        format!("<tool_call>\n<function=bash>\n{command}{timeout}</function>\n</tool_call>"),
        format!("<tool_call>\n<function=bash>\n{timeout}{command}</function>\n</tool_call>"),
    ] {
        let (_, complete) = QwenXmlParser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        let expected: Value = serde_json::from_str(&complete[0].function.arguments).unwrap();
        assert_eq!(expected["command"], "echo \"a\\b\"\t<tag>");
        assert_eq!(expected["timeout"], 30);

        let mut feeds: Vec<Vec<String>> = vec![text.chars().map(|c| c.to_string()).collect()];
        for (split, _) in text.char_indices().skip(1) {
            feeds.push(vec![text[..split].to_string(), text[split..].to_string()]);
        }
        for chunks in feeds {
            let mut parser = QwenXmlParser::new();
            let mut args = String::new();
            for chunk in &chunks {
                for call in parser.parse_incremental(chunk, &tools).await.unwrap().calls {
                    args.push_str(&call.parameters);
                }
            }
            assert!(parser.get_unstreamed_tool_args().is_none(), "{chunks:?}");
            assert_eq!(
                serde_json::from_str::<Value>(&args).ok().as_ref(),
                Some(&expected),
                "args {args}; chunks {chunks:?}"
            );
        }
    }
}

// Coercion needs the complete value: non-string parameters and JSON
// string literals (which the complete parse unwraps) still arrive whole.
#[tokio::test]
async fn test_values_that_need_the_complete_text_arrive_whole() {
    let tools = bash_tool();
    for (text, expected) in [
        (
            "<tool_call><function=bash><parameter=timeout>30</parameter></function></tool_call>",
            ["{\"timeout\": 30", "}"],
        ),
        (
            "<tool_call><function=bash><parameter=command>\"ls\"</parameter></function></tool_call>",
            ["{\"command\": \"ls\"", "}"],
        ),
    ] {
        let chunks: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let mut parser = QwenXmlParser::new();
        let deltas: Vec<String> = argument_deltas(&mut parser, &chunk_refs, &tools)
            .await
            .into_iter()
            .map(|(_, delta)| delta)
            .collect();
        assert_eq!(deltas, expected);
    }
}

// A value the stream ends in is closed with what arrived, as the engine's
// own streaming parser does; a parameter without any value is not invented.
#[tokio::test]
async fn test_end_of_stream_closes_a_partially_streamed_value() {
    let tools = bash_tool();
    let mut parser = QwenXmlParser::new();
    let deltas = argument_deltas(
        &mut parser,
        &[
            "<tool_call><function=bash><parameter=command>ls -l",
            "a /tm",
        ],
        &tools,
    )
    .await;
    let mut args: String = deltas.into_iter().map(|(_, d)| d).collect();
    assert_eq!(args, "{\"command\": \"ls -la /tm");
    for item in parser.get_unstreamed_tool_args().unwrap() {
        args.push_str(&item.parameters);
    }
    assert_eq!(args, "{\"command\": \"ls -la /tm\"}");
    assert!(parser.take_unstreamed_normal_text().is_empty());

    let mut parser = QwenXmlParser::new();
    let deltas = argument_deltas(
        &mut parser,
        &["<tool_call><function=bash><parameter=command>\n"],
        &tools,
    )
    .await;
    assert!(deltas.is_empty(), "{deltas:?}");
    let pending: String = parser
        .get_unstreamed_tool_args()
        .unwrap()
        .into_iter()
        .map(|item| item.parameters)
        .collect();
    assert_eq!(pending, "{}");
}
