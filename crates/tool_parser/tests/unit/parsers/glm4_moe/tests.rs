use openai_protocol::common::Function;

use super::*;

fn tool(name: &str, strict: Option<bool>, parameters: Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters: Some(parameters),
            strict,
            extra: Default::default(),
        },
    }
}

/// The tag follows xgrammar's built-in `glm_4_7` shape: one `<tool_call>`
/// trigger, a `glm_xml`-styled schema per tool passed through as written
/// (a no-argument tool's `{}` included), and `at_least_one` forcing a call.
#[test]
fn structural_tag_mirrors_the_xgrammar_glm_4_7_model() {
    let schema = serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let tools = vec![
        tool("get_weather", None, schema.clone()),
        tool("lookup", Some(false), schema.clone()),
        tool("ping", None, serde_json::json!({})),
        tool("", None, schema.clone()),
    ];
    let tag = Glm4MoeParser::build_structural_tag(&tools, true);
    let format = &tag["format"];
    assert_eq!(format["type"], "triggered_tags");
    assert_eq!(format["triggers"], serde_json::json!(["<tool_call>"]));
    assert_eq!(format["at_least_one"], true);
    let tags = format["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 3, "a nameless tool has no tag");
    assert_eq!(tags[0]["begin"], "<tool_call>get_weather");
    assert_eq!(tags[0]["end"], "</tool_call>");
    assert_eq!(tags[0]["content"]["type"], "json_schema");
    assert_eq!(tags[0]["content"]["style"], "glm_xml");
    assert_eq!(tags[0]["content"]["json_schema"], schema);
    assert_eq!(
        tags[1]["content"]["json_schema"], schema,
        "strict is not a constraint switch here, as in the other builders"
    );
    assert_eq!(
        tags[2]["content"]["json_schema"],
        serde_json::json!({}),
        "a no-argument tool keeps its empty schema"
    );

    let tag = Glm4MoeParser::build_structural_tag(&tools, false);
    assert_eq!(tag["format"]["at_least_one"], false);
}

#[test]
fn reasoning_prefix_is_free_text_that_owes_nothing_at_its_end() {
    let prefix = Glm4MoeParser::reasoning_prefix(GrammarStart::Unknown).unwrap();
    assert_eq!(prefix["type"], "any_text");
    assert!(
        prefix.get("end").is_none(),
        "nothing the model must write before the calls: {prefix}"
    );
    let excludes = prefix["excludes"].as_array().unwrap();
    for token in [
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
    ] {
        assert!(
            excludes.contains(&Value::String(token.to_string())),
            "{token} must not appear before the forced call"
        );
    }
    assert!(
        !excludes.contains(&Value::String("</think>".to_string())),
        "the model closes its thought here when the engine applies the grammar from the first token"
    );
}

/// Where the worker says where its engine starts the grammar, the prefix
/// is the one shape that engine needs: none at all behind a reasoning
/// parser's gate, the `</think>`-closed thought block (xgrammar's
/// `glm_4_7` with `reasoning=True`) from the first token.
#[test]
fn reasoning_prefix_follows_where_the_engine_starts_the_grammar() {
    assert_eq!(
        Glm4MoeParser::reasoning_prefix(GrammarStart::AfterReasoning),
        None,
        "the grammar starts at the engine's gate: the call is forced right there"
    );
    let block = Glm4MoeParser::reasoning_prefix(GrammarStart::FirstToken).unwrap();
    assert_eq!(block["type"], "tag");
    assert_eq!(block["begin"], "");
    assert_eq!(block["end"], "</think>");
    assert_eq!(block["content"]["type"], "any_text");
    let excludes = block["content"]["excludes"].as_array().unwrap();
    for token in ["<think>", "</think>", "<tool_call>", "</arg_value>"] {
        assert!(
            excludes.contains(&Value::String(token.to_string())),
            "{token} must not appear inside the thought block"
        );
    }
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

// String-typed params stay strings even when they look numeric/bool/array.
#[tokio::test]
async fn test_schema_aware_coercion_keeps_strings() {
    let tools = tool_with_props(serde_json::json!({
        "limit": {"type": "string"},
        "flag": {"type": "string"},
        "coords": {"type": "string"},
        "count": {"type": "integer"},
    }));
    let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n\
            <arg_key>coords</arg_key>\n<arg_value>[60,30]</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
    let (_, calls) = Glm4MoeParser::glm45()
        .parse_complete_with_tools(text, &tools)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], Value::String("4".to_string()));
    assert_eq!(args["flag"], Value::String("true".to_string()));
    assert_eq!(args["coords"], Value::String("[60,30]".to_string()));
    assert_eq!(args["count"], Value::Number(5.into()));
}

// The streaming path threads `tools` separately, so cover it too.
#[tokio::test]
async fn test_streaming_schema_aware_coercion() {
    let tools = tool_with_props(serde_json::json!({
        "limit": {"type": "string"},
        "count": {"type": "integer"},
    }));
    let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
    let result = Glm4MoeParser::glm45()
        .parse_incremental(text, &tools)
        .await
        .unwrap();
    let args: Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
    assert_eq!(args["limit"], Value::String("4".to_string()));
    assert_eq!(args["count"], Value::Number(5.into()));
}

#[tokio::test]
async fn test_declared_strings_preserve_whitespace() {
    let tools = tool_with_props(serde_json::json!({
        "code": {"type": "string"},
        "needle": {"type": "string"},
        "blank": {"type": "string"},
        "quoted": {"type": "string"},
        "count": {"type": "integer"},
    }));
    let args_text = concat!(
        "<arg_key>code</arg_key><arg_value>    return 1\n</arg_value>",
        "<arg_key>needle</arg_key><arg_value>    total = 1\n</arg_value>",
        "<arg_key>blank</arg_key><arg_value> \t\n</arg_value>",
        "<arg_key>quoted</arg_key><arg_value> \"    return 1\\n\" </arg_value>",
        "<arg_key>count</arg_key><arg_value> 42\n</arg_value>",
        "<arg_key>unknown</arg_key><arg_value> legacy text\n</arg_value>",
    );
    let expected = serde_json::json!({
        "code": "    return 1\n",
        "needle": "    total = 1\n",
        "blank": " \t\n",
        "quoted": "    return 1\n",
        "count": 42,
        "unknown": "legacy text",
    });

    for (make_parser, name_separator) in [
        (Glm4MoeParser::glm45 as fn() -> Glm4MoeParser, "\n"),
        (Glm4MoeParser::glm47 as fn() -> Glm4MoeParser, ""),
    ] {
        let text = format!("<tool_call>f{name_separator}{args_text}</tool_call>");
        let (_, calls) = make_parser()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, expected);

        // Keep the model's opening special token intact; split the argument
        // text into small chunks, including inside values and closing tags.
        let mut parser = make_parser();
        let start = "<tool_call>".len();
        let result = parser
            .parse_incremental(&text[..start], &tools)
            .await
            .unwrap();
        assert!(result.calls.is_empty());
        let mut arguments = String::new();
        for chunk in text.as_bytes()[start..].chunks(7) {
            let result = parser
                .parse_incremental(std::str::from_utf8(chunk).unwrap(), &tools)
                .await
                .unwrap();
            for call in result.calls {
                arguments.push_str(&call.parameters);
            }
        }
        let args: Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(args, expected);
    }
}
