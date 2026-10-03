use openai_protocol::common::Tool;
use serde_json::{json, Value};
use tool_parser::{parsers::QwenXmlParser, Glm4MoeParser, MinimaxM2Parser, ToolParser};

#[tokio::test]
async fn test_additional_properties_string_complete_and_streaming() {
    let tools: Vec<Tool> = serde_json::from_value(json!([{
        "type": "function",
        "function": {
            "name": "save_config",
            "parameters": {"type": "object", "additionalProperties": {"type": "string"}}
        }
    }]))
    .unwrap();
    let inputs = [
        "<tool_call>save_config<arg_key>code</arg_key><arg_value>42</arg_value><arg_key>empty</arg_key><arg_value>null</arg_value></tool_call>",
        "<tool_call><function=save_config><parameter=code>42</parameter><parameter=empty>null</parameter></function></tool_call>",
        "<minimax:tool_call><invoke name=\"save_config\"><parameter name=\"code\">42</parameter><parameter name=\"empty\">null</parameter></invoke></minimax:tool_call>",
    ];
    for (dialect, input) in inputs.iter().enumerate() {
        for streaming in [false, true] {
            let mut parser: Box<dyn ToolParser> = match dialect {
                0 => Box::new(Glm4MoeParser::glm47()),
                1 => Box::new(QwenXmlParser::new()),
                _ => Box::new(MinimaxM2Parser::new()),
            };
            let arguments = if streaming {
                let mut arguments = String::new();
                let prefix = if dialect == 2 {
                    "<minimax:tool_call>"
                } else {
                    "<tool_call>"
                };
                let chunks = std::iter::once(&input[..prefix.len()]).chain(
                    input.as_bytes()[prefix.len()..]
                        .chunks(7)
                        .map(|chunk| std::str::from_utf8(chunk).unwrap()),
                );
                for chunk in chunks {
                    for call in parser.parse_incremental(chunk, &tools).await.unwrap().calls {
                        assert_eq!(call.tool_index, 0);
                        arguments.push_str(&call.parameters);
                    }
                }
                if let Some(calls) = parser.get_unstreamed_tool_args() {
                    for call in calls {
                        arguments.push_str(&call.parameters);
                    }
                }
                arguments
            } else {
                let (_, calls) = parser
                    .parse_complete_with_tools(input, &tools)
                    .await
                    .unwrap();
                assert_eq!(calls.len(), 1);
                calls[0].function.arguments.clone()
            };
            assert_eq!(
                serde_json::from_str::<Value>(&arguments).unwrap(),
                json!({"code": "42", "empty": "null"}),
                "dialect={dialect}, streaming={streaming}"
            );
        }
    }
}
