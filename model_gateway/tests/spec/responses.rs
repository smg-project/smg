use openai_protocol::{
    common::{ConversationRef, Function, StringOrArray},
    responses::{
        FunctionTool, IncludeField, McpTool, ResponseInput, ResponseInputOutputItem, ResponseTool,
        ResponsesRequest, ResponsesToolChoice, StringOrContentParts, TextConfig, TextFormat,
        ToolChoiceOptions,
    },
};
use serde_json::json;
use validator::Validate;

/// Test that valid conversation IDs pass validation
#[test]
fn test_validate_conversation_id_valid() {
    let valid_ids = vec![
        "conv_123",
        "conv_test-123_abc",
        "conv_ABC_123",
        "conv_my_conversation_123",
        "conv_456",
        "conv_test123",
    ];

    for id in valid_ids {
        let request = ResponsesRequest {
            conversation: Some(ConversationRef::Id(id.to_string())),
            input: ResponseInput::Text("test".to_string()),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "Expected '{id}' to be valid, but got error: {:?}",
            request.validate().err()
        );
    }
}

/// Object-form `{"conversation": {"id": "conv_abc"}}` must deserialize into
/// `ConversationRef::Object { .. }` and pass the same validation as the
/// string-form.
#[test]
fn test_validate_conversation_id_object_form_valid() {
    let v = json!({
        "input": "test",
        "model": "gpt-4",
        "conversation": { "id": "conv_abc" },
    });
    let request: ResponsesRequest =
        serde_json::from_value(v).expect("object-form conversation should deserialize");
    assert!(
        matches!(
            request.conversation,
            Some(ConversationRef::Object { ref id }) if id == "conv_abc"
        ),
        "Expected ConversationRef::Object variant, got {:?}",
        request.conversation
    );
    assert!(
        request.validate().is_ok(),
        "Valid object-form conversation should pass validation, got: {:?}",
        request.validate().err()
    );
}

/// Object-form with an empty id must fail validation with the same
/// `invalid_conversation_id` error code used for the string-form.
#[test]
fn test_validate_conversation_id_object_form_empty_invalid() {
    let v = json!({
        "input": "test",
        "model": "gpt-4",
        "conversation": { "id": "" },
    });
    let request: ResponsesRequest =
        serde_json::from_value(v).expect("object-form conversation should deserialize");
    let result = request.validate();
    assert!(
        result.is_err(),
        "Object-form conversation with empty id should fail validation"
    );
    let errors = result.unwrap_err();
    let field_errors = errors.field_errors();
    let conversation_errors = field_errors
        .get("conversation")
        .expect("Expected error for 'conversation' field");
    let code = conversation_errors.first().map(|e| e.code.as_ref());
    assert_eq!(
        code,
        Some("invalid_conversation_id"),
        "Expected 'invalid_conversation_id' error code, got: {code:?}"
    );
}

/// Test that invalid conversation IDs fail validation
#[test]
fn test_validate_conversation_id_invalid() {
    let invalid_ids = vec![
        // Missing 'conv_' prefix
        "test-conv-streaming",
        "conversation-456",
        "my_conversation_123",
        "ABC123",
        "test_123_conv",
        "conv123", // missing underscore
        // Invalid characters
        "conv_.test",     // contains dot
        "conv_ test",     // contains space
        "conv_@test",     // contains @
        "conv_/test",     // contains /
        "conv_\\test",    // contains backslash
        "conv_:test",     // contains colon
        "conv_;test",     // contains semicolon
        "conv_,test",     // contains comma
        "conv_+test",     // contains plus
        "conv_=test",     // contains equals
        "conv_[test]",    // contains brackets
        "conv_{test}",    // contains braces
        "conv_(test)",    // contains parentheses
        "conv_!test",     // contains exclamation
        "conv_?test",     // contains question mark
        "conv_#test",     // contains hash
        "conv_$test",     // contains dollar sign
        "conv_%test",     // contains percent
        "conv_&test",     // contains ampersand
        "conv_*test",     // contains asterisk
        "conv_ test-123", // contains space
    ];

    for id in invalid_ids {
        let request = ResponsesRequest {
            conversation: Some(ConversationRef::Id(id.to_string())),
            input: ResponseInput::Text("test".to_string()),
            ..Default::default()
        };
        let result = request.validate();
        assert!(
            result.is_err(),
            "Expected '{id}' to be invalid, but validation passed"
        );

        // Verify error is for conversation field
        if let Err(errors) = result {
            let field_errors = errors.field_errors();
            let conversation_errors = field_errors.get("conversation");
            assert!(
                conversation_errors.is_some(),
                "Expected error for 'conversation' field, but got errors for: {:?}",
                field_errors.keys()
            );

            let error_msg = conversation_errors
                .and_then(|errs| errs.first())
                .and_then(|err| err.message.as_ref())
                .map(|msg| msg.to_string());

            assert!(
                error_msg.is_some(),
                "Expected error message for conversation field"
            );
            let msg = error_msg.unwrap();
            assert!(
                msg.contains("Invalid 'conversation'"),
                "Error message should mention 'conversation', got: {msg}"
            );
            assert!(
                msg.contains(id),
                "Error message should include the invalid ID '{id}', got: {msg}"
            );
        }
    }
}

/// Test that None conversation ID is valid
#[test]
fn test_validate_conversation_id_none() {
    let request = ResponsesRequest {
        conversation: None,
        input: ResponseInput::Text("test".to_string()),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "Request with no conversation ID should be valid"
    );
}

/// Test the exact error format matches OpenAI's error message for invalid characters
#[test]
fn test_validate_conversation_id_error_message_format() {
    let invalid_id = "conv_.test-conv-streaming";
    let request = ResponsesRequest {
        conversation: Some(ConversationRef::Id(invalid_id.to_string())),
        input: ResponseInput::Text("test".to_string()),
        ..Default::default()
    };

    let result = request.validate();
    assert!(result.is_err());

    if let Err(errors) = result {
        let error_msg = errors
            .field_errors()
            .get("conversation")
            .and_then(|errs| errs.first())
            .and_then(|err| err.message.as_ref())
            .map(|msg| msg.to_string())
            .unwrap();

        // Verify the error message matches OpenAI's format
        assert!(
            error_msg.starts_with("Invalid 'conversation':"),
            "Error should start with \"Invalid 'conversation':\""
        );
        assert!(
            error_msg.contains("letters, numbers, underscores, or dashes"),
            "Error should mention valid characters"
        );
        assert!(
            error_msg.contains(invalid_id),
            "Error should include the invalid conversation ID"
        );
    }
}

/// Test the exact error format for missing 'conv_' prefix
#[test]
fn test_validate_conversation_id_missing_prefix() {
    let invalid_id = "test-conv-streaming";
    let request = ResponsesRequest {
        conversation: Some(ConversationRef::Id(invalid_id.to_string())),
        input: ResponseInput::Text("test".to_string()),
        ..Default::default()
    };

    let result = request.validate();
    assert!(result.is_err());

    if let Err(errors) = result {
        let error_msg = errors
            .field_errors()
            .get("conversation")
            .and_then(|errs| errs.first())
            .and_then(|err| err.message.as_ref())
            .map(|msg| msg.to_string())
            .unwrap();

        // Verify the error message matches OpenAI's format
        assert!(
            error_msg.starts_with("Invalid 'conversation':"),
            "Error should start with \"Invalid 'conversation':\""
        );
        assert!(
            error_msg.contains("begins with 'conv_'"),
            "Error should mention the required prefix, got: {error_msg}"
        );
        assert!(
            error_msg.contains(invalid_id),
            "Error should include the invalid conversation ID"
        );
    }
}

// ============================================================================
// Field-Level Validation Tests
// ============================================================================

/// Test temperature range validation
#[test]
fn test_validate_temperature_range() {
    // Valid temperatures
    for temp in [0.0, 1.0, 2.0, 0.5, 1.5] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            temperature: Some(temp),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "Temperature {temp} should be valid"
        );
    }

    // Invalid temperatures
    for temp in [-0.1, 2.1, -1.0, 3.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            temperature: Some(temp),
            ..Default::default()
        };
        assert!(
            request.validate().is_err(),
            "Temperature {temp} should be invalid"
        );
    }
}

/// Test frequency_penalty range validation
#[test]
fn test_validate_frequency_penalty_range() {
    // Valid penalties
    for penalty in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            frequency_penalty: Some(penalty),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "Frequency penalty {penalty} should be valid"
        );
    }

    // Invalid penalties
    for penalty in [-2.1, 2.1, -3.0, 3.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            frequency_penalty: Some(penalty),
            ..Default::default()
        };
        assert!(
            request.validate().is_err(),
            "Frequency penalty {penalty} should be invalid"
        );
    }
}

/// Test presence_penalty range validation
#[test]
fn test_validate_presence_penalty_range() {
    // Valid penalties
    for penalty in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            presence_penalty: Some(penalty),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "Presence penalty {penalty} should be valid"
        );
    }

    // Invalid penalties
    for penalty in [-2.1, 2.1, -3.0, 3.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            presence_penalty: Some(penalty),
            ..Default::default()
        };
        assert!(
            request.validate().is_err(),
            "Presence penalty {penalty} should be invalid"
        );
    }
}

/// Test top_logprobs range validation
#[test]
fn test_validate_top_logprobs_range() {
    // Valid values
    for val in [0, 1, 10, 20] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            top_logprobs: Some(val),
            include: Some(vec![IncludeField::MessageOutputTextLogprobs]),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "top_logprobs {val} should be valid"
        );
    }

    // Invalid values
    for val in [21, 30, 100] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            top_logprobs: Some(val),
            include: Some(vec![IncludeField::MessageOutputTextLogprobs]),
            ..Default::default()
        };
        assert!(
            request.validate().is_err(),
            "top_logprobs {val} should be invalid"
        );
    }
}

/// Test top_p range validation
#[test]
fn test_validate_top_p_range() {
    // Valid values (> 0.0 and <= 1.0)
    for val in [0.01, 0.5, 1.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            top_p: Some(val),
            ..Default::default()
        };
        assert!(request.validate().is_ok(), "top_p {val} should be valid");
    }

    // Invalid values (0.0 is invalid because it means no tokens, < 0 or > 1)
    for val in [0.0, -0.1, 1.1, 2.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            top_p: Some(val),
            ..Default::default()
        };
        assert!(request.validate().is_err(), "top_p {val} should be invalid");
    }
}

/// Test top_k validation
#[test]
fn test_validate_top_k() {
    // Valid values (-1 means disabled, or >= 1)
    for val in [-1, 1, 10, 100] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            top_k: val,
            ..Default::default()
        };
        assert!(request.validate().is_ok(), "top_k {val} should be valid");
    }

    // Invalid values (0 or < -1)
    for val in [0, -2, -10] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            top_k: val,
            ..Default::default()
        };
        assert!(request.validate().is_err(), "top_k {val} should be invalid");
    }
}

/// Test min_p range validation
#[test]
fn test_validate_min_p_range() {
    // Valid values
    for val in [0.0, 0.5, 1.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            min_p: val,
            ..Default::default()
        };
        assert!(request.validate().is_ok(), "min_p {val} should be valid");
    }

    // Invalid values
    for val in [-0.1, 1.1, 2.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            min_p: val,
            ..Default::default()
        };
        assert!(request.validate().is_err(), "min_p {val} should be invalid");
    }
}

/// Test repetition_penalty range validation
#[test]
fn test_validate_repetition_penalty_range() {
    // Valid values
    for val in [0.0, 1.0, 2.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            repetition_penalty: val,
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "repetition_penalty {val} should be valid"
        );
    }

    // Invalid values
    for val in [-0.1, 2.1, 3.0] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            repetition_penalty: val,
            ..Default::default()
        };
        assert!(
            request.validate().is_err(),
            "repetition_penalty {val} should be invalid"
        );
    }
}

/// Test max_output_tokens minimum validation
#[test]
fn test_validate_max_output_tokens() {
    // Valid values
    for val in [1, 100, 1000] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            max_output_tokens: Some(val),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "max_output_tokens {val} should be valid"
        );
    }

    // Invalid values
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        max_output_tokens: Some(0),
        ..Default::default()
    };
    assert!(
        request.validate().is_err(),
        "max_output_tokens 0 should be invalid"
    );
}

/// Test max_tool_calls minimum validation
#[test]
fn test_validate_max_tool_calls() {
    // Valid values
    for val in [1, 5, 10] {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            max_tool_calls: Some(val),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "max_tool_calls {val} should be valid"
        );
    }

    // Invalid values
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        max_tool_calls: Some(0),
        ..Default::default()
    };
    assert!(
        request.validate().is_err(),
        "max_tool_calls 0 should be invalid"
    );
}

/// Test input validation (empty text)
#[test]
fn test_validate_input_empty_text() {
    let request = ResponsesRequest {
        input: ResponseInput::Text(String::new()),
        ..Default::default()
    };
    let result = request.validate();
    assert!(result.is_err(), "Empty input text should be invalid");

    if let Err(errors) = result {
        let error_msg = errors.to_string();
        assert!(
            error_msg.contains("input") || error_msg.contains("empty"),
            "Error should mention input or empty"
        );
    }
}

/// Test input validation (empty items array)
#[test]
fn test_validate_input_empty_items() {
    let request = ResponsesRequest {
        input: ResponseInput::Items(vec![]),
        ..Default::default()
    };
    let result = request.validate();
    assert!(result.is_err(), "Empty input items should be invalid");
}

/// Test input validation (items with empty content)
#[test]
fn test_validate_input_items_empty_content() {
    let request = ResponsesRequest {
        input: ResponseInput::Items(vec![ResponseInputOutputItem::SimpleInputMessage {
            content: StringOrContentParts::String(String::new()),
            role: "user".to_string(),
            r#type: None,
            phase: None,
        }]),
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "Input item with empty content should be invalid"
    );
}

/// Test stop sequences validation (max 4)
#[test]
fn test_validate_stop_sequences_max() {
    // Valid: 4 or fewer stop sequences
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        stop: Some(StringOrArray::Array(vec![
            "stop1".to_string(),
            "stop2".to_string(),
            "stop3".to_string(),
            "stop4".to_string(),
        ])),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "4 stop sequences should be valid"
    );

    // Invalid: more than 4 stop sequences
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        stop: Some(StringOrArray::Array(vec![
            "stop1".to_string(),
            "stop2".to_string(),
            "stop3".to_string(),
            "stop4".to_string(),
            "stop5".to_string(),
        ])),
        ..Default::default()
    };
    assert!(
        request.validate().is_err(),
        "5 stop sequences should be invalid"
    );
}

/// Test stop sequences validation (non-empty)
#[test]
fn test_validate_stop_sequences_non_empty() {
    // Invalid: empty string stop sequence
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        stop: Some(StringOrArray::String(String::new())),
        ..Default::default()
    };
    assert!(
        request.validate().is_err(),
        "Empty stop sequence should be invalid"
    );

    // Invalid: array with empty string
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        stop: Some(StringOrArray::Array(vec![String::new()])),
        ..Default::default()
    };
    assert!(
        request.validate().is_err(),
        "Array with empty stop sequence should be invalid"
    );
}

/// Test tools validation (function tool must have function)
#[test]
fn test_validate_tools_function_missing() {
    // With type-discriminated tools, a function tool without the required fields
    // should fail deserialization.
    let v = json!({
        "input": "test",
        "tools": [{ "type": "function" }]
    });
    let parsed: Result<ResponsesRequest, _> = serde_json::from_value(v);
    assert!(parsed.is_err(), "Expected deserialization to fail");
}

#[test]
fn test_deserialize_function_tool_rejects_unknown_fields() {
    let v = json!({
        "input": "test",
        "tools": [
            {
                "type": "function",
                "name": "test_func",
                "parameters": {},
                "extra_field": 1
            }
        ]
    });

    let parsed: Result<ResponsesRequest, _> = serde_json::from_value(v);
    assert!(parsed.is_err(), "Expected deserialization to fail");
}

/// Test tools validation (valid MCP tool should pass)
#[test]
fn test_validate_tools_mcp_valid_ok() {
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![ResponseTool::Mcp(McpTool {
            server_url: None, // server_url is optional when server_label is provided
            authorization: None,
            headers: None,
            server_label: "mock".to_string(),
            server_description: None,
            require_approval: None,
            allowed_tools: None,
            connector_id: None,
            defer_loading: None,
        })]),
        ..Default::default()
    };

    assert!(
        request.validate().is_ok(),
        "Valid MCP tool configuration should be accepted"
    );
}

/// Test tools validation (MCP tool must have server_label; server_url is optional)
#[test]
fn test_validate_tools_mcp_missing_server_label() {
    let v = json!({
        "input": "test",
        "tools": [{ "type": "mcp" }]
    });
    let parsed: Result<ResponsesRequest, _> = serde_json::from_value(v);
    assert!(parsed.is_err(), "Expected deserialization to fail");
}

/// Test tools validation (MCP tool server_label must be unique; case-insensitive)
#[test]
fn test_validate_tools_mcp_duplicate_server_label() {
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![
            ResponseTool::Mcp(McpTool {
                server_url: None,
                authorization: None,
                headers: None,
                server_label: "Foo".to_string(),
                server_description: None,
                require_approval: None,
                allowed_tools: None,
                connector_id: None,
                defer_loading: None,
            }),
            ResponseTool::Mcp(McpTool {
                server_url: None,
                authorization: None,
                headers: None,
                server_label: "foo".to_string(),
                server_description: None,
                require_approval: None,
                allowed_tools: None,
                connector_id: None,
                defer_loading: None,
            }),
        ]),
        ..Default::default()
    };

    let result = request.validate();
    assert!(
        result.is_err(),
        "MCP tools with duplicate server_label should be invalid"
    );

    let err = result.unwrap_err();
    assert!(
        format!("{err:?}").contains("mcp_tool_duplicate_server_label"),
        "Expected error code mcp_tool_duplicate_server_label, got: {err:?}",
    );
}

/// Test tools validation (MCP tool server_label must follow OpenAI label rules)
#[test]
fn test_validate_tools_mcp_server_label_invalid_cases() {
    let invalid_labels = vec!["   deepwiki", "1deepwiki", "deepwiki."];

    for label in invalid_labels {
        let request = ResponsesRequest {
            input: ResponseInput::Text("test".to_string()),
            tools: Some(vec![ResponseTool::Mcp(McpTool {
                server_url: Some("https://example.com/mcp".to_string()),
                authorization: None,
                headers: None,
                server_label: label.to_string(),
                server_description: None,
                require_approval: None,
                allowed_tools: None,
                connector_id: None,
                defer_loading: None,
            })]),
            ..Default::default()
        };

        let result = request.validate();
        assert!(
            result.is_err(),
            "Expected '{label}' to be invalid, but validation passed",
        );

        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("invalid_server_label"),
            "Expected error code invalid_server_label, got: {err:?}",
        );
    }
}

/// Test text format validation (JSON schema name cannot be empty)
#[test]
fn test_validate_text_format_json_schema_empty_name() {
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        text: Some(TextConfig {
            format: Some(TextFormat::JsonSchema {
                name: String::new(), // Empty name
                schema: json!({}),
                description: None,
                strict: None,
            }),
        }),
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "JSON schema with empty name should be invalid"
    );
}

// ============================================================================
// Cross-Field Validation Tests (Schema-Level)
// ============================================================================

/// Test tool_choice requires tools
#[test]
fn test_validate_tool_choice_requires_tools() {
    // Valid: tool_choice with tools
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "test_func".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        })]),
        tool_choice: Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto)),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "tool_choice with tools should be valid"
    );

    // Valid: tool_choice=none without tools is OK
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: None,
        tool_choice: Some(ResponsesToolChoice::Options(ToolChoiceOptions::None)),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "tool_choice=none without tools should be valid"
    );

    // Invalid: tool_choice=auto without tools
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: None,
        tool_choice: Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto)),
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "tool_choice=auto without tools should be invalid"
    );

    if let Err(errors) = result {
        let error_msg = errors.to_string();
        assert!(
            error_msg.contains("tool_choice") && error_msg.contains("tools"),
            "Error should mention tool_choice requires tools"
        );
    }
}

/// Test top_logprobs requires include field
#[test]
fn test_validate_top_logprobs_requires_include() {
    // Valid: top_logprobs with correct include field
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        top_logprobs: Some(5),
        include: Some(vec![IncludeField::MessageOutputTextLogprobs]),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "top_logprobs with include field should be valid"
    );

    // Invalid: top_logprobs without include field
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        top_logprobs: Some(5),
        include: None,
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "top_logprobs without include field should be invalid"
    );

    // Invalid: top_logprobs with wrong include field
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        top_logprobs: Some(5),
        include: Some(vec![IncludeField::ReasoningEncryptedContent]),
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "top_logprobs with wrong include field should be invalid"
    );
}

/// Test previous_response_id format validation
/// NOTE: Format validation removed - previous_response_id format is not validated
/// response_id generated by the grpc router is not necessarily start with 'resp_'
#[test]
#[ignore]
fn test_validate_previous_response_id_format() {
    // Valid: starts with "resp_"
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        previous_response_id: Some("resp_123abc".to_string()),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "previous_response_id with resp_ prefix should be valid"
    );

    // Invalid: doesn't start with "resp_"
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        previous_response_id: Some("response_123abc".to_string()),
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "previous_response_id without resp_ prefix should be invalid"
    );

    if let Err(errors) = result {
        let error_msg = errors.to_string();
        assert!(
            error_msg.contains("previous_response_id") && error_msg.contains("resp_"),
            "Error should mention previous_response_id format"
        );
    }
}

/// Test conversation and previous_response_id mutual exclusion
#[test]
fn test_validate_conversation_previous_response_mutual_exclusion() {
    // Valid: only conversation
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        conversation: Some(ConversationRef::Id("conv_123".to_string())),
        previous_response_id: None,
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "Only conversation should be valid"
    );

    // Valid: only previous_response_id
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        conversation: None,
        previous_response_id: Some("resp_123".to_string()),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "Only previous_response_id should be valid"
    );

    // Invalid: both conversation and previous_response_id
    let request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        conversation: Some(ConversationRef::Id("conv_123".to_string())),
        previous_response_id: Some("resp_123".to_string()),
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "Both conversation and previous_response_id should be invalid"
    );

    if let Err(errors) = result {
        let error_msg = errors.to_string();
        assert!(
            error_msg.contains("mutually exclusive")
                || (error_msg.contains("conversation")
                    && error_msg.contains("previous_response_id")),
            "Error should mention mutual exclusion, got: {error_msg}"
        );
    }
}

/// Test input items structure validation
#[test]
fn test_validate_input_items_structure() {
    // Valid: items with at least one message
    let request = ResponsesRequest {
        input: ResponseInput::Items(vec![ResponseInputOutputItem::SimpleInputMessage {
            content: StringOrContentParts::String("Hello".to_string()),
            role: "user".to_string(),
            r#type: None,
            phase: None,
        }]),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "Input items with message should be valid"
    );

    // Invalid: items with no messages (only function calls)
    let request = ResponsesRequest {
        input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
            id: None,
            call_id: "call_123".to_string(),
            output: "result".to_string().into(),
            status: None,
        }]),
        ..Default::default()
    };
    let result = request.validate();
    assert!(
        result.is_err(),
        "Input items without messages should be invalid"
    );
}

// ============================================================================
// Normalization Tests (Normalizable Trait)
// ============================================================================

/// Test tool_choice defaults to auto when tools are present
#[test]
fn test_normalize_tool_choice_auto() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "test_func".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        })]),
        tool_choice: None,
        ..Default::default()
    };

    request.normalize();

    assert!(
        request.tool_choice.is_some(),
        "tool_choice should be set after normalization"
    );
    assert!(
        matches!(
            request.tool_choice,
            Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto))
        ),
        "tool_choice should default to auto when tools are present"
    );
}

/// Test tool_choice defaults to none when tools array is empty
#[test]
fn test_normalize_tool_choice_none() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![]),
        tool_choice: None,
        ..Default::default()
    };

    request.normalize();

    assert!(
        request.tool_choice.is_some(),
        "tool_choice should be set after normalization"
    );
    assert!(
        matches!(
            request.tool_choice,
            Some(ResponsesToolChoice::Options(ToolChoiceOptions::None))
        ),
        "tool_choice should default to none when tools array is empty"
    );
}

/// Test tool_choice is not overridden if already set
#[test]
fn test_normalize_tool_choice_no_override() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "test_func".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        })]),
        tool_choice: Some(ResponsesToolChoice::Options(ToolChoiceOptions::Required)),
        ..Default::default()
    };

    request.normalize();

    assert!(
        matches!(
            request.tool_choice,
            Some(ResponsesToolChoice::Options(ToolChoiceOptions::Required))
        ),
        "tool_choice should not be overridden if already set"
    );
}

/// Test parallel_tool_calls defaults to true when tools are present
#[test]
fn test_normalize_parallel_tool_calls() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "test_func".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        })]),
        parallel_tool_calls: None,
        ..Default::default()
    };

    request.normalize();

    assert!(
        request.parallel_tool_calls.is_some(),
        "parallel_tool_calls should be set after normalization"
    );
    assert_eq!(
        request.parallel_tool_calls,
        Some(true),
        "parallel_tool_calls should default to true when tools are present"
    );
}

/// Test parallel_tool_calls is not set when tools are absent
#[test]
fn test_normalize_parallel_tool_calls_no_tools() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: None,
        parallel_tool_calls: None,
        ..Default::default()
    };

    request.normalize();

    assert!(
        request.parallel_tool_calls.is_none(),
        "parallel_tool_calls should remain None when tools are absent"
    );
}

/// Test parallel_tool_calls is not overridden if already set
#[test]
fn test_normalize_parallel_tool_calls_no_override() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "test_func".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        })]),
        parallel_tool_calls: Some(false),
        ..Default::default()
    };

    request.normalize();

    assert_eq!(
        request.parallel_tool_calls,
        Some(false),
        "parallel_tool_calls should not be overridden if already set"
    );
}

/// Test store defaults to true
#[test]
fn test_normalize_store_default() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        store: None,
        ..Default::default()
    };

    request.normalize();

    assert!(
        request.store.is_some(),
        "store should be set after normalization"
    );
    assert_eq!(request.store, Some(true), "store should default to true");
}

/// Test store is not overridden if already set
#[test]
fn test_normalize_store_no_override() {
    use openai_protocol::validated::Normalizable;

    let mut request = ResponsesRequest {
        input: ResponseInput::Text("test".to_string()),
        store: Some(false),
        ..Default::default()
    };

    request.normalize();

    assert_eq!(
        request.store,
        Some(false),
        "store should not be overridden if already set to false"
    );
}
