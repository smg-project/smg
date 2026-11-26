use serde_json::json;

use super::*;

#[test]
fn test_passthrough_transform() {
    let result = json!({"key": "value"});
    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "call_test-1",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { id, output, .. } => {
            assert_eq!(id, "mcp_test-1");
            assert!(output.contains("key"));
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_flattens_single_text_block() {
    let result = json!([
        {"type": "text", "text": "hello from mcp"}
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "test-2",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { output, .. } => {
            assert_eq!(output, "hello from mcp");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_fc_id_to_mcp_prefix() {
    let result = json!({"key": "value"});
    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "fc_abc123",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { id, .. } => {
            assert_eq!(id, "mcp_abc123");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_flattens_multiple_text_blocks() {
    let result = json!([
        {"type": "text", "text": "first block"},
        {"type": "text", "text": "second block"}
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "test-3",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { output, .. } => {
            assert_eq!(output, "first block\nsecond block");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_preserves_existing_mcp_prefix() {
    let result = json!({"key": "value"});
    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "mcp_existing",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { id, .. } => {
            assert_eq!(id, "mcp_existing");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_ignores_non_text_blocks() {
    let result = json!([
        {"type": "text", "text": "kept text"},
        {"type": "image", "url": "https://example.com/image.png"},
        {"type": "resource", "uri": "file:///tmp/test.txt"}
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "test-4",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { output, .. } => {
            assert_eq!(output, "kept text");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_ignores_typed_non_text_blocks_with_text_fields() {
    let result = json!([
        {"type": "text", "text": "kept text"},
        {"type": "image", "text": "caption that should be ignored"}
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "test-4b",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { output, .. } => {
            assert_eq!(output, "kept text");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_uses_error_message_for_structured_errors() {
    let result = json!({
        "error": {
            "code": "tool_failed",
            "message": "tool execution failed"
        }
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "test-5",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { output, .. } => {
            assert_eq!(output, "tool execution failed");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_passthrough_transform_keeps_content_when_error_has_no_text() {
    let result = json!([
        {"type": "text", "text": "hello"},
        {
            "error": {"code": "tool_failed"},
            "content": [
                {"type": "text", "text": "important"}
            ]
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::Passthrough,
        "test-6",
        "server",
        "tool",
        "{}",
    );

    match transformed {
        ResponseOutputItem::McpCall { output, .. } => {
            assert_eq!(output, "hello\nimportant");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn test_web_search_transform() {
    let result = json!([
        {
            "type": "text",
            "text": r#"{
                  "queries": ["rust examples"],
                  "openai_response": {
                    "sources": [
                      {"url": "https://example.com", "title": "Example"},
                      {"url": "https://rust-lang.org", "title": "Rust"}
                    ]
                  }
                }"#
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::WebSearchCall,
        "req-123",
        "server",
        "web_search",
        r#"{"query":"rust from args"}"#,
    );

    match transformed {
        ResponseOutputItem::WebSearchCall {
            id,
            status,
            action,
            results,
        } => {
            assert_eq!(id, "ws_req-123");
            assert_eq!(status, WebSearchCallStatus::Completed);
            assert!(results.is_none());
            match action {
                WebSearchAction::Search {
                    query,
                    queries,
                    sources,
                } => {
                    assert_eq!(query, Some("rust from args".to_string()));
                    assert_eq!(queries, vec!["rust examples".to_string()]);
                    assert_eq!(sources.len(), 2);
                    assert_eq!(sources[0].url, "https://example.com");
                }
                _ => panic!("Expected Search action"),
            }
        }
        _ => panic!("Expected WebSearchCall"),
    }
}

#[test]
fn test_web_search_transform_extracts_sources_and_query_from_embedded_json_text() {
    let result = json!([
        {
            "type": "text",
            "text": r#"{
                  "execution_id": "123",
                  "openai_response": {
                    "sources": [
                      {"type": "url", "url": "https://example.com/a"},
                      {"type": "url", "url": "https://example.com/b"}
                    ]
                  }
                }"#
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::WebSearchCall,
        "req-embedded",
        "server",
        "web_search",
        r#"{"query":"positive news story April 8 2026"}"#,
    );

    match transformed {
        ResponseOutputItem::WebSearchCall { action, .. } => match action {
            WebSearchAction::Search {
                query,
                queries,
                sources,
            } => {
                assert_eq!(query, Some("positive news story April 8 2026".to_string()));
                assert!(queries.is_empty());
                assert_eq!(sources.len(), 2);
                assert_eq!(sources[0].url, "https://example.com/a");
                assert_eq!(sources[1].url, "https://example.com/b");
            }
            _ => panic!("Expected Search action"),
        },
        _ => panic!("Expected WebSearchCall"),
    }
}

#[test]
fn test_web_search_transform_falls_back_to_results_sources() {
    let result = json!({
        "results": [
            {"type": "url", "url": "https://example.com/legacy-a"},
            {"type": "url", "url": "https://example.com/legacy-b"}
        ]
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::WebSearchCall,
        "req-legacy",
        "server",
        "web_search",
        r#"{"query":"legacy search"}"#,
    );

    match transformed {
        ResponseOutputItem::WebSearchCall { action, .. } => match action {
            WebSearchAction::Search {
                query,
                queries,
                sources,
            } => {
                assert_eq!(query, Some("legacy search".to_string()));
                assert!(queries.is_empty());
                assert_eq!(sources.len(), 2);
                assert_eq!(sources[0].url, "https://example.com/legacy-a");
                assert_eq!(sources[1].url, "https://example.com/legacy-b");
            }
            _ => panic!("Expected Search action"),
        },
        _ => panic!("Expected WebSearchCall"),
    }
}

#[test]
fn test_code_interpreter_transform() {
    let result = json!({
        "code": "print('hello')",
        "container_id": "cntr_abc123",
        "stdout": "hello\n"
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::CodeInterpreterCall,
        "req-456",
        "server",
        "code_interpreter",
        "{}",
    );

    match transformed {
        ResponseOutputItem::CodeInterpreterCall {
            id,
            status,
            code,
            outputs,
            ..
        } => {
            assert_eq!(id, "ci_req-456");
            assert_eq!(status, CodeInterpreterCallStatus::Completed);
            assert_eq!(code, Some("print('hello')".to_string()));
            assert!(outputs.is_some());
            assert_eq!(outputs.unwrap().len(), 1);
        }
        _ => panic!("Expected CodeInterpreterCall"),
    }
}

#[test]
fn test_file_search_transform() {
    let result = json!({
        "query": "async patterns",
        "results": [
            {"file_id": "file_1", "filename": "async.md", "score": 0.95, "text": "..."},
            {"file_id": "file_2", "filename": "patterns.md", "score": 0.87}
        ]
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::FileSearchCall,
        "req-789",
        "server",
        "file_search",
        "{}",
    );

    match transformed {
        ResponseOutputItem::FileSearchCall {
            id,
            status,
            queries,
            results,
        } => {
            assert_eq!(id, "fs_req-789");
            assert_eq!(status, FileSearchCallStatus::Completed);
            assert_eq!(queries, vec!["async patterns"]);
            let results = results.unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].file_id, "file_1");
            assert_eq!(results[0].score, Some(0.95));
        }
        _ => panic!("Expected FileSearchCall"),
    }
}

#[test]
fn test_image_generation_transform_direct_fields() {
    let result = json!({
        "result": "BASE64_IMAGE_BYTES",
        "revised_prompt": "a serene mountain at sunrise"
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-1",
        "server",
        "image_generation",
        r#"{"prompt":"mountain"}"#,
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
            ..
        } => {
            assert_eq!(id, "ig_req-img-1");
            assert_eq!(result, "BASE64_IMAGE_BYTES");
            assert_eq!(
                revised_prompt,
                Some("a serene mountain at sunrise".to_string())
            );
            assert_eq!(status, ImageGenerationCallStatus::Completed);
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_image_generation_transform_b64_json_alias() {
    let result = json!({
        "b64_json": "PNG_BASE64",
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-2",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            result,
            revised_prompt,
            ..
        } => {
            assert_eq!(result, "PNG_BASE64");
            assert!(revised_prompt.is_none());
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_image_generation_transform_embedded_openai_response() {
    let result = json!([
        {
            "type": "text",
            "text": r#"{
                  "openai_response": {
                    "result": "EMBEDDED_BYTES",
                    "revised_prompt": "tweaked prompt"
                  }
                }"#
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-3",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            result,
            revised_prompt,
            ..
        } => {
            assert_eq!(result, "EMBEDDED_BYTES");
            assert_eq!(revised_prompt, Some("tweaked prompt".to_string()));
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_image_generation_transform_text_block_top_level_fields() {
    // MCP servers authored against the FastMCP SDK (and any server that
    // returns a plain `dict` from its tool handler) emit a single text
    // block whose JSON body has `result` / `revised_prompt` at the TOP
    // level — there is no `openai_response` wrapper. This shape is
    // equally valid per the MCP spec. The extractor must read those
    // top-level fields the same way it reads direct-object and
    // embedded-openai_response payloads.
    let result = json!([
        {
            "type": "text",
            "text": r#"{
                  "result": "TOP_LEVEL_BYTES",
                  "revised_prompt": "a happy cat",
                  "status": "completed"
                }"#
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-toplevel",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            result,
            revised_prompt,
            ..
        } => {
            assert_eq!(result, "TOP_LEVEL_BYTES");
            assert_eq!(revised_prompt, Some("a happy cat".to_string()));
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_image_generation_transform_text_block_b64_json_alias() {
    // Alias check: when a text-block payload uses `b64_json` (the OpenAI
    // Images API field name) instead of `result`, the extractor must
    // still pick it up.
    let result = json!([
        {
            "type": "text",
            "text": r#"{"b64_json": "ALIAS_BYTES"}"#
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-alias",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall { result, .. } => {
            assert_eq!(result, "ALIAS_BYTES");
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_image_generation_transform_fields_distributed_across_blocks() {
    // MCP servers may split revised_prompt and image bytes across
    // different text blocks. The extractor must accumulate fields
    // from all blocks rather than returning early on the first hit.
    let result = json!([
        {
            "type": "text",
            "text": r#"{
                  "openai_response": {
                    "revised_prompt": "distributed prompt"
                  }
                }"#
        },
        {
            "type": "text",
            "text": r#"{
                  "openai_response": {
                    "result": "DISTRIBUTED_BYTES"
                  }
                }"#
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-dist",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            result,
            revised_prompt,
            ..
        } => {
            assert_eq!(result, "DISTRIBUTED_BYTES");
            assert_eq!(revised_prompt, Some("distributed prompt".to_string()));
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_image_generation_transform_missing_image_data() {
    // When the MCP server returns neither image nor prompt, the transformer
    // still produces a well-formed ImageGenerationCall with an empty result.
    // Per-router wiring owns surfacing an error status.
    let result = json!({});

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-4",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
            ..
        } => {
            assert_eq!(id, "ig_req-img-4");
            assert!(result.is_empty());
            assert!(revised_prompt.is_none());
            assert_eq!(status, ImageGenerationCallStatus::Completed);
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

/// When the MCP tool result surfaces the OpenAI-side metadata
/// (`action`, `background`, `output_format`, `quality`, `size`) on the
/// direct object, the transformer forwards them verbatim onto the
/// `image_generation_call` output item so cloud-passthrough fidelity is
/// preserved and integration assertions can read them.
#[test]
fn test_image_generation_transform_forwards_metadata_direct_object() {
    let result = json!({
        "result": "BASE64_IMAGE_BYTES",
        "revised_prompt": "a serene mountain at sunrise",
        "action": "generate",
        "background": "opaque",
        "output_format": "png",
        "quality": "high",
        "size": "1024x1024",
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-meta",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            id,
            action,
            background,
            output_format,
            quality,
            result,
            revised_prompt,
            size,
            status,
        } => {
            assert_eq!(id, "ig_req-img-meta");
            assert_eq!(action.as_deref(), Some("generate"));
            assert_eq!(background.as_deref(), Some("opaque"));
            assert_eq!(output_format.as_deref(), Some("png"));
            assert_eq!(quality.as_deref(), Some("high"));
            assert_eq!(result, "BASE64_IMAGE_BYTES");
            assert_eq!(
                revised_prompt.as_deref(),
                Some("a serene mountain at sunrise")
            );
            assert_eq!(size.as_deref(), Some("1024x1024"));
            assert_eq!(status, ImageGenerationCallStatus::Completed);
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

/// Metadata forwarding also works from a top-level text-block payload
/// (the shape produced by FastMCP-style servers that return a plain
/// `dict`). The extractor must pick metadata out of that payload the
/// same way it already reads `result`/`revised_prompt`.
#[test]
fn test_image_generation_transform_forwards_metadata_from_text_block() {
    let result = json!([
        {
            "type": "text",
            "text": r#"{
                  "result": "TOP_LEVEL_BYTES",
                  "revised_prompt": "a happy cat",
                  "action": "generate",
                  "background": "transparent",
                  "output_format": "webp",
                  "quality": "medium",
                  "size": "1536x1024",
                  "status": "completed"
                }"#
        }
    ]);

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-meta-textblock",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            action,
            background,
            output_format,
            quality,
            size,
            ..
        } => {
            assert_eq!(action.as_deref(), Some("generate"));
            assert_eq!(background.as_deref(), Some("transparent"));
            assert_eq!(output_format.as_deref(), Some("webp"));
            assert_eq!(quality.as_deref(), Some("medium"));
            assert_eq!(size.as_deref(), Some("1536x1024"));
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

/// An MCP server that surfaces no metadata emits `None` for every
/// optional field — the transformer must not invent defaults. Guards
/// against an accidental `unwrap_or("")` that would pin an empty string
/// on the wire and round-trip as `""` instead of being skipped entirely.
#[test]
fn test_image_generation_transform_metadata_absent_when_not_surfaced() {
    let result = json!({
        "result": "BASE64_IMAGE_BYTES",
    });

    let transformed = ResponseTransformer::transform(
        &result,
        ResponseFormat::ImageGenerationCall,
        "req-img-nometa",
        "server",
        "image_generation",
        "{}",
    );

    match transformed {
        ResponseOutputItem::ImageGenerationCall {
            action,
            background,
            output_format,
            quality,
            size,
            ..
        } => {
            assert!(action.is_none());
            assert!(background.is_none());
            assert!(output_format.is_none());
            assert!(quality.is_none());
            assert!(size.is_none());
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_compact_image_generation_output_strips_base64() {
    let mut item = ResponseOutputItem::ImageGenerationCall {
        id: "ig_abc".to_string(),
        action: None,
        background: None,
        output_format: None,
        quality: None,
        result: "A_VERY_LONG_BASE64_STRING".to_string(),
        revised_prompt: Some("mountain".to_string()),
        size: None,
        status: ImageGenerationCallStatus::Completed,
    };

    compact_image_generation_output(&mut item);

    match item {
        ResponseOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
            ..
        } => {
            assert_eq!(id, "ig_abc");
            assert!(result.is_empty());
            assert_eq!(revised_prompt, Some("mountain".to_string()));
            assert_eq!(status, ImageGenerationCallStatus::Completed);
        }
        _ => panic!("Expected ImageGenerationCall"),
    }
}

#[test]
fn test_compact_image_generation_output_is_noop_for_other_items() {
    let mut item = ResponseOutputItem::WebSearchCall {
        id: "ws_xyz".to_string(),
        status: WebSearchCallStatus::Completed,
        action: WebSearchAction::Search {
            query: Some("rust".to_string()),
            queries: vec!["rust".to_string()],
            sources: vec![],
        },
        results: None,
    };

    compact_image_generation_output(&mut item);

    // WebSearchCall must be unchanged — no panic, id still present.
    match item {
        ResponseOutputItem::WebSearchCall { id, .. } => assert_eq!(id, "ws_xyz"),
        _ => panic!("Expected WebSearchCall"),
    }
}

fn failed_tool_output(tool_name: &str) -> smg_mcp::ToolExecutionOutput {
    smg_mcp::ToolExecutionOutput::new_for_test(
        "call_xyz",
        tool_name,
        "srv",
        "srv-label",
        "{\"q\":\"v\"}",
        serde_json::json!({}),
        /* is_error */ true,
        Some("upstream broke".to_string()),
        std::time::Duration::default(),
    )
}

#[test]
fn transform_tool_output_emits_mcp_call_failed_with_error_message() {
    let item = transform_tool_output(
        &failed_tool_output("brave_web_search"),
        ResponseFormat::Passthrough,
    );
    match item {
        ResponseOutputItem::McpCall {
            status,
            error,
            output,
            arguments,
            name,
            ..
        } => {
            assert_eq!(status, "failed");
            assert_eq!(error, Some(McpToolCallError::execution("upstream broke")));
            assert_eq!(output, "");
            assert_eq!(arguments, "{\"q\":\"v\"}");
            assert_eq!(name, "brave_web_search");
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn transform_tool_output_falls_back_to_output_text_when_error_message_missing() {
    // MCP `CallToolResult { is_error: true }` can leave error_message
    // unset and put the failure text inside the result blocks.
    let mut output = failed_tool_output("brave_web_search");
    output.error_message = None;
    output.output = serde_json::json!([
        {"type": "text", "text": "rate limited"}
    ]);

    let item = transform_tool_output(&output, ResponseFormat::Passthrough);
    match item {
        ResponseOutputItem::McpCall { error, .. } => {
            assert_eq!(error, Some(McpToolCallError::execution("rate limited")));
        }
        _ => panic!("Expected McpCall"),
    }
}

#[test]
fn transform_tool_output_emits_hosted_completed_even_on_is_error() {
    // Hosted-builtin variants always emit status=completed regardless of
    // upstream is_error. Real OpenAI cloud doesn't emit Failed for soft
    // failures (rate-limited search, no results, etc.) and many MCP
    // servers set isError=true for those exact cases — propagating it
    // would break clients that expect cloud-parity wire shape. The
    // failure context lives in the item content, not in `status`.
    for fmt in [
        ResponseFormat::WebSearchCall,
        ResponseFormat::CodeInterpreterCall,
        ResponseFormat::FileSearchCall,
        ResponseFormat::ImageGenerationCall,
    ] {
        let item = transform_tool_output(&failed_tool_output("search"), fmt);
        let serialized = serde_json::to_value(&item).expect("serialize failed item");
        assert_eq!(
            serialized["status"],
            serde_json::json!("completed"),
            "{fmt:?}"
        );
    }
}

#[test]
fn test_compact_image_generation_outputs_json_strips_base64() {
    let mut outputs = vec![
        serde_json::json!({
            "type": "image_generation_call",
            "id": "ig_1",
            "result": "AAAA_LONG_BASE64",
            "revised_prompt": "cat",
            "status": "completed"
        }),
        // Untouched: not an image_generation_call.
        serde_json::json!({
            "type": "mcp_call",
            "id": "mcp_1",
            "output": "raw text",
        }),
        // Image item with no `result` field: must not panic.
        serde_json::json!({
            "type": "image_generation_call",
            "id": "ig_2",
            "status": "in_progress"
        }),
    ];

    compact_image_generation_outputs_json(&mut outputs);

    assert!(outputs[0].get("result").is_none());
    assert_eq!(outputs[0]["revised_prompt"], serde_json::json!("cat"));
    assert_eq!(outputs[0]["status"], serde_json::json!("completed"));
    assert_eq!(outputs[1]["output"], serde_json::json!("raw text"));
    assert!(outputs[2].get("result").is_none());
}
