//! Protocol-surface roundtrip tests for the OpenAI Responses API types.

use openai_protocol::{
    common::{
        ContextManagementType, Detail, PromptCacheRetention, ToolChoice as ChatToolChoice,
        ToolChoiceValue as ChatToolChoiceValue, ToolReference,
    },
    responses::*,
};
use serde_json::json;
use validator::Validate;

#[test]
fn summary_text_content_round_trips_spec_shape() {
    // Spec: `summary: array of SummaryTextContent { text, type: "summary_text" }`.
    let item: ResponseOutputItem = serde_json::from_value(json!({
        "type": "reasoning",
        "id": "r_1",
        "summary": [{"text": "step 1", "type": "summary_text"}],
        "content": [],
        "status": "completed",
    }))
    .expect("spec-shape reasoning should deserialize");

    match &item {
        ResponseOutputItem::Reasoning { summary, .. } => {
            assert_eq!(summary.len(), 1);
            match &summary[0] {
                SummaryTextContent::SummaryText { text } => {
                    assert_eq!(text, "step 1");
                }
            }
        }
        _ => panic!("expected Reasoning variant"),
    }

    let v = serde_json::to_value(&item).expect("serialize");
    assert_eq!(
        v["summary"],
        json!([{"text": "step 1", "type": "summary_text"}])
    );
}

#[test]
fn reasoning_items_preserve_empty_summary_arrays() {
    let output_payload = json!({
        "type": "reasoning",
        "id": "r_empty",
        "summary": [],
        "content": [],
        "status": "completed",
    });
    let output_item: ResponseOutputItem = serde_json::from_value(output_payload.clone())
        .expect("output reasoning item with empty summary should deserialize");
    let serialized = serde_json::to_value(&output_item).expect("serialize output reasoning");
    assert_eq!(serialized, output_payload);

    let input_payload = json!({
        "type": "reasoning",
        "id": "r_input_empty",
        "summary": []
    });
    let input_item: ResponseInputOutputItem = serde_json::from_value(input_payload.clone())
        .expect("input reasoning item with empty summary should deserialize");
    let serialized = serde_json::to_value(&input_item).expect("serialize input reasoning");
    assert_eq!(serialized, input_payload);
}

#[test]
fn legacy_vec_string_summary_fails_to_deserialize() {
    let legacy = r#"{"type":"reasoning","id":"r_x","summary":["text"]}"#;
    let result: Result<ResponseInputOutputItem, _> = serde_json::from_str(legacy);
    assert!(
        result.is_err(),
        "legacy Vec<String> summary must no longer deserialize"
    );
}

#[test]
fn test_responses_request_omitted_top_p_deserializes_to_none() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello"
    }))
    .expect("request should deserialize");

    assert_eq!(request.top_p, None);

    let serialized = serde_json::to_value(&request).expect("request should serialize");
    assert!(serialized.get("top_p").is_none());
}

#[test]
fn test_responses_request_null_top_p_deserializes_to_none() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "top_p": null
    }))
    .expect("request should deserialize");

    assert_eq!(request.top_p, None);

    let serialized = serde_json::to_value(&request).expect("request should serialize");
    assert!(serialized.get("top_p").is_none());
}

#[test]
fn test_responses_request_explicit_top_p_preserved() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "top_p": 0.9
    }))
    .expect("request should deserialize");

    assert_eq!(request.top_p, Some(0.9));
}

#[test]
fn test_require_approval_string_round_trip() {
    let tool: McpTool = serde_json::from_value(json!({
        "server_label": "deepwiki",
        "require_approval": "never"
    }))
    .expect("mcp tool should deserialize");

    assert_eq!(
        tool.require_approval,
        Some(RequireApproval::Mode(RequireApprovalMode::Never))
    );

    let serialized = serde_json::to_value(&tool).expect("mcp tool should serialize");
    assert_eq!(serialized["require_approval"], json!("never"));
}

#[test]
fn test_require_approval_object_round_trip() {
    let tool: McpTool = serde_json::from_value(json!({
        "server_label": "deepwiki",
        "require_approval": {
            "always": null,
            "never": {
                "tool_names": ["ask_question", "read_wiki_structure"],
                "read_only": null
            }
        }
    }))
    .expect("mcp tool should deserialize");

    assert_eq!(
        tool.require_approval,
        Some(RequireApproval::Rules(RequireApprovalRules {
            always: None,
            never: Some(RequireApprovalFilter {
                tool_names: Some(vec![
                    "ask_question".to_string(),
                    "read_wiki_structure".to_string()
                ]),
                read_only: None,
            }),
        }))
    );

    let serialized = serde_json::to_value(&tool).expect("mcp tool should serialize");
    assert_eq!(
        serialized["require_approval"],
        json!({
            "never": {
                "tool_names": ["ask_question", "read_wiki_structure"]
            }
        })
    );
}

#[test]
fn test_include_field_web_search_call_variants_round_trip() {
    let fields: Vec<IncludeField> = serde_json::from_value(json!([
        "web_search_call.action.sources",
        "web_search_call.results",
    ]))
    .expect("include fields should deserialize");

    assert_eq!(
        fields,
        vec![
            IncludeField::WebSearchCallActionSources,
            IncludeField::WebSearchCallResults,
        ]
    );

    let serialized = serde_json::to_value(&fields).expect("include fields should serialize");
    assert_eq!(
        serialized,
        json!(["web_search_call.action.sources", "web_search_call.results"])
    );
}

#[test]
fn test_file_search_tool_round_trip() {
    // Spec fixture (openai-responses-api-spec.md §tools): `file_search` with
    // vector_store_ids, compound filter, and ranking_options incl. hybrid_search,
    // ranker enum, and score_threshold.
    let payload = json!({
        "type": "file_search",
        "vector_store_ids": ["vs_1234567890"],
        "max_num_results": 20,
        "filters": {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "region", "value": "us-east-1"},
                {"type": "in", "key": "tag", "value": ["alpha", "beta"]}
            ]
        },
        "ranking_options": {
            "ranker": "default-2024-11-15",
            "score_threshold": 0.5,
            "hybrid_search": {"embedding_weight": 0.7, "text_weight": 0.3}
        }
    });

    let tool: ResponseTool =
        serde_json::from_value(payload.clone()).expect("file_search tool should deserialize");
    assert!(matches!(tool, ResponseTool::FileSearch(_)));

    let serialized = serde_json::to_value(&tool).expect("file_search tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_file_search_tool_round_trip_hybrid_weights_omitted() {
    // Spec (openai-responses-api-spec.md §tools): `embedding_weight` and
    // `text_weight` are optional. When omitted they must deserialize to
    // `None` and be absent again on re-serialization (absent→None→absent).
    let payload = json!({
        "type": "file_search",
        "vector_store_ids": ["vs_1234567890"],
        "ranking_options": {
            "hybrid_search": {}
        }
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("file_search tool with empty hybrid_search should deserialize");
    match &tool {
        ResponseTool::FileSearch(fs) => {
            let ranking = fs
                .ranking_options
                .as_ref()
                .expect("ranking_options should be present");
            let hybrid = ranking
                .hybrid_search
                .as_ref()
                .expect("hybrid_search should be present");
            assert!(hybrid.embedding_weight.is_none());
            assert!(hybrid.text_weight.is_none());
        }
        other => panic!("expected FileSearch variant, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("file_search tool should serialize");
    assert_eq!(serialized, payload);
}

// ------------------------------------------------------------------
// T3: non-preview web_search tool + WebSearchCall.results
// ------------------------------------------------------------------

/// Spec fixture (openai-responses-api-spec.md §tools line 439):
/// `{ type: "web_search" | "web_search_2025_08_26", filters? { allowed_domains? },
/// return_token_budget?: "default"|"unlimited", search_context_size?: "low"|"medium"|"high",
/// user_location? }`. Covers the canonical tag, the versioned alias, and full-field nested shape.
#[test]
fn test_web_search_tool_round_trip() {
    let payload = json!({
        "type": "web_search",
        "filters": {"allowed_domains": ["example.com", "rust-lang.org"]},
        "return_token_budget": "default",
        "search_context_size": "high",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "country": "US",
            "region": "California",
            "timezone": "America/Los_Angeles"
        }
    });
    let tool: ResponseTool =
        serde_json::from_value(payload.clone()).expect("web_search tool should deserialize");
    assert!(matches!(tool, ResponseTool::WebSearch(_)));
    assert_eq!(
        serde_json::to_value(&tool).expect("web_search tool should serialize"),
        payload
    );

    // Versioned alias deserializes into the same variant (canonical serialization re-tested above).
    let alias: ResponseTool = serde_json::from_value(json!({"type": "web_search_2025_08_26"}))
        .expect("web_search_2025_08_26 alias should deserialize");
    assert!(matches!(alias, ResponseTool::WebSearch(_)));
}

#[test]
fn test_web_search_response_tool_echo_accepts_return_token_budget() {
    let payload = json!({
        "id": "resp_web_echo",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": null,
        "model": "gpt-5.4",
        "output": [],
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "reasoning": null,
        "store": true,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [{
            "type": "web_search",
            "return_token_budget": "default",
            "search_context_size": "low",
            "user_location": {
                "type": "approximate",
                "city": null,
                "country": null,
                "region": null,
                "timezone": null
            }
        }],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": null,
        "user": null,
        "metadata": {}
    });

    let response: ResponsesResponse = serde_json::from_value(payload.clone())
        .expect("upstream web_search echo with return_token_budget should deserialize");
    let serialized = serde_json::to_value(&response).expect("serialize response");
    assert_eq!(
        serialized["tools"][0]["return_token_budget"],
        json!("default")
    );
}

/// Acceptance: `web_search_call` output item carries an optional typed
/// `results` field populated when callers request `web_search_call.results`
/// via the top-level `include[]` array. When absent, the default wire shape
/// `{id, action, status, type}` must stay spec-byte-identical.
#[test]
fn test_web_search_call_results_round_trip() {
    let with_results = json!({
        "type": "web_search_call",
        "id": "ws_abc",
        "status": "completed",
        "action": {"type": "search", "query": "rust", "queries": ["rust"]},
        "results": [
            {"url": "https://tokio.rs", "title": "Tokio", "snippet": "rt", "score": 0.5},
            {"url": "https://async.rs"}
        ]
    });
    let item: ResponseOutputItem = serde_json::from_value(with_results.clone())
        .expect("web_search_call with results should deserialize");
    let ResponseOutputItem::WebSearchCall { results, .. } = &item else {
        panic!("expected WebSearchCall");
    };
    let results = results.as_ref().expect("results present");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].score, Some(0.5));
    assert!(results[1].title.is_none());
    assert_eq!(
        serde_json::to_value(&item).expect("web_search_call should serialize"),
        with_results
    );

    // Absent results: deserializes to None and re-serializes without the key.
    let no_results = json!({
        "type": "web_search_call",
        "id": "ws_no",
        "status": "completed",
        "action": {"type": "search"}
    });
    let bare: ResponseOutputItem = serde_json::from_value(no_results.clone())
        .expect("web_search_call without results should deserialize");
    let ResponseOutputItem::WebSearchCall { results, .. } = &bare else {
        panic!("expected WebSearchCall");
    };
    assert!(results.is_none());
    let serialized = serde_json::to_value(&bare).expect("web_search_call should serialize");
    assert_eq!(serialized, no_results);
    assert!(serialized.get("results").is_none());
}

// ------------------------------------------------------------------
// P2: new top-level ResponsesRequest fields
// ------------------------------------------------------------------

/// Acceptance: the six new top-level fields deserialize and re-serialize
/// without loss (`prompt`, `prompt_cache_key`, `prompt_cache_retention`,
/// `safety_identifier`, `stream_options`, `context_management`).
#[test]
fn test_responses_request_new_top_level_fields_round_trip() {
    let payload = json!({
        "model": "gpt-5.4",
        "input": "hello",
        "prompt": {
            "id": "pmpt_abc",
            "variables": {
                "name": "ada",
                "picture": {
                    "type": "input_image",
                    "image_url": "https://example.com/pic.png",
                    "detail": "high"
                }
            },
            "version": "1"
        },
        "prompt_cache_key": "pck-123",
        "prompt_cache_retention": "24h",
        "safety_identifier": "sid-123",
        "stream_options": { "include_obfuscation": false },
        "context_management": [{
            "type": "compaction",
            "compact_threshold": 4096
        }]
    });

    let request: ResponsesRequest =
        serde_json::from_value(payload.clone()).expect("request should deserialize");

    assert!(request.prompt.is_some());
    assert_eq!(
        request.prompt.as_ref().map(|p| p.id.as_str()),
        Some("pmpt_abc")
    );
    assert_eq!(
        request.prompt.as_ref().and_then(|p| p.version.as_deref()),
        Some("1")
    );
    assert_eq!(request.prompt_cache_key.as_deref(), Some("pck-123"));
    assert_eq!(
        request.prompt_cache_retention,
        Some(PromptCacheRetention::Duration24h)
    );
    assert_eq!(request.safety_identifier.as_deref(), Some("sid-123"));
    assert_eq!(
        request
            .stream_options
            .as_ref()
            .and_then(|s| s.include_obfuscation),
        Some(false)
    );
    let ctx = request
        .context_management
        .as_ref()
        .expect("context_management must round-trip");
    assert_eq!(ctx.len(), 1);
    assert_eq!(ctx[0].r#type, ContextManagementType::Compaction);
    assert_eq!(ctx[0].compact_threshold, Some(4096));

    // Re-serialize and confirm the wire form matches the inputs.
    let reserialized = serde_json::to_value(&request).expect("should serialize");
    assert_eq!(reserialized["prompt"]["id"], "pmpt_abc");
    assert_eq!(reserialized["prompt_cache_key"], "pck-123");
    assert_eq!(reserialized["prompt_cache_retention"], "24h");
    assert_eq!(reserialized["safety_identifier"], "sid-123");
    assert_eq!(reserialized["stream_options"]["include_obfuscation"], false);
    assert_eq!(reserialized["context_management"][0]["type"], "compaction");
    assert_eq!(
        reserialized["context_management"][0]["compact_threshold"],
        4096
    );
}

/// `prompt_cache_retention` accepts the other spec value and serializes
/// back with the hyphenated rename.
#[test]
fn test_prompt_cache_retention_in_memory_round_trip() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello",
        "prompt_cache_retention": "in-memory"
    }))
    .expect("should deserialize");

    assert_eq!(
        request.prompt_cache_retention,
        Some(PromptCacheRetention::InMemory)
    );

    let reserialized = serde_json::to_value(&request).expect("should serialize");
    assert_eq!(reserialized["prompt_cache_retention"], "in-memory");
}

/// Absent fields must stay absent on the wire (no `"prompt": null` etc.),
/// matching every other `Option<_>` field on `ResponsesRequest`.
#[test]
fn test_responses_request_new_fields_omitted_when_absent() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hello"
    }))
    .expect("should deserialize");

    assert!(request.prompt.is_none());
    assert!(request.prompt_cache_key.is_none());
    assert!(request.prompt_cache_retention.is_none());
    assert!(request.safety_identifier.is_none());
    assert!(request.stream_options.is_none());
    assert!(request.context_management.is_none());

    let serialized = serde_json::to_value(&request).expect("should serialize");
    for key in [
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "stream_options",
        "context_management",
    ] {
        assert!(
            serialized.get(key).is_none(),
            "field {key} should be skipped when absent"
        );
    }
}

// ---- P1: rich content parts + typed annotations ----

#[test]
fn content_part_input_image_roundtrip() {
    // `original` is the spec-new detail level; exercising it here
    // also covers the rest of the variant shape.
    let raw = json!({
        "type": "input_image",
        "detail": "original",
        "image_url": "https://example.com/cat.png"
    });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("input_image should deserialize");
    match &part {
        ResponseContentPart::InputImage {
            detail,
            file_id,
            image_url,
        } => {
            assert!(matches!(detail, Some(Detail::Original)));
            assert!(file_id.is_none());
            assert_eq!(image_url.as_deref(), Some("https://example.com/cat.png"));
        }
        other => panic!("expected InputImage, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_input_file_roundtrip() {
    let raw = json!({
        "type": "input_file",
        "detail": "low",
        "file_data": "BASE64DATA",
        "filename": "report.pdf"
    });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("input_file should deserialize");
    match &part {
        ResponseContentPart::InputFile {
            detail,
            file_data,
            file_id,
            file_url,
            filename,
        } => {
            assert!(matches!(detail, Some(FileDetail::Low)));
            assert_eq!(file_data.as_deref(), Some("BASE64DATA"));
            assert!(file_id.is_none());
            assert!(file_url.is_none());
            assert_eq!(filename.as_deref(), Some("report.pdf"));
        }
        other => panic!("expected InputFile, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_refusal_roundtrip() {
    let raw = json!({ "type": "refusal", "refusal": "I cannot help with that." });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("refusal should deserialize");
    match &part {
        ResponseContentPart::Refusal { refusal } => {
            assert_eq!(refusal, "I cannot help with that.");
        }
        other => panic!("expected Refusal, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_output_text_with_typed_annotations() {
    let raw = json!({
        "type": "output_text",
        "text": "See [1] and [2].",
        "annotations": [
            {
                "type": "file_citation",
                "file_id": "file-abc",
                "filename": "source.pdf",
                "index": 5
            },
            {
                "type": "url_citation",
                "url": "https://example.com",
                "title": "Example",
                "start_index": 0,
                "end_index": 18
            },
            {
                "type": "container_file_citation",
                "container_id": "cntr-1",
                "file_id": "file-xyz",
                "filename": "out.txt",
                "start_index": 0,
                "end_index": 4
            },
            {
                "type": "file_path",
                "file_id": "file-gen",
                "index": 7
            }
        ]
    });
    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("output_text should deserialize");
    match &part {
        ResponseContentPart::OutputText {
            text,
            annotations,
            logprobs,
        } => {
            assert_eq!(text, "See [1] and [2].");
            assert_eq!(annotations.len(), 4);
            assert!(logprobs.is_none());
            assert!(matches!(&annotations[0], Annotation::FileCitation { .. }));
            assert!(matches!(&annotations[1], Annotation::UrlCitation { .. }));
            assert!(matches!(
                &annotations[2],
                Annotation::ContainerFileCitation { .. }
            ));
            assert!(matches!(&annotations[3], Annotation::FilePath { .. }));
        }
        other => panic!("expected OutputText, got {other:?}"),
    }
    let roundtripped = serde_json::to_value(&part).unwrap();
    assert_eq!(roundtripped, raw);
}

#[test]
fn content_part_output_text_preserves_empty_annotations() {
    let raw = json!({
        "type": "output_text",
        "text": "No citations here.",
        "annotations": []
    });

    let part: ResponseContentPart =
        serde_json::from_value(raw.clone()).expect("output_text should deserialize");
    match &part {
        ResponseContentPart::OutputText { annotations, .. } => {
            assert!(annotations.is_empty());
        }
        other => panic!("expected OutputText, got {other:?}"),
    }

    let roundtripped = serde_json::to_value(&part).expect("output_text should serialize");
    assert_eq!(roundtripped, raw);
    assert_eq!(roundtripped["annotations"], json!([]));
}

#[test]
fn function_call_items_accept_missing_optional_id() {
    let input_payload = json!({
        "type": "function_call",
        "call_id": "call_123",
        "name": "lookup",
        "arguments": "{\"q\":\"rust\"}"
    });
    let input_item: ResponseInputOutputItem = serde_json::from_value(input_payload.clone())
        .expect("function_call input item without id should deserialize");
    match &input_item {
        ResponseInputOutputItem::FunctionToolCall { id, .. } => assert!(id.is_none()),
        other => panic!("expected FunctionToolCall, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&input_item).expect("serialize input function_call"),
        input_payload
    );

    let output_payload = json!({
        "type": "function_call",
        "call_id": "call_123",
        "name": "lookup",
        "arguments": "{\"q\":\"rust\"}",
        "status": "completed"
    });
    let output_item: ResponseOutputItem = serde_json::from_value(output_payload.clone())
        .expect("function_call output item without id should deserialize");
    match &output_item {
        ResponseOutputItem::FunctionToolCall { id, output, .. } => {
            assert!(id.is_none());
            assert!(output.is_none());
        }
        other => panic!("expected FunctionToolCall, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&output_item).expect("serialize output function_call"),
        output_payload
    );

    let function_output_payload = json!({
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "done"
    });
    let function_output_item: ResponseInputOutputItem =
        serde_json::from_value(function_output_payload.clone())
            .expect("function_call_output input item without id should deserialize");
    match &function_output_item {
        ResponseInputOutputItem::FunctionCallOutput { id, .. } => assert!(id.is_none()),
        other => panic!("expected FunctionCallOutput, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&function_output_item).expect("serialize input function_call_output"),
        function_output_payload
    );
}

#[test]
fn content_part_unknown_type_fails_fast() {
    // Previously `#[serde(other)] Unknown` silently swallowed unknown
    // types; P1 removes that arm so spec-invalid payloads fail cleanly.
    let raw = json!({ "type": "totally_made_up", "text": "x" });
    let err = serde_json::from_value::<ResponseContentPart>(raw)
        .expect_err("unknown content-part type must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("totally_made_up") || msg.contains("unknown variant"),
        "error should mention the unknown variant, got: {msg}"
    );
}

// ------------------------------------------------------------------
// P5: ResponseInputOutputItem fail-fast on unknown `type` discriminator
//
// `ResponseInputOutputItem` is `#[serde(tag = "type")]` with a single
// `#[serde(untagged)] SimpleInputMessage` fallback. Before P5, that
// fallback accepted any `{role, content, type: "<anything>"}` payload
// because `type` was `Option<String>`. The spec (EasyInputMessage.type)
// constrains `type` to absent or `"message"`, so P5 replaces the loose
// `Option<String>` with `Option<SimpleInputMessageTypeTag>` — unknown
// discriminators must now return a deserialize error (→ 400).
// ------------------------------------------------------------------

#[test]
fn input_item_unknown_type_fails_fast() {
    // Spec-invalid discriminator must NOT silently fall into
    // SimpleInputMessage. Previously `type: "totally_made_up"` deserialized
    // cleanly because `r#type` was `Option<String>`.
    let raw = json!({
        "type": "totally_made_up",
        "role": "user",
        "content": "hello"
    });
    let err = serde_json::from_value::<ResponseInputOutputItem>(raw)
        .expect_err("unknown input-item type must fail");
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "deserialize error must carry a message, got empty: {msg}"
    );
}

#[test]
fn simple_input_message_without_type_still_deserializes() {
    // Per spec, `EasyInputMessage.type` is optional. Omitting it must
    // continue to route to the untagged SimpleInputMessage variant.
    let raw = json!({
        "role": "user",
        "content": "hello"
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(raw).expect("missing type must still deserialize");
    match item {
        ResponseInputOutputItem::SimpleInputMessage {
            role,
            content,
            r#type,
            ..
        } => {
            assert_eq!(role, "user");
            assert!(matches!(content, StringOrContentParts::String(ref s) if s == "hello"));
            assert!(r#type.is_none(), "absent type must deserialize as None");
        }
        other => panic!("expected SimpleInputMessage, got {other:?}"),
    }
}

#[test]
fn simple_input_message_with_type_message_deserializes() {
    // Per spec, `type: "message"` is the only non-null value permitted on
    // EasyInputMessage. Round-trip it intact.
    let raw = json!({
        "type": "message",
        "role": "user",
        "content": "hi"
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(raw.clone()).expect("`type: message` must deserialize");
    match &item {
        ResponseInputOutputItem::SimpleInputMessage { r#type, .. } => {
            assert_eq!(*r#type, Some(SimpleInputMessageTypeTag::Message));
        }
        other => panic!("expected SimpleInputMessage, got {other:?}"),
    }
    // Serializes back to the same wire bytes (canonical shape).
    let roundtripped = serde_json::to_value(&item).expect("serialize");
    assert_eq!(roundtripped, raw);
}

// ------------------------------------------------------------------
// P7: ResponsesToolChoice round-trip tests (all 8 spec variants)
// ------------------------------------------------------------------

/// Options: `"none"` | `"auto"` | `"required"` deserialize as bare strings
/// and re-serialize identically.
#[test]
fn responses_tool_choice_options_round_trip() {
    for (payload, expected) in [
        (json!("none"), ToolChoiceOptions::None),
        (json!("auto"), ToolChoiceOptions::Auto),
        (json!("required"), ToolChoiceOptions::Required),
    ] {
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("options string should deserialize");
        match &choice {
            ResponsesToolChoice::Options(opt) => assert_eq!(opt, &expected),
            other => panic!("expected Options, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }
}

/// Types: `{"type": "<built-in>"}` for every hosted tool the spec enumerates.
#[test]
fn responses_tool_choice_types_round_trip() {
    for (payload, expected) in [
        (
            json!({"type": "file_search"}),
            BuiltInToolChoiceType::FileSearch,
        ),
        (
            json!({"type": "web_search"}),
            BuiltInToolChoiceType::WebSearch,
        ),
        (
            json!({"type": "web_search_preview"}),
            BuiltInToolChoiceType::WebSearchPreview,
        ),
        (
            json!({"type": "web_search_preview_2025_03_11"}),
            BuiltInToolChoiceType::WebSearchPreview20250311,
        ),
        (
            json!({"type": "image_generation"}),
            BuiltInToolChoiceType::ImageGeneration,
        ),
        (
            json!({"type": "computer_use_preview"}),
            BuiltInToolChoiceType::ComputerUsePreview,
        ),
        (
            json!({"type": "code_interpreter"}),
            BuiltInToolChoiceType::CodeInterpreter,
        ),
    ] {
        let choice: ResponsesToolChoice =
            serde_json::from_value(payload.clone()).expect("types object should deserialize");
        match &choice {
            ResponsesToolChoice::Types { tool_type } => assert_eq!(tool_type, &expected),
            other => panic!("expected Types, got {other:?}"),
        }
        let reserialized = serde_json::to_value(&choice).expect("serialize");
        assert_eq!(reserialized, payload);
    }
}

/// Function: `{"type": "function", "name": "..."}` — spec-canonical flat
/// shape round-trips unchanged.
#[test]
fn responses_tool_choice_function_round_trip() {
    let payload = json!({"type": "function", "name": "get_weather"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("function flat shape should deserialize");
    match &choice {
        ResponsesToolChoice::Function(payload) => assert_eq!(payload.name, "get_weather"),
        other => panic!("expected Function, got {other:?}"),
    }
    assert_eq!(choice.function_name(), Some("get_weather"));
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Backward compat: the Chat-style nested `{"function": {"name": ...}}`
/// wire shape was accepted on the Responses endpoint before the P7
/// split and existing smg clients + e2e tests rely on it. We accept it
/// on deserialize (Postel: liberal on input) but always emit the
/// canonical flat shape on serialize (conservative on output).
#[test]
fn responses_tool_choice_function_accepts_legacy_nested() {
    let legacy_nested = json!({
        "type": "function",
        "function": {"name": "search_web"}
    });
    let choice: ResponsesToolChoice = serde_json::from_value(legacy_nested)
        .expect("legacy nested function shape must deserialize for backward compat");

    // Internal state is normalized to the flat `name` field regardless of
    // which wire shape we read.
    match &choice {
        ResponsesToolChoice::Function(payload) => {
            assert_eq!(payload.name, "search_web");
            assert_eq!(payload.tool_type, FunctionToolChoiceTag::Function);
        }
        other => panic!("expected Function, got {other:?}"),
    }
    assert_eq!(choice.function_name(), Some("search_web"));

    // Serialize MUST emit the canonical flat shape, not the nested legacy
    // shape that came in on the wire.
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(
        reserialized,
        json!({"type": "function", "name": "search_web"}),
        "Responses `Function` must serialize as canonical flat even when \
         deserialized from the legacy nested shape"
    );
    assert!(
        reserialized.get("function").is_none(),
        "canonical flat serialize must not emit a nested `function` object"
    );
}

/// Payloads without either `name` or `function.name` must fail —
/// we accept both wire shapes, but at least one of them must carry the
/// function name. (The exact error string comes from serde's untagged
/// enum routing which swallows inner errors, so we assert only that
/// deserialization fails.)
#[test]
fn responses_tool_choice_function_rejects_missing_name() {
    let payload = json!({"type": "function"});
    assert!(
        serde_json::from_value::<ResponsesToolChoice>(payload).is_err(),
        "function without `name` or `function.name` must be rejected",
    );
}

/// AllowedTools: `{"type": "allowed_tools", "mode", "tools"}` with Function
/// references; validates ToolReference interop.
#[test]
fn responses_tool_choice_allowed_tools_round_trip() {
    let payload = json!({
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "get_weather"}]
    });
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("allowed_tools should deserialize");
    match &choice {
        ResponsesToolChoice::AllowedTools { mode, tools, .. } => {
            assert_eq!(mode, "auto");
            assert_eq!(tools.len(), 1);
            match &tools[0] {
                ToolReference::Function { name } => assert_eq!(name, "get_weather"),
                other => panic!("expected Function reference, got {other:?}"),
            }
        }
        other => panic!("expected AllowedTools, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Mcp: `{"type": "mcp", "server_label", "name"?}` with and without the
/// optional `name`.
#[test]
fn responses_tool_choice_mcp_round_trip() {
    // With `name`
    let payload_with_name = json!({
        "type": "mcp",
        "server_label": "my_server",
        "name": "search"
    });
    let choice: ResponsesToolChoice = serde_json::from_value(payload_with_name.clone())
        .expect("mcp with name should deserialize");
    match &choice {
        ResponsesToolChoice::Mcp {
            server_label, name, ..
        } => {
            assert_eq!(server_label, "my_server");
            assert_eq!(name.as_deref(), Some("search"));
        }
        other => panic!("expected Mcp, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload_with_name);

    // Without `name`
    let payload_no_name = json!({
        "type": "mcp",
        "server_label": "my_server"
    });
    let choice: ResponsesToolChoice = serde_json::from_value(payload_no_name.clone())
        .expect("mcp without name should deserialize");
    match &choice {
        ResponsesToolChoice::Mcp {
            server_label, name, ..
        } => {
            assert_eq!(server_label, "my_server");
            assert!(name.is_none());
        }
        other => panic!("expected Mcp, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload_no_name);
}

/// Custom: `{"type": "custom", "name": "..."}` — user-registered tool by name.
#[test]
fn responses_tool_choice_custom_round_trip() {
    let payload = json!({"type": "custom", "name": "my_custom_tool"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("custom should deserialize");
    match &choice {
        ResponsesToolChoice::Custom { name, .. } => assert_eq!(name, "my_custom_tool"),
        other => panic!("expected Custom, got {other:?}"),
    }
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// ApplyPatch: `{"type": "apply_patch"}`.
#[test]
fn responses_tool_choice_apply_patch_round_trip() {
    let payload = json!({"type": "apply_patch"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("apply_patch should deserialize");
    assert!(matches!(choice, ResponsesToolChoice::ApplyPatch { .. }));
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Shell: `{"type": "shell"}`.
#[test]
fn responses_tool_choice_shell_round_trip() {
    let payload = json!({"type": "shell"});
    let choice: ResponsesToolChoice =
        serde_json::from_value(payload.clone()).expect("shell should deserialize");
    assert!(matches!(choice, ResponsesToolChoice::Shell { .. }));
    let reserialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(reserialized, payload);
}

/// Projection onto Chat Completions' ToolChoice enum.
/// - Options map to the same Value variants.
/// - Function flattens to Chat's nested `{function: {name}}` wire form.
/// - AllowedTools preserves mode and tool list.
/// - Hosted / custom / apply_patch / shell / mcp have no Chat equivalent
///   and fall back to `auto`.
#[test]
fn responses_tool_choice_to_chat_projection() {
    assert!(matches!(
        ResponsesToolChoice::Options(ToolChoiceOptions::None).to_chat_tool_choice(),
        ChatToolChoice::Value(ChatToolChoiceValue::None)
    ));
    assert!(matches!(
        ResponsesToolChoice::Options(ToolChoiceOptions::Auto).to_chat_tool_choice(),
        ChatToolChoice::Value(ChatToolChoiceValue::Auto)
    ));
    assert!(matches!(
        ResponsesToolChoice::Options(ToolChoiceOptions::Required).to_chat_tool_choice(),
        ChatToolChoice::Value(ChatToolChoiceValue::Required)
    ));

    let fn_choice = ResponsesToolChoice::Function(ResponsesFunctionToolChoice {
        tool_type: FunctionToolChoiceTag::Function,
        name: "get_weather".to_string(),
    });
    match fn_choice.to_chat_tool_choice() {
        ChatToolChoice::Function {
            tool_type,
            function,
        } => {
            assert_eq!(tool_type, "function");
            assert_eq!(function.name, "get_weather");
        }
        other => panic!("expected Function, got {other:?}"),
    }

    let allowed = ResponsesToolChoice::AllowedTools {
        tool_type: AllowedToolsToolChoiceTag::AllowedTools,
        mode: "auto".to_string(),
        tools: vec![ToolReference::Function {
            name: "get_weather".to_string(),
        }],
    };
    match allowed.to_chat_tool_choice() {
        ChatToolChoice::AllowedTools {
            tool_type,
            mode,
            tools,
        } => {
            assert_eq!(tool_type, "allowed_tools");
            assert_eq!(mode, "auto");
            assert_eq!(tools.len(), 1);
        }
        other => panic!("expected AllowedTools, got {other:?}"),
    }

    // Responses-only variants collapse onto Chat's `auto` — there is no
    // spec-valid Chat projection for hosted / mcp / custom / apply_patch / shell.
    for variant in [
        ResponsesToolChoice::Types {
            tool_type: BuiltInToolChoiceType::FileSearch,
        },
        ResponsesToolChoice::Mcp {
            tool_type: McpToolChoiceTag::Mcp,
            server_label: "s".into(),
            name: None,
        },
        ResponsesToolChoice::Custom {
            tool_type: CustomToolChoiceTag::Custom,
            name: "c".into(),
        },
        ResponsesToolChoice::ApplyPatch {
            tool_type: ApplyPatchToolChoiceTag::ApplyPatch,
        },
        ResponsesToolChoice::Shell {
            tool_type: ShellToolChoiceTag::Shell,
        },
    ] {
        assert!(matches!(
            variant.to_chat_tool_choice(),
            ChatToolChoice::Value(ChatToolChoiceValue::Auto)
        ));
    }
}

/// The Default impl is `Options(Auto)`.
#[test]
fn responses_tool_choice_default_is_auto() {
    assert!(matches!(
        ResponsesToolChoice::default(),
        ResponsesToolChoice::Options(ToolChoiceOptions::Auto)
    ));
}

// ------------------------------------------------------------------
// T4: image_generation tool + image_generation_call item round-trips
// ------------------------------------------------------------------

/// `ResponseTool::ImageGeneration` — bare `{type}` and full spec payload
/// round-trip with every optional field preserved byte-for-byte.
#[test]
fn image_generation_tool_round_trips_spec_shape() {
    // Minimal spec shape: just the discriminator, no config.
    let minimal = json!({"type": "image_generation"});
    let tool: ResponseTool =
        serde_json::from_value(minimal.clone()).expect("minimal image_generation deserialize");
    match &tool {
        ResponseTool::ImageGeneration(cfg) => {
            assert!(cfg.action.is_none());
            assert!(cfg.background.is_none());
            assert!(cfg.input_fidelity.is_none());
            assert!(cfg.input_image_mask.is_none());
            assert!(cfg.model.is_none());
            assert!(cfg.moderation.is_none());
            assert!(cfg.output_compression.is_none());
            assert!(cfg.output_format.is_none());
            assert!(cfg.partial_images.is_none());
            assert!(cfg.quality.is_none());
            assert!(cfg.size.is_none());
        }
        other => panic!("expected ImageGeneration, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), minimal);

    // Full spec shape: every optional field populated.
    let full = json!({
        "type": "image_generation",
        "action": "edit",
        "background": "transparent",
        "input_fidelity": "high",
        "input_image_mask": {
            "file_id": "file_abc",
            "image_url": "https://example.com/mask.png"
        },
        "model": "gpt-image-1",
        "moderation": "low",
        "output_compression": 80,
        "output_format": "png",
        "partial_images": 2,
        "quality": "high",
        "size": "1024x1024"
    });
    let tool: ResponseTool =
        serde_json::from_value(full.clone()).expect("full image_generation deserialize");
    match &tool {
        ResponseTool::ImageGeneration(cfg) => {
            assert_eq!(cfg.action.as_deref(), Some("edit"));
            assert_eq!(cfg.background.as_deref(), Some("transparent"));
            assert_eq!(cfg.input_fidelity.as_deref(), Some("high"));
            let mask = cfg.input_image_mask.as_ref().expect("mask present");
            assert_eq!(mask.file_id.as_deref(), Some("file_abc"));
            assert_eq!(
                mask.image_url.as_deref(),
                Some("https://example.com/mask.png")
            );
            assert_eq!(cfg.model.as_deref(), Some("gpt-image-1"));
            assert_eq!(cfg.moderation.as_deref(), Some("low"));
            assert_eq!(cfg.output_compression, Some(80));
            assert_eq!(cfg.output_format.as_deref(), Some("png"));
            assert_eq!(cfg.partial_images, Some(2));
            assert_eq!(cfg.quality.as_deref(), Some("high"));
            assert_eq!(cfg.size.as_deref(), Some("1024x1024"));
        }
        other => panic!("expected ImageGeneration, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), full);
}

/// `ResponseOutputItem::ImageGenerationCall` — `{id, result, status, type}`
/// round-trips with every spec-listed status variant. Absent
/// `revised_prompt` deserializes to `None` and is omitted on serialize, so
/// the default wire shape stays byte-identical to the spec example.
#[test]
fn image_generation_call_output_item_round_trips_spec_shape() {
    for (status_json, expected) in [
        ("in_progress", ImageGenerationCallStatus::InProgress),
        ("completed", ImageGenerationCallStatus::Completed),
        ("generating", ImageGenerationCallStatus::Generating),
        ("failed", ImageGenerationCallStatus::Failed),
    ] {
        let payload = json!({
            "type": "image_generation_call",
            "id": "ig_1",
            "result": "aGVsbG8=",
            "status": status_json,
        });
        let item: ResponseOutputItem = serde_json::from_value(payload.clone())
            .expect("image_generation_call output item deserialize");
        match &item {
            ResponseOutputItem::ImageGenerationCall {
                id,
                result,
                revised_prompt,
                status,
                ..
            } => {
                assert_eq!(id, "ig_1");
                assert_eq!(result, "aGVsbG8=");
                assert!(revised_prompt.is_none());
                assert_eq!(*status, expected);
            }
            other => panic!("expected ImageGenerationCall, got {other:?}"),
        }
        assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
    }
}

/// `ResponseOutputItem::ImageGenerationCall` preserves `revised_prompt`
/// through a full (de)serialization cycle. OpenAI populates this when
/// the mainline model rewrites the user prompt before dispatching the
/// image-generation call; dropping it here would silently lose prompt
/// provenance during storage/replay.
#[test]
fn image_generation_call_output_item_round_trips_with_revised_prompt() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_1",
        "result": "aGVsbG8=",
        "revised_prompt": "A red fox sitting on a rock at sunset.",
        "status": "completed",
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("image_generation_call output item deserialize");
    match &item {
        ResponseOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
            ..
        } => {
            assert_eq!(id, "ig_1");
            assert_eq!(result, "aGVsbG8=");
            assert_eq!(
                revised_prompt.as_deref(),
                Some("A red fox sitting on a rock at sunset.")
            );
            assert_eq!(*status, ImageGenerationCallStatus::Completed);
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// `ResponseInputOutputItem::ImageGenerationCall` mirrors the output-item
/// shape so clients can replay a prior-turn image generation through the
/// `input` array for stateless round-trips.
#[test]
fn image_generation_call_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_2",
        "result": "d29ybGQ=",
        "status": "completed",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
            ..
        } => {
            assert_eq!(id, "ig_2");
            assert_eq!(result.as_deref(), Some("d29ybGQ="));
            assert!(revised_prompt.is_none());
            assert_eq!(*status, Some(ImageGenerationCallStatus::Completed));
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// `ResponseInputOutputItem::ImageGenerationCall` accepts the documented
/// multi-turn id-only reference form: clients resubmitting
/// `{ "type": "image_generation_call", "id": ... }` to continue an
/// image-edit conversation must deserialize and round-trip without
/// forcing the full `result` / `status` payload. (See OpenAI
/// Responses API image-generation tool guide.)
#[test]
fn image_generation_call_input_item_accepts_id_only_reference() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_3",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("id-only image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
            ..
        } => {
            assert_eq!(id, "ig_3");
            assert!(result.is_none());
            assert!(revised_prompt.is_none());
            assert!(status.is_none());
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    // Serialized form must stay minimal — no `"result": null` or
    // `"status": null` leaks through `skip_serializing_if`.
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// `ResponseInputOutputItem::ImageGenerationCall` preserves
/// `revised_prompt` on replay — the stateless client path must carry the
/// rewritten prompt forward to match what the output variant emitted on
/// the originating turn.
#[test]
fn image_generation_call_input_item_round_trips_with_revised_prompt() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_2",
        "result": "d29ybGQ=",
        "revised_prompt": "A cozy cabin in a snowy pine forest at dusk.",
        "status": "completed",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
            id,
            result,
            revised_prompt,
            status,
            ..
        } => {
            assert_eq!(id, "ig_2");
            assert_eq!(result.as_deref(), Some("d29ybGQ="));
            assert_eq!(
                revised_prompt.as_deref(),
                Some("A cozy cabin in a snowy pine forest at dusk.")
            );
            assert_eq!(*status, Some(ImageGenerationCallStatus::Completed));
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// Real OpenAI production responses carry additional metadata on
/// `image_generation_call` output items (`action`, `background`,
/// `output_format`, `quality`, `size`) that the OpenAI Rust SDK v2.8.1 omits.
/// This roundtrip pins the full OpenAI-shaped output item and verifies every
/// metadata field is preserved through (de)serialization — silently dropping
/// any of these would break cloud-passthrough fidelity and integration
/// test assertions.
#[test]
fn image_generation_call_output_item_round_trips_with_full_metadata() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_prod_1",
        "action": "generate",
        "background": "opaque",
        "output_format": "png",
        "quality": "high",
        "result": "aGVsbG8=",
        "revised_prompt": "A red fox sitting on a mossy rock at golden hour.",
        "size": "1024x1024",
        "status": "completed",
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("full-metadata image_generation_call output item deserialize");
    match &item {
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
            assert_eq!(id, "ig_prod_1");
            assert_eq!(action.as_deref(), Some("generate"));
            assert_eq!(background.as_deref(), Some("opaque"));
            assert_eq!(output_format.as_deref(), Some("png"));
            assert_eq!(quality.as_deref(), Some("high"));
            assert_eq!(result, "aGVsbG8=");
            assert_eq!(
                revised_prompt.as_deref(),
                Some("A red fox sitting on a mossy rock at golden hour.")
            );
            assert_eq!(size.as_deref(), Some("1024x1024"));
            assert_eq!(*status, ImageGenerationCallStatus::Completed);
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// A minimal `image_generation_call` output item (only `id`, `result`,
/// `status`) must round-trip without introducing `null` entries for the
/// absent metadata fields — `skip_serializing_if = "Option::is_none"` keeps
/// the wire shape spec-minimal. Guards against an accidental
/// `Option::default()` that would materialize `"action": null` etc. on emit.
#[test]
fn image_generation_call_output_item_round_trips_minimal_without_metadata() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_min_1",
        "result": "aGVsbG8=",
        "status": "completed",
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("minimal image_generation_call output item deserialize");
    match &item {
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
            assert_eq!(id, "ig_min_1");
            assert!(action.is_none());
            assert!(background.is_none());
            assert!(output_format.is_none());
            assert!(quality.is_none());
            assert_eq!(result, "aGVsbG8=");
            assert!(revised_prompt.is_none());
            assert!(size.is_none());
            assert_eq!(*status, ImageGenerationCallStatus::Completed);
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    // Must be byte-for-byte equal — no `"action": null`, `"size": null` etc.
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// Symmetry with the output-side variant — the input-side
/// `ImageGenerationCall` also carries the five metadata fields so a client
/// resubmitting a prior-turn image generation item gets byte-identical
/// round-tripping. Without this, stateless multi-turn replay strips metadata
/// that the output item emitted.
#[test]
fn image_generation_call_input_item_round_trips_with_full_metadata() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_prod_2",
        "action": "edit",
        "background": "transparent",
        "output_format": "webp",
        "quality": "medium",
        "result": "d29ybGQ=",
        "revised_prompt": "A winter cabin with snow falling softly outside.",
        "size": "1024x1536",
        "status": "completed",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("full-metadata image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
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
            assert_eq!(id, "ig_prod_2");
            assert_eq!(action.as_deref(), Some("edit"));
            assert_eq!(background.as_deref(), Some("transparent"));
            assert_eq!(output_format.as_deref(), Some("webp"));
            assert_eq!(quality.as_deref(), Some("medium"));
            assert_eq!(result.as_deref(), Some("d29ybGQ="));
            assert_eq!(
                revised_prompt.as_deref(),
                Some("A winter cabin with snow falling softly outside.")
            );
            assert_eq!(size.as_deref(), Some("1024x1536"));
            assert_eq!(*status, Some(ImageGenerationCallStatus::Completed));
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// Minimal id-only input-side image_generation_call must keep serializing
/// without any `null` metadata fields — the documented multi-turn reference
/// form stays byte-identical after the metadata-field additions.
#[test]
fn image_generation_call_input_item_round_trips_minimal_without_metadata() {
    let payload = json!({
        "type": "image_generation_call",
        "id": "ig_min_2",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("minimal image_generation_call input item deserialize");
    match &item {
        ResponseInputOutputItem::ImageGenerationCall {
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
            assert_eq!(id, "ig_min_2");
            assert!(action.is_none());
            assert!(background.is_none());
            assert!(output_format.is_none());
            assert!(quality.is_none());
            assert!(result.is_none());
            assert!(revised_prompt.is_none());
            assert!(size.is_none());
            assert!(status.is_none());
        }
        other => panic!("expected ImageGenerationCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn compaction_output_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "compaction",
        "id": "cmp_abc123",
        "encrypted_content": "opaque-base64-blob",
    });
    let item: ResponseOutputItem =
        serde_json::from_value(payload.clone()).expect("compaction output item deserialize");
    match &item {
        ResponseOutputItem::Compaction {
            id,
            encrypted_content,
        } => {
            assert_eq!(id, "cmp_abc123");
            assert_eq!(encrypted_content, "opaque-base64-blob");
        }
        other => panic!("expected Compaction, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn compaction_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "compaction",
        "id": "cmp_abc123",
        "encrypted_content": "opaque-base64-blob",
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("compaction input item deserialize");
    match &item {
        ResponseInputOutputItem::Compaction {
            encrypted_content,
            id,
        } => {
            assert_eq!(encrypted_content, "opaque-base64-blob");
            assert_eq!(id.as_deref(), Some("cmp_abc123"));
        }
        other => panic!("expected Compaction, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn compaction_input_item_accepts_missing_id() {
    let payload = json!({
        "type": "compaction",
        "encrypted_content": "opaque-base64-blob",
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("id-less compaction input item deserialize");
    match &item {
        ResponseInputOutputItem::Compaction {
            encrypted_content,
            id,
        } => {
            assert_eq!(encrypted_content, "opaque-base64-blob");
            assert!(id.is_none());
        }
        other => panic!("expected Compaction, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_tool_round_trips_spec_shape() {
    let payload = json!({ "type": "computer" });
    let tool: ResponseTool = serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(tool, ResponseTool::Computer));
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn computer_use_preview_tool_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_use_preview",
        "display_height": 1080,
        "display_width": 1920,
        "environment": "browser",
    });
    let tool: ResponseTool = serde_json::from_value(payload.clone()).expect("deserialize");
    match &tool {
        ResponseTool::ComputerUsePreview(cup) => {
            assert_eq!(cup.display_height, 1080);
            assert_eq!(cup.display_width, 1920);
            assert_eq!(cup.environment, ComputerEnvironment::Browser);
        }
        other => panic!("expected ComputerUsePreview, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn computer_action_variants_round_trip() {
    let fixtures = [
        json!({"type": "click", "button": "left", "x": 10, "y": 20}),
        json!({"type": "double_click", "x": 100, "y": 200}),
        json!({
            "type": "drag",
            "path": [{"x": 0, "y": 0}, {"x": 10, "y": 10}],
        }),
        json!({"type": "keypress", "keys": ["ctrl", "c"]}),
        json!({"type": "move", "x": 50, "y": 60}),
        json!({"type": "screenshot"}),
        json!({
            "type": "scroll",
            "scroll_x": 0,
            "scroll_y": -120,
            "x": 100,
            "y": 200,
        }),
        json!({"type": "type", "text": "hello"}),
        json!({"type": "wait"}),
    ];
    for payload in fixtures {
        let action: ComputerAction =
            serde_json::from_value(payload.clone()).expect("action deserialize");
        assert_eq!(
            serde_json::to_value(&action).expect("action serialize"),
            payload,
        );
    }
}

#[test]
fn test_custom_tool_text_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L471-474):
    // `Custom { name, type: "custom", defer_loading?, description?, format? }`
    // with `format: CustomToolInputFormat = Text { type: "text" }`.
    let payload = json!({
        "type": "custom",
        "name": "run_sql",
        "description": "Run a SELECT against the analytics warehouse",
        "defer_loading": false,
        "format": {"type": "text"}
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("custom tool with text format should deserialize");
    match &tool {
        ResponseTool::Custom(c) => {
            assert_eq!(c.name, "run_sql");
            assert_eq!(c.defer_loading, Some(false));
            assert!(matches!(c.format, Some(CustomToolInputFormat::Text)));
        }
        other => panic!("expected ResponseTool::Custom, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("custom tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_grammar_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L472-474):
    // `Grammar { definition, syntax: "lark" | "regex", type: "grammar" }`.
    for syntax in ["lark", "regex"] {
        let payload = json!({
            "type": "custom",
            "name": "parse_expr",
            "format": {
                "type": "grammar",
                "definition": "start: NUMBER",
                "syntax": syntax
            }
        });

        let tool: ResponseTool = serde_json::from_value(payload.clone())
            .expect("custom tool with grammar format should deserialize");
        match &tool {
            ResponseTool::Custom(c) => match &c.format {
                Some(CustomToolInputFormat::Grammar(g)) => {
                    assert_eq!(g.definition, "start: NUMBER");
                    let expected = if syntax == "lark" {
                        CustomToolGrammarSyntax::Lark
                    } else {
                        CustomToolGrammarSyntax::Regex
                    };
                    assert_eq!(g.syntax, expected);
                }
                other => panic!("expected Grammar format, got {other:?}"),
            },
            other => panic!("expected ResponseTool::Custom, got {other:?}"),
        }

        let serialized = serde_json::to_value(&tool).expect("custom tool should serialize");
        assert_eq!(serialized, payload);
    }
}

#[test]
fn test_namespace_tool_with_function_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L475):
    // `Namespace { description, name, tools: array of Function | Custom,
    // type: "namespace" }`. Inner elements reuse the top-level Function
    // shape but are restricted by the spec to Function or Custom.
    let payload = json!({
        "type": "namespace",
        "name": "sql",
        "description": "Tools for interacting with the warehouse",
        "tools": [
            {
                "type": "function",
                "name": "select",
                "description": "Run a read-only SELECT",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                },
                "strict": true
            }
        ]
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("namespace tool with function element should deserialize");
    match &tool {
        ResponseTool::Namespace(def) => {
            assert_eq!(def.name, "sql");
            assert_eq!(def.description, "Tools for interacting with the warehouse");
            assert_eq!(def.tools.len(), 1);
            match &def.tools[0] {
                NamespaceTool::Function(ft) => {
                    assert_eq!(ft.function.name, "select");
                    assert_eq!(ft.function.strict, Some(true));
                }
                other @ NamespaceTool::Custom(_) => {
                    panic!("expected NamespaceTool::Function, got {other:?}")
                }
            }
        }
        other => panic!("expected ResponseTool::Namespace, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_namespace_tool_with_custom_text_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L471-475): namespace elements
    // may be Custom tools. `Custom { name, type: "custom", defer_loading?,
    // description?, format? }` with `format: Text { type: "text" }`.
    let payload = json!({
        "type": "namespace",
        "name": "shell",
        "description": "User-owned shell helpers",
        "tools": [
            {
                "type": "custom",
                "name": "raw_cmd",
                "description": "Emit a free-form shell command",
                "format": {"type": "text"}
            }
        ]
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("namespace tool with custom/text element should deserialize");
    match &tool {
        ResponseTool::Namespace(def) => {
            assert_eq!(def.tools.len(), 1);
            match &def.tools[0] {
                NamespaceTool::Custom(c) => {
                    assert_eq!(c.name, "raw_cmd");
                    assert!(matches!(c.format, Some(CustomToolInputFormat::Text)));
                }
                other @ NamespaceTool::Function(_) => {
                    panic!("expected NamespaceTool::Custom, got {other:?}")
                }
            }
        }
        other => panic!("expected ResponseTool::Namespace, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_namespace_tool_with_custom_grammar_format_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L472-475): namespace elements
    // may be Custom tools whose `format: Grammar { definition, syntax: "lark"
    // | "regex", type: "grammar" }` constrains free-form input at decode time.
    for syntax in ["lark", "regex"] {
        let payload = json!({
            "type": "namespace",
            "name": "parse",
            "description": "Grammar-constrained parsers",
            "tools": [
                {
                    "type": "custom",
                    "name": "expr_parser",
                    "format": {
                        "type": "grammar",
                        "definition": "start: NUMBER",
                        "syntax": syntax
                    }
                }
            ]
        });

        let tool: ResponseTool = serde_json::from_value(payload.clone())
            .expect("namespace tool with custom/grammar element should deserialize");
        match &tool {
            ResponseTool::Namespace(def) => match &def.tools[0] {
                NamespaceTool::Custom(c) => match &c.format {
                    Some(CustomToolInputFormat::Grammar(g)) => {
                        assert_eq!(g.definition, "start: NUMBER");
                        let expected = if syntax == "lark" {
                            CustomToolGrammarSyntax::Lark
                        } else {
                            CustomToolGrammarSyntax::Regex
                        };
                        assert_eq!(g.syntax, expected);
                    }
                    other => panic!("expected Grammar format, got {other:?}"),
                },
                other @ NamespaceTool::Function(_) => {
                    panic!("expected NamespaceTool::Custom, got {other:?}")
                }
            },
            other => panic!("expected ResponseTool::Namespace, got {other:?}"),
        }

        let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
        assert_eq!(serialized, payload);
    }
}

#[test]
fn test_namespace_tool_mixed_elements_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L475): a single namespace may
    // mix Function and Custom elements in any order.
    let payload = json!({
        "type": "namespace",
        "name": "fs",
        "description": "Filesystem helpers",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file by path",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                },
                "strict": true
            },
            {
                "type": "custom",
                "name": "write_file",
                "description": "Write a file by free-form payload"
            }
        ]
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("namespace tool with mixed elements should deserialize");
    match &tool {
        ResponseTool::Namespace(def) => {
            assert_eq!(def.tools.len(), 2);
            assert!(
                matches!(&def.tools[0], NamespaceTool::Function(ft) if ft.function.name == "read_file")
            );
            assert!(matches!(&def.tools[1], NamespaceTool::Custom(c) if c.name == "write_file"));
        }
        other => panic!("expected ResponseTool::Namespace, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("namespace tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_namespace_tool_rejects_nested_namespace_element() {
    // Spec (openai-responses-api-spec.md §tools L475): a Namespace's `tools`
    // may contain only Function or Custom — nested Namespace elements are
    // forbidden. The dedicated NamespaceTool enum (without a Namespace arm)
    // is what enforces this at the protocol layer.
    let payload = json!({
        "type": "namespace",
        "name": "outer",
        "description": "outer namespace",
        "tools": [
            {
                "type": "namespace",
                "name": "inner",
                "description": "inner namespace",
                "tools": []
            }
        ]
    });

    assert!(
        serde_json::from_value::<ResponseTool>(payload).is_err(),
        "nested namespace elements must be rejected"
    );
}

#[test]
fn test_namespace_tool_rejects_hosted_tool_element() {
    // Spec (openai-responses-api-spec.md §tools L475): hosted / built-in tool
    // types (e.g. `file_search`, `web_search_preview`, `code_interpreter`)
    // may appear as top-level ResponseTool entries but MUST NOT appear as
    // namespace elements.
    let payload = json!({
        "type": "namespace",
        "name": "ns",
        "description": "namespace",
        "tools": [
            { "type": "file_search", "vector_store_ids": ["vs_123"] }
        ]
    });

    assert!(
        serde_json::from_value::<ResponseTool>(payload).is_err(),
        "hosted/built-in tools must be rejected inside namespace.tools"
    );
}

#[test]
fn computer_call_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call",
        "id": "cc_1",
        "call_id": "call_1",
        "action": {"type": "screenshot"},
        "status": "completed",
        "pending_safety_checks": [],
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(item, ResponseInputOutputItem::ComputerCall { .. }));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_call_output_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call_output",
        "call_id": "call_1",
        "output": {"type": "computer_screenshot", "file_id": "file_1"},
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(
        item,
        ResponseInputOutputItem::ComputerCallOutput { .. }
    ));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_call_output_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call",
        "id": "cc_1",
        "call_id": "call_1",
        "action": {"type": "click", "button": "left", "x": 10, "y": 20},
        "status": "in_progress",
        "pending_safety_checks": [],
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(item, ResponseOutputItem::ComputerCall { .. }));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_call_output_output_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "computer_call_output",
        "call_id": "call_1",
        "output": {"type": "computer_screenshot", "image_url": "https://example/s.png"},
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone()).expect("deserialize");
    assert!(matches!(
        item,
        ResponseOutputItem::ComputerCallOutput { .. }
    ));
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn computer_safety_check_accepts_optional_code_and_message() {
    // Per SDK v2.8.1 (types/responses/response_computer_tool_call.py::PendingSafetyCheck),
    // both `code` and `message` are `Optional[str] = None`. Payloads that omit
    // either field must deserialize, and the serialization round-trip must
    // preserve the missing fields (not invent empty strings).
    let payload = json!({
        "type": "computer_call",
        "id": "cc_1",
        "call_id": "call_1",
        "action": {"type": "screenshot"},
        "status": "in_progress",
        "pending_safety_checks": [
            {"id": "psc_only_id"},
            {"id": "psc_with_code", "code": "malicious_instructions"},
            {"id": "psc_with_message", "message": "details"},
            {"id": "psc_full", "code": "c", "message": "m"},
        ],
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone()).expect("deserialize");
    match &item {
        ResponseOutputItem::ComputerCall {
            pending_safety_checks,
            ..
        } => {
            assert_eq!(pending_safety_checks.len(), 4);
            assert!(pending_safety_checks[0].code.is_none());
            assert!(pending_safety_checks[0].message.is_none());
            assert_eq!(
                pending_safety_checks[1].code.as_deref(),
                Some("malicious_instructions")
            );
            assert!(pending_safety_checks[1].message.is_none());
            assert!(pending_safety_checks[2].code.is_none());
            assert_eq!(pending_safety_checks[2].message.as_deref(), Some("details"));
            assert_eq!(pending_safety_checks[3].code.as_deref(), Some("c"));
            assert_eq!(pending_safety_checks[3].message.as_deref(), Some("m"));
        }
        other => panic!("expected ComputerCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn test_custom_tool_call_input_item_round_trip() {
    // Spec (openai-responses-api-spec.md L272-273):
    // `CustomToolCall object { call_id, input, name, type, id, namespace }`
    // `type: "custom_tool_call"`.
    let payload = json!({
        "type": "custom_tool_call",
        "call_id": "call_abc123",
        "input": "SELECT count(*) FROM events",
        "name": "run_sql",
        "id": "ctc_01",
        "namespace": "analytics"
    });

    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("custom_tool_call should deserialize");
    match &item {
        ResponseInputOutputItem::CustomToolCall {
            call_id,
            input,
            name,
            id,
            namespace,
        } => {
            assert_eq!(call_id, "call_abc123");
            assert_eq!(input, "SELECT count(*) FROM events");
            assert_eq!(name, "run_sql");
            assert_eq!(id.as_deref(), Some("ctc_01"));
            assert_eq!(namespace.as_deref(), Some("analytics"));
        }
        other => panic!("expected CustomToolCall, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_call_output_string_round_trip() {
    // Spec (openai-responses-api-spec.md L268-270):
    // `CustomToolCallOutput object { call_id, output, type, id }` where
    // `output: string or array of ResponseInputText | ResponseInputImage |
    // ResponseInputFile`. No `status` field per spec.
    let payload = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": "42",
        "id": "ctco_01"
    });

    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("custom_tool_call_output should deserialize");
    match &item {
        ResponseInputOutputItem::CustomToolCallOutput {
            call_id,
            output,
            id,
        } => {
            assert_eq!(call_id, "call_abc123");
            assert_eq!(id.as_deref(), Some("ctco_01"));
            match output {
                CustomToolCallOutputContent::Text(s) => assert_eq!(s, "42"),
                CustomToolCallOutputContent::Parts(_) => panic!("expected Text output"),
            }
        }
        other => panic!("expected CustomToolCallOutput, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_call_output_array_round_trip() {
    // Spec (openai-responses-api-spec.md L269): array of
    // ResponseInputText | ResponseInputImage | ResponseInputFile.
    let payload = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "input_text", "text": "rows returned:"},
            {"type": "input_file", "filename": "result.csv", "file_id": "file_xyz"}
        ]
    });

    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("array-form custom_tool_call_output should deserialize");
    match &item {
        ResponseInputOutputItem::CustomToolCallOutput { output, .. } => match output {
            CustomToolCallOutputContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(
                    parts[0],
                    CustomToolInputContentPart::InputText { .. }
                ));
                assert!(matches!(
                    parts[1],
                    CustomToolInputContentPart::InputFile { .. }
                ));
            }
            CustomToolCallOutputContent::Text(_) => panic!("expected Parts output"),
        },
        other => panic!("expected CustomToolCallOutput, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_custom_tool_call_output_parts_reject_output_text() {
    // Spec (openai-responses-api-spec.md L269): the `output` array only
    // accepts input-typed parts. Assistant-facing shapes such as
    // `output_text` and `refusal` must be rejected at the type boundary
    // rather than silently coerced.
    let payload_output_text = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "output_text", "text": "not allowed here", "annotations": []}
        ]
    });
    let err = serde_json::from_value::<ResponseInputOutputItem>(payload_output_text)
        .expect_err("output_text must not be accepted inside custom_tool_call_output parts");
    let msg = err.to_string();
    assert!(
        msg.contains("output_text") || msg.contains("did not match any variant"),
        "expected untagged-variant rejection carrying `output_text`, got: {msg}"
    );

    let payload_refusal = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "refusal", "refusal": "nope"}
        ]
    });
    assert!(
        serde_json::from_value::<ResponseInputOutputItem>(payload_refusal).is_err(),
        "refusal must not be accepted inside custom_tool_call_output parts"
    );

    // Sanity: an input-typed part still deserializes cleanly.
    let payload_ok = json!({
        "type": "custom_tool_call_output",
        "call_id": "call_abc123",
        "output": [
            {"type": "input_text", "text": "ok"}
        ]
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload_ok)
        .expect("input_text part must still deserialize after the tightening");
    match &item {
        ResponseInputOutputItem::CustomToolCallOutput { output, .. } => match output {
            CustomToolCallOutputContent::Parts(parts) => {
                assert_eq!(parts.len(), 1);
                assert!(matches!(
                    parts[0],
                    CustomToolInputContentPart::InputText { .. }
                ));
            }
            CustomToolCallOutputContent::Text(_) => panic!("expected Parts output"),
        },
        other => panic!("expected CustomToolCallOutput, got {other:?}"),
    }
}

#[test]
fn test_custom_tool_call_output_validation_rejects_empty_text_and_parts() {
    use validator::Validate;

    // cross-parameter validation (L2145-2160) requires a Message/SimpleInputMessage
    // alongside any tool item, so every payload below pairs the custom_tool_call_output
    // with a plain user message to isolate the CustomToolCallOutput branch we want
    // to cover.
    let user_msg = json!({"role": "user", "content": "hi"});

    // Empty string output — validate_input_item's
    // `CustomToolCallOutputContent::Text(s) if s.is_empty()` branch.
    let empty_text: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg,
            {
                "type": "custom_tool_call_output",
                "call_id": "call_abc123",
                "output": ""
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        empty_text.validate().is_err(),
        "empty CustomToolCallOutput text must be rejected"
    );

    // Empty parts array output — validate_input_item's
    // `CustomToolCallOutputContent::Parts(parts) if parts.is_empty()` branch.
    let empty_parts: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg,
            {
                "type": "custom_tool_call_output",
                "call_id": "call_abc123",
                "output": []
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        empty_parts.validate().is_err(),
        "empty CustomToolCallOutput parts array must be rejected"
    );

    // Sanity: a non-empty string output passes the same validator.
    let ok_text: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg,
            {
                "type": "custom_tool_call_output",
                "call_id": "call_abc123",
                "output": "42"
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        ok_text.validate().is_ok(),
        "non-empty CustomToolCallOutput must validate",
    );
}

// ---------------------------------------------------------------------------
// T6 — containerized `shell` tool + call/output items
// ---------------------------------------------------------------------------

#[test]
fn shell_tool_no_environment_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L463-464):
    // `Shell { type: "shell", environment? }`. Environment is optional.
    let payload = json!({
        "type": "shell",
    });
    let tool: ResponseTool =
        serde_json::from_value(payload.clone()).expect("shell tool w/o env should deserialize");
    match &tool {
        ResponseTool::Shell(s) => assert!(s.environment.is_none()),
        other => panic!("expected ResponseTool::Shell, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn shell_tool_container_auto_environment_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L465-468):
    // `ContainerAuto { type: "container_auto", file_ids?, memory_limit?,
    //  network_policy? }`.
    let payload = json!({
        "type": "shell",
        "environment": {
            "type": "container_auto",
            "file_ids": ["file_001", "file_002"],
            "memory_limit": "4g",
            "network_policy": {
                "type": "allowlist",
                "allowed_domains": ["api.github.com", "pypi.org"],
                "domain_secrets": [
                    {"domain": "api.github.com", "name": "GH_TOKEN", "value": "sk-abc"}
                ]
            }
        }
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("shell tool w/ container_auto env should deserialize");
    match &tool {
        ResponseTool::Shell(s) => match &s.environment {
            Some(ShellEnvironment::ContainerAuto(auto)) => {
                assert_eq!(
                    auto.file_ids.as_deref(),
                    Some(&["file_001".into(), "file_002".into()][..])
                );
                assert_eq!(auto.memory_limit.as_deref(), Some("4g"));
                match auto
                    .network_policy
                    .as_ref()
                    .expect("network_policy present")
                {
                    ContainerNetworkPolicy::Allowlist(a) => {
                        assert_eq!(a.allowed_domains, vec!["api.github.com", "pypi.org"]);
                        let secrets = a.domain_secrets.as_ref().expect("domain_secrets present");
                        assert_eq!(secrets.len(), 1);
                        assert_eq!(secrets[0].domain, "api.github.com");
                        assert_eq!(secrets[0].name, "GH_TOKEN");
                        assert_eq!(secrets[0].value, "sk-abc");
                    }
                    ContainerNetworkPolicy::Disabled => {
                        panic!("expected Allowlist policy, got Disabled")
                    }
                }
            }
            other => panic!("expected ContainerAuto, got {other:?}"),
        },
        other => panic!("expected ResponseTool::Shell, got {other:?}"),
    }

    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn shell_tool_container_auto_disabled_network_policy_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L448):
    // `ContainerNetworkPolicyDisabled { type: "disabled" }` carries no fields.
    let payload = json!({
        "type": "shell",
        "environment": {
            "type": "container_auto",
            "network_policy": {"type": "disabled"}
        }
    });
    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("shell tool w/ disabled network policy should deserialize");
    match &tool {
        ResponseTool::Shell(s) => match &s.environment {
            Some(ShellEnvironment::ContainerAuto(auto)) => {
                assert!(matches!(
                    auto.network_policy.as_ref().expect("policy present"),
                    ContainerNetworkPolicy::Disabled
                ));
            }
            other => panic!("expected ContainerAuto env, got {other:?}"),
        },
        other => panic!("expected ResponseTool::Shell, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn shell_tool_local_environment_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L469):
    // `LocalEnvironment { type: "local" }`.
    let payload = json!({
        "type": "shell",
        "environment": {
            "type": "local"
        }
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("shell tool w/ local env should deserialize");
    match &tool {
        ResponseTool::Shell(s) => match &s.environment {
            Some(ShellEnvironment::Local(_)) => {}
            other => panic!("expected Local env, got {other:?}"),
        },
        other => panic!("expected ResponseTool::Shell, got {other:?}"),
    }

    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn shell_tool_container_reference_environment_round_trip() {
    // Spec (openai-responses-api-spec.md §tools, L470):
    // `ContainerReference { container_id, type: "container_reference" }`.
    let payload = json!({
        "type": "shell",
        "environment": {
            "type": "container_reference",
            "container_id": "container_abc123"
        }
    });

    let tool: ResponseTool = serde_json::from_value(payload.clone())
        .expect("shell tool w/ container_reference env should deserialize");
    match &tool {
        ResponseTool::Shell(s) => match &s.environment {
            Some(ShellEnvironment::ContainerReference(r)) => {
                assert_eq!(r.container_id, "container_abc123");
            }
            other => panic!("expected ContainerReference env, got {other:?}"),
        },
        other => panic!("expected ResponseTool::Shell, got {other:?}"),
    }

    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

#[test]
fn shell_call_input_item_round_trips_spec_shape() {
    // Spec (openai-responses-api-spec.md §ShellCall, L228-231):
    // `{ action, call_id, type, id, environment, status }` with
    // `action: { commands: array of string, max_output_length?, timeout_ms? }`
    // and `environment: optional LocalEnvironment | ContainerReference`
    // (no `container_auto` on the call form per L512-513).
    let payload = json!({
        "type": "shell_call",
        "id": "sc_01",
        "call_id": "call_shell_1",
        "action": {
            "commands": ["bash", "-lc", "echo hi"],
            "max_output_length": 65536,
            "timeout_ms": 10000
        },
        "environment": {
            "type": "container_reference",
            "container_id": "container_abc"
        },
        "status": "in_progress"
    });

    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("shell_call should deserialize");
    match &item {
        ResponseInputOutputItem::ShellCall {
            action,
            call_id,
            id,
            environment,
            status,
            created_by,
        } => {
            assert_eq!(call_id, "call_shell_1");
            assert_eq!(id.as_deref(), Some("sc_01"));
            assert_eq!(action.commands, vec!["bash", "-lc", "echo hi"]);
            assert_eq!(action.max_output_length, Some(65536));
            assert_eq!(action.timeout_ms, Some(10000));
            assert!(matches!(
                environment.as_ref().expect("env present"),
                ShellCallEnvironment::ContainerReference(_)
            ));
            assert_eq!(*status, Some(ShellCallStatus::InProgress));
            assert!(created_by.is_none());
        }
        other => panic!("expected ShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn item_reference_input_item_round_trips_with_type() {
    // Spec (openai-responses-api-spec.md L275-276):
    // `ItemReference { id, type }` where `type` is `optional "item_reference"`.
    // The tagged form carries the discriminator explicitly.
    let payload = json!({
        "type": "item_reference",
        "id": "msg_abc123",
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("tagged item_reference should deserialize");
    match &item {
        ResponseInputOutputItem::ItemReference { id, r#type } => {
            assert_eq!(id, "msg_abc123");
            assert_eq!(*r#type, Some(ItemReferenceTypeTag::ItemReference));
        }
        other => panic!("expected ItemReference, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_minimal_shape_round_trips() {
    // Spec: `id`, `environment`, `status` are all acceptable as absent on
    // the input side; `action` and `call_id` are the only required fields
    // per the audit's Desired shape (§T6 L528).
    let payload = json!({
        "type": "shell_call",
        "call_id": "call_shell_2",
        "action": {"commands": ["ls"]}
    });

    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("minimal shell_call should deserialize");
    match &item {
        ResponseInputOutputItem::ShellCall {
            action,
            call_id,
            id,
            environment,
            status,
            created_by,
        } => {
            assert_eq!(call_id, "call_shell_2");
            assert!(id.is_none());
            assert!(environment.is_none());
            assert!(status.is_none());
            assert_eq!(action.commands, vec!["ls"]);
            assert!(action.max_output_length.is_none());
            assert!(action.timeout_ms.is_none());
            assert!(created_by.is_none());
        }
        other => panic!("expected ShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn item_reference_input_item_accepts_missing_type() {
    // Spec L276: `type: optional "item_reference"` — the discriminator can
    // be omitted on the wire. Bare `{id}` must still land in the
    // `ItemReference` variant and not in any other catch-all shape.
    let payload = json!({
        "id": "msg_abc123",
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("id-only item_reference should deserialize");
    match &item {
        ResponseInputOutputItem::ItemReference { id, r#type } => {
            assert_eq!(id, "msg_abc123");
            assert!(r#type.is_none());
        }
        other => panic!("expected ItemReference, got {other:?}"),
    }
    // `r#type: None` is `skip_serializing_if = Option::is_none`, so the
    // serialized shape round-trips exactly to the id-only payload.
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_output_input_item_round_trips_spec_shape() {
    // Spec (openai-responses-api-spec.md §ShellCallOutput, L233-238):
    // `{ call_id, output, type, id, max_output_length, status }` with
    // `output: array of { outcome, stderr, stdout }` where
    // `outcome: Timeout { type: "timeout" } | Exit { exit_code, type: "exit" }`.
    let payload = json!({
        "type": "shell_call_output",
        "id": "sco_01",
        "call_id": "call_shell_1",
        "output": [
            {
                "outcome": {"type": "exit", "exit_code": 0},
                "stderr": "",
                "stdout": "hi\n",
                "created_by": "system"
            },
            {
                "outcome": {"type": "timeout"},
                "stderr": "killed\n",
                "stdout": ""
            }
        ],
        "max_output_length": 65536,
        "status": "completed"
    });

    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("shell_call_output should deserialize");
    match &item {
        ResponseInputOutputItem::ShellCallOutput {
            call_id,
            output,
            id,
            max_output_length,
            status,
            created_by,
        } => {
            assert_eq!(call_id, "call_shell_1");
            assert_eq!(id.as_deref(), Some("sco_01"));
            assert_eq!(*max_output_length, Some(65536));
            assert_eq!(*status, Some(ShellCallStatus::Completed));
            assert_eq!(output.len(), 2);
            match &output[0].outcome {
                ShellOutcome::Exit(e) => assert_eq!(e.exit_code, 0),
                ShellOutcome::Timeout => panic!("expected Exit outcome, got Timeout"),
            }
            assert_eq!(output[0].stdout, "hi\n");
            assert_eq!(output[0].stderr, "");
            assert_eq!(output[0].created_by.as_deref(), Some("system"));
            assert!(matches!(output[1].outcome, ShellOutcome::Timeout));
            assert_eq!(output[1].stderr, "killed\n");
            assert!(output[1].created_by.is_none());
            assert!(created_by.is_none());
        }
        other => panic!("expected ShellCallOutput, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_output_item_round_trips_on_response_side() {
    // Spec (openai-responses-api-spec.md §returns `output: array of
    // ResponseOutputItem`, L512-514): ShellCall / ShellCallOutput are
    // listed as legal output items. Mirrors the input-side variant
    // byte-for-byte.
    let payload = json!({
        "type": "shell_call",
        "id": "sc_10",
        "call_id": "call_shell_10",
        "action": {"commands": ["echo", "ok"]},
        "environment": {"type": "local"},
        "status": "completed"
    });

    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("response-side shell_call should deserialize");
    match &item {
        ResponseOutputItem::ShellCall {
            id,
            call_id,
            environment,
            status,
            ..
        } => {
            assert_eq!(id, "sc_10");
            assert_eq!(call_id, "call_shell_10");
            match environment.as_ref().expect("env present") {
                ResponseShellCallEnvironment::Local(_) => {}
                ResponseShellCallEnvironment::ContainerReference(_) => {
                    panic!("expected Local env, got ContainerReference")
                }
            }
            assert_eq!(*status, ShellCallStatus::Completed);
        }
        other => panic!("expected ResponseOutputItem::ShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

// ============================================================================
// T5: local_shell tool + local_shell_call / local_shell_call_output round-trips
// ============================================================================

/// `ResponseTool::LocalShell` is a unit tool whose on-wire shape is just
/// `{ "type": "local_shell" }`. Spec (openai-responses-api-spec.md §tools
/// L462): `LocalShell { type: "local_shell" }`.
#[test]
fn local_shell_tool_round_trips_spec_shape() {
    let payload = json!({ "type": "local_shell" });
    let tool: ResponseTool =
        serde_json::from_value(payload.clone()).expect("local_shell tool should deserialize");
    assert!(matches!(tool, ResponseTool::LocalShell));
    assert_eq!(serde_json::to_value(&tool).expect("serialize"), payload);
}

/// `ResponseInputOutputItem::LocalShellCall` carries the minimum spec shape:
/// `{ id, action: { command, env, type: "exec" }, call_id, status, type }`.
/// Spec (openai-responses-api-spec.md §LocalShellCall L219-222).
#[test]
fn local_shell_call_input_item_round_trips_spec_shape() {
    let payload = json!({
        "type": "local_shell_call",
        "id": "ls_1",
        "call_id": "call_ls_1",
        "action": {
            "type": "exec",
            "command": ["/bin/echo", "hello"],
            "env": {}
        },
        "status": "in_progress"
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("local_shell_call input item should deserialize");
    match &item {
        ResponseInputOutputItem::LocalShellCall {
            id,
            call_id,
            action,
            status,
        } => {
            assert_eq!(id, "ls_1");
            assert_eq!(call_id, "call_ls_1");
            assert_eq!(*status, LocalShellCallStatus::InProgress);
            match action {
                LocalShellExec::Exec {
                    command,
                    env,
                    timeout_ms,
                    user,
                    working_directory,
                } => {
                    assert_eq!(command, &vec!["/bin/echo".to_string(), "hello".to_string()]);
                    assert!(env.is_empty());
                    assert!(timeout_ms.is_none());
                    assert!(user.is_none());
                    assert!(working_directory.is_none());
                }
            }
        }
        other => panic!("expected LocalShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_container_reference_on_response_side_round_trip() {
    // Spec (openai-responses-api-spec.md §returns L512-513): the
    // response-side shell_call environment covers both `local` and
    // `container_reference`. The sibling
    // `shell_call_output_item_round_trips_on_response_side` test only
    // exercises the `Local` arm; this fixture closes coverage for
    // [`ResponseShellCallEnvironment::ContainerReference`] so a broken
    // container_reference arm can't slip through the suite even though
    // the input- and response-side enums are intentionally distinct.
    let payload = json!({
        "type": "shell_call",
        "id": "sc_20",
        "call_id": "call_shell_20",
        "action": {"commands": ["echo", "ok"]},
        "environment": {
            "type": "container_reference",
            "container_id": "container_xyz"
        },
        "status": "completed"
    });

    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("response-side shell_call w/ container_reference should deserialize");
    match &item {
        ResponseOutputItem::ShellCall {
            id,
            call_id,
            environment,
            status,
            ..
        } => {
            assert_eq!(id, "sc_20");
            assert_eq!(call_id, "call_shell_20");
            match environment.as_ref().expect("env present") {
                ResponseShellCallEnvironment::ContainerReference(r) => {
                    assert_eq!(r.container_id, "container_xyz");
                }
                ResponseShellCallEnvironment::Local(_) => {
                    panic!("expected ContainerReference env, got Local")
                }
            }
            assert_eq!(*status, ShellCallStatus::Completed);
        }
        other => panic!("expected ResponseOutputItem::ShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// All optional `LocalShellExec::Exec` fields (`timeout_ms`, `user`,
/// `working_directory`) plus populated `env` round-trip through a full
/// (de)serialize cycle.
#[test]
fn local_shell_call_input_item_round_trips_with_all_optionals() {
    let payload = json!({
        "type": "local_shell_call",
        "id": "ls_2",
        "call_id": "call_ls_2",
        "action": {
            "type": "exec",
            "command": ["python", "-c", "print('hi')"],
            "env": { "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8" },
            "timeout_ms": 5000,
            "user": "root",
            "working_directory": "/tmp"
        },
        "status": "completed"
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("local_shell_call with optionals should deserialize");
    match &item {
        ResponseInputOutputItem::LocalShellCall { action, status, .. } => {
            assert_eq!(*status, LocalShellCallStatus::Completed);
            match action {
                LocalShellExec::Exec {
                    command,
                    env,
                    timeout_ms,
                    user,
                    working_directory,
                } => {
                    assert_eq!(command.len(), 3);
                    assert_eq!(env.get("LANG").map(String::as_str), Some("C.UTF-8"));
                    assert_eq!(env.get("PYTHONUNBUFFERED").map(String::as_str), Some("1"));
                    assert_eq!(*timeout_ms, Some(5000));
                    assert_eq!(user.as_deref(), Some("root"));
                    assert_eq!(working_directory.as_deref(), Some("/tmp"));
                }
            }
        }
        other => panic!("expected LocalShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_output_response_side_round_trip() {
    // Mirror of shell_call_output_input_item_round_trips_spec_shape but
    // deserialized into [`ResponseOutputItem`] to prove the output union
    // accepts the same wire shape. The response-side variant requires `id`,
    // `max_output_length`, and `status` per spec L233.
    let payload = json!({
        "type": "shell_call_output",
        "id": "sco_10",
        "call_id": "call_shell_10",
        "output": [
            {
                "outcome": {"type": "exit", "exit_code": 2},
                "stderr": "boom",
                "stdout": ""
            }
        ],
        "max_output_length": 65536,
        "status": "incomplete"
    });

    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("response-side shell_call_output should deserialize");
    match &item {
        ResponseOutputItem::ShellCallOutput {
            id,
            call_id,
            output,
            status,
            max_output_length,
            created_by,
        } => {
            assert_eq!(id, "sco_10");
            assert_eq!(call_id, "call_shell_10");
            assert_eq!(*status, ShellCallStatus::Incomplete);
            assert_eq!(*max_output_length, Some(65536));
            assert_eq!(output.len(), 1);
            match &output[0].outcome {
                ShellOutcome::Exit(e) => assert_eq!(e.exit_code, 2),
                ShellOutcome::Timeout => panic!("expected Exit outcome, got Timeout"),
            }
            assert!(created_by.is_none());
        }
        other => panic!("expected ResponseOutputItem::ShellCallOutput, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_with_created_by_round_trip() {
    // OpenAI SDK v2.8.1 `ResponseFunctionShellToolCall` carries an optional
    // `created_by: Optional[str]` provenance tag (see
    // `response_function_shell_tool_call.py`). Items returned by the
    // platform populate it; client-authored calls omit it. This fixture
    // locks both the deserialize-with-value and serialize-preserves-value
    // paths on the input side.
    let payload = json!({
        "type": "shell_call",
        "id": "sc_cb",
        "call_id": "call_shell_cb",
        "action": {"commands": ["ls"]},
        "status": "completed",
        "created_by": "user_abc"
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("shell_call w/ created_by should deserialize");
    match &item {
        ResponseInputOutputItem::ShellCall { created_by, .. } => {
            assert_eq!(created_by.as_deref(), Some("user_abc"));
        }
        other => panic!("expected ShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_output_with_created_by_round_trip() {
    // OpenAI SDK v2.8.1 `ResponseFunctionShellToolCallOutput` carries an
    // optional `created_by: Optional[str]` (see
    // `response_function_shell_tool_call_output.py`). Mirror of the
    // `shell_call_with_created_by_round_trip` fixture for the
    // `shell_call_output` variant — different SDK type, same
    // `Optional[str]` contract.
    let payload = json!({
        "type": "shell_call_output",
        "id": "sco_cb",
        "call_id": "call_shell_cb",
        "output": [],
        "status": "completed",
        "created_by": "platform"
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("shell_call_output w/ created_by should deserialize");
    match &item {
        ResponseInputOutputItem::ShellCallOutput { created_by, .. } => {
            assert_eq!(created_by.as_deref(), Some("platform"));
        }
        other => panic!("expected ShellCallOutput, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_output_without_max_output_length_round_trips_on_response_side() {
    // OpenAI SDK v2.8.1 `ResponseFunctionShellToolCallOutput.max_output_length`
    // is typed `Optional[int]`, so the platform may emit `shell_call_output`
    // items where the originating `shell_call.action.max_output_length` was
    // not set. This fixture proves the response-side union deserializes the
    // SDK-optional shape without requiring `max_output_length` to be
    // present on the wire.
    let payload = json!({
        "type": "shell_call_output",
        "id": "sco_no_max",
        "call_id": "call_shell_no_max",
        "output": [],
        "status": "completed"
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("response-side shell_call_output w/o max_output_length should deserialize");
    match &item {
        ResponseOutputItem::ShellCallOutput {
            max_output_length,
            created_by,
            ..
        } => {
            assert!(max_output_length.is_none());
            assert!(created_by.is_none());
        }
        other => panic!("expected ResponseOutputItem::ShellCallOutput, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_environment_rejects_container_auto() {
    // Spec (openai-responses-api-spec.md §returns, L512-513): the call-side
    // environment union only carries `local` / `container_reference` —
    // `container_auto` is request-only. A payload that tries to sneak it
    // onto a call form must fail to deserialize rather than silently
    // succeed, preserving the response-form restriction.
    let payload = json!({
        "type": "shell_call",
        "call_id": "call_shell_3",
        "action": {"commands": ["ls"]},
        "environment": {
            "type": "container_auto",
            "file_ids": []
        }
    });
    // Rejection must happen — the exact message varies depending on which
    // internal serde branch fires first (outer `type`-tagged arm vs. the
    // untagged SimpleInputMessage fallback), but no variant of
    // [`ResponseInputOutputItem`] accepts `container_auto` on a
    // `shell_call.environment` payload, so deserialization must fail.
    assert!(
        serde_json::from_value::<ResponseInputOutputItem>(payload).is_err(),
        "container_auto must be rejected on shell_call.environment"
    );
}

#[test]
fn shell_call_environment_rejects_container_auto_on_response_side() {
    // Mirror of `shell_call_environment_rejects_container_auto` targeting
    // the response-side union. [`ResponseShellCallEnvironment`] is a
    // separate enum from the input-side [`ShellCallEnvironment`], so the
    // input-side rejection test doesn't prove the response-side variant
    // also refuses `container_auto`. Spec §returns L512-513 constrains
    // the response-side shell_call environment to `local` /
    // `container_reference` only.
    let payload = json!({
        "type": "shell_call",
        "id": "sc_bad",
        "call_id": "call_shell_bad",
        "action": {"commands": ["ls"]},
        "environment": {
            "type": "container_auto",
            "file_ids": []
        },
        "status": "completed"
    });
    assert!(
        serde_json::from_value::<ResponseOutputItem>(payload).is_err(),
        "container_auto must be rejected on response-side shell_call.environment"
    );
}

#[test]
fn shell_call_environment_local_round_trips_on_input_side() {
    // Spec (openai-responses-api-spec.md §ShellCall L228-230): the
    // input-side `shell_call.environment.local` arm accepts the tool-side
    // `LocalEnvironment { type: "local" }` shape verbatim, round-tripping
    // losslessly through [`ResponseInputOutputItem`].
    let payload = json!({
        "type": "shell_call",
        "call_id": "call_shell_local",
        "action": {"commands": ["ls"]},
        "environment": {
            "type": "local"
        }
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("input-side shell_call with local env should deserialize");
    match &item {
        ResponseInputOutputItem::ShellCall { environment, .. } => {
            match environment.as_ref().expect("env present") {
                ShellCallEnvironment::Local(_) => {}
                ShellCallEnvironment::ContainerReference(_) => {
                    panic!("expected Local env, got ContainerReference")
                }
            }
        }
        other => panic!("expected ShellCall, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

#[test]
fn shell_call_output_validation_accepts_empty_output_for_replay() {
    // Spec (openai-responses-api-spec.md §ShellCallOutput L233-238):
    // `output: array of ResponseFunctionShellCallOutputContent`. The array
    // is legitimately empty on replay-only fixtures where the platform
    // surfaced an outcome without per-chunk stdout/stderr — request-level
    // validation (`validate_input_item`) keeps empty arrays valid. This
    // regression fixture locks that relaxed shape in so it cannot be
    // accidentally tightened, matching the pattern used for apply_patch
    // replay acceptance elsewhere in this file.
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            {"role": "user", "content": "hi"},
            {
                "type": "shell_call_output",
                "call_id": "call_shell_empty",
                "output": []
            }
        ]
    }))
    .expect("request with empty shell_call_output.output should deserialize");
    assert!(
        Validate::validate(&request).is_ok(),
        "empty shell_call_output.output must remain valid for lossless replay"
    );
}

#[test]
fn item_reference_input_item_rejects_unknown_type_tag() {
    // Spec L276 constrains `type` to exactly `"item_reference"` when
    // present. A payload carrying a different `type` must not silently land
    // in the `ItemReference` variant — the untagged catch-all is pinned to
    // [`ItemReferenceTypeTag::ItemReference`] for this reason (P5 fail-fast).
    let payload = json!({
        "type": "totally_made_up",
        "id": "msg_abc123",
    });
    let result: Result<ResponseInputOutputItem, _> = serde_json::from_value(payload);
    assert!(
        result.is_err(),
        "item_reference must reject unknown `type` discriminator, got: {result:?}"
    );
}

// ============================================================================
// T7 — apply_patch tool + call/output items
// ============================================================================

#[test]
fn test_apply_patch_tool_round_trip() {
    // Spec (openai-responses-api-spec.md §tools L478):
    // `ApplyPatch { type: "apply_patch" }`. Unit variant with no payload.
    let payload = json!({"type": "apply_patch"});

    let tool: ResponseTool =
        serde_json::from_value(payload.clone()).expect("apply_patch tool should deserialize");
    assert!(matches!(tool, ResponseTool::ApplyPatch));

    let serialized = serde_json::to_value(&tool).expect("apply_patch tool should serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_operation_create_file_round_trip() {
    // Spec (openai-responses-api-spec.md §ApplyPatchCall L242):
    // `CreateFile { diff, path, type: "create_file" }`.
    let payload = json!({
        "type": "create_file",
        "diff": "+ hello world\n",
        "path": "src/greeting.rs",
    });

    let op: ApplyPatchOperation =
        serde_json::from_value(payload.clone()).expect("create_file should deserialize");
    match &op {
        ApplyPatchOperation::CreateFile { diff, path } => {
            assert_eq!(diff, "+ hello world\n");
            assert_eq!(path, "src/greeting.rs");
        }
        other => panic!("expected CreateFile, got {other:?}"),
    }

    let serialized = serde_json::to_value(&op).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_operation_delete_file_round_trip() {
    // Spec (openai-responses-api-spec.md §ApplyPatchCall L243):
    // `DeleteFile { path, type: "delete_file" }`. No `diff` — whole file is
    // removed, so the payload must deserialize cleanly without one and
    // serialize back without a spurious diff field.
    let payload = json!({
        "type": "delete_file",
        "path": "src/obsolete.rs",
    });

    let op: ApplyPatchOperation =
        serde_json::from_value(payload.clone()).expect("delete_file should deserialize");
    match &op {
        ApplyPatchOperation::DeleteFile { path } => {
            assert_eq!(path, "src/obsolete.rs");
        }
        other => panic!("expected DeleteFile, got {other:?}"),
    }

    let serialized = serde_json::to_value(&op).expect("serialize");
    assert_eq!(serialized, payload);
    // Ensure no stray `diff` key is emitted for delete_file operations.
    assert!(
        serialized.get("diff").is_none(),
        "delete_file must not emit a diff field"
    );
}

#[test]
fn simple_input_message_with_id_does_not_match_item_reference() {
    // Regression: `ItemReference` is declared after `SimpleInputMessage` in
    // the untagged fallback chain, so a `{id, role, content}` payload — the
    // id-carrying shape of [`ResponseInputOutputItem::SimpleInputMessage`] —
    // lands in `SimpleInputMessage` first and does NOT silently drop
    // `role` / `content` by matching the bare-`id` `ItemReference` arm
    // behind it.
    let payload = json!({
        "id": "msg_abc123",
        "role": "user",
        "content": "hello",
    });
    let item: ResponseInputOutputItem =
        serde_json::from_value(payload).expect("message-shaped payload with id must deserialize");
    match &item {
        ResponseInputOutputItem::SimpleInputMessage { role, content, .. } => {
            assert_eq!(role, "user");
            assert!(matches!(content, StringOrContentParts::String(s) if s == "hello"));
        }
        other => panic!("expected SimpleInputMessage, got {other:?}"),
    }
}

#[test]
fn test_apply_patch_operation_update_file_round_trip() {
    // Spec (openai-responses-api-spec.md §ApplyPatchCall L244):
    // `UpdateFile { diff, path, type: "update_file" }`.
    let payload = json!({
        "type": "update_file",
        "diff": "@@ -1,1 +1,1 @@\n-old\n+new\n",
        "path": "src/main.rs",
    });

    let op: ApplyPatchOperation =
        serde_json::from_value(payload.clone()).expect("update_file should deserialize");
    match &op {
        ApplyPatchOperation::UpdateFile { diff, path } => {
            assert_eq!(diff, "@@ -1,1 +1,1 @@\n-old\n+new\n");
            assert_eq!(path, "src/main.rs");
        }
        other => panic!("expected UpdateFile, got {other:?}"),
    }

    let serialized = serde_json::to_value(&op).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_call_input_item_round_trip() {
    // Spec (openai-responses-api-spec.md §ApplyPatchCall L240-L246):
    // `{ call_id, operation, status, type, id }` with
    // `status: "in_progress" | "completed"` and `type: "apply_patch_call"`.
    let payload = json!({
        "type": "apply_patch_call",
        "call_id": "call_patch_001",
        "operation": {
            "type": "update_file",
            "diff": "@@ -1 +1 @@\n-a\n+b\n",
            "path": "src/lib.rs",
        },
        "status": "completed",
        "id": "apc_01",
    });

    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("apply_patch_call should deserialize");
    match &item {
        ResponseInputOutputItem::ApplyPatchCall {
            call_id,
            operation,
            status,
            id,
        } => {
            assert_eq!(call_id, "call_patch_001");
            assert_eq!(*status, ApplyPatchCallStatus::Completed);
            assert_eq!(id.as_deref(), Some("apc_01"));
            match operation {
                ApplyPatchOperation::UpdateFile { diff, path } => {
                    assert_eq!(diff, "@@ -1 +1 @@\n-a\n+b\n");
                    assert_eq!(path, "src/lib.rs");
                }
                other => panic!("expected UpdateFile operation, got {other:?}"),
            }
        }
        other => panic!("expected ApplyPatchCall, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_call_input_item_in_progress_status_and_omitted_id() {
    // Spec (L245): status `"in_progress"` is also legal. `id` is
    // `Option<String>` on the input side, so a newly-minted client-side call
    // may omit it; `skip_serializing_if` keeps it off the wire.
    let payload = json!({
        "type": "apply_patch_call",
        "call_id": "call_patch_002",
        "operation": {"type": "delete_file", "path": "old.rs"},
        "status": "in_progress",
    });

    let item: ResponseInputOutputItem =
        serde_json::from_value(payload.clone()).expect("apply_patch_call should deserialize");
    match &item {
        ResponseInputOutputItem::ApplyPatchCall {
            call_id,
            operation,
            status,
            id,
        } => {
            assert_eq!(call_id, "call_patch_002");
            assert_eq!(*status, ApplyPatchCallStatus::InProgress);
            assert!(id.is_none(), "id should be absent when omitted on the wire");
            assert!(matches!(operation, ApplyPatchOperation::DeleteFile { .. }));
        }
        other => panic!("expected ApplyPatchCall, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
    assert!(
        serialized.get("id").is_none(),
        "omitted id must not be serialized back as null",
    );
}

#[test]
fn test_apply_patch_call_output_input_item_completed_round_trip() {
    // Spec (openai-responses-api-spec.md §ApplyPatchCallOutput L248-L251):
    // `{ call_id, status, type, id, output }` with
    // `status: "completed" | "failed"` and `output: optional string`.
    let payload = json!({
        "type": "apply_patch_call_output",
        "call_id": "call_patch_001",
        "status": "completed",
        "id": "apco_01",
        "output": "1 file updated",
    });

    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("apply_patch_call_output should deserialize");
    match &item {
        ResponseInputOutputItem::ApplyPatchCallOutput {
            call_id,
            status,
            id,
            output,
        } => {
            assert_eq!(call_id, "call_patch_001");
            assert_eq!(*status, ApplyPatchCallOutputStatus::Completed);
            assert_eq!(id.as_deref(), Some("apco_01"));
            assert_eq!(output.as_deref(), Some("1 file updated"));
        }
        other => panic!("expected ApplyPatchCallOutput, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_call_output_input_item_failed_without_output() {
    // Spec (L251): `output` is optional — a `failed` output with no log text
    // must round-trip without emitting an explicit `null`.
    let payload = json!({
        "type": "apply_patch_call_output",
        "call_id": "call_patch_003",
        "status": "failed",
    });

    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("apply_patch_call_output without output should deserialize");
    match &item {
        ResponseInputOutputItem::ApplyPatchCallOutput {
            call_id,
            status,
            id,
            output,
        } => {
            assert_eq!(call_id, "call_patch_003");
            assert_eq!(*status, ApplyPatchCallOutputStatus::Failed);
            assert!(id.is_none());
            assert!(output.is_none());
        }
        other => panic!("expected ApplyPatchCallOutput, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
    assert!(serialized.get("id").is_none());
    assert!(serialized.get("output").is_none());
}

#[test]
fn test_apply_patch_call_output_item_round_trip() {
    // Spec: output-side mirror of the input variant. `id` is always populated
    // by the server on emit, so it appears unconditionally on the wire.
    let payload = json!({
        "type": "apply_patch_call",
        "id": "apc_out_01",
        "call_id": "call_patch_004",
        "operation": {
            "type": "create_file",
            "diff": "+new line\n",
            "path": "src/new.rs",
        },
        "status": "completed",
    });

    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("apply_patch_call output item should deserialize");
    match &item {
        ResponseOutputItem::ApplyPatchCall {
            id,
            call_id,
            operation,
            status,
        } => {
            assert_eq!(id, "apc_out_01");
            assert_eq!(call_id, "call_patch_004");
            assert_eq!(*status, ApplyPatchCallStatus::Completed);
            assert!(matches!(operation, ApplyPatchOperation::CreateFile { .. }));
        }
        other => panic!("expected ResponseOutputItem::ApplyPatchCall, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_call_output_response_item_round_trip() {
    // Spec: output-side mirror of `ApplyPatchCallOutput`. `id` is required on
    // the output wire.
    let payload = json!({
        "type": "apply_patch_call_output",
        "id": "apco_out_01",
        "call_id": "call_patch_005",
        "status": "failed",
        "output": "merge conflict at src/lib.rs:42",
    });

    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("apply_patch_call_output output item should deserialize");
    match &item {
        ResponseOutputItem::ApplyPatchCallOutput {
            id,
            call_id,
            status,
            output,
        } => {
            assert_eq!(id, "apco_out_01");
            assert_eq!(call_id, "call_patch_005");
            assert_eq!(*status, ApplyPatchCallOutputStatus::Failed);
            assert_eq!(output.as_deref(), Some("merge conflict at src/lib.rs:42"));
        }
        other => panic!("expected ResponseOutputItem::ApplyPatchCallOutput, got {other:?}"),
    }

    let serialized = serde_json::to_value(&item).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_operation_rejects_unknown_fields() {
    // P5 fail-fast contract: extra fields on an operation variant must be
    // rejected rather than silently dropped. `deny_unknown_fields` on the
    // `ApplyPatchOperation` enum enforces this per-variant.
    //
    // Regression for CodeRabbit feedback on PR #1339: a `delete_file`
    // operation carrying a stray `diff` key would otherwise deserialize
    // successfully and lose the foreign field, masking client bugs that
    // send the wrong operation shape.
    let stray_diff_on_delete = json!({
        "type": "delete_file",
        "path": "src/obsolete.rs",
        "diff": "should not be accepted",
    });
    assert!(
        serde_json::from_value::<ApplyPatchOperation>(stray_diff_on_delete).is_err(),
        "delete_file must reject a stray `diff` field per deny_unknown_fields"
    );

    // A stray foreign key on a variant that does carry `diff` / `path` must
    // also be rejected — deny_unknown_fields is not relaxed for the
    // create_file / update_file shapes.
    let foreign_key_on_create = json!({
        "type": "create_file",
        "diff": "+ hi\n",
        "path": "src/new.rs",
        "not_a_field": "nope",
    });
    assert!(
        serde_json::from_value::<ApplyPatchOperation>(foreign_key_on_create).is_err(),
        "create_file must reject foreign keys per deny_unknown_fields"
    );

    let foreign_key_on_update = json!({
        "type": "update_file",
        "diff": "@@ -1 +1 @@\n-a\n+b\n",
        "path": "src/main.rs",
        "mode": "force",
    });
    assert!(
        serde_json::from_value::<ApplyPatchOperation>(foreign_key_on_update).is_err(),
        "update_file must reject foreign keys per deny_unknown_fields"
    );

    // Sanity: the spec-canonical shapes still deserialize cleanly after
    // the tightening.
    for payload in [
        json!({"type": "create_file", "diff": "+ hi\n", "path": "src/new.rs"}),
        json!({"type": "delete_file", "path": "src/old.rs"}),
        json!({"type": "update_file", "diff": "@@ -1 +1 @@\n-a\n+b\n", "path": "src/main.rs"}),
    ] {
        assert!(
            serde_json::from_value::<ApplyPatchOperation>(payload.clone()).is_ok(),
            "spec-canonical shape must still deserialize: {payload}"
        );
    }
}

#[test]
fn test_apply_patch_tool_choice_round_trip() {
    // Spec (openai-responses-api-spec.md §tool_choice L424):
    // `ToolChoiceApplyPatch { type: "apply_patch" }`. Pairs with the
    // `ResponseTool::ApplyPatch` declaration so callers can force usage.
    let payload = json!({"type": "apply_patch"});
    let choice: ResponsesToolChoice = serde_json::from_value(payload.clone())
        .expect("apply_patch tool_choice should deserialize");
    match &choice {
        ResponsesToolChoice::ApplyPatch { tool_type } => {
            assert_eq!(*tool_type, ApplyPatchToolChoiceTag::ApplyPatch);
        }
        other => panic!("expected ResponsesToolChoice::ApplyPatch, got {other:?}"),
    }

    let serialized = serde_json::to_value(&choice).expect("serialize");
    assert_eq!(serialized, payload);
}

#[test]
fn test_apply_patch_request_validate_accepts_relaxed_shapes() {
    // Regression for CodeRabbit feedback on PR #1339: the `validate_input_item`
    // arms for `ApplyPatchCall` and `ApplyPatchCallOutput` intentionally do
    // no content checking — the spec permits empty/absent `output` on the
    // call-output side (L251), and the `ApplyPatchOperation` enum
    // structurally enforces the `diff` / `path` shape so an empty `diff`
    // string on `create_file` / `update_file` is spec-legal. This test
    // locks those relaxed branches in so they cannot regress back to a
    // stricter non-empty check.
    //
    // The cross-parameter validator (see `validate_response_request`)
    // requires at least one Message/SimpleInputMessage alongside any tool
    // item, so every fixture below pairs the apply_patch item with a plain
    // user message to isolate the apply_patch branch under test.
    use validator::Validate;

    let user_msg = json!({"role": "user", "content": "hi"});

    // (a) apply_patch_call_output with `output: ""` — empty-string log
    // must validate cleanly because the output-empty path was intentionally
    // omitted (cf. the `custom_tool_call_output_empty` check we did NOT
    // mirror here).
    let empty_output: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg.clone(),
            {
                "type": "apply_patch_call_output",
                "call_id": "call_patch_1",
                "status": "completed",
                "output": "",
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        empty_output.validate().is_ok(),
        "empty ApplyPatchCallOutput.output string must pass validation"
    );

    // (b) apply_patch_call_output with `output` omitted entirely — spec
    // L251 marks `output` as optional, so a no-log `failed` output must
    // validate without raising.
    let omitted_output: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg.clone(),
            {
                "type": "apply_patch_call_output",
                "call_id": "call_patch_2",
                "status": "failed",
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        omitted_output.validate().is_ok(),
        "omitted ApplyPatchCallOutput.output must pass validation"
    );

    // (c) apply_patch_call UpdateFile with `diff: ""` — an empty-diff
    // update is structurally legal (the `ApplyPatchOperation` enum only
    // enforces presence of the fields, not their contents) and the
    // validator intentionally does not content-check the operation.
    let empty_update_diff: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": [
            user_msg,
            {
                "type": "apply_patch_call",
                "call_id": "call_patch_3",
                "operation": {
                    "type": "update_file",
                    "diff": "",
                    "path": "src/main.rs"
                },
                "status": "completed"
            }
        ]
    }))
    .expect("request should deserialize");
    assert!(
        empty_update_diff.validate().is_ok(),
        "empty ApplyPatchCall.operation.diff on update_file must pass validation"
    );
}

/// `ResponseInputOutputItem::LocalShellCallOutput` carries a string
/// `output` + optional `status`. Spec
/// (openai-responses-api-spec.md §LocalShellCallOutput L224-226):
/// `{ id, output, type, status }`.
#[test]
fn local_shell_call_output_input_item_round_trips_spec_shape() {
    // With status present.
    let payload_with_status = json!({
        "type": "local_shell_call_output",
        "id": "lso_1",
        "output": "{\"exit_code\":0,\"stdout\":\"hello\\n\",\"stderr\":\"\"}",
        "status": "completed"
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload_with_status.clone())
        .expect("local_shell_call_output should deserialize");
    match &item {
        ResponseInputOutputItem::LocalShellCallOutput { id, output, status } => {
            assert_eq!(id, "lso_1");
            assert!(output.contains("exit_code"));
            assert_eq!(*status, Some(LocalShellCallStatus::Completed));
        }
        other => panic!("expected LocalShellCallOutput, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&item).expect("serialize"),
        payload_with_status,
    );

    // Without status — SDK v2.8.1 types the field as Optional.
    let payload_no_status = json!({
        "type": "local_shell_call_output",
        "id": "lso_2",
        "output": "raw log"
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload_no_status.clone())
        .expect("local_shell_call_output without status should deserialize");
    match &item {
        ResponseInputOutputItem::LocalShellCallOutput { id, output, status } => {
            assert_eq!(id, "lso_2");
            assert_eq!(output, "raw log");
            assert!(status.is_none());
        }
        other => panic!("expected LocalShellCallOutput, got {other:?}"),
    }
    // No `"status": null` should leak via skip_serializing_if.
    assert_eq!(
        serde_json::to_value(&item).expect("serialize"),
        payload_no_status,
    );
}

/// Output-side mirror: `ResponseOutputItem::LocalShellCall` +
/// `LocalShellCallOutput` round-trip the same on-wire shape the input-side
/// variants carry. Spec (openai-responses-api-spec.md L507-516) lists both
/// under `output: array of ResponseOutputItem`.
#[test]
fn local_shell_output_item_variants_round_trip_spec_shape() {
    let call_payload = json!({
        "type": "local_shell_call",
        "id": "ls_out_1",
        "call_id": "call_ls_out_1",
        "action": {
            "type": "exec",
            "command": ["ls", "-la"],
            "env": {}
        },
        "status": "incomplete"
    });
    let call_item: ResponseOutputItem = serde_json::from_value(call_payload.clone())
        .expect("local_shell_call output item should deserialize");
    match &call_item {
        ResponseOutputItem::LocalShellCall {
            id,
            call_id,
            action,
            status,
        } => {
            assert_eq!(id, "ls_out_1");
            assert_eq!(call_id, "call_ls_out_1");
            assert_eq!(*status, LocalShellCallStatus::Incomplete);
            match action {
                LocalShellExec::Exec { command, .. } => {
                    assert_eq!(command, &vec!["ls".to_string(), "-la".to_string()]);
                }
            }
        }
        other => panic!("expected ResponseOutputItem::LocalShellCall, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&call_item).expect("serialize"),
        call_payload,
    );

    let output_payload = json!({
        "type": "local_shell_call_output",
        "id": "lso_out_1",
        "output": "{\"exit_code\":0}",
        "status": "completed"
    });
    let output_item: ResponseOutputItem = serde_json::from_value(output_payload.clone())
        .expect("local_shell_call_output output item should deserialize");
    match &output_item {
        ResponseOutputItem::LocalShellCallOutput { id, output, status } => {
            assert_eq!(id, "lso_out_1");
            assert!(output.contains("exit_code"));
            assert_eq!(*status, Some(LocalShellCallStatus::Completed));
        }
        other => panic!("expected ResponseOutputItem::LocalShellCallOutput, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&output_item).expect("serialize"),
        output_payload,
    );
}

// ---------------------------------------------------------------------------
// T11: hosted-MCP protocol-surface coverage
//
// Spec refs:
//   .claude/_audit/openai-responses-api-spec.md L441-445 (McpTool: allowed_tools
//   union, connector_id, defer_loading), L253-255 (McpListTools), L264-266
//   (McpCall). SDK reference: openai==2.8.1 `types/responses/tool.py::Mcp`
//   (connector_id literal set, allowed_tools union) and
//   `types/responses/response_input_item.py::{McpCall, McpListTools}`.
// ---------------------------------------------------------------------------

/// Backward compat: pre-T11 `allowed_tools: ["foo","bar"]` wire shape still
/// deserializes via the untagged `McpAllowedTools::List` variant. Verifies the
/// Postel-style union migration is wire-compatible with existing callers.
#[test]
fn test_mcp_tool_allowed_tools_list_backward_compat() {
    let tool: McpTool = serde_json::from_value(json!({
        "server_label": "deepwiki",
        "allowed_tools": ["ask_question", "read_wiki_structure"]
    }))
    .expect("legacy list form should deserialize into McpAllowedTools::List");

    match &tool.allowed_tools {
        Some(McpAllowedTools::List(names)) => {
            assert_eq!(
                names,
                &vec![
                    "ask_question".to_string(),
                    "read_wiki_structure".to_string()
                ]
            );
        }
        other => panic!("expected McpAllowedTools::List, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("serialize");
    assert_eq!(
        serialized["allowed_tools"],
        json!(["ask_question", "read_wiki_structure"])
    );
}

/// `allowed_tools` object filter with only `read_only` set round-trips into
/// `McpAllowedTools::Filter`.
#[test]
fn test_mcp_tool_allowed_tools_filter_read_only_round_trip() {
    let payload = json!({
        "server_label": "deepwiki",
        "allowed_tools": { "read_only": true }
    });
    let tool: McpTool = serde_json::from_value(payload.clone())
        .expect("filter form with read_only should deserialize");

    match &tool.allowed_tools {
        Some(McpAllowedTools::Filter(filter)) => {
            assert_eq!(filter.read_only, Some(true));
            assert_eq!(filter.tool_names, None);
        }
        other => panic!("expected McpAllowedTools::Filter, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("serialize");
    assert_eq!(serialized["allowed_tools"], json!({ "read_only": true }));
}

/// Typoed keys on the `McpToolFilter` branch fail fast rather than silently
/// collapsing to an empty filter (which would broaden tool exposure downstream).
///
/// `McpAllowedTools` is `#[serde(untagged)]`, so a typoed object fails the
/// `Filter` arm (via `deny_unknown_fields`) and the `List` arm (wrong type),
/// surfacing serde's generic "did not match any variant" rejection. The
/// important guarantee is that the payload is rejected rather than silently
/// deserialized into an empty filter.
#[test]
fn test_mcp_tool_allowed_tools_filter_rejects_unknown_fields() {
    let err = serde_json::from_value::<McpTool>(json!({
        "server_label": "deepwiki",
        "allowed_tools": { "tool_namse": ["x"] }, // typo
    }))
    .expect_err("typoed filter key must not deserialize");
    // Regression guard: under the pre-T11 `Option<Vec<String>>` wire shape the
    // same payload was also rejected (wrong type). The union migration MUST
    // preserve that rejection — empty filters would broaden MCP tool exposure.
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "serde should surface some rejection message, got empty string",
    );
}

/// `allowed_tools` object filter with only `tool_names` set round-trips into
/// `McpAllowedTools::Filter`.
#[test]
fn test_mcp_tool_allowed_tools_filter_tool_names_round_trip() {
    let payload = json!({
        "server_label": "deepwiki",
        "allowed_tools": { "tool_names": ["a", "b"] }
    });
    let tool: McpTool = serde_json::from_value(payload.clone())
        .expect("filter form with tool_names should deserialize");

    match &tool.allowed_tools {
        Some(McpAllowedTools::Filter(filter)) => {
            assert_eq!(filter.read_only, None);
            assert_eq!(
                filter.tool_names.as_deref(),
                Some(["a".to_string(), "b".to_string()].as_slice())
            );
        }
        other => panic!("expected McpAllowedTools::Filter, got {other:?}"),
    }

    let serialized = serde_json::to_value(&tool).expect("serialize");
    assert_eq!(
        serialized["allowed_tools"],
        json!({ "tool_names": ["a", "b"] })
    );
}

/// `connector_id` deserializes into the [`McpConnectorId`] enum; all eight spec
/// literal variants map correctly and round-trip back to the same wire string.
#[test]
fn test_mcp_tool_connector_id_round_trip() {
    // Map wire value -> enum variant for every connector in SDK 2.8.1
    // `types/responses/tool.py::Mcp.connector_id`.
    let cases: &[(&str, McpConnectorId)] = &[
        ("connector_dropbox", McpConnectorId::Dropbox),
        ("connector_gmail", McpConnectorId::Gmail),
        ("connector_googlecalendar", McpConnectorId::GoogleCalendar),
        ("connector_googledrive", McpConnectorId::GoogleDrive),
        ("connector_microsoftteams", McpConnectorId::MicrosoftTeams),
        ("connector_outlookcalendar", McpConnectorId::OutlookCalendar),
        ("connector_outlookemail", McpConnectorId::OutlookEmail),
        ("connector_sharepoint", McpConnectorId::SharePoint),
    ];

    for (wire, expected) in cases {
        let tool: McpTool = serde_json::from_value(json!({
            "server_label": "ctr",
            "connector_id": wire,
        }))
        .unwrap_or_else(|e| panic!("connector_id={wire} should deserialize: {e}"));
        assert_eq!(tool.connector_id, Some(*expected), "connector_id={wire}");

        let serialized = serde_json::to_value(&tool).expect("serialize");
        assert_eq!(serialized["connector_id"], json!(wire));
    }
}

/// `defer_loading` optional bool round-trips; unknown `connector_id` values are
/// rejected by the enum parser (no silent `Unknown` catch-all).
#[test]
fn test_mcp_tool_defer_loading_round_trip_and_unknown_connector_rejected() {
    let tool: McpTool = serde_json::from_value(json!({
        "server_label": "deepwiki",
        "defer_loading": true,
    }))
    .expect("defer_loading should deserialize");
    assert_eq!(tool.defer_loading, Some(true));
    let serialized = serde_json::to_value(&tool).expect("serialize");
    assert_eq!(serialized["defer_loading"], json!(true));

    // Unknown connector_id must fail-fast (no Unknown silent fallback).
    let err = serde_json::from_value::<McpTool>(json!({
        "server_label": "deepwiki",
        "connector_id": "connector_totally_made_up",
    }))
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("connector_"),
        "expected connector enum error, got: {msg}"
    );
}

/// Legacy McpTool payloads (no `connector_id`, `defer_loading`, or
/// `allowed_tools`) still deserialize cleanly with all new fields as `None`.
#[test]
fn test_mcp_tool_legacy_payload_still_deserializes() {
    let tool: McpTool = serde_json::from_value(json!({
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.example",
    }))
    .expect("legacy payload should deserialize");
    assert_eq!(tool.allowed_tools, None);
    assert_eq!(tool.connector_id, None);
    assert_eq!(tool.defer_loading, None);

    // Serialization should not emit the absent optional fields.
    let serialized = serde_json::to_value(&tool).expect("serialize");
    assert!(serialized.get("allowed_tools").is_none());
    assert!(serialized.get("connector_id").is_none());
    assert!(serialized.get("defer_loading").is_none());
}

/// Spec contract (openai-responses-api-spec.md L445): `server_url` XOR
/// `connector_id` is required. A payload that sets both MUST be rejected by
/// validation so downstream target resolution is unambiguous.
#[test]
fn test_validate_tools_mcp_server_url_and_connector_id_conflict() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "gpt-5.4",
        "input": "hi",
        "tools": [{
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.example",
            "connector_id": "connector_dropbox",
        }],
    }))
    .expect("deserialize");
    let err = request
        .validate()
        .expect_err("server_url + connector_id must fail validation");
    assert!(
        format!("{err:?}").contains("mcp_tool_conflicting_targets"),
        "expected mcp_tool_conflicting_targets, got: {err:?}"
    );
}

/// Input-item `mcp_call` minimal payload (required-only) round-trips cleanly.
/// Required fields per SDK 2.8.1 `response_input_item.py::McpCall`: `id`,
/// `arguments`, `name`, `server_label`, `type`.
#[test]
fn test_mcp_call_input_item_minimal_round_trip() {
    let payload = json!({
        "type": "mcp_call",
        "id": "mcp_call_1",
        "arguments": "{\"query\":\"hello\"}",
        "name": "ask_question",
        "server_label": "deepwiki",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("mcp_call minimal input item should deserialize");

    match &item {
        ResponseInputOutputItem::McpCall {
            id,
            arguments,
            name,
            server_label,
            approval_request_id,
            error,
            output,
            status,
        } => {
            assert_eq!(id, "mcp_call_1");
            assert_eq!(arguments, "{\"query\":\"hello\"}");
            assert_eq!(name, "ask_question");
            assert_eq!(server_label, "deepwiki");
            assert_eq!(approval_request_id, &None);
            assert_eq!(error, &None);
            assert_eq!(output, &None);
            assert_eq!(status, &None);
        }
        other => panic!("expected McpCall, got {other:?}"),
    }

    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// Input-item `mcp_call` full payload including all optional fields round-trips
/// byte-for-byte.
#[test]
fn test_mcp_call_input_item_full_round_trip() {
    let payload = json!({
        "type": "mcp_call",
        "id": "mcp_call_2",
        "arguments": "{\"query\":\"hello\"}",
        "name": "ask_question",
        "server_label": "deepwiki",
        "approval_request_id": "apr_1",
        "error": null,
        "output": "{\"answer\":\"42\"}",
        "status": "completed",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("mcp_call full input item should deserialize");

    match &item {
        ResponseInputOutputItem::McpCall {
            approval_request_id,
            output,
            status,
            ..
        } => {
            assert_eq!(approval_request_id.as_deref(), Some("apr_1"));
            assert_eq!(output.as_deref(), Some("{\"answer\":\"42\"}"));
            assert_eq!(status.as_deref(), Some("completed"));
        }
        other => panic!("expected McpCall, got {other:?}"),
    }

    // The `error: null` field skips serializing, so compare against the
    // absent-error shape on re-serialization.
    let expected_serialized = json!({
        "type": "mcp_call",
        "id": "mcp_call_2",
        "arguments": "{\"query\":\"hello\"}",
        "name": "ask_question",
        "server_label": "deepwiki",
        "approval_request_id": "apr_1",
        "output": "{\"answer\":\"42\"}",
        "status": "completed",
    });
    assert_eq!(
        serde_json::to_value(&item).expect("serialize"),
        expected_serialized
    );
}

/// Structured `mcp_call.error` variants round-trip byte-for-byte on both the
/// input and output item shapes.
#[test]
fn test_mcp_call_structured_error_round_trips() {
    let errors = [
        json!({"type": "mcp_protocol_error", "code": -32601, "message": "method not found"}),
        json!({"type": "mcp_tool_execution_error", "content": "rate limited"}),
        json!({"type": "mcp_tool_execution_error", "content": [{"type": "text", "text": "boom"}]}),
        json!({"type": "http_error", "code": 502, "message": "bad gateway"}),
    ];
    for error in errors {
        let input_payload = json!({
            "type": "mcp_call",
            "id": "mcp_call_3",
            "arguments": "{}",
            "name": "ask_question",
            "server_label": "deepwiki",
            "error": error,
        });
        let item: ResponseInputOutputItem = serde_json::from_value(input_payload.clone())
            .unwrap_or_else(|e| panic!("structured error should deserialize: {e}"));
        assert_eq!(
            serde_json::to_value(&item).expect("serialize"),
            input_payload
        );

        let output_payload = json!({
            "type": "mcp_call",
            "id": "mcp_call_3",
            "status": "failed",
            "arguments": "{}",
            "name": "ask_question",
            "output": "",
            "server_label": "deepwiki",
            "error": error,
        });
        let item: ResponseOutputItem = serde_json::from_value(output_payload.clone())
            .unwrap_or_else(|e| panic!("structured error should deserialize: {e}"));
        assert_eq!(
            serde_json::to_value(&item).expect("serialize"),
            output_payload
        );
    }
}

/// Legacy plain-string `mcp_call.error` (pre-union wire shape, still present in
/// stored history) coerces to the tool-execution variant so replay serializes
/// the object form OpenAI now requires.
#[test]
fn test_mcp_call_legacy_string_error_coerces_to_structured() {
    let item: ResponseInputOutputItem = serde_json::from_value(json!({
        "type": "mcp_call",
        "id": "mcp_call_4",
        "arguments": "{}",
        "name": "brave_web_search",
        "server_label": "brave",
        "status": "failed",
        "error": "Tool call failed: rate limited",
    }))
    .expect("legacy string error should still deserialize");

    match &item {
        ResponseInputOutputItem::McpCall { error, .. } => {
            assert_eq!(
                error,
                &Some(McpToolCallError::execution(
                    "Tool call failed: rate limited"
                ))
            );
        }
        other => panic!("expected McpCall, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&item).expect("serialize")["error"],
        json!({"type": "mcp_tool_execution_error", "content": "Tool call failed: rate limited"})
    );

    let item: ResponseOutputItem = serde_json::from_value(json!({
        "type": "mcp_call",
        "id": "mcp_call_4",
        "status": "failed",
        "arguments": "{}",
        "name": "brave_web_search",
        "output": "",
        "server_label": "brave",
        "error": "Tool call failed: rate limited",
    }))
    .expect("legacy string error should still deserialize");
    assert_eq!(
        serde_json::to_value(&item).expect("serialize")["error"],
        json!({"type": "mcp_tool_execution_error", "content": "Tool call failed: rate limited"})
    );
}

/// Input-item `mcp_list_tools` round-trips with a non-empty `tools` array; the
/// optional `error` field is omitted on serialize when absent.
#[test]
fn test_mcp_list_tools_input_item_round_trip() {
    let payload = json!({
        "type": "mcp_list_tools",
        "id": "mcplist_1",
        "server_label": "deepwiki",
        "tools": [
            {
                "name": "ask_question",
                "description": "Ask the wiki a question",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "read_wiki_structure",
                "input_schema": {"type": "object"},
            }
        ],
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("mcp_list_tools input item should deserialize");

    match &item {
        ResponseInputOutputItem::McpListTools {
            id,
            server_label,
            tools,
            error,
        } => {
            assert_eq!(id, "mcplist_1");
            assert_eq!(server_label, "deepwiki");
            assert_eq!(tools.len(), 2);
            assert_eq!(tools[0].name, "ask_question");
            assert_eq!(
                tools[0].description.as_deref(),
                Some("Ask the wiki a question")
            );
            assert_eq!(tools[1].name, "read_wiki_structure");
            assert_eq!(tools[1].description, None);
            assert_eq!(error, &None);
        }
        other => panic!("expected McpListTools, got {other:?}"),
    }

    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}

/// Symmetry check: `ResponseOutputItem::McpListTools` carries the same
/// optional `error` field as its input-side counterpart so a failing list-tools
/// item can round-trip emit → replay losslessly.
#[test]
fn test_mcp_list_tools_output_item_error_round_trip() {
    let payload = json!({
        "type": "mcp_list_tools",
        "id": "mcplist_out_1",
        "server_label": "deepwiki",
        "tools": [],
        "error": "server unreachable",
    });
    let item: ResponseOutputItem = serde_json::from_value(payload.clone())
        .expect("mcp_list_tools output item with error should deserialize");
    match &item {
        ResponseOutputItem::McpListTools { tools, error, .. } => {
            assert!(tools.is_empty());
            assert_eq!(error.as_deref(), Some("server unreachable"));
        }
        other => panic!("expected ResponseOutputItem::McpListTools, got {other:?}"),
    }
    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);

    // error absent: field is omitted rather than emitted as null.
    let payload_no_error = json!({
        "type": "mcp_list_tools",
        "id": "mcplist_out_2",
        "server_label": "deepwiki",
        "tools": [],
    });
    let item_no_error: ResponseOutputItem = serde_json::from_value(payload_no_error.clone())
        .expect("mcp_list_tools output item without error should deserialize");
    match &item_no_error {
        ResponseOutputItem::McpListTools { error, .. } => assert_eq!(error, &None),
        other => panic!("expected ResponseOutputItem::McpListTools, got {other:?}"),
    }
    assert_eq!(
        serde_json::to_value(&item_no_error).expect("serialize"),
        payload_no_error
    );
}

/// Input-item `mcp_list_tools` with a populated `error` field round-trips.
#[test]
fn test_mcp_list_tools_input_item_with_error_round_trip() {
    let payload = json!({
        "type": "mcp_list_tools",
        "id": "mcplist_2",
        "server_label": "deepwiki",
        "tools": [],
        "error": "server unreachable",
    });
    let item: ResponseInputOutputItem = serde_json::from_value(payload.clone())
        .expect("mcp_list_tools with error should deserialize");

    match &item {
        ResponseInputOutputItem::McpListTools { tools, error, .. } => {
            assert!(tools.is_empty());
            assert_eq!(error.as_deref(), Some("server unreachable"));
        }
        other => panic!("expected McpListTools, got {other:?}"),
    }

    assert_eq!(serde_json::to_value(&item).expect("serialize"), payload);
}
