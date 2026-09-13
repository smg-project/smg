//! Response transformer for MCP to OpenAI format conversion.

use openai_protocol::responses::{
    CodeInterpreterCallStatus, CodeInterpreterOutput, FileSearchCallStatus, FileSearchResult,
    ImageGenerationCallStatus, McpToolCallError, ResponseOutputItem, WebSearchAction,
    WebSearchCallStatus, WebSearchSource,
};
use tracing::warn;

use super::ResponseFormat;

/// Transform a `ToolExecutionOutput` into a `ResponseOutputItem` using a
/// pre-resolved `ResponseFormat`.
///
/// `response_format` must be resolved via the session's exposed-name map
/// (e.g. [`super::lookup_tool_format`]) — `output.tool_name` carries the
/// *invoked/exposed* name after session-side rewriting, so a fresh
/// registry lookup against `(output.server_key, output.tool_name)` would
/// miss for disambiguated names like `mcp_<server>_<tool>`.
///
/// On `is_error`: only `mcp_call` surfaces the failure (`status: "failed"`,
/// `error: Some(msg)`). The four hosted-builtin variants always emit
/// `status: "completed"` — that matches OpenAI cloud's de-facto behavior
/// (the cloud's native `web_search_call`/`code_interpreter_call`/etc.
/// don't emit `Failed` for soft failures like empty results or rate
/// limits, and many MCP servers set `isError: true` for those very cases).
/// The error context lives in the item's content, not in `status`.
pub fn transform_tool_output(
    output: &smg_mcp::ToolExecutionOutput,
    response_format: ResponseFormat,
) -> ResponseOutputItem {
    if output.is_error && response_format == ResponseFormat::Passthrough {
        return failed_mcp_call(output);
    }
    ResponseTransformer::transform(
        &output.output,
        response_format,
        &output.call_id,
        &output.server_label,
        &output.tool_name,
        &output.arguments_str,
    )
}

fn failed_mcp_call(output: &smg_mcp::ToolExecutionOutput) -> ResponseOutputItem {
    // The MCP `CallToolResult { is_error: true, .. }` path can leave
    // `error_message` unset while carrying the real failure text inside
    // `output.output` (typed text blocks). Fall back to that before the
    // generic placeholder so client-visible mcp_call.error stays useful.
    let err_msg = output
        .error_message
        .clone()
        .filter(|msg| !msg.is_empty())
        .or_else(|| {
            let text = ResponseTransformer::flatten_mcp_output(&output.output);
            (!text.is_empty()).then_some(text)
        })
        .unwrap_or_else(|| "Tool execution failed".to_string());
    ResponseOutputItem::McpCall {
        id: mcp_response_item_id(&output.call_id),
        status: "failed".to_string(),
        approval_request_id: None,
        arguments: output.arguments_str.clone(),
        error: Some(McpToolCallError::execution(err_msg)),
        name: output.tool_name.clone(),
        output: String::new(),
        server_label: output.server_label.clone(),
    }
}

/// Normalize an MCP response item id source into an external `mcp_call.id`.
///
/// The input may be an upstream output item id (`fc_*`), an internal call id
/// (`call_*`), or an already-normalized MCP id (`mcp_*`).
pub fn mcp_response_item_id(source_id: &str) -> String {
    if source_id.starts_with("mcp_") {
        return source_id.to_string();
    }

    if let Some(stripped) = source_id
        .strip_prefix("call_")
        .or_else(|| source_id.strip_prefix("fc_"))
    {
        return format!("mcp_{stripped}");
    }

    format!("mcp_{source_id}")
}

/// Extract non-null `openai_response` objects from embedded MCP text-block payloads.
pub fn extract_embedded_openai_responses(result: &serde_json::Value) -> Vec<serde_json::Value> {
    let text_blocks = result
        .as_array()
        .map(|items| items.as_slice())
        .unwrap_or(&[]);

    let mut openai_responses = Vec::new();

    for item in text_blocks {
        let Some(payload) = parse_text_block_payload(item) else {
            continue;
        };

        let Some(openai_response) = payload.get("openai_response") else {
            continue;
        };

        if !openai_response.is_null() {
            openai_responses.push(openai_response.clone());
        }
    }

    openai_responses
}

/// Fields SMG pulls out of an MCP `image_generation` tool result before
/// assembling the `image_generation_call` output item.
///
/// Using a named struct instead of a tuple keeps the extractor future-proof:
/// adding another optional metadata field does not touch the extractor's
/// signature or its call sites. All fields are `Option<String>` so they
/// serialize as absent (not `null`) when the MCP server did not surface them.
#[derive(Default)]
struct ImageGenerationFields {
    image_b64: Option<String>,
    revised_prompt: Option<String>,
    action: Option<String>,
    background: Option<String>,
    output_format: Option<String>,
    quality: Option<String>,
    size: Option<String>,
}

/// Transforms MCP CallToolResult to OpenAI Responses API output items.
pub struct ResponseTransformer;

impl ResponseTransformer {
    /// Transform an MCP result based on the configured response format.
    ///
    /// Returns a `ResponseOutputItem` from the protocols crate.
    pub fn transform(
        result: &serde_json::Value,
        format: ResponseFormat,
        tool_call_id: &str,
        server_label: &str,
        tool_name: &str,
        arguments: &str,
    ) -> ResponseOutputItem {
        match format {
            ResponseFormat::Passthrough => {
                Self::to_mcp_call(result, tool_call_id, server_label, tool_name, arguments)
            }
            ResponseFormat::WebSearchCall => {
                Self::to_web_search_call(result, tool_call_id, arguments)
            }
            ResponseFormat::CodeInterpreterCall => {
                Self::to_code_interpreter_call(result, tool_call_id)
            }
            ResponseFormat::FileSearchCall => Self::to_file_search_call(result, tool_call_id),
            ResponseFormat::ImageGenerationCall => {
                Self::to_image_generation_call(result, tool_call_id)
            }
        }
    }

    /// Transform to mcp_call output (passthrough).
    fn to_mcp_call(
        result: &serde_json::Value,
        tool_call_id: &str,
        server_label: &str,
        tool_name: &str,
        arguments: &str,
    ) -> ResponseOutputItem {
        ResponseOutputItem::McpCall {
            id: mcp_response_item_id(tool_call_id),
            status: "completed".to_string(),
            approval_request_id: None,
            arguments: arguments.to_string(),
            error: None,
            name: tool_name.to_string(),
            output: Self::flatten_mcp_output(result),
            server_label: server_label.to_string(),
        }
    }

    /// Flatten passthrough MCP results into the plain-string output shape used by OpenAI.
    fn flatten_mcp_output(result: &serde_json::Value) -> String {
        match result {
            serde_json::Value::String(text) => text.clone(),
            _ => {
                let mut text_parts = Vec::new();
                Self::collect_text_parts(result, &mut text_parts);
                if text_parts.is_empty() {
                    result.to_string()
                } else {
                    text_parts.join("\n")
                }
            }
        }
    }

    fn collect_text_parts(value: &serde_json::Value, text_parts: &mut Vec<String>) {
        match value {
            serde_json::Value::Array(items) => {
                for item in items {
                    Self::collect_text_parts(item, text_parts);
                }
            }
            serde_json::Value::Object(obj) => {
                if obj.get("type").and_then(|v| v.as_str()) == Some("text") {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(text.to_string());
                    }
                    return;
                }

                if obj.get("type").is_none() {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(text.to_string());
                        return;
                    }
                }

                if let Some(message) = obj.get("message").and_then(|v| v.as_str()) {
                    text_parts.push(message.to_string());
                    return;
                }

                if let Some(error) = obj.get("error") {
                    let before = text_parts.len();
                    Self::collect_text_parts(error, text_parts);
                    if text_parts.len() > before {
                        return;
                    }
                }

                if let Some(content) = obj.get("content") {
                    Self::collect_text_parts(content, text_parts);
                }
            }
            _ => {}
        }
    }

    /// Transform MCP web search results to OpenAI web_search_call format.
    fn to_web_search_call(
        result: &serde_json::Value,
        tool_call_id: &str,
        arguments: &str,
    ) -> ResponseOutputItem {
        let sources = Self::extract_web_sources(result);
        let queries = Self::extract_queries(result);
        let query = Self::extract_query_from_arguments(arguments);

        ResponseOutputItem::WebSearchCall {
            id: format!("ws_{tool_call_id}"),
            status: WebSearchCallStatus::Completed,
            action: WebSearchAction::Search {
                query,
                queries,
                sources,
            },
            // Populated only when the caller asks for `web_search_call.results`
            // via include[]; transformer leaves it unset so default omits the field.
            results: None,
        }
    }

    /// Transform MCP code interpreter results to OpenAI code_interpreter_call format.
    fn to_code_interpreter_call(
        result: &serde_json::Value,
        tool_call_id: &str,
    ) -> ResponseOutputItem {
        let obj = result.as_object();

        let container_id = obj
            .and_then(|o| o.get("container_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let code = obj
            .and_then(|o| o.get("code"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let outputs = Self::extract_code_outputs(result);

        ResponseOutputItem::CodeInterpreterCall {
            id: format!("ci_{tool_call_id}"),
            status: CodeInterpreterCallStatus::Completed,
            container_id,
            code,
            outputs: (!outputs.is_empty()).then_some(outputs),
        }
    }

    /// Transform MCP file search results to OpenAI file_search_call format.
    fn to_file_search_call(result: &serde_json::Value, tool_call_id: &str) -> ResponseOutputItem {
        let obj = result.as_object();

        let queries = Self::extract_queries(result);
        let results = Self::extract_file_results(result);

        ResponseOutputItem::FileSearchCall {
            id: format!("fs_{tool_call_id}"),
            status: FileSearchCallStatus::Completed,
            queries: if queries.is_empty() {
                obj.and_then(|o| o.get("query"))
                    .and_then(|v| v.as_str())
                    .map(|q| vec![q.to_string()])
                    .unwrap_or_default()
            } else {
                queries
            },
            results: (!results.is_empty()).then_some(results),
        }
    }

    /// Transform MCP image generation results to OpenAI image_generation_call format.
    ///
    /// Extracts fields from the MCP `CallToolResult`:
    /// - `result` / `image_base64` / `b64_json`: base64-encoded image bytes
    /// - `revised_prompt`: optional rewritten prompt preserved for replay
    /// - `action` / `background` / `output_format` / `quality` / `size`:
    ///   OpenAI-side output metadata. Forwarded verbatim when the MCP server
    ///   surfaces them so cloud passthrough does not drop them.
    ///
    /// Also probes embedded text blocks carrying `{"openai_response": {...}}`
    /// payloads for the same fields, mirroring the pattern used by
    /// `to_web_search_call`.
    fn to_image_generation_call(
        result: &serde_json::Value,
        tool_call_id: &str,
    ) -> ResponseOutputItem {
        let fields = Self::extract_image_generation_fields(result);

        ResponseOutputItem::ImageGenerationCall {
            id: format!("ig_{tool_call_id}"),
            action: fields.action,
            background: fields.background,
            output_format: fields.output_format,
            quality: fields.quality,
            // `result` is non-optional in the spec; fall back to an empty
            // string when the MCP server returns no image data so the shape
            // still matches `ResponseOutputItem::ImageGenerationCall`. Router
            // wiring is responsible for surfacing an error status when
            // `image_b64` is missing.
            result: fields.image_b64.unwrap_or_default(),
            revised_prompt: fields.revised_prompt,
            size: fields.size,
            status: ImageGenerationCallStatus::Completed,
        }
    }

    /// Extract the image-generation fields SMG surfaces from an MCP result
    /// payload. Returns a struct (not a tuple) so adding/removing fields
    /// does not ripple through call sites.
    fn extract_image_generation_fields(result: &serde_json::Value) -> ImageGenerationFields {
        let mut fields = ImageGenerationFields::default();

        // Merge helper: fills any still-unset output slot from an object.
        // Using mutable accumulation (instead of early-returning on first
        // match) prevents losing fields that an MCP server distributes
        // across multiple text blocks — e.g. `revised_prompt` in one and
        // `image_base64` in another. First occurrence wins for each slot.
        let mut update_fields = |obj: &serde_json::Map<String, serde_json::Value>| {
            if fields.image_b64.is_none() {
                fields.image_b64 = obj
                    .get("result")
                    .and_then(|v| v.as_str())
                    .or_else(|| obj.get("image_base64").and_then(|v| v.as_str()))
                    .or_else(|| obj.get("b64_json").and_then(|v| v.as_str()))
                    .map(String::from);
            }
            if fields.revised_prompt.is_none() {
                fields.revised_prompt = obj
                    .get("revised_prompt")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
            // Metadata passthrough: the five OpenAI-documented
            // `image_generation_call` output-item metadata fields. Each is
            // optional and independently extracted so an MCP server that
            // surfaces only some of them (e.g. just `size`) is preserved.
            if fields.action.is_none() {
                fields.action = obj.get("action").and_then(|v| v.as_str()).map(String::from);
            }
            if fields.background.is_none() {
                fields.background = obj
                    .get("background")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
            if fields.output_format.is_none() {
                fields.output_format = obj
                    .get("output_format")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
            if fields.quality.is_none() {
                fields.quality = obj
                    .get("quality")
                    .and_then(|v| v.as_str())
                    .map(String::from);
            }
            if fields.size.is_none() {
                fields.size = obj.get("size").and_then(|v| v.as_str()).map(String::from);
            }
        };

        // 1) Direct object fields: { result | image_base64 | b64_json, revised_prompt }
        if let Some(obj) = result.as_object() {
            update_fields(obj);
        }

        // 2) Array of MCP text blocks. An MCP server returning a JSON-shaped
        //    tool result is serialized by `call_result_to_json` as an array
        //    of text blocks (`[{"type":"text","text":"{...}"}, ...]`). Two
        //    text-block shapes are accepted:
        //
        //    a) `{"openai_response": {"result": ..., "revised_prompt": ...}}`
        //       (internal convention used by web_search_call and other
        //       SMG-authored MCP servers).
        //    b) Top-level fields: `{"result": ..., "revised_prompt": ...}`.
        //       This is the natural shape produced by servers that return
        //       a plain `dict` from their tool handler (e.g. FastMCP's
        //       `@tool` decorator wrapping a `dict` return value — see
        //       `e2e_test/infra/mock_mcp_server.py`). It also covers any
        //       third-party MCP server that emits the OpenAI Images API
        //       shape directly.
        //
        //    Both shapes are read so either style works interchangeably;
        //    first occurrence wins for each slot (matches the accumulator
        //    semantics in `update_fields`).
        if let Some(text_blocks) = result.as_array() {
            for item in text_blocks {
                let Some(payload) = parse_text_block_payload(item) else {
                    continue;
                };
                // First-occurrence-wins accumulation means the
                // `openai_response`-wrapped shape keeps priority over the
                // top-level shape when a single block carries both.
                if let Some(inner) = payload
                    .get("openai_response")
                    .filter(|value| !value.is_null())
                    .and_then(|value| value.as_object())
                {
                    update_fields(inner);
                }
                if let Some(obj) = payload.as_object() {
                    update_fields(obj);
                }
            }
        }

        fields
    }

    /// Extract web sources from MCP result.
    fn extract_web_sources(result: &serde_json::Value) -> Vec<WebSearchSource> {
        let mut sources = Vec::new();

        if result.is_array() {
            for openai_response in extract_embedded_openai_responses(result) {
                let Some(raw_sources) = openai_response.get("sources") else {
                    continue;
                };

                let source_items = raw_sources
                    .as_array()
                    .map(|items| items.as_slice())
                    .unwrap_or_else(|| {
                        warn!("Expected openai_response.sources to be an array");
                        &[]
                    });

                for source_item in source_items {
                    match Self::parse_web_source(source_item) {
                        Some(source) => sources.push(source),
                        None => warn!("Skipping malformed web source item"),
                    }
                }
            }

            if !sources.is_empty() {
                return sources;
            }
        }

        let fallback_items = result
            .as_array()
            .or_else(|| result.as_object()?.get("results")?.as_array())
            .map(|items| items.as_slice())
            .unwrap_or_default();

        for source_item in fallback_items {
            match Self::parse_web_source(source_item) {
                Some(source) => sources.push(source),
                None => warn!("Skipping malformed web source item"),
            }
        }

        sources
    }

    /// Parse a single web source from JSON.
    fn parse_web_source(item: &serde_json::Value) -> Option<WebSearchSource> {
        let obj = item.as_object()?;
        let url = obj.get("url").and_then(|v| v.as_str())?;
        let source_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("url");
        Some(WebSearchSource {
            source_type: source_type.to_string(),
            url: url.to_string(),
        })
    }

    /// Extract queries from MCP result.
    fn extract_queries(result: &serde_json::Value) -> Vec<String> {
        let mut queries = Vec::new();

        if let Some(text_blocks) = result.as_array() {
            for item in text_blocks {
                let Some(payload) = parse_text_block_payload(item) else {
                    continue;
                };

                let Some(raw_queries) = payload.get("queries") else {
                    continue;
                };

                let query_items = raw_queries
                    .as_array()
                    .map(|items| items.as_slice())
                    .unwrap_or_else(|| {
                        warn!("Expected queries to be an array");
                        &[]
                    });

                queries.extend(
                    query_items
                        .iter()
                        .filter_map(|query| query.as_str().map(String::from)),
                );
            }

            if !queries.is_empty() {
                return queries;
            }
        }

        result
            .as_object()
            .and_then(|obj| obj.get("queries"))
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|query| query.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or(queries)
    }

    fn extract_query_from_arguments(arguments: &str) -> Option<String> {
        let args_json = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
        args_json
            .as_object()
            .and_then(|obj| obj.get("query"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
    }

    /// Extract code interpreter outputs from MCP result.
    fn extract_code_outputs(result: &serde_json::Value) -> Vec<CodeInterpreterOutput> {
        let mut outputs = Vec::new();

        if let Some(obj) = result.as_object() {
            // Check for logs/stdout
            if let Some(logs) = obj.get("logs").and_then(|v| v.as_str()) {
                outputs.push(CodeInterpreterOutput::Logs {
                    logs: logs.to_string(),
                });
            }
            if let Some(stdout) = obj.get("stdout").and_then(|v| v.as_str()) {
                outputs.push(CodeInterpreterOutput::Logs {
                    logs: stdout.to_string(),
                });
            }

            // Check for image outputs
            if let Some(image_url) = obj.get("image_url").and_then(|v| v.as_str()) {
                outputs.push(CodeInterpreterOutput::Image {
                    url: image_url.to_string(),
                });
            }

            // Check for outputs array
            if let Some(out_array) = obj.get("outputs").and_then(|v| v.as_array()) {
                outputs.extend(out_array.iter().filter_map(|item| {
                    let item_obj = item.as_object()?;
                    match item_obj.get("type").and_then(|v| v.as_str())? {
                        "logs" => item_obj.get("logs").and_then(|v| v.as_str()).map(|logs| {
                            CodeInterpreterOutput::Logs {
                                logs: logs.to_string(),
                            }
                        }),
                        "image" => item_obj.get("url").and_then(|v| v.as_str()).map(|url| {
                            CodeInterpreterOutput::Image {
                                url: url.to_string(),
                            }
                        }),
                        _ => None,
                    }
                }));
            }
        }

        outputs
    }

    /// Extract file search results from MCP result.
    fn extract_file_results(result: &serde_json::Value) -> Vec<FileSearchResult> {
        result
            .as_object()
            .and_then(|obj| obj.get("results"))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(Self::parse_file_result).collect())
            .unwrap_or_default()
    }

    /// Parse a file search result from JSON.
    fn parse_file_result(item: &serde_json::Value) -> Option<FileSearchResult> {
        let obj = item.as_object()?;
        let file_id = obj.get("file_id").and_then(|v| v.as_str())?.to_string();
        let filename = obj.get("filename").and_then(|v| v.as_str())?.to_string();
        let text = obj.get("text").and_then(|v| v.as_str()).map(String::from);
        let score = obj.get("score").and_then(|v| v.as_f64()).map(|f| f as f32);

        Some(FileSearchResult {
            file_id,
            filename,
            text,
            score,
            attributes: None,
        })
    }
}

/// Strip the base64 `result` payload from an `ImageGenerationCall` output
/// item so stored multi-turn context does not balloon with large image bytes.
///
/// Per OpenAI spec the `result` field is required on the wire when the item
/// is freshly emitted, but multi-turn replay/storage references the image by
/// `id` only (see `ResponseInputOutputItem::ImageGenerationCall` where
/// `result` is `Option<String>`). This helper clears the payload for storage
/// while preserving `id`, `revised_prompt`, and `status`.
///
/// For any non-image item this is a no-op.
pub fn compact_image_generation_output(item: &mut ResponseOutputItem) {
    if let ResponseOutputItem::ImageGenerationCall { result, .. } = item {
        // Drop the heap buffer; `clear()` would retain (multi-MB) capacity.
        *result = String::new();
    }
}

/// Compact every `image_generation_call` item inside `outputs` in-place. JSON
/// counterpart to [`compact_image_generation_output`] used by the storage
/// layer, which works with `serde_json::Value` arrays rather than typed
/// `ResponseOutputItem`s. Non-image items are untouched.
pub fn compact_image_generation_outputs_json(outputs: &mut [serde_json::Value]) {
    for item in outputs {
        let Some(obj) = item.as_object_mut() else {
            continue;
        };
        if obj.get("type").and_then(|v| v.as_str()) != Some("image_generation_call") {
            continue;
        }
        // Remove rather than blank: the input/replay shape models `result` as
        // `Option<String>` with `skip_serializing_if`, so the id-only multi-turn
        // reference form requires field absence — an empty string would still
        // surface on the wire and diverge from the documented replay payload.
        obj.remove("result");
    }
}

fn parse_text_block_payload(item: &serde_json::Value) -> Option<serde_json::Value> {
    let Some(obj) = item.as_object() else {
        warn!("Expected MCP result item to be an object");
        return None;
    };

    if obj.get("type").and_then(|v| v.as_str()) != Some("text") {
        return None;
    }

    let Some(text) = obj.get("text").and_then(|v| v.as_str()) else {
        warn!("MCP text block is missing a text field");
        return None;
    };

    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(payload) => Some(payload),
        Err(error) => {
            warn!("Failed to parse embedded MCP text block payload: {error}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
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
}
