//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use openai_protocol::{
    common::Tool,
    responses::{CustomTool, NamespaceTool, ResponseTool, ResponsesRequest, ResponsesResponse},
};
use serde_json::to_value;
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpOrchestrator, McpServerBinding};
use tracing::{debug, error, warn};

use crate::{
    routers::{
        common::{
            mcp_utils::ensure_request_mcp_client, openai_bridge,
            persistence_utils::persist_conversation_items,
        },
        error,
    },
    worker::WorkerRegistry,
};

/// Ensure MCP connection succeeds if MCP tools or builtin tools are declared.
///
/// Checks if the request declares MCP tools or builtin tool types
/// (`web_search_preview`, `code_interpreter`, `image_generation`) and,
/// if so, validates that the MCP clients can be created and connected.
///
/// Returns Ok((has_mcp_tools, mcp_servers)) on success.
pub(crate) async fn ensure_mcp_connection(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    format_registry: &openai_bridge::FormatRegistry,
    tools: Option<&[ResponseTool]>,
) -> Result<(bool, Vec<McpServerBinding>), Response> {
    // Check for explicit MCP tools (must error if connection fails)
    let has_explicit_mcp_tools = tools
        .map(|t| t.iter().any(|tool| matches!(tool, ResponseTool::Mcp(_))))
        .unwrap_or(false);

    // Check for builtin tools that MAY have MCP routing configured.
    //
    // `ImageGeneration` is included here because gpt-oss via the
    // harmony pipeline, and Qwen/Llama via the regular pipeline, both
    // dispatch hosted `image_generation` calls through the same MCP
    // routing path — the only difference is how the tool is advertised in
    // the prompt. Without this arm, the short-circuit below would return
    // `(false, Vec::new())`, the MCP loop would never be entered, and the
    // registered `image_generation` MCP server would receive zero
    // dispatches.
    let has_builtin_tools = tools
        .map(|t| {
            t.iter()
                .any(|tool| openai_bridge::builtin_type_for_response_tool(tool).is_some())
        })
        .unwrap_or(false);

    // Only process if we have MCP or builtin tools
    if !has_explicit_mcp_tools && !has_builtin_tools {
        return Ok((false, Vec::new()));
    }

    if let Some(tools) = tools {
        // TODO: Thread real request headers through the gRPC responses path if/when
        // gRPC MCP flows need the same forwarded-header preservation contract.
        match ensure_request_mcp_client(mcp_orchestrator, format_registry, tools).await {
            Some(mcp_servers) => {
                return Ok((true, mcp_servers));
            }
            None => {
                // No MCP servers available
                if has_explicit_mcp_tools {
                    // Explicit MCP tools MUST have working connections
                    error!(
                        function = "ensure_mcp_connection",
                        "Failed to connect to MCP servers"
                    );
                    return Err(error::failed_dependency(
                        "connect_mcp_server_failed",
                        "Failed to connect to MCP servers. Check server_url and authorization.",
                    ));
                }
                // Builtin tools without MCP routing - pass through to model
                debug!(
                    function = "ensure_mcp_connection",
                    "No MCP routing configured for builtin tools, passing through to model"
                );
                return Ok((false, Vec::new()));
            }
        }
    }

    Ok((false, Vec::new()))
}

/// Validate that workers are available for the requested model.
///
/// Runs on the client-supplied name, before the pipeline canonicalizes it, so
/// it has to accept aliases as well as canonical model IDs. `contains_model`
/// covers both; listing `get_models()` and testing membership would reject
/// every alias here.
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    if !worker_registry.contains_model(model) {
        return Some(error::model_not_found(model));
    }

    None
}

/// Extract function tools from ResponseTools
///
/// This utility consolidates the logic for extracting tools with schemas from ResponseTools.
/// It's used by both Harmony and Regular routers for different purposes:
///
/// - **Harmony router**: Extracts function tools because MCP tools are exposed to the model as
///   function tools (via `convert_mcp_tools_to_response_tools()`), and those are used to
///   generate structural constraints in the Harmony preparation stage.
///
/// - **Regular router**: Extracts function tools during the initial conversion from
///   ResponsesRequest to ChatCompletionRequest. MCP tools are merged later by the tool loop.
///
/// Custom tools (`{"type": "custom"}`) are downgraded to function tools with a
/// single `input: string` parameter; `chat_to_responses` maps the calls back.
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .flat_map(|tool| match tool {
            ResponseTool::Function(ft) => vec![Tool {
                tool_type: "function".to_string(),
                function: ft.function.clone(),
            }],
            ResponseTool::Custom(ct) => vec![Tool {
                tool_type: "function".to_string(),
                function: custom_tool_as_function(ct),
            }],
            ResponseTool::Namespace(namespace) => namespace
                .tools
                .iter()
                .map(|member| {
                    let mut function = match member {
                        NamespaceTool::Function(ft) => ft.function.clone(),
                        NamespaceTool::Custom(ct) => custom_tool_as_function(ct),
                    };
                    function.name = format!("{}.{}", namespace.name, function.name);
                    Tool {
                        tool_type: "function".to_string(),
                        function,
                    }
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

/// Downgrade a Responses custom tool to a chat function tool whose single
/// `input` string parameter carries the free-form payload.
pub(crate) fn custom_tool_as_function(ct: &CustomTool) -> openai_protocol::common::Function {
    openai_protocol::common::Function {
        name: ct.name.clone(),
        description: ct.description.clone(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The raw payload to pass to this custom tool."
                }
            },
            "required": ["input"],
            "additionalProperties": false
        }),
        strict: Some(false),
    }
}

/// Recover a custom tool's raw `input` from the `{"input": "..."}` arguments
/// of its downgraded function call.
pub(crate) fn custom_tool_input(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("input")?.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Collect the names of all declared custom tools (top-level and namespaced),
/// used to map function-shaped calls back to `custom_tool_call` items.
pub(crate) fn custom_tool_names(
    response_tools: Option<&[ResponseTool]>,
) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    let Some(tools) = response_tools else {
        return names;
    };
    for tool in tools {
        match tool {
            ResponseTool::Custom(ct) => {
                names.insert(ct.name.clone());
            }
            ResponseTool::Namespace(namespace) => {
                for member in &namespace.tools {
                    if let NamespaceTool::Custom(ct) = member {
                        names.insert(format!("{}.{}", namespace.name, ct.name));
                    }
                }
            }
            _ => {}
        }
    }
    names
}

/// Version tag for the opaque `reasoning.encrypted_content` payload.
const REASONING_BLOB_PREFIX: &str = "smg-reasoning-v1.";

/// Encode reasoning text as the opaque `encrypted_content` blob a `store=false`
/// client hands back on its next turn. The gateway is stateless, so the blob
/// carries the text itself (version-tagged base64): opaque, not confidential.
pub(crate) fn encode_reasoning_content(text: &str) -> String {
    format!("{REASONING_BLOB_PREFIX}{}", BASE64_STANDARD.encode(text))
}

/// Recover reasoning text from a blob produced by [`encode_reasoning_content`].
/// Blobs from other producers yield `None`.
pub(crate) fn decode_reasoning_content(blob: &str) -> Option<String> {
    let payload = blob.strip_prefix(REASONING_BLOB_PREFIX)?;
    String::from_utf8(BASE64_STANDARD.decode(payload).ok()?).ok()
}

/// Recover structured identity only for a declared namespace member.
/// Literal top-level names (including dots) take precedence over namespace matches.
pub(crate) fn resolve_function_identity(
    tools: Option<&[ResponseTool]>,
    name: &str,
) -> (String, Option<String>) {
    let tools = tools.unwrap_or_default();
    if !tools
        .iter()
        .any(|tool| matches!(tool, ResponseTool::Function(ft) if ft.function.name == name))
    {
        for tool in tools {
            if let ResponseTool::Namespace(namespace) = tool {
                for member in &namespace.tools {
                    if let NamespaceTool::Function(ft) = member {
                        if name == format!("{}.{}", namespace.name, ft.function.name) {
                            return (ft.function.name.clone(), Some(namespace.name.clone()));
                        }
                    }
                }
            }
        }
    }
    (name.to_string(), None)
}

/// Synthetic same-name members used by namespace routing regression tests.
#[cfg(test)]
pub(crate) fn namespace_test_request() -> ResponsesRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "input": "Check the weather",
        "tools": [
            {"type": "namespace", "name": "weather", "description": "Weather tools", "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
            ]},
            {"type": "namespace", "name": "travel", "description": "Travel tools", "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
            ]}
        ]
    })).unwrap()
}

/// Persist response to storage if store=true
///
/// Common helper function to avoid duplication across sync and streaming paths
/// in both harmony and regular responses implementations.
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
            request_context,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, UNKNOWN_MODEL_ID};

    fn registry_with_aliased_worker() -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .model(ModelCard::new("canonical-model").with_alias("model-alias"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        registry.register_or_replace(Arc::new(worker));
        registry
    }

    #[test]
    fn worker_availability_accepts_alias_and_preserves_unknown_rejection() {
        let registry = registry_with_aliased_worker();

        assert!(validate_worker_availability(&registry, "canonical-model").is_none());
        assert!(validate_worker_availability(&registry, "model-alias").is_none());

        let response = validate_worker_availability(&registry, UNKNOWN_MODEL_ID)
            .expect("unknown model should remain rejected for Responses");
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    }

    #[test]
    fn worker_availability_rejects_alias_once_its_worker_is_gone() {
        let registry = registry_with_aliased_worker();
        let worker_id = registry.get_id_by_url("http://worker:8080").unwrap();
        assert!(registry.remove(&worker_id).is_some());

        let response = validate_worker_availability(&registry, "model-alias")
            .expect("alias must stop resolving with no workers behind it");
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    }
    #[test]
    fn namespace_function_identity_roundtrips_and_preserves_literal_names() {
        let mut request: ResponsesRequest = namespace_test_request();
        let tools = extract_tools_from_response_tools(request.tools.as_deref());
        assert_eq!(
            tools
                .iter()
                .map(|t| t.function.name.as_str())
                .collect::<Vec<_>>(),
            vec!["weather.lookup", "travel.lookup"]
        );
        for namespace in ["weather", "travel"] {
            assert_eq!(
                resolve_function_identity(request.tools.as_deref(), &format!("{namespace}.lookup")),
                ("lookup".into(), Some(namespace.into()))
            );
        }
        for name in ["lookup", "unknown.lookup"] {
            assert_eq!(
                resolve_function_identity(request.tools.as_deref(), name),
                (name.into(), None)
            );
        }
        request.tools.as_mut().unwrap().push(
            serde_json::from_value(
                serde_json::json!({"type":"function","name":"weather.lookup","parameters":{}}),
            )
            .unwrap(),
        );
        assert_eq!(
            resolve_function_identity(request.tools.as_deref(), "weather.lookup"),
            ("weather.lookup".into(), None)
        );
    }
}
