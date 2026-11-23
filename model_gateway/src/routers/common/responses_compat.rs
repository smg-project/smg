//! Local-backend compatibility lowering for Codex / Responses API input items.
//!
//! OpenAI's Responses API and Codex clients (including Codex Responses Lite and
//! native multi-agent `spawn_agent`) emit input items that OpenAI's hosted API
//! accepts natively but self-hosted engines (SGLang `/v1/responses`, gRPC chat
//! and Harmony pipelines) do not implement directly:
//!
//! - `additional_tools`: declares tools inline at a point in `input` instead of
//!   at the top-level `tools` field. Codex Lite also wraps default top-level
//!   functions/custom tools in a `"functions"` namespace. For local backends,
//!   we lift those declarations into top-level `tools` (replacing duplicate
//!   declarations by name so the latest declaration wins), unwrapping the
//!   default `"functions"` namespace unless the request already declares an
//!   explicit top-level `"functions"` namespace, and keeping non-default
//!   namespaces (e.g. `clock`, `collaboration`) intact.
//! - `agent_message`: carries a task or peer message from a parent/peer agent
//!   (`author`, `recipient`, `content`). For local backends, plaintext
//!   `agent_message` items are lowered in-place to ordinary `role: "user"`
//!   `message` items preserving order, sender/recipient metadata, and task text
//!   without elevating the payload to a system/developer instruction. Non-
//!   plaintext parts (such as `encrypted_content`) are rejected explicitly with
//!   HTTP 400 rather than silently dropped.
//!
//! External OpenAI-compatible cloud providers bypass this lowering so the
//! original wire items pass through untouched.

use axum::response::Response;
use openai_protocol::{
    common::ToolReference,
    responses::{
        generate_id, AgentMessageContentPart, NamespaceTool, ResponseContentPart, ResponseInput,
        ResponseInputOutputItem, ResponseTool, ResponsesRequest, ResponsesToolChoice,
    },
};

use crate::routers::error;

/// Default namespace Codex Responses Lite uses to group top-level function and
/// custom tools inside an `additional_tools` input item.
const DEFAULT_FUNCTION_NAMESPACE: &str = "functions";

/// Render a plaintext [`ResponseInputOutputItem::AgentMessage`] into a single
/// user-message string that preserves sender (`author`), recipient
/// (`recipient`), and task body text.
///
/// Returns `Err(part_type_name)` if any content part is non-plaintext (such as
/// `"encrypted_content"`) so local backends refuse it explicitly instead of
/// dropping part of the payload.
pub(crate) fn format_agent_message_text(
    author: &str,
    recipient: &str,
    content: &[AgentMessageContentPart],
) -> Result<String, &'static str> {
    let mut text_parts = Vec::with_capacity(content.len());
    for part in content {
        match part {
            AgentMessageContentPart::InputText { text } => text_parts.push(text.as_str()),
            AgentMessageContentPart::EncryptedContent { .. } => {
                return Err(part.type_name());
            }
        }
    }
    let body = text_parts.join("\n");
    let mut out = String::new();
    if !author.is_empty() || !recipient.is_empty() {
        if !author.is_empty() {
            out.push_str("Sender: ");
            out.push_str(author);
            out.push('\n');
        }
        if !recipient.is_empty() {
            out.push_str("Recipient: ");
            out.push_str(recipient);
            out.push('\n');
        }
        out.push('\n');
    }
    out.push_str(&body);
    Ok(out)
}

/// Lower Codex-specific Responses input items (`additional_tools`,
/// `agent_message`, and `reasoning` items missing an `id`) into standard forms
/// understood by local HTTP/PD/gRPC inference backends.
///
/// Requests that need no lowering return immediately without allocating.
pub(crate) fn lower_responses_request_for_local_backend(
    request: &mut ResponsesRequest,
) -> Result<(), Response> {
    let ResponseInput::Items(items) = &mut request.input else {
        return Ok(());
    };

    let needs_lowering = items.iter().any(|item| match item {
        ResponseInputOutputItem::AdditionalTools { .. }
        | ResponseInputOutputItem::AgentMessage { .. } => true,
        ResponseInputOutputItem::Reasoning { id, .. } => id.as_deref().is_none_or(str::is_empty),
        _ => false,
    });
    if !needs_lowering {
        return Ok(());
    }

    let original_items = std::mem::take(items);

    // If top-level `tools` already declares an explicit `"functions"` namespace,
    // coalesce inline `"functions"` members into it instead of flattening so
    // `tool_choice` and history calls targeting `"functions"` remain consistent.
    let has_explicit_functions_namespace = request.tools.as_deref().is_some_and(|tools| {
        tools.iter().any(|tool| {
            matches!(tool, ResponseTool::Namespace(ns) if ns.name == DEFAULT_FUNCTION_NAMESPACE)
        })
    });

    let custom_choice_name = match &request.tool_choice {
        Some(ResponsesToolChoice::Custom { name, .. }) => Some(name.as_str()),
        _ => None,
    };
    let mut has_literal_custom_choice = custom_choice_name.is_some_and(|choice_name| {
        request.tools.as_deref().is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| callable_tool_name(tool) == Some(choice_name))
        })
    });

    let mut lowered_items = Vec::with_capacity(original_items.len());
    let mut lifted_tools: Vec<ResponseTool> = Vec::new();
    let mut unwrapped_default_namespace = false;
    let mut unwrapped_custom_names: Vec<String> = Vec::new();

    for item in original_items {
        match item {
            ResponseInputOutputItem::AdditionalTools { tools, .. } => {
                for tool in tools {
                    match tool {
                        ResponseTool::Namespace(namespace)
                            if !has_explicit_functions_namespace
                                && namespace.name == DEFAULT_FUNCTION_NAMESPACE =>
                        {
                            unwrapped_default_namespace = true;
                            for member in namespace.tools {
                                let lifted = match member {
                                    NamespaceTool::Function(ft) => ResponseTool::Function(ft),
                                    NamespaceTool::Custom(ct) => {
                                        unwrapped_custom_names.push(ct.name.clone());
                                        ResponseTool::Custom(ct)
                                    }
                                };
                                push_or_replace_tool(&mut lifted_tools, lifted);
                            }
                        }
                        other => {
                            if custom_choice_name.is_some_and(|choice_name| {
                                callable_tool_name(&other) == Some(choice_name)
                            }) {
                                has_literal_custom_choice = true;
                            }
                            push_or_replace_tool(&mut lifted_tools, other);
                        }
                    }
                }
            }
            ResponseInputOutputItem::AgentMessage {
                id,
                author,
                recipient,
                content,
                ..
            } => {
                let text = format_agent_message_text(&author, &recipient, &content).map_err(
                    |part_kind| {
                        error::bad_request(
                            "unsupported_content_part",
                            format!(
                                "Unsupported content part type {part_kind:?} in an agent_message for local backends: only plaintext \"input_text\" parts are supported"
                            ),
                        )
                    },
                )?;
                lowered_items.push(ResponseInputOutputItem::Message {
                    id: id.unwrap_or_else(|| generate_id("msg")),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text }],
                    status: Some("completed".to_string()),
                    phase: None,
                });
            }
            mut other => {
                if let ResponseInputOutputItem::Reasoning { id, .. } = &mut other {
                    if id.as_deref().is_none_or(str::is_empty) {
                        *id = Some(generate_id("rs"));
                    }
                }
                lowered_items.push(other);
            }
        }
    }

    if lowered_items.is_empty() {
        return Err(error::bad_request(
            "input_items_empty",
            "Input must contain at least one message or item beyond additional_tools",
        ));
    }

    if !lifted_tools.is_empty() {
        let target = request.tools.get_or_insert_with(Vec::new);
        for tool in lifted_tools {
            push_or_replace_tool(target, tool);
        }
        if has_explicit_functions_namespace {
            // Codex permits bare names for inline functions. A real bare
            // declaration takes precedence when both spellings exist.
            let needs_namespace = |name: &str| {
                !target.iter().any(|tool| callable_tool_name(tool) == Some(name))
                    && target.iter().any(|tool| {
                        matches!(tool, ResponseTool::Namespace(ns)
                            if ns.name == DEFAULT_FUNCTION_NAMESPACE
                                && ns.tools.iter().any(|member| namespace_member_name(member) == name))
                    })
            };
            match &mut request.tool_choice {
                Some(ResponsesToolChoice::Function(choice))
                    if choice.namespace.is_none() && needs_namespace(&choice.name) =>
                {
                    choice.namespace = Some(DEFAULT_FUNCTION_NAMESPACE.to_owned());
                }
                Some(ResponsesToolChoice::AllowedTools { tools, .. }) => {
                    for tool in tools {
                        if let ToolReference::Function { name } = tool {
                            if needs_namespace(name) {
                                *name = format!("{DEFAULT_FUNCTION_NAMESPACE}.{name}");
                            }
                        }
                    }
                }
                Some(ResponsesToolChoice::Custom { name, .. }) if needs_namespace(name) => {
                    *name = format!("{DEFAULT_FUNCTION_NAMESPACE}.{name}");
                }
                _ => {}
            }
        }
    }

    if unwrapped_default_namespace {
        for item in &mut lowered_items {
            match item {
                ResponseInputOutputItem::FunctionToolCall { namespace, .. }
                | ResponseInputOutputItem::CustomToolCall { namespace, .. }
                    if namespace.as_deref() == Some(DEFAULT_FUNCTION_NAMESPACE) =>
                {
                    *namespace = None;
                }
                _ => {}
            }
        }
        match &mut request.tool_choice {
            Some(ResponsesToolChoice::Function(choice))
                if choice.namespace.as_deref() == Some(DEFAULT_FUNCTION_NAMESPACE) =>
            {
                choice.namespace = None;
            }
            Some(ResponsesToolChoice::Custom { name, .. }) => {
                if let Some(bare) = name
                    .strip_prefix(DEFAULT_FUNCTION_NAMESPACE)
                    .and_then(|rest| rest.strip_prefix('.'))
                {
                    if !has_literal_custom_choice
                        && unwrapped_custom_names
                            .iter()
                            .any(|unwrapped| unwrapped == bare)
                    {
                        *name = bare.to_owned();
                    }
                }
            }
            _ => {}
        }
    }

    request.input = ResponseInput::Items(lowered_items);
    Ok(())
}

fn callable_tool_name(tool: &ResponseTool) -> Option<&str> {
    match tool {
        ResponseTool::Function(ft) => Some(ft.function.name.as_str()),
        ResponseTool::Custom(ct) => Some(ct.name.as_str()),
        _ => None,
    }
}

fn namespace_member_name(member: &NamespaceTool) -> &str {
    match member {
        NamespaceTool::Function(ft) => ft.function.name.as_str(),
        NamespaceTool::Custom(ct) => ct.name.as_str(),
    }
}

/// Append `tool` to `dest`, coalescing same-named `ResponseTool::Namespace`
/// declarations and replacing duplicate callable tool names in-place so the
/// latest inline declaration wins.
fn push_or_replace_tool(dest: &mut Vec<ResponseTool>, tool: ResponseTool) {
    match tool {
        ResponseTool::Namespace(incoming) => {
            if let Some(existing) = dest.iter_mut().find_map(|entry| match entry {
                ResponseTool::Namespace(ns) if ns.name == incoming.name => Some(ns),
                _ => None,
            }) {
                if existing.description.is_empty() && !incoming.description.is_empty() {
                    existing.description = incoming.description;
                }
                if incoming.defer_loading.is_some() {
                    existing.defer_loading = incoming.defer_loading;
                }
                for member in incoming.tools {
                    let name = namespace_member_name(&member);
                    if let Some(slot) = existing
                        .tools
                        .iter_mut()
                        .find(|existing_member| namespace_member_name(existing_member) == name)
                    {
                        *slot = member;
                    } else {
                        existing.tools.push(member);
                    }
                }
                return;
            }
            dest.push(ResponseTool::Namespace(incoming));
        }
        incoming => {
            if let Some(name) = callable_tool_name(&incoming) {
                if let Some(slot) = dest
                    .iter_mut()
                    .find(|existing| callable_tool_name(existing) == Some(name))
                {
                    *slot = incoming;
                    return;
                }
            }
            dest.push(incoming);
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use openai_protocol::{
        responses::{
            ResponseInput, ResponseInputOutputItem, ResponsesRequest, ResponsesToolChoice,
            ToolChoiceOptions,
        },
        validated::Normalizable,
    };
    use serde_json::{json, Value};
    use validator::Validate;

    use super::*;
    use crate::routers::grpc::common::responses::utils::extract_tools_from_response_tools;

    fn parse_request(body: Value) -> ResponsesRequest {
        serde_json::from_value(body).expect("request should deserialize")
    }

    #[test]
    fn lowers_codex_subagent_request_with_additional_tools_and_agent_message() {
        let mut req = parse_request(json!({
            "model": "test-model",
            "stream": true,
            "tools": [],
            "input": [
                {
                    "type": "additional_tools",
                    "id": "at_1",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "description": "",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "exec_command",
                                    "description": "Run shell command",
                                    "parameters": {"type": "object", "properties": {}},
                                    "strict": false
                                },
                                {
                                    "type": "custom",
                                    "name": "apply_patch",
                                    "description": "Patch files"
                                }
                            ]
                        },
                        {
                            "type": "namespace",
                            "name": "clock",
                            "description": "Clock tools",
                            "tools": [{
                                "type": "function",
                                "name": "now",
                                "description": "Current time",
                                "parameters": {"type": "object", "properties": {}}
                            }]
                        },
                        {
                            "type": "namespace",
                            "name": "collaboration",
                            "description": "Multi-agent tools",
                            "tools": [{
                                "type": "function",
                                "name": "send_message",
                                "description": "Send message",
                                "parameters": {"type": "object", "properties": {}}
                            }]
                        }
                    ]
                },
                {
                    "type": "message",
                    "id": "msg_sys",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Base instructions"}]
                },
                {
                    "type": "agent_message",
                    "id": "am_1",
                    "author": "/root",
                    "recipient": "/root/worker_1",
                    "content": [{"type": "input_text", "text": "Reply with OK-123."}]
                }
            ]
        }));

        // ValidatedJson runs normalize() before routing; empty top-level tools: []
        // must default to auto (not none) when inline additional_tools exist.
        req.normalize();
        assert!(matches!(
            req.tool_choice,
            Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto))
        ));

        // Custom tool_choice targeting an unwrapped default-namespace tool must
        // strip the `functions.` qualifier to match the bare offered tool name.
        let mut custom_choice_req = req.clone();
        custom_choice_req.tool_choice = Some(
            serde_json::from_value(json!({"type": "custom", "name": "functions.apply_patch"}))
                .unwrap(),
        );
        lower_responses_request_for_local_backend(&mut custom_choice_req)
            .expect("lowering succeeds with custom tool_choice");
        assert_eq!(
            serde_json::to_value(&custom_choice_req.tool_choice).unwrap(),
            json!({"type": "custom", "name": "apply_patch"})
        );

        // Literal top-level custom names with a `functions.` prefix must stay
        // intact when an unrelated inline `functions` namespace is unwrapped.
        let mut literal_custom_req = parse_request(json!({
            "model": "test-model",
            "tools": [{"type": "custom", "name": "functions.apply_patch"}],
            "tool_choice": {"type": "custom", "name": "functions.apply_patch"},
            "input": [
                {
                    "type": "additional_tools",
                    "tools": [{
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]
                    }]
                },
                {"role": "user", "content": "patch it"}
            ]
        }));
        lower_responses_request_for_local_backend(&mut literal_custom_req)
            .expect("lowering preserves literal dotted custom tool");
        assert_eq!(
            serde_json::to_value(&literal_custom_req.tool_choice).unwrap(),
            json!({"type": "custom", "name": "functions.apply_patch"})
        );

        // An unwrapped `functions.apply_patch` member must not block rewriting
        // `tool_choice: functions.apply_patch` to the unwrapped `apply_patch`.
        let mut collision_req = parse_request(json!({
            "model": "test-model",
            "tool_choice": {"type": "custom", "name": "functions.apply_patch"},
            "input": [
                {
                    "type": "additional_tools",
                    "tools": [{
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [
                            {"type": "custom", "name": "apply_patch"},
                            {"type": "custom", "name": "functions.apply_patch"}
                        ]
                    }]
                },
                {"role": "user", "content": "patch it"}
            ]
        }));
        lower_responses_request_for_local_backend(&mut collision_req)
            .expect("lowering rewrites choice to unwrapped apply_patch");
        assert_eq!(
            serde_json::to_value(&collision_req.tool_choice).unwrap(),
            json!({"type": "custom", "name": "apply_patch"})
        );

        lower_responses_request_for_local_backend(&mut req).expect("lowering succeeds");

        let wire = serde_json::to_value(&req).unwrap();
        let tools = wire["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 4);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "exec_command");
        assert_eq!(tools[1]["type"], "custom");
        assert_eq!(tools[1]["name"], "apply_patch");
        assert_eq!(tools[2]["type"], "namespace");
        assert_eq!(tools[2]["name"], "clock");
        assert_eq!(tools[3]["type"], "namespace");
        assert_eq!(tools[3]["name"], "collaboration");

        let input = wire["input"].as_array().expect("input array");
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "developer");

        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["id"], "am_1");
        assert_eq!(input[1]["role"], "user");
        let text = input[1]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Sender: /root"), "{text}");
        assert!(text.contains("Recipient: /root/worker_1"), "{text}");
        assert!(text.ends_with("Reply with OK-123."), "{text}");
    }

    #[test]
    fn inline_tool_declarations_alone_leave_no_local_model_input() {
        let mut req = parse_request(json!({
            "model":"test-model",
            "input":[{"type":"additional_tools", "role":"developer",
                "tools":[{"type":"function","name":"exec_command"}]}]
        }));
        let response = lower_responses_request_for_local_backend(&mut req).unwrap_err();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn rejects_encrypted_agent_message_for_local_backends() {
        let mut req = parse_request(json!({
            "model": "test-model",
            "input": [{
                "type": "agent_message",
                "id": "am_enc",
                "author": "/root",
                "recipient": "/root/worker_1",
                "content": [
                    {"type": "input_text", "text": "Header"},
                    {"type": "encrypted_content", "encrypted_content": "gAAAAB-secret"}
                ]
            }]
        }));

        let response = lower_responses_request_for_local_backend(&mut req)
            .expect_err("encrypted_content must be refused");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn coalesces_duplicate_tools_and_keeps_explicit_functions_namespace_consistent() {
        // 1. Repeated inline declarations deduplicate by name (latest wins).
        let mut dedup = parse_request(json!({
            "model": "test-model",
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [{
                            "type": "function",
                            "name": "exec_command",
                            "description": "v1",
                            "parameters": {"type": "object"}
                        }]
                    }, {
                        "type": "namespace",
                        "name": "collaboration",
                        "description": "Collab",
                        "tools": [{
                            "type": "function",
                            "name": "send_message",
                            "description": "v1",
                            "parameters": {"type": "object"}
                        }]
                    }]
                },
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [{
                            "type": "function",
                            "name": "exec_command",
                            "description": "v2",
                            "parameters": {"type": "object"}
                        }]
                    }, {
                        "type": "namespace",
                        "name": "collaboration",
                        "description": "Collab",
                        "tools": [{
                            "type": "function",
                            "name": "send_message",
                            "description": "v2",
                            "parameters": {"type": "object"}
                        }]
                    }]
                },
                {"role": "user", "content": "run"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "namespace": "functions",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok"
                }
            ]
        }));
        lower_responses_request_for_local_backend(&mut dedup).unwrap();
        let wire = serde_json::to_value(&dedup).unwrap();
        let tools = wire["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "exec_command");
        assert_eq!(tools[0]["description"], "v2");
        assert_eq!(tools[1]["tools"].as_array().unwrap().len(), 1);
        assert_eq!(tools[1]["tools"][0]["description"], "v2");
        let ResponseInput::Items(items) = &dedup.input else {
            panic!("expected items");
        };
        assert!(matches!(
            &items[1],
            ResponseInputOutputItem::FunctionToolCall { namespace: None, name, .. } if name == "exec_command"
        ));

        // 2. When top-level tools already declare a `functions` namespace, inline
        // `functions` members coalesce into it so `tool_choice` targeting
        // `functions.<new>` remains valid.
        let mut mixed = parse_request(json!({
            "model": "test-model",
            "tools": [{
                "type": "namespace",
                "name": "functions",
                "description": "Explicit functions namespace",
                "tools": [{
                    "type": "function",
                    "name": "write_stdin",
                    "parameters": {"type": "object"}
                }]
            }],
            "tool_choice": {"type": "function", "name": "exec_command", "namespace": "functions"},
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [{
                            "type": "function",
                            "name": "exec_command",
                            "parameters": {"type": "object"}
                        }, {"type": "custom", "name": "apply_patch"}]
                    }]
                },
                {"role": "user", "content": "Run git status"}
            ]
        }));
        for (choice, expected) in [
            (
                json!({"type": "function", "name": "exec_command"}),
                "functions.exec_command",
            ),
            (
                json!({"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "exec_command"}]}),
                "functions.exec_command",
            ),
            (
                json!({"type": "custom", "name": "apply_patch"}),
                "functions.apply_patch",
            ),
        ] {
            let mut request = mixed.clone();
            request.tool_choice = Some(serde_json::from_value(choice).unwrap());
            request.validate().unwrap();
            lower_responses_request_for_local_backend(&mut request).unwrap();
            let tools = extract_tools_from_response_tools(request.tools.as_deref());
            let offered = request
                .tool_choice
                .unwrap()
                .to_chat_tool_choice()
                .narrow_tools(&tools)
                .unwrap();
            assert_eq!(offered.len(), 1);
            assert_eq!(offered[0].function.name, expected);
        }
        lower_responses_request_for_local_backend(&mut mixed).unwrap();
        let wire = serde_json::to_value(&mixed).unwrap();
        assert_eq!(wire["tools"].as_array().unwrap().len(), 1);
        assert_eq!(wire["tools"][0]["type"], "namespace");
        assert_eq!(wire["tools"][0]["name"], "functions");
        let members = wire["tools"][0]["tools"].as_array().unwrap();
        assert_eq!(members.len(), 3);
        assert_eq!(members[0]["name"], "write_stdin");
        assert_eq!(members[1]["name"], "exec_command");
        assert_eq!(wire["tool_choice"]["namespace"], "functions");
        assert_eq!(wire["tool_choice"]["name"], "exec_command");
    }

    #[test]
    fn lowers_missing_reasoning_id_for_local_backends() {
        let mut req = parse_request(json!({
            "model": "test-model",
            "input": [
                {"role": "user", "content": "check status"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                    "content": null
                },
                {
                    "type": "reasoning",
                    "id": null,
                    "summary": []
                },
                {
                    "type": "reasoning",
                    "id": "rs_existing",
                    "summary": [],
                    "encrypted_content": "opaque"
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok"
                }
            ]
        }));

        lower_responses_request_for_local_backend(&mut req)
            .expect("missing/null reasoning ids should be lowered");

        let wire = serde_json::to_value(&req).unwrap();
        let items = wire["input"].as_array().unwrap();
        let generated_1 = items[1]["id"].as_str().expect("missing id synthesized");
        let generated_2 = items[2]["id"].as_str().expect("null id synthesized");
        assert!(generated_1.starts_with("rs_"), "{generated_1}");
        assert!(generated_2.starts_with("rs_"), "{generated_2}");
        assert_ne!(generated_1, generated_2);
        assert!(items[1].get("content").is_none());
        assert_eq!(items[3]["id"], "rs_existing");
        assert_eq!(items[3]["encrypted_content"], "opaque");
    }
}
