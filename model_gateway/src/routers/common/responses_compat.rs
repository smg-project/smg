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
//! - `tool_search_output`: validates and lifts discovered tools into top-level
//!   `tools`. The corresponding hosted search records (`tool_search_call` and
//!   `web_search_call`) are omitted on local backends.
//! - `function_call_output` without a call ID becomes a user message carrying
//!   the tool name, namespace, and output. Reasoning items missing an ID get one.
//!
//! External OpenAI-compatible cloud providers bypass this lowering so the
//! original wire items pass through untouched.

use axum::response::Response;
use openai_protocol::{
    common::ToolReference,
    responses::{
        generate_id, validate_response_tools_at, AgentMessageContentPart, NamespaceTool,
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseTool,
        ResponsesRequest, ResponsesToolChoice, StringOrContentParts,
    },
};

use crate::routers::error;

/// Default namespace Codex Responses Lite uses to group top-level function and
/// custom tools inside an `additional_tools` input item.
const DEFAULT_FUNCTION_NAMESPACE: &str = "functions";

/// Header prefix used when lowering a `function_call_output` without `call_id`
/// into a user message for local backends.
pub(crate) fn format_call_id_less_tool_output_header(
    name: Option<&str>,
    namespace: Option<&str>,
) -> String {
    let ns = namespace.filter(|s| !s.is_empty());
    let tool = name.filter(|s| !s.is_empty());
    match (ns, tool) {
        (Some(ns), Some(tool)) => format!("[tool_output {ns}.{tool}]"),
        (None, Some(tool)) => format!("[tool_output {tool}]"),
        (Some(ns), None) => format!("[tool_output {ns}]"),
        (None, None) => "[tool_output]".to_string(),
    }
}

/// Lower a `function_call_output` that has no `call_id` into user-message
/// content parts preserving tool identity (`namespace`/`name`) and all output
/// parts.
pub(crate) fn lower_call_id_less_tool_output_parts(
    name: Option<&str>,
    namespace: Option<&str>,
    output: StringOrContentParts,
) -> Vec<ResponseContentPart> {
    let header = format_call_id_less_tool_output_header(name, namespace);
    match output {
        StringOrContentParts::String(text) => {
            let combined = if text.is_empty() {
                header
            } else {
                format!("{header}\n{text}")
            };
            vec![ResponseContentPart::InputText { text: combined }]
        }
        StringOrContentParts::Array(mut parts) => {
            if let Some(ResponseContentPart::InputText { text }) = parts.first_mut() {
                if text.is_empty() {
                    *text = header;
                } else {
                    *text = format!("{header}\n{text}");
                }
            } else {
                parts.insert(0, ResponseContentPart::InputText { text: header });
            }
            parts
        }
    }
}

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

/// Lower inline and discovered tool declarations, agent messages, unpaired
/// function outputs, reasoning items, and hosted search records for local backends.
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
        | ResponseInputOutputItem::AgentMessage { .. }
        | ResponseInputOutputItem::ToolSearchCall { .. }
        | ResponseInputOutputItem::ToolSearchOutput { .. }
        | ResponseInputOutputItem::WebSearchCall { .. } => true,
        ResponseInputOutputItem::Reasoning { id, .. } => id.as_deref().is_none_or(str::is_empty),
        ResponseInputOutputItem::FunctionCallOutput { call_id, .. } => {
            call_id.as_deref().is_none_or(str::is_empty)
        }
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

    let mut lift_tools = |tools: Vec<ResponseTool>| {
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
                    if custom_choice_name
                        .is_some_and(|choice_name| callable_tool_name(&other) == Some(choice_name))
                    {
                        has_literal_custom_choice = true;
                    }
                    push_or_replace_tool(&mut lifted_tools, other);
                }
            }
        }
    };

    for (item_idx, item) in original_items.into_iter().enumerate() {
        match item {
            ResponseInputOutputItem::AdditionalTools { tools, .. } => {
                lift_tools(tools);
            }
            ResponseInputOutputItem::ToolSearchOutput { tools, .. } => {
                let prefix = format!("input[{item_idx}].tools");
                let parsed: Vec<ResponseTool> = tools
                    .into_iter()
                    .enumerate()
                    .map(|(tool_idx, value)| {
                        serde_json::from_value(value).map_err(|err| {
                            error::bad_request(
                                "invalid_tool",
                                format!("Invalid '{prefix}[{tool_idx}]': {err}"),
                            )
                        })
                    })
                    .collect::<Result<_, _>>()?;
                validate_response_tools_at(&parsed, &prefix).map_err(|err| {
                    error::bad_request(
                        err.code,
                        err.message
                            .unwrap_or_else(|| format!("Invalid '{prefix}'").into()),
                    )
                })?;
                lift_tools(parsed);
            }
            ResponseInputOutputItem::ToolSearchCall { .. }
            | ResponseInputOutputItem::WebSearchCall { .. } => {}
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
            ResponseInputOutputItem::FunctionCallOutput {
                id,
                call_id,
                name,
                namespace,
                output,
                ..
            } if call_id.as_deref().is_none_or(str::is_empty) => {
                let content = lower_call_id_less_tool_output_parts(
                    name.as_deref(),
                    namespace.as_deref(),
                    output,
                );
                lowered_items.push(ResponseInputOutputItem::Message {
                    id: id.unwrap_or_else(|| generate_id("msg")),
                    role: "user".to_string(),
                    content,
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
            "Input must contain at least one item beyond tool declarations and hosted search records",
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
#[path = "../../../tests/unit/routers/common/responses_compat/tests.rs"]
mod tests;
