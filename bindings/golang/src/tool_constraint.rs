//! The tools of a request as the bindings read them: the decoding constraint
//! of its tool choice, and the tool list its response parser resolves calls
//! against.

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{Tool, ToolChoice, ToolChoiceValue},
};
use symphony::formats::{forced::arguments_schema, json::required_calls_schema};

/// Every tool the model sees: the request-level `tools` followed by the tools
/// declared on messages, as the request's provider profile defines them (the
/// gateway's `effective_tools`). The constraint and the response parser both
/// work from this set, so a forced call can land on a message-declared tool
/// and a call to one comes back as a call, not as text.
fn effective_tools(request: &ChatCompletionRequest) -> Vec<Tool> {
    request.effective_tools().cloned().collect()
}

/// The tools the response parser resolves calls against, as the JSON array
/// the converter takes; `None` when the request declares none.
pub(crate) fn effective_tools_json(request: &ChatCompletionRequest) -> Option<String> {
    let tools = effective_tools(request);
    if tools.is_empty() {
        return None;
    }
    serde_json::to_string(&tools).ok()
}

/// Whether the engine must keep its special tokens for the response parser:
/// tools at either level (request-level or message-declared), a tool choice
/// that allows calls, and no JSON-schema constraint (whose output is plain
/// JSON from the first token, with no markers to keep). The request paths
/// derive `skip_special_tokens = false` from it.
pub(crate) fn tools_keep_special_tokens(
    request: &ChatCompletionRequest,
    constraint: Option<&(String, String)>,
) -> bool {
    constraint.is_none_or(|(kind, _)| kind != "json_schema")
        && !effective_tools(request).is_empty()
        && !matches!(
            request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        )
}

/// The constraint the engine is held to for the request's tool choice, as the
/// `(kind, value)` pair the engine request takes: the JSON schema the
/// Symphony tables read the forced call under, built from the request's
/// effective tools narrowed by the choice (the named function, an
/// allowed-tools list), the gateway's callable set. A named function is
/// constrained with that function's parameters (the `forced` table reads
/// the arguments object alone), `required` with the list of calls to any of
/// the tools (the `json` table reads it). `None` when the request declares no
/// tool or its choice forces no call; an error when the choice needs a tool
/// and the request declares none, or narrows the tools to none (a function
/// the request does not declare, an allowed-tools list naming none of them),
/// both of which the gateway's validation rejects with the same words, or
/// when two tools define one `$defs` name differently.
pub(crate) fn tool_constraint_for(
    request: &ChatCompletionRequest,
) -> Result<Option<(String, String)>, String> {
    let Some(tool_choice) = request.tool_choice.as_ref() else {
        return Ok(None);
    };
    let tools = effective_tools(request);
    if tools.is_empty() {
        let needs_none = matches!(
            tool_choice,
            ToolChoice::Value(ToolChoiceValue::None | ToolChoiceValue::Auto)
        );
        return if needs_none {
            Ok(None)
        } else {
            Err(TOOL_CHOICE_REQUIRES_TOOLS.to_string())
        };
    }
    let narrowed = tool_choice.narrow_tools(&tools);
    let allowed: &[Tool] = match &narrowed {
        Some(narrowed) if narrowed.is_empty() => return Err(narrowed_to_none(tool_choice)),
        Some(narrowed) => narrowed,
        None => &tools,
    };
    if !tool_choice.forces_call() {
        return Ok(None);
    }
    let schema = match tool_choice {
        // The helper finds the function among every tool of the request; the
        // narrowing above refused a name no tool declares.
        ToolChoice::Function { function, .. } => {
            arguments_schema(&tools, &function.name).ok_or_else(|| narrowed_to_none(tool_choice))?
        }
        ToolChoice::Value(_) | ToolChoice::AllowedTools { .. } => required_calls_schema(allowed)?,
    };
    let schema = serde_json::to_string(&schema)
        .map_err(|e| format!("Failed to serialize tool parameters: {e}"))?;
    Ok(Some(("json_schema".to_string(), schema)))
}

/// The gateway's refusal of a choice that needs a tool on a request that
/// declares none (`tool_choice_requires_tools` in the protocol crate's
/// validation), word for word.
const TOOL_CHOICE_REQUIRES_TOOLS: &str =
    "Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.";

/// The error of a choice that leaves no tool to call, worded as the
/// gateway's validation words it.
fn narrowed_to_none(tool_choice: &ToolChoice) -> String {
    match tool_choice {
        ToolChoice::Function { function, .. } => format!(
            "Invalid value for 'tool_choice': function '{}' not found in 'tools'.",
            function.name
        ),
        ToolChoice::AllowedTools { .. } => {
            "Invalid value for 'tool_choice.allowed_tools.tools': the listed tools are not found in 'tools'."
                .to_string()
        }
        ToolChoice::Value(_) => "Invalid value for 'tool_choice': no tool to call.".to_string(),
    }
}

#[cfg(test)]
#[path = "../tests/unit/tool_constraint/tests.rs"]
mod tests;
