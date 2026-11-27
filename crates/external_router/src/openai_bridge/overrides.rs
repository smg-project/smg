//! Pure-function helpers that merge caller-declared hosted-tool configuration
//! into model-emitted tool-call arguments before MCP dispatch.
//!
//! # Motivation
//!
//! OpenAI's Responses API lets callers declare hosted-tool configuration on the
//! request, e.g.:
//!
//! ```json
//! { "tools": [{ "type": "image_generation", "size": "512x512", "quality": "high" }] }
//! ```
//!
//! When the model emits the hosted-tool call, the arguments field often
//! contains only the dynamic subset (e.g. `{"prompt": "a cat"}`). The remaining
//! caller-declared knobs (`size`, `quality`, `model`, ...) live on the
//! `tools` declaration and must be merged into the dispatch payload so the
//! MCP server receives the effective configuration.
//!
//! These helpers are intentionally:
//! - **pure**: no session state, no I/O; testable in isolation.
//! - **symmetric** with [`super::transformer`], which handles the
//!   inverse direction (MCP result → `ResponseOutputItem`).
//!
//! # Contract
//!
//! [`extract_hosted_tool_overrides`] walks `tools` for the first declaration of
//! the requested kind, serializes it to `Value::Object`, drops null entries
//! (so absent fields don't overwrite model-supplied values), strips the `type`
//! discriminator (the kind is already known to the caller), and returns the
//! remaining object — or `None` if no configurable fields remain.
//!
//! [`apply_hosted_tool_overrides`] mutates the dispatch args in-place, with
//! overrides winning over model args for overlapping keys. Non-object or empty
//! inputs are no-ops.

use openai_protocol::responses::ResponseTool;
use serde_json::Value;
use smg_mcp::BuiltinToolType;

/// Extract caller-declared configuration for the given hosted-tool kind from
/// the request's `tools` declarations.
///
/// Returns `Some(Value::Object(...))` containing only non-null, non-default
/// fields; returns `None` if there is no matching declaration or the matching
/// declaration has no overridable fields.
///
/// # Contract
///
/// - Walks `tools` in order and matches the first declaration whose variant
///   corresponds to `kind` (e.g. `BuiltinToolType::ImageGeneration` →
///   `ResponseTool::ImageGeneration(..)`).
/// - Serializes the inner struct to `Value::Object`, then:
///   - drops entries whose value is `Value::Null`;
///   - drops the `type` discriminator (the kind is already known to the
///     caller and merging it into dispatch args would be spurious).
/// - Returns `None` if the resulting object is empty.
///
/// This preserves forward compatibility: unknown / newly-added fields on a
/// hosted-tool struct are forwarded verbatim without requiring changes here.
pub fn extract_hosted_tool_overrides(
    tools: &[ResponseTool],
    kind: BuiltinToolType,
) -> Option<Value> {
    let declaration = tools.iter().find(|tool| matches_kind(tool, kind))?;
    let serialized = serde_json::to_value(declaration).ok()?;
    let Value::Object(mut map) = serialized else {
        return None;
    };

    map.remove("type");
    map.retain(|_, value| !value.is_null());

    if map.is_empty() {
        None
    } else {
        Some(Value::Object(map))
    }
}

/// Merge caller-declared overrides into model-emitted tool-call arguments.
///
/// Overrides win over model args for overlapping keys; non-overlapping model
/// args are preserved. This is a no-op if `arguments` is not an object or if
/// `overrides` does not contain an object payload.
pub fn apply_hosted_tool_overrides(arguments: &mut Value, overrides: &Value) {
    let Value::Object(args_map) = arguments else {
        return;
    };
    let Some(overrides_map) = overrides.as_object() else {
        return;
    };

    for (key, value) in overrides_map {
        args_map.insert(key.clone(), value.clone());
    }
}

/// Check whether a `ResponseTool` variant matches the given `BuiltinToolType`.
///
/// Only hosted-tool kinds that are routable via MCP are handled; the arms for
/// tool kinds without currently configurable fields still match so the helper
/// stays consistent as `ResponseTool::*` inner structs grow overridable
/// fields over time.
fn matches_kind(tool: &ResponseTool, kind: BuiltinToolType) -> bool {
    matches!(
        (tool, kind),
        (
            ResponseTool::ImageGeneration(_),
            BuiltinToolType::ImageGeneration,
        ) | (
            ResponseTool::WebSearchPreview(_),
            BuiltinToolType::WebSearchPreview,
        ) | (
            ResponseTool::CodeInterpreter(_),
            BuiltinToolType::CodeInterpreter,
        ) | (ResponseTool::FileSearch(_), BuiltinToolType::FileSearch,)
    )
}

#[cfg(test)]
#[path = "../../tests/unit/openai_bridge/overrides/tests.rs"]
mod tests;
