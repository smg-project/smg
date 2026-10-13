//! The chat renderers with an encoder of their own, and what each answers
//! for the template questions the gateway asks around a render.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    chat_template::{ChatTemplateContentFormat, ThinkingKeyName, ThinkingToggle},
    traits::RendererCapabilities,
};

mod deepseek_common;
pub mod deepseek_v32;
pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod kimi_k25_tools;
pub mod kimi_k3_xtml;

/// A renderer's own rule for the thinking mode a request renders with, from
/// the template kwargs and the request's `thinking` field.
pub(crate) type ThinkingModeRule = fn(Option<&HashMap<String, Value>>, Option<bool>) -> bool;

/// What a renderer with its own encoder answers for the questions the
/// gateway asks a tokenizer around a render (the `Tokenizer` trait's
/// `chat_template_content_format`, `thinking_toggle`, `thinking_key_name`,
/// the `reasoning_effort` words, `think_in_prefill`, `renderer_capabilities`,
/// `native_thinking_mode`): one table per renderer, kept beside it. The Jinja
/// renderer answers them from the template instead
/// ([`crate::chat_template::ChatTemplateState`]), so a wrapper reads the
/// table when a native renderer applies and asks the template otherwise.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RendererProfile {
    /// The renderer's own content format; `None` for a renderer with no
    /// opinion on content parts, which reports the template's (usually
    /// absent, so the default).
    pub(crate) content_format: Option<ChatTemplateContentFormat>,
    /// The thinking switch and its default. The Jinja processor knows
    /// nothing of a native renderer, so the renderer states it.
    pub(crate) thinking_toggle: ThinkingToggle,
    /// The kwarg the switch reads.
    pub(crate) thinking_key_name: ThinkingKeyName,
    /// `chat_template_kwargs.reasoning_effort` values that switch thinking
    /// on, so the gateway's parser is armed from the kwarg as the prompt is;
    /// empty when the renderer takes only the protocol's own words.
    pub(crate) native_effort_values: &'static [&'static str],
    /// The values that switch it off; empty when the renderer switches off
    /// on the protocol's `none` / `minimal` and declares no words of its own.
    pub(crate) native_effort_off_values: &'static [&'static str],
    /// The prompt ends inside an open think block when thinking is on, so a
    /// completion starts mid-reasoning and the parser must be told. Every
    /// native renderer reports a thinking switch, so the switchless-template
    /// question `prefill_opens_think_block` mirrors this.
    pub(crate) think_in_prefill: bool,
    /// The behaviours the gateway mirrors when it prepares the request.
    pub(crate) capabilities: RendererCapabilities,
    /// The renderer's own rule for the thinking mode a request renders with,
    /// when the parser must be armed from the same answer; `None` leaves the
    /// mode to the gateway's reading of the kwargs.
    pub(crate) thinking_mode: Option<ThinkingModeRule>,
}

impl RendererProfile {
    /// The behaviours no native renderer has unless its table says so.
    pub(crate) const NO_CAPABILITIES: RendererCapabilities = RendererCapabilities {
        enable_thinking_alias: false,
        native_assistant_continuation: false,
        raw_tool_call_arguments: false,
    };
}
