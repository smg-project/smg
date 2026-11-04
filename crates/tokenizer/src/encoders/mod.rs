//! The chat renderers with an encoder of their own, what each answers for
//! the template questions the gateway asks around a render, and the segment
//! form of a prompt such a renderer hands its backend: control tokens the
//! renderer wrote, and plain text in which the same spellings stay text.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    chat_template::{
        ChatTemplateContentFormat, ChatTemplateState, ThinkingKeyName, ThinkingToggle,
    },
    traits::{MediaHandling, RendererCapabilities},
};

mod deepseek_common;
pub mod deepseek_v32;
pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod glm53;
pub mod kimi_k25_tools;
pub mod kimi_k3_xtml;
pub(crate) mod marks;

/// One piece of a rendered prompt, tagged with how special-token strings in
/// it must be encoded (the Kimi reference's `EncodeSegment { text,
/// allow_special }`; the DeepSeek reference's special ranges). Each backend
/// encodes a list of these: the tiktoken backend for the K3 renderer, the
/// Hugging Face backend for the DeepSeek renderers (see `marks`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptSegment {
    pub(crate) text: String,
    /// `true`: special-token strings map to their control ids (structural
    /// markers emitted by the renderer). `false`: ordinary BPE, so a literal
    /// marker string inside message text stays text.
    pub(crate) allow_special: bool,
}

impl PromptSegment {
    pub(crate) fn control(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            allow_special: true,
        }
    }

    pub(crate) fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            allow_special: false,
        }
    }
}

/// The flat prompt string: segment texts concatenated in order.
pub(crate) fn join_segments(segments: &[PromptSegment]) -> String {
    segments.iter().map(|s| s.text.as_str()).collect()
}

/// Python's truthiness of a JSON value, the test the reference encoders and
/// the Jinja templates put on an optional field (`if response_format:`):
/// `null`, `false`, zero and an empty string, array or object are all
/// nothing.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

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
        media_handling: MediaHandling::Preprocess,
        enable_thinking_alias: false,
        native_assistant_continuation: false,
        raw_tool_call_arguments: false,
        leading_system_only: false,
    };

    /// The behaviours of a renderer without a profile: the Jinja template,
    /// which continues a trailing assistant message the way transformers'
    /// `continue_final_message` does.
    pub(crate) const TEMPLATE_CAPABILITIES: RendererCapabilities = RendererCapabilities {
        native_assistant_continuation: true,
        ..Self::NO_CAPABILITIES
    };
}

/// What a tokenizer answers to the template questions the gateway asks: a
/// renderer with an encoder of its own answers from its profile, the Jinja
/// template answers otherwise. The two tokenizers hold the same pair and read
/// it through this one view, so the rule is written once.
pub(crate) struct TemplateAnswers<'a> {
    pub(crate) profile: Option<&'a RendererProfile>,
    pub(crate) template: &'a ChatTemplateState,
}

impl TemplateAnswers<'_> {
    pub(crate) fn content_format(&self) -> ChatTemplateContentFormat {
        self.profile
            .and_then(|profile| profile.content_format)
            .unwrap_or_else(|| self.template.content_format())
    }

    /// The content format of the template a request renders through, by
    /// whether it declares tools; a profile's own format wins.
    pub(crate) fn content_format_for_tools(&self, has_tools: bool) -> ChatTemplateContentFormat {
        self.profile
            .and_then(|profile| profile.content_format)
            .unwrap_or_else(|| self.template.content_format_for_tools(has_tools))
    }

    pub(crate) fn thinking_toggle(&self) -> ThinkingToggle {
        self.profile.map_or_else(
            || self.template.thinking_toggle(),
            |profile| profile.thinking_toggle,
        )
    }

    pub(crate) fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        self.profile.map_or_else(
            || self.template.thinking_key_name(),
            |profile| Some(profile.thinking_key_name),
        )
    }

    pub(crate) fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        self.profile.map_or_else(
            || self.template.native_reasoning_effort_values(),
            |profile| profile.native_effort_values,
        )
    }

    pub(crate) fn native_reasoning_effort_off_values(&self) -> &'static [&'static str] {
        self.profile.map_or_else(
            || self.template.native_reasoning_effort_off_values(),
            |profile| profile.native_effort_off_values,
        )
    }

    pub(crate) fn think_in_prefill(&self) -> bool {
        self.profile.map_or_else(
            || self.template.think_in_prefill(),
            |profile| profile.think_in_prefill,
        )
    }

    /// Every native renderer reports a thinking switch, so its profile's
    /// `think_in_prefill` answers the switchless-template question too.
    pub(crate) fn prefill_opens_think_block(&self) -> bool {
        self.profile.map_or_else(
            || self.template.prefill_opens_think_block(),
            |profile| profile.think_in_prefill,
        )
    }

    /// The profile's capabilities, or the Jinja template's: it continues a
    /// trailing assistant message the way the reference's
    /// `continue_final_message` does, takes a history call's arguments as the
    /// text the request sent when it writes them that way, and handles media
    /// as `media_handling` says (resolved by the tokenizer that owns the
    /// template).
    pub(crate) fn renderer_capabilities(
        &self,
        media_handling: MediaHandling,
    ) -> RendererCapabilities {
        self.profile.map_or_else(
            || RendererCapabilities {
                media_handling,
                raw_tool_call_arguments: self.template.raw_tool_call_arguments(),
                ..RendererProfile::TEMPLATE_CAPABILITIES
            },
            |profile| profile.capabilities,
        )
    }

    /// The thinking mode the profile's own rule renders for the kwargs and
    /// the request-level toggle; `None` without a profile or a rule.
    pub(crate) fn native_thinking_mode(
        &self,
        template_kwargs: Option<&HashMap<String, Value>>,
        thinking: Option<bool>,
    ) -> Option<bool> {
        self.profile
            .and_then(|profile| profile.thinking_mode)
            .map(|resolve| resolve(template_kwargs, thinking))
    }
}

/// Whether a marked render of the V4 family ends in the generation header its
/// encoder writes at the final transition: the assistant token and the think
/// token that opens or closes the reasoning, with the action task's token
/// after them for an action task. Read from the render, so it follows the
/// encoder's own preprocessing (merged tool turns, dropped messages), and
/// request text cannot forge a mark.
pub(crate) fn ends_with_generation_header(marked: &str) -> bool {
    let tail = deepseek_common::task_sp_token("action")
        .and_then(|action| marked.strip_suffix(action))
        .unwrap_or(marked);
    let header = |think: &str| {
        tail.strip_suffix(think)
            .is_some_and(|rest| rest.ends_with(deepseek_common::ASSISTANT))
    };
    header(deepseek_common::THINKING_START) || header(deepseek_common::THINKING_END)
}
