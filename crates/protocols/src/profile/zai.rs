//! z.ai (GLM) contract rules, from the providers-verifier golden set recorded
//! against GLM-5.3-Flash on 2026-09-22 with the native and the OpenAI SDK.
//!
//! What the vendor does, and what this profile encodes:
//! - thinking cannot be switched off: `thinking.type = disabled` is a 400,
//!   and so is every `reasoning_effort` outside low/high/max (`none`,
//!   `minimal`, `medium`, `xhigh` and unknown values all 400); a GLM-5.3
//!   series rule (`is_glm53`), pinned the way the Kimi profile pins K3:
//!   docs.z.ai has GLM-5.2 taking `disabled` and mapping the other efforts,
//!   and earlier generations taking every value; `adaptive`, a Kimi value
//!   no GLM schema defines, is a 400 for every model;
//! - `thinking.clear_thinking` round-trips: `false` keeps the history's
//!   reasoning in the rendered prompt, `true` drops it; it is carried to the
//!   chat template as the `clear_thinking` kwarg;
//! - `tool_stream` is accepted; SMG streams tool-call deltas regardless, so
//!   the flag is consumed here rather than forwarded to an engine that does
//!   not know it;
//! - sampling defaults are temperature 1 and top_p 0.95 for the models
//!   docs.z.ai lists them for (GLM-5.x, GLM-4.7, GLM-4.6; GLM-4.5 and older
//!   differ and keep the engine's own); out-of-range values the vendor
//!   clamps silently stay 400 here (the OpenAI contract);
//! - an unknown `tools[].type` is a 400;
//! - `file_url` content blocks are z.ai-only (the vendor fetches the file) and
//!   are rejected here with a named code rather than as an unknown part.
//! `max_tokens` beyond the model's window is the gateway's business, where the
//! window is known (`context_length_exceeded`, as for over-long inputs).

use serde_json::Value;

use crate::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent, ThinkingType},
    common::ContentPart,
};

/// The vendor's sampling defaults, applied when the client omits the field so
/// a self-hosted engine does not substitute its own.
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_TOP_P: f32 = 0.95;

/// The model ids docs.z.ai lists those defaults for.
const DEFAULTS_MARKERS: [&str; 5] = ["glm-5", "glm5", "glm_5", "glm-4.6", "glm-4.7"];

/// The GLM-5.3 series, the one that cannot switch thinking off.
const GLM53_MARKERS: [&str; 3] = ["glm-5.3", "glm5.3", "glm_5.3"];

/// The effort levels the GLM-5.3 series accepts.
const REASONING_EFFORTS: [&str; 3] = ["low", "high", "max"];

/// The one tool type the vendor takes.
const TOOL_TYPE: &str = "function";

/// The vendor's streamed-tool-arguments switch; consumed, see the module docs.
const TOOL_STREAM: &str = "tool_stream";

/// The chat-template kwarg `thinking.clear_thinking` maps to.
const CLEAR_THINKING: &str = "clear_thinking";

/// The vendor-side file content block SMG does not fetch.
const FILE_URL: &str = "file_url";

/// Whether any `/`-separated segment of a model id starts with one of the
/// markers, the marker's last digit not continued by another (`glm-5.3`
/// names the 5.3 series, not a `glm-5.30`).
fn model_matches(model: &str, markers: &[&str]) -> bool {
    model.split('/').any(|segment| {
        markers.iter().any(|marker| {
            super::starts_with_ignore_ascii_case(segment, marker)
                && !segment[marker.len()..].starts_with(|c: char| c.is_ascii_digit())
        })
    })
}

/// The GLM-5.3 series (`GLM-5.3`, `glm-5.3-flash`, `glm5.3-air`), whose
/// thinking rules were recorded and are documented as its own.
fn is_glm53(model: &str) -> bool {
    model_matches(model, &GLM53_MARKERS)
}

pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    if model_matches(&req.model, &DEFAULTS_MARKERS) {
        req.temperature.get_or_insert(DEFAULT_TEMPERATURE);
        req.top_p.get_or_insert(DEFAULT_TOP_P);
    }
    if let Some(clear) = req.thinking.as_ref().and_then(|t| t.clear_thinking) {
        req.chat_template_kwargs
            .get_or_insert_default()
            .entry(CLEAR_THINKING.to_string())
            .or_insert(Value::Bool(clear));
    }
    req.other.remove(TOOL_STREAM);
}

pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    let glm53 = is_glm53(&req.model);
    validate_thinking_type(req, glm53)?;
    if glm53 {
        validate_glm53_thinking(req)?;
    }
    validate_tools(req)?;
    validate_content_parts(req)
}

/// `thinking.type` is `enabled` or `disabled` at z.ai, whatever the model;
/// `adaptive` is a Kimi value and is rejected profile-wide. The hint names
/// what the model takes: the GLM-5.3 series has no `disabled`.
fn validate_thinking_type(
    req: &ChatCompletionRequest,
    glm53: bool,
) -> Result<(), validator::ValidationError> {
    let adaptive = req
        .thinking
        .as_ref()
        .is_some_and(|thinking| thinking.r#type == Some(ThinkingType::Adaptive));
    if adaptive {
        let allowed = if glm53 {
            "enabled"
        } else {
            "enabled or disabled"
        };
        return Err(pinned(
            "thinking_type_not_supported",
            "thinking.type",
            allowed,
        ));
    }
    Ok(())
}

/// The GLM-5.3 series thinks always: `disabled` (documented as unsupported)
/// is rejected, and the effort must be one the series takes, in each
/// spelling the request carries: both are forwarded, so a bad value hidden
/// behind the preferred one still counts.
fn validate_glm53_thinking(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if req
        .thinking
        .as_ref()
        .is_some_and(|thinking| thinking.r#type == Some(ThinkingType::Disabled))
    {
        return Err(error(
            "thinking_disabled_not_supported",
            "thinking cannot be disabled for this model".into(),
        ));
    }
    let efforts = [
        (
            "thinking.effort",
            req.thinking.as_ref().and_then(|t| t.effort.as_deref()),
        ),
        ("reasoning_effort", req.reasoning_effort.as_deref()),
    ];
    for (field, effort) in efforts {
        if effort.is_some_and(|effort| !REASONING_EFFORTS.contains(&effort)) {
            return Err(pinned(
                "reasoning_effort_not_allowed",
                field,
                "low, high or max",
            ));
        }
    }
    Ok(())
}

fn validate_tools(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if let Some(tool) = req
        .tools
        .iter()
        .flatten()
        .find(|tool| tool.tool_type != TOOL_TYPE)
    {
        return Err(error(
            "tool_type_not_supported",
            format!(
                "invalid tools[].type '{}': only '{TOOL_TYPE}' is supported",
                tool.tool_type
            ),
        ));
    }
    Ok(())
}

/// `file_url` is served by the vendor's own fetcher; SMG has none, so the
/// block is refused up front instead of failing as an unknown part later.
fn validate_content_parts(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    let parts = req.messages.iter().filter_map(|message| match message {
        ChatMessage::System { content, .. }
        | ChatMessage::User { content, .. }
        | ChatMessage::Developer { content, .. }
        | ChatMessage::Root { content, .. }
        | ChatMessage::Tool { content, .. } => Some(content),
        ChatMessage::Assistant { content, .. } => content.as_ref(),
        ChatMessage::Function { .. } => None,
    });
    for content in parts {
        let MessageContent::Parts(parts) = content else {
            continue;
        };
        let is_file_url = |part: &ContentPart| {
            matches!(part, ContentPart::Unknown(fields)
                if fields.get("type").and_then(|t| t.as_str()) == Some(FILE_URL))
        };
        if parts.iter().any(is_file_url) {
            return Err(error(
                "content_part_not_supported",
                format!("{FILE_URL} content blocks are not supported"),
            ));
        }
    }
    Ok(())
}

fn error(code: &'static str, message: String) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(message.into());
    e
}

fn pinned(code: &'static str, field: &str, allowed: &str) -> validator::ValidationError {
    error(
        code,
        format!("invalid {field}: only {allowed} is allowed for this model"),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{chat::ChatCompletionRequest, profile::ProviderProfile};

    fn request(fields: Value) -> ChatCompletionRequest {
        let mut body = json!({
            "model": "zai-org/GLM-5.3-Flash",
            "messages": [{"role": "user", "content": "Is 97 prime? One word."}]
        });
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).expect("request deserializes")
    }

    fn validate(req: &ChatCompletionRequest) -> Result<(), String> {
        ProviderProfile::for_model(&req.model)
            .validate_chat(req)
            .map_err(|e| e.code.to_string())
    }

    #[test]
    fn sampling_defaults_fill_only_what_the_client_omitted() {
        let mut req = request(json!({}));
        ProviderProfile::Zai.normalize_chat(&mut req);
        assert_eq!(req.temperature, Some(1.0));
        assert_eq!(req.top_p, Some(0.95));

        let mut req = request(json!({"temperature": 0.0, "top_p": 0.5}));
        ProviderProfile::Zai.normalize_chat(&mut req);
        assert_eq!(req.temperature, Some(0.0));
        assert_eq!(req.top_p, Some(0.5));
    }

    #[test]
    fn thinking_cannot_be_disabled() {
        assert_eq!(
            validate(&request(json!({"thinking": {"type": "disabled"}}))),
            Err("thinking_disabled_not_supported".into())
        );
        assert_eq!(
            validate(&request(json!({"thinking": {"type": "adaptive"}}))),
            Err("thinking_type_not_supported".into())
        );
        assert_eq!(
            validate(&request(json!({"thinking": {"type": "enabled"}}))),
            Ok(())
        );
        assert_eq!(validate(&request(json!({}))), Ok(()));
    }

    #[test]
    fn only_low_high_and_max_efforts_are_accepted() {
        for effort in ["low", "high", "max"] {
            assert_eq!(
                validate(&request(json!({"reasoning_effort": effort}))),
                Ok(()),
                "{effort}"
            );
            assert_eq!(
                validate(&request(json!({"thinking": {"effort": effort}}))),
                Ok(()),
                "thinking.effort {effort}"
            );
        }
        for effort in ["none", "minimal", "medium", "xhigh", "ultra", "turbo"] {
            assert_eq!(
                validate(&request(json!({"reasoning_effort": effort}))),
                Err("reasoning_effort_not_allowed".into()),
                "{effort}"
            );
        }
        assert_eq!(
            validate(&request(
                json!({"thinking": {"type": "enabled", "effort": "medium"}})
            )),
            Err("reasoning_effort_not_allowed".into())
        );
        // Both spellings are forwarded, so the shadowed one is checked too.
        assert_eq!(
            validate(&request(json!({
                "thinking": {"type": "enabled", "effort": "low"},
                "reasoning_effort": "medium"
            }))),
            Err("reasoning_effort_not_allowed".into())
        );
    }

    #[test]
    fn clear_thinking_reaches_the_chat_template_without_overriding_an_explicit_kwarg() {
        let mut req = request(json!({"thinking": {"type": "enabled", "clear_thinking": false}}));
        ProviderProfile::Zai.normalize_chat(&mut req);
        assert_eq!(
            req.chat_template_kwargs.as_ref().unwrap()[CLEAR_THINKING],
            Value::Bool(false)
        );

        let mut req = request(json!({
            "thinking": {"type": "enabled", "clear_thinking": true},
            "chat_template_kwargs": {"clear_thinking": false}
        }));
        ProviderProfile::Zai.normalize_chat(&mut req);
        assert_eq!(
            req.chat_template_kwargs.as_ref().unwrap()[CLEAR_THINKING],
            Value::Bool(false),
            "an explicit chat_template_kwargs entry wins"
        );

        let mut req = request(json!({"thinking": {"type": "enabled"}}));
        ProviderProfile::Zai.normalize_chat(&mut req);
        assert!(req.chat_template_kwargs.is_none());
    }

    #[test]
    fn tool_stream_is_accepted_and_consumed() {
        let mut req = request(json!({"tool_stream": true, "stream": true}));
        assert_eq!(req.other.get(TOOL_STREAM), Some(&Value::Bool(true)));
        ProviderProfile::Zai.normalize_chat(&mut req);
        assert!(!req.other.contains_key(TOOL_STREAM));
        assert!(req.stream);
    }

    #[test]
    fn only_function_tools_are_accepted() {
        let tool = |kind: &str| json!({"tools": [{"type": kind, "function": {"name": "x", "parameters": {"type": "object"}}}]});
        assert_eq!(validate(&request(tool("function"))), Ok(()));
        assert_eq!(
            validate(&request(tool("retrieval"))),
            Err("tool_type_not_supported".into())
        );
    }

    #[test]
    fn a_file_url_content_block_is_rejected_by_name() {
        let req = request(json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Summarize this file."},
                {"type": "file_url", "file_url": {"url": "https://cdn.bigmodel.cn/static/demo/demo2.txt"}}
            ]}]
        }));
        assert_eq!(validate(&req), Err("content_part_not_supported".into()));
        let image = request(json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://a/1.png"}}
            ]}]
        }));
        assert_eq!(validate(&image), Ok(()));
    }

    /// Only the GLM-5.3 series cannot switch thinking off (docs.z.ai: "can
    /// only be enabled", efforts low/high/max); GLM-5.2 takes `disabled`
    /// and maps the other efforts, earlier generations take everything,
    /// while the vendor-level rules apply to all of them.
    #[test]
    fn glm53_thinking_rules_do_not_reach_other_generations() {
        let with_model = |model: &str, fields: Value| {
            let mut req = request(fields);
            req.model = model.to_string();
            req
        };
        for model in ["zai-org/GLM-5.2", "glm-5", "zai-org/GLM-4.6"] {
            assert_eq!(
                validate(&with_model(
                    model,
                    json!({"thinking": {"type": "disabled"}})
                )),
                Ok(()),
                "{model}"
            );
            assert_eq!(
                validate(&with_model(model, json!({"reasoning_effort": "none"}))),
                Ok(()),
                "{model}"
            );
            // `adaptive` is a Kimi value; docs.z.ai defines enabled/disabled only.
            assert_eq!(
                validate(&with_model(
                    model,
                    json!({"thinking": {"type": "adaptive"}})
                )),
                Err("thinking_type_not_supported".into()),
                "{model}"
            );
            assert_eq!(
                validate(&with_model(
                    model,
                    json!({
                        "messages": [{"role": "user", "content": [
                            {"type": "file_url", "file_url": {"url": "https://a/f.txt"}}
                        ]}]
                    })
                )),
                Err("content_part_not_supported".into()),
                "{model}"
            );
            let mut req = with_model(model, json!({"tool_stream": true}));
            ProviderProfile::Zai.normalize_chat(&mut req);
            assert!(!req.other.contains_key(TOOL_STREAM), "{model}");
        }
        assert_eq!(
            validate(&with_model(
                "glm-5.3",
                json!({"thinking": {"type": "disabled"}})
            )),
            Err("thinking_disabled_not_supported".into())
        );
        for model in ["GLM-5.3-Flash", "glm-5.3", "zai-org/glm5.3-air", "glm_5.3"] {
            assert!(is_glm53(model), "{model}");
        }
        for model in [
            "zai-org/GLM-5.2",
            "glm-5",
            "glm-5.30",
            "glm-4.6",
            "chatglm3-6b",
        ] {
            assert!(!is_glm53(model), "{model}");
        }
    }

    /// docs.z.ai lists temperature 1 / top_p 0.95 for GLM-5.x, GLM-4.7 and
    /// GLM-4.6; GLM-4.5 and older differ, so they keep the engine's own.
    #[test]
    fn vendor_sampling_defaults_follow_the_documented_models() {
        for model in [
            "zai-org/GLM-5.3-Flash",
            "glm-5.2",
            "glm-5",
            "GLM-4.7",
            "zai-org/glm-4.6",
        ] {
            let mut req = request(json!({}));
            req.model = model.to_string();
            ProviderProfile::Zai.normalize_chat(&mut req);
            assert_eq!(req.temperature, Some(1.0), "{model}");
            assert_eq!(req.top_p, Some(0.95), "{model}");
        }
        for model in ["glm-4.5-air", "THUDM/glm-4-9b-chat", "glm-4.60"] {
            let mut req = request(json!({}));
            req.model = model.to_string();
            ProviderProfile::Zai.normalize_chat(&mut req);
            assert_eq!(req.temperature, None, "{model}");
            assert_eq!(req.top_p, None, "{model}");
        }
    }

    #[test]
    fn root_messages_are_rejected() {
        let req = request(json!({
            "messages": [{"role": "root", "content": "x"}, {"role": "user", "content": "hi"}]
        }));
        assert_eq!(validate(&req), Err("invalid_role".into()));
    }
}
