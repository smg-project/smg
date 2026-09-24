//! z.ai (GLM) contract rules, from the providers-verifier golden set recorded
//! against GLM-5.3-Flash and docs.z.ai.
//!
//! The GLM-5.3 series cannot switch thinking off and takes efforts
//! low/high/max; other GLM generations keep their thinking switch and the
//! documented wider set (GLM-5-Next is not pinned: its contract is not
//! recorded, and the registry's media grouping says nothing about it). The
//! sampling defaults go to the models docs.z.ai lists them for. Everything
//! else is z.ai-wide: `thinking.clear_thinking` reaches the chat template,
//! `tool_stream` is consumed, unknown tool types and `file_url` blocks are
//! 400s. `file_url` is refused at ingress on every backend, deliberately:
//! the vendor fetches the file itself and no self-hosted engine does, and
//! validation runs before the backend is known; a pass-through for an
//! upstream z.ai proxy would need the backend at validation time.
//! `max_tokens` beyond the window is the gateway's (`context_length_exceeded`).

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

/// The GLM-5.3 series, the one that cannot switch thinking off, in the
/// spellings the multimodal registry matches too (`registry/glm53_flash.rs`).
const GLM53_MARKERS: [&str; 5] = ["glm-5.3", "glm5.3", "glm_5.3", "glm-5-3", "glm5-3"];

/// The effort levels the GLM-5.3 series accepts.
const GLM53_EFFORTS: [&str; 3] = ["low", "high", "max"];

/// The effort levels docs.z.ai documents for the other GLM generations.
const REASONING_EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

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
        validate_thinking_enabled(req)?;
        validate_efforts(req, &GLM53_EFFORTS, "low, high or max")?;
    } else {
        validate_efforts(
            req,
            &REASONING_EFFORTS,
            "none, minimal, low, medium, high, xhigh or max",
        )?;
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
/// is rejected.
fn validate_thinking_enabled(
    req: &ChatCompletionRequest,
) -> Result<(), validator::ValidationError> {
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
    Ok(())
}

/// The effort must be one the model takes, in each spelling the request
/// carries: both are forwarded, so a bad value hidden behind the preferred
/// one still counts.
fn validate_efforts(
    req: &ChatCompletionRequest,
    allowed: &[&str],
    hint: &str,
) -> Result<(), validator::ValidationError> {
    let efforts = [
        (
            "thinking.effort",
            req.thinking.as_ref().and_then(|t| t.effort.as_deref()),
        ),
        ("reasoning_effort", req.reasoning_effort.as_deref()),
    ];
    for (field, effort) in efforts {
        if effort.is_some_and(|effort| !allowed.contains(&effort)) {
            return Err(pinned("reasoning_effort_not_allowed", field, hint));
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
            // The documented set is wider than GLM-5.3's, but still a set.
            for effort in ["turbo", "ultra"] {
                assert_eq!(
                    validate(&with_model(model, json!({"reasoning_effort": effort}))),
                    Err("reasoning_effort_not_allowed".into()),
                    "{model} {effort}"
                );
            }
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
        // Dash-spelled ids are the same family (registry/glm53_flash.rs);
        // GLM-5-Next is not, its thinking contract being unrecorded.
        for model in [
            "GLM-5.3-Flash",
            "glm-5.3",
            "zai-org/glm5.3-air",
            "glm_5.3",
            "glm-5-3-flash",
            "glm5-3",
        ] {
            assert!(is_glm53(model), "{model}");
        }
        for model in [
            "zai-org/GLM-5.2",
            "glm-5",
            "glm-5.30",
            "zai-org/GLM-5-Next",
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
