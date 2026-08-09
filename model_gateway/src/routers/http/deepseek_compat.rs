//! DeepSeek V4 compatibility for HTTP workers.
//!
//! HTTP workers render the prompt themselves, so the gateway's native
//! tokenizer path cannot apply the official DeepSeek V4 controls for them.
//! Both the regular router and the PD router translate the public controls
//! into the SGLang HTTP worker controls here before forwarding.

use serde_json::Value;

#[inline]
pub(crate) fn is_deepseek_v4_model(model: &str) -> bool {
    const DEEPSEEK_V4: &[u8] = b"deepseek-v4";
    model
        .as_bytes()
        .windows(DEEPSEEK_V4.len())
        .any(|candidate| candidate.eq_ignore_ascii_case(DEEPSEEK_V4))
}

/// Translate the public DeepSeek V4 controls to the SGLang HTTP worker
/// controls: remap `reasoning_effort` to the official levels and project the
/// `thinking` field onto `chat_template_kwargs.thinking` (default on). An
/// explicit `chat_template_kwargs.thinking` from the caller wins. No-op for
/// other models — the `thinking` kwarg name is DeepSeek-specific.
pub(crate) fn apply_deepseek_v4_http_compat(json_request: &mut Value, model_id: &str) {
    if !is_deepseek_v4_model(model_id) {
        return;
    }
    let Some(request) = json_request.as_object_mut() else {
        return;
    };

    // Normalize to the official levels. `none`/`minimal` are not levels the
    // V4 worker understands — they are only the gateway's thinking-off
    // signal, so consume them and drop the field from the forwarded JSON.
    let mut effort_thinking = None;
    if let Some(Value::String(effort)) = request.get_mut("reasoning_effort") {
        match effort.as_str() {
            "low" | "medium" => {
                effort.clear();
                effort.push_str("high");
            }
            "xhigh" => {
                effort.clear();
                effort.push_str("max");
            }
            "none" | "minimal" => {
                effort_thinking = Some(false);
            }
            _ => {}
        }
    }
    if effort_thinking.is_some() {
        request.remove("reasoning_effort");
    }

    // Consume the public `thinking` object unconditionally: the worker only
    // gets the translated kwarg, so it can never contradict the translation.
    let explicit_thinking = request.remove("thinking").and_then(|thinking| {
        thinking
            .get("type")
            .and_then(Value::as_str)
            .and_then(|kind| match kind {
                "enabled" => Some(true),
                "disabled" => Some(false),
                _ => None,
            })
    });

    let template_thinking = request
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .and_then(|kwargs| kwargs.get("thinking"))
        .and_then(Value::as_bool);
    if template_thinking.is_some() {
        return;
    }

    let enabled = explicit_thinking.or(effort_thinking).unwrap_or(true);

    let template_kwargs = request
        .entry("chat_template_kwargs")
        .or_insert_with(|| Value::Object(Default::default()));
    if let Some(template_kwargs) = template_kwargs.as_object_mut() {
        template_kwargs.insert("thinking".to_string(), Value::Bool(enabled));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn deepseek_v4_http_compat_translates_official_controls() {
        for (mut request, expected) in [
            (json!({}), true),
            (json!({"thinking": {"type": "enabled"}}), true),
            (json!({"thinking": {"type": "disabled"}}), false),
            (json!({"reasoning_effort": "none"}), false),
            (json!({"reasoning_effort": "minimal"}), false),
        ] {
            apply_deepseek_v4_http_compat(&mut request, "DeepSeek-V4-Pro");
            assert_eq!(request["chat_template_kwargs"]["thinking"], expected);
            // Translated public controls must not reach the worker: off-signal
            // efforts are not official levels, and the `thinking` object could
            // contradict the injected kwarg on workers that understand it.
            assert!(request.get("reasoning_effort").is_none());
            assert!(request.get("thinking").is_none());
        }
    }

    #[test]
    fn deepseek_v4_http_compat_drops_public_controls_even_with_kwarg_override() {
        let mut request = json!({
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
            "chat_template_kwargs": {"thinking": true},
        });
        apply_deepseek_v4_http_compat(&mut request, "DeepSeek-V4-Pro");
        // The explicit kwarg wins, and a worker that natively understands the
        // official fields must not see the contradicting originals.
        assert_eq!(request["chat_template_kwargs"]["thinking"], true);
        assert!(request.get("reasoning_effort").is_none());
        assert!(request.get("thinking").is_none());
    }

    #[test]
    fn deepseek_v4_http_compat_preserves_internal_override_and_maps_effort() {
        let mut request = json!({
            "thinking": {"type": "enabled"},
            "reasoning_effort": "xhigh",
            "chat_template_kwargs": {"thinking": false},
        });
        apply_deepseek_v4_http_compat(&mut request, "DeepSeek-V4-Pro");
        assert_eq!(request["chat_template_kwargs"]["thinking"], false);
        assert_eq!(request["reasoning_effort"], "max");

        let mut other_model = json!({"reasoning_effort": "high"});
        apply_deepseek_v4_http_compat(&mut other_model, "other-model");
        assert!(other_model.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn deepseek_v4_http_compat_leaves_other_models_untouched() {
        // The `thinking` kwarg name is only meaningful to DeepSeek's renderer;
        // other models (e.g. Qwen `enable_thinking`) must not receive it, and
        // their reasoning_effort must pass through verbatim.
        let original = json!({
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
        });
        let mut request = original.clone();
        apply_deepseek_v4_http_compat(&mut request, "qwen3-30b");
        assert_eq!(request, original);

        let original = json!({"reasoning_effort": "xhigh"});
        let mut request = original.clone();
        apply_deepseek_v4_http_compat(&mut request, "other-model");
        assert_eq!(request, original);
    }
}
