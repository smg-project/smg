//! vLLM Chat request construction for reasoning-aware structured output.

use openai_protocol::chat::ChatCompletionRequest;
use serde_json::{json, Value};
use smg_grpc_client::{vllm_proto, vllm_proto::sampling_params::Constraint, VllmEngineClient};

pub(super) fn build_request(
    request_id: String,
    body: &ChatCompletionRequest,
    processed_text: String,
    token_ids: Vec<u32>,
    multimodal_inputs: Option<vllm_proto::MultimodalInputs>,
    tool_constraint: Option<(String, String)>,
    starts_in_reasoning: bool,
) -> Result<vllm_proto::GenerateRequest, String> {
    // Build first so conflicting request constraints retain their existing errors.
    let has_tool_constraint = tool_constraint.is_some();
    let mut request = VllmEngineClient::build_generate_request_from_chat(
        request_id,
        body,
        processed_text,
        token_ids,
        multimodal_inputs,
        tool_constraint,
    )?;
    if !starts_in_reasoning
        || has_tool_constraint
        || !body
            .model
            .split('/')
            .any(|part| part.eq_ignore_ascii_case("deepseek-v4.1-flash"))
    {
        return Ok(request);
    }
    if let Some(params) = request.sampling_params.as_mut() {
        if let Some(Constraint::JsonSchema(schema)) = params.constraint.as_ref() {
            let schema: Value = serde_json::from_str(schema)
                .map_err(|err| format!("Invalid JSON schema for reasoning output: {err}"))?;
            // vLLM without a reasoning parser constrains generation from the first
            // token. Leave reasoning free and constrain only the final-answer region.
            // separate_reasoning controls extraction, not this template state.
            let tag = json!({
                "type": "structural_tag",
                "format": {"type": "sequence", "elements": [
                    {"type": "any_text", "excludes": ["</think>"]},
                    {"type": "const_string", "value": "</think>"},
                    {"type": "json_schema", "json_schema": schema}
                ]}
            });
            params.constraint = Some(Constraint::StructuralTag(tag.to_string()));
            params.skip_special_tokens = false;
        }
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use openai_protocol::{common::ResponseFormat, validated::Normalizable};
    use serde_json::{json, Value};
    use smg_grpc_client::vllm_proto::sampling_params::Constraint;

    use super::*;

    fn request(extra: Value) -> ChatCompletionRequest {
        let mut value = json!({"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Return JSON."}],"response_format":{"type":"json_object"}});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let mut req: ChatCompletionRequest = serde_json::from_value(value).unwrap();
        req.normalize();
        req
    }

    fn build(
        req: &ChatCompletionRequest,
        reasoning: bool,
        tools: Option<(String, String)>,
    ) -> vllm_proto::SamplingParams {
        build_request(
            "test".into(),
            req,
            "prefill".into(),
            vec![42],
            None,
            tools,
            reasoning,
        )
        .unwrap()
        .sampling_params
        .unwrap()
    }

    #[test]
    fn reasoning_json_preserves_schema_after_required_reasoning_boundary() {
        let schema = json!({"$defs":{"label":{"enum":["amber"]}},"type":"object","properties":{"label":{"$ref":"#/$defs/label"},"count":{"type":["integer","null"]}},"required":["label","count"],"additionalProperties":false});
        for format in [
            json!({"type":"json_object"}),
            json!({"type":"json_schema","json_schema":{"name":"fixture","strict":true,"schema":schema}}),
        ] {
            let req = request(json!({"response_format":format}));
            let p = build(&req, true, None);
            let Some(Constraint::StructuralTag(tag)) = p.constraint else {
                panic!("thinking JSON needs a structural tag")
            };
            let tag: Value = serde_json::from_str(&tag).unwrap();
            let expected_schema = if matches!(req.response_format, Some(ResponseFormat::JsonObject))
            {
                json!({"type":"object"})
            } else {
                schema.clone()
            };
            assert_eq!(tag["format"]["type"], "sequence");
            assert_eq!(
                tag["format"]["elements"],
                json!([
                    {"type":"any_text","excludes":["</think>"]},
                    {"type":"const_string","value":"</think>"},
                    {"type":"json_schema","json_schema":expected_schema}
                ])
            );
            assert!(
                !p.skip_special_tokens,
                "retain delimiters in any backend-decoded text"
            );
        }
    }

    #[test]
    fn reasoning_json_does_not_change_other_models_or_final_content_state() {
        for model in [
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-ai/DeepSeek-R1",
            "Qwen/Qwen3-8B",
            "openai/gpt-oss-20b",
            "opaque-alias",
            "DeepSeek-V4.1-Flash-extra",
        ] {
            assert!(
                matches!(
                    build(&request(json!({"model":model})), true, None).constraint,
                    Some(Constraint::JsonSchema(_))
                ),
                "{model}"
            );
        }
        assert!(matches!(
            build(&request(json!({})), false, None).constraint,
            Some(Constraint::JsonSchema(_))
        ));
        assert!(build(
            &request(json!({"response_format":{"type":"text"}})),
            true,
            None
        )
        .constraint
        .is_none());
    }

    #[test]
    fn reasoning_json_preserves_tool_precedence_and_constraint_conflicts() {
        for (kind, value) in [
            ("json_schema", "{\"type\":\"array\"}"),
            ("structural_tag", "{\"format\":{\"type\":\"any_text\"}}"),
        ] {
            let p = build(&request(json!({})), true, Some((kind.into(), value.into())));
            match p.constraint.unwrap() {
                Constraint::JsonSchema(s) | Constraint::StructuralTag(s) => assert_eq!(s, value),
                other => panic!("unexpected tool constraint: {other:?}"),
            }
        }
        for extra in [json!({"regex":"[a-z]+"}), json!({"ebnf":"root ::= \"a\""})] {
            assert!(build_request(
                "test".into(),
                &request(extra),
                "prefill".into(),
                vec![42],
                None,
                None,
                true
            )
            .is_err());
        }
    }
    #[test]
    fn reasoning_json_uses_resolved_template_state_including_continuation() {
        use llm_tokenizer::{
            chat_template::{ThinkingKeyName, ThinkingToggle},
            traits::RendererCapabilities,
        };

        use crate::routers::grpc::utils::chat_reasoning_starts_in_prefill;
        let tokenizer = llm_tokenizer::MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::DefaultOn)
            .with_thinking_key_name(ThinkingKeyName::Thinking)
            .with_native_reasoning_effort_values(&["low", "high", "xhigh", "max"])
            .with_renderer_capabilities(RendererCapabilities {
                enable_thinking_alias: true,
                native_assistant_continuation: true,
                raw_tool_call_arguments: true,
            });
        for (extra, wrapped) in [
            (json!({}), true),
            (json!({"separate_reasoning":false}), true),
            (json!({"model":"/models/DEEPSEEK-V4.1-FLASH"}), true),
            (json!({"thinking":{"type":"disabled"}}), false),
            (json!({"reasoning_effort":"none"}), false),
            (
                json!({"chat_template_kwargs":{"enable_thinking":false}}),
                false,
            ),
            (
                json!({"thinking":{"type":"disabled"},"chat_template_kwargs":{"thinking":true}}),
                true,
            ),
            (
                json!({"continue_final_message":true,"messages":[{"role":"assistant","content":"{\"label\":"}]}),
                false,
            ),
        ] {
            let req = request(extra.clone());
            let p = build(
                &req,
                chat_reasoning_starts_in_prefill(&req, &tokenizer),
                None,
            );
            assert_eq!(
                matches!(p.constraint, Some(Constraint::StructuralTag(_))),
                wrapped,
                "{extra}"
            );
        }
    }
}
