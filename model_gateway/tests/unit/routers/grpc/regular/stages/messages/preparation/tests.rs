/// A byte-level tokenizer over `template`, in a directory the caller keeps
/// alive for as long as the tokenizer is used.
fn byte_tokenizer_with_template(
    template: &str,
) -> (
    tempfile::TempDir,
    std::sync::Arc<dyn llm_tokenizer::traits::Tokenizer>,
) {
    use serde_json::json;

    let dir = tempfile::TempDir::new().unwrap();
    let vocab: serde_json::Map<_, _> = (0..=255u32)
        .map(|byte| (format!("<0x{byte:02X}>"), json!(byte)))
        .collect();
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": null, "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": true}
    });
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, tokenizer.to_string()).unwrap();
    let template_path = dir.path().join("template.jinja");
    std::fs::write(&template_path, template).unwrap();
    let tokenizer = llm_tokenizer::HuggingFaceTokenizer::from_file_with_chat_template(
        path.to_str().unwrap(),
        Some(template_path.to_str().unwrap()),
    )
    .unwrap();
    (dir, std::sync::Arc::new(tokenizer))
}

/// The stage renders `output_config.effort` as the `reasoning_effort`
/// template kwarg, renders none without it, and refuses another type.
#[tokio::test]
async fn the_stage_renders_output_config_effort() {
    use std::sync::Arc;

    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::messages::CreateMessageRequest;
    use serde_json::{json, Value};

    use crate::{
        config::ParserChoice,
        routers::{
            common::snapshot::WorkerRegistry,
            grpc::{
                common::stages::PipelineStage,
                context::{PreparationOutput, RequestContext},
                regular::stages::{messages::MessagePreparationStage, testing::components_over},
            },
        },
    };

    const MODEL: &str = "effort-model";
    let (_dir, tokenizer) = byte_tokenizer_with_template(
        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
             effort={{ reasoning_effort | default('unset') }}",
    );
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    tokenizer_registry
        .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
        .await
        .unwrap();
    let components = components_over(
        tokenizer_registry,
        Arc::new(WorkerRegistry::new()),
        ParserChoice::Legacy,
    );
    for (output_config, rendered) in [
        (json!({"effort": "high"}), Ok("effort=high")),
        (json!({"effort": 2}), Ok("effort=2")),
        (json!({"effort": null}), Ok("effort=unset")),
        (Value::Null, Ok("effort=unset")),
    ] {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": MODEL, "max_tokens": 8, "output_config": output_config,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let mut ctx = RequestContext::for_messages_carrying_its_tools(
            Arc::new(request),
            None,
            MODEL.into(),
            components.clone(),
        );
        let got = match MessagePreparationStage::new(&[]).execute(&mut ctx).await {
            Err(response) => Err(response.status()),
            Ok(()) => match &ctx.state.preparation {
                Some(PreparationOutput::Messages {
                    processed_messages, ..
                }) => Ok(processed_messages.text.lines().last().unwrap().to_string()),
                _ => panic!("no messages preparation"),
            },
        };
        assert_eq!(got, rendered.map(str::to_string), "{output_config}");
    }
}

/// The stage derives the request model's profile once, keeps it on the
/// request state and hands that one to the render: `output_config.effort:
/// none` switches thinking off under both, renders no effort under a
/// MiniMax model id and renders as sent under a self-hosted id, and the
/// stored profile is the model's in both cases.
#[tokio::test]
async fn the_stage_hands_the_render_the_request_models_profile() {
    use std::sync::Arc;

    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::{messages::CreateMessageRequest, profile::ProviderProfile};
    use serde_json::json;

    use crate::{
        config::ParserChoice,
        routers::{
            common::snapshot::WorkerRegistry,
            grpc::{
                common::stages::PipelineStage,
                context::{PreparationOutput, RequestContext},
                regular::stages::{messages::MessagePreparationStage, testing::components_over},
            },
        },
    };

    let (_dir, tokenizer) = byte_tokenizer_with_template(
        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
             thinking={{ enable_thinking | default('unset') }} \
             effort={{ reasoning_effort | default('unset') }}",
    );
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    for model in ["MiniMax-M3", "self-hosted-model"] {
        let loaded = Arc::clone(&tokenizer);
        tokenizer_registry
            .load(model, model, "test", || async move { Ok(loaded) })
            .await
            .unwrap();
    }
    let components = components_over(
        tokenizer_registry,
        Arc::new(WorkerRegistry::new()),
        ParserChoice::Legacy,
    );
    for (model, profile, rendered) in [
        (
            "MiniMax-M3",
            ProviderProfile::Minimax,
            "thinking=False effort=unset",
        ),
        (
            "self-hosted-model",
            ProviderProfile::Generic,
            "thinking=False effort=none",
        ),
    ] {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": model, "max_tokens": 8, "output_config": {"effort": "none"},
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let mut ctx = RequestContext::for_messages_carrying_its_tools(
            Arc::new(request),
            None,
            model.into(),
            components.clone(),
        );
        MessagePreparationStage::new(&[])
            .execute(&mut ctx)
            .await
            .unwrap_or_else(|response| panic!("{model}: {}", response.status()));
        assert_eq!(ctx.state.provider, Some(profile), "{model}");
        let text = match &ctx.state.preparation {
            Some(PreparationOutput::Messages {
                processed_messages, ..
            }) => processed_messages.text.lines().last().unwrap().to_string(),
            _ => panic!("{model}: no messages preparation"),
        };
        assert_eq!(text, rendered, "{model}");
    }
}

/// A structured output format, under `output_config.format` or the
/// field's earlier top-level name, is the answer's schema: where no table
/// reads the response the stage hands it to the engine as its own
/// JSON-schema constraint (a table gives it its tag instead, see
/// `regular::symphony::constraint`). A request without one is constrained
/// by nothing.
#[tokio::test]
async fn the_stage_constrains_the_answer_to_the_output_formats_schema() {
    use std::sync::Arc;

    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::messages::CreateMessageRequest;
    use serde_json::json;

    use crate::{
        config::ParserChoice,
        routers::{
            common::snapshot::WorkerRegistry,
            grpc::{
                common::stages::PipelineStage,
                context::{PreparationOutput, RequestContext},
                regular::{
                    stages::{messages::MessagePreparationStage, testing::components_over},
                    symphony,
                },
            },
        },
    };

    const MODEL: &str = "format-model";
    let (_dir, tokenizer) = byte_tokenizer_with_template(
        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
    );
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    tokenizer_registry
        .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
        .await
        .unwrap();
    let components = components_over(
        tokenizer_registry,
        Arc::new(WorkerRegistry::new()),
        ParserChoice::Legacy,
    );
    let schema = json!({
        "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]
    });
    let format = json!({"type": "json_schema", "schema": schema});
    for (extra, constrained) in [
        (json!({"output_config": {"format": format}}), true),
        (json!({"output_format": format}), true),
        (json!({"output_config": {"effort": "high"}}), false),
        (json!({}), false),
    ] {
        let mut body = json!({
            "model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let request: CreateMessageRequest = serde_json::from_value(body).unwrap();
        let mut ctx = RequestContext::for_messages_carrying_its_tools(
            Arc::new(request),
            None,
            MODEL.into(),
            components.clone(),
        );
        MessagePreparationStage::new(&[])
            .execute(&mut ctx)
            .await
            .unwrap_or_else(|response| panic!("{extra}: {}", response.status()));
        let constraint = match ctx.state.preparation.take() {
            Some(PreparationOutput::Messages {
                tool_constraints, ..
            }) => tool_constraints,
            _ => panic!("{extra}: no messages preparation"),
        };
        let pair = match constraint {
            Ok(symphony::Prepared::Constraint {
                constraint,
                first_token,
                ..
            }) => {
                assert!(!first_token, "{extra}");
                Some(constraint.to_tuple())
            }
            Ok(symphony::Prepared::None) => None,
            other => panic!("{extra}: {other:?}"),
        };
        assert_eq!(
            pair,
            constrained.then(|| ("json_schema".to_string(), schema.to_string())),
            "{extra}"
        );
    }
}

/// Media of a model without a media spec is judged by the template the
/// request renders through, the one for requests with tools when the
/// request declares tools and the default one otherwise: refused where
/// that template takes one string per message, served where it takes
/// content parts.
#[tokio::test]
async fn media_without_a_spec_is_judged_by_the_template_the_request_renders_through() {
    use std::sync::Arc;

    use llm_tokenizer::{chat_template::ChatTemplateContentFormat, mock::MockTokenizer};
    use openai_protocol::messages::CreateMessageRequest;
    use serde_json::json;

    use crate::routers::grpc::{
        common::stages::PipelineStage,
        context::RequestContext,
        regular::stages::{
            messages::MessagePreparationStage, testing::components_forwarding_media_of,
        },
    };

    for (default, with_tools) in [
        (
            ChatTemplateContentFormat::String,
            ChatTemplateContentFormat::OpenAI,
        ),
        (
            ChatTemplateContentFormat::OpenAI,
            ChatTemplateContentFormat::String,
        ),
    ] {
        let tokenizer = MockTokenizer::new()
            .with_deferred_chat_ids(vec![1, 2, 3])
            .with_content_format(default)
            .with_tool_content_format(with_tools);
        let (_dir, components) = components_forwarding_media_of("m", Arc::new(tokenizer)).await;
        for (declares_tools, format) in [(true, with_tools), (false, default)] {
            let mut body = json!({
                "model": "m", "max_tokens": 8,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="
                    }}
                ]}]
            });
            if declares_tools {
                body["tools"] = json!([{
                    "name": "lookup", "description": "Look a thing up",
                    "input_schema": {"type": "object", "properties": {}}
                }]);
            }
            let request: CreateMessageRequest = serde_json::from_value(body).unwrap();
            let mut ctx = RequestContext::for_messages_carrying_its_tools(
                Arc::new(request),
                None,
                "m".into(),
                components.clone(),
            );
            let refused_with = MessagePreparationStage::new(&[])
                .execute(&mut ctx)
                .await
                .err()
                .map(|response| {
                    response
                        .headers()
                        .get("x-smg-error-code")
                        .and_then(|code| code.to_str().ok())
                        .unwrap_or_default()
                        .to_owned()
                });
            assert_eq!(
                refused_with.as_deref(),
                (format == ChatTemplateContentFormat::String).then_some("multimodal_not_supported"),
                "declares tools: {declares_tools}"
            );
        }
    }
}
