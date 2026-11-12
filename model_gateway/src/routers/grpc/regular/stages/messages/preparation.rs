//! Message API preparation stage: Convert tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    common::{ToolChoice, ToolChoiceValue},
    messages::CreateMessageRequest,
    profile::ProviderProfile,
};
use tracing::error;

use crate::{
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{PreparationOutput, RequestContext},
            regular::{
                stages::{
                    reading,
                    rendered_prompt::{prepare_rendered_prompt, RenderedPrompt},
                },
                symphony,
            },
            spec::MessagesResponseSpec,
            utils::{self, message_utils},
        },
    },
    worker::RoutingPool,
};

/// Message API preparation stage
///
/// Parallel to `ChatPreparationStage` but works with `CreateMessageRequest`.
/// Converts Anthropic Messages API types into the internal chat template format,
/// tokenizes, and builds tool constraints.
pub(crate) struct MessagePreparationStage {
    /// The pools whose workers this pipeline's requests can reach (the
    /// mode's pools): the multimodal wire dtype is resolved over them.
    wire_pools: &'static [RoutingPool],
}

impl MessagePreparationStage {
    pub(crate) fn new(wire_pools: &'static [RoutingPool]) -> Self {
        Self { wire_pools }
    }
}

#[async_trait]
impl PipelineStage for MessagePreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let request = ctx.messages_request_arc();
        self.prepare_messages(ctx, &request).await?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "MessagePreparation"
    }
}

impl MessagePreparationStage {
    async fn prepare_messages(
        &self,
        ctx: &mut RequestContext,
        request: &CreateMessageRequest,
    ) -> Result<(), Response> {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer = utils::resolve_tokenizer(ctx, "MessagePreparationStage::prepare_messages")
            .map_err(|e| *e)?;
        // Which parser reads this message: the request's X-SMG-Parser header
        // when it carries one, else the gateway's --parser flag (see
        // `regular::symphony`). Decided once here; the constraint below and
        // request building read the same choice.
        let parser = symphony::choice(ctx.components.parser, ctx.input.headers.as_ref())
            .map_err(|why| error::bad_request("invalid_parser_header", why))?;
        ctx.state.parser = Some(parser);

        // Step 1: The response spec, built here because its chat tools (the
        // custom tools, converted once) are what the template renders,
        // narrowed by tool_choice and shared with count_tokens so both render
        // the same prompt; the constraints read the narrowed tools too, the
        // parsers read them all from the spec.
        let spec = Box::new(MessagesResponseSpec::from(request));
        let filtered_tools =
            message_utils::template_tools(&spec.chat_tools, request.tool_choice.as_ref());
        let chat_tool_choice = request.tool_choice.as_ref().map(ToolChoice::from);
        let tools_for_template = (!filtered_tools.is_empty()).then_some(&*filtered_tools);

        // The media-part order and, for a request with media, the plan and the
        // placeholder tokens: resolved once, shared with count_tokens.
        let model_id = ctx.input.model_id.as_str();
        let media = message_utils::resolve_messages_media(
            &ctx.components,
            model_id,
            &*tokenizer,
            &request.messages,
            true,
        )
        .await?;
        let media_order = media.media_order;

        // The model's provider profile, derived once for the request, before
        // the render that reads it (the thinking the prompt is rendered with)
        // and read by the later stages from the state.
        let provider = ProviderProfile::for_model(&request.model);
        ctx.state.provider = Some(provider);

        // Step 2: Process messages and apply chat template
        let (processed_messages, prompt_encoding) = match message_utils::process_messages(
            request,
            &*tokenizer,
            tools_for_template,
            media.placeholders(),
            media_order,
            provider,
        ) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!(function = "MessagePreparationStage::execute", error = %e, "Failed to process messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Steps 3 and 4: the prompt tokenized the way its renderer said to,
        // the media anchors checked, the media processed or kept as
        // references (shared with the chat stage).
        let RenderedPrompt {
            token_ids,
            multimodal_intermediate,
            multimodal_refs,
        } = prepare_rendered_prompt(
            ctx,
            &tokenizer,
            &processed_messages.text,
            prompt_encoding,
            media.media,
            self.wire_pools,
            "MessagePreparationStage::execute",
        )
        .await?;

        // Step 4: Build tool constraints if tools present. When Symphony reads
        // the message, its table gives a forced call its tag (see
        // `regular::symphony`, which decides both with one predicate); the older
        // registry decides for every other request. On a thinking prompt a
        // parser with a reasoning prefix gets its tag wrapped so a forced call
        // follows the reasoning instead of preempting it. The table's refusal
        // waits for request building, which decides on the selected worker
        // (`regular::symphony::worker_constraint`). All of it, and whether the
        // decoder keeps the special tokens the reader needs, is decided as for
        // a chat request (`reading::prepare`), over the tools the template
        // renders.
        let reading::Prepared {
            constraint: tool_call_constraint,
            json_schema: json_schema_constraint,
            keeps_special_tokens: preserve_reasoning_special_tokens,
            names: parser_names,
            prompt_thought,
            reading,
        } = reading::prepare(
            ctx,
            symphony::Asks::Messages {
                request,
                continues: request.continues_final_assistant(),
            },
            &filtered_tools,
            &processed_messages.text,
            utils::messages_reasoning_starts_in_prefill_for(
                request,
                provider,
                processed_messages.continued_final_message,
                tokenizer.as_ref(),
            ),
            &tokenizer,
            "MessagePreparationStage::execute",
        )?;
        ctx.state.parser_names = Some(parser_names);
        ctx.state.prompt_thought = prompt_thought;
        ctx.state.reading = reading;

        // Derive skip_special_tokens from parser and constraint type:
        // - typed reasoning parsers need their control tokens preserved
        // - json_schema: backend forces JSON, no trigger tokens to preserve
        // - structural_tag or no constraint (auto): parser needs trigger tokens
        let skip_special_tokens = if preserve_reasoning_special_tokens {
            false
        } else if json_schema_constraint {
            true
        } else {
            filtered_tools.is_empty()
                || matches!(
                    chat_tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                )
        };

        // Step 5: the stop sequence decoder, over the stop sequences the spec
        // carries on to the stream's decoder.
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            spec.stop_sequences.as_ref(),
            None, // no stop_token_ids in Messages API
            skip_special_tokens,
            false, // no_stop_trim default
            false, // ignore_eos — not available in Messages API
        );

        // Store results in context.
        ctx.state.multimodal_intermediate = multimodal_intermediate;
        ctx.state.multimodal_refs = multimodal_refs;
        ctx.state.preparation = Some(PreparationOutput::Messages {
            token_ids,
            processed_messages,
            tool_constraints: tool_call_constraint,
            spec,
        });

        // Store stop decoder and derived skip_special_tokens for response processing.
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(skip_special_tokens);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use crate::routers::grpc::utils::message_utils::invalid_multimodal_request;

    #[test]
    fn invalid_multimodal_input_maps_to_bad_request() {
        let response = invalid_multimodal_request("unsupported audio modality");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

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
            routers::grpc::{
                common::stages::PipelineStage,
                context::{PreparationOutput, RequestContext},
                regular::stages::{messages::MessagePreparationStage, testing::components_over},
            },
            worker::WorkerRegistry,
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
            crate::config::ParserChoice::Legacy,
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
            let mut ctx = RequestContext::for_messages(
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
            routers::grpc::{
                common::stages::PipelineStage,
                context::{PreparationOutput, RequestContext},
                regular::stages::{messages::MessagePreparationStage, testing::components_over},
            },
            worker::WorkerRegistry,
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
            crate::config::ParserChoice::Legacy,
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
            let mut ctx = RequestContext::for_messages(
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
            routers::grpc::{
                common::stages::PipelineStage,
                context::{PreparationOutput, RequestContext},
                regular::{
                    stages::{messages::MessagePreparationStage, testing::components_over},
                    symphony,
                },
            },
            worker::WorkerRegistry,
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
            crate::config::ParserChoice::Legacy,
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
            let mut ctx = RequestContext::for_messages(
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
}
