//! Message API preparation stage: Convert tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    common::{ToolChoice, ToolChoiceValue},
    messages::CreateMessageRequest,
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
        if let Some(refusal) = output_format_refusal(request) {
            return Err(refusal);
        }

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

        // Step 2: Process messages and apply chat template
        let (processed_messages, prompt_encoding) = match message_utils::process_messages(
            request,
            &*tokenizer,
            tools_for_template,
            media.placeholders(),
            media_order,
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
        } = reading::prepare(
            ctx,
            symphony::Asks::Messages {
                request,
                continues: request.continues_final_assistant(),
            },
            &filtered_tools,
            &processed_messages.text,
            utils::messages_reasoning_starts_in_prefill(
                request,
                processed_messages.continued_final_message,
                tokenizer.as_ref(),
            ),
            &tokenizer,
            "MessagePreparationStage::execute",
        )?;

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

/// A structured output format would be dropped with the other pass-through
/// fields and the answer returned unconstrained, so it is refused by name.
fn output_format_refusal(request: &CreateMessageRequest) -> Option<Response> {
    let field = request.output_format_field()?;
    Some(error::bad_request(
        "unsupported_output_format",
        format!(
            "{field} is not supported: the gateway cannot constrain a Messages answer to a format"
        ),
    ))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::{output_format_refusal, CreateMessageRequest};
    use crate::routers::grpc::utils::message_utils::invalid_multimodal_request;

    #[test]
    fn invalid_multimodal_input_maps_to_bad_request() {
        let response = invalid_multimodal_request("unsupported audio modality");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// The stage renders `output_config.effort` as the `reasoning_effort`
    /// template kwarg, renders none without it, and refuses another type.
    #[tokio::test]
    async fn the_stage_renders_output_config_effort() {
        use std::sync::Arc;

        use llm_tokenizer::{traits::Tokenizer, HuggingFaceTokenizer, TokenizerRegistry};
        use openai_protocol::messages::CreateMessageRequest;
        use reasoning_parser::ParserFactory as ReasoningParserFactory;
        use serde_json::{json, Value};
        use tool_parser::ParserFactory as ToolParserFactory;

        use crate::{
            routers::grpc::{
                common::stages::PipelineStage,
                context::{PreparationOutput, RequestContext, SharedComponents},
                regular::{
                    stages::messages::MessagePreparationStage, symphony::template::Templates,
                },
                utils::ParserResolver,
            },
            worker::WorkerRegistry,
        };

        const MODEL: &str = "effort-model";
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
        let template = dir.path().join("template.jinja");
        std::fs::write(
            &template,
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
             effort={{ reasoning_effort | default('unset') }}",
        )
        .unwrap();
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(
            HuggingFaceTokenizer::from_file_with_chat_template(
                path.to_str().unwrap(),
                Some(template.to_str().unwrap()),
            )
            .unwrap(),
        );
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        tokenizer_registry
            .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
            .await
            .unwrap();
        let components = Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(tokenizer_registry.clone())),
            tokenizer_registry,
            worker_registry: Arc::new(WorkerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
        });
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

    /// An output format under either field is a 400 naming the field; a
    /// request without one is not refused.
    #[test]
    fn an_output_format_is_refused_by_the_field_it_comes_under() {
        let request = |extra: serde_json::Value| -> CreateMessageRequest {
            let mut body = serde_json::json!({
                "model": "m", "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}]
            });
            body.as_object_mut()
                .expect("an object")
                .extend(extra.as_object().expect("an object").clone());
            serde_json::from_value(body).expect("a Messages request")
        };
        let format = serde_json::json!({"type": "json_schema", "schema": {"type": "object"}});
        for (extra, field) in [
            (
                serde_json::json!({"output_format": format}),
                "output_format",
            ),
            (
                serde_json::json!({"output_config": {"format": format}}),
                "output_config.format",
            ),
        ] {
            let response = output_format_refusal(&request(extra)).expect("a refusal");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
            assert_eq!(
                response
                    .headers()
                    .get("x-smg-error-code")
                    .and_then(|v| v.to_str().ok()),
                Some("unsupported_output_format"),
                "{field}"
            );
        }
        assert!(
            output_format_refusal(&request(serde_json::json!({"output_format": null}))).is_none()
        );
    }
}
