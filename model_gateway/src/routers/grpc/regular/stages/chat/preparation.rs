//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ToolChoice, ToolChoiceValue},
};
use tracing::error;

use crate::{
    config::ParserChoice,
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{PreparationOutput, RequestContext},
            multimodal,
            regular::{
                stages::rendered_prompt::{prepare_rendered_prompt, RenderedPrompt},
                symphony,
            },
            utils, ProcessedMessages,
        },
    },
    worker::RoutingPool,
};

/// Chat preparation stage
///
/// Extracts chat-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct ChatPreparationStage {
    /// The pools whose workers this pipeline's requests can reach (the
    /// mode's pools): the multimodal wire dtype is resolved over them.
    wire_pools: &'static [RoutingPool],
}

impl ChatPreparationStage {
    pub(crate) fn new(wire_pools: &'static [RoutingPool]) -> Self {
        Self { wire_pools }
    }
}

#[async_trait]
impl PipelineStage for ChatPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let request = ctx.chat_request_arc();
        self.prepare_chat(ctx, &request).await?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChatPreparation"
    }
}

impl ChatPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Which parser reads this response: the request's X-SMG-Parser header
        // when it carries one, else the gateway's --parser flag (see
        // `regular::symphony`). Decided once here; the constraint below and
        // request building read the same choice.
        let parser = symphony::choice(ctx.components.parser, ctx.input.headers.as_ref())
            .map_err(|why| error::bad_request("invalid_parser_header", why))?;
        ctx.state.parser = Some(parser);
        let (token_ids, processed_messages, tool_constraints) =
            prepare_chat_like(ctx, request, self.wire_pools).await?;
        ctx.state.preparation = Some(PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints,
        });
        Ok(())
    }
}

/// The chat request → prepared inputs pipeline, shared by the chat endpoint
/// and the transcription endpoint (whose backend request is chat-shaped).
///
/// Applies the model's chat template + multimodal expansion, tokenizes,
/// derives `skip_special_tokens`, and builds the stop decoder — writing the
/// intermediate/decoder/skip-special onto `ctx.state`. Returns the pieces the
/// caller stores in its own `PreparationOutput` variant. Does NOT set
/// `ctx.state.preparation`.
pub(crate) async fn prepare_chat_like(
    ctx: &mut RequestContext,
    request: &ChatCompletionRequest,
    wire_pools: &'static [RoutingPool],
) -> Result<(Vec<u32>, ProcessedMessages, symphony::PreparedConstraint), Response> {
    utils::validate_chat_content_parts(&request.messages)
        .map_err(|e| error::bad_request("unsupported_content_part", e))?;
    {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "ChatPreparationStage::prepare_chat").map_err(|e| *e)?;

        // Step 1: the request whose tools the template renders (every tool
        // for a named tool_choice; the constraint built below forces the call;
        // under `none` the tools stay only for a renderer that declares them
        // itself)
        let body_ref = utils::request_as_rendered(request, tokenizer.renderer_capabilities());

        // The media-part order and, for a request with media, the plan and
        // the placeholder tokens, resolved from the registries as the
        // Messages stage resolves them (`utils::resolve_media`).
        let model_id = ctx.input.model_id.as_str();
        let media = utils::resolve_media(&ctx.components, model_id, &*tokenizer, || {
            multimodal::media_plan_chat(&request.messages)
        })
        .await?;
        let media_order = media.media_order;

        // Step 2: Process messages and apply chat template
        let (processed_messages, prompt_encoding) =
            match utils::process_chat_messages_with_placeholders(
                &body_ref,
                &*tokenizer,
                media.placeholders(),
                media_order,
            ) {
                Ok(msgs) => msgs,
                Err(e) => {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process chat messages");
                    return Err(error::bad_request("process_messages_failed", e));
                }
            };

        // Steps 3 and 4: the prompt tokenized the way its renderer said to,
        // the media anchors checked, the media processed or kept as
        // references (shared with the Messages stage).
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
            wire_pools,
            "ChatPreparationStage::execute",
        )
        .await?;

        // Step 4: Build tool constraints if needed
        // The tool parser registry handles both structural tag (for native format
        // parsers like Mistral, KimiK2) and generic JSON schema fallback. When
        // the prompt ends inside the model's thinking block, a parser with a
        // reasoning prefix gets its tag wrapped so a forced call follows the
        // reasoning instead of preempting it. The constraint covers every tool
        // the choice lets the model call, dynamic tools declared on messages
        // included (see `ChatCompletionRequest::callable_tools`).
        let constraint_tools = request.callable_tools();
        // When Symphony reads the response, its table gives a forced call, or a
        // JSON response_format, its tag, so the output comes back in the syntax
        // the table parses (see `regular::symphony`, which decides both with one
        // predicate); the older registry decides for every other request. Only
        // the chat stage chooses a parser: transcription, which shares this
        // preparation, keeps the legacy parsers. The table's refusal waits for
        // request building, which decides on the selected worker: one whose
        // engine takes the whole turn's grammar may serve what the table's
        // tags cannot (`regular::symphony::worker_constraint`).
        let parser: ParserChoice = ctx.state.parser.unwrap_or_default();
        let names = ctx.components.parser_resolver.names(&request.model);
        // The tag takes the start the model's workers name, when they agree; the selected
        // worker's own start decides the `auto` tag at request building.
        let workers = ctx
            .components
            .worker_registry
            .get_by_model(&ctx.input.model_id);
        // The thinking switch's answer, or the table's replay of the rendered
        // prompt where a table reads the response: a turn the prompt leaves
        // unopened starts inside the thought whatever the switch says, and the
        // tag must read it as the engine will (`regular::symphony::reasoning_open`).
        let reasoning = utils::chat_reasoning_starts_in_rendered_prefill(
            request,
            processed_messages.continued_final_message,
            tokenizer.as_ref(),
        ) || symphony::reasoning_open(
            parser,
            &names,
            &request.model,
            &ctx.components.templates,
            &processed_messages.text,
        )
        .is_some_and(|thought| thought.open);
        let symphony_constraint = symphony::constraint(
            parser,
            &names,
            symphony::Asks::Chat(request),
            &constraint_tools,
            reasoning,
            symphony::grammar_start_of(&workers),
            &ctx.components.templates,
        );
        let tool_call_constraint = match (symphony_constraint, request.tool_choice.as_ref()) {
            (Ok(None), Some(tool_choice)) if !constraint_tools.is_empty() => Ok(ctx
                .components
                .tool_parser_factory
                .registry()
                .generate_tool_constraint(
                    ctx.components.parser_resolver.legacy_tool(&names),
                    &constraint_tools,
                    tool_choice,
                    reasoning,
                )
                .map_err(|e| {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request(
                        "invalid_tool_configuration",
                        format!("Invalid tool configuration: {e}"),
                    )
                })?),
            (prepared, _) => prepared,
        };

        let json_schema_constraint = tool_call_constraint
            .as_ref()
            .is_ok_and(|constraint| constraint.as_ref().is_some_and(|c| c.is_json_schema()));
        // A Symphony table whose markers the decoder would drop as special tokens needs them
        // kept whatever the request asks of the channels: without them the table reads neither
        // the thought nor the calls. Where a table reads the response its markers alone decide,
        // and no legacy reasoning parser is built to be asked; under a JSON-schema constraint no
        // table reads (request building makes no assignment: `regular::symphony::assign`), so
        // the legacy parsers answer for themselves, as they do where no table reads at all.
        let preserve_reasoning_special_tokens = match symphony::keeps_special_tokens(
            parser,
            &names,
            &request.model,
            &ctx.components.templates,
            &tokenizer,
        ) {
            Some(kept) if !json_schema_constraint => kept,
            _ => {
                request.separate_reasoning
                    && utils::reasoning_parser_requires_special_tokens(
                        &ctx.components.reasoning_parser_factory,
                        ctx.components
                            .parser_resolver
                            .reasoning_parser(&request.model)
                            .as_deref(),
                        &request.model,
                    )
            }
        };

        let skip_special_tokens = skips_special_tokens(
            request,
            json_schema_constraint,
            preserve_reasoning_special_tokens,
        );

        // Step 5: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            skip_special_tokens,
            request.no_stop_trim,
            request.ignore_eos,
        );

        // Store the intermediate + decoder + derived skip_special_tokens on
        // ctx (PreparationOutput is consumed by request_building before
        // response_processing runs); hand the rest to the caller.
        ctx.state.multimodal_intermediate = multimodal_intermediate;
        ctx.state.multimodal_refs = multimodal_refs;
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(skip_special_tokens);

        Ok((token_ids, processed_messages, tool_call_constraint))
    }
}

/// Whether the engine drops its special tokens from the response. A reasoning
/// parser or a table whose markers are special tokens needs them kept, whatever
/// the request says; a JSON-schema constraint forces the call's JSON, so there
/// are no markers to keep and the request's own setting stands; a request that
/// offers tools the model may call (declared at the request level or on a
/// message, see `ChatCompletionRequest::declares_tools`, the calls not switched
/// off by `tool_choice: none`) keeps them for the tool parser's markers; any
/// other request keeps its own setting.
fn skips_special_tokens(
    request: &ChatCompletionRequest,
    json_schema_constraint: bool,
    preserve_reasoning_special_tokens: bool,
) -> bool {
    if preserve_reasoning_special_tokens {
        return false;
    }
    if json_schema_constraint {
        return request.skip_special_tokens;
    }
    let calls_off = matches!(
        request.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    );
    if request.declares_tools() && !calls_off {
        false
    } else {
        request.skip_special_tokens
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::validated::Normalizable;
    use serde_json::{json, Value};

    use super::*;

    fn tool(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {}}}
        })
    }

    /// A system message declaring `tools`, as the profiles that read such
    /// declarations take it.
    fn declaring(tools: &[Value]) -> Value {
        json!({"role": "system", "content": "", "tools": tools})
    }

    fn user() -> Value {
        json!({"role": "user", "content": "weather?"})
    }

    /// A request as the HTTP layer hands it on: normalized, so an empty `tools`
    /// list has become `tool_choice: none`.
    fn request(value: Value) -> ChatCompletionRequest {
        let mut request: ChatCompletionRequest =
            serde_json::from_value(value).expect("request deserializes");
        request.normalize();
        request
    }

    fn message_level_only() -> ChatCompletionRequest {
        request(json!({
            "model": "kimi-k3",
            "messages": [declaring(&[tool("get_weather")]), user()]
        }))
    }

    #[test]
    fn tools_keep_the_special_tokens_wherever_they_are_declared() {
        let request_level_only = request(json!({
            "model": "kimi-k3",
            "messages": [user()],
            "tools": [tool("get_time")]
        }));
        assert!(!skips_special_tokens(&request_level_only, false, false));

        assert!(!skips_special_tokens(&message_level_only(), false, false));

        let both = request(json!({
            "model": "kimi-k3",
            "messages": [declaring(&[tool("get_weather")]), user()],
            "tools": [tool("get_time")]
        }));
        assert!(!skips_special_tokens(&both, false, false));

        // No tools anywhere: the request's own setting, whichever way it goes.
        let none = request(json!({"model": "kimi-k3", "messages": [user()]}));
        assert!(skips_special_tokens(&none, false, false));
        let kept = request(json!({
            "model": "kimi-k3",
            "messages": [user()],
            "skip_special_tokens": false
        }));
        assert!(!skips_special_tokens(&kept, false, false));
    }

    #[test]
    fn conflicting_declarations_resolve_as_the_other_gates_resolve_them() {
        // An empty request-level list beside a message's declaration: the
        // normalizer's `tool_choice: none` switches the calls off, as it does
        // without the message, so the parser is off too and nothing is kept.
        let empty_list = request(json!({
            "model": "kimi-k3",
            "messages": [declaring(&[tool("get_weather")]), user()],
            "tools": []
        }));
        assert!(matches!(
            empty_list.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        ));
        assert!(skips_special_tokens(&empty_list, false, false));

        // `tool_choice: none` written by the client, over a message's declaration.
        let switched_off = request(json!({
            "model": "kimi-k3",
            "messages": [declaring(&[tool("get_weather")]), user()],
            "tool_choice": "none"
        }));
        assert!(skips_special_tokens(&switched_off, false, false));

        // A message's declaration under a profile that reads none widens nothing.
        let unread = request(json!({
            "model": "self-hosted-model",
            "messages": [declaring(&[tool("get_weather")]), user()]
        }));
        assert!(skips_special_tokens(&unread, false, false));
    }

    #[test]
    fn the_constraint_and_the_reasoning_markers_decide_before_the_tools() {
        // A JSON-schema constraint forces the call's JSON: nothing to keep, the
        // request's setting stands even with a message-declared tool.
        assert!(skips_special_tokens(&message_level_only(), true, false));

        // Reasoning markers the decoder would drop are kept whatever the request says.
        let none = request(json!({"model": "kimi-k3", "messages": [user()]}));
        assert!(!skips_special_tokens(&none, false, true));
        assert!(!skips_special_tokens(&none, true, true));
    }
}
