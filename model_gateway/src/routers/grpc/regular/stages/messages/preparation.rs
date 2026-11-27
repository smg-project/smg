//! Message API preparation stage: Convert tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    common::{ToolChoice, ToolChoiceValue},
    messages::CreateMessageRequest,
    profile::ProviderProfile,
};
use tracing::{error, warn};

use crate::routers::{
    common::snapshot::RoutingPool,
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
        let spec = Box::new(MessagesResponseSpec::with_tools(
            request,
            ctx.state.messages_tools.clone(),
        ));
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
            tools_for_template.is_some(),
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
                // As in the chat preparation: one WARN and the mark only for
                // a refusal typed as the request's own; every other failure
                // keeps ERROR and stays unmarked, the one that is the
                // template's own whatever the request holds answered in the
                // server's class.
                let refused = e.is_refusal();
                if refused {
                    warn!(function = "MessagePreparationStage::execute", error = %e, "Failed to process messages");
                } else {
                    error!(function = "MessagePreparationStage::execute", error = %e, "Failed to process messages");
                }
                let answer = e.answer();
                return Err(if refused {
                    utils::reported(answer)
                } else {
                    answer
                });
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
                has_tools: spec.has_tools,
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
        ctx.state.prompt_thought = Some(prompt_thought);
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
#[path = "../../../../../../tests/unit/routers/grpc/regular/stages/messages/preparation/tests.rs"]
mod tests;
