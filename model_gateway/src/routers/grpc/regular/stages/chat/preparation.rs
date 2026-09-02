//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ToolChoice, ToolChoiceValue},
};
use tracing::{debug, error};

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        multimodal, utils, ProcessedMessages,
    },
};

/// Chat preparation stage
///
/// Extracts chat-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct ChatPreparationStage;

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
        let (token_ids, processed_messages, tool_constraints) =
            prepare_chat_like(ctx, request).await?;
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
) -> Result<(Vec<u32>, ProcessedMessages, Option<(String, String)>), Response> {
    {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "ChatPreparationStage::prepare_chat").map_err(|e| *e)?;

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Resolve media-part ordering from the model registry so it stays owned
        // by the per-model spec. Falls back to vLLM-compatible media-first when
        // the model has no multimodal components or matches no spec.
        let model_id = ctx.input.model_id.as_str();
        let tokenizer_entry = ctx
            .components
            .tokenizer_registry
            .get_by_name(model_id)
            .or_else(|| ctx.components.tokenizer_registry.get_by_id(model_id));
        let media_order = match (ctx.components.multimodal.as_ref(), tokenizer_entry.as_ref()) {
            (Some(mm_components), Some(entry)) => {
                multimodal::resolve_media_part_order(
                    model_id,
                    &*tokenizer,
                    mm_components,
                    &entry.id,
                    &entry.source,
                )
                .await
            }
            _ => llm_multimodal::MediaPartOrder::MediaFirst,
        };

        // Normalize media once. The same plan drives placeholder resolution,
        // rendering, fetching, preprocessing, and final count validation.
        let media_plan = multimodal::media_plan_chat(&request.messages);
        let (placeholder_tokens, mm_context) = if media_plan.is_empty() {
            (None, None)
        } else if let Some(mm_components) = ctx.components.multimodal.as_ref() {
            let model_id = ctx.input.model_id.as_str();
            let (tokenizer_id, tokenizer_source) = match tokenizer_entry {
                Some(e) => (e.id, e.source),
                None => {
                    error!(
                        function = "ChatPreparationStage::execute",
                        model = %model_id,
                        "Tokenizer entry not found for multimodal processing"
                    );
                    return Err(error::bad_request(
                        "multimodal_config_missing",
                        format!("Tokenizer not found for model: {model_id}"),
                    ));
                }
            };

            let placeholders = multimodal::prepare_placeholder_tokens(
                &media_plan,
                model_id,
                &*tokenizer,
                mm_components,
                &tokenizer_id,
                &tokenizer_source,
            )
            .await
            .map_err(|e| {
                error!(
                    function = "ChatPreparationStage::execute",
                    model = %model_id,
                    error = %e,
                    "Failed to prepare multimodal prompt plan"
                );
                error::bad_request(
                    "invalid_multimodal_request",
                    format!("Invalid multimodal request: {e}"),
                )
            })?;

            (
                Some(placeholders),
                Some((
                    mm_components,
                    model_id,
                    tokenizer_id,
                    tokenizer_source,
                    media_plan,
                )),
            )
        } else {
            error!(
                function = "ChatPreparationStage::execute",
                "Multimodal content detected but multimodal components not initialized"
            );
            return Err(error::bad_request(
                "multimodal_not_supported",
                "Multimodal content detected but multimodal processing is not available",
            ));
        };

        // Step 2: Process messages and apply chat template
        let processed_messages = match utils::process_chat_messages_with_placeholders(
            &body_ref,
            &*tokenizer,
            placeholder_tokens.as_ref(),
            media_order,
        ) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process chat messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Step 3: Tokenize the processed text (no special tokens - chat template already handles them)
        let encoding = match utils::encode_blocking(
            tokenizer.clone(),
            processed_messages.text.clone(),
            false,
        )
        .await
        {
            Ok(encoding) => encoding,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Tokenization failed");
                return Err(error::internal_error(
                    "tokenization_failed",
                    format!("Tokenization failed: {e}"),
                ));
            }
        };

        let mut token_ids = encoding.token_ids().to_vec();

        if let (Some(placeholders), Some((_, _, _, _, media_plan))) =
            (placeholder_tokens.as_ref(), mm_context.as_ref())
        {
            multimodal::validate_rendered_media_anchors(
                media_plan,
                placeholders,
                &*tokenizer,
                &token_ids,
            )
            .map_err(|error| {
                error!(
                    function = "ChatPreparationStage::execute",
                    %error,
                    "Rendered multimodal anchors do not match request media"
                );
                error::bad_request("multimodal_prompt_contract_mismatch", error.to_string())
            })?;
        }

        // Step 4: Full multimodal processing (fetch + preprocess + expand tokens + hash)
        let mut multimodal_intermediate = None;
        if let Some((mm_components, model_id, tokenizer_id, tokenizer_source, media_plan)) =
            mm_context
        {
            match multimodal::process_multimodal_plan(
                media_plan,
                model_id,
                &*tokenizer,
                token_ids,
                mm_components,
                &tokenizer_id,
                &tokenizer_source,
            )
            .await
            {
                Ok(output) => {
                    debug!(
                        function = "ChatPreparationStage::execute",
                        expanded_tokens = output.expanded_token_ids.len(),
                        "Multimodal processing complete"
                    );
                    token_ids = output.expanded_token_ids;
                    multimodal_intermediate = Some(output.intermediate);
                }
                Err(e) => {
                    error!(
                        function = "ChatPreparationStage::execute",
                        error = %e,
                        "Multimodal processing failed"
                    );
                    return Err(error::bad_request(
                        "multimodal_processing_failed",
                        format!("Multimodal processing failed: {e}"),
                    ));
                }
            }
        }

        // Step 4: Build tool constraints if needed
        // The tool parser registry handles both structural tag (for native format
        // parsers like Mistral, KimiK2) and generic JSON schema fallback.
        let tool_call_constraint = if let (Some(tools), Some(tool_choice)) =
            (body_ref.tools.as_ref(), request.tool_choice.as_ref())
        {
            ctx.components
                .tool_parser_factory
                .registry()
                .generate_tool_constraint(
                    ctx.components
                        .parser_resolver
                        .tool_parser(&request.model)
                        .as_deref(),
                    tools,
                    tool_choice,
                )
                .map_err(|e| {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request(
                        "invalid_tool_configuration",
                        format!("Invalid tool configuration: {e}"),
                    )
                })?
        } else {
            None
        };

        let preserve_reasoning_special_tokens = request.separate_reasoning
            && utils::reasoning_parser_requires_special_tokens(
                &ctx.components.reasoning_parser_factory,
                ctx.components
                    .parser_resolver
                    .reasoning_parser(&request.model)
                    .as_deref(),
                &request.model,
            );

        // Derive skip_special_tokens from parser and constraint type:
        // - typed reasoning parsers need their control tokens preserved
        // - json_schema: backend forces JSON, no trigger tokens to preserve
        // - structural_tag or no constraint (auto): parser needs trigger tokens
        let skip_special_tokens = if preserve_reasoning_special_tokens {
            false
        } else {
            match &tool_call_constraint {
                Some(c) if c.is_json_schema() => request.skip_special_tokens,
                _ if request.tools.is_some()
                    && !matches!(
                        request.tool_choice,
                        Some(ToolChoice::Value(ToolChoiceValue::None))
                    ) =>
                {
                    false
                }
                _ => request.skip_special_tokens,
            }
        };

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
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(skip_special_tokens);

        Ok((
            token_ids,
            processed_messages,
            tool_call_constraint.map(|c| c.to_tuple()),
        ))
    }
}
