//! Message API preparation stage: Convert tools, process messages, tokenize, build constraints

use std::fmt::Display;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    common::{StringOrArray, ToolChoice, ToolChoiceValue},
    messages::CreateMessageRequest,
};
use tracing::{debug, error};

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        multimodal,
        utils::{self, message_utils},
    },
};

/// Message API preparation stage
///
/// Parallel to `ChatPreparationStage` but works with `CreateMessageRequest`.
/// Converts Anthropic Messages API types into the internal chat template format,
/// tokenizes, and builds tool constraints.
pub(crate) struct MessagePreparationStage;

fn invalid_multimodal_request(error: impl Display) -> Response {
    error::bad_request(
        "invalid_multimodal_request",
        format!("Invalid multimodal request: {error}"),
    )
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

        // Step 1: Convert Messages API tools to chat tools and filter by tool_choice
        let chat_tools = request
            .tools
            .as_deref()
            .map(message_utils::extract_chat_tools);

        let chat_tool_choice = request
            .tool_choice
            .as_ref()
            .map(message_utils::convert_message_tool_choice);

        // Filter tools by tool_choice (reuse chat utility)
        let filtered_tools = match (&chat_tools, &chat_tool_choice) {
            (Some(tools), Some(tc)) => {
                utils::filter_tools_by_tool_choice(tools, Some(tc)).unwrap_or_else(|| tools.clone())
            }
            (Some(tools), None) => tools.clone(),
            _ => Vec::new(),
        };

        let tools_for_template = if filtered_tools.is_empty() {
            None
        } else {
            Some(filtered_tools.as_slice())
        };

        // Resolve media-part ordering from the model registry so /v1/messages
        // renders each model consistently with /v1/chat/completions.
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

        // Resolve multimodal context once (see chat/preparation.rs for details).
        let media_plan = multimodal::media_plan_messages(&request.messages);
        let (placeholder_tokens, mm_context) = if media_plan.is_empty() {
            (None, None)
        } else if let Some(mm_components) = ctx.components.multimodal.as_ref() {
            let model_id = ctx.input.model_id.as_str();
            let (tokenizer_id, tokenizer_source) = match tokenizer_entry {
                Some(e) => (e.id, e.source),
                None => {
                    error!(
                        function = "MessagePreparationStage::execute",
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
                    function = "MessagePreparationStage::execute",
                    model = %model_id,
                    error = %e,
                    "Failed to resolve multimodal placeholder token"
                );
                invalid_multimodal_request(e)
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
                function = "MessagePreparationStage::execute",
                "Multimodal content detected but multimodal components not initialized"
            );
            return Err(error::bad_request(
                "multimodal_not_supported",
                "Multimodal content detected but multimodal processing is not available",
            ));
        };

        // Step 2: Process messages and apply chat template
        let processed_messages = match message_utils::process_messages(
            request,
            &*tokenizer,
            tools_for_template,
            placeholder_tokens.as_ref(),
            media_order,
        ) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!(function = "MessagePreparationStage::execute", error = %e, "Failed to process messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Step 3: Tokenize the processed text
        let encoding = match utils::encode_blocking(
            tokenizer.clone(),
            processed_messages.text.clone(),
            false,
        )
        .await
        {
            Ok(encoding) => encoding,
            Err(e) => {
                error!(function = "MessagePreparationStage::execute", error = %e, "Tokenization failed");
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
                    function = "MessagePreparationStage::execute",
                    %error,
                    "Rendered multimodal anchors do not match request media"
                );
                error::bad_request("multimodal_prompt_contract_mismatch", error.to_string())
            })?;
        }

        // Step 4: Full multimodal processing (fetch + preprocess + expand tokens + hash),
        // or keep the media references for a worker that processes them itself.
        let mut multimodal_intermediate = None;
        let mut multimodal_refs = None;
        if let (
            Some(placeholders),
            Some((mm_components, model_id, tokenizer_id, tokenizer_source, media_plan)),
        ) = (placeholder_tokens.as_ref(), mm_context)
        {
            let processing = multimodal::resolve_mm_processing(
                mm_components,
                &ctx.components.worker_registry,
                model_id,
                &media_plan,
                placeholders,
            )
            .map_err(|e| error::bad_request(e.code(), e.to_string()))?;
            if processing == multimodal::MmProcessing::Worker {
                debug!(
                    function = "MessagePreparationStage::execute",
                    media_items = media_plan.parts().len(),
                    "Forwarding media references for worker-side processing"
                );
                multimodal_refs = Some(media_plan);
            } else {
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
                            function = "MessagePreparationStage::execute",
                            expanded_tokens = output.expanded_token_ids.len(),
                            "Multimodal processing complete"
                        );
                        token_ids = output.expanded_token_ids;
                        multimodal_intermediate = Some(output.intermediate);
                    }
                    Err(e) => {
                        error!(
                            function = "MessagePreparationStage::execute",
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
        }

        // Step 4: Build tool constraints if tools present
        let tool_call_constraint = if let (false, Some(tool_choice)) =
            (filtered_tools.is_empty(), chat_tool_choice.as_ref())
        {
            ctx.components
                .tool_parser_factory
                .registry()
                .generate_tool_constraint(
                    ctx.components
                        .parser_resolver
                        .tool_parser(&request.model)
                        .as_deref(),
                    &filtered_tools,
                    tool_choice,
                )
                .map_err(|e| {
                    error!(function = "MessagePreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request(
                        "invalid_tool_configuration",
                        format!("Invalid tool configuration: {e}"),
                    )
                })?
        } else {
            None
        };

        // Step 5: Create stop sequence decoder
        let stop_for_decoder = request
            .stop_sequences
            .as_ref()
            .map(|seqs| StringOrArray::Array(seqs.clone()));

        let preserve_reasoning_special_tokens = utils::reasoning_parser_requires_special_tokens(
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
                Some(c) if c.is_json_schema() => true,
                _ if !filtered_tools.is_empty()
                    && !matches!(
                        chat_tool_choice,
                        Some(ToolChoice::Value(ToolChoiceValue::None))
                    ) =>
                {
                    false
                }
                _ => true,
            }
        };

        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            stop_for_decoder.as_ref(),
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
            tool_constraints: tool_call_constraint.map(|c| c.to_tuple()),
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

    use super::invalid_multimodal_request;

    #[test]
    fn invalid_multimodal_input_maps_to_bad_request() {
        let response = invalid_multimodal_request("unsupported audio modality");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
