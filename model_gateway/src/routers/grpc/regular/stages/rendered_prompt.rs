//! What the chat and Messages preparation stages do alike once the template
//! has rendered: tokenize the prompt the way its renderer said to, check the
//! rendered media anchors against the plan, and either process the media
//! (fetch, preprocess, expand the tokens, hash) or keep the references for a
//! worker that processes them itself. The two stages render differently and
//! continue from the rendered text through one function.

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::traits::{PromptEncoding, Tokenizer};
use tracing::{debug, error};

use crate::{
    routers::{
        error,
        grpc::{
            context::RequestContext,
            multimodal::{self, MediaPlan, MultimodalIntermediate},
            utils::{self, MediaContext},
        },
    },
    worker::RoutingPool,
};

/// A rendered prompt as the stages hand it on: the token ids, expanded for
/// media processed here, and the media as processed or as references.
pub(crate) struct RenderedPrompt {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) multimodal_intermediate: Option<MultimodalIntermediate>,
    pub(crate) multimodal_refs: Option<MediaPlan>,
}

/// Tokenize the rendered `text` as `prompt_encoding` says and process the
/// request's `media`, when it carries any. `wire_pools` are the pools the
/// pipeline's mode selects from, for the multimodal wire dtype; `stage`
/// names the caller in the log lines. A plan a worker processes itself pins
/// `ctx.state.media_refs_pipeline` and comes back as the references.
pub(crate) async fn prepare_rendered_prompt(
    ctx: &mut RequestContext,
    tokenizer: &Arc<dyn Tokenizer>,
    text: &str,
    prompt_encoding: PromptEncoding,
    media: Option<MediaContext>,
    wire_pools: &'static [RoutingPool],
    stage: &'static str,
) -> Result<RenderedPrompt, Response> {
    // Step 3: Tokenize the prompt the way its renderer said to (a flat
    // encode of the text, or the encode the renderer prepared)
    let encoding =
        match utils::encode_prompt_blocking(tokenizer.clone(), text, prompt_encoding).await {
            Ok(encoding) => encoding,
            Err(e) => {
                error!(function = stage, error = %e, "Tokenization failed");
                return Err(error::internal_error(
                    "tokenization_failed",
                    format!("Tokenization failed: {e}"),
                ));
            }
        };

    let mut token_ids = encoding.token_ids().to_vec();

    if let Some(media) = media.as_ref() {
        multimodal::validate_rendered_media_anchors(
            &media.plan,
            &media.placeholders,
            &**tokenizer,
            &token_ids,
        )
        .map_err(|error| {
            error!(
                function = stage,
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
    if let Some(MediaContext {
        plan: media_plan,
        placeholders,
        components: mm_components,
        tokenizer_id,
        tokenizer_source,
    }) = media
    {
        let model_id = ctx.input.model_id.as_str();
        let processing = multimodal::resolve_mm_processing(
            &mm_components,
            &ctx.components.worker_registry,
            model_id,
            &media_plan,
            &placeholders,
        )
        .map_err(multimodal::MmRefsError::into_response)?;
        if processing == multimodal::MmProcessing::Worker {
            debug!(
                function = stage,
                media_items = media_plan.parts().len(),
                "Forwarding media references for worker-side processing"
            );
            // Which of the workers that process references may take the
            // plan (see `resolve_mm_processing` and `PipelinePin`).
            ctx.state.media_refs_pipeline =
                multimodal::media_refs_pipeline_pin(&media_plan, &placeholders);
            multimodal_refs = Some(media_plan);
        } else {
            // The dtype every worker the selection can pick reads image
            // pixels in, when they agree (and the one-pass write is not
            // switched off, `SMG_MM_ONE_PASS_WIRE=0`): the preprocessor
            // then writes it directly; otherwise `None`, and the write
            // happens at assembly. Resolved over the pools of this
            // pipeline's mode, the pools the pixels can reach, as the
            // selection draws them.
            let image_wire_dtype = multimodal::mm_shared_image_wire_dtype_over(
                &ctx.components.worker_registry,
                model_id,
                wire_pools,
            );
            // The frames decode under the destination engine's own
            // transparency rule, resolved over the same pools as the dtype.
            let transparency = multimodal::engine_transparency_policy(
                &ctx.components.worker_registry,
                model_id,
                wire_pools,
            );
            // The engine's per-item bound, the smallest its workers
            // advertise: an item above it would never be scheduled.
            let encoder_budget =
                multimodal::engine_encoder_budget(&ctx.components.worker_registry, model_id);
            match multimodal::process_multimodal_plan(
                media_plan,
                model_id,
                &**tokenizer,
                token_ids,
                &mm_components,
                &tokenizer_id,
                &tokenizer_source,
                image_wire_dtype,
                transparency,
                encoder_budget,
            )
            .await
            {
                Ok(output) => {
                    debug!(
                        function = stage,
                        expanded_tokens = output.expanded_token_ids.len(),
                        "Multimodal processing complete"
                    );
                    token_ids = output.expanded_token_ids;
                    multimodal_intermediate = Some(output.intermediate);
                }
                Err(e) => {
                    error!(
                        function = stage,
                        error = %multimodal::processing_failed_message(&e),
                        "Multimodal processing failed"
                    );
                    return Err(multimodal::processing_failed_response(&e));
                }
            }
        }
    }

    Ok(RenderedPrompt {
        token_ids,
        multimodal_intermediate,
        multimodal_refs,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_tokenizer::{
        mock::MockTokenizer,
        traits::{PromptEncoding, Tokenizer},
        TokenizerRegistry,
    };
    use openai_protocol::chat::ChatCompletionRequest;
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::prepare_rendered_prompt;
    use crate::{
        routers::{
            common::placement::PipelinePin,
            grpc::{
                context::{RequestContext, SharedComponents},
                regular::symphony::template::Templates,
                utils::ParserResolver,
            },
        },
        worker::WorkerRegistry,
    };

    /// A prompt without media is the renderer's tokens and nothing else: no
    /// intermediate, no references, no pipeline pin.
    #[tokio::test]
    async fn a_prompt_without_media_is_the_renderers_tokens_and_nothing_else() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let loaded = tokenizer.clone();
        tokenizer_registry
            .load("tokenizer-id", "m", "test", || async move { Ok(loaded) })
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
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "Hello world"}]
        }))
        .unwrap();
        let mut ctx = RequestContext::for_chat(Arc::new(request), None, "m".into(), components);

        let Ok(rendered) = prepare_rendered_prompt(
            &mut ctx,
            &tokenizer,
            "Hello world",
            PromptEncoding::FromText,
            None,
            &[],
            "test",
        )
        .await
        else {
            panic!("a prompt without media is prepared");
        };

        let expected = tokenizer.encode("Hello world", false).unwrap();
        assert_eq!(rendered.token_ids, expected.token_ids());
        assert_eq!(rendered.token_ids.len(), 2);
        assert!(rendered.multimodal_intermediate.is_none());
        assert!(rendered.multimodal_refs.is_none());
        assert_eq!(ctx.state.media_refs_pipeline, PipelinePin::Any);
    }
}
