//! What the chat and Messages preparation stages do alike once the template
//! has rendered: tokenize the prompt the way its renderer said to, check the
//! rendered media anchors against the plan, and either process the media
//! (fetch, preprocess, expand the tokens, hash) or keep the references for a
//! worker that processes them itself. The two stages render differently and
//! continue from the rendered text through one function.

use std::sync::Arc;

use axum::response::Response;
use llm_multimodal::Modality;
use llm_tokenizer::traits::{PromptEncoding, Tokenizer};
use openai_protocol::worker::MmProcessingMode;
use tracing::{debug, error, warn};

use crate::routers::{
    common::snapshot::RoutingPool,
    error,
    grpc::{
        context::{EncodeOutputs, RequestContext},
        multimodal::{self, MediaContext, MediaPlan, MultimodalIntermediate},
        utils,
    },
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

    let mut token_ids = encoding.into_token_ids();

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

    if matches!(
        ctx.state.encode_outputs,
        Some(EncodeOutputs::PreparedInput(_))
    ) {
        return Ok(RenderedPrompt {
            token_ids,
            multimodal_intermediate: None,
            multimodal_refs: media.map(|media| media.plan),
        });
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
            if media_plan.modalities() == [Modality::Image] {
                ctx.state.image_anchor = placeholders
                    .get(Modality::Image)
                    .and_then(|anchor| tokenizer.token_to_id(anchor));
            }
            // Whether the language model reads the grid tensors decides if a
            // language-model-only decode worker can take the stripped side;
            // the spec answered in placeholder preparation, and worker
            // selection reads it off the plan.
            let mut media_plan = media_plan;
            media_plan.set_language_model_reads_grids(placeholders.language_model_reads_grids());
            multimodal_refs = Some(media_plan);
        } else {
            // Preparation asked the matrix with a worker that processes
            // references itself counted as taking every modality, which
            // holds only while a reference can travel to it. This request
            // resolved to the gateway's own pipeline under a placement that
            // forwards where it can (`auto`, or a hinted request under
            // `worker`), so none travels: the matrix alone answers, before
            // any byte is fetched, with the refusal worker selection gave
            // after the decode. Under `router` preparation asked it so.
            if mm_components.processing != MmProcessingMode::Router {
                multimodal::ensure_model_accepts_media(
                    &ctx.components.worker_registry,
                    model_id,
                    &media_plan,
                    true,
                    mm_components.pixels_go_to,
                )?;
            }
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
            // The byte bound the pixels must fit before any image is decoded,
            // where every worker the selection can pick takes them inline in
            // the gateway's own Generate message; none where any may take
            // them beside it (an SHM handle, the RDMA lane, a TRT-LLM
            // worker's file bytes, a ZMQ wire), and the framing itself answers.
            let image_wire_cap = multimodal::mm_image_wire_cap_over(
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
                image_wire_cap,
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
                    let response = multimodal::processing_failed_response(&e);
                    // The request's own media, by an explicit verdict, is the
                    // client's to fix: one WARN with the cause, the answer
                    // marked reported. Every other failure keeps ERROR and
                    // stays unmarked: the media host's, this process's, and
                    // one with no verdict, whose 400 also covers a model
                    // configuration this host could not load.
                    if multimodal::is_request_media_fault(&e) {
                        warn!(
                            function = stage,
                            error = %format!("{e:#}"),
                            "Multimodal processing failed"
                        );
                        return Err(utils::reported(response));
                    }
                    error!(
                        function = stage,
                        error = %format!("{e:#}"),
                        "Multimodal processing failed"
                    );
                    return Err(response);
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
#[path = "../../../../../tests/unit/routers/grpc/regular/stages/rendered_prompt/tests.rs"]
mod tests;
