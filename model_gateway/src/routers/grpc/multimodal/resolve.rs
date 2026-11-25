//! Media resolution: what a chat-like request's media needs before its
//! template renders, resolved from the registries.
//!
//! [`resolve_media`] reads the [`MediaPlan`] its caller builds from the
//! request and returns the model's media-part order and, for a request with
//! media, the plan with the placeholder tokens the template writes for it.
//! The chat and Messages preparation stages and count_tokens resolve through
//! it, so the three refuse a request alike. A generation request's media is
//! counted here, before any refusal or limit; a token count's is not.
//! Nothing is fetched or preprocessed here.

use std::sync::Arc;

use axum::response::Response;
use llm_multimodal::{MediaPartOrder, Modality};
use llm_tokenizer::{
    chat_template::ChatTemplateContentFormat,
    traits::{MediaHandling, Tokenizer},
};
use openai_protocol::worker::MmProcessingMode;
use smg_worker::identity::basic::WorkerType;
use tracing::{error, warn};

use crate::{
    observability::metrics::Metrics,
    routers::{
        error,
        grpc::{
            context::SharedComponents,
            multimodal::{
                self, invalid_multimodal_request, MediaPlan, MultimodalComponents,
                PlaceholderTokens,
            },
            utils::reported,
        },
    },
};

/// The media context of a chat-like request (chat or Messages), resolved
/// from the registries before the template renders, as the preparation
/// stages resolve it: nothing is fetched or preprocessed here.
pub(crate) struct ResolvedMedia {
    /// Where the model's template wants media parts relative to text; the
    /// default (`MediaFirst`) for a request without media, which no renderer
    /// acts on: with no media part there is nothing to place.
    pub(crate) media_order: MediaPartOrder,
    /// The request's media and the placeholders the template writes for it,
    /// when the request carries media.
    pub(crate) media: Option<MediaContext>,
}

impl ResolvedMedia {
    pub(crate) fn placeholders(&self) -> Option<&PlaceholderTokens> {
        self.media.as_ref().map(|media| &media.placeholders)
    }
}

/// A request's media as the preparation stages process it after rendering
/// (`regular::stages::rendered_prompt`).
pub(crate) struct MediaContext {
    pub(crate) plan: MediaPlan,
    pub(crate) placeholders: PlaceholderTokens,
    pub(crate) components: Arc<MultimodalComponents>,
    pub(crate) tokenizer_id: String,
    pub(crate) tokenizer_source: String,
}

/// Resolve what a chat-like request's media needs before the template
/// renders: for a request with media, the model's media-part order and the
/// plan `plan` builds from the request and the placeholder tokens (the
/// engine's own limits applied); a request without media returns before the
/// registries are read, with the default order and no plan. A renderer that writes media as template
/// text has no plan and keeps the authored order, its request's media
/// recorded all the same. The chat and Messages
/// preparation stages and count_tokens resolve through here, so the three
/// refuse a request alike; `stage` names the caller in the log lines, as
/// each caller's own lines named it before. `has_tools` picks the template
/// the request renders through, as its renderer picks it: media of a model
/// without a media spec is refused here when that template takes one string
/// per message.
pub(crate) async fn resolve_media(
    components: &SharedComponents,
    model_id: &str,
    tokenizer: &dyn Tokenizer,
    stage: &'static str,
    plan: impl FnOnce() -> MediaPlan,
    record_media: bool,
    has_tools: bool,
) -> Result<ResolvedMedia, Response> {
    // What a generation request carries, counted once and before any
    // refusal or limit (a refused request's media included); a token count
    // resolves the same plan and is not a request here.
    let record = |plan: &MediaPlan| {
        for &modality in plan.modalities() {
            Metrics::record_mm_request_media(
                model_id,
                modality,
                plan.count(modality),
                &plan.source_counts(modality),
            );
        }
        // The request's images by the source class of its media, once per
        // request, one without media included (0 under `none`). The class
        // is read over all the request's media, not its images alone.
        Metrics::record_mm_request_images(
            model_id,
            plan.source_class(),
            plan.count(Modality::Image),
        );
    };
    if tokenizer.renderer_capabilities().media_handling == MediaHandling::TemplateText {
        // A template that writes media as text takes no plan; a request
        // that is recorded builds one for the count alone, a token count
        // builds none.
        if record_media {
            record(&plan());
        }
        return Ok(ResolvedMedia {
            media_order: MediaPartOrder::Authored,
            media: None,
        });
    }
    let plan = plan();
    if record_media {
        record(&plan);
    }
    // No images or video to a pool whose every worker is language model only (the engine admits
    // such a request and never schedules it), and no modality to a pool whose engines take none of
    // it (the capability matrix): both are refused here, for the chat and the Messages paths alike,
    // on the declared parts, before any byte of media is fetched, decoded or sent.
    let router_processes = components
        .multimodal
        .as_ref()
        .is_some_and(|mm_components| mm_components.processing == MmProcessingMode::Router);
    // The first rule judges the workers the pixels go to in this router's mode.
    let pixels_go_to = components
        .multimodal
        .as_ref()
        .map_or(WorkerType::Regular, |mm_components| {
            mm_components.pixels_go_to
        });
    multimodal::ensure_model_accepts_media(
        &components.worker_registry,
        model_id,
        &plan,
        router_processes,
        pixels_go_to,
    )?;
    if plan.is_empty() {
        // A request without media renders the same under either order (its text parts in
        // their own order): the registries are not asked for a model with nothing to place.
        return Ok(ResolvedMedia {
            media_order: MediaPartOrder::MediaFirst,
            media: None,
        });
    }
    // The tokenizer's registry entry (its id and source) is read for the
    // media paths alone, which need the multimodal components; a gateway
    // without them looks nothing up.
    let Some(mm_components) = components.multimodal.as_ref() else {
        error!(
            function = stage,
            "Multimodal content detected but multimodal components not initialized"
        );
        return Err(error::bad_request(
            "multimodal_not_supported",
            "Multimodal content detected but multimodal processing is not available",
        ));
    };
    // Bound straight from the lookup: a named `Option` moved from here would
    // keep its slot in this future beside `entry` across both awaits below.
    let Some(entry) = components
        .tokenizer_registry
        .get_by_name(model_id)
        .or_else(|| components.tokenizer_registry.get_by_id(model_id))
    else {
        error!(
            function = stage,
            model = %model_id,
            "Tokenizer entry not found for multimodal processing"
        );
        return Err(error::bad_request(
            "multimodal_config_missing",
            format!("Tokenizer not found for model: {model_id}"),
        ));
    };

    // The per-request media limits the model's workers advertise for their
    // engine, which the plan is held to before any fetch. They live for the
    // preparation only, not across the order lookup after it: this future is
    // inlined into the route and stage futures boxed once per request, and its
    // largest state is the one that holds the prepared placeholders.
    let placeholders = {
        let engine_limits = multimodal::engine_item_limits(&components.worker_registry, model_id);
        multimodal::prepare_placeholder_tokens(
            &plan,
            model_id,
            tokenizer,
            mm_components,
            &entry.id,
            &entry.source,
            &engine_limits,
            |format| multimodal::fleet_reads_format(&components.worker_registry, model_id, format),
        )
        .await
    }
    .map_err(|e| {
        // Only a refusal the plan types as the request's own is the client's
        // to fix: one WARN with the cause, the answer marked reported.
        if multimodal::RequestRefusal::is(&e) {
            warn!(
                function = stage,
                model = %model_id,
                error = %format!("{e:#}"),
                "Failed to prepare the multimodal prompt plan"
            );
            return reported(invalid_multimodal_request(e));
        }
        // Every other failure keeps the ERROR line and an unmarked answer.
        // The chain, not the outermost context alone: the cause (the file,
        // the token, the hub's answer and its address) lives here, and the
        // answers of the server's class below are generic.
        error!(
            function = stage,
            model = %model_id,
            error = %format!("{e:#}"),
            "Failed to prepare the multimodal prompt plan"
        );
        // A hub that failed to serve the configuration answers for itself,
        // and a failure its site marks as the deployment's own is answered
        // in the server's class. Everything else keeps its existing 400,
        // including unclassified failures while resolving the configuration
        // directory: an absent or inaccessible configuration, or local
        // client or cache setup failures.
        multimodal::config_unavailable_response(&e).unwrap_or_else(|| {
            if multimodal::DeploymentFault::is(&e) {
                multimodal::preparation_fault_response()
            } else {
                invalid_multimodal_request(e)
            }
        })
    })?;
    // A model without a media spec is served where its template takes
    // content parts and writes the engine's own placeholders. A template
    // that takes one string per message leaves the anchors to the gateway,
    // which knows none for this model: rendered, the prompt would carry no
    // anchor for media the worker is still handed.
    if placeholders.is_worker_only()
        && tokenizer.chat_template_content_format_for_request(has_tools)
            == ChatTemplateContentFormat::String
    {
        let modalities = plan
            .modalities()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(error::bad_request(
            "multimodal_not_supported",
            format!(
                "{modalities} input is not supported by model '{model_id}': its chat template \
                 takes text content only and the gateway writes no media placeholder for this \
                 model"
            ),
        ));
    }
    // The media-part order comes from the model registry so every path
    // renders a model consistently. It is looked up after the placeholders,
    // whose preparation resolved the model's configs and cached them, so the
    // lookup is a cache hit and a request resolves the checkpoint once: looked
    // up first, it would resolve on its own, fall back to media-first on a
    // failure it does not report, and leave the preparation to resolve again.
    let media_order = multimodal::resolve_media_part_order(
        model_id,
        tokenizer,
        mm_components,
        &entry.id,
        &entry.source,
    )
    .await;

    Ok(ResolvedMedia {
        media_order,
        media: Some(MediaContext {
            plan,
            placeholders,
            components: Arc::clone(mm_components),
            tokenizer_id: entry.id,
            tokenizer_source: entry.source,
        }),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    /// The media resolution of one image for model "m", which has no media
    /// spec, where a worker would process the media, through a template of
    /// `format`.
    async fn resolve_image_without_a_spec(
        format: ChatTemplateContentFormat,
    ) -> Result<ResolvedMedia, Response> {
        use llm_tokenizer::mock::MockTokenizer;

        use crate::routers::grpc::regular::stages::testing::components_forwarding_media_of;

        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(MockTokenizer::new().with_content_format(format));
        let (_dir, components) = components_forwarding_media_of("m", Arc::clone(&tokenizer)).await;
        resolve_media(
            &components,
            "m",
            &*tokenizer,
            "test",
            || {
                MediaPlan::new([llm_multimodal::MediaContentPart::ImageUrl {
                    url: "https://a/1.png".to_string().into(),
                    detail: None,
                    uuid: None,
                    max_long_side_pixel: None,
                }])
            },
            false,
            false,
        )
        .await
    }

    /// Media of a model without a media spec is refused before rendering
    /// when its template takes one string per message, with the capability
    /// code and a sentence naming the modality and the model: neither the
    /// template nor the gateway would write an anchor for it. A template
    /// that takes content parts writes the engine's own placeholders, and
    /// the request goes on to a worker that processes its media.
    #[tokio::test]
    async fn media_nothing_would_anchor_is_refused_before_rendering() {
        let Err(refused) = resolve_image_without_a_spec(ChatTemplateContentFormat::String).await
        else {
            panic!("an image that nothing would anchor in the prompt is refused");
        };
        assert_eq!(refused.status(), http::StatusCode::BAD_REQUEST);
        assert_eq!(
            refused
                .headers()
                .get("x-smg-error-code")
                .and_then(|code| code.to_str().ok()),
            Some("multimodal_not_supported")
        );
        let body = axum::body::to_bytes(refused.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body["error"]["message"],
            "image input is not supported by model 'm': its chat template takes text content \
             only and the gateway writes no media placeholder for this model"
        );

        let Ok(served) = resolve_image_without_a_spec(ChatTemplateContentFormat::OpenAI).await
        else {
            panic!("a template that takes content parts writes the anchor itself");
        };
        assert!(served
            .placeholders()
            .is_some_and(PlaceholderTokens::is_worker_only));
    }

    /// A generation request without media is observed once in
    /// `smg_mm_request_images`, with no images under `none`, ahead of the
    /// return that reads no registry for it. A token count resolves the
    /// same plan and is observed nowhere.
    #[test]
    fn a_request_without_media_is_observed_with_no_images_and_a_token_count_is_not() {
        use llm_tokenizer::mock::MockTokenizer;

        use crate::{
            observability::metrics::test_support::{assert_metric, render_with_recorder},
            routers::grpc::regular::stages::testing::components_forwarding_media_of,
        };

        let scrape_after_resolving = |record_media: bool| {
            render_with_recorder(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async {
                        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
                        let (_dir, components) =
                            components_forwarding_media_of("m", Arc::clone(&tokenizer)).await;
                        let resolved = resolve_media(
                            &components,
                            "m",
                            &*tokenizer,
                            "test",
                            || MediaPlan::new([]),
                            record_media,
                            false,
                        )
                        .await;
                        assert!(resolved.is_ok_and(|resolved| resolved.media.is_none()));
                    });
            })
        };

        let request = scrape_after_resolving(true);
        let series = ["model=\"m\"", "source_class=\"none\""];
        assert_metric(&request, "smg_mm_request_images_count", &series, "1");
        assert_metric(&request, "smg_mm_request_images_sum", &series, "0");

        let token_count = scrape_after_resolving(false);
        assert!(
            !token_count.contains("smg_mm_request_images"),
            "a token count is not a request the family observes:\n{token_count}"
        );
    }
}
