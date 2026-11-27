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

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new().with_content_format(format));
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
    let Err(refused) = resolve_image_without_a_spec(ChatTemplateContentFormat::String).await else {
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

    let Ok(served) = resolve_image_without_a_spec(ChatTemplateContentFormat::OpenAI).await else {
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
