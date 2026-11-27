use std::sync::Arc;

use axum::{http::StatusCode, response::Response};
use llm_multimodal::{MediaContentPart, Modality};
use llm_tokenizer::{
    mock::MockTokenizer,
    traits::{PromptEncoding, Tokenizer},
    TokenizerRegistry,
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    model_card::ModelCard,
    worker::{MmProcessingMode, RuntimeType, WorkerType},
};
use smg_worker::{
    identity::basic::{ConnectionMode, Worker},
    test_support::no_health_check,
};
use tracing::Level;

use super::prepare_rendered_prompt;
use crate::{
    middleware::ReportedRefusal,
    observability::logging::test_support::{test_subscriber, LinesAtWarnOrAbove},
    routers::{
        common::{
            placement::PipelinePin,
            snapshot::{
                test_support::{grpc_worker, registry_with},
                WorkerRegistry,
            },
        },
        grpc::{
            backend_client::BasicWorkerBuilder,
            context::RequestContext,
            multimodal::{
                ensure_model_accepts_media, media_refs_pipeline_pin, MediaContext, MediaPlan,
                MultimodalComponents, MultimodalConfigRegistry, MultimodalSettings,
                PlaceholderTokens, GATEWAY_PIPELINE_PROCESSOR, MM_ITEM_HINTS_LABEL,
                MM_PROCESSOR_LABEL,
            },
            regular::stages::testing::components_over,
            testing::unreadable_model_dir,
        },
    },
};

/// A request context for model "m" over the mock tokenizer, and the
/// tokenizer.
async fn context() -> (RequestContext, Arc<dyn Tokenizer>) {
    context_over(WorkerRegistry::new()).await
}

/// The same over `registry`, whose workers serve model "m".
async fn context_over(registry: WorkerRegistry) -> (RequestContext, Arc<dyn Tokenizer>) {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let loaded = tokenizer.clone();
    tokenizer_registry
        .load("tokenizer-id", "m", "test", || async move { Ok(loaded) })
        .await
        .unwrap();
    let components = components_over(
        tokenizer_registry,
        Arc::new(registry),
        crate::config::ParserChoice::Legacy,
    );
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "Hello world"}]
    }))
    .unwrap();
    (
        RequestContext::for_chat(Arc::new(request), None, "m".into(), components),
        tokenizer,
    )
}

/// A media context for one image, under multimodal components in `mode`.
fn media(mode: MmProcessingMode, placeholders: PlaceholderTokens) -> MediaContext {
    media_for(
        MediaPlan::new([MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }]),
        mode,
        placeholders,
    )
}

/// A media context for `plan`, under multimodal components in `mode`.
fn media_for(
    plan: MediaPlan,
    mode: MmProcessingMode,
    placeholders: PlaceholderTokens,
) -> MediaContext {
    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.processing = mode;
    MediaContext {
        plan,
        placeholders,
        components: Arc::new(components),
        tokenizer_id: "tokenizer-id".to_string(),
        tokenizer_source: "test".to_string(),
    }
}

/// A rendered prompt whose anchors do not match the plan is refused by
/// name before any media is fetched: one image planned, no anchor
/// token in the prompt.
#[tokio::test]
async fn a_prompt_whose_anchors_miss_the_plan_is_refused_by_name() {
    let (mut ctx, tokenizer) = context().await;
    let mut placeholders = PlaceholderTokens::default();
    placeholders.insert(Modality::Image, "token".to_string());
    let media = media(MmProcessingMode::Router, placeholders);

    let Err(response) = prepare_rendered_prompt(
        &mut ctx,
        &tokenizer,
        "Hello world",
        PromptEncoding::FromText,
        Some(media),
        &[],
        "test",
    )
    .await
    else {
        panic!("a prompt without its anchor is refused");
    };
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get("x-smg-error-code")
            .and_then(|v| v.to_str().ok()),
        Some("multimodal_prompt_contract_mismatch")
    );
}

/// A processing failure with no verdict that the request's media is at
/// fault (here, a model whose `config.json` does not parse) keeps the
/// stage's ERROR line and stays unmarked, though its answer is a 400.
#[tokio::test]
async fn a_processing_failure_with_no_client_verdict_keeps_the_error_line() {
    // The media context names where the model's configuration is looked
    // for: a local directory, alive to the end of the test, so the load
    // fails on this host, with no fetch.
    let model_dir = unreadable_model_dir();
    let (mut ctx, tokenizer) = context().await;
    let mut placeholders = PlaceholderTokens::default();
    placeholders.insert(Modality::Image, "token".to_string());
    let mut media = media(MmProcessingMode::Router, placeholders);
    media.tokenizer_source = model_dir.path().to_str().unwrap().to_string();
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let Err(response) = prepare_rendered_prompt(
        &mut ctx,
        &tokenizer,
        "Hello token",
        PromptEncoding::FromText,
        Some(media),
        &[],
        "test",
    )
    .await
    else {
        panic!("media of a model without a media config is refused");
    };

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(response.extensions().get::<ReportedRefusal>().is_none());
    assert_eq!(
        lines.lines(),
        [(Level::ERROR, "Multimodal processing failed".to_string())]
    );
}

/// Media a worker processes itself leaves as references: the tokens are
/// the renderer's, no intermediate is built, and the pipeline pin the
/// worker selection reads is set from the plan. The plan is an audio
/// reference, whose pin is not the state's default, so a pin left
/// unwritten fails the test.
#[tokio::test]
async fn media_a_worker_processes_itself_leaves_as_references_with_the_pin_set() {
    let (mut ctx, tokenizer) = context().await;
    let placeholders = PlaceholderTokens::worker_only();
    let media = media_for(
        MediaPlan::new([MediaContentPart::AudioUrl {
            url: "https://a/1.wav".to_string().into(),
            uuid: None,
        }]),
        MmProcessingMode::Worker,
        placeholders.clone(),
    );
    let expected_pin = media_refs_pipeline_pin(&media.plan, &placeholders);
    assert_ne!(
        expected_pin,
        PipelinePin::default(),
        "the plan must pin a pipeline the default does not"
    );

    let Ok(rendered) = prepare_rendered_prompt(
        &mut ctx,
        &tokenizer,
        "Hello world",
        PromptEncoding::FromText,
        Some(media),
        &[],
        "test",
    )
    .await
    else {
        panic!("worker-side media is prepared as references");
    };
    let expected = tokenizer.encode("Hello world", false).unwrap();
    assert_eq!(rendered.token_ids, expected.token_ids());
    assert!(rendered.multimodal_intermediate.is_none());
    assert_eq!(
        rendered
            .multimodal_refs
            .as_ref()
            .map(|plan| plan.parts().len()),
        Some(1)
    );
    assert_eq!(ctx.state.media_refs_pipeline, expected_pin);
}

/// A hinted image forwarded to a fleet whose workers apply the hints leaves
/// with the hint demand set, the one worker selection and the dispatch's
/// wire read for every attempt; the same image without hints leaves it
/// unset.
#[tokio::test]
async fn hinted_media_forwarded_to_workers_that_apply_hints_sets_the_hint_demand() {
    let image = |max_long_side_pixel| {
        MediaPlan::new([MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel,
        }])
    };
    let placeholders = PlaceholderTokens::worker_only();
    for (name, plan, hinted) in [
        ("a hinted image", image(Some(504)), true),
        ("an image without hints", image(None), false),
    ] {
        let (mut ctx, tokenizer) = context_over(registry_with(vec![grpc_worker(
            "grpc://127.0.0.1:9830",
            "m",
            RuntimeType::Vllm,
            &[
                (MM_PROCESSOR_LABEL, GATEWAY_PIPELINE_PROCESSOR),
                (MM_ITEM_HINTS_LABEL, "true"),
            ],
        )]))
        .await;
        let media = media_for(plan, MmProcessingMode::Worker, placeholders.clone());
        let rendered = match prepare_rendered_prompt(
            &mut ctx,
            &tokenizer,
            "Hello world",
            PromptEncoding::FromText,
            Some(media),
            &[],
            "test",
        )
        .await
        {
            Ok(rendered) => rendered,
            Err(response) => panic!("{name}: refused with {}", response.status()),
        };
        assert!(rendered.multimodal_refs.is_some(), "{name}");
        assert_eq!(ctx.state.media_refs_hints, hinted, "{name}");
    }
}

/// A prompt without media is the renderer's tokens and nothing else: no
/// intermediate, no references, no pipeline pin.
#[tokio::test]
async fn a_prompt_without_media_is_the_renderers_tokens_and_nothing_else() {
    let (mut ctx, tokenizer) = context().await;

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

/// The code and the message of a 400.
async fn refusal(response: Response) -> (String, String) {
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let code = response
        .headers()
        .get("x-smg-error-code")
        .and_then(|v| v.to_str().ok())
        .expect("code header")
        .to_string();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
    let message = body["error"]["message"]
        .as_str()
        .expect("message")
        .to_string();
    (code, message)
}

/// Media the gateway processes itself under a placement that forwards
/// references where it can is refused for a modality no engine of the
/// pool takes before any byte is fetched, with the matrix's sentence:
/// the refusal worker selection gave after the fetch and the decode.
/// One row for each way the placement resolves to the gateway. Every
/// pool holds a worker that processes references itself, so
/// preparation's check passes. The media is inline and does not decode,
/// and the model's configuration does not load, so a request that
/// reached the pipeline would fail there, with another code and no fetch.
/// Two controls after the rows do: media an engine of the pool takes
/// passes the check and fails in the pipeline, under the pipeline's code.
#[tokio::test]
async fn media_the_gateway_resolves_to_process_is_refused_before_the_fetch() {
    fn refs_worker(url: &str) -> Arc<dyn Worker> {
        grpc_worker(
            url,
            "m",
            RuntimeType::Sglang,
            &[(MM_PROCESSOR_LABEL, GATEWAY_PIPELINE_PROCESSOR)],
        )
    }
    fn audio() -> MediaPlan {
        MediaPlan::new([MediaContentPart::AudioBase64 {
            media_type: "audio/wav".to_string(),
            data: Arc::new("AAAA".to_string()),
            uuid: None,
        }])
    }
    fn image() -> MediaPlan {
        MediaPlan::new([MediaContentPart::ImageBase64 {
            media_type: "image/png".to_string(),
            data: Arc::new("AAAA".to_string()),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }])
    }
    fn hinted_video() -> MediaPlan {
        MediaPlan::new([MediaContentPart::VideoUrl {
            url: "data:video/mp4;base64,AAAA".to_string().into(),
            uuid: None,
            fps: Some(1.0),
            max_long_side_pixel: None,
        }])
    }
    fn anchored(modality: Modality) -> PlaceholderTokens {
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(modality, "token".to_string());
        placeholders
    }
    // An anchor the worker expands itself: the model's own opt-in.
    fn expandable(modality: Modality) -> PlaceholderTokens {
        let mut placeholders = anchored(modality);
        placeholders.set_worker_expandable(modality, true);
        placeholders
    }
    const NO_AUDIO: &str = "backend sglang does not support audio inputs";
    const NO_VIDEO: &str = "backend sglang does not support video inputs";
    let encode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:9813")
            .model(ModelCard::new("m"))
            .worker_type(WorkerType::Encode)
            .runtime_type(RuntimeType::Sglang)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    let model_dir = unreadable_model_dir();

    for (name, mode, workers, plan, placeholders, text, sentence) in [
        (
            "auto, a pool where one worker processes references and one does not",
            MmProcessingMode::Auto,
            vec![
                refs_worker("grpc://127.0.0.1:9810"),
                grpc_worker("grpc://127.0.0.1:9811", "m", RuntimeType::Sglang, &[]),
            ],
            audio(),
            expandable(Modality::Audio),
            "Hello token",
            NO_AUDIO,
        ),
        (
            "auto, an encode-prefill-decode pool",
            MmProcessingMode::Auto,
            vec![refs_worker("grpc://127.0.0.1:9812"), encode],
            audio(),
            expandable(Modality::Audio),
            "Hello token",
            NO_AUDIO,
        ),
        (
            "auto, a model whose anchor the worker does not expand",
            MmProcessingMode::Auto,
            vec![refs_worker("grpc://127.0.0.1:9814")],
            audio(),
            anchored(Modality::Audio),
            "Hello token",
            NO_AUDIO,
        ),
        (
            "auto, a hinted item",
            MmProcessingMode::Auto,
            vec![refs_worker("grpc://127.0.0.1:9815")],
            hinted_video(),
            anchored(Modality::Video),
            "Hello token",
            NO_VIDEO,
        ),
        (
            "worker, a hinted item the gateway's pipeline serves",
            MmProcessingMode::Worker,
            vec![refs_worker("grpc://127.0.0.1:9816")],
            hinted_video(),
            anchored(Modality::Video),
            "Hello token",
            NO_VIDEO,
        ),
    ] {
        let registry = registry_with(workers);
        assert!(
            ensure_model_accepts_media(&registry, "m", &plan, false, WorkerType::Regular).is_ok(),
            "{name}: preparation's check passes"
        );
        let (mut ctx, tokenizer) = context_over(registry).await;
        let mut media = media_for(plan, mode, placeholders);
        media.tokenizer_source = model_dir.path().to_str().unwrap().to_string();

        let Err(response) = prepare_rendered_prompt(
            &mut ctx,
            &tokenizer,
            text,
            PromptEncoding::FromText,
            Some(media),
            &[],
            "test",
        )
        .await
        else {
            panic!("{name}: the media is refused");
        };
        let (code, message) = refusal(response).await;
        assert_eq!(code, "multimodal_not_supported", "{name}");
        assert_eq!(message, sentence, "{name}");
    }

    // The controls: the check refuses only what no engine of the pool
    // takes. Each request resolves to the gateway as the first row's
    // does, passes the check, reaches the pipeline and fails there on
    // the model's configuration.
    for (name, workers, plan, placeholders) in [
        (
            "the first row's pool and an image, which its engine takes",
            vec![
                refs_worker("grpc://127.0.0.1:9817"),
                grpc_worker("grpc://127.0.0.1:9818", "m", RuntimeType::Sglang, &[]),
            ],
            image(),
            expandable(Modality::Image),
        ),
        (
            "the first row's request over a pool where one engine takes audio",
            vec![
                refs_worker("grpc://127.0.0.1:9819"),
                grpc_worker("grpc://127.0.0.1:9820", "m", RuntimeType::TokenSpeed, &[]),
            ],
            audio(),
            expandable(Modality::Audio),
        ),
    ] {
        let (mut ctx, tokenizer) = context_over(registry_with(workers)).await;
        let mut media = media_for(plan, MmProcessingMode::Auto, placeholders);
        media.tokenizer_source = model_dir.path().to_str().unwrap().to_string();

        let Err(response) = prepare_rendered_prompt(
            &mut ctx,
            &tokenizer,
            "Hello token",
            PromptEncoding::FromText,
            Some(media),
            &[],
            "test",
        )
        .await
        else {
            panic!("{name}: the pipeline fails on the model's configuration");
        };
        let (code, _) = refusal(response).await;
        assert_eq!(code, "multimodal_processing_failed", "{name}");
    }
}
