//! The generate route over the gRPC pipeline refuses a request that names
//! media, as the chat route refuses media no worker takes: the pipeline
//! carries text and token ids to the engine and nothing else, so a body with
//! `image_data`, `video_data` or `audio_data` is answered 400
//! `multimodal_not_supported` instead of being served without its media. A
//! text-only body is served as before.
#![expect(clippy::unwrap_used, reason = "test failures should panic")]

use std::{sync::Arc, time::Duration};

use axum::http::StatusCode;
use common::{
    grpc_harness::{
        build_grpc_router, grpc_router_config, grpc_worker, serve_canned_worker, Canned,
    },
    test_app::read_body,
};
use openai_protocol::{generate::GenerateRequest, model_card::ModelCard};
use serde_json::{json, Value};
use smg::{
    config::RoutingMode,
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};

use crate::common;

const MODEL: &str = "generate-media-test-model";

/// A regular gRPC router over one canned worker of `MODEL`.
async fn router() -> Box<dyn RouterTrait> {
    let port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 2,
    })
    .await;
    let worker = Arc::new(
        grpc_worker(
            port,
            WorkerType::Regular,
            RuntimeType::TokenSpeed,
            ModelCard::new(MODEL),
        )
        .build(),
    );
    let config = grpc_router_config()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .random_policy()
        .build_unchecked();
    build_grpc_router(config, MODEL, vec![worker]).await
}

/// A text generate request of `MODEL` with `extra` merged into its body.
fn request(extra: &Value) -> GenerateRequest {
    let mut body = json!({
        "model": MODEL,
        "text": "Hello Hello",
        "sampling_params": {"max_new_tokens": 2}
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("generate-media-test-tenant"))
}

/// Each media field, alone or beside another, is refused before any worker
/// sees the request, under the chat route's code.
#[tokio::test]
async fn a_generate_request_with_media_is_refused_as_the_chat_route_refuses_it() {
    let router = router().await;
    for media in [
        json!({"image_data": "data:image/png;base64,iVBORw0KGgo="}),
        json!({"video_data": ["clip.mp4"]}),
        json!({"audio_data": "data:audio/wav;base64,UklGRg=="}),
        json!({"image_data": ["a.png"], "audio_data": "b.wav"}),
    ] {
        let response = router
            .route_generate(None, &tenant(), request(&media), MODEL)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{media}");
        let body: Value = serde_json::from_str(&read_body(response).await).unwrap();
        assert_eq!(
            body["error"]["code"], "multimodal_not_supported",
            "{media}: {body}"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("generate route")),
            "{media}: {body}"
        );
    }
}

/// A text-only body is served as before, a media field set to null, to an
/// empty list or to a list of nulls and empty lists naming no media.
#[tokio::test]
async fn a_text_only_generate_request_is_served_as_before() {
    let router = router().await;
    for extra in [
        json!({}),
        json!({"image_data": null}),
        json!({"image_data": [], "video_data": []}),
        json!({"image_data": [null]}),
        json!({"video_data": [[]]}),
    ] {
        let response = router
            .route_generate(None, &tenant(), request(&extra), MODEL)
            .await;
        assert_eq!(response.status(), StatusCode::OK, "{extra}");
    }
}
