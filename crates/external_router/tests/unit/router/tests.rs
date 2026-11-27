use super::*;

#[derive(Debug)]
struct StubRouter;

#[async_trait]
impl ExternalRouter for StubRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn router_type(&self) -> &'static str {
        "stub"
    }
}

/// The status of an answer and the code and message of its JSON error object.
async fn error_shape(answer: Response) -> (http::StatusCode, String, String) {
    let status = answer.status();
    let body = axum::body::to_bytes(answer.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
    let field = |name: &str| {
        body["error"][name]
            .as_str()
            .unwrap_or_else(|| panic!("error.{name} in {body}"))
            .to_owned()
    };
    (status, field("code"), field("message"))
}

fn body<T: serde::de::DeserializeOwned>(json: serde_json::Value) -> T {
    serde_json::from_value(json).expect("a request body")
}

/// Every default answer of the trait is the JSON error object, a 501
/// under the endpoint's own code, like the refusals the routers write.
#[tokio::test]
async fn every_default_answer_is_the_json_error_object_under_its_code() {
    use serde_json::json;

    use crate::tenant::TenantKey;

    let router = StubRouter;
    let meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let empty = || Request::new(Body::empty());
    let answers = [
        (
            router.health_generate(empty()).await,
            "health_generate_not_implemented",
            "Health generate not implemented",
        ),
        (
            router.get_server_info(empty()).await,
            "server_info_not_implemented",
            "Server info not implemented",
        ),
        (
            router
                .route_chat(
                    None,
                    &meta,
                    body(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})),
                    "m",
                )
                .await,
            "chat_completions_not_implemented",
            "Chat completions not implemented",
        ),
        (
            router
                .route_responses(None, &meta, body(json!({"model": "m", "input": "hi"})), "m")
                .await,
            "responses_not_implemented",
            "Responses endpoint not implemented",
        ),
        (
            router
                .route_messages(
                    None,
                    &meta,
                    body(json!({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})),
                    "m",
                )
                .await,
            "messages_not_implemented",
            "Messages API not yet implemented for this router",
        ),
        (
            router
                .route_interactions(None, &meta, body(json!({"input": "hi"})), Some("m"))
                .await,
            "interactions_not_implemented",
            "Interactions API not implemented for this router",
        ),
        (
            router
                .route_realtime_session(None, &body(json!({"type": "realtime"})))
                .await,
            "realtime_sessions_not_implemented",
            "Realtime sessions not implemented",
        ),
        (
            router
                .route_realtime_client_secret(None, &body(json!({"session": {"type": "realtime"}})))
                .await,
            "realtime_client_secrets_not_implemented",
            "Realtime client secrets not implemented",
        ),
        (
            router
                .route_realtime_transcription_session(None, &body(json!({"type": "transcription"})))
                .await,
            "realtime_transcription_sessions_not_implemented",
            "Realtime transcription sessions not implemented",
        ),
        (
            router.route_realtime_ws(empty(), "m").await,
            "realtime_websocket_not_implemented",
            "Realtime WebSocket not implemented",
        ),
        (
            router.route_realtime_webrtc(empty(), "m").await,
            "realtime_webrtc_not_implemented",
            "Realtime WebRTC not implemented",
        ),
    ];
    for (answer, code, message) in answers {
        assert_eq!(
            error_shape(answer).await,
            (
                http::StatusCode::NOT_IMPLEMENTED,
                code.to_owned(),
                message.to_owned()
            )
        );
    }
}

fn providers() -> Vec<ProviderType> {
    vec![
        ProviderType::OpenAI,
        ProviderType::XAI,
        ProviderType::Anthropic,
        ProviderType::Gemini,
        ProviderType::Custom("oci".to_string()),
    ]
}

#[test]
fn every_provider_has_exactly_one_specific_home_or_the_fallback() {
    let specs = known::all();
    assert_eq!(specs.iter().filter(|s| s.fallback).count(), 1);
    for provider in providers() {
        let specific = specs
            .iter()
            .filter(|s| !s.fallback && (s.serves)(&provider))
            .count();
        assert!(
            specific <= 1,
            "{provider:?} is claimed by {specific} routers"
        );
        assert!(
            spec_for_provider(Some(&provider)).is_some(),
            "{provider:?} has no home"
        );
    }
    assert_eq!(spec_for_provider(None).map(|s| s.backend), Some("openai"));
}

#[test]
fn a_specific_router_beats_the_fallback() {
    assert_eq!(
        spec_for_provider(Some(&ProviderType::Anthropic)).map(|s| s.backend),
        Some("anthropic")
    );
    assert_eq!(
        spec_for_provider(Some(&ProviderType::XAI)).map(|s| s.backend),
        Some("openai")
    );
    // A future provider-specific router wins over the fallback even
    // though the fallback also takes custom providers.
    let mut specs = known::all().to_vec();
    let oci = ExternalRouterSpec {
        router_id: "http-oci",
        backend: "oci",
        label: "OCI",
        feature: "provider-oci",
        serves: |p| matches!(p, ProviderType::Custom(name) if name == "oci"),
        fallback: false,
        compiled: true,
        build: |_ctx| Box::pin(async { Err("not built in tests".to_string()) }),
    };
    specs.push(oci);
    let custom = ProviderType::Custom("oci".to_string());
    assert_eq!(
        home_among(&specs, Some(&custom)).map(|s| s.backend),
        Some("oci")
    );
    let other = ProviderType::Custom("together".to_string());
    assert_eq!(
        home_among(&specs, Some(&other)).map(|s| s.backend),
        Some("openai")
    );
}

#[test]
fn a_missing_router_names_its_feature() {
    assert!(known::ANTHROPIC
        .not_compiled()
        .contains("`provider-anthropic`"));
    assert_eq!(spec_for_backend("gemini").map(|s| s.label), Some("Gemini"));
    assert!(spec_for_backend("oci").is_none());
}
