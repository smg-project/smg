use std::{sync::Arc, time::Duration};

use openai_protocol::interactions::InteractionsRequest;
use serde_json::json;

use super::request_building;
use crate::{
    gemini::context::{RequestContext, SharedComponents},
    tenant::{TenantKey, TenantRequestMeta},
    test_support::{http_client, StubWorkers},
};

/// The step's upstream payload for a request with this body, routed to
/// `gemini-routed`.
async fn payload_of(body: serde_json::Value) -> serde_json::Value {
    let request: InteractionsRequest = serde_json::from_value(body).expect("a request");
    let components = SharedComponents {
        client: http_client(),
        workers: Arc::new(StubWorkers::default()),
        request_timeout: Duration::from_secs(5),
    };
    let mut ctx = RequestContext::new(
        Arc::new(request),
        None,
        Some("gemini-routed".to_string()),
        TenantRequestMeta::new(TenantKey::new("t")),
        Arc::new(components),
    );
    assert!(request_building(&mut ctx).await.is_ok());
    ctx.processing.payload.expect("a payload")
}

/// The upstream payload carries the request's fields the gateway does not
/// model as the client wrote them, and the gateway's own overrides still
/// apply: the model the route names, and `store: false` for a model
/// request.
#[tokio::test]
async fn the_payload_keeps_the_fields_the_gateway_does_not_model() {
    let payload = payload_of(json!({
        "model": "m", "input": "hi", "store": false,
        "continuation_token": "tok_1",
        "safety_settings": [{"type": "harassment", "threshold": "block_none"}],
        "service_tier": "flex"
    }))
    .await;
    assert_eq!(payload["continuation_token"], "tok_1");
    assert_eq!(payload["safety_settings"][0]["threshold"], "block_none");
    assert_eq!(payload["service_tier"], "flex");
    assert_eq!(payload["model"], "gemini-routed");
    assert_eq!(payload["store"], false);
}

/// A generation or agent config written in camelCase, keys and all,
/// reaches the upstream with each of its values, in snake_case.
#[tokio::test]
async fn a_camel_case_config_reaches_the_payload_with_its_values() {
    let payload = payload_of(json!({
        "model": "m", "input": "hi", "store": false,
        "generationConfig": {"maxOutputTokens": 1, "stopSequences": ["END"]}
    }))
    .await;
    assert_eq!(
        payload["generation_config"],
        json!({"max_output_tokens": 1, "stop_sequences": ["END"]})
    );
    assert!(payload.get("generationConfig").is_none(), "{payload}");

    let payload = payload_of(json!({
        "agent": "deep-research", "input": "hi",
        "agentConfig": {"type": "deep-research", "thinkingSummaries": "auto"}
    }))
    .await;
    assert_eq!(
        payload["agent_config"],
        json!({"type": "deep-research", "thinking_summaries": "auto"})
    );
}
