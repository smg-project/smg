use std::{sync::Arc, time::Duration};

use axum::http::StatusCode;
use openai_protocol::interactions::InteractionsRequest;
use serde_json::json;

use super::previous_interaction_loading;
use crate::{
    gemini::context::{RequestContext, SharedComponents},
    tenant::{TenantKey, TenantRequestMeta},
    test_support::{http_client, StubWorkers},
};

/// A request that links a prior interaction is answered 501 in either
/// spelling of the key: the camelCase one fills the typed field this step
/// reads, so it cannot ride past the step to the upstream.
#[tokio::test]
async fn a_previous_interaction_in_either_spelling_is_not_implemented() {
    for key in ["previous_interaction_id", "previousInteractionId"] {
        let mut body = json!({"agent": "deep-research", "input": "hi"});
        body[key] = json!("int_1");
        let request: InteractionsRequest = serde_json::from_value(body).expect("a request");
        let components = SharedComponents {
            client: http_client(),
            workers: Arc::new(StubWorkers::default()),
            request_timeout: Duration::from_secs(5),
        };
        let mut ctx = RequestContext::new(
            Arc::new(request),
            None,
            None,
            TenantRequestMeta::new(TenantKey::new("t")),
            Arc::new(components),
        );
        let Err(response) = previous_interaction_loading(&mut ctx).await else {
            panic!("{key}: the link is refused");
        };
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{key}");
    }
}
