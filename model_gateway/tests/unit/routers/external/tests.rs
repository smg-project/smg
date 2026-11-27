use std::sync::Mutex;

use axum::{http::StatusCode, response::IntoResponse};
use serde_json::{json, Value};
use smg_external_router::spec_for_provider;

use super::*;
use crate::{
    middleware::{RouteRequestMeta, TenantKey},
    routers::grpc::backend_client::BasicWorkerBuilder,
};

/// A provider router of the kind the adapter mounts, keeping the
/// Messages request it was handed as the JSON it would post: the trait's
/// two required methods, every other route its default.
#[derive(Debug, Default)]
struct RecordingProvider {
    posted: Mutex<Option<Value>>,
}

#[async_trait]
impl ExternalRouter for RecordingProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn router_type(&self) -> &'static str {
        "stub"
    }

    async fn route_messages(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        _model_id: &str,
    ) -> Response {
        let posted = serde_json::to_value(&body).expect("a request serializes");
        *self.posted.lock().expect("the recorder") = Some(posted);
        StatusCode::OK.into_response()
    }
}

/// A Messages request reaches its provider without the gateway's `rid`:
/// a thread turn carries the id the gateway minted, a plain request the
/// client's own, and a provider refuses the unknown field.
#[tokio::test]
async fn a_provider_gets_no_rid() {
    let provider = Arc::new(RecordingProvider::default());
    let adapter = ExternalRouterAdapter {
        spec: spec_for_provider(None).expect("the OpenAI-compatible spec"),
        inner: provider.clone(),
        rate_limit_manager: None,
        worker_registry: Arc::new(WorkerRegistry::new()),
    };
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 1,
        "rid": "msg_minted",
        "messages": [{"role": "user", "content": "q"}],
    }))
    .expect("a request");
    let tenant_meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let response = adapter
        .route_messages(None, &tenant_meta, Arc::new(request), "m")
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let posted = provider
        .posted
        .lock()
        .expect("the recorder")
        .take()
        .expect("a posted request");
    assert!(posted.get("rid").is_none(), "{posted}");
    assert_eq!(posted["model"], "m");
    assert_eq!(posted["messages"][0]["content"], "q");
}

/// The adapter holds a worker's load under the request's routing key:
/// the key the header names counts on the worker until the hold drops,
/// and a request without the header holds unkeyed load.
#[test]
fn a_hold_counts_the_requests_routing_key() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://external:8000").build());
    let adapter = GatewayWorker::handle(Arc::clone(&worker));
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-smg-routing-key",
        "key-123".parse().expect("a header value"),
    );

    let keyed = adapter.hold_load(Some(&headers));
    assert_eq!((worker.load(), worker.routing_key_load()), (1, 1));
    let unkeyed = adapter.hold_load(None);
    assert_eq!((worker.load(), worker.routing_key_load()), (2, 1));

    drop(keyed);
    drop(unkeyed);
    assert_eq!((worker.load(), worker.routing_key_load()), (0, 0));
}
