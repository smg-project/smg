//! The concurrency bulkheads' hot path: one acquisition and release per
//! request on the limiter alone, and the gateway's dispatch of a request
//! with the caps off and on.
#![expect(clippy::expect_used)]

use std::{hint::black_box, sync::Arc};

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use criterion::{criterion_group, criterion_main, Criterion};
use openai_protocol::generate::GenerateRequest;
use smg::{
    middleware::{Bulkhead, Bulkheads, RouteRequestMeta, TenantKey, TenantRequestMeta},
    routers::{factory::router_ids, gateway::Gateway, RouterTrait},
    worker::WorkerRegistry,
};

/// A router that answers at once: what remains is the gateway's dispatch.
#[derive(Debug)]
struct ImmediateRouter;

#[async_trait]
impl RouterTrait for ImmediateRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        _body: GenerateRequest,
        _model_id: &str,
    ) -> Response {
        (StatusCode::OK, "ok").into_response()
    }

    fn router_type(&self) -> &'static str {
        "immediate"
    }
}

fn gateway(bulkheads: Bulkheads) -> Arc<Gateway> {
    let gateway = Arc::new(Gateway::new(Arc::new(WorkerRegistry::new())).with_bulkheads(bulkheads));
    gateway.register_router(router_ids::HTTP_REGULAR, Arc::new(ImmediateRouter));
    gateway
}

fn bench_limiter(c: &mut Criterion) {
    let bulkhead = Bulkhead::new(64);
    // A key with a request in flight: the entry exists, the acquisition is
    // a lookup and a compare-and-swap.
    let anchor = bulkhead.try_acquire("auth:tenant").expect("the first slot");
    c.bench_function("bulkhead/acquire_release/key_in_flight", |b| {
        b.iter(|| black_box(bulkhead.try_acquire(black_box("auth:tenant"))));
    });
    drop(anchor);
    // A key with nothing in flight: the acquisition inserts the entry, the
    // release removes it.
    c.bench_function("bulkhead/acquire_release/key_idle", |b| {
        b.iter(|| black_box(bulkhead.try_acquire(black_box("auth:tenant"))));
    });

    let both = Bulkheads::new(1024, 1024);
    let anchor = both.try_acquire(Some("auth:tenant"), Some("model-a"));
    c.bench_function("bulkheads/try_acquire/tenant_and_model", |b| {
        b.iter(|| black_box(both.try_acquire(Some("auth:tenant"), Some("model-a")).ok()));
    });
    drop(anchor);
    let off = Bulkheads::none();
    c.bench_function("bulkheads/try_acquire/off", |b| {
        b.iter(|| black_box(off.try_acquire(Some("auth:tenant"), Some("model-a")).ok()));
    });
}

fn bench_dispatch(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");
    let body: GenerateRequest =
        serde_json::from_value(serde_json::json!({ "text": "hello", "model": "model-a" }))
            .expect("a generate request");
    let meta = RouteRequestMeta::new(TenantKey::from("auth:tenant"));

    // The caps on, with the request's tenant and model in flight already
    // (the steady state under load) and with neither in flight (each request
    // inserts and removes both keys' entries).
    let steady = Bulkheads::new(1024, 1024);
    let anchor = steady.try_acquire(Some("auth:tenant"), Some("model-a"));
    for (name, bulkheads) in [
        ("off", Bulkheads::none()),
        ("tenant_and_model_in_flight", steady),
        ("tenant_and_model_idle", Bulkheads::new(1024, 1024)),
    ] {
        let gateway = gateway(bulkheads);
        c.bench_function(&format!("gateway/dispatch_generate/caps_{name}"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let response = gateway
                        .route_generate(None, &meta, body.clone(), "model-a")
                        .await;
                    black_box(response.status())
                });
            });
        });
    }
    drop(anchor);
}

criterion_group!(benches, bench_limiter, bench_dispatch);
criterion_main!(benches);
