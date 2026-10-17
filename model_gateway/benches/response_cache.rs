//! The response cache's per-request cost: the canonical form and key of a
//! chat request (what a cacheable request pays before the router runs, hit
//! or miss), the whole decision, and a hit's lookup.
//!
//! Run: `cargo bench -p smg --bench response_cache`.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a bench: a failed set-up should panic"
)]

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, HeaderValue, Response},
};
use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openai_protocol::chat::ChatCompletionRequest;
use serde_json::json;
use smg::{
    middleware::{RouteRequestMeta, TenantKey},
    routers::response_cache::{canonical_bytes, key_for, Decision, ResponseCache, Scope},
};

/// A deterministic chat request whose last user message is `prompt_bytes`
/// of text (the typical request is a kilobyte; a long context tens of them).
fn request(prompt_bytes: usize) -> ChatCompletionRequest {
    let value = json!({
        "model": "bench-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "x".repeat(prompt_bytes)}
        ],
        "temperature": 0,
        "max_tokens": 64,
        "logit_bias": {"1": 1.0, "2": -1.0},
        "user": "bench"
    });
    serde_json::from_value(value).expect("a valid chat request")
}

fn bench_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_cache_key");
    for prompt_bytes in [1usize << 10, 32 << 10] {
        let request = request(prompt_bytes);
        let canonical_len = canonical_bytes(&request).expect("serializes").len();
        group.throughput(Throughput::Bytes(canonical_len as u64));
        group.bench_with_input(
            BenchmarkId::new("canonical_and_hash", prompt_bytes),
            &request,
            |b, request| {
                b.iter(|| {
                    let canonical = canonical_bytes(request).expect("serializes");
                    key_for(Scope::Chat, "header:bench", &canonical, &[])
                });
            },
        );
    }
    group.finish();
}

fn bench_decide_and_hit(c: &mut Criterion) {
    let cache = Arc::new(ResponseCache::new(
        Duration::from_secs(300),
        10_000,
        256 << 20,
        1 << 20,
    ));
    let tenant = RouteRequestMeta::new(TenantKey::from("header:bench"));
    let request = request(1 << 10);
    let mut group = c.benchmark_group("response_cache_request");
    group.bench_function("decide_1k", |b| {
        b.iter(|| cache.decide(None, &tenant, &request));
    });

    // A stored 4 KB answer: the hit path builds the client's response.
    let Decision::Lookup { key, ttl } = cache.decide(None, &tenant, &request) else {
        panic!("a deterministic request is cacheable");
    };
    let answer = Bytes::from(json!({"id": "chatcmpl-bench", "choices": [{"message": {"role": "assistant", "content": "y".repeat(4000)}}]}).to_string());
    let response = Response::builder()
        .status(200)
        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
        .body(Body::from(answer))
        .expect("a response");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");
    runtime.block_on(async {
        let recorded = cache.record(key, ttl, response);
        to_bytes(recorded.into_body(), 1 << 20)
            .await
            .expect("the body");
    });
    assert_eq!(cache.occupancy().0, 1, "the answer is stored");
    group.bench_function("hit_4k", |b| {
        b.iter(|| cache.hit(&key).expect("a hit"));
    });
    group.finish();
}

criterion_group!(benches, bench_key, bench_decide_and_hit);
criterion_main!(benches);
