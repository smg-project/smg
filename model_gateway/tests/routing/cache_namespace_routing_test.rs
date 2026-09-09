//! Cache-aware routing under cache namespaces.
//!
//! Engines partition their prefix caches by `cache_salt` / `extra_key` /
//! the LoRA adapter. Requests that differ in any of those can never share
//! engine cache, so the router must not assert prefix affinity between them,
//! while requests inside one partition keep full affinity and unpartitioned
//! requests behave exactly as before.

use std::sync::Arc;

use openai_protocol::{common::CachePartition, worker::HealthCheckConfig};
use smg::{
    policies::{
        CacheAwareConfig, CacheAwarePolicy, CacheNamespace, LoadBalancingPolicy, SelectWorkerInfo,
    },
    worker::{BasicWorkerBuilder, Worker, WorkerType},
};

fn workers() -> Vec<Arc<dyn Worker>> {
    ["http://w1:8000", "http://w2:8000"]
        .iter()
        .map(|url| {
            Arc::new(
                BasicWorkerBuilder::new(*url)
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

fn policy() -> CacheAwarePolicy {
    CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        cache_threshold: 0.5,
        ..Default::default()
    })
}

fn salted(salt: &str) -> Option<CacheNamespace> {
    CacheNamespace::derive(&CachePartition {
        cache_salt: Some(salt),
        extra_key: None,
        lora_path: None,
    })
}

fn route_text(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    text: &str,
    cache_namespace: Option<CacheNamespace>,
) -> Option<usize> {
    policy.select_worker(
        workers,
        &SelectWorkerInfo {
            request_text: Some(text),
            cache_namespace,
            ..Default::default()
        },
    )
}

fn route_tokens(
    policy: &CacheAwarePolicy,
    workers: &[Arc<dyn Worker>],
    tokens: &[u32],
    cache_namespace: Option<CacheNamespace>,
) -> Option<usize> {
    policy.select_worker(
        workers,
        &SelectWorkerInfo {
            tokens: Some(tokens),
            cache_namespace,
            ..Default::default()
        },
    )
}

/// HTTP path (string tree): a shared prompt under two salts is two affinities.
#[test]
fn salted_prompts_do_not_share_affinity_on_the_string_tree() {
    let policy = policy();
    let workers = workers();
    policy.init_workers(&workers);
    let prompt = "You are a helpful assistant. Summarize the quarterly report.";

    // w1 is busier, so tenant A's first request misses onto w2.
    workers[0].increment_load();
    assert_eq!(
        route_text(&policy, &workers, prompt, salted("tenant-a")),
        Some(1)
    );

    // Tenant A keeps its affinity once w2 is the busier worker...
    workers[1].increment_load();
    workers[1].increment_load();
    assert_eq!(
        route_text(&policy, &workers, prompt, salted("tenant-a")),
        Some(1)
    );

    // ...but tenant B shares only the prompt bytes, never the engine cache:
    // its request is a miss and goes to the least-loaded worker.
    assert_eq!(
        route_text(&policy, &workers, prompt, salted("tenant-b")),
        Some(0)
    );

    // An unpartitioned request does not see tenant entries either.
    assert_eq!(route_text(&policy, &workers, prompt, None), Some(0));
}

/// gRPC path (token tree): same contract on token ids.
#[test]
fn salted_prompts_do_not_share_affinity_on_the_token_tree() {
    let policy = policy();
    let workers = workers();
    policy.init_workers(&workers);
    // Two full pages: the token tree is page-aligned (16 tokens by default).
    let prompt: Vec<u32> = (101..133).collect();

    workers[0].increment_load();
    assert_eq!(
        route_tokens(&policy, &workers, &prompt, salted("tenant-a")),
        Some(1)
    );

    workers[1].increment_load();
    workers[1].increment_load();
    assert_eq!(
        route_tokens(&policy, &workers, &prompt, salted("tenant-a")),
        Some(1)
    );
    assert_eq!(
        route_tokens(&policy, &workers, &prompt, salted("tenant-b")),
        Some(0)
    );
    assert_eq!(route_tokens(&policy, &workers, &prompt, None), Some(0));
}

/// The namespace marker never counts toward the match ratio: a prompt that
/// shares nothing but the marker with a tenant's entries is a miss.
#[test]
fn the_namespace_marker_alone_is_not_a_prefix_hit() {
    let policy = policy();
    let workers = workers();
    policy.init_workers(&workers);

    let seed: Vec<u32> = (1..33).collect();
    workers[0].increment_load();
    assert_eq!(
        route_tokens(&policy, &workers, &seed, salted("tenant-a")),
        Some(1)
    );
    assert_eq!(
        route_text(
            &policy,
            &workers,
            "a long enough prompt",
            salted("tenant-a")
        ),
        Some(1)
    );
    workers[1].increment_load();
    workers[1].increment_load();
    // One unrelated token (or char): were the marker counted, this would be
    // a near-total match and stick to w2; it must miss and take the
    // least-loaded w1.
    assert_eq!(
        route_tokens(&policy, &workers, &[999], salted("tenant-a")),
        Some(0)
    );
    assert_eq!(
        route_text(&policy, &workers, "z", salted("tenant-a")),
        Some(0)
    );
}

/// Unpartitioned requests behave exactly as before the namespace existed.
#[test]
fn unpartitioned_requests_keep_plain_affinity() {
    let policy = policy();
    let workers = workers();
    policy.init_workers(&workers);

    workers[0].increment_load();
    assert_eq!(route_text(&policy, &workers, "hello world", None), Some(1));
    workers[1].increment_load();
    workers[1].increment_load();
    assert_eq!(route_text(&policy, &workers, "hello world", None), Some(1));
    assert_eq!(route_text(&policy, &workers, "hello", None), Some(1));
}
