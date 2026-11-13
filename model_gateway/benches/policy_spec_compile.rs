//! Converter cost: what a start pays to compile the routing flags into a policy specification,
//! for a configuration with no flag off its default, for the full regular-mode flag set (every
//! cache-aware knob, the sticky override, the warmup slice, the overload block, the boundaries)
//! with and without the command line's given set, and for a disaggregated pair with a policy
//! per side.
//!
//! Run with: cargo bench --bench policy_spec_compile

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use smg::config::{
    policy_spec::{compile, compile_given, GivenFlags},
    types::{
        CacheIndexKind, ManualAssignmentMode, PdPairingMode, PolicyConfig, PowerOfTwoLoadMetric,
        RouterConfig, RoutingKeyOverrideConfig, RoutingMode,
    },
};

fn workers() -> Vec<String> {
    vec![
        "http://worker-a:8000".to_string(),
        "http://worker-b:8000".to_string(),
    ]
}

/// No routing flag off its default: the default profile alone.
fn default_config() -> RouterConfig {
    RouterConfig::builder()
        .regular_mode(workers())
        .build_unchecked()
}

/// Every regular-mode routing flag off its default.
fn full_regular() -> RouterConfig {
    RouterConfig::builder()
        .regular_mode(workers())
        .policy(PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            balance_prefix_margin_tokens: 1024,
            balance_uncached_credit: true,
            eviction_interval_secs: 60,
            max_tree_size: 16_777_216,
            block_size: 32,
            balance_token_usage_threshold: 0.7,
            overload_token_usage_threshold: 0.95,
            overlap_decay: 0.5,
            selection_temperature: 0.05,
            cache_index: CacheIndexKind::Hash,
            cache_ttl_secs: 300,
            cache_boundaries: vec![256, 1024, 4096],
            selection_policy: None,
            selection_accounting_ttl_ms: 200,
        })
        .cache_boundaries(vec![256, 1024, 4096])
        .routing_key_override(RoutingKeyOverrideConfig {
            enabled: true,
            eviction_interval_secs: 120,
            max_idle_secs: 600,
            assignment_mode: ManualAssignmentMode::MinLoad,
            headers: vec!["x-session-id".to_string(), "x-tenant".to_string()],
        })
        .worker_warmup(30, 0.25, 1024, 0.5, 8)
        .worker_overload_waiting_requests(Some(8))
        .worker_overload_token_usage(Some(0.8))
        .worker_overload_shed(true)
        .build_unchecked()
}

/// A disaggregated pair with a policy per side and the main policy tuned.
fn full_pair() -> RouterConfig {
    RouterConfig::builder()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill-a:8000".to_string(), Some(8998))],
            decode_urls: vec!["http://decode-a:8000".to_string()],
            prefill_policy: Some(PolicyConfig::Bucket {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                bucket_adjust_interval_secs: 5,
            }),
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::Requests,
            }),
        })
        .policy(PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.3,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        })
        .pd_pairing_mode(PdPairingMode::Strict)
        .build_unchecked()
}

fn bench_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_spec_compile");
    let default = default_config();
    let regular = full_regular();
    let pair = full_pair();
    let given: GivenFlags = compile(&regular)
        .report
        .flags()
        .map(str::to_string)
        .collect();
    group.bench_function("default", |b| b.iter(|| compile(black_box(&default))));
    group.bench_function("full_regular", |b| b.iter(|| compile(black_box(&regular))));
    group.bench_function("full_regular_given", |b| {
        b.iter(|| compile_given(black_box(&regular), black_box(&given)));
    });
    group.bench_function("full_pair", |b| b.iter(|| compile(black_box(&pair))));
    group.finish();
}

criterion_group!(benches, bench_compile);
criterion_main!(benches);
