//! The tenant rate limiter's per-request cost: one reservation taken and
//! settled, as the routers do it on every request of a tenant under the
//! limiter, over the in-memory backend. Two groups: the tenant-global
//! bucket alone, and with a model rule's bucket beside it. Driven through
//! a `current_thread` runtime with `block_on`, as `scheduler_admission`
//! drives `admit_async`, so the runtime's share is the same in both.
//!
//! Run: `cargo bench -p smg --bench rate_limit_reserve`.

#![expect(clippy::expect_used, clippy::unwrap_used)]

use std::{hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, Criterion};
use smg::{
    config::types::RouterConfig,
    rate_limit::{RateLimitManager, Reservation, ReserveRequest, UsageSettlement},
    tenant::TenantKey,
};
use tokio::runtime::Builder;
use uuid::Uuid;

/// A manager over the in-memory backend: one tenant budget with a model rule
/// the `ruled-` models match.
fn manager(dir: &std::path::Path) -> Arc<RateLimitManager> {
    let path = dir.join("rate_limit.yaml");
    std::fs::write(
        &path,
        "default_policy:\n  tokens_per_minute: 1000000000\n  requests_per_minute: 1000000000\n  \
         model_rules:\n    - rule_id: ruled\n      matcher:\n        type: prefix\n        \
         value: ruled-\n      tokens_per_minute: 1000000000\n      requests_per_minute: 1000000000\n",
    )
    .unwrap();
    let config = RouterConfig::builder()
        .tenant_rate_limit_enabled(true)
        .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
        .build_unchecked();
    RateLimitManager::from_config(&config).unwrap().unwrap()
}

/// One reservation taken and settled; a denial comes back with its
/// retry-after seconds.
async fn reserve_and_settle(
    manager: &RateLimitManager,
    tenant: &TenantKey,
    model: &str,
) -> Result<(), u64> {
    let request = ReserveRequest {
        request_charge_id: Uuid::now_v7(),
        tenant_key: tenant.clone(),
        model_id: Some(model),
        estimated_input_tokens: 40,
    };
    match manager.reserve(request).await {
        Reservation::Admitted(handle) => {
            handle
                .settle_success(UsageSettlement {
                    actual_input_tokens: 42,
                    completion_tokens: 8,
                    ..UsageSettlement::default()
                })
                .await;
            Ok(())
        }
        Reservation::Denied { retry_after_secs } => Err(retry_after_secs),
    }
}

fn bench_reserve_settle(c: &mut Criterion) {
    let dir = tempfile::tempdir().expect("a temp dir");
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    // The manager spawns its idle-bucket sweep on the runtime it is built in.
    let manager = {
        let _runtime = runtime.enter();
        manager(dir.path())
    };
    let tenant = TenantKey::new("auth:team-red");
    let mut group = c.benchmark_group("rate_limit_reserve_settle");
    for (name, model) in [
        ("tenant_global", "plain-model"),
        ("with_model_rule", "ruled-model"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                runtime
                    .block_on(reserve_and_settle(
                        black_box(&manager),
                        black_box(&tenant),
                        black_box(model),
                    ))
                    .expect("admitted under an ample budget");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_reserve_settle);
criterion_main!(benches);
