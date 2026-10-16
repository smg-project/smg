//! The request-path cost of per-tenant usage accounting: one `record` for a
//! tenant with a row (the steady state), for a tenant past the maximum (the
//! `other` bucket), for a hundred tenants interleaved (the lookup under a
//! populated table), and the disabled path (nothing installed).

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use smg::observability::tenant_usage::{self, RequestUsage, TenantUsage};

fn usage() -> RequestUsage {
    RequestUsage {
        prompt_tokens: 512,
        completion_tokens: 128,
        elapsed: Duration::from_millis(850),
    }
}

fn bench_record(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_usage_record");

    // The disabled path: the free function with no table installed.
    group.bench_function("disabled", |b| {
        b.iter(|| tenant_usage::record("auth:tenant-0", "model-a", usage()));
    });

    let table = TenantUsage::new(128);
    table.record("auth:tenant-0", "model-a", usage());
    group.bench_function("tracked_tenant_hit", |b| {
        b.iter(|| table.record("auth:tenant-0", "model-a", usage()));
    });

    let tenants: Vec<String> = (0..100).map(|i| format!("auth:tenant-{i}")).collect();
    for tenant in &tenants {
        table.record(tenant, "model-a", usage());
        table.record(tenant, "model-b", usage());
    }
    group.bench_function("hundred_tenants_two_models_round_robin", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let tenant = &tenants[i % tenants.len()];
            let model = if i.is_multiple_of(2) {
                "model-a"
            } else {
                "model-b"
            };
            i += 1;
            table.record(tenant, model, usage());
        });
    });

    let full = TenantUsage::new(4);
    for i in 0..4 {
        full.record(&format!("auth:tenant-{i}"), "model-a", usage());
    }
    group.bench_function("overflow_tenant_to_other", |b| {
        b.iter(|| full.record("ip:198.51.100.7", "model-a", usage()));
    });

    // A first sighting: the row's creation, once per tenant for the life of
    // the process; measured on a fresh table per iteration.
    group.bench_function("first_sighting_new_row", |b| {
        b.iter_batched(
            || TenantUsage::new(128),
            |table| table.record("auth:new-tenant", "model-a", usage()),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_render(c: &mut Criterion) {
    let table = TenantUsage::new(256);
    for i in 0..128 {
        for model in ["model-a", "model-b", "model-c"] {
            table.record(&format!("auth:tenant-{i}"), model, usage());
        }
    }
    c.bench_function("tenant_usage_render_128_tenants_3_models", |b| {
        b.iter(|| {
            let mut out = String::with_capacity(64 * 1024);
            table.render_into(&mut out);
            out
        });
    });
}

criterion_group!(benches, bench_record, bench_render);
criterion_main!(benches);
