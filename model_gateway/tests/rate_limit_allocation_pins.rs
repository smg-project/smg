//! What a tenant reservation allocates, pinned at today's count, so the
//! next change that adds a copy to the reserve path fails a test rather
//! than moving a bench inside its noise. A separate executable, so it can
//! declare the counting allocator the pin needs.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]
#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a reserve allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
};

use smg::{
    config::types::RouterConfig,
    rate_limit::{RateLimitManager, Reservation, ReserveRequest, UsageSettlement},
    tenant::TenantKey,
};
use uuid::Uuid;

thread_local! {
    /// Allocation and reallocation calls this thread made.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = CALLS.with(Cell::get);
    let out = work();
    (out, CALLS.with(Cell::get) - before)
}

/// A manager over the in-memory backend: one tenant budget with a model rule
/// the reserved model matches, so a reservation takes both buckets.
fn manager(dir: &std::path::Path) -> Arc<RateLimitManager> {
    let path = dir.join("rate_limit.yaml");
    std::fs::write(
        &path,
        "default_policy:\n  tokens_per_minute: 1000000\n  requests_per_minute: 1000000\n  \
         model_rules:\n    - rule_id: ruled\n      matcher:\n        type: prefix\n        \
         value: ruled-\n      tokens_per_minute: 1000000\n      requests_per_minute: 1000000\n",
    )
    .expect("the policy file");
    let config = RouterConfig::builder()
        .tenant_rate_limit_enabled(true)
        .tenant_rate_limit_config(Some(path.to_str().expect("utf-8 path").to_string()))
        .build_unchecked();
    RateLimitManager::from_config(&config)
        .expect("a valid policy")
        .expect("the limiter is enabled")
}

/// One logical request's reservation under `charge_id` and its settlement,
/// as the routers do it; a denial comes back with its retry-after seconds.
async fn reserve_and_settle(
    manager: &RateLimitManager,
    charge_id: Uuid,
    tenant: &TenantKey,
    model: &str,
) -> Result<(), u64> {
    let request = ReserveRequest {
        request_charge_id: charge_id,
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

/// A reservation that takes the tenant-global bucket and a model rule's,
/// settled: the shared handle alone, with or without a rule. The in-memory
/// backend is called by static dispatch, so its reserve and settle build no
/// boxed future, and a rule's scope key shares the rule's id with the
/// compiled policy; nothing per bucket beyond its key.
///
/// The map of active reservations is sharded, and a shard allocates its
/// table on its first insert; which shard a charge id lands in follows the
/// process's hash seed, so a reservation under a fresh id grows the map or
/// not by chance (the more shards, the more often). Each measured
/// reservation therefore reuses a charge id a warm-up reservation settled:
/// its shard has its table, and the count is the reserve path's own.
#[test]
fn a_reservation_with_a_model_rule_allocates_this_much() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    // The manager spawns its idle-bucket sweep on the runtime it is built in.
    let manager = {
        let _runtime = runtime.enter();
        manager(dir.path())
    };
    let tenant = TenantKey::new("auth:team-red");
    let ruled_id = Uuid::now_v7();
    let plain_id = Uuid::now_v7();
    // Warm: the buckets exist, and each measured charge id has been through
    // the active map once.
    for _ in 0..8 {
        runtime
            .block_on(reserve_and_settle(
                &manager,
                Uuid::now_v7(),
                &tenant,
                "ruled-model",
            ))
            .expect("admitted under an ample budget");
    }
    for (charge_id, model) in [(ruled_id, "ruled-model"), (plain_id, "other-model")] {
        runtime
            .block_on(reserve_and_settle(&manager, charge_id, &tenant, model))
            .expect("admitted under an ample budget");
    }
    let (admitted, ruled) = allocations_during(|| {
        runtime.block_on(reserve_and_settle(
            &manager,
            ruled_id,
            &tenant,
            "ruled-model",
        ))
    });
    admitted.expect("admitted under an ample budget");
    let (admitted, plain) = allocations_during(|| {
        runtime.block_on(reserve_and_settle(
            &manager,
            plain_id,
            &tenant,
            "other-model",
        ))
    });
    admitted.expect("admitted under an ample budget");
    eprintln!(
        "allocation calls per reservation: with a model rule {ruled}, tenant-global alone {plain}"
    );
    assert_eq!(ruled, 1, "with a model rule");
    assert_eq!(plain, 1, "tenant-global alone");
}
