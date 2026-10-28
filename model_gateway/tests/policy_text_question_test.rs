//! The policy registry answers whether some policy wants a request's text
//! without allocating: the answer is kept where a policy is added or
//! removed, and a request reads it. A separate executable, so it can declare
//! the counting allocator the pin needs.

#![expect(unsafe_code, reason = "a global allocator that counts allocations")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
};

use http::HeaderMap;
use smg::{
    config::types::{PolicyConfig, RoutingKeyOverrideConfig},
    policies::PolicyRegistry,
};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations_during(work: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    work();
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

/// A cache-aware default (text wanted, no eviction thread) and 64 models on
/// hinted cache-aware policies of their own: a request's question allocates
/// nothing, with and without a routing key and token ids, however many
/// models are registered.
#[test]
fn a_request_asks_whether_text_is_wanted_without_allocating() {
    let registry = PolicyRegistry::with_override(
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
        RoutingKeyOverrideConfig {
            enabled: true,
            ..Default::default()
        },
    );
    for model in 0..64 {
        registry.on_worker_added(&format!("model-{model}"), Some("cache_aware"));
    }
    let plain = HeaderMap::new();
    let mut keyed = HeaderMap::new();
    keyed.insert(
        "x-smg-routing-key",
        "session-A".parse().expect("a header value"),
    );

    assert!(registry.any_policy_needs_request_text(Some(&plain)));
    assert!(!registry.any_policy_needs_request_text(Some(&keyed)));
    assert!(!registry.any_policy_needs_request_text_with_tokens(Some(&plain)));

    let allocations = allocations_during(|| {
        for _ in 0..1_000 {
            black_box(registry.any_policy_needs_request_text(Some(&plain)));
            black_box(registry.any_policy_needs_request_text(Some(&keyed)));
            black_box(registry.any_policy_needs_request_text_with_tokens(Some(&plain)));
            black_box(registry.any_policy_needs_request_text_with_tokens(None));
        }
    });
    assert_eq!(
        allocations, 0,
        "the text questions allocated {allocations} times over 4,000 requests"
    );
}
