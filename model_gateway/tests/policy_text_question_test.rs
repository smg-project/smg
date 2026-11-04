//! Policy paths a request walks without allocating, pinned: the registry's
//! question whether some policy wants the request's text (the answer is kept
//! where a policy is added or removed), and the least-load decision, which
//! allocates its eligible list and nothing else once it has dispatched to a
//! worker. A separate executable, so it can declare the counting allocator
//! the pins need.

#![expect(unsafe_code, reason = "a global allocator that counts allocations")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    sync::Arc,
};

use http::HeaderMap;
use openai_protocol::worker::HealthCheckConfig;
use smg::{
    config::types::{PolicyConfig, RoutingKeyOverrideConfig},
    policies::{LeastLoadPolicy, LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    worker::{
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, Worker, WorkerType,
    },
};

struct Counting;

thread_local! {
    /// The allocation calls of this thread while one of its pins measures,
    /// `None` outside a measurement. A thread counts its own calls alone, so
    /// neither the test harness nor a sibling test on another thread reaches
    /// a pin's count. Const-initialised with no destructor, so reading it
    /// inside the allocator allocates nothing and never runs a destructor.
    static COUNTED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Counts one allocation call on this thread while it measures; a thread
/// whose local is gone (tearing down) counts nothing.
fn count() {
    let _ = COUNTED.try_with(|counted| {
        if let Some(calls) = counted.get() {
            counted.set(Some(calls + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocation calls `work` makes on this thread: nothing another thread
/// does in the meantime is counted.
fn allocations_during(work: impl FnOnce()) -> usize {
    COUNTED.with(|counted| counted.set(Some(0)));
    work();
    COUNTED.with(|counted| counted.replace(None)).unwrap_or(0)
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

/// Eight ready regular workers (no health checker behind them), each with a
/// published report and requests in flight, under a least-load policy that
/// has dispatched to every one of them: a decision allocates its eligible
/// list and nothing else. The winner's credit finds its tally by a borrowed
/// key; only a worker's first dispatch allocates the key. A report is
/// published before every decision, so the credit of the one before is
/// released and the tally holds at most one dispatch per worker.
#[test]
fn a_least_load_decision_allocates_its_eligible_list_and_nothing_else() {
    let policy = LeastLoadPolicy::new();
    let feed = TestFeed::attached_to(&policy);
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{i}:8000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let reports = workers
        .iter()
        .enumerate()
        .map(|(i, worker)| {
            for _ in 0..i % 3 {
                worker.increment_load();
            }
            (
                worker.url().to_string(),
                one_rank_load(512 * i as i32, 0.1 * i as f64, 2_000.0 + 100.0 * i as f64),
            )
        })
        .collect();
    feed.publish(&reports);
    let tokens: Vec<u32> = (0..1_024).collect();
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        ..Default::default()
    };
    // Every worker has been dispatched to: the tally knows each key.
    for _ in 0..64 {
        black_box(policy.select_worker(&workers, &info));
    }
    let decisions = 1_000;
    let mut allocations = 0;
    for _ in 0..decisions {
        feed.publish(&reports);
        allocations += allocations_during(|| {
            black_box(policy.select_worker(&workers, &info));
        });
    }
    // The eligible list is collected from a filter without a size hint: four
    // indices, then eight, two blocks for this pool.
    assert_eq!(
        allocations,
        2 * decisions,
        "least-load decisions allocated {allocations} times over {decisions} decisions; the \
         eligible list (two blocks for eight workers) is the budget"
    );
}
