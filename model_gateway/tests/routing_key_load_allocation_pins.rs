//! What a request's routing-key load bookkeeping allocates, pinned at the
//! guard's one copy of the key, so the next change that builds a map key
//! per request fails a test rather than moving the keyed request's cost
//! inside a bench's noise: the bookkeeping runs twice per request that
//! carries a sticky key (the sticky-session header, or a rid-derived key),
//! once when the load guard takes the worker's load and once when it
//! releases it. A separate executable, so it can declare the counting
//! allocator the pin needs and install the gateway's own metrics recorder,
//! under the runtime the gateway's request path runs on.

#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a request's bookkeeping allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::{Arc, OnceLock},
    time::Instant,
};

use metrics_exporter_prometheus::PrometheusHandle;
use smg::{
    observability::metrics::{start_prometheus, Metrics, PrometheusConfig},
    worker::{BasicWorkerBuilder, Worker, WorkerLoadGuard, WorkerType},
};

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

/// The gateway's Prometheus recorder, installed once per process as the
/// gateway installs it (the recorder alone: its HTTP server is a task of
/// its own and binds nothing here), so the keyed-load gauge costs what it
/// costs in the gateway.
fn install_recorder() {
    static RECORDER: OnceLock<PrometheusHandle> = OnceLock::new();
    RECORDER.get_or_init(|| start_prometheus(PrometheusConfig::default()));
}

const WORKER_URL: &str = "http://engine-a:8000";
const KEY: &str = "session-a";

fn worker() -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(WORKER_URL)
            .worker_type(WorkerType::Regular)
            .build(),
    )
}

/// One keyed request on the worker: the guard takes the load and releases
/// it, as a request's handler does around the engine call.
fn keyed_request(worker: &Arc<dyn Worker>) {
    drop(WorkerLoadGuard::with_key(Arc::clone(worker), Some(KEY)));
}

/// A keyed request whose key is in flight already, the steady state of a
/// sticky session (every request after its first), allocates once: the
/// guard's own copy of the key. The increment finds the key's count and
/// bumps it in place, the decrement moves it in place too, so neither
/// builds a key of its own; the key's first request adds the map's copy,
/// and nothing more. The keyed-load gauge is set on both, through the
/// gateway's recorder; what one set costs is read beside the requests and
/// taken out, so the pin reads the bookkeeping alone.
#[tokio::test(flavor = "current_thread")]
async fn a_keyed_request_with_its_key_in_flight_allocates_the_guards_copy_alone() {
    install_recorder();
    let worker = worker();
    // The first request of all: the gauge's label is interned and its
    // series registered, the key's shard holds its table.
    keyed_request(&worker);
    let ((), gauge_set) =
        allocations_during(|| Metrics::set_worker_routing_keys_active(WORKER_URL, 1));

    let held = WorkerLoadGuard::with_key(Arc::clone(&worker), Some(KEY));
    let ((), in_flight) = allocations_during(|| keyed_request(&worker));
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-routing-key", KEY.parse().expect("an ASCII key"));
    let ((), from_header) = allocations_during(|| {
        drop(WorkerLoadGuard::new(Arc::clone(&worker), Some(&headers)));
    });
    drop(held);
    // The key's first request, on a shard whose table has room.
    let ((), first_of_key) = allocations_during(|| keyed_request(&worker));
    let ((), unkeyed) =
        allocations_during(|| drop(WorkerLoadGuard::new(Arc::clone(&worker), None)));

    eprintln!(
        "routing-key load: {in_flight} allocation calls per keyed request with its key in flight \
         ({gauge_set} per gauge set, two sets per request), {from_header} with the key read from \
         the header, {first_of_key} for a key's first request, {unkeyed} for an unkeyed request"
    );
    let bookkeeping = |calls: usize| {
        calls
            .checked_sub(2 * gauge_set)
            .expect("a keyed request sets the gauge twice")
    };
    assert_eq!(
        bookkeeping(in_flight),
        1,
        "a keyed request with its key in flight: the guard's copy of the key, no key built for the increment or the decrement"
    );
    assert_eq!(
        bookkeeping(from_header),
        1,
        "a key read from the header is copied once, into the guard"
    );
    assert_eq!(
        bookkeeping(first_of_key),
        2,
        "a key's first request: the guard's copy and the map's"
    );
    assert_eq!(unkeyed, 0, "an unkeyed request touches no map and no gauge");
}

/// The cost of one keyed request's bookkeeping with its key in flight, for
/// the Numbers table: `KEYED_REQUESTS` requests (200,000 by default) through
/// the load guard with the gateway's recorder installed, the allocation
/// calls and the time per request printed; run with `--ignored --nocapture`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "a timing, not a check"]
async fn cost_per_keyed_request() {
    let requests: u64 = std::env::var("KEYED_REQUESTS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(200_000);
    install_recorder();
    let worker = worker();
    let held = WorkerLoadGuard::with_key(Arc::clone(&worker), Some(KEY));
    let start = Instant::now();
    let ((), calls) = allocations_during(|| {
        for _ in 0..requests {
            keyed_request(&worker);
        }
    });
    let elapsed = start.elapsed();
    drop(held);
    eprintln!(
        "keyed request cost: {requests} requests, {:.3} allocation calls and {:.0} ns per request",
        calls as f64 / requests as f64,
        elapsed.as_nanos() as f64 / requests as f64
    );
}
