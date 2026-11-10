//! What the HTTP edge's three outermost layers (the request id, the HTTP
//! metrics, the request span) allocate for one request, pinned at today's
//! count, so the next change that adds a box, a copy or a per-request label
//! to the edge fails a test rather than moving a bench inside its noise. A
//! separate executable, so it can declare the counting allocator the pin
//! needs.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]
#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a request allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    convert::Infallible,
    future::Future,
    pin::pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll, Waker},
};

use axum::{body::Body, extract::Request, response::Response};
use http_body::Body as _;
use smg::{
    middleware::{create_logging_layer, AccessLogFields, HttpMetricsLayer, RequestIdLayer},
    observability::inflight_tracker::InFlightRequestTracker,
};
use smg_external_router::metrics::handles::force_on_this_thread;
use tower::{service_fn, util::BoxCloneSyncService, Service, ServiceBuilder};

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

/// The route the edge wraps, boxed as the router boxes every route (a clone
/// of it is a heap copy of the service inside, as it is in the server); it
/// answers at once with an empty body and counts its calls.
fn route(calls: Arc<AtomicUsize>) -> BoxCloneSyncService<Request, Response, Infallible> {
    BoxCloneSyncService::new(service_fn(move |_request: Request| {
        calls.fetch_add(1, Ordering::Relaxed);
        std::future::ready(Ok::<_, Infallible>(Response::new(Body::empty())))
    }))
}

/// The server's three outermost layers as it stacks them: the request id
/// outside, the HTTP metrics, the request span inside, the access log with
/// no optional field, as it is by default.
fn edge(calls: Arc<AtomicUsize>) -> impl Service<Request, Response = Response, Error = Infallible> {
    ServiceBuilder::new()
        .layer(RequestIdLayer::new(vec!["x-request-id".to_owned()]))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
        .layer(create_logging_layer(AccessLogFields::NONE))
        .service(route(calls))
}

/// One request through `edge`, polled to its answer and to its body's end on
/// this thread (the route answers at once, so no runtime is needed). The
/// edge's own id is on the response.
fn one_request(edge: &mut impl Service<Request, Response = Response, Error = Infallible>) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .body(Body::empty())
        .expect("a request");
    let mut cx = Context::from_waker(Waker::noop());
    assert!(
        matches!(edge.poll_ready(&mut cx), Poll::Ready(Ok(()))),
        "the edge is ready"
    );
    let mut answer = pin!(edge.call(request));
    let response = match answer.as_mut().poll(&mut cx) {
        Poll::Ready(Ok(response)) => Some(response),
        Poll::Ready(Err(_)) | Poll::Pending => None,
    }
    .expect("the route answers at once");
    assert!(
        response.headers()["x-request-id"]
            .as_bytes()
            .starts_with(b"chatcmpl-"),
        "the edge's own id is echoed"
    );
    let mut body = pin!(response.into_body());
    while let Poll::Ready(Some(frame)) = body.as_mut().poll_frame(&mut cx) {
        frame.expect("a frame");
    }
}

/// What a request costs the three layers, with the route's own answer in
/// the count: the fresh id and its header value; the request's extension
/// map (its box and its table) with the id in it, the entry stamp being a
/// task-local of the request, not an entry; the
/// route's empty body and its boxed future; the response's extension map
/// with the duties the span layer hands the edge; the edge's one body
/// wrapper; the response header map's two tables for the echoed id. Nothing
/// per request beyond those: no boxed layer future, no copy of the route a
/// layer clones to call it, no header name built from its text.
///
/// The in-flight tracker's map is sharded and a shard allocates its table
/// on its first insert; which shard a request lands in follows the process's
/// hash seed, so after a warm-up the count is read as the least over a run
/// of requests: a cold shard shows as the odd request above it.
const EDGE_ALLOCATIONS_PER_REQUEST: usize = 14;

#[test]
fn the_edge_s_layers_allocate_this_much_per_request() {
    // The records keep their handles, as they do once the exporter is installed.
    force_on_this_thread(Some(true));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut edge = edge(Arc::clone(&calls));
    // Warm: the kept handles of the request's series exist and the maps that
    // hold them have their tables.
    for _ in 0..512 {
        one_request(&mut edge);
    }
    let least = (0..256)
        .map(|_| allocations_during(|| one_request(&mut edge)).1)
        .min()
        .expect("measured requests");
    force_on_this_thread(None);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        768,
        "the route answered every request"
    );
    eprintln!("allocation calls per request through the edge's three outermost layers: {least}");
    assert_eq!(least, EDGE_ALLOCATIONS_PER_REQUEST);
}
