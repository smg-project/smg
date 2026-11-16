//! What a request's one read of the parser names allocates on a card that
//! sets both kinds, pinned at today's count: the registry walk's own
//! allocations plus the two Strings of the names, once per request (the
//! preparation stage's read), and the two Strings alone where the names were
//! resolved already (request building's read of them: no second walk). The
//! mock bed's cards set no parser names, so its allocation cell cannot see
//! this path. A separate executable, so it can declare the counting allocator
//! the pin needs.

#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a read allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
};

use openai_protocol::model_card::ModelCard;
use smg::{
    routers::grpc::utils::ParserResolver,
    worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
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

/// A registry holding one worker of model `m` whose card sets both parser
/// kinds: the case where a reader of one kind has another name to leave alone.
fn registry_with_both_kinds() -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = BasicWorkerBuilder::new("http://w0:8000")
        .model(
            ModelCard::new("m")
                .with_tool_parser("json")
                .with_reasoning_parser("qwen3"),
        )
        .worker_type(WorkerType::Regular)
        .build();
    registry.register(Arc::new(worker));
    registry
}

/// A request's read of the names costs the walk over the model's workers and
/// the two Strings of the names; a read of names resolved already costs the
/// two Strings and no walk.
#[test]
fn a_requests_read_of_the_names_costs_the_walk_and_two_strings_once() {
    let registry = registry_with_both_kinds();
    let resolver = ParserResolver::new(Arc::clone(&registry), None, None);
    // The first read: whatever the registry or the resolver builds lazily.
    let resolved = resolver.names_for(None, "m").labels;
    assert_eq!(resolved.tool.as_deref(), Some("json"));
    assert_eq!(resolved.reasoning.as_deref(), Some("qwen3"));

    let (workers, walk) = allocations_during(|| registry.get_by_model("m"));
    assert_eq!(workers.len(), 1);
    drop(workers);

    let (names, calls) = allocations_during(|| resolver.names_for(None, "m"));
    assert_eq!(names.labels, resolved);
    assert_eq!(
        calls,
        walk + 2,
        "a read costs the walk and the two Strings: {calls} calls, the walk alone {walk}"
    );
    let (names, calls) = allocations_during(|| resolver.names_for(Some(&resolved), "m"));
    assert_eq!(names.labels, resolved);
    assert_eq!(
        calls, 2,
        "a read of resolved names costs their two Strings and no walk: {calls} calls"
    );
    eprintln!("allocation calls per read of the names on a card with both kinds: {} (the walk alone: {walk})", walk + 2);
}
