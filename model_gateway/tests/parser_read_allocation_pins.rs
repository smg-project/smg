//! What a single parser-name read allocates on a card that sets both parser
//! kinds, pinned at today's count: the registry walk's own allocations plus
//! the one String of the kind asked for, so a reader that clones both names
//! and drops one fails a test rather than adding a String per request on the
//! paths that read one kind (the preparation stage's reasoning read, the
//! fallbacks that read the pair). The mock bed's cards set no parser names, so
//! its allocation cell cannot see this path. A separate executable, so it can
//! declare the counting allocator the pin needs.

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
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use smg::{
    routers::grpc::utils::ParserResolver,
    worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
};
use tool_parser::ParserFactory as ToolParserFactory;

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

/// A single read costs the walk over the model's workers and one String, the
/// name of the kind it asked for; the other kind's name is read, not cloned.
#[test]
fn a_single_parser_read_clones_only_its_kind() {
    let registry = registry_with_both_kinds();
    let resolver = ParserResolver::new(
        Arc::clone(&registry),
        None,
        None,
        &ToolParserFactory::new(),
        &ReasoningParserFactory::new(),
    );
    // The first reads: whatever the registry or the resolver builds lazily.
    assert_eq!(resolver.tool_parser("m").as_deref(), Some("json"));
    assert_eq!(resolver.reasoning_parser("m").as_deref(), Some("qwen3"));

    let (workers, walk) = allocations_during(|| registry.get_by_model("m"));
    assert_eq!(workers.len(), 1);
    drop(workers);

    let (tool, calls) = allocations_during(|| resolver.tool_parser("m"));
    assert_eq!(tool.as_deref(), Some("json"));
    assert_eq!(
        calls,
        walk + 1,
        "a tool read costs the walk and its one String: {calls} calls, the walk alone {walk}"
    );
    let (reasoning, calls) = allocations_during(|| resolver.reasoning_parser("m"));
    assert_eq!(reasoning.as_deref(), Some("qwen3"));
    assert_eq!(
        calls,
        walk + 1,
        "a reasoning read costs the walk and its one String: {calls} calls, the walk alone {walk}"
    );
    eprintln!("allocation calls per single parser read on a card with both kinds: {calls} (the walk alone: {walk})");
}
