//! What a local tree insert allocates while the mesh adapter is attached,
//! pinned by shape: an insert for a model whose buffer exists allocates
//! nothing, because the lookup borrows the model id and the owned key is
//! built only for the model's first insert of a round. The counts are
//! printed for the Numbers table. A separate executable, so it can declare
//! the counting allocator.

#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what an insert allocated"
)]
#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
};

use smg::{
    mesh::{
        adapters::tree_sync::{PeerList, RepairEntry, TreeRepairPage},
        TreeDelta, TreeSyncAdapter,
    },
    policies::{TreeHandle, TreeKind},
};
use smg_mesh::{MeshKV, StreamConfig, StreamRouting};

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

/// A tree that holds no path: the adapter's inbound side is not this pin's.
#[derive(Debug)]
struct NoTree;

impl TreeHandle for NoTree {
    fn apply_known_remote_insert(
        &self,
        _model_id: &str,
        _tree_kind: TreeKind,
        _node_hash: u64,
        _worker_url: &str,
    ) -> bool {
        false
    }

    fn open_repair_stream(
        &self,
        _model_id: &str,
        _tree_kind: TreeKind,
    ) -> Option<Box<dyn Iterator<Item = RepairEntry> + Send>> {
        None
    }

    fn apply_repair_page(&self, _page: &TreeRepairPage) -> usize {
        0
    }
}

/// No peer in reach: the adapter's repair side is not this pin's.
#[derive(Debug)]
struct NoPeers;

impl PeerList for NoPeers {
    fn alive_peers(&self) -> Vec<String> {
        Vec::new()
    }
}

/// The adapter as the mesh wiring builds it, over an in-process mesh: the
/// tenant-delta broadcast namespace and the two targeted repair namespaces,
/// the tree and the peer list behind it empty.
fn adapter() -> Arc<TreeSyncAdapter> {
    let mesh = MeshKV::new("node-a".to_string());
    let stream = |prefix: &str, routing: StreamRouting| {
        mesh.configure_stream_prefix(
            prefix,
            StreamConfig {
                max_buffer_bytes: 64 * 1024,
                routing,
            },
        )
    };
    TreeSyncAdapter::new(
        stream("td:", StreamRouting::Broadcast),
        stream("tree:req:", StreamRouting::Targeted),
        stream("tree:page:", StreamRouting::Targeted),
        Arc::new(NoTree),
        Arc::new(NoPeers),
        "node-a".to_string(),
    )
}

/// One delta as the cache-aware policy builds it after a routed request.
fn delta(node_hash: u64) -> TreeDelta {
    TreeDelta {
        tree_kind: TreeKind::String,
        node_hash,
        worker_url: "http://w1".to_string(),
        epoch: 1,
    }
}

/// A model's inserts after its first find the buffer by a borrowed lookup
/// and allocate nothing (the delta is built before the measured call; the
/// buffer has room from its first growth). The first insert builds the key
/// and the buffer; its count is printed beside the pinned one.
#[test]
fn an_insert_into_an_existing_buffer_allocates_nothing() {
    let adapter = adapter();
    let first = delta(1);
    let ((), cold) = allocations_during(|| adapter.on_local_insert("model-1", first));
    let second = delta(2);
    let ((), warm) = allocations_during(|| adapter.on_local_insert("model-1", second));
    eprintln!(
        "allocation calls per local tree insert: the model's first {cold}, an insert into its existing buffer {warm}"
    );
    assert_eq!(
        warm, 0,
        "an insert into an existing buffer borrows the model id: the owned key is built only for the model's first insert"
    );
}
