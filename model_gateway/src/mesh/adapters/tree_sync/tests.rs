use smg_mesh::{MeshKV, StreamConfig, StreamRouting};

use super::*;

fn td_namespace(mesh: &MeshKV) -> Arc<StreamNamespace> {
    mesh.configure_stream_prefix(
        PREFIX,
        StreamConfig {
            max_buffer_bytes: 64 * 1024,
            routing: StreamRouting::Broadcast,
        },
    )
}

fn req_namespace(mesh: &MeshKV) -> Arc<StreamNamespace> {
    mesh.configure_stream_prefix(
        REPAIR_REQUEST_PREFIX,
        StreamConfig {
            max_buffer_bytes: 64 * 1024,
            routing: StreamRouting::Targeted,
        },
    )
}

fn page_namespace(mesh: &MeshKV) -> Arc<StreamNamespace> {
    mesh.configure_stream_prefix(
        REPAIR_PAGE_PREFIX,
        StreamConfig {
            // Generous so tests with multi-MB pages don't
            // FIFO-evict before assertions inspect them.
            max_buffer_bytes: 8 * 1024 * 1024,
            routing: StreamRouting::Targeted,
        },
    )
}

fn delta(hash: u64, worker: &str) -> TreeDelta {
    TreeDelta {
        tree_kind: TreeKind::String,
        node_hash: hash,
        worker_url: worker.into(),
        epoch: 1,
    }
}

/// Test-only [`TreeHandle`]: a set of known `(model, kind,
/// hash)` tuples plus a log of every applied call so tests
/// can assert on apply-side effects. Apply succeeds (returns
/// `true`) iff the hash is in the known set. `streams` lets
/// d-2b tests pre-load `RepairEntry` sequences for
/// `open_repair_stream` lookups.
#[derive(Debug, Default)]
struct MockTreeHandle {
    known: DashMap<(String, TreeKind, u64), ()>,
    applied: parking_lot::Mutex<Vec<(String, TreeKind, u64, String)>>,
    streams: DashMap<(String, TreeKind), Vec<RepairEntry>>,
    /// Pages received via `apply_repair_page` (d-2c). Tests
    /// use this to assert what the receiver actually applied.
    applied_pages: parking_lot::Mutex<Vec<TreeRepairPage>>,
}

impl MockTreeHandle {
    fn mark_known(&self, model_id: &str, tree_kind: TreeKind, node_hash: u64) {
        self.known
            .insert((model_id.to_string(), tree_kind, node_hash), ());
    }

    fn applied_calls(&self) -> Vec<(String, TreeKind, u64, String)> {
        self.applied.lock().clone()
    }

    fn set_repair_stream(&self, model_id: &str, tree_kind: TreeKind, entries: Vec<RepairEntry>) {
        self.streams
            .insert((model_id.to_string(), tree_kind), entries);
    }
}

impl TreeHandle for MockTreeHandle {
    fn apply_known_remote_insert(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
        node_hash: u64,
        worker_url: &str,
    ) -> bool {
        if !self
            .known
            .contains_key(&(model_id.to_string(), tree_kind, node_hash))
        {
            return false;
        }
        self.applied.lock().push((
            model_id.to_string(),
            tree_kind,
            node_hash,
            worker_url.to_string(),
        ));
        true
    }

    fn open_repair_stream(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
    ) -> Option<Box<dyn Iterator<Item = RepairEntry> + Send>> {
        let entries = self.streams.get(&(model_id.to_string(), tree_kind))?;
        Some(Box::new(entries.value().clone().into_iter()))
    }

    fn apply_repair_page(&self, page: &TreeRepairPage) -> usize {
        // Variant-matching count: only entries whose variant
        // matches `page.tree_kind` count toward the applied
        // total — mirroring the production impl's semantics.
        let applied = page
            .entries
            .iter()
            .filter(|e| {
                matches!(
                    (page.tree_kind, e),
                    (TreeKind::String, RepairEntry::String { .. })
                        | (TreeKind::Token, RepairEntry::Token { .. })
                )
            })
            .count();
        self.applied_pages.lock().push(page.clone());
        applied
    }
}

fn empty_handle() -> Arc<MockTreeHandle> {
    Arc::new(MockTreeHandle::default())
}

/// Test-only [`PeerList`] returning a fixed list of peers.
#[derive(Debug, Default)]
struct MockPeerList {
    peers: parking_lot::Mutex<Vec<String>>,
}

impl MockPeerList {
    fn with(peers: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            peers: parking_lot::Mutex::new(peers.iter().map(|s| (*s).into()).collect()),
        })
    }
}

impl PeerList for MockPeerList {
    fn alive_peers(&self) -> Vec<String> {
        self.peers.lock().clone()
    }
}

fn empty_peers() -> Arc<MockPeerList> {
    Arc::new(MockPeerList::default())
}

fn adapter_with_empty_handle(mesh: &MeshKV, node_name: &str) -> Arc<TreeSyncAdapter> {
    let td = td_namespace(mesh);
    let req = req_namespace(mesh);
    let pages = page_namespace(mesh);
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = empty_peers();
    TreeSyncAdapter::new(td, req, pages, tree, peers, node_name.into())
}

#[tokio::test]
async fn tree_delta_bincode_round_trip() {
    let batch = vec![
        TreeDelta {
            tree_kind: TreeKind::String,
            node_hash: 7,
            worker_url: "http://w1".into(),
            epoch: 42,
        },
        TreeDelta {
            tree_kind: TreeKind::Token,
            node_hash: u64::MAX,
            worker_url: "http://w2".into(),
            epoch: 0,
        },
    ];
    let bytes = bincode::serialize(&batch).unwrap();
    let decoded: Vec<TreeDelta> = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded, batch);
}

#[tokio::test]
async fn on_local_insert_buffers_per_model() {
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");

    adapter.on_local_insert("model-1", delta(1, "http://w1"));
    adapter.on_local_insert("model-1", delta(2, "http://w1"));
    adapter.on_local_insert("model-2", delta(3, "http://w2"));

    assert_eq!(adapter.pending_deltas.get("model-1").unwrap().len(), 2);
    assert_eq!(adapter.pending_deltas.get("model-2").unwrap().len(), 1);
}

#[tokio::test]
async fn drain_batches_per_model_and_clears_buffer() {
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");

    adapter.on_local_insert("model-1", delta(1, "http://w1"));
    adapter.on_local_insert("model-1", delta(2, "http://w1"));
    adapter.on_local_insert("model-2", delta(3, "http://w2"));

    let entries = adapter.drain_pending_deltas();
    assert_eq!(entries.len(), 2, "one batch per model");

    // Each batch round-trips into the original per-model deltas.
    let mut by_key: std::collections::HashMap<String, Vec<TreeDelta>> =
        std::collections::HashMap::new();
    for (key, bytes) in entries {
        let batch: Vec<TreeDelta> = bincode::deserialize(&bytes).unwrap();
        by_key.insert(key, batch);
    }
    assert_eq!(by_key.get("td:model-1").unwrap().len(), 2);
    assert_eq!(by_key.get("td:model-2").unwrap().len(), 1);

    // Buffer is emptied on drain so the next round starts fresh.
    assert!(adapter.pending_deltas.is_empty());
}

#[tokio::test]
async fn drain_skips_empty_model_buffers() {
    // Cleared model buckets must not emit empty batches.
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");

    adapter
        .pending_deltas
        .insert("model-empty".into(), Vec::new());

    let entries = adapter.drain_pending_deltas();
    assert!(entries.is_empty());
}

#[tokio::test]
async fn start_registers_drain_with_mesh_round_collector() {
    // End-to-end outbound: start → drain registration →
    // collect_round_batch pulls our entries.
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");
    adapter.start();

    adapter.on_local_insert("model-1", delta(10, "http://w1"));
    adapter.on_local_insert("model-2", delta(20, "http://w2"));

    let round = mesh.collect_round_batch();
    let keys: std::collections::HashSet<String> =
        round.drain_entries.iter().map(|(k, _)| k.clone()).collect();
    assert!(keys.contains("td:model-1"));
    assert!(keys.contains("td:model-2"));
}

#[tokio::test]
async fn drain_closure_uses_weak_reference() {
    // Dropping the last strong Arc must actually drop the
    // adapter; a strong Arc in the drain closure would cycle
    // through DrainRegistry and leak it.
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");
    adapter.start();

    let weak = Arc::downgrade(&adapter);
    drop(adapter);
    assert!(
        weak.upgrade().is_none(),
        "drain closure must not strongly hold the adapter",
    );

    // Drain is now a no-op; round produces no td: entries.
    let round = mesh.collect_round_batch();
    let td_entries: Vec<_> = round
        .drain_entries
        .iter()
        .filter(|(k, _)| k.starts_with("td:"))
        .collect();
    assert!(td_entries.is_empty());
}

#[tokio::test]
async fn handle_incoming_batch_applies_known_and_skips_unknown() {
    // Known deltas hit `apply_known_remote_insert`; unknown
    // ones don't (and trigger repair, exercised in dedicated
    // tests below). Kinds must not alias: same hash on a
    // different `TreeKind` is a different (model, kind, hash)
    // tuple and stays unknown.
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    tree.mark_known("model-1", TreeKind::String, 42);
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let batch = vec![
        TreeDelta {
            tree_kind: TreeKind::String,
            node_hash: 42, // known
            worker_url: "http://w1".into(),
            epoch: 1,
        },
        TreeDelta {
            tree_kind: TreeKind::String,
            node_hash: 99, // unknown
            worker_url: "http://w2".into(),
            epoch: 1,
        },
        TreeDelta {
            tree_kind: TreeKind::Token,
            node_hash: 42, // same hash, different kind — must not alias
            worker_url: "http://w3".into(),
            epoch: 1,
        },
    ];
    let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
    adapter.handle_incoming_batch("model-1", &[bytes]);

    let applied = tree.applied_calls();
    assert_eq!(
        applied,
        vec![(
            "model-1".to_string(),
            TreeKind::String,
            42,
            "http://w1".to_string(),
        )],
        "only the (String, 42) delta applied; unknown String:99 and Token:42 did not",
    );
}

#[tokio::test]
async fn handle_incoming_batch_ignores_malformed_payload() {
    // Corrupt batch → no propagation, no panic.
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");

    adapter.handle_incoming_batch("model-1", &[Bytes::from_static(b"not-bincode")]);
}

#[tokio::test]
#[should_panic(expected = "TreeSyncAdapter requires a tenant-delta namespace scoped to `td:`")]
async fn new_rejects_wrong_td_prefix() {
    // Pass the repair-request namespace as the td: arg.
    let mesh = MeshKV::new("node-a".into());
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let req_again = mesh.configure_stream_prefix(
        "td-misnamed:",
        StreamConfig {
            max_buffer_bytes: 64 * 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = empty_peers();
    let _ = TreeSyncAdapter::new(req_again, req, pages, tree, peers, "node-a".into());
}

#[tokio::test]
#[should_panic(
    expected = "TreeSyncAdapter requires a repair-request namespace scoped to `tree:req:`"
)]
async fn new_rejects_wrong_repair_prefix() {
    // Pass the td: namespace as the repair-request arg.
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let td_again = mesh.configure_stream_prefix(
        "td-also:",
        StreamConfig {
            max_buffer_bytes: 64 * 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = empty_peers();
    let _ = TreeSyncAdapter::new(td, td_again, pages, tree, peers, "node-a".into());
}

#[tokio::test]
#[should_panic(expected = "node_name must not be empty")]
async fn new_rejects_empty_node_name() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = empty_peers();
    let _ = TreeSyncAdapter::new(td, req, pages, tree, peers, String::new());
}

#[tokio::test]
#[should_panic(expected = "drain already registered for prefix 'td:'")]
async fn start_is_fused() {
    // Second start must panic at the mesh's
    // one-drain-per-prefix invariant.
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");
    adapter.start();
    adapter.start();
}

#[tokio::test]
async fn tree_repair_request_bincode_round_trip() {
    let req = TreeRepairRequest {
        session_id: Uuid::now_v7(),
        requester_peer_id: "node-a".into(),
        target_peer_id: "node-b".into(),
        model_id: "model-1".into(),
        tree_kind: TreeKind::Token,
        cursor: Some(vec![1, 2, 3]),
        reason: RepairReason::UnknownHash(42),
    };
    let bytes = bincode::serialize(&req).unwrap();
    let decoded: TreeRepairRequest = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded, req);
}

/// Drain the targeted-stream side buffer for repair requests
/// so we can inspect what the adapter published.
fn drain_repair_publishes(mesh: &MeshKV) -> Vec<(String, String, Bytes)> {
    let round = mesh.collect_round_batch();
    round
        .targeted_entries
        .into_iter()
        .filter(|(_, key, _)| key.starts_with(REPAIR_REQUEST_PREFIX))
        .collect()
}

#[tokio::test]
async fn unknown_hash_publishes_repair_request_to_alive_peer() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree: Arc<dyn TreeHandle> = Arc::new(MockTreeHandle::default());
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, tree, peers, "node-a".into());

    // Single unknown delta → one targeted publish to node-b.
    let batch = vec![delta(99, "http://w1")];
    let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
    adapter.handle_incoming_batch("model-1", &[bytes]);

    let publishes = drain_repair_publishes(&mesh);
    assert_eq!(publishes.len(), 1, "exactly one repair request");
    let (target, key, payload) = &publishes[0];
    assert_eq!(target, "node-b");
    assert!(key.starts_with(REPAIR_REQUEST_PREFIX));

    let request: TreeRepairRequest = bincode::deserialize(payload).unwrap();
    assert_eq!(request.requester_peer_id, "node-a");
    assert_eq!(request.target_peer_id, "node-b");
    assert_eq!(request.model_id, "model-1");
    assert_eq!(request.tree_kind, TreeKind::String);
    assert_eq!(request.reason, RepairReason::UnknownHash(99));
    assert!(request.cursor.is_none());

    // Outstanding-session bookkeeping records the in-flight key.
    assert!(adapter
        .outstanding_repairs
        .contains_key(&("model-1".to_string(), TreeKind::String)),);
}

#[tokio::test]
async fn coalesces_duplicate_repair_for_same_model_kind() {
    // Two unknown hashes for the same (model, kind) within one
    // batch must produce only one repair request — the
    // in-flight session covers both.
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree: Arc<dyn TreeHandle> = Arc::new(MockTreeHandle::default());
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, tree, peers, "node-a".into());

    let batch = vec![delta(101, "http://w1"), delta(102, "http://w2")];
    let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
    adapter.handle_incoming_batch("model-1", &[bytes]);

    let publishes = drain_repair_publishes(&mesh);
    assert_eq!(publishes.len(), 1, "duplicate request should coalesce");
}

#[tokio::test]
async fn separate_kinds_get_separate_repair_sessions() {
    // String and Token unknown hashes for the same model are
    // distinct sessions — coalescing keys on (model, kind).
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree: Arc<dyn TreeHandle> = Arc::new(MockTreeHandle::default());
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, tree, peers, "node-a".into());

    let batch = vec![
        TreeDelta {
            tree_kind: TreeKind::String,
            node_hash: 1,
            worker_url: "http://w1".into(),
            epoch: 1,
        },
        TreeDelta {
            tree_kind: TreeKind::Token,
            node_hash: 1,
            worker_url: "http://w2".into(),
            epoch: 1,
        },
    ];
    let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
    adapter.handle_incoming_batch("model-1", &[bytes]);

    let publishes = drain_repair_publishes(&mesh);
    assert_eq!(publishes.len(), 2, "one per (model, kind) pair");
}

#[tokio::test]
async fn no_alive_peers_skips_repair_silently() {
    // Empty peer list → no publish, no panic, and no entry
    // recorded so a future delta retries once peers come up.
    let mesh = MeshKV::new("node-a".into());
    let adapter = adapter_with_empty_handle(&mesh, "node-a");

    let batch = vec![delta(99, "http://w1")];
    let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
    adapter.handle_incoming_batch("model-1", &[bytes]);

    let publishes = drain_repair_publishes(&mesh);
    assert!(publishes.is_empty(), "no peers means nothing published");
    assert!(
        adapter.outstanding_repairs.is_empty(),
        "no session recorded so the next delta can retry",
    );
}

#[tokio::test]
async fn known_hashes_do_not_trigger_repair() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    tree.mark_known("model-1", TreeKind::String, 42);
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let batch = vec![delta(42, "http://w1")];
    let bytes = Bytes::from(bincode::serialize(&batch).unwrap());
    adapter.handle_incoming_batch("model-1", &[bytes]);

    let publishes = drain_repair_publishes(&mesh);
    assert!(publishes.is_empty(), "known hash skips repair entirely");
}

// ----- d-2b: repair page wire format + responder -----

fn string_entry(path: &str, tenants: &[(&str, u64)]) -> RepairEntry {
    RepairEntry::String {
        path: path.into(),
        tenants: tenants.iter().map(|(t, e)| (Arc::from(*t), *e)).collect(),
    }
}

fn token_entry(tokens: &[u32], tenants: &[(&str, u64)]) -> RepairEntry {
    RepairEntry::Token {
        tokens: tokens.to_vec(),
        tenants: tenants.iter().map(|(t, e)| (Arc::from(*t), *e)).collect(),
    }
}

fn drain_page_publishes(mesh: &MeshKV) -> Vec<(String, String, Bytes)> {
    let round = mesh.collect_round_batch();
    round
        .targeted_entries
        .into_iter()
        .filter(|(_, key, _)| key.starts_with(REPAIR_PAGE_PREFIX))
        .collect()
}

fn make_request(
    session_id: Uuid,
    target: &str,
    requester: &str,
    model_id: &str,
    kind: TreeKind,
    cursor: Option<Vec<u8>>,
) -> TreeRepairRequest {
    TreeRepairRequest {
        session_id,
        requester_peer_id: requester.into(),
        target_peer_id: target.into(),
        model_id: model_id.into(),
        tree_kind: kind,
        cursor,
        reason: RepairReason::UnknownHash(0xdead_beef),
    }
}

#[tokio::test]
async fn repair_entry_and_page_round_trip_through_bincode() {
    let entries = vec![
        string_entry("hello", &[("worker-1", 7)]),
        string_entry("hello world", &[("worker-1", 8), ("worker-2", 9)]),
    ];
    let page = TreeRepairPage {
        session_id: Uuid::now_v7(),
        model_id: "model-1".into(),
        tree_kind: TreeKind::String,
        page_index: 3,
        entries: entries.clone(),
        next_cursor: Some(vec![1, 2, 3]),
        is_last: false,
    };
    let bytes = bincode::serialize(&page).unwrap();
    let decoded: TreeRepairPage = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded, page);

    // Token variant round-trips too.
    let tk_entries = vec![token_entry(&[1, 2, 3, 4], &[("worker-1", 0)])];
    let tk_page = TreeRepairPage {
        session_id: Uuid::now_v7(),
        model_id: "model-2".into(),
        tree_kind: TreeKind::Token,
        page_index: 0,
        entries: tk_entries.clone(),
        next_cursor: None,
        is_last: true,
    };
    let bytes = bincode::serialize(&tk_page).unwrap();
    let decoded: TreeRepairPage = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded, tk_page);
}

#[tokio::test]
async fn paging_drops_entries_above_wire_ceiling() {
    // Entry serialized size > REPAIR_ENTRY_HARD_CEILING must be
    // dropped, not emitted — otherwise the mesh chunking layer
    // would split the page across wire frames and the spec
    // forbids `tree:page:*` from entering that path.
    let huge_path: String = "x".repeat(REPAIR_ENTRY_HARD_CEILING + 1024);
    let small = string_entry("small", &[("w1", 1)]);
    let entries: Vec<RepairEntry> = vec![
        RepairEntry::String {
            path: huge_path,
            tenants: vec![(Arc::from("w1"), 0)],
        },
        small.clone(),
    ];

    let pager = PagingIter::new(
        Box::new(entries.into_iter()),
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        TREE_REPAIR_PAGE_BYTE_CAP,
        0,
    );
    let pages: Vec<_> = pager.collect();
    // The huge entry is dropped; only `small` survives. We
    // should see exactly one page (the terminal one) carrying
    // just the small entry.
    let surviving: Vec<RepairEntry> = pages.iter().flat_map(|p| p.entries.clone()).collect();
    assert_eq!(surviving, vec![small], "huge entry dropped, small survives");
    assert!(pages.last().unwrap().is_last);
}

#[tokio::test]
async fn paging_pages_never_exceed_wire_ceiling() {
    // Invariant test: no `TreeRepairPage` produced by paging
    // serializes above `TREE_REPAIR_PAGE_HARD_CEILING`. Mix
    // entries at three sizes — small, between soft and hard,
    // and above hard — and assert every emitted page fits.
    let between = "y".repeat(TREE_REPAIR_PAGE_BYTE_CAP + 16 * 1024); // ~2 MB + 16 KB
    let above = "z".repeat(REPAIR_ENTRY_HARD_CEILING + 1024);

    let entries: Vec<RepairEntry> = vec![
        string_entry("small-1", &[("w1", 1)]),
        RepairEntry::String {
            path: between,
            tenants: vec![(Arc::from("w1"), 0)],
        },
        string_entry("small-2", &[("w1", 2)]),
        RepairEntry::String {
            path: above,
            tenants: vec![(Arc::from("w1"), 0)],
        },
        string_entry("small-3", &[("w1", 3)]),
    ];

    let pager = PagingIter::new(
        Box::new(entries.into_iter()),
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        TREE_REPAIR_PAGE_BYTE_CAP,
        0,
    );
    let pages: Vec<_> = pager.collect();
    for (i, p) in pages.iter().enumerate() {
        let size = bincode::serialized_size(p).unwrap() as usize;
        assert!(
            size <= TREE_REPAIR_PAGE_HARD_CEILING,
            "page {i} (size {size}) exceeds hard wire ceiling \
                 ({TREE_REPAIR_PAGE_HARD_CEILING}) — would invoke \
                 multi-chunk path forbidden for tree:page:*",
        );
    }
}

#[tokio::test]
async fn paging_empty_stream_emits_one_is_last_page() {
    // Empty input → one page so the requester sees an ACK.
    let stream: Box<dyn Iterator<Item = RepairEntry> + Send> = Box::new(std::iter::empty());
    let pager = PagingIter::new(
        stream,
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        TREE_REPAIR_PAGE_BYTE_CAP,
        0,
    );
    let pages: Vec<_> = pager.collect();
    assert_eq!(pages.len(), 1);
    assert!(pages[0].is_last);
    assert!(pages[0].entries.is_empty());
    assert!(pages[0].next_cursor.is_none());
    assert_eq!(pages[0].page_index, 0);
}

#[tokio::test]
async fn paging_chunks_by_byte_cap_and_marks_last() {
    // Build entries large enough that a tiny cap forces
    // multiple pages. Use ~1 KB string paths and a 4 KB
    // entry budget so each page holds ~3-4 entries.
    let big_path: String = "a".repeat(1024);
    let entries: Vec<RepairEntry> = (0..12)
        .map(|i| RepairEntry::String {
            path: format!("{big_path}-{i}"),
            tenants: vec![(Arc::from("worker-1"), i as u64)],
        })
        .collect();
    let total = entries.len();

    // 4 KB cap; entry size ~1 KB + bincode overhead.
    let pager = PagingIter::new(
        Box::new(entries.clone().into_iter()),
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        4 * 1024,
        0,
    );
    let pages: Vec<_> = pager.collect();
    assert!(pages.len() >= 2, "small cap must produce multiple pages");

    // Total entries preserved; only the final page is marked
    // last; each non-last page carries a non-None cursor.
    let total_entries: usize = pages.iter().map(|p| p.entries.len()).sum();
    assert_eq!(total_entries, total);
    for (i, p) in pages.iter().enumerate() {
        let is_last = i == pages.len() - 1;
        assert_eq!(p.is_last, is_last, "page {i} is_last");
        assert_eq!(p.next_cursor.is_none(), is_last, "page {i} cursor parity");
        assert_eq!(p.page_index, i as u32, "page index sequential");
        // Every page that's not the last must have at least
        // one entry (we always pack at least one per page).
        if !is_last {
            assert!(!p.entries.is_empty(), "non-last page must be non-empty");
        }
    }

    // Each non-last page's serialized bincode size stays
    // close to the cap once header overhead is added back.
    for p in &pages {
        let size = bincode::serialized_size(p).unwrap() as usize;
        // Final page may be smaller; just assert non-final
        // pages don't exceed `cap + slop` for the slop budget
        // (header overhead already reserved).
        assert!(
            size <= 4 * 1024 + TREE_REPAIR_PAGE_HEADER_OVERHEAD,
            "page size {size} exceeds cap + header budget",
        );
    }
}

#[tokio::test]
async fn paging_oversized_single_entry_still_emits_one_per_page() {
    // A single entry larger than the cap must still go out
    // — we never produce an empty non-last page.
    let huge: RepairEntry = RepairEntry::String {
        path: "x".repeat(8 * 1024),
        tenants: vec![(Arc::from("w1"), 0)],
    };
    let pager = PagingIter::new(
        Box::new(vec![huge].into_iter()),
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        1024, // cap < entry size
        0,
    );
    let pages: Vec<_> = pager.collect();
    assert_eq!(pages.len(), 1, "single entry yields exactly one page");
    assert!(pages[0].is_last);
    assert_eq!(pages[0].entries.len(), 1);
}

#[tokio::test]
async fn paging_cursor_resume_yields_same_total() {
    let big_path: String = "p".repeat(1024);
    let entries: Vec<RepairEntry> = (0..10)
        .map(|i| RepairEntry::String {
            path: format!("{big_path}-{i}"),
            tenants: vec![(Arc::from("w1"), i as u64)],
        })
        .collect();

    // Full pass.
    let full: Vec<RepairEntry> = PagingIter::new(
        Box::new(entries.clone().into_iter()),
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        4 * 1024,
        0,
    )
    .flat_map(|p| p.entries)
    .collect();
    assert_eq!(full.len(), entries.len());
    assert_eq!(full, entries);

    // Resume halfway: skip 5 → final pass yields the last 5.
    let resumed: Vec<RepairEntry> = PagingIter::new(
        Box::new(entries.clone().into_iter()),
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        4 * 1024,
        5,
    )
    .flat_map(|p| p.entries)
    .collect();
    assert_eq!(resumed.len(), 5);
    assert_eq!(resumed, entries[5..]);
}

#[tokio::test]
async fn paging_cursor_encodes_cumulative_count() {
    // Each non-last page's `next_cursor` must decode to the
    // total items consumed so far — usable as a resume key.
    let big_path: String = "c".repeat(1024);
    let entries: Vec<RepairEntry> = (0..8)
        .map(|i| RepairEntry::String {
            path: format!("{big_path}-{i}"),
            tenants: vec![(Arc::from("w1"), i as u64)],
        })
        .collect();
    let pager = PagingIter::new(
        Box::new(entries.into_iter()),
        Uuid::now_v7(),
        "m".into(),
        TreeKind::String,
        4 * 1024,
        0,
    );
    let pages: Vec<_> = pager.collect();
    let mut running_total = 0usize;
    for (i, p) in pages.iter().enumerate() {
        running_total += p.entries.len();
        if i < pages.len() - 1 {
            let cursor_bytes = p.next_cursor.as_ref().expect("non-last has cursor");
            let decoded: u64 = bincode::deserialize(cursor_bytes).unwrap();
            assert_eq!(decoded as usize, running_total, "page {i} cursor matches");
        }
    }
}

#[tokio::test]
async fn responder_publishes_pages_targeted_at_requester() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    tree.set_repair_stream(
        "model-1",
        TreeKind::String,
        vec![
            string_entry("hello", &[("w1", 1)]),
            string_entry("hello world", &[("w1", 2), ("w2", 3)]),
        ],
    );
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session_id = Uuid::now_v7();
    let request = make_request(
        session_id,
        "node-a", // targeted at us
        "node-b", // requester
        "model-1",
        TreeKind::String,
        None,
    );
    adapter.respond_to_repair_request(request);

    let publishes = drain_page_publishes(&mesh);
    assert!(
        !publishes.is_empty(),
        "responder must publish at least one page"
    );
    for (target, key, _payload) in &publishes {
        assert_eq!(target, "node-b", "all pages targeted at requester");
        let expected_prefix = format!("{REPAIR_PAGE_PREFIX}{session_id}:");
        assert!(
            key.starts_with(&expected_prefix),
            "key {key} matches tree:page:{session_id}:N",
        );
    }

    // Reconstruct the page sequence and verify entries cover
    // the input set.
    let mut all_entries: Vec<RepairEntry> = Vec::new();
    for (_, _, payload) in &publishes {
        let page: TreeRepairPage = bincode::deserialize(payload).unwrap();
        all_entries.extend(page.entries);
    }
    assert_eq!(all_entries.len(), 2);
}

#[tokio::test]
async fn responder_emits_empty_ack_for_unknown_model() {
    // Same shape as "tree exists but is empty" — one is_last
    // page with no entries — gives the requester an ACK.
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, tree, peers, "node-a".into());

    let request = make_request(
        Uuid::now_v7(),
        "node-a",
        "node-b",
        "unknown-model",
        TreeKind::String,
        None,
    );
    adapter.respond_to_repair_request(request);

    let publishes = drain_page_publishes(&mesh);
    assert_eq!(publishes.len(), 1, "exactly one ACK page");
    let page: TreeRepairPage = bincode::deserialize(&publishes[0].2).unwrap();
    assert!(page.is_last);
    assert!(page.entries.is_empty());
    assert!(page.next_cursor.is_none());
}

#[tokio::test]
async fn responder_drops_request_with_mismatched_target() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    tree.set_repair_stream(
        "model-1",
        TreeKind::String,
        vec![string_entry("p", &[("w1", 1)])],
    );
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    // Target is some other peer — drop, no spawn, no publish.
    let request = make_request(
        Uuid::now_v7(),
        "node-elsewhere",
        "node-b",
        "model-1",
        TreeKind::String,
        None,
    );
    let bytes = bincode::serialize(&request).unwrap();
    adapter.handle_incoming_repair_request(&[Bytes::from(bytes)]);
    // Yield once to let any spurious spawn observe state;
    // none is expected.
    tokio::task::yield_now().await;

    let publishes = drain_page_publishes(&mesh);
    assert!(
        publishes.is_empty(),
        "mismatched target_peer_id must drop the request",
    );
}

#[tokio::test]
async fn responder_handles_two_distinct_sessions_independently() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    tree.set_repair_stream(
        "model-1",
        TreeKind::String,
        vec![string_entry("a", &[("w1", 1)])],
    );
    tree.set_repair_stream(
        "model-2",
        TreeKind::Token,
        vec![token_entry(&[1, 2, 3], &[("w2", 2)])],
    );
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session_a = Uuid::now_v7();
    let session_b = Uuid::now_v7();
    adapter.respond_to_repair_request(make_request(
        session_a,
        "node-a",
        "node-b",
        "model-1",
        TreeKind::String,
        None,
    ));
    adapter.respond_to_repair_request(make_request(
        session_b,
        "node-a",
        "node-c",
        "model-2",
        TreeKind::Token,
        None,
    ));

    let publishes = drain_page_publishes(&mesh);
    let mut by_session: std::collections::HashMap<Uuid, Vec<TreeRepairPage>> =
        std::collections::HashMap::new();
    for (_, _, payload) in &publishes {
        let page: TreeRepairPage = bincode::deserialize(payload).unwrap();
        by_session.entry(page.session_id).or_default().push(page);
    }
    assert!(by_session.contains_key(&session_a));
    assert!(by_session.contains_key(&session_b));
    // Each session ends with exactly one is_last=true page.
    for pages in by_session.values() {
        let last = pages.iter().filter(|p| p.is_last).count();
        assert_eq!(last, 1, "each session has exactly one terminal page");
    }
}

#[tokio::test]
async fn responder_resumes_from_request_cursor() {
    let big_path: String = "r".repeat(1024);
    let canned: Vec<RepairEntry> = (0..6)
        .map(|i| RepairEntry::String {
            path: format!("{big_path}-{i}"),
            tenants: vec![(Arc::from("w1"), i as u64)],
        })
        .collect();
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    tree.set_repair_stream("model-1", TreeKind::String, canned.clone());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    // Resume at offset 4 → response covers the last 2 entries.
    let cursor = bincode::serialize(&4u64).unwrap();
    adapter.respond_to_repair_request(make_request(
        Uuid::now_v7(),
        "node-a",
        "node-b",
        "model-1",
        TreeKind::String,
        Some(cursor),
    ));

    let publishes = drain_page_publishes(&mesh);
    let mut all_entries: Vec<RepairEntry> = Vec::new();
    for (_, _, payload) in &publishes {
        let page: TreeRepairPage = bincode::deserialize(payload).unwrap();
        all_entries.extend(page.entries);
    }
    assert_eq!(all_entries.len(), 2);
    assert_eq!(all_entries, canned[4..]);
}

#[tokio::test]
#[should_panic(
    expected = "TreeSyncAdapter requires a repair-page namespace scoped to `tree:page:`"
)]
async fn new_rejects_wrong_pages_prefix() {
    // Pass the repair-request namespace as the pages arg.
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let req_again = mesh.configure_stream_prefix(
        "tree:page-misnamed:",
        StreamConfig {
            max_buffer_bytes: 64 * 1024,
            routing: StreamRouting::Targeted,
        },
    );
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = empty_peers();
    let _ = TreeSyncAdapter::new(td, req, req_again, tree, peers, "node-a".into());
}

// ----- d-2c: receiver side of the repair protocol -----

fn make_page(
    session_id: Uuid,
    model_id: &str,
    tree_kind: TreeKind,
    page_index: u32,
    entries: Vec<RepairEntry>,
    next_cursor: Option<u64>,
    is_last: bool,
) -> TreeRepairPage {
    TreeRepairPage {
        session_id,
        model_id: model_id.into(),
        tree_kind,
        page_index,
        entries,
        next_cursor: next_cursor.map(|n| bincode::serialize(&n).unwrap()),
        is_last,
    }
}

fn page_bytes(page: &TreeRepairPage) -> Bytes {
    Bytes::from(bincode::serialize(page).unwrap())
}

fn seed_outstanding(
    adapter: &TreeSyncAdapter,
    model_id: &str,
    tree_kind: TreeKind,
    session_id: Uuid,
    target: &str,
) {
    adapter.outstanding_repairs.insert(
        (model_id.to_string(), tree_kind),
        RepairProgress::new(session_id, target.into(), Instant::now()),
    );
}

fn applied_pages(tree: &MockTreeHandle) -> Vec<TreeRepairPage> {
    tree.applied_pages.lock().clone()
}

#[tokio::test]
async fn receiver_applies_page_and_records_session_progress() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session = Uuid::now_v7();
    seed_outstanding(&adapter, "model-1", TreeKind::String, session, "node-b");

    let page = make_page(
        session,
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("p", &[("w1", 7)])],
        Some(1),
        false,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);

    // Page was applied via TreeHandle.
    let calls = applied_pages(&tree);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].session_id, session);
    assert_eq!(calls[0].page_index, 0);

    // Session progress recorded with cursor and no terminal
    // page observed yet.
    let entry = adapter
        .outstanding_repairs
        .get(&("model-1".to_string(), TreeKind::String))
        .expect("session present");
    assert!(entry.applied.contains_key(&0));
    assert_eq!(
        entry.contiguous_cursor(),
        Some(bincode::serialize(&1u64).unwrap())
    );
    assert_eq!(entry.terminal_page_index, None);
    assert!(!entry.is_contiguously_complete());
}

#[tokio::test]
async fn receiver_dedupes_same_page_index() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session = Uuid::now_v7();
    seed_outstanding(&adapter, "model-1", TreeKind::String, session, "node-b");
    let page = make_page(
        session,
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("p", &[("w1", 1)])],
        Some(1),
        false,
    );

    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);
    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);

    assert_eq!(
        applied_pages(&tree).len(),
        1,
        "duplicate page must not be applied twice",
    );
}

#[tokio::test]
async fn receiver_drops_page_for_unknown_session() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    // No outstanding session — page should be dropped.
    let page = make_page(
        Uuid::now_v7(),
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("p", &[("w1", 1)])],
        None,
        true,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);

    assert!(
        applied_pages(&tree).is_empty(),
        "page with no matching session must not be applied",
    );
}

#[tokio::test]
async fn receiver_drops_page_with_mismatched_session_id() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    // Live session is `live_session`; incoming page carries
    // a different session_id (e.g., a delayed retry response).
    let live_session = Uuid::now_v7();
    let stale_session = Uuid::now_v7();
    seed_outstanding(
        &adapter,
        "model-1",
        TreeKind::String,
        live_session,
        "node-b",
    );

    let page = make_page(
        stale_session,
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("p", &[("w1", 1)])],
        None,
        true,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);

    assert!(applied_pages(&tree).is_empty(), "stale session_id rejected");
    // Live session not affected.
    let entry = adapter
        .outstanding_repairs
        .get(&("model-1".to_string(), TreeKind::String))
        .expect("live session retained");
    assert_eq!(entry.session_id, live_session);
    assert!(entry.applied.is_empty());
}

#[tokio::test]
async fn terminal_page_clears_outstanding_session() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session = Uuid::now_v7();
    seed_outstanding(&adapter, "model-1", TreeKind::String, session, "node-b");
    let page = make_page(
        session,
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("p", &[("w1", 1)])],
        None,
        true,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);

    assert!(
        !adapter
            .outstanding_repairs
            .contains_key(&("model-1".to_string(), TreeKind::String)),
        "terminal page must clear the outstanding session",
    );
}

#[tokio::test]
async fn terminal_page_with_gap_keeps_session_open() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session = Uuid::now_v7();
    seed_outstanding(&adapter, "model-1", TreeKind::String, session, "node-b");
    let page0 = make_page(
        session,
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("a", &[("w1", 1)])],
        Some(1),
        false,
    );
    let terminal_page2 = make_page(
        session,
        "model-1",
        TreeKind::String,
        2,
        vec![string_entry("c", &[("w1", 1)])],
        None,
        true,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&page0)]);
    adapter.handle_incoming_repair_page(&[page_bytes(&terminal_page2)]);

    let entry = adapter
        .outstanding_repairs
        .get(&("model-1".to_string(), TreeKind::String))
        .expect("gapped terminal session stays open for retry");
    assert_eq!(entry.terminal_page_index, Some(2));
    assert!(!entry.is_contiguously_complete());
    assert_eq!(
        entry.contiguous_cursor(),
        Some(bincode::serialize(&1u64).unwrap()),
        "retry should resume after the contiguous prefix only",
    );
}

#[tokio::test]
async fn contiguous_cursor_handles_out_of_order_pages() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session = Uuid::now_v7();
    seed_outstanding(&adapter, "model-1", TreeKind::String, session, "node-b");

    // Apply page 0 (cursor=1), then page 2 (cursor=3),
    // skipping page 1. contiguous cursor must reflect
    // page 0 only — not page 2.
    let p0 = make_page(
        session,
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("a", &[("w1", 1)])],
        Some(1),
        false,
    );
    let p2 = make_page(
        session,
        "model-1",
        TreeKind::String,
        2,
        vec![string_entry("c", &[("w1", 1)])],
        Some(3),
        false,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&p0)]);
    adapter.handle_incoming_repair_page(&[page_bytes(&p2)]);

    let entry = adapter
        .outstanding_repairs
        .get(&("model-1".to_string(), TreeKind::String))
        .expect("session present");
    assert!(entry.applied.contains_key(&0));
    assert!(entry.applied.contains_key(&2));
    assert_eq!(
        entry.contiguous_cursor(),
        Some(bincode::serialize(&1u64).unwrap()),
        "contiguous cursor must reflect page 0, not page 2",
    );
}

#[tokio::test]
async fn retry_scan_reissues_stale_session_with_contiguous_cursor() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b", "node-c"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    // Seed an old session that has applied page 0; pretend
    // it started 10s ago (> 5s timeout).
    let old_session = Uuid::now_v7();
    let mut progress = RepairProgress::new(
        old_session,
        "node-b".into(),
        Instant::now() - Duration::from_secs(10),
    );
    progress
        .applied
        .insert(0, Some(bincode::serialize(&5u64).unwrap()));
    adapter
        .outstanding_repairs
        .insert(("model-1".to_string(), TreeKind::String), progress);

    // Run scan with retry_timeout=5s, max_retries=3.
    adapter.scan_for_retries(Instant::now(), Duration::from_secs(5), 3);

    // A new request was published — same model+kind, fresh
    // session_id, cursor = bincode(5).
    let publishes = drain_repair_publishes(&mesh);
    assert_eq!(publishes.len(), 1, "exactly one retry request");
    let (_target, _key, payload) = &publishes[0];
    let request: TreeRepairRequest = bincode::deserialize(payload).unwrap();
    assert_eq!(request.model_id, "model-1");
    assert_eq!(request.tree_kind, TreeKind::String);
    assert_ne!(request.session_id, old_session, "fresh session_id on retry");
    assert_eq!(
        request.cursor.as_deref(),
        Some(bincode::serialize(&5u64).unwrap().as_slice()),
        "retry carries contiguous cursor",
    );

    // outstanding_repairs replaced with a fresh progress
    // entry under the new session_id, retry_count=1.
    let entry = adapter
        .outstanding_repairs
        .get(&("model-1".to_string(), TreeKind::String))
        .expect("session retained");
    assert_eq!(entry.session_id, request.session_id);
    assert_eq!(entry.retry_count, 1);
    assert!(entry.applied.is_empty(), "applied set reset on retry");
}

#[tokio::test]
async fn retry_scan_skips_session_with_recent_page_progress() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session = Uuid::now_v7();
    adapter.outstanding_repairs.insert(
        ("model-1".to_string(), TreeKind::String),
        RepairProgress::new(
            session,
            "node-b".into(),
            Instant::now() - Duration::from_secs(10),
        ),
    );

    let page = make_page(
        session,
        "model-1",
        TreeKind::String,
        0,
        vec![string_entry("a", &[("w1", 1)])],
        Some(1),
        false,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);

    adapter.scan_for_retries(Instant::now(), Duration::from_secs(5), 3);

    assert!(
        drain_repair_publishes(&mesh).is_empty(),
        "recent page progress should suppress retry",
    );
    let entry = adapter
        .outstanding_repairs
        .get(&("model-1".to_string(), TreeKind::String))
        .expect("session retained");
    assert_eq!(entry.session_id, session);
    assert_eq!(entry.retry_count, 0);
}

#[tokio::test]
async fn retry_scan_gives_up_after_max_retries() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, tree, peers, "node-a".into());

    let mut progress = RepairProgress::new(
        Uuid::now_v7(),
        "node-b".into(),
        Instant::now() - Duration::from_secs(60),
    );
    progress.retry_count = 3; // already at max
    adapter
        .outstanding_repairs
        .insert(("model-1".to_string(), TreeKind::String), progress);

    adapter.scan_for_retries(Instant::now(), Duration::from_secs(5), 3);

    assert!(
        adapter.outstanding_repairs.is_empty(),
        "session removed after exceeding max retries",
    );
    let publishes = drain_repair_publishes(&mesh);
    assert!(publishes.is_empty(), "no retry request emitted at limit");
}

#[tokio::test]
async fn retry_scan_prefers_different_peer() {
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree: Arc<dyn TreeHandle> = empty_handle();
    let peers: Arc<dyn PeerList> = MockPeerList::with(&["node-b", "node-c"]);
    let adapter = TreeSyncAdapter::new(td, req, pages, tree, peers, "node-a".into());

    let progress = RepairProgress::new(
        Uuid::now_v7(),
        "node-b".into(),
        Instant::now() - Duration::from_secs(10),
    );
    adapter
        .outstanding_repairs
        .insert(("model-1".to_string(), TreeKind::String), progress);

    adapter.scan_for_retries(Instant::now(), Duration::from_secs(5), 3);

    let publishes = drain_repair_publishes(&mesh);
    assert_eq!(publishes.len(), 1);
    let (target, _, _) = &publishes[0];
    assert_eq!(target, "node-c", "retry must avoid the previous target");
}

#[tokio::test]
async fn variant_mismatch_is_skipped_not_panic() {
    // A page declared as String containing a Token entry
    // must drop the Token entry without panicking and not
    // count it toward the applied total.
    let mesh = MeshKV::new("node-a".into());
    let td = td_namespace(&mesh);
    let req = req_namespace(&mesh);
    let pages = page_namespace(&mesh);
    let tree = Arc::new(MockTreeHandle::default());
    let adapter_tree: Arc<dyn TreeHandle> = tree.clone();
    let peers: Arc<dyn PeerList> = empty_peers();
    let adapter = TreeSyncAdapter::new(td, req, pages, adapter_tree, peers, "node-a".into());

    let session = Uuid::now_v7();
    seed_outstanding(&adapter, "model-1", TreeKind::String, session, "node-b");
    let page = make_page(
        session,
        "model-1",
        TreeKind::String,
        0,
        vec![
            string_entry("ok", &[("w1", 1)]),
            token_entry(&[1, 2, 3], &[("w1", 2)]), // wrong variant
        ],
        None,
        true,
    );
    adapter.handle_incoming_repair_page(&[page_bytes(&page)]);

    // Mock counts only matching variants — string entry yes,
    // token entry no.
    let calls = applied_pages(&tree);
    assert_eq!(calls.len(), 1);
}
