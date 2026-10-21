//! `td:` stream adapter: gateway ↔ mesh bridge for the distributed
//! prefix tree.
//!
//! - Outbound: `on_local_insert` buffers per-model `TreeDelta`s;
//!   the drain callback batches each model into one
//!   `td:{model_id}` stream entry per gossip round.
//! - Inbound: a spawned task subscribes to `td:`, decodes each
//!   batch, and hands every delta to [`TreeHandle`] which either
//!   applies it (the matched prefix is already known locally and
//!   the worker becomes a tenant of that node) or returns false.
//!   On false, the adapter publishes a `tree:req:` repair request
//!   targeted at a random ALIVE peer; the response (`tree:page:`)
//!   is consumed in a later slice.
//!
//! The adapter holds no tree-membership state. The tree owner
//! (`CacheAwarePolicy` in production) implements [`TreeHandle`];
//! dependency direction is adapter → policy. Membership lookups
//! likewise go through a [`PeerList`] trait so the adapter doesn't
//! reach into the gossip controller directly.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use bytes::Bytes;
use dashmap::DashMap;
use kv_index::TenantId;
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};
use smg_mesh::{DrainHandle, StreamNamespace, MAX_STREAM_CHUNK_BYTES};
use tracing::{debug, error, trace, warn};
use uuid::Uuid;

use crate::policies::{TreeHandle, TreeKind};

const PREFIX: &str = "td:";
pub(crate) const REPAIR_REQUEST_PREFIX: &str = "tree:req:";
pub(crate) const REPAIR_PAGE_PREFIX: &str = "tree:page:";
pub(crate) const TENANT_DELTA_PREFIX: &str = PREFIX;

/// Duration a repair session may sit without progress before
/// the periodic retry scan reissues it. Spec §15 default: 5 s.
//
// TODO(d-3): read from `MeshConfig.tree_repair_retry_timeout`
// once `server.rs` wires the adapter.
const TREE_REPAIR_RETRY_TIMEOUT: Duration = Duration::from_secs(5);

/// How often the retry scan runs. Smaller than the timeout so
/// stale sessions are caught within a tick of becoming stale.
const TREE_REPAIR_RETRY_SCAN_INTERVAL: Duration = Duration::from_secs(1);

/// Maximum retry attempts for a single (model_id, tree_kind)
/// repair session before giving up. The next unknown-hash
/// delta restarts a fresh session.
const TREE_REPAIR_MAX_RETRIES: u32 = 3;

/// Soft cap on the bincode-serialized size of a single
/// [`TreeRepairPage`]. Spec default (mesh-v2 §3.1,
/// `max_tree_repair_page_bytes`). The responder fills entries
/// until the next would push the page over this budget.
//
// TODO(d-3): read from the live `MeshConfig.max_tree_repair_page_bytes`
// once `server.rs` wires the adapter, instead of using a const.
pub const TREE_REPAIR_PAGE_BYTE_CAP: usize = 2 * 1024 * 1024;

/// Reserved budget for the [`TreeRepairPage`] header (everything
/// other than `entries`). Subtracted from the byte cap so the
/// assembled page stays under the cap once header bytes are
/// added back. UUID + model_id + tree_kind + counters comfortably
/// fit; 256 is conservative for realistic model-id lengths
/// (typical ~30 chars, max ~100). A pathological multi-hundred-
/// char model_id could theoretically push the actual header
/// past this budget, but SMG does not produce such ids.
const TREE_REPAIR_PAGE_HEADER_OVERHEAD: usize = 256;

/// Hard ceiling on the bincode-serialized size of a single
/// [`TreeRepairPage`]. Above this, the mesh layer's chunking path
/// fires a `debug_assert!` for `tree:page:*` (spec forbids
/// multi-chunk reassembly for tree repair) and would split the
/// value across multiple wire frames in release. The ceiling
/// already accounts for gossip envelope overhead via
/// `STREAM_CHUNK_OVERHEAD_MARGIN`.
const TREE_REPAIR_PAGE_HARD_CEILING: usize = MAX_STREAM_CHUNK_BYTES;

/// Hard ceiling on a single [`RepairEntry`]'s serialized size:
/// the wire ceiling minus the page header reserve. Entries
/// exceeding this cannot be shipped at all without triggering
/// the multi-chunk path.
const REPAIR_ENTRY_HARD_CEILING: usize =
    TREE_REPAIR_PAGE_HARD_CEILING - TREE_REPAIR_PAGE_HEADER_OVERHEAD;

/// Live-membership view consumed by the adapter when picking a
/// peer to send a repair request to. Defined here (not in the
/// mesh crate) so adapter tests can supply a mock without spinning
/// up gossip; the production impl wraps the gossip controller's
/// alive-peer list and lands with the slice that wires the
/// adapter into `server.rs`.
pub trait PeerList: Send + Sync + std::fmt::Debug {
    /// Currently-ALIVE peer names, excluding the local node.
    /// Order is implementation-defined; the adapter picks one at
    /// random.
    fn alive_peers(&self) -> Vec<String>;
}

/// One prefix-tree change observed on a producing node. `node_hash`
/// is the blake3 8-byte id scoped by `(model_id, tree_kind, path)`;
/// `model_id` lives in the stream key, `tree_kind` is inside this
/// struct. Receivers resolve the hash via `TreeHandle`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeDelta {
    pub tree_kind: TreeKind,
    pub node_hash: u64,
    /// Worker URL that cached the prefix.
    pub worker_url: String,
    /// Reserved on-wire slot for a future intra-batch ordering
    /// value. The current receiver does not consult it — it appears
    /// only in `trace!`/`debug!` log lines — so producers may set
    /// `0` until a consumer that orders by epoch lands. Kept on
    /// `TreeDelta` (rather than added later) so a newer receiver
    /// paired with an older producer just sees zeros instead of a
    /// deserialization mismatch.
    pub epoch: u64,
}

/// Why a node is asking for repair. Carried on the wire so the
/// responder can log/diagnose; doesn't change the response shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairReason {
    /// Inbound `TreeDelta` referenced a hash this node didn't know.
    UnknownHash(u64),
}

/// Wire format for a `tree:req:{session_id}` message. Targeted at
/// one peer; the responder generates `tree:page:{session_id}:N`
/// fragments back to `requester_peer_id`. Slice 5d implements the
/// responder + page handler; slice 5c only emits these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeRepairRequest {
    pub session_id: Uuid,
    pub requester_peer_id: String,
    pub target_peer_id: String,
    pub model_id: String,
    pub tree_kind: TreeKind,
    /// Resumable pagination cursor. `None` means "from the
    /// beginning"; slice 5d will set this on retry after a
    /// timeout to skip already-applied pages.
    pub cursor: Option<Vec<u8>>,
    pub reason: RepairReason,
}

/// One reconstructable tree entry on the repair-page wire. The
/// variant matches the requested [`TreeKind`]; receivers may
/// validate the variant against the page's `tree_kind` and drop
/// mismatched entries as malformed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairEntry {
    String {
        path: String,
        tenants: Vec<(TenantId, u64)>,
    },
    Token {
        tokens: Vec<u32>,
        tenants: Vec<(TenantId, u64)>,
    },
}

/// One page of tree-repair payload, targeted at the requester
/// keyed `tree:page:{session_id}:{page_index}`. `next_cursor`
/// is `None` iff `is_last`; resumed requests echo the cursor
/// back in [`TreeRepairRequest::cursor`] so the responder skips
/// already-shipped entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeRepairPage {
    pub session_id: Uuid,
    pub model_id: String,
    pub tree_kind: TreeKind,
    pub page_index: u32,
    pub entries: Vec<RepairEntry>,
    pub next_cursor: Option<Vec<u8>>,
    pub is_last: bool,
}

/// Iterator that chunks a stream of [`RepairEntry`] into
/// byte-capped [`TreeRepairPage`]s. Always emits at least one
/// page (even on an empty stream — the empty page acts as an
/// ACK for the requester and gives d-2c a clean session-cleanup
/// signal).
struct PagingIter {
    stream: Box<dyn Iterator<Item = RepairEntry> + Send>,
    /// Entry pulled from the stream that didn't fit in the
    /// previous page's budget; consumed first on the next call.
    deferred: Option<RepairEntry>,
    session_id: Uuid,
    model_id: String,
    tree_kind: TreeKind,
    /// Per-page budget for entry bytes (cap minus reserved header
    /// overhead). At least one entry always fits per page even
    /// if the entry alone exceeds the budget.
    entry_budget: usize,
    /// Total entries already consumed across all pages, including
    /// any entries skipped via the start cursor. Used as the
    /// `next_cursor` value for the page being built.
    cumulative_consumed: u64,
    page_index: u32,
    finished: bool,
}

impl PagingIter {
    fn new(
        mut stream: Box<dyn Iterator<Item = RepairEntry> + Send>,
        session_id: Uuid,
        model_id: String,
        tree_kind: TreeKind,
        byte_cap: usize,
        start_skip: u64,
    ) -> Self {
        // Best-effort cursor: skip entries the requester already
        // has. If the tree shrunk between rounds we end early —
        // the final page just has fewer entries.
        for _ in 0..start_skip {
            if stream.next().is_none() {
                break;
            }
        }
        Self {
            stream,
            deferred: None,
            session_id,
            model_id,
            tree_kind,
            entry_budget: byte_cap.saturating_sub(TREE_REPAIR_PAGE_HEADER_OVERHEAD),
            cumulative_consumed: start_skip,
            page_index: 0,
            finished: false,
        }
    }
}

impl PagingIter {
    /// Classify and try to pack a single entry into the current
    /// page being built. Returns `Some(entry)` if the entry didn't
    /// fit and must be deferred to the next page; otherwise `None`
    /// (entry was either packed, or dropped above the wire ceiling).
    ///
    /// Four cases:
    ///   - size > `REPAIR_ENTRY_HARD_CEILING`: drop with `error!`
    ///     (multi-chunk path is forbidden for `tree:page:*`).
    ///   - page non-empty AND would overflow `entry_budget`:
    ///     defer to next page.
    ///   - page empty AND size > `entry_budget`: emit anyway as
    ///     sole-entry page with `warn!` flagged as spec debt.
    ///   - otherwise: pack normally.
    fn try_pack_entry(
        &mut self,
        entries: &mut Vec<RepairEntry>,
        size: &mut usize,
        entry: RepairEntry,
    ) -> Option<RepairEntry> {
        let entry_size =
            bincode::serialized_size(&entry).unwrap_or(REPAIR_ENTRY_HARD_CEILING as u64) as usize;

        if entry_size > REPAIR_ENTRY_HARD_CEILING {
            error!(
                entry_size,
                ceiling = REPAIR_ENTRY_HARD_CEILING,
                model_id = %self.model_id,
                kind = ?self.tree_kind,
                "RepairEntry exceeds wire-level ceiling; dropping. \
                 Receiver will not learn this prefix via repair until \
                 direct traffic populates it. See d-2c follow-up for \
                 entry fragmentation."
            );
            // TODO(d-2c): emit a `router_mesh_tree_repair_drops_total{reason=\"oversized_entry\"}` counter.
            self.cumulative_consumed += 1;
            return None;
        }

        if !entries.is_empty() && *size + entry_size > self.entry_budget {
            return Some(entry);
        }

        if entries.is_empty() && entry_size > self.entry_budget {
            warn!(
                entry_size,
                budget = self.entry_budget,
                ceiling = REPAIR_ENTRY_HARD_CEILING,
                model_id = %self.model_id,
                kind = ?self.tree_kind,
                "tree-repair spec debt: RepairEntry exceeds soft page \
                 cap; emitting as sole-entry oversized page (still \
                 under the wire ceiling). See d-2c follow-up for \
                 entry fragmentation."
            );
        }

        entries.push(entry);
        *size += entry_size;
        self.cumulative_consumed += 1;
        None
    }
}

impl Iterator for PagingIter {
    type Item = TreeRepairPage;

    fn next(&mut self) -> Option<TreeRepairPage> {
        if self.finished {
            return None;
        }

        let mut entries: Vec<RepairEntry> = Vec::new();
        let mut size: usize = 0;
        let mut hit_budget = false;

        // Pull deferred entry first, then fall through to the
        // stream. Both sources share one classification path.
        // Pull-then-process avoids holding a `&mut self.stream`
        // borrow across the `&mut self` call to `try_pack_entry`.
        loop {
            let entry = if let Some(entry) = self.deferred.take() {
                entry
            } else if let Some(entry) = self.stream.next() {
                entry
            } else {
                break;
            };
            if let Some(deferred) = self.try_pack_entry(&mut entries, &mut size, entry) {
                self.deferred = Some(deferred);
                hit_budget = true;
                break;
            }
        }

        let is_last = !hit_budget;
        let next_cursor = if is_last {
            None
        } else {
            // u64 always bincode-serializes; if it ever didn't,
            // an empty cursor decodes back to 0 ("start over") —
            // safe best-effort fallback rather than a panic.
            Some(bincode::serialize(&self.cumulative_consumed).unwrap_or_default())
        };

        let page = TreeRepairPage {
            session_id: self.session_id,
            model_id: self.model_id.clone(),
            tree_kind: self.tree_kind,
            page_index: self.page_index,
            entries,
            next_cursor,
            is_last,
        };

        self.page_index += 1;
        if is_last {
            self.finished = true;
        }
        Some(page)
    }
}

/// Decode a [`TreeRepairRequest::cursor`] (bincode-encoded `u64`
/// skip count). Treats `None` and decode errors as "start from
/// the beginning" — a malformed cursor is best handled by re-
/// shipping the whole tree rather than refusing to respond.
fn decode_cursor(cursor: Option<&[u8]>) -> u64 {
    cursor
        .and_then(|bytes| bincode::deserialize::<u64>(bytes).ok())
        .unwrap_or(0)
}

/// Per-session bookkeeping for an in-flight repair request.
/// Stored in `outstanding_repairs[(model_id, tree_kind)]` —
/// at most one session per (model_id, tree_kind) at a time
/// (subsequent unknown-hash triggers coalesce into the active
/// session).
struct RepairProgress {
    session_id: Uuid,
    target_peer_id: String,
    /// Time the *current* request was issued. Reset on retry.
    started_at: Instant,
    /// Time the current session last made receiver-side progress.
    /// Refreshed on every accepted repair page so long-running
    /// transfers do not retry while actively advancing.
    last_activity_at: Instant,
    /// Number of times this (model_id, tree_kind) session has
    /// been retried. New session_id is generated on each retry.
    retry_count: u32,
    /// Pages received and applied, with their `next_cursor`.
    /// BTreeMap so iteration order = page-index order — used
    /// to compute the contiguous-prefix cursor for retry.
    applied: BTreeMap<u32, Option<Vec<u8>>>,
    /// Page index of the terminal page once observed. The repair
    /// is complete only when every page through this index has
    /// been applied.
    terminal_page_index: Option<u32>,
}

impl RepairProgress {
    fn new(session_id: Uuid, target_peer_id: String, now: Instant) -> Self {
        Self {
            session_id,
            target_peer_id,
            started_at: now,
            last_activity_at: now,
            retry_count: 0,
            applied: BTreeMap::new(),
            terminal_page_index: None,
        }
    }

    /// Cursor at the end of the highest contiguous applied
    /// prefix. `None` if no pages applied or page 0 is missing.
    fn contiguous_cursor(&self) -> Option<Vec<u8>> {
        let mut cursor: Option<Vec<u8>> = None;
        for (expected, (&idx, c)) in (0u32..).zip(&self.applied) {
            if idx != expected {
                break;
            }
            cursor.clone_from(c);
        }
        cursor
    }

    fn is_contiguously_complete(&self) -> bool {
        let Some(last_page_index) = self.terminal_page_index else {
            return false;
        };
        (0..=last_page_index).all(|idx| self.applied.contains_key(&idx))
    }
}

/// Bridges the `td:` broadcast stream namespace to per-model
/// tenant buffers, querying a [`TreeHandle`] for inbound hash
/// resolution and a [`PeerList`] for repair-request targeting.
pub struct TreeSyncAdapter {
    tenant_deltas: Arc<StreamNamespace>,
    /// Targeted namespace for `tree:req:*` repair requests.
    /// The adapter both publishes (5c, when a local node hits an
    /// unknown hash) and subscribes (d-2b, when a peer asks us
    /// for our tree).
    tree_repair_requests: Arc<StreamNamespace>,
    /// Targeted namespace for `tree:page:*` repair pages.
    /// The adapter publishes pages here in response to
    /// `tree:req:` messages; the receiver side lands in d-2c.
    tree_repair_pages: Arc<StreamNamespace>,
    pending_deltas: DashMap<String, Vec<TreeDelta>>,
    /// Hash-membership handle provided by the tree owner.
    tree: Arc<dyn TreeHandle>,
    /// Live-peer source for repair-request targeting.
    peers: Arc<dyn PeerList>,
    /// Outstanding repair sessions, keyed by (model_id, kind).
    /// At most one session per (model_id, kind) at a time:
    /// subsequent unknown-hash triggers coalesce into the
    /// active session.  Each [`RepairProgress`] tracks the
    /// session_id, target peer, applied pages with cursors,
    /// retry count, and completion. Cleared on terminal page
    /// or after `TREE_REPAIR_MAX_RETRIES` failed retries.
    outstanding_repairs: DashMap<(String, TreeKind), RepairProgress>,
    node_name: String,
    /// Keeps the drain registration alive; dropping it unregisters
    /// from the mesh. `OnceLock` guards against a second `start`.
    drain_handle: OnceLock<DrainHandle>,
}

impl std::fmt::Debug for TreeSyncAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeSyncAdapter")
            .field("prefix", &self.tenant_deltas.prefix())
            .field("node_name", &self.node_name)
            .field("pending_models", &self.pending_deltas.len())
            .field("outstanding_repairs", &self.outstanding_repairs.len())
            .finish()
    }
}

impl TreeSyncAdapter {
    /// Build an adapter wrapping the `td:` broadcast namespace
    /// (tenant deltas), the `tree:req:` targeted namespace
    /// (repair requests), the local tree handle, peer list, and
    /// the local node name. Panics if either namespace prefix is
    /// wrong so a mis-wired caller fails loudly at startup instead
    /// of fanning entries into the wrong stream.
    pub fn new(
        tenant_deltas: Arc<StreamNamespace>,
        tree_repair_requests: Arc<StreamNamespace>,
        tree_repair_pages: Arc<StreamNamespace>,
        tree: Arc<dyn TreeHandle>,
        peers: Arc<dyn PeerList>,
        node_name: String,
    ) -> Arc<Self> {
        assert_eq!(
            tenant_deltas.prefix(),
            PREFIX,
            "TreeSyncAdapter requires a tenant-delta namespace scoped to `{PREFIX}`",
        );
        assert_eq!(
            tree_repair_requests.prefix(),
            REPAIR_REQUEST_PREFIX,
            "TreeSyncAdapter requires a repair-request namespace scoped to `{REPAIR_REQUEST_PREFIX}`",
        );
        assert_eq!(
            tree_repair_pages.prefix(),
            REPAIR_PAGE_PREFIX,
            "TreeSyncAdapter requires a repair-page namespace scoped to `{REPAIR_PAGE_PREFIX}`",
        );
        assert!(
            !node_name.is_empty(),
            "TreeSyncAdapter node_name must not be empty",
        );
        Arc::new(Self {
            tenant_deltas,
            tree_repair_requests,
            tree_repair_pages,
            pending_deltas: DashMap::new(),
            tree,
            peers,
            outstanding_repairs: DashMap::new(),
            node_name,
            drain_handle: OnceLock::new(),
        })
    }

    /// Register the drain callback and start the inbound task. Call
    /// once per adapter — a second call panics via the mesh's
    /// one-drain-per-prefix invariant.
    pub fn start(self: &Arc<Self>) {
        // `Weak` avoids the `TreeSyncAdapter → DrainHandle →
        // DrainRegistry → drain closure → TreeSyncAdapter` strong
        // cycle that would leak the drain registration past adapter
        // drop. If upgrade fails, return an empty batch; the mesh
        // tears down the `DrainHandle` on its own `Drop`.
        let drain_owner = Arc::downgrade(self);
        let handle = self.tenant_deltas.register_drain(Box::new(move || {
            drain_owner
                .upgrade()
                .map(|this| this.drain_pending_deltas())
                .unwrap_or_default()
        }));
        assert!(
            self.drain_handle.set(handle).is_ok(),
            "TreeSyncAdapter::start called more than once",
        );

        // Same `Weak` pattern for the subscription task so a late
        // channel close can't strand the adapter alive. Exit on
        // first upgrade-None.
        let sub_owner = Arc::downgrade(self);
        let mut sub = self.tenant_deltas.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, value)) = sub.receiver.recv().await {
                let Some(this) = sub_owner.upgrade() else {
                    debug!("TreeSyncAdapter dropped, exiting tenant-delta subscription");
                    break;
                };
                let Some(model_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                    warn!(key, "td: subscription yielded unexpected key shape");
                    continue;
                };
                match value {
                    Some(fragments) => this.handle_incoming_batch(model_id, &fragments),
                    None => {
                        // Shared CRDT/stream subscription API — td:
                        // never emits tombstones today.
                        debug!(model_id, "unexpected td: tombstone event");
                    }
                }
            }
            debug!("TreeSyncAdapter tenant-delta subscription closed");
        });

        // Responder side of the repair protocol: subscribe to
        // `tree:req:`, dispatch each request to a bounded per-
        // session task that pages the local tree out via
        // `tree:page:` to the requester.
        let req_owner = Arc::downgrade(self);
        let mut req_sub = self.tree_repair_requests.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, value)) = req_sub.receiver.recv().await {
                let Some(this) = req_owner.upgrade() else {
                    debug!("TreeSyncAdapter dropped, exiting tree:req: subscription");
                    break;
                };
                if !key.starts_with(REPAIR_REQUEST_PREFIX) {
                    warn!(key, "tree:req: subscription yielded unexpected key shape");
                    continue;
                }
                match value {
                    Some(fragments) => this.handle_incoming_repair_request(&fragments),
                    None => {
                        debug!("unexpected tree:req: tombstone event");
                    }
                }
            }
            debug!("TreeSyncAdapter tree:req: subscription closed");
        });

        // Receiver side of the repair protocol: subscribe to
        // `tree:page:`, decode each page, dedupe + apply via
        // `TreeHandle::apply_repair_page`, and clean up the
        // outstanding session on the terminal page.
        let page_owner = Arc::downgrade(self);
        let mut page_sub = self.tree_repair_pages.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, value)) = page_sub.receiver.recv().await {
                let Some(this) = page_owner.upgrade() else {
                    debug!("TreeSyncAdapter dropped, exiting tree:page: subscription");
                    break;
                };
                if !key.starts_with(REPAIR_PAGE_PREFIX) {
                    warn!(key, "tree:page: subscription yielded unexpected key shape");
                    continue;
                }
                match value {
                    Some(fragments) => this.handle_incoming_repair_page(&fragments),
                    None => {
                        debug!("unexpected tree:page: tombstone event");
                    }
                }
            }
            debug!("TreeSyncAdapter tree:page: subscription closed");
        });

        // Periodic retry/timeout scan: reissues stale repair
        // sessions with a fresh session_id and the
        // last-contiguous-applied cursor. Sessions exceeding
        // `TREE_REPAIR_MAX_RETRIES` are dropped from
        // `outstanding_repairs` so the next unknown-hash delta
        // can restart from scratch.
        let retry_owner = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "periodic scan ends automatically when the adapter is dropped (Weak upgrade fails); no handle needed"
        )]
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(TREE_REPAIR_RETRY_SCAN_INTERVAL);
            // Skip the immediate first tick — no progress to scan
            // for in the first millisecond after start().
            interval.tick().await;
            loop {
                interval.tick().await;
                let Some(this) = retry_owner.upgrade() else {
                    debug!("TreeSyncAdapter dropped, exiting retry scan task");
                    break;
                };
                this.scan_for_retries(
                    Instant::now(),
                    TREE_REPAIR_RETRY_TIMEOUT,
                    TREE_REPAIR_MAX_RETRIES,
                );
            }
        });
    }

    /// Buffer a local tree insert for the next gossip round. Hot
    /// path — keep it cheap; the drain does the serialisation.
    /// `delta.node_hash` must be non-zero: 0 is
    /// [`kv_index::GLOBAL_EVICTION_HASH`] (producers remap 0→1 to
    /// keep the space disjoint).
    pub fn on_local_insert(&self, model_id: &str, delta: TreeDelta) {
        debug_assert!(
            !model_id.is_empty(),
            "TreeSyncAdapter::on_local_insert requires non-empty model_id",
        );
        debug_assert_ne!(
            delta.node_hash, 0,
            "TreeDelta.node_hash must be non-zero (0 is reserved for GLOBAL_EVICTION_HASH)",
        );
        self.pending_deltas
            .entry(model_id.to_string())
            .or_default()
            .push(delta);
    }

    /// Test-only view onto the outbound buffer so integration tests
    /// outside this file can assert that a producer hook fired without
    /// having to wait a full gossip tick for drain.
    #[cfg(test)]
    pub(crate) fn pending_delta_count_for_test(&self, model_id: &str) -> usize {
        self.pending_deltas
            .get(model_id)
            .map(|d| d.len())
            .unwrap_or(0)
    }

    /// Collect each model's buffer into one `td:{model_id}` stream
    /// entry. Called once per gossip round. Iterate→collect→remove
    /// avoids the deadlock DashMap hits on iterate-and-mutate.
    fn drain_pending_deltas(&self) -> Vec<(String, Bytes)> {
        let model_ids: Vec<String> = self
            .pending_deltas
            .iter()
            .filter(|e| !e.value().is_empty())
            .map(|e| e.key().clone())
            .collect();

        let mut entries = Vec::with_capacity(model_ids.len());
        for model_id in model_ids {
            let Some((_, deltas)) = self.pending_deltas.remove(&model_id) else {
                continue;
            };
            if deltas.is_empty() {
                continue;
            }
            match bincode::serialize(&deltas) {
                Ok(bytes) => {
                    entries.push((format!("{PREFIX}{model_id}"), Bytes::from(bytes)));
                }
                Err(err) => {
                    // Should be unreachable for this schema; drop
                    // the batch rather than re-enter next round.
                    warn!(model_id, %err, "failed to serialize tenant deltas");
                }
            }
        }
        entries
    }

    fn handle_incoming_batch(&self, model_id: &str, fragments: &[Bytes]) {
        let total = fragments.iter().map(Bytes::len).sum();
        let mut bytes = Vec::with_capacity(total);
        for frag in fragments {
            bytes.extend_from_slice(frag);
        }
        let batch: Vec<TreeDelta> = match bincode::deserialize(&bytes) {
            Ok(batch) => batch,
            Err(err) => {
                warn!(model_id, %err, "failed to decode tenant-delta batch");
                return;
            }
        };
        debug!(
            model_id,
            count = batch.len(),
            "remote tenant-delta batch received"
        );
        for delta in &batch {
            if self.tree.apply_known_remote_insert(
                model_id,
                delta.tree_kind,
                delta.node_hash,
                &delta.worker_url,
            ) {
                trace!(
                    model_id,
                    kind = ?delta.tree_kind,
                    hash = delta.node_hash,
                    worker_url = %delta.worker_url,
                    epoch = delta.epoch,
                    "applied remote tenant delta against local tree",
                );
            } else {
                debug!(
                    model_id,
                    kind = ?delta.tree_kind,
                    hash = delta.node_hash,
                    worker_url = %delta.worker_url,
                    epoch = delta.epoch,
                    "unknown remote tenant delta hash, requesting repair",
                );
                self.request_repair_for_unknown_hash(model_id, delta.tree_kind, delta.node_hash);
            }
        }
    }

    /// Decode an incoming `tree:req:` payload and dispatch a
    /// per-session paging task. Mismatched `target_peer_id` is
    /// dropped silently — targeted routing should already filter,
    /// but defense-in-depth against future routing changes.
    fn handle_incoming_repair_request(self: &Arc<Self>, fragments: &[Bytes]) {
        let total: usize = fragments.iter().map(Bytes::len).sum();
        let mut bytes = Vec::with_capacity(total);
        for frag in fragments {
            bytes.extend_from_slice(frag);
        }
        let request: TreeRepairRequest = match bincode::deserialize(&bytes) {
            Ok(req) => req,
            Err(err) => {
                warn!(%err, "failed to decode TreeRepairRequest");
                return;
            }
        };
        if request.target_peer_id != self.node_name {
            trace!(
                target = %request.target_peer_id,
                local = %self.node_name,
                session_id = %request.session_id,
                "dropping repair request not addressed to this node",
            );
            return;
        }

        // CPU-bound: tree walk + bincode-serialize multi-MB
        // pages. Park it on the blocking pool so a busy responder
        // doesn't starve tokio worker threads. Bounded by stream
        // length; loss on shutdown is fine (the requester retries
        // or times out via d-2c session cleanup).
        let this = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            this.respond_to_repair_request(request);
        });
    }

    /// Walk the local tree for the requested model+kind, chunk
    /// it into byte-capped pages, and publish each page to the
    /// requester. Always emits at least one page (an empty
    /// `is_last=true` page if the tree is missing or empty) so
    /// the requester sees an ACK and d-2c session-cleanup has a
    /// clean signal.
    fn respond_to_repair_request(&self, request: TreeRepairRequest) {
        let stream: Box<dyn Iterator<Item = RepairEntry> + Send> = self
            .tree
            .open_repair_stream(&request.model_id, request.tree_kind)
            .unwrap_or_else(|| Box::new(std::iter::empty()));

        let start_skip = decode_cursor(request.cursor.as_deref());
        let pager = PagingIter::new(
            stream,
            request.session_id,
            request.model_id.clone(),
            request.tree_kind,
            TREE_REPAIR_PAGE_BYTE_CAP,
            start_skip,
        );

        let mut pages_sent: u32 = 0;
        for page in pager {
            let key = format!(
                "{REPAIR_PAGE_PREFIX}{}:{}",
                page.session_id, page.page_index,
            );
            let bytes = match bincode::serialize(&page) {
                Ok(b) => Bytes::from(b),
                Err(err) => {
                    warn!(
                        session_id = %page.session_id,
                        page_index = page.page_index,
                        entry_count = page.entries.len(),
                        %err,
                        "failed to serialize TreeRepairPage; skipping",
                    );
                    continue;
                }
            };
            self.tree_repair_pages
                .publish_to(&request.requester_peer_id, &key, bytes);
            pages_sent += 1;
        }

        debug!(
            session_id = %request.session_id,
            model_id = %request.model_id,
            kind = ?request.tree_kind,
            requester = %request.requester_peer_id,
            pages_sent,
            "completed repair response",
        );
    }

    /// Issue a `tree:req:{session_id}` repair request for an
    /// unknown hash, targeted at a random ALIVE peer. Coalesces
    /// duplicates: while a session is already in flight for the
    /// same `(model_id, tree_kind)`, subsequent unknown-hash
    /// callbacks are dropped — the in-flight request asks for the
    /// whole tree (`cursor=None`), so it'll cover the new hash
    /// when the response lands (slice 5d).
    fn request_repair_for_unknown_hash(&self, model_id: &str, tree_kind: TreeKind, node_hash: u64) {
        let key = (model_id.to_string(), tree_kind);
        if self.outstanding_repairs.contains_key(&key) {
            trace!(
                model_id,
                kind = ?tree_kind,
                hash = node_hash,
                "repair already in flight, coalescing unknown-hash trigger",
            );
            return;
        }

        let alive = self.peers.alive_peers();
        let Some(target) = alive.choose(&mut rand::rng()) else {
            warn!(
                model_id,
                kind = ?tree_kind,
                hash = node_hash,
                "no alive peers to request repair from; will retry on next unknown delta",
            );
            return;
        };

        let request = TreeRepairRequest {
            session_id: Uuid::now_v7(),
            requester_peer_id: self.node_name.clone(),
            target_peer_id: target.clone(),
            model_id: model_id.to_string(),
            tree_kind,
            cursor: None,
            reason: RepairReason::UnknownHash(node_hash),
        };

        let bytes = match bincode::serialize(&request) {
            Ok(bytes) => Bytes::from(bytes),
            Err(err) => {
                // Schema is fixed-shape — should be unreachable.
                warn!(model_id, %err, "failed to serialize tree repair request");
                return;
            }
        };

        let stream_key = format!("{REPAIR_REQUEST_PREFIX}{}", request.session_id);
        self.tree_repair_requests
            .publish_to(target, &stream_key, bytes);

        // Record the in-flight session AFTER publish so a publish
        // that panics on a programming error doesn't leave a
        // ghost entry blocking future requests.
        self.outstanding_repairs.insert(
            key,
            RepairProgress::new(request.session_id, target.clone(), Instant::now()),
        );

        debug!(
            model_id,
            kind = ?tree_kind,
            hash = node_hash,
            target = %target,
            session_id = %request.session_id,
            "sent tree repair request",
        );
    }

    /// Decode an incoming `tree:page:` payload, validate it
    /// against the in-flight session, dedupe by `page_index`,
    /// apply via [`TreeHandle::apply_repair_page`], and clean up
    /// on terminal page. Stale pages (no matching session, or
    /// session_id / target peer mismatch) are dropped at debug
    /// level — the wire layer does not provide authenticated
    /// sender identity, so the (session_id, target_peer_id) tuple
    /// stored at request time is the strongest defense available
    /// against a delayed retry response from a different peer
    /// stomping on a fresh session.
    fn handle_incoming_repair_page(&self, fragments: &[Bytes]) {
        let total: usize = fragments.iter().map(Bytes::len).sum();
        let mut bytes = Vec::with_capacity(total);
        for frag in fragments {
            bytes.extend_from_slice(frag);
        }
        let page: TreeRepairPage = match bincode::deserialize(&bytes) {
            Ok(p) => p,
            Err(err) => {
                warn!(%err, "failed to decode TreeRepairPage");
                return;
            }
        };

        let key = (page.model_id.clone(), page.tree_kind);
        let mut entry = match self.outstanding_repairs.get_mut(&key) {
            Some(e) => e,
            None => {
                debug!(
                    model_id = %page.model_id,
                    kind = ?page.tree_kind,
                    session_id = %page.session_id,
                    page_index = page.page_index,
                    "received tree:page: with no matching outstanding session; dropping",
                );
                return;
            }
        };

        if entry.session_id != page.session_id {
            debug!(
                model_id = %page.model_id,
                kind = ?page.tree_kind,
                page_session_id = %page.session_id,
                live_session_id = %entry.session_id,
                page_index = page.page_index,
                "tree:page: session_id does not match the live session; dropping (likely a stale retry response)",
            );
            return;
        }

        if entry.applied.contains_key(&page.page_index) {
            trace!(
                model_id = %page.model_id,
                kind = ?page.tree_kind,
                session_id = %page.session_id,
                page_index = page.page_index,
                "duplicate tree:page: already applied; ignoring",
            );
            return;
        }

        let applied = self.tree.apply_repair_page(&page);
        entry
            .applied
            .insert(page.page_index, page.next_cursor.clone());
        entry.last_activity_at = Instant::now();
        let is_last = page.is_last;
        if is_last {
            entry.terminal_page_index = Some(page.page_index);
        }
        let is_complete = entry.is_contiguously_complete();

        debug!(
            model_id = %page.model_id,
            kind = ?page.tree_kind,
            session_id = %page.session_id,
            page_index = page.page_index,
            entries_in_page = page.entries.len(),
            entries_applied = applied,
            is_last,
            is_complete,
            "applied tree:page:",
        );

        // Drop the entry guard before mutating the map, otherwise
        // `remove()` would deadlock on the held shard write lock.
        drop(entry);

        if is_complete {
            if self
                .outstanding_repairs
                .remove_if(&key, |_, progress| progress.session_id == page.session_id)
                .is_some()
            {
                debug!(
                    model_id = %page.model_id,
                    kind = ?page.tree_kind,
                    session_id = %page.session_id,
                    "repair session complete",
                );
            } else {
                trace!(
                    model_id = %page.model_id,
                    kind = ?page.tree_kind,
                    session_id = %page.session_id,
                    "repair session completed but current progress changed before cleanup",
                );
            }
        }
    }

    /// Scan `outstanding_repairs` for sessions older than
    /// `retry_timeout` and reissue them (up to `max_retries`).
    /// Reissued requests get a fresh `session_id` and carry the
    /// last contiguous-applied cursor. Sessions exceeding the
    /// retry budget are removed from `outstanding_repairs` with
    /// an `error!` log; the next unknown-hash delta restarts
    /// from scratch.
    ///
    /// Called periodically by the spawned retry task; tests
    /// invoke it directly with synthetic timestamps.
    fn scan_for_retries(&self, now: Instant, retry_timeout: Duration, max_retries: u32) {
        // Snapshot keys that look stale to avoid holding shard
        // locks across publish_to / new RepairProgress creation.
        let stale_keys: Vec<(String, TreeKind)> = self
            .outstanding_repairs
            .iter()
            .filter(|e| {
                let p = e.value();
                now.duration_since(p.last_activity_at) >= retry_timeout
            })
            .map(|e| e.key().clone())
            .collect();

        for key in stale_keys {
            let (session_id, cursor, retry_count, prev_target, started_at, last_activity_at) =
                match self.outstanding_repairs.get(&key) {
                    Some(p) => (
                        p.session_id,
                        p.contiguous_cursor(),
                        p.retry_count,
                        p.target_peer_id.clone(),
                        p.started_at,
                        p.last_activity_at,
                    ),
                    None => continue,
                };
            if now.duration_since(last_activity_at) < retry_timeout {
                continue;
            }
            let session_age_ms = now.duration_since(started_at).as_millis();

            if retry_count >= max_retries {
                if self
                    .outstanding_repairs
                    .remove_if(&key, |_, progress| {
                        progress.session_id == session_id
                            && progress.retry_count >= max_retries
                            && now.duration_since(progress.last_activity_at) >= retry_timeout
                    })
                    .is_some()
                {
                    error!(
                        model_id = %key.0,
                        kind = ?key.1,
                        retry_count,
                        session_age_ms,
                        "tree repair session exceeded max retries; giving up. \
                         Next unknown-hash delta will restart from scratch.",
                    );
                }
                continue;
            }

            // Prefer a different peer on retry; fall back to the
            // full alive set if `prev_target` was the only peer.
            let mut alive = self.peers.alive_peers();
            alive.retain(|p| p != &prev_target);
            if alive.is_empty() {
                alive = self.peers.alive_peers();
            }
            let Some(target) = alive.choose(&mut rand::rng()) else {
                debug!(
                    model_id = %key.0,
                    kind = ?key.1,
                    "no alive peers to retry repair request; will re-check next scan",
                );
                continue;
            };

            let new_session_id = Uuid::now_v7();
            let request = TreeRepairRequest {
                session_id: new_session_id,
                requester_peer_id: self.node_name.clone(),
                target_peer_id: target.clone(),
                model_id: key.0.clone(),
                tree_kind: key.1,
                cursor,
                reason: RepairReason::UnknownHash(0),
            };

            let bytes = match bincode::serialize(&request) {
                Ok(b) => Bytes::from(b),
                Err(err) => {
                    warn!(model_id = %key.0, %err, "failed to serialize retry repair request");
                    continue;
                }
            };
            let stream_key = format!("{REPAIR_REQUEST_PREFIX}{new_session_id}");
            self.tree_repair_requests
                .publish_to(target, &stream_key, bytes);

            // Replace progress only if the same stale session is
            // still current. If a page made progress or completion
            // won the race after our snapshot, the extra retry
            // request is harmless: its fresh session_id has no live
            // progress entry, so responses are dropped as stale.
            let replaced = self
                .outstanding_repairs
                .get_mut(&key)
                .is_some_and(|mut progress| {
                    if progress.session_id != session_id
                        || progress.retry_count != retry_count
                        || now.duration_since(progress.last_activity_at) < retry_timeout
                    {
                        return false;
                    }
                    *progress = RepairProgress {
                        session_id: new_session_id,
                        target_peer_id: target.clone(),
                        started_at: now,
                        last_activity_at: now,
                        retry_count: retry_count + 1,
                        applied: BTreeMap::new(),
                        terminal_page_index: None,
                    };
                    true
                });

            if replaced {
                debug!(
                    model_id = %key.0,
                    kind = ?key.1,
                    target = %target,
                    new_session_id = %new_session_id,
                    retry_count = retry_count + 1,
                    session_age_ms,
                    "reissued repair request",
                );
            } else {
                trace!(
                    model_id = %key.0,
                    kind = ?key.1,
                    new_session_id = %new_session_id,
                    "skipped retry progress replacement because session changed or made progress",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
