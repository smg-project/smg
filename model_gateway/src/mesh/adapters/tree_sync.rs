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
mod tests {
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

        fn set_repair_stream(
            &self,
            model_id: &str,
            tree_kind: TreeKind,
            entries: Vec<RepairEntry>,
        ) {
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
}
