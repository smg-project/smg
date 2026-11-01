//! MeshKV: Generic, application-agnostic mesh transport with explicit namespace handles.
//!
//! Provides two explicit namespace types:
//! - `CrdtNamespace` for durable, mergeable state (workers, policies, rate limits, config)
//! - `StreamNamespace` for ephemeral, lossy, application-regenerated traffic (tenant deltas, tree repair)
//!

use std::{
    collections::{HashSet, VecDeque},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use bytes::Bytes;
use dashmap::{mapref::entry::Entry as DashMapEntry, DashMap};
use parking_lot::RwLock;
use tokio::sync::mpsc;

use crate::{
    crdt_kv::{CrdtOrMap, MergeStrategy, Operation, OperationLog},
    transport::chunk_assembler::ChunkAssembler,
};

// ============================================================================
// Type Definitions
// ============================================================================

/// Routing mode for stream namespaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamRouting {
    /// Send to all connected peers (e.g., tenant deltas).
    Broadcast,
    /// Send to exactly one peer (e.g., tree repair requests/pages).
    Targeted,
}

/// Configuration for a stream namespace.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum bytes buffered before backpressure (oldest entries dropped).
    pub max_buffer_bytes: usize,
    /// Routing mode: broadcast to all peers or targeted to one peer.
    pub routing: StreamRouting,
}

// ============================================================================
// Subscription
// ============================================================================

/// A subscription to changes for keys matching a prefix.
/// Delivers via a bounded mpsc channel. Capacity is per-prefix:
/// - worker:  1000
/// - policy:  100
/// - rl:      100
/// - config:  100
/// - td:      50
/// - tree:req: 100
/// - tree:page: 32
pub struct Subscription {
    /// Receives (key, value) pairs. `None` value means the key was deleted.
    pub receiver: mpsc::Receiver<SubscriptionEvent>,
}

/// Function signature for stream drain callbacks. Called exactly once per
/// gossip round. Returns accumulated entries to be sent in this round's
/// batch. Values are `Bytes` so fan-out to N peers is an Arc refcount bump
/// per peer rather than N heap copies.
pub type StreamDrainFn = Box<dyn Fn() -> Vec<(String, Bytes)> + Send + Sync>;

/// Handle returned by `register_drain`. Dropping unregisters the drain callback.
/// Use `drop(handle)` to explicitly unregister.
pub struct DrainHandle {
    prefix: String,
    drain_registry: Arc<DrainRegistry>,
}

impl Drop for DrainHandle {
    fn drop(&mut self) {
        self.drain_registry.remove(&self.prefix);
    }
}

// ============================================================================
// Subscriber Registry (internal)
// ============================================================================

/// A single subscription event: (key, value). `None` value means deletion.
///
/// Values are a `Vec<Bytes>` of zero-copy fragments: a 1-element Vec for
/// single-value writes, an N-element Vec for reassembled chunked receives
/// (each element wraps one chunk's original allocation). The fragmented
/// shape avoids the 2× peak a contiguous reassembly would impose when a
/// near-cap multi-chunk value completes.
type SubscriptionEvent = (String, Option<Vec<Bytes>>);

/// Tracks all active subscriptions by prefix.
struct SubscriberRegistry {
    /// prefix -> list of senders. Multiple subscribers can watch the same prefix.
    subscribers: DashMap<String, Vec<mpsc::Sender<SubscriptionEvent>>>,
}

impl SubscriberRegistry {
    fn new() -> Self {
        Self {
            subscribers: DashMap::new(),
        }
    }

    fn register(&self, prefix: &str, tx: mpsc::Sender<SubscriptionEvent>) {
        self.subscribers
            .entry(prefix.to_string())
            .or_default()
            .push(tx);
    }

    /// Notify all subscribers whose prefix matches the given key.
    /// Uses try_send to never block the gossip loop.
    /// `value` is `Some(bytes)` for puts, `None` for deletes.
    fn notify(&self, key: &str, value: Option<Vec<Bytes>>) {
        for entry in &self.subscribers {
            let prefix = entry.key();
            if key.starts_with(prefix.as_str()) {
                // try_send: never block. If full, entry is dropped.
                // For CRDT: watermark not advanced, resent next round.
                // For Stream: dropped permanently (ephemeral).
                for tx in entry.value() {
                    let _ = tx.try_send((key.to_string(), value.clone()));
                }
            }
        }
    }
}

// ============================================================================
// Drain Registry (internal)
// ============================================================================

/// Tracks registered drain callbacks for stream namespaces.
struct DrainRegistry {
    drains: DashMap<String, Arc<StreamDrainFn>>,
}

impl DrainRegistry {
    fn new() -> Self {
        Self {
            drains: DashMap::new(),
        }
    }

    fn register(&self, prefix: &str, drain: StreamDrainFn) {
        let entry = self.drains.entry(prefix.to_string());
        assert!(
            matches!(&entry, DashMapEntry::Vacant(_)),
            "drain already registered for prefix '{prefix}'"
        );
        entry.or_insert(Arc::new(drain));
    }

    fn remove(&self, prefix: &str) {
        self.drains.remove(prefix);
    }

    /// Call all registered drains. Returns accumulated entries.
    /// Called exactly once per gossip round.
    fn drain_all(&self) -> Vec<(String, Bytes)> {
        let mut all_entries = Vec::new();
        for entry in &self.drains {
            let drain_fn = entry.value();
            all_entries.extend(drain_fn());
        }
        all_entries
    }
}

// ============================================================================
// Prefix Configuration (internal)
// ============================================================================

/// Per-prefix subscriber channel capacity.
fn subscriber_capacity_for_prefix(prefix: &str) -> usize {
    match prefix {
        "worker:" => 1000,
        "policy:" => 100,
        "rl:" => 100,
        "config:" => 100,
        "td:" => 50,
        "tree:req:" => 100,
        "tree:page:" => 32,
        _ => 100, // default for unknown prefixes
    }
}

// ============================================================================
// CrdtNamespace
// ============================================================================

/// Handle for durable, mergeable state. Scoped to a key prefix.
/// Provides put/get/delete/keys/subscribe.
pub struct CrdtNamespace {
    prefix: String,
    store: Arc<CrdtOrMap>,
    subscriber_registry: Arc<SubscriberRegistry>,
    merge_strategy: MergeStrategy,
}

impl CrdtNamespace {
    /// Insert or update a key-value pair. The key must start with this
    /// namespace's prefix.
    pub fn put(&self, key: &str, value: Vec<u8>) {
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.store.insert(key.to_string(), value);
        // Notify with the canonical post-insert value (matching `get`), not the
        // raw caller payload: for normalizing engines (e.g. `rl:`) the stored
        // shard differs from the input, so this keeps the local-write and
        // remote-merge notification shapes identical.
        let canonical = self.store.get(key);
        self.subscriber_registry
            .notify(key, canonical.map(|v| vec![Bytes::from(v)]));
    }

    /// Get the current value for a key, or None if not present or tombstoned.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.store.get(key)
    }

    /// Delete a key by writing a tombstone. The tombstone propagates via gossip
    /// and wins by higher version. Actual removal happens after GC grace period.
    /// Subscribers receive `(key, None)` to indicate deletion.
    pub fn delete(&self, key: &str) {
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        self.store.remove(key);
        self.subscriber_registry.notify(key, None);
    }

    /// List all live keys matching a sub-prefix within this namespace.
    pub fn keys(&self, sub_prefix: &str) -> Vec<String> {
        let full_prefix = format!("{}{}", self.prefix, sub_prefix);
        self.store
            .keys()
            .into_iter()
            .filter(|k| k.starts_with(&full_prefix))
            .collect()
    }

    /// Subscribe to changes for keys matching a sub-prefix within this namespace.
    /// Channel capacity is determined by the namespace prefix.
    pub fn subscribe(&self, sub_prefix: &str) -> Subscription {
        let full_prefix = format!("{}{}", self.prefix, sub_prefix);
        let capacity = subscriber_capacity_for_prefix(&self.prefix);
        let (tx, rx) = mpsc::channel(capacity);
        self.subscriber_registry.register(&full_prefix, tx);
        Subscription { receiver: rx }
    }

    /// Get the merge strategy for this namespace.
    pub fn merge_strategy(&self) -> &MergeStrategy {
        &self.merge_strategy
    }

    /// Get the prefix for this namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ============================================================================
// StreamNamespace
// ============================================================================

/// Handle for ephemeral, lossy, application-regenerated traffic. Scoped to a
/// key prefix with a fixed routing mode (Broadcast or Targeted).
pub struct StreamNamespace {
    prefix: String,
    routing: StreamRouting,
    max_buffer_bytes: usize,
    /// Targeted entries: (target_peer_id, key, value). VecDeque for O(1) FIFO eviction.
    targeted_buffer: parking_lot::Mutex<VecDeque<(String, String, Bytes)>>,
    /// Current total bytes in the targeted buffer.
    targeted_buffer_bytes: AtomicUsize,
    subscriber_registry: Arc<SubscriberRegistry>,
    drain_registry: Arc<DrainRegistry>,
}

impl StreamNamespace {
    /// Publish a value to exactly one peer (Targeted namespaces only).
    /// If the buffer exceeds `max_buffer_bytes`, the oldest entries are dropped.
    pub fn publish_to(&self, peer_id: &str, key: &str, value: Bytes) {
        assert_eq!(
            self.routing,
            StreamRouting::Targeted,
            "publish_to() is only valid on Targeted namespaces, not Broadcast (prefix: '{}')",
            self.prefix
        );
        assert!(
            key.starts_with(&self.prefix),
            "key '{key}' does not match prefix '{}'",
            self.prefix
        );
        let value_len = value.len();
        let mut buf = self.targeted_buffer.lock();
        buf.push_back((peer_id.to_string(), key.to_string(), value));
        self.targeted_buffer_bytes
            .fetch_add(value_len, Ordering::Relaxed);
        // Drop the oldest entries (FIFO) while over limit. O(1) per pop_front.
        while self.targeted_buffer_bytes.load(Ordering::Relaxed) > self.max_buffer_bytes
            && !buf.is_empty()
        {
            if let Some((_, _, dropped)) = buf.pop_front() {
                self.targeted_buffer_bytes
                    .fetch_sub(dropped.len(), Ordering::Relaxed);
            }
        }
    }

    /// Subscribe to messages for keys matching a sub-prefix within this namespace.
    pub fn subscribe(&self, sub_prefix: &str) -> Subscription {
        let full_prefix = format!("{}{}", self.prefix, sub_prefix);
        let capacity = subscriber_capacity_for_prefix(&self.prefix);
        let (tx, rx) = mpsc::channel(capacity);
        self.subscriber_registry.register(&full_prefix, tx);
        Subscription { receiver: rx }
    }

    /// Register a drain callback. Called exactly once per gossip round by the
    /// centralized collector. Only valid on Broadcast namespaces — drain entries
    /// carry (key, value) without peer_id, so targeted routing is not possible.
    /// Returns a DrainHandle for unregistration.
    pub fn register_drain(&self, drain: StreamDrainFn) -> DrainHandle {
        assert_eq!(
            self.routing,
            StreamRouting::Broadcast,
            "register_drain() is only valid on Broadcast namespaces (prefix: '{}')",
            self.prefix
        );
        self.drain_registry.register(&self.prefix, drain);
        DrainHandle {
            prefix: self.prefix.clone(),
            drain_registry: self.drain_registry.clone(),
        }
    }

    /// Drain all targeted buffer entries. Returns (peer_id, key, value) tuples
    /// and resets the buffer. Called by the gossip loop once per round.
    pub fn drain_targeted_buffer(&self) -> Vec<(String, String, Bytes)> {
        let mut buf = self.targeted_buffer.lock();
        self.targeted_buffer_bytes.store(0, Ordering::Relaxed);
        std::mem::take(&mut *buf).into()
    }

    /// Get the routing mode for this namespace.
    pub fn routing(&self) -> StreamRouting {
        self.routing
    }

    /// Get the prefix for this namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ============================================================================
// MeshKV
// ============================================================================

/// A batch of entries collected once per gossip round by the central collector.
#[derive(Debug, Default)]
pub struct RoundBatch {
    /// Targeted stream entries: (peer_id, key, value). Sent to one specific peer.
    pub targeted_entries: Vec<(String, String, Bytes)>,
    /// Entries from registered drain callbacks (e.g., TreeSyncAdapter pending deltas).
    /// Broadcast traffic (td:*) flows through this path, not through a buffer.
    /// Values are `Bytes` so per-peer senders clone by Arc-refcount bump when
    /// fanning out, not by a full heap copy per peer.
    pub drain_entries: Vec<(String, Bytes)>,
    /// Shared snapshot of the CRDT operation log for this round. The `Arc`
    /// is reused across rounds while no engine's log mutates, so idle
    /// rounds clone no op data; per-peer senders filter it through their
    /// send watermarks. Merge is idempotent by op-id.
    pub crdt_ops: Arc<OperationLog>,
}

/// Generic, application-agnostic mesh transport. Provides explicit namespace
/// handles for CRDT and stream modes. Application code MUST use namespace
/// handles, not MeshKV directly.
impl std::fmt::Debug for MeshKV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshKV")
            .field("server_name", &self.server_name)
            .field("replica_id", &self.replica_id)
            .finish_non_exhaustive()
    }
}

pub struct MeshKV {
    /// CRDT store shared across all CRDT namespaces.
    store: Arc<CrdtOrMap>,
    /// The configured prefixes, so a second configuration of one fails closed.
    configured_prefixes: RwLock<HashSet<String>>,
    /// Stream namespaces, stored for round batch collection.
    stream_namespaces: RwLock<Vec<Arc<StreamNamespace>>>,
    /// Shared subscriber registry.
    subscriber_registry: Arc<SubscriberRegistry>,
    /// Shared drain registry.
    drain_registry: Arc<DrainRegistry>,
    /// Receiver-side chunk reassembly buffer shared across all inbound
    /// SyncStream connections on this node.
    chunk_assembler: Arc<ChunkAssembler>,
    /// Auto-registered `config:` CRDT namespace (LastWriterWins).
    /// Pre-created at `new()` time so every gateway (admin API,
    /// middleware, adapters) can read and write config without the
    /// application having to remember to `configure_crdt_prefix`
    /// explicitly. Also used by the gossip receive path to mirror
    /// incoming v1 `StoreType::App` entries into `config:{key}` for
    /// rolling-upgrade compatibility.
    configs: Arc<CrdtNamespace>,
    /// Server name for this node (used to derive replica_id).
    server_name: String,
    /// Replica ID: hash(server_name) as u64.
    replica_id: u64,
}

impl MeshKV {
    /// Create a new MeshKV instance. Auto-registers the `config:`
    /// CRDT namespace (LastWriterWins) so gateway readers and the
    /// gossip receive path can always reach the config store via
    /// `mesh_kv.configs()` without a separate wiring step.
    pub fn new(server_name: String) -> Self {
        let replica_id = Self::derive_replica_id(&server_name);
        let store = Arc::new(CrdtOrMap::new());
        store.register_merge_strategy("config:".to_string(), MergeStrategy::LastWriterWins);
        let subscriber_registry = Arc::new(SubscriberRegistry::new());
        let configured_prefixes = HashSet::from(["config:".to_string()]);
        let configs = Arc::new(CrdtNamespace {
            prefix: "config:".to_string(),
            store: store.clone(),
            subscriber_registry: subscriber_registry.clone(),
            merge_strategy: MergeStrategy::LastWriterWins,
        });
        Self {
            store,
            configured_prefixes: RwLock::new(configured_prefixes),
            stream_namespaces: RwLock::new(Vec::new()),
            subscriber_registry,
            drain_registry: Arc::new(DrainRegistry::new()),
            chunk_assembler: Arc::new(ChunkAssembler::new()),
            configs,
            server_name,
            replica_id,
        }
    }

    /// Shared handle to the auto-registered `config:` CRDT namespace.
    /// Gateway middleware, admin API, and the gossip receive path use
    /// this to read and write cluster-wide configuration (rate-limit
    /// limits, feature flags, etc.) with LastWriterWins merge.
    pub fn configs(&self) -> Arc<CrdtNamespace> {
        self.configs.clone()
    }

    /// Handle to the node-wide chunk reassembly buffer. Used by the
    /// gossip receive path to route `StreamBatch` chunks through
    /// reassembly before firing subscribers.
    pub(crate) fn chunk_assembler(&self) -> Arc<ChunkAssembler> {
        self.chunk_assembler.clone()
    }

    /// Reclaim tombstone metadata older than the default grace period
    /// across every CRDT engine. Returns the number reclaimed.
    ///
    /// Deliberately not driven anywhere: time-based collection is unsound.
    /// A peer absent longer than the grace still holds the deleted key's
    /// older insert in its op-log and replays it on reconnect; with the
    /// tombstone metadata gone, the stale insert lands on an empty entry
    /// and resurrects the key. Collection is only safe at causal stability
    /// — every live peer acked past the tombstone and absent peers
    /// declared dead — so the caller must be the dead-node cleanup layer.
    pub fn gc_tombstones(&self) -> usize {
        self.store.gc_tombstones()
    }

    /// Fire subscribers whose prefix matches `key`. Used by the gossip
    /// receive path when a chunked value completes (or a single-chunk
    /// entry arrives), so handlers can deliver into adapter-owned
    /// mpsc channels without reaching into internal registries.
    pub(crate) fn notify_subscribers(&self, key: &str, value: Option<Vec<Bytes>>) {
        self.subscriber_registry.notify(key, value);
    }

    /// Derive replica_id from server_name using blake3 hash truncated to u64.
    /// Collision risk: 2^-64 per pair — negligible.
    fn derive_replica_id(server_name: &str) -> u64 {
        let hash = blake3::hash(server_name.as_bytes());
        // blake3::Hash::as_bytes() returns &[u8; 32], so first_chunk is infallible.
        let bytes: &[u8; 8] = hash.as_bytes().first_chunk().unwrap_or(&[0; 8]);
        u64::from_le_bytes(*bytes)
    }

    /// Configure a CRDT namespace for a key prefix. Returns a handle scoped to
    /// that prefix.
    ///
    /// # Panics
    /// Panics if the prefix is already configured (fail-closed).
    pub fn configure_crdt_prefix(
        &self,
        prefix: &str,
        merge_strategy: MergeStrategy,
    ) -> Arc<CrdtNamespace> {
        {
            let mut prefixes = self.configured_prefixes.write();
            assert!(
                !prefixes.contains(prefix),
                "Prefix '{prefix}' is already configured. Each prefix must be configured exactly once."
            );
            prefixes.insert(prefix.to_string());
        }
        self.store
            .register_merge_strategy(prefix.to_string(), merge_strategy);

        Arc::new(CrdtNamespace {
            prefix: prefix.to_string(),
            store: self.store.clone(),
            subscriber_registry: self.subscriber_registry.clone(),
            merge_strategy,
        })
    }

    /// Configure a stream namespace for a key prefix. Returns a handle scoped to
    /// that prefix.
    ///
    /// # Panics
    /// Panics if the prefix is already configured (fail-closed).
    pub fn configure_stream_prefix(
        &self,
        prefix: &str,
        config: StreamConfig,
    ) -> Arc<StreamNamespace> {
        {
            let mut prefixes = self.configured_prefixes.write();
            assert!(
                !prefixes.contains(prefix),
                "Prefix '{prefix}' is already configured. Each prefix must be configured exactly once."
            );
            prefixes.insert(prefix.to_string());
        }

        let ns = Arc::new(StreamNamespace {
            prefix: prefix.to_string(),
            routing: config.routing,
            max_buffer_bytes: config.max_buffer_bytes,
            targeted_buffer: parking_lot::Mutex::new(VecDeque::new()),
            targeted_buffer_bytes: AtomicUsize::new(0),
            subscriber_registry: self.subscriber_registry.clone(),
            drain_registry: self.drain_registry.clone(),
        });
        self.stream_namespaces.write().push(ns.clone());
        ns
    }

    /// Get the server name for this node.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Get the replica ID for this node.
    pub fn replica_id(&self) -> u64 {
        self.replica_id
    }

    /// Collect all stream entries for one gossip round. Called exactly once
    /// per round by the centralized gossip loop. Drains all stream buffers
    /// and calls all registered drain callbacks.
    pub fn collect_round_batch(&self) -> RoundBatch {
        let mut targeted_entries = Vec::new();

        // Drain targeted stream namespace buffers.
        for ns in self.stream_namespaces.read().iter() {
            targeted_entries.extend(ns.drain_targeted_buffer());
        }

        // Call registered drain callbacks (e.g., TreeSyncAdapter pending deltas).
        // Broadcast traffic (td:*) flows through this path.
        let drain_entries = self.drain_registry.drain_all();

        // Shared CRDT op-log snapshot for the per-peer senders; reused
        // across rounds while no engine's log mutates.
        let crdt_ops = self.store.operation_log_snapshot();

        RoundBatch {
            targeted_entries,
            drain_entries,
            crdt_ops,
        }
    }

    /// Merge a batch of CRDT operations received from a peer into the local
    /// store and fire subscribers for keys whose live value changed. Used by
    /// the gossip receive path (`dispatch_crdt_batch`). Merge is idempotent by
    /// op-id, so re-applying an already-seen batch is a no-op and fires no
    /// subscriber events. Notifications carry the canonical post-merge value
    /// (matching `get`), so remote-merge subscribers see the same value shape
    /// as local writes.
    pub(crate) fn merge_crdt_ops(&self, ops: Vec<Operation>) {
        let mut log = OperationLog::new();
        for op in ops {
            log.append(op);
        }
        for change in self.store.merge(&log) {
            self.subscriber_registry
                .notify(&change.key, change.value.map(|v| vec![Bytes::from(v)]));
        }
    }

    /// Check if a prefix has been configured.
    pub fn is_prefix_configured(&self, prefix: &str) -> bool {
        self.configured_prefixes.read().contains(prefix)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_derive_replica_id_deterministic() {
        let id1 = MeshKV::derive_replica_id("gateway-1");
        let id2 = MeshKV::derive_replica_id("gateway-1");
        assert_eq!(id1, id2, "Same server_name must produce same replica_id");
    }

    #[test]
    fn test_derive_replica_id_different_names() {
        let id1 = MeshKV::derive_replica_id("gateway-1");
        let id2 = MeshKV::derive_replica_id("gateway-2");
        assert_ne!(
            id1, id2,
            "Different server_names should produce different replica_ids"
        );
    }

    #[test]
    fn test_configure_crdt_prefix() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        assert_eq!(ns.prefix(), "worker:");
        assert!(kv.is_prefix_configured("worker:"));
    }

    #[test]
    fn test_configs_prefix_auto_registered() {
        // The `config:` prefix is always registered at construction;
        // gateway code must not have to call configure_crdt_prefix for
        // it, and a redundant call would panic per the one-configure-
        // per-prefix rule.
        let kv = MeshKV::new("test-node".to_string());
        assert!(kv.is_prefix_configured("config:"));
        let configs = kv.configs();
        assert_eq!(configs.prefix(), "config:");
    }

    #[test]
    fn test_configs_put_get_round_trip() {
        let kv = MeshKV::new("test-node".to_string());
        let configs = kv.configs();
        configs.put("config:rate_limit", b"100".to_vec());
        assert_eq!(
            configs.get("config:rate_limit"),
            Some(b"100".to_vec()),
            "config namespace round-trips through the shared CRDT store"
        );
    }

    #[test]
    #[should_panic(expected = "already configured")]
    fn test_configure_config_prefix_twice_panics() {
        // Application code must not try to reconfigure `config:` — the
        // auto-registration at MeshKV::new() already owns the slot.
        // This guards against accidental reconfiguration that would
        // replace the merge strategy or subscriber capacity.
        let kv = MeshKV::new("test-node".to_string());
        kv.configure_crdt_prefix("config:", MergeStrategy::LastWriterWins);
    }

    #[test]
    fn test_configure_stream_prefix() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "td:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        );
        assert_eq!(ns.prefix(), "td:");
        assert_eq!(ns.routing(), StreamRouting::Broadcast);
        assert!(kv.is_prefix_configured("td:"));
    }

    #[test]
    #[should_panic(expected = "already configured")]
    fn test_duplicate_prefix_panics() {
        let kv = MeshKV::new("test-node".to_string());
        kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins); // panics
    }

    #[test]
    #[should_panic(expected = "already configured")]
    fn test_duplicate_prefix_across_modes_panics() {
        let kv = MeshKV::new("test-node".to_string());
        kv.configure_crdt_prefix("data:", MergeStrategy::LastWriterWins);
        kv.configure_stream_prefix(
            "data:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        ); // panics
    }

    #[test]
    fn test_crdt_put_get() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        ns.put("worker:7", b"healthy".to_vec());
        let val = ns.get("worker:7");
        assert_eq!(val, Some(b"healthy".to_vec()));
    }

    #[test]
    fn test_crdt_delete() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        ns.put("worker:7", b"healthy".to_vec());
        ns.delete("worker:7");
        assert_eq!(ns.get("worker:7"), None);
    }

    #[test]
    fn test_crdt_keys() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        ns.put("worker:1", b"a".to_vec());
        ns.put("worker:2", b"b".to_vec());
        ns.put("worker:3", b"c".to_vec());

        let mut keys = ns.keys("");
        keys.sort();
        assert_eq!(keys, vec!["worker:1", "worker:2", "worker:3"]);
    }

    #[test]
    #[should_panic(expected = "does not match prefix")]
    fn test_crdt_put_wrong_prefix_panics() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        ns.put("policy:1", b"wrong".to_vec()); // panics
    }

    #[test]
    #[should_panic(expected = "only valid on Targeted")]
    fn test_stream_publish_to_on_broadcast_panics() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "td:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        );
        ns.publish_to("peer-1", "td:model-x", Bytes::from("data")); // panics
    }

    #[tokio::test]
    async fn test_crdt_subscribe() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        let mut sub = ns.subscribe("");
        ns.put("worker:7", b"healthy".to_vec());

        let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
            .await
            .expect("timeout")
            .expect("channel closed");

        assert_eq!(key, "worker:7");
        let frags = value.expect("put yields Some");
        assert_eq!(frags.len(), 1, "single local write is a single fragment");
        assert_eq!(frags[0].as_ref(), b"healthy");
    }

    #[tokio::test]
    async fn test_crdt_subscribe_delete() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

        let mut sub = ns.subscribe("");
        ns.put("worker:7", b"healthy".to_vec());
        ns.delete("worker:7");

        // First event: put
        let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
            .await
            .expect("timeout")
            .expect("channel closed");
        assert_eq!(key, "worker:7");
        assert!(value.is_some());

        // Second event: delete
        let (key, value) = tokio::time::timeout(Duration::from_millis(100), sub.receiver.recv())
            .await
            .expect("timeout")
            .expect("channel closed");
        assert_eq!(key, "worker:7");
        assert!(value.is_none(), "delete should notify with None");
    }

    #[test]
    fn test_targeted_backpressure_drops_oldest() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "tree:page:",
            StreamConfig {
                max_buffer_bytes: 20, // tiny limit
                routing: StreamRouting::Targeted,
            },
        );

        ns.publish_to("peer-A", "tree:page:m1", Bytes::from("aaaaaaaaaa")); // 10 bytes
        ns.publish_to("peer-A", "tree:page:m2", Bytes::from("bbbbbbbbbb")); // 10 bytes
        ns.publish_to("peer-A", "tree:page:m3", Bytes::from("cccccccccc")); // over limit

        let drained = ns.drain_targeted_buffer();
        let total_bytes: usize = drained.iter().map(|(_, _, v)| v.len()).sum();
        assert!(
            total_bytes <= 20,
            "buffer should be within limit, got {total_bytes}"
        );
        // Oldest entry (m1) should have been dropped, keeping m2 and m3.
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].1, "tree:page:m2");
        assert_eq!(drained[1].1, "tree:page:m3");
    }

    #[test]
    fn test_drain_targeted_buffer() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "tree:page:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Targeted,
            },
        );

        ns.publish_to("peer-A", "tree:page:m1", Bytes::from("page1"));
        ns.publish_to("peer-B", "tree:page:m2", Bytes::from("page2"));

        let entries = ns.drain_targeted_buffer();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "peer-A");
        assert_eq!(entries[1].0, "peer-B");

        // Buffer should be empty after drain.
        let entries2 = ns.drain_targeted_buffer();
        assert!(entries2.is_empty());
    }

    #[test]
    fn test_collect_round_batch() {
        let kv = MeshKV::new("test-node".to_string());
        let targeted_ns = kv.configure_stream_prefix(
            "tree:page:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Targeted,
            },
        );

        targeted_ns.publish_to("peer-A", "tree:page:m1", Bytes::from("page"));

        let batch = kv.collect_round_batch();
        assert_eq!(batch.targeted_entries.len(), 1);
        assert_eq!(batch.targeted_entries[0].0, "peer-A");

        // Second collect should be empty (buffers drained).
        let batch2 = kv.collect_round_batch();
        assert!(batch2.targeted_entries.is_empty());
    }

    #[test]
    fn test_collect_round_batch_with_drain_callback() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "td:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        );

        let _handle = ns.register_drain(Box::new(|| {
            vec![(
                "td:from-drain".to_string(),
                Bytes::from_static(b"drain-data"),
            )]
        }));

        let batch = kv.collect_round_batch();
        assert_eq!(batch.drain_entries.len(), 1);
        assert_eq!(batch.drain_entries[0].0, "td:from-drain");
    }

    #[test]
    #[should_panic(expected = "drain already registered")]
    fn test_duplicate_drain_registration_panics() {
        let kv = MeshKV::new("test-node".to_string());
        let ns = kv.configure_stream_prefix(
            "td:",
            StreamConfig {
                max_buffer_bytes: 1024,
                routing: StreamRouting::Broadcast,
            },
        );

        let _h1 = ns.register_drain(Box::new(Vec::new));
        let _h2 = ns.register_drain(Box::new(Vec::new)); // panics
    }
}
