//! The membership store: the worker row the handshake produces, the writes
//! that change it and the watch the gateways' pools derive from (design 5.1,
//! the workers row; 5.2, `WatchWorkers`).

use std::{collections::BTreeMap, time::Instant};

use crate::{
    lease::Membership,
    store::GatewayId,
    view::{fingerprint, Role, WireModes, WorkerId},
};

/// The stable identity of a registration: the key (the URL today) and the
/// incarnation the store numbered, so a same-key replacement is a new id and
/// every replica derives the same id from the same facts. Never `0`, which
/// reads as absent wherever an id travels.
pub fn identity(key: &str, incarnation: u64) -> WorkerId {
    identity_of(fingerprint(key.as_bytes()), incarnation)
}

/// [`identity`] from the key's fingerprint, for a caller that holds it
/// already: a host building a candidate row takes one pass over the URL for
/// the id and the row's placement key both.
pub fn identity_of(fingerprint: u64, incarnation: u64) -> WorkerId {
    let id = fingerprint ^ incarnation.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    WorkerId(if id == 0 { 1 } else { id })
}

/// The lifecycle state of a row, as the prober drives it: the registry's
/// status machine. Only `Ready` is routable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WorkerStatus {
    /// Registered, not yet proven healthy; probed until the first pass.
    #[default]
    Pending,
    Ready,
    /// Was ready and fails its probes; kept and probed, it may return.
    NotReady,
    /// A sustained liveness failure; removed when the deployment removes
    /// unhealthy workers, else probed in place.
    Failed,
    /// Leaving: in-flight requests settle, nothing new is placed.
    Draining,
}

/// The prober's settings for one worker, from its registration (the row's
/// own, else the deployment's defaults).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HealthPolicy {
    /// Consecutive failed probes before a ready worker leaves rotation;
    /// three times as many fail it.
    pub failure_threshold: u32,
    /// Consecutive passed probes before a worker returns.
    pub success_threshold: u32,
    pub check_interval_ms: u32,
    pub timeout_ms: u32,
    /// Whether the prober runs at all: a worker with its checks off is
    /// promoted by its connect signal alone.
    pub probes: bool,
}

impl Default for HealthPolicy {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            check_interval_ms: 60_000,
            timeout_ms: 5_000,
            probes: true,
        }
    }
}

/// The overload thresholds a worker's reports are judged against, resolved
/// at registration; a field left `None` is off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overload {
    /// KV usage at or above which the worker is overloaded, in permille.
    pub kv_usage_permille: Option<u16>,
    /// Queued requests at or above which the worker is overloaded.
    pub waiting_requests: Option<u32>,
}

impl Default for Overload {
    fn default() -> Self {
        Self {
            kv_usage_permille: Some(800),
            waiting_requests: Some(8),
        }
    }
}

/// How a worker's signals reach the store (the row's `signals`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Signals {
    /// KV events into the index (an events subscription).
    pub events: bool,
    /// Load records (a records-only subscription when no events are wanted).
    pub loads: bool,
    /// The scrape interval for an engine without a servicer; `0` for none.
    pub scrape_interval_ms: u32,
}

/// From where a worker is reachable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reachability {
    /// Every gateway dials it.
    #[default]
    Any,
    /// A host-local wire: one gateway holds the socket and relays its loads.
    Gateway(GatewayId),
}

/// What the registration handshake learned about a worker before its row
/// exists: the workers row of the design, in the host's own words.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkerFacts {
    /// The registration key: the URL a gateway dials, unique per row.
    pub key: String,
    /// The engine the row belongs to, equal for the ranks of one
    /// data-parallel engine and equal to `key` for a single-rank worker: the
    /// subscription is per engine, the rows per rank.
    pub engine: String,
    /// The store's number for this life of the key; a same-key upsert with
    /// another incarnation is a replacement.
    pub incarnation: u64,
    pub role: Role,
    pub wire_modes: WireModes,
    /// The engine family as the handshake named it; empty when unknown.
    pub engine_kind: String,
    /// The model ids the worker serves, aliases resolved.
    pub models: Vec<String>,
    pub dp_rank: u16,
    pub dp_size: u16,
    /// The labels of record: the registration's, over those discovered from
    /// the engine (`policy`, zone, the discovery provenance, the pairing
    /// descriptor's fields).
    pub labels: BTreeMap<String, String>,
    pub zone: Option<String>,
    /// The prefill-decode pairing descriptor, when the worker has one.
    pub pairing: Option<String>,
    /// The pairing class, derived by the store at each membership change;
    /// `0` outside any pairing.
    pub pair_class: u32,
    /// The engine's admission window (maximum running requests); `None`
    /// when it reports none.
    pub max_running: Option<u32>,
    pub kv_block_size: Option<u32>,
    pub bootstrap_port: Option<u16>,
    pub priority: i32,
    pub reachable_from: Reachability,
    pub signals: Signals,
    pub health: HealthPolicy,
    pub overload: Overload,
}

/// One worker as the store holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerRow {
    pub id: WorkerId,
    pub facts: WorkerFacts,
    pub status: WorkerStatus,
    /// Moves on every write to the row; the guard a racing write names.
    pub revision: u64,
    /// When the row was created on the host's clock.
    pub admitted_at: Instant,
}

/// The mutable facts of a row a patch may change (`PatchWorker`; today the
/// replace of a worker's properties); a field left `None` is kept.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkerPatch {
    pub labels: Option<BTreeMap<String, String>>,
    pub priority: Option<i32>,
    pub overload: Option<Overload>,
}

/// What a membership write produced: the event `WatchWorkers` carries to the
/// gateways and the replication stream to the standbys. The engine's
/// lifecycle takes it through [`MembershipChange::as_lifecycle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MembershipChange {
    Registered {
        id: WorkerId,
    },
    /// The same key with a new incarnation: the old id leaves, the new one
    /// enters, runtime state carried over where the registration matches.
    Replaced {
        old: WorkerId,
        new: WorkerId,
    },
    Removed {
        id: WorkerId,
    },
    StatusChanged {
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
    },
    /// The facts of a row changed without a new incarnation.
    Relabelled {
        id: WorkerId,
    },
    /// The worker's data-parallel width changed.
    Resized {
        id: WorkerId,
    },
}

impl MembershipChange {
    /// The engine's membership event for this change; a status change is a
    /// health signal, not a membership event, so it maps to none.
    pub fn as_lifecycle(&self) -> Option<Membership<'static>> {
        Some(match *self {
            Self::Registered { id } => Membership::Added(id),
            Self::Replaced { old, new } => Membership::Replaced { old, new },
            Self::Removed { id } => Membership::Removed(id),
            Self::Relabelled { id } => Membership::Relabelled(id),
            Self::Resized { id } => Membership::Resized(id),
            Self::StatusChanged { .. } => return None,
        })
    }

    /// The row the change is about (the new one of a replacement).
    pub fn id(&self) -> WorkerId {
        match *self {
            Self::Registered { id }
            | Self::Removed { id }
            | Self::StatusChanged { id, .. }
            | Self::Relabelled { id }
            | Self::Resized { id } => id,
            Self::Replaced { new, .. } => new,
        }
    }
}

/// A watcher's tail since the epoch it last saw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MembershipTail {
    /// The changes after the epoch, each with the epoch it made.
    Changes(Vec<(u64, MembershipChange)>),
    /// The epoch is past the store's buffer: re-read a snapshot, then tail
    /// from its epoch (a lagged registry subscriber reconciles the same way
    /// today).
    SnapshotRequired,
}

/// The membership store's write side and its watch.
///
/// Every write moves the store's epoch and the row's revision. A watcher
/// reads a snapshot with its epoch, then the changes since it; past the
/// store's buffer it is told to snapshot again. The snapshot carries the
/// facts the gateways' pools derive from (role, wire modes, engine kind,
/// reachability), never a pool list.
pub trait WorkersStore: Send + Sync {
    /// A registration or its refresh. A new key is `Registered`; the same key
    /// with another incarnation is `Replaced` (the old id leaves with its
    /// state); the same key and incarnation refresh the facts in place
    /// (`Resized` when the data-parallel width moved, else `Relabelled`).
    fn upsert(&self, facts: WorkerFacts) -> MembershipChange;

    /// Change a row's mutable facts; `None` when the row is gone or
    /// `revision` moved.
    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange>;

    /// Remove a row; `None` when it is gone already or `revision` moved (a
    /// removal never takes a newer incarnation of the same key).
    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange>;

    /// Move a row's status; `None` when the row is gone, `revision` moved or
    /// the status already stood (nothing is emitted then, as today).
    fn set_status(
        &self,
        id: WorkerId,
        status: WorkerStatus,
        revision: u64,
    ) -> Option<MembershipChange>;

    fn get(&self, id: WorkerId) -> Option<WorkerRow>;

    /// The row under a registration key, whatever its incarnation.
    fn by_key(&self, key: &str) -> Option<WorkerRow>;

    /// The epoch of the last write.
    fn epoch(&self) -> u64;

    /// Every row with the epoch it stands at.
    fn snapshot(&self) -> (u64, Vec<WorkerRow>);

    /// The changes after `epoch`, or the instruction to snapshot again.
    fn changes_since(&self, epoch: u64) -> MembershipTail;
}
