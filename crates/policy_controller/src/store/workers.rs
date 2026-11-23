//! The membership store: the worker row the handshake produces, the writes
//! that change it and the watch the gateways' pools derive from (design 5.1,
//! the workers row; 5.2, `WatchWorkers`).

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use crate::{
    engine::{
        demands::DemandFacts,
        view::{fingerprint, Role, WireModes, WorkerId},
    },
    store::{encoding::EncodeOffer, pairing::PairingDescriptor, GatewayId},
    tracking::{in_flight::HostCount, lease::Membership},
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

/// The overload thresholds a worker's registration names for itself. A
/// field it names stands over the value of the binding that matches the
/// worker's pool, whether that binding's block is switched on or off; a
/// field left `None` takes the binding's. The store holds no default of
/// its own: the spec's is the only one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overload {
    /// KV usage in permille, the mean over the engine's ranks, at or above
    /// which the worker's rows are vetoed.
    pub kv_usage_permille: Option<u16>,
    /// Queued requests, summed over the engine's ranks, at or above which
    /// the worker's rows are vetoed.
    pub waiting_requests: Option<u32>,
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
    /// What the worker offers a prefill-decode partner, when the handshake
    /// learned any of it. The store derives the pair classes from it, with
    /// the engine kind as the runtime, where a membership write moves a
    /// member or an offer and where a generation moves the pairing mode
    /// ([`pairing`](super::pairing)): a registration hands no class in. The
    /// door drops a blank value and reads a name past its blanks. Behind a
    /// shared pointer: a worker that reports none holds one word for it,
    /// and every copy of a row shares the one its worker reported.
    pub pairing: Option<Arc<PairingDescriptor>>,
    /// The worker's end of the encode handoff, as the handshake learned it;
    /// the default states none. The store derives the encode classes from
    /// it, with the engine kind as the runtime, where a membership write
    /// moves a member or an offer and where a generation moves the pairing
    /// mode (`encoding`). The door drops a blank
    /// connector and reads its name past its blanks.
    pub encode: EncodeOffer,
    /// The worker takes images, as its registration states: an encoder
    /// encodes them and a decode attends to them. A request that needs
    /// vision ([`Demands::vision`](crate::engine::demands::Demands::vision)) takes
    /// no other worker, so a registration that changes the bit starts
    /// another life ([`Self::compare_to_standing`]).
    pub vision: bool,
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
    /// The seconds a removal waits behind a drain, where the worker's
    /// registration states another window than the default. A host that
    /// builds the worker's object from the row gives the object this window.
    pub drain_settle_secs: Option<u32>,
    /// What the worker states for a request's hard demands beside its
    /// vision ([`DemandFacts`]), in its registering host's own numbers. A
    /// host that states none registers a worker that meets no demand that
    /// reads them.
    pub demand_facts: DemandFacts,
}

impl WorkerFacts {
    /// A URL's bytes.
    pub const URL_BYTES: usize = 2_048;
    /// The bytes of a name: a runtime, a zone, a model id.
    pub const NAME_BYTES: usize = 256;
    /// A label's key, in bytes.
    pub const LABEL_KEY_BYTES: usize = 128;
    /// A label's value, in bytes.
    pub const LABEL_VALUE_BYTES: usize = 4_096;
    /// The labels of one row.
    pub const LABELS: usize = 128;

    /// Whether these facts name the models `other` names, in any order and
    /// however often. The order is a fact of its own (a row's events are
    /// indexed under its first model), and a reorder alone is no other
    /// worker.
    pub(crate) fn same_models(&self, other: &Self) -> bool {
        fn named(models: &[String]) -> Vec<&str> {
            let mut ids: Vec<&str> = models.iter().map(String::as_str).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        }
        self.models == other.models || named(&self.models) == named(&other.models)
    }
}

/// Who made a row: the caller whose registration put it there. A row has
/// one maker while its registration stands, through every refresh and every
/// new incarnation of its key. It is what the membership door knows of a
/// row and no fact a registrant states: it moves no revision and is no part
/// of the facts. The door writes it, and a caller that is not a row's maker
/// changes nothing of the row there. Two makers are one where they agree
/// whole.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Registrant {
    /// A control call. Any caller of the control scope changes the row.
    #[default]
    Control,
    /// A provider that finds workers by itself. Its key stands apart from
    /// the row, held once for every copy of it: a copy of the row copies no
    /// string of its maker.
    Provider(Arc<ProviderKey>),
}

impl Registrant {
    /// The bytes of a provider's name: a label key's bound.
    pub const NAME_BYTES: usize = WorkerFacts::LABEL_KEY_BYTES;
    /// The bytes of a provider's key for one worker: a URL's bound.
    pub const KEY_BYTES: usize = WorkerFacts::URL_BYTES;

    /// The maker `provider` is for the worker it knows by `key`, or the part
    /// that breaks its rule. The rules are checked here and nowhere else, so
    /// a leader makes no maker its follower refuses.
    ///
    /// The name is one [`token`] of at most [`Self::NAME_BYTES`] bytes, the
    /// form of a principal's stated name: the step that registers for a
    /// provider acts as a principal of that name. A refusal says it as it
    /// stands.
    /// The key is not empty and is at most [`Self::KEY_BYTES`] bytes. It is
    /// the provider's own text and may hold an address with a credential:
    /// no refusal, log line or metric label carries it. A key past its bound
    /// is refused and never shortened, since two shortened keys can meet.
    pub fn provider(
        provider: impl Into<String>,
        key: impl Into<String>,
    ) -> Result<Self, ProviderKeyError> {
        let (provider, key) = (provider.into(), key.into());
        if provider.is_empty() {
            return Err(ProviderKeyError::Provider(
                "a provider's row names its provider",
            ));
        }
        if provider.len() > Self::NAME_BYTES {
            return Err(ProviderKeyError::Provider(
                "a provider's name is at most 128 bytes",
            ));
        }
        if !token(&provider) {
            return Err(ProviderKeyError::Provider(
                "a provider's name is one token of visible ASCII with no comma",
            ));
        }
        if key.is_empty() {
            return Err(ProviderKeyError::Key(
                "a provider's row names its key for the worker",
            ));
        }
        if key.len() > Self::KEY_BYTES {
            return Err(ProviderKeyError::Key(
                "a provider's key is at most 2048 bytes",
            ));
        }
        Ok(Self::Provider(Arc::new(ProviderKey { provider, key })))
    }
}

/// Whether `name` is one token: not empty, every byte visible ASCII (0x21 to
/// 0x7E) and none a comma. A principal's stated name and a provider's name
/// take this form, so one name reads the same in both; each holds its own
/// bound. The wire lets a hop join a key's entries into one value with a
/// comma between them, so a comma is two names.
#[inline]
pub fn token(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && byte != b',')
}

/// Why a provider's name and key make no maker: the part that breaks its
/// rule, with the rule as a refusal says it. The words hold neither text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProviderKeyError {
    /// The provider's name.
    #[error("{0}")]
    Provider(&'static str),
    /// The provider's key for the worker.
    #[error("{0}")]
    Key(&'static str),
}

/// A provider's name with its own key for one worker, each inside its rule:
/// [`Registrant::provider`] alone makes one.
#[derive(Debug, PartialEq, Eq)]
pub struct ProviderKey {
    provider: String,
    key: String,
}

impl ProviderKey {
    /// The provider's name.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The provider's own key for the worker.
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// One worker as the store holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerRow {
    pub id: WorkerId,
    pub facts: WorkerFacts,
    pub status: WorkerStatus,
    /// Counts the row's stated facts: it moves on every write of them (a
    /// registration's refresh, a patch) and on no status change. The guard
    /// a racing write names. The world's writer carries a member's fields
    /// under the view's lists while it stands, so a write of the facts must
    /// move it.
    pub revision: u64,
    /// When the row was created on the host's clock.
    pub admitted_at: Instant,
    /// Who made the row.
    pub maker: Registrant,
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
/// Every write moves the store's epoch; a write of a row's facts moves the
/// row's revision, a status change does not. A watcher
/// reads a snapshot with its epoch, then the changes since it; past the
/// store's buffer it is told to snapshot again, and it snapshots again
/// where the store took another store's rows ([`Self::history`]). The
/// snapshot carries the
/// facts the gateways' pools derive from (role, wire modes, engine kind,
/// reachability), never a pool list.
///
/// A write is answered once its row is written, told to the journal and
/// counted in the epoch. What a decision reads follows it. The crate's
/// store builds its pools from the membership once a publication window
/// ([`crate::store::PUBLISH_WINDOW`]), however many writes the window
/// took. The build of a registration, its refresh and a patch starts
/// within one window of their answer, where the store's host pays the due
/// instant ([`crate::store::LoadSink::publish_due`]), and they are in force
/// for decisions when it ends. A caller that needs its own write in force
/// at once, or whose host runs no loop, asks for the build
/// ([`crate::store::MemoryStore::publish_now`]). A removal and a replaced
/// incarnation pay the build before they answer, and so does a
/// registration that takes its worker off a model, for the pools it
/// leaves. A status change is published with its answer: no decision made
/// after it is handed the worker.
pub trait WorkersStore: Send + Sync {
    /// A registration or its refresh. A new key is `Registered`; the same key
    /// with another incarnation is `Replaced` (the old id leaves with its
    /// state); the same key and incarnation refresh the facts in place
    /// (`Resized` when the data-parallel width moved, else `Relabelled`).
    /// A new key's row is a control call's ([`Registrant::Control`]).
    fn upsert(&self, facts: WorkerFacts) -> MembershipChange {
        self.upsert_by(facts, Registrant::Control)
    }

    /// [`Self::upsert`] from `maker`: a new key's row is made by it. A row
    /// that stands under the key keeps the maker it has, through a refresh
    /// and through a new incarnation, whoever calls. Who may write a row is
    /// the membership door's test, not the store's.
    ///
    /// The store takes the incarnation as stated. A caller numbers a new
    /// life of a key with a number no life of the key had: the membership
    /// doors do ([`life`](super::membership::life)), and a row they made
    /// never comes back under its id.
    fn upsert_by(&self, facts: WorkerFacts, maker: Registrant) -> MembershipChange;

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
    fn all_rows(&self) -> (u64, Vec<WorkerRow>);

    /// The changes after `epoch`, or the instruction to snapshot again.
    fn changes_since(&self, epoch: u64) -> MembershipTail;

    /// Which history of rows the store holds: it moves when the rows are
    /// replaced from a snapshot
    /// ([`MemoryStore::restore_from`](super::MemoryStore::restore_from) or
    /// [`MemoryStore::watch_from`](super::MemoryStore::watch_from)).
    /// An epoch names a place in one history, and a row
    /// under one id and one revision may state other facts in the next, so
    /// a watcher that keeps its place keeps this beside it and starts again
    /// from a snapshot, trusting nothing it held, when it moves. A store
    /// that never takes another's rows stays at `0`.
    fn history(&self) -> u64 {
        0
    }

    /// Takes the count the store's host keeps for the row `id`
    /// ([`HostCount`]): the store keeps it equal to the count its leases
    /// on file book on that worker. Whether the store keeps it: `false`
    /// for a row that is gone. The holder of a host's worker objects hands
    /// one as it lays a row's object. A store that books no count keeps
    /// none, which is the default.
    fn host_count(&self, _id: WorkerId, _count: Box<dyn HostCount>) -> bool {
        false
    }
}
