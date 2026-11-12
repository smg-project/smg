//! The store ingress (design 3, 5.1 and appendix A): the contracts through
//! which the controller's store is fed, and through which a gateway in the
//! embedded mode feeds the same store from its own process.
//!
//! The store holds what a decision reads and nothing a request path writes:
//! membership, health, loads, the KV event cursors beside the index,
//! reservations, session pins and leases. Every path that writes it today
//! ends in one of these sinks (the step-3 inventory counted thirteen paths
//! into six writes), so the drivers that move into the controller (discovery,
//! the handshake, the prober, the poll, the subscriptions, the relays) and the
//! gateway's own copies of them in the embedded mode speak one set of traits:
//!
//! | trait | who calls | how often |
//! |---|---|---|
//! | [`WorkersStore`] | discovery and the registration handshake, the control scope's register, patch and deregister calls | per membership change; a rollout of 512 workers is under one change per second |
//! | [`HealthSink`] | the prober (one probe per worker per interval, 128 in flight), the connect signal, every gateway's vetoes and completed dispatches | per probe; two frames per request |
//! | [`LoadSink`] | the subscription task per record, the poll per engine per tick, the scrape per interval, a relay per frame | up to ten records per rank per second busy, one per second idle |
//! | [`KvEventSink`] | the subscription task per batch, a relay per frame | one batch per scheduler step that moved blocks |
//! | [`ReservationStore`] | a commit (a prefill slot, a decode room), a lease's end | two per pair dispatch |
//! | [`SessionsStore`] | a keyed commit, the TTL sweep, a gateway's `Hello` pages | per keyed decision |
//! | [`LeasesStore`] | every decision, dispatch and completion report, the dispatch-deadline sweep, `Hello` pages | two to three frames per request |
//!
//! Rules every implementation keeps:
//!
//! - **Revision guards.** A write that may race a replacement of the same
//!   registration key names the row revision it read; a stale one is refused
//!   (`None`), never applied to the successor. The registry's status
//!   transitions and probe completions work this way today.
//! - **Late and duplicate records.** Load records are in stream order per
//!   rank and replace the rank's row; a poll's row for a rank that pushed
//!   since the poll began is dropped. A KV batch below its rank's cursor is a
//!   duplicate and is skipped; a batch past it is a gap answered by one replay
//!   request; a sequence far below the cursor, or a clear below it, is a
//!   restarted publisher that clears the rank. A release of a lease or a
//!   reservation is idempotent.
//! - **Instants.** The store is in-process; its instants are the host's
//!   `Instant`. The wire carries durations only (a record's age, a lease's
//!   deadline as an offset), converted at the edge by the relay or the
//!   session stream.
//! - **The fencing term.** A lease carries the term of the leader that issued
//!   it. A record whose term is above the store's own is refused (the wire's
//!   `FAILED_PRECONDITION`); one below it is kept, as a lease restored from a
//!   gateway's `Hello` pages after a failover is. Reservations are exclusive
//!   under one fenced leader; credits are rebuilt from `Hello`, never
//!   replicated.
//! - **Wire messages are the host's.** No second load or event message exists
//!   in process: a load record enters as the view's [`LoadRow`] (the record's
//!   routing core in its own names) and a KV batch as the host's own message
//!   behind [`KvEventSink::Batch`], with the header facts the cursor reads
//!   beside it.
//!
//! [`MemoryStore`] is the one-process store behind every trait: the
//! controller's own store once its drivers move in, and the double the
//! contract tests run over. A gateway in the embedded mode implements the
//! same traits over the structures it already holds.
//!
//! [`LoadRow`]: crate::view::LoadRow

pub mod health;
pub mod kv_events;
pub mod leases;
pub mod loads;
pub mod memory;
pub mod reservations;
pub mod sessions;
pub mod workers;

#[cfg(test)]
mod tests;

pub use health::HealthSink;
pub use kv_events::{
    Admission, BatchFacts, BatchOutcome, KvCursor, KvEventSink, RankCursor, ResyncReason,
    SnapshotChunk,
};
pub use leases::{LeaseRecord, LeaseRefused, LeaseState, LeasesStore};
pub use loads::{LoadSample, LoadSink, LoadSource, PollMode};
pub use memory::MemoryStore;
pub use reservations::{ReservationKind, ReservationStore};
pub use sessions::{Pin, SessionsStore};
pub use workers::{
    identity, identity_of, HealthPolicy, MembershipChange, MembershipTail, Overload, Reachability,
    Signals, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
};

/// The verified identity of a gateway: the fingerprint of the subject its
/// session was authenticated as, never a message field. Leases, vetoes and
/// completed reports are keyed by it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GatewayId(pub u64);

impl GatewayId {
    /// The gateway deciding in its own process: the embedded mode's one
    /// principal.
    pub const LOCAL: Self = Self(0);
}
