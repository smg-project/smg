//! The journal: where the store tells its replicated writes, each inside the
//! write's own critical section, so the order of the calls is the order the
//! store applied the writes (design 5.4, the leader's log).
//!
//! The store knows the trait alone. The replication log
//! ([`crate::service::replication::Log`]) is its one implementation: a store with a
//! journal is a store whose history a standby or a watcher of its rows can
//! follow, and a store without one (a gateway deciding in its own process)
//! pays one branch a write.

use std::sync::Arc;

use crate::{
    engine::view::WorkerId,
    store::workers::{MembershipChange, WorkerRow, WorkerStatus},
};

/// A place in a store's history: the number of a replicated write, and the
/// term the store wrote it under. The default is the place before every
/// write.
///
/// Two stores that agree on a position and its term share the history
/// before it (the replication contract's rejoin rule): a term is led by one
/// replica, and a follower writes a position only as its leader wrote it.
///
/// The record gains fields: code outside this crate starts from the default
/// and assigns what it knows, and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Position {
    /// The write's number, above that of the write before it: the next
    /// number under a leader's term, and under no term past every number
    /// the store's own writes took before, which may skip some. `0` before
    /// any write.
    pub seq: u64,
    /// The term the write at `seq` was made under; `0` before any lease.
    pub term: u64,
}

/// What a store tells at each replicated write: the kinds a watcher of the
/// rows or a standby can ask for. They are membership and status writes,
/// the starts of terms and the generations applied. A watch is told nothing
/// of a generation yet: its needs are what one would ask for.
///
/// Every call but [`Journal::released`] arrives under a lock of the store
/// (the membership lock or the turn of the leases' file),
/// so an implementation returns at once: it takes no lock a reader may
/// hold for long, it never waits on a standby and it calls nothing of the
/// store.
pub trait Journal: Send + Sync {
    /// A membership write: the epoch it makes, the change, the row it is
    /// about as the write leaves it (`None` for a removal) and the row the
    /// write took out (a removal's row, a replacement's old incarnation).
    /// Where a write's writer handed the store the row it leaves, that row
    /// is told and the table then takes it. Where it handed the row the
    /// write takes out, that row is told with the table's status. A write
    /// that hands none tells that row as the table held it. A status is
    /// not told here: it has its own call ([`Journal::status`]). `at` is
    /// the place of the leader's event the write applies, on a store that
    /// follows ([`crate::store::MemoryStore::follow`]), and none for a
    /// write of the store's own: a journal keeps that place for the
    /// applier's write alone. A write whose place was given up before it
    /// came is the store's own where the store took a term since, and is
    /// journaled nowhere otherwise.
    ///
    /// `term` is the term the write is made under, where its writer names
    /// one. The answer is whether the write goes on. A journal says no in
    /// one case alone: its store does not lead `term` in the moment the
    /// event would take its place. The store then moves nothing, neither
    /// a table nor its epoch. A write that names no term always goes on,
    /// and a journal that leads no terms answers yes to every write. The
    /// store steps its epoch behind a yes: `epoch` is the one that step
    /// gives.
    #[must_use = "a write the journal refuses moves nothing in the store"]
    fn member(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool;

    /// A status write: the epoch it makes, the row's id, the status the
    /// row leaves and the one it takes. No row is told. `models` are the
    /// models the row serves at the write, the one fact of its row a
    /// reader of a status reads: a watch that asked for some. They are the
    /// writer's where it handed them, as a holder of worker objects does
    /// on a leader and on a standby, and the table's row's otherwise.
    /// `at`, `term` and the answer are as [`Journal::member`] has
    /// them. A status of the store's own is told ahead of the table's
    /// move, and the table takes the status behind a yes alone. A status
    /// the store follows names no term.
    #[must_use = "a write the journal refuses moves nothing in the store"]
    #[expect(
        clippy::too_many_arguments,
        reason = "a status entry's fields, with the write's place and term"
    )]
    fn status(
        &self,
        at: Option<Position>,
        term: Option<u64>,
        epoch: u64,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Arc<[String]>,
    ) -> bool;

    /// A generation applied, by its number: told under the membership lock
    /// behind the publication of its first world, so its place orders it
    /// among the membership writes, with the place of the leader's event
    /// it applies as [`Journal::member`] has it. The number alone is told:
    /// the door that applied the generation holds the document it runs
    /// from, and whoever sends the event joins the two.
    ///
    /// A generation applied under no term takes no place: a standby joins a
    /// history at a term's start, so the store tells the standing number
    /// again behind each term it takes
    /// ([`crate::store::LeasesStore::set_term`]). That one is told under
    /// the term's rise, not under the membership lock: an apply that
    /// publishes between the read of the standing number and its telling
    /// leaves the older number behind the newer one. The older entry is
    /// inert: it crosses by its number alone, since the door joins a
    /// document only to the generation it still holds, and an event with
    /// no document changes nothing on a standby.
    fn generation(&self, at: Option<Position>, generation: u64);

    /// The store's term rose, told once a rise: the store took `term`, and
    /// its later writes are made under it while it leads.
    fn term_rose(&self, term: u64);

    /// The store follows the start of a leader's term, the event at `at`
    /// ([`crate::store::MemoryStore::follow_term`]): it writes no row, and
    /// the journal keeps the place. The store took nothing by it, and
    /// leads no term.
    fn term_followed(&self, at: Position);

    /// The last write told, with its term.
    fn head(&self) -> Position;

    /// The term the journal followed, `None` while the store leads the
    /// journal's term: from a take that starts the term until the store
    /// follows another history. A take that starts no term changes nothing.
    fn followed(&self) -> Option<u64>;

    /// The store has let go of every lock of a journaled write: the one
    /// call that arrives under no lock of the store. It follows
    /// each such write, and a call that wrote nothing may be followed by
    /// it too. What a journal wakes at its writes it wakes here, so no
    /// writer and no reservation waits behind the wake; a journal that
    /// wakes nobody leaves it empty. A journal that hands its writes on to
    /// another hands this call on too, or the other's wake is never made.
    fn released(&self) {}

    /// The store restores a snapshot that stands at `at`: the journal starts
    /// again there, inside the restore's own locks, and wakes nobody there:
    /// the restore's release follows ([`Journal::released`]). `false` where
    /// the journal refuses the place, and the store then restores nothing.
    fn restart(&self, at: Position) -> bool;
}
