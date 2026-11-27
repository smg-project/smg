//! The membership writer: a host's writes of worker rows into the store.

#![cfg_attr(
    not(any(test, feature = "discovery")),
    expect(
        dead_code,
        reason = "A drain's, a removal's and a refresh's rules run at the replica's own door, which the host owns."
    )
)]

use std::{mem, slice, sync::Arc};

use smg_worker::{
    identity::{basic::Worker, builder::rank_url},
    settings::UnhealthyRemovalScope,
};

use super::{kv_feed::asks_events, RowRevision};
use crate::{
    engine::view::WorkerId,
    store::{
        membership::{FactPastBound, NotRegistrant, WorkerFactsChange},
        pairing::PairingDescriptor,
        MembershipChange, MemoryStore, Registrant, WorkerFacts, WorkerPatch, WorkerRow,
        WorkerStatus,
    },
};

/// What a write of the membership writer made of the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MembershipWriteOutcome {
    Registered(WorkerId),
    InPlace(WorkerId),
    Another { old: WorkerId, new: WorkerId },
    Unmoved(WorkerId),
    Draining(WorkerId),
    Removed(WorkerId),
}

impl MembershipWriteOutcome {
    /// The row a registration's write stands on: the new one of a
    /// replacement. A removal's answers name no such row.
    pub(super) fn written_row(&self) -> Option<WorkerId> {
        match *self {
            Self::Registered(row) | Self::InPlace(row) | Self::Unmoved(row) => Some(row),
            Self::Another { new, .. } => Some(new),
            Self::Draining(_) | Self::Removed(_) => None,
        }
    }
}

/// A write the door refused. Nothing was written, with one exception: a
/// registration's rows refused as [`MembershipWriteRefused::NotLeading`] behind
/// their first write keep the rows written before the lead was lost.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MembershipWriteRefused {
    /// Another registrant's row, named by its registrant.
    NotRegistrant(String),
    /// A fact the door does not take, or the row past its size. A fact is
    /// refused past its bound, under a key that is not the call's URL, or,
    /// at the controller's door, when its log does not carry it to a standby
    /// or when it states a host-local wire and names no gateway.
    Past(FactPastBound),
    /// No row stands under the URL.
    Absent,
    /// The host does not lead now, with its reason.
    NotLeading(String),
    /// Another writer moved the row under the call.
    Moved,
    /// A row stands where an add expects none: under a URL it states, or
    /// one it would retire. The rows of the same add, tried again, are not
    /// such rows ([`Precondition::Absent`]).
    Stands,
    /// The row a replacement or a refresh names is draining: its removal
    /// takes the worker.
    Draining,
    /// The row's maker is outside the health removal scope.
    OutOfScope,
    /// A refresh's worker states another fact than the row's labels and
    /// the pairing they derive. A refresh writes those alone.
    NotLabelsAlone,
}

impl MembershipWriteRefused {
    pub(crate) const OUT_OF_SCOPE: &str = "the row's maker is outside the health removal scope";
    pub(crate) const NOT_LABELS_ALONE: &str =
        "a refresh writes a row's labels and the pairing they derive, and states no other fact";
}

/// Who asks to drain or remove a row.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Remover<'a> {
    /// A caller acting as the row's whole registrant.
    Registrant(&'a Registrant),
    /// The host's discovery source, acting under a row's maker if it has
    /// this provider name. The guard names the row under any key.
    Provider(&'a str),
    /// The host's health checks, acting under the maker of a row in scope.
    Health(UnhealthyRemovalScope),
}

/// What a registration expects to stand under the URLs it states. Once that
/// holds, its facts are laid by one rule whatever it expects: an absent URL
/// is registered, and a standing row is left, changed in place or replaced
/// as its facts say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Precondition {
    /// Nothing: an add. A row standing under a URL it states, or one the
    /// registration would retire, refuses it whole
    /// ([`MembershipWriteRefused::Stands`]), whoever made the row.
    ///
    /// One standing is the add's own. A lead lost between two rows leaves
    /// the rows written before it, and the same add tried again lays the
    /// rest: every row that stands under a URL it states is its maker's
    /// and states equal facts, and a row is left to write. The standing
    /// rows answer unmoved and the others are registered. An engine that
    /// stands whole refuses the add, and so does a standing row that
    /// differs or is another maker's.
    Absent,
    /// The row its asker saw, at the revision it saw: a replacement by id,
    /// which states that one row. No row refuses it
    /// ([`MembershipWriteRefused::Absent`]), and so does another row or
    /// revision ([`MembershipWriteRefused::Moved`]). A draining row refuses it
    /// too ([`MembershipWriteRefused::Draining`]): its removal wins.
    AtRevision(RowRevision),
    /// Anything: an upsert.
    Any,
}

/// The object a guarded write replaces, named by the write's verb: the
/// object its asker read and built from.
///
/// An object can change while its row's revision stands: a registration
/// that states another credential alone lays another. So a registration
/// line that keeps credentials refuses the write as moved where another
/// object stands for the row by then. A door holds the write against no
/// object, and reads the verb.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Over<'a> {
    /// A replacement by id. It is its maker's whole statement of what a
    /// registration can state: a threshold it leaves out is cleared.
    Replaced(&'a Arc<dyn Worker>),
    /// An update in place. It moves what it names: a threshold the built
    /// worker does not state stays the row's.
    Updated(&'a Arc<dyn Worker>),
}

impl<'a> Over<'a> {
    /// The object the write replaces.
    pub fn object(self) -> &'a Arc<dyn Worker> {
        match self {
            Self::Replaced(object) | Self::Updated(object) => object,
        }
    }
}

/// One row a registration states.
#[derive(Clone, Copy)]
pub(crate) struct Stated<'a> {
    /// The row's URL: its key.
    pub(crate) url: &'a str,
    /// The facts the registration states, over what stands under the URL.
    pub(crate) state: &'a dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
}

/// The membership writer a host hands its registration and removal steps.
pub trait MembershipWriter: Send + Sync + 'static {
    /// Lay a registration's facts under `url`, over what it expects to
    /// stand there. The callback states facts for `url` during the door's
    /// turn and must leave writes to the door. Facts whose key is not `url`
    /// are refused: the maker was asked of the row under `url`, and the
    /// store writes by the facts' key. The door states the row's event
    /// signal itself, by the generation in force, whatever the facts state.
    ///
    /// This is the write of one row by the row's own rule. It holds no rule
    /// of the engine's other rows and no row leaves with it: a worker's
    /// rows are registered through [`found_rows`](Self::found_rows).
    fn found(
        &self,
        maker: &Registrant,
        expect: Precondition,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused>;

    /// Lays the rows one registration states of one engine, each with the
    /// worker its registration built, over what it expects to stand. A
    /// row's URL is its worker's, and its facts are the ones the worker
    /// states over the standing row ([`facts`]), by the
    /// write's verb.
    /// The engine is the one the rows' own facts state, so no caller names
    /// it another way.
    ///
    /// A registration is held whole: each row to the rule of
    /// [`found`](Self::found), the rows together to the engine's full
    /// width, and the engine's rows it does not state leave with it. The
    /// rows are one a rank, each under its rank's URL, or one row under the
    /// engine's own URL that states no rank. That one row is the engine
    /// whole at any width it states: a worker that serves several engines
    /// over one connection states their count. Every check holds before
    /// the first write, so a refusal of a row or of the group writes
    /// nothing. At a door whose lead can move, the term is asked ahead of
    /// each row's write and each removal: a lead lost between two of them
    /// refuses there ([`MembershipWriteRefused::NotLeading`]), and what was
    /// written before it stands. An upsert lays the rest, and so does the same
    /// add tried again ([`Precondition::Absent`]). A guarded write
    /// ([`Precondition::AtRevision`]) states the one row its asker saw and
    /// changes it where it stands, by that row's own rule.
    ///
    /// A door writes the facts alone, and its host's holder builds each
    /// object from its row. A door over a door passes the call on.
    ///
    /// `over` names the object a guarded write replaces, by the write's
    /// verb ([`Over`]). Every door reads the verb: a replacement by id
    /// states its thresholds whole, and any other write keeps the standing
    /// row's where a built worker states none.
    ///
    /// [`facts`]: super::facts
    fn found_rows(
        &self,
        maker: &Registrant,
        expect: Precondition,
        built: &[Arc<dyn Worker>],
        over: Option<Over<'_>>,
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused>;

    /// Lay the registration of one built worker:
    /// [`found_rows`](Self::found_rows) with one row.
    fn found_built(
        &self,
        maker: &Registrant,
        expect: Precondition,
        built: &Arc<dyn Worker>,
        over: Option<Over<'_>>,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
        let wrote = self.found_rows(maker, expect, slice::from_ref(built), over)?;
        wrote.first().copied().ok_or(MembershipWriteRefused::Moved)
    }

    /// Mark the row the asker saw as draining. Its revision stays the same.
    fn drained(
        &self,
        by: Remover<'_>,
        url: &str,
        seen: RowRevision,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused>;

    /// Remove the row the asker saw under the registered URL.
    fn gone(
        &self,
        by: Remover<'_>,
        url: &str,
        seen: RowRevision,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused>;

    /// Write the labels of the worker the health loop's refresh rebuilt,
    /// and the pairing descriptor they derive, over the row the asker saw
    /// under the worker's URL. Both are the engine's word and nobody's
    /// registration: no maker is asked, a control call's row and a
    /// provider's are served alike, and the row keeps the maker it has.
    ///
    /// The facts `built` states over the standing row ([`facts`]) may
    /// differ from the row's in the labels and the descriptor alone. Any
    /// other difference is refused
    /// ([`MembershipWriteRefused::NotLabelsAlone`]), so a refresh starts no
    /// life. No row, another row or revision, and a draining row refuse it too.
    /// The stated facts' room is tested as a registration's is: the
    /// controller's door, whose log carries no descriptor, refuses a refresh
    /// that states one as it refuses such a registration. Equal labels and an
    /// equal descriptor write nothing and answer
    /// [`MembershipWriteOutcome::Unmoved`].
    ///
    /// A door writes the row alone, as for [`Self::found_built`]. A door
    /// over a door passes the call on.
    ///
    /// `over` is the object the refresh rebuilt from. Equal labels and an
    /// equal descriptor move no revision, so the row's guard cannot tell an
    /// older rebuild from a newer one, and a door that writes the row alone
    /// has nothing to hold the refresh against.
    ///
    /// [`facts`]: super::facts
    fn refreshed(
        &self,
        seen: RowRevision,
        built: &Arc<dyn Worker>,
        over: &Arc<dyn Worker>,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused>;
}

/// What a door owns of a write and of the reads ahead of it, beside the
/// rules every door shares.
pub(crate) trait Turn {
    /// Whether a found row of `facts` asks for its engine's cache events:
    /// the door's answer under the generation in force at its own store
    /// ([`asks_events_under`]).
    fn asks_events(&self, facts: &WorkerFacts) -> bool;

    /// The row that stands under the id `id`, as the door reads a record:
    /// a door with a holder reads it off its worker objects
    /// ([`Objects::record`](super::Objects::record)).
    fn row(&self, id: WorkerId) -> Option<WorkerRow>;

    /// The row that stands under `url`, as [`Self::row`] reads one.
    fn row_under(&self, url: &str) -> Option<WorkerRow>;

    /// Every row whose facts name `engine`, as [`Self::row`] reads one.
    fn rows_of(&self, engine: &str) -> Vec<WorkerRow>;

    /// The turn's write of a registration's facts: a door with a holder
    /// decides it over its worker objects
    /// ([`Objects::upsert`](super::Objects::upsert)). `None` where another
    /// writer moved the row under the turn. A turn under a term hands it
    /// on with the write, and answers `None` too where the log left the
    /// term ahead of the row's event: the routine then asks
    /// [`Self::check_term`].
    fn upsert(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange>;

    /// The turn's write of a patch, as [`Self::upsert`] writes a
    /// registration.
    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange>;

    /// The turn's removal of a row, as [`Self::upsert`] writes a
    /// registration.
    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange>;

    /// The turn's connect signal for the row `id`, read at `revision`, a
    /// row whose checks are off: ready at once, unless it drains or is
    /// ready. A door with a holder writes it on the row's worker object,
    /// and the store is told. A turn under a term hands it on with the
    /// status: the store's journal compares it where the status's event
    /// takes its place, and nothing is written where the log has left it
    /// ([`Objects::connect_row_under`](super::Objects::connect_row_under)).
    /// A registration of equal facts sends it too, where its row waits for
    /// it still ([`awaits_connect`]).
    fn connect(&self, id: WorkerId, revision: u64);

    /// The turn's drain of the row `id`, read at `revision`, as
    /// [`Self::connect`] writes a status
    /// ([`Objects::drain_row_under`](super::Objects::drain_row_under)).
    /// Whether the row moved: `false` where another writer moved it under
    /// the turn, and where the log left the turn's term ahead of the
    /// status's event.
    fn drain(&self, id: WorkerId, revision: u64) -> bool;

    /// Check the term ahead of a write. A registration asks before each
    /// row it writes and each row it removes, and once more behind a
    /// write the store did not take.
    fn check_term(&self) -> Result<(), MembershipWriteRefused> {
        Ok(())
    }

    /// The number of the life a registration makes now, or a refusal where
    /// the turn may make none: under a term its log no longer leads.
    fn life(&self) -> Result<u64, MembershipWriteRefused>;

    /// The door's own rules of the facts it is about to write, behind the
    /// store's measure.
    fn room(&self, facts: &WorkerFacts, maker: &Registrant) -> Result<(), MembershipWriteRefused>;

    /// Hear a row as its write left it, ahead of the connect signal of a
    /// row whose checks are off.
    fn laid(&self, _: &WorkerRow) {}

    /// Hear a row's write at its place in the turn.
    fn told(&self, _: MembershipWriteOutcome) {}
}

/// Whether a found row of `facts` asks for its engine's cache events under
/// the generation in force at `store` ([`asks_events`]): what a door over
/// that store answers its turn ([`Turn::asks_events`]).
pub(crate) fn asks_events_under(facts: &WorkerFacts, store: &MemoryStore) -> bool {
    store.deciding(|world| asks_events(facts, world.spec().engine()))
}

fn made_by(row: &WorkerRow, maker: &Registrant) -> Result<(), MembershipWriteRefused> {
    row.made_by(maker)
        .map_err(|not| MembershipWriteRefused::NotRegistrant(not.to_string()))
}

/// The refusal of a write the turn's store did not take, asked behind that
/// write. Where the turn's term has ended by this ask, it is refused as a
/// turn that does not lead, whatever refused the write: its row need not
/// have moved. Otherwise it is refused as a row another writer moved: the
/// table did not hold what the turn read.
fn unwritten(turn: &impl Turn) -> MembershipWriteRefused {
    turn.check_term()
        .err()
        .unwrap_or(MembershipWriteRefused::Moved)
}

/// The row an asker saw, where it still stands under the URL at the
/// revision the asker saw.
fn seen_row(
    turn: &impl Turn,
    url: &str,
    seen: RowRevision,
) -> Result<WorkerRow, MembershipWriteRefused> {
    let row = turn.row_under(url).ok_or(MembershipWriteRefused::Absent)?;
    if (row.id, row.revision) != (seen.row, seen.revision) {
        return Err(MembershipWriteRefused::Moved);
    }
    Ok(row)
}

const RANKS: MembershipWriteRefused = MembershipWriteRefused::Past(FactPastBound {
    fact: "dp_rank",
    why: "a registration states every rank of one engine under its own URL",
});

/// How a door reads the facts a built worker states over the row that
/// stands under its URL, by the write's verb. The controller's door, whose
/// log carries no pairing descriptor, reads the facts alone, and the row
/// keeps the descriptor that stands.
pub(crate) type Reading = fn(&dyn Worker, Option<&WorkerFacts>, Option<Over<'_>>) -> WorkerFacts;

/// The rows `built` states, handed to `write`: each worker's URL, and its
/// facts over the row that stands there, as the door reads them (`read`).
pub(crate) fn stating<T>(
    built: &[Arc<dyn Worker>],
    over: Option<Over<'_>>,
    read: Reading,
    write: impl FnOnce(&[Stated<'_>]) -> T,
) -> T {
    let states: Vec<_> = built
        .iter()
        .map(|worker| move |standing: Option<&WorkerFacts>| read(worker.as_ref(), standing, over))
        .collect();
    let rows: Vec<_> = built
        .iter()
        .zip(&states)
        .map(|(worker, state)| Stated {
            url: worker.url(),
            state,
        })
        .collect();
    write(&rows)
}

/// What a stated row leaves to write: its facts, and what laying them makes
/// of the row that stands under its URL.
pub(crate) type Ready = (WorkerFacts, Option<(WorkerId, WorkerFactsChange)>);

/// Whether a registration states its engine as one row: the row stands
/// under the engine's own URL and states no rank. Such a row is the engine
/// whole at any width it states: a worker that serves several engines over
/// one connection states their count and no rank. Every other registration
/// states one row a rank, of an engine two or more wide.
fn states_the_engine_whole(ready: &[Ready]) -> bool {
    matches!(ready, [(only, _)] if only.key == only.engine && only.dp_rank == 0)
}

/// The row a write expects to stand under `url`, read once for the row's
/// rule.
fn standing_under(
    turn: &impl Turn,
    expect: Precondition,
    url: &str,
) -> Result<Option<WorkerRow>, MembershipWriteRefused> {
    Ok(match expect {
        Precondition::AtRevision(seen) => Some(seen_row(turn, url, seen)?),
        Precondition::Absent | Precondition::Any => turn.row_under(url),
    })
}

/// Check one stated row by its own rule, over the row its write expects to
/// stand under its URL ([`standing_under`]). Nothing is written. The row's
/// event signal is stated here, over the facts its callback states
/// ([`Turn::asks_events`]).
fn ready_row(
    turn: &impl Turn,
    maker: &Registrant,
    expect: Precondition,
    Stated { url, state }: &Stated<'_>,
    standing: Option<WorkerRow>,
) -> Result<Ready, MembershipWriteRefused> {
    if let Some(row) = &standing {
        // An add of several rows asks no maker here. What it meets is
        // judged over its rows together, and its refusal is the same
        // whoever made the row ([`same_add`]).
        if expect != Precondition::Absent {
            made_by(row, maker)?;
        }
        // A draining row is leaving. Laid over, it would take the
        // URL's next life and its removal would leave the worker: the
        // removal wins over a replacement.
        if matches!(expect, Precondition::AtRevision(_)) && row.status == WorkerStatus::Draining {
            return Err(MembershipWriteRefused::Draining);
        }
    }
    let mut facts = state(standing.as_ref().map(|row| &row.facts));
    if facts.key != *url {
        return Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "key",
            why: "the facts name the URL of the call",
        }));
    }
    // The host states the row's stream in the write that makes the row: the
    // follow is told of a worker once for each object laid, and reads the
    // row then.
    facts.signals.events = turn.asks_events(&facts);
    if let Some(row) = &standing {
        facts.incarnation = row.facts.incarnation;
    }
    facts.bounded(maker).map_err(MembershipWriteRefused::Past)?;
    turn.room(&facts, maker)?;
    let laid = standing.map(|row| (row.id, row.compare_new_facts(&facts)));
    Ok((facts, laid))
}

/// What an add meets, over the rows it states. An add writes no row that
/// stands, and one standing is its own: a lead lost between two rows leaves
/// the rows written before it, and the same add tried again lays the rest.
/// It is the same add where every standing row is its maker's and would
/// answer unmoved, and a row is left to write. An engine that stands whole,
/// another maker's row and a row whose facts differ refuse it, the same
/// whoever made the row.
///
/// The rule reads what stands and nothing of a first attempt, so it holds
/// at the next leader's door too.
fn same_add(
    turn: &impl Turn,
    maker: &Registrant,
    ready: &[Ready],
) -> Result<(), MembershipWriteRefused> {
    let mut writes = false;
    for (_, laid) in ready {
        let Some((id, laid)) = laid else {
            writes = true;
            continue;
        };
        let own = turn.row(*id).is_some_and(|row| row.made_by(maker).is_ok());
        if !own || *laid != WorkerFactsChange::Unmoved {
            return Err(MembershipWriteRefused::Stands);
        }
    }
    if writes {
        Ok(())
    } else {
        Err(MembershipWriteRefused::Stands)
    }
}

/// Lay one row by its own rule alone: the write of [`MembershipWriter::found`].
/// No rule of the engine's other rows is held and no row leaves. One row
/// has no rest to lay, so an add of it is refused by any row that stands,
/// before a fact is stated and whoever made the row.
pub(crate) fn lay_one(
    turn: &impl Turn,
    maker: &Registrant,
    expect: Precondition,
    url: &str,
    state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    let standing = standing_under(turn, expect, url)?;
    if expect == Precondition::Absent && standing.is_some() {
        return Err(MembershipWriteRefused::Stands);
    }
    let (facts, laid) = ready_row(turn, maker, expect, &Stated { url, state }, standing)?;
    turn.check_term()?;
    write_row(turn, maker, facts, laid)
}

/// Check each row and the group, over what the registration expects to
/// stand, before writing either the rows that stay or the rows that leave.
/// One turn covers the whole registration.
pub(crate) fn lay(
    turn: &impl Turn,
    maker: &Registrant,
    expect: Precondition,
    rows: &[Stated<'_>],
) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
    if rows.is_empty() {
        return Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "rows",
            why: "a registration states at least one row",
        }));
    }
    // A guarded write states the one row its asker saw and changes it where
    // it stands, by that row's own rule: the group's rule is a
    // registration's.
    if matches!(expect, Precondition::AtRevision(_)) {
        let [row] = rows else {
            return Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "rows",
                why: "a replacement states the one row its asker saw",
            }));
        };
        return lay_one(turn, maker, expect, row.url, row.state).map(|wrote| vec![wrote]);
    }
    let mut standing = Vec::with_capacity(rows.len());
    for row in rows {
        standing.push(standing_under(turn, expect, row.url)?);
    }
    // An add with no row left to write is refused before a fact is stated,
    // whoever made the rows that stand.
    if expect == Precondition::Absent && standing.iter().all(Option::is_some) {
        return Err(MembershipWriteRefused::Stands);
    }
    let mut ready = Vec::with_capacity(rows.len());
    for (row, standing) in rows.iter().zip(standing) {
        ready.push(ready_row(turn, maker, expect, row, standing)?);
    }
    lay_ready(turn, maker, expect == Precondition::Absent, ready)
}

/// The group's rule over the rows one registration states, then the rows
/// that leave and the writes. A door that states its rows by a rule of its
/// own enters here, behind that rule: `adds` says the registration is an
/// add, which meets the rows that stand by the add's rule ([`same_add`])
/// and retires none. Such a door hands an add's standing row in with what
/// laying it would make, and asks its maker of no row.
pub(crate) fn lay_ready(
    turn: &impl Turn,
    maker: &Registrant,
    adds: bool,
    ready: Vec<Ready>,
) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
    let Some((first, _)) = ready.first() else {
        return Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "rows",
            why: "a registration states at least one row",
        }));
    };
    if adds {
        same_add(turn, maker, &ready)?;
    }
    // The engine is the one the first row states. The others are held to
    // it below, and the door's rows are asked by it.
    let engine = first.engine.as_str();
    if !states_the_engine_whole(&ready) {
        let width = usize::from(first.dp_size);
        if width < 2 || width != ready.len() {
            return Err(RANKS);
        }
        let mut seen = vec![false; width];
        for (facts, _) in &ready {
            let rank = usize::from(facts.dp_rank);
            let Some(seen) = seen.get_mut(rank) else {
                return Err(RANKS);
            };
            if facts.engine != engine
                || usize::from(facts.dp_size) != width
                || facts.key != rank_url(engine, rank)
                || mem::replace(seen, true)
            {
                return Err(RANKS);
            }
        }
    }

    // Every standing row of the engine that the registration does not
    // state leaves with it: the ranks a narrower worker no longer has, the
    // plain row of one that comes back ranked, the ranks of one that comes
    // back plain, and a rank left above ranks that were removed. The rows
    // are the engine's by their own fact, whatever their keys read as.
    let mut left: Vec<_> = turn
        .rows_of(engine)
        .into_iter()
        .filter(|row| ready.iter().all(|(facts, _)| facts.key != row.facts.key))
        .collect();
    left.sort_by_key(|row| row.facts.dp_rank);
    for row in &left {
        // An add retires no row: the engine stands in another form.
        if adds {
            return Err(MembershipWriteRefused::Stands);
        }
        made_by(row, maker)?;
    }

    // The term is asked ahead of each row's write and of each removal,
    // whatever the write makes of its row: a lead lost between two of them
    // refuses at the next.
    let mut answer = Vec::with_capacity(ready.len() + left.len());
    for (facts, laid) in ready {
        turn.check_term()?;
        answer.push(write_row(turn, maker, facts, laid)?);
    }
    for row in left {
        turn.check_term()?;
        let wrote = changed(
            turn,
            turn.remove(row.id, row.revision),
            MembershipWriteOutcome::Removed,
        )?;
        turn.told(wrote);
        answer.push(wrote);
    }
    Ok(answer)
}

/// Whether `row` waits for its connect signal still: its checks are off,
/// so no verdict moves it, and it stands neither ready nor draining, the
/// two statuses that signal leaves. A registration whose signal was not
/// written leaves such a row. The one test of it: a registration of equal
/// facts asks it, at a control call's arm and at [`write_row`]'s, and
/// sends the signal again.
pub(crate) fn awaits_connect(row: &WorkerRow) -> bool {
    !row.facts.health.probes && !matches!(row.status, WorkerStatus::Ready | WorkerStatus::Draining)
}

fn write_row(
    turn: &impl Turn,
    maker: &Registrant,
    mut facts: WorkerFacts,
    laid: Option<(WorkerId, WorkerFactsChange)>,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    match laid {
        None | Some((_, WorkerFactsChange::Another)) => facts.incarnation = turn.life()?,
        Some((id, WorkerFactsChange::Unmoved)) => {
            // Equal facts write no row. A row that waits for its connect
            // signal still is sent it again, with this turn's word. The
            // facts are the row's own: a row that is checked waits for
            // none, and is not read again.
            let waits = (!facts.health.probes).then(|| turn.row(id)).flatten();
            if let Some(row) = waits.filter(awaits_connect) {
                turn.connect(row.id, row.revision);
            }
            let wrote = MembershipWriteOutcome::Unmoved(id);
            turn.told(wrote);
            return Ok(wrote);
        }
        Some((_, WorkerFactsChange::InPlace)) => {}
    }
    let probes = facts.health.probes;
    let change = turn.upsert(facts, maker.clone());
    let change = change.ok_or_else(|| unwritten(turn))?;
    let wrote = match change {
        MembershipChange::Registered { id } => MembershipWriteOutcome::Registered(id),
        MembershipChange::Relabelled { id } | MembershipChange::Resized { id } => {
            MembershipWriteOutcome::InPlace(id)
        }
        MembershipChange::Replaced { old, new } => MembershipWriteOutcome::Another { old, new },
        MembershipChange::Removed { .. } | MembershipChange::StatusChanged { .. } => {
            return Err(MembershipWriteRefused::Moved);
        }
    };
    // The row as its write left it: a connect signal moves it next.
    if let Some(row) = turn.row(change.id()) {
        turn.laid(&row);
        if !probes {
            turn.connect(row.id, row.revision);
        }
    }
    turn.told(wrote);
    Ok(wrote)
}

fn changed(
    turn: &impl Turn,
    change: Option<MembershipChange>,
    wrote: fn(WorkerId) -> MembershipWriteOutcome,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    change
        .map(|change| wrote(change.id()))
        .ok_or_else(|| unwritten(turn))
}

fn removal_row(
    turn: &impl Turn,
    by: Remover<'_>,
    url: &str,
    seen: RowRevision,
) -> Result<WorkerRow, MembershipWriteRefused> {
    let row = seen_row(turn, url, seen)?;
    match by {
        Remover::Registrant(maker) => made_by(&row, maker)?,
        Remover::Provider(name) => match &row.maker {
            Registrant::Provider(provider) if provider.provider() == name => {}
            Registrant::Provider(provider) => {
                return Err(MembershipWriteRefused::NotRegistrant(
                    NotRegistrant::Provider {
                        provider: provider.provider(),
                    }
                    .to_string(),
                ));
            }
            Registrant::Control => {
                return Err(MembershipWriteRefused::NotRegistrant(
                    NotRegistrant::Control.to_string(),
                ))
            }
        },
        Remover::Health(scope) => {
            let allowed = match scope {
                UnhealthyRemovalScope::Disabled => false,
                UnhealthyRemovalScope::DiscoveryOwned => {
                    matches!(row.maker, Registrant::Provider(_))
                }
                UnhealthyRemovalScope::AllLocal => true,
            };
            if !allowed {
                return Err(MembershipWriteRefused::OutOfScope);
            }
        }
    }
    Ok(row)
}

/// Mark the guarded row as draining while the door holds its turn.
pub(crate) fn drain(
    turn: &impl Turn,
    by: Remover<'_>,
    url: &str,
    seen: RowRevision,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    let row = removal_row(turn, by, url, seen)?;
    if row.status == WorkerStatus::Draining {
        return Ok(MembershipWriteOutcome::Unmoved(row.id));
    }
    if !turn.drain(row.id, row.revision) {
        return Err(unwritten(turn));
    }
    Ok(MembershipWriteOutcome::Draining(row.id))
}

/// Remove the guarded row while the door holds its turn.
pub(crate) fn take(
    turn: &impl Turn,
    by: Remover<'_>,
    url: &str,
    seen: RowRevision,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    let row = removal_row(turn, by, url, seen)?;
    changed(
        turn,
        turn.remove(row.id, row.revision),
        MembershipWriteOutcome::Removed,
    )
}

/// Write what a refresh's rebuilt worker states over the guarded row while
/// the door holds its turn: its labels, and the pairing descriptor they
/// derive. The row's other facts are the test of the refresh's scope and
/// are not written: the door states a row's event signal at its
/// registration, and a refresh leaves it as it stands. The worker is read
/// as the door reads one (`read`), under no verb: a door whose log carries
/// no descriptor keeps the standing one, so its refresh moves labels alone.
pub(crate) fn refresh(
    turn: &impl Turn,
    seen: RowRevision,
    built: &dyn Worker,
    read: Reading,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    refresh_stated(turn, seen, built.url(), &|standing| {
        read(built, Some(standing), None)
    })
}

/// [`refresh`] over the facts `state` makes of the guarded row.
///
/// Labels are no part of a row's offer, and a patch writes them under the
/// row's guard. A descriptor is part of it. Where it moved, the row takes
/// the write a registration takes in place, and the pair classes are
/// derived again from the membership that write leaves. That write has no
/// guard of its own: the door's turn holds the row as the guard read it.
fn refresh_stated(
    turn: &impl Turn,
    seen: RowRevision,
    url: &str,
    state: &dyn Fn(&WorkerFacts) -> WorkerFacts,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    let row = seen_row(turn, url, seen)?;
    // A draining row is leaving: its removal wins over a refresh, as over
    // a replacement.
    if row.status == WorkerStatus::Draining {
        return Err(MembershipWriteRefused::Draining);
    }
    let mut stated = state(&row.facts);
    let labels = mem::replace(&mut stated.labels, row.facts.labels.clone());
    let mut pairing = mem::replace(&mut stated.pairing, row.facts.pairing.clone());
    if stated != row.facts {
        return Err(MembershipWriteRefused::NotLabelsAlone);
    }
    // The row holds its descriptor as the pairing rule reads it. A stated
    // one that differs is read the same way before it counts as moved: its
    // blanks are dropped. One that reports nothing is none on either side,
    // as the rule reads a row that has no descriptor.
    let held = row.facts.pairing.as_ref();
    if let Some(descriptor) = pairing.as_mut().filter(|said| held != Some(&**said)) {
        Arc::make_mut(descriptor).normalize();
    }
    let reports = |said: &Arc<PairingDescriptor>| **said != PairingDescriptor::default();
    pairing = pairing.filter(reports);
    let pairing_moved = pairing.as_ref() != held.filter(|held| reports(held));
    if labels == row.facts.labels && !pairing_moved {
        return Ok(MembershipWriteOutcome::Unmoved(row.id));
    }
    stated.labels = labels;
    stated.pairing = pairing;
    stated
        .bounded(&row.maker)
        .map_err(MembershipWriteRefused::Past)?;
    turn.room(&stated, &row.maker)?;
    // The write, in place or by the patch, is a write of the turn: the term
    // is asked ahead of it.
    turn.check_term()?;
    if pairing_moved {
        return match turn.upsert(stated, row.maker) {
            Some(MembershipChange::Relabelled { id } | MembershipChange::Resized { id }) => {
                Ok(MembershipWriteOutcome::InPlace(id))
            }
            Some(
                MembershipChange::Registered { .. }
                | MembershipChange::Replaced { .. }
                | MembershipChange::Removed { .. }
                | MembershipChange::StatusChanged { .. },
            ) => Err(MembershipWriteRefused::Moved),
            None => Err(unwritten(turn)),
        };
    }
    let patch = WorkerPatch {
        labels: Some(stated.labels),
        ..WorkerPatch::default()
    };
    changed(
        turn,
        turn.patch(row.id, patch, row.revision),
        MembershipWriteOutcome::InPlace,
    )
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/membership_writer/tests.rs"]
pub(super) mod tests;
