//! The membership writer: a host's writes of worker rows into the store.

#![cfg_attr(
    not(test),
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
        HealthSink, MembershipChange, MemoryStore, Registrant, WorkerFacts, WorkerPatch, WorkerRow,
        WorkerStatus, WorkersStore,
    },
};

/// What a write of the membership writer made of the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MembershipWriteOutcome {
    Registered(WorkerId),
    InPlace(WorkerId),
    Another { old: WorkerId, new: WorkerId },
    Unmoved(WorkerId),
    Draining(WorkerId),
    Removed(WorkerId),
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
    fn store(&self) -> &MemoryStore;

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
    /// writer moved the row under the turn.
    fn upsert(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange>;

    /// The turn's write of a patch, as [`Self::upsert`] writes a
    /// registration.
    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange>;

    /// The turn's removal of a row, as [`Self::upsert`] writes a
    /// registration.
    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange>;

    /// Check the term ahead of a write. A registration asks before each
    /// row it writes and each row it removes.
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

fn made_by(row: &WorkerRow, maker: &Registrant) -> Result<(), MembershipWriteRefused> {
    row.made_by(maker)
        .map_err(|not| MembershipWriteRefused::NotRegistrant(not.to_string()))
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
/// ([`asks_events`]).
fn ready_row(
    turn: &impl Turn,
    maker: &Registrant,
    expect: Precondition,
    Stated { url, state }: &Stated<'_>,
    standing: Option<WorkerRow>,
) -> Result<Ready, MembershipWriteRefused> {
    let store = turn.store();
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
    facts.signals.events = store.deciding(|world| asks_events(&facts, world.spec().engine()));
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
            turn.remove(row.id, row.revision),
            MembershipWriteOutcome::Removed,
        )?;
        turn.told(wrote);
        answer.push(wrote);
    }
    Ok(answer)
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
            let wrote = MembershipWriteOutcome::Unmoved(id);
            turn.told(wrote);
            return Ok(wrote);
        }
        Some((_, WorkerFactsChange::InPlace)) => {}
    }
    let store = turn.store();
    let probes = facts.health.probes;
    let change = turn.upsert(facts, maker.clone());
    let change = change.ok_or(MembershipWriteRefused::Moved)?;
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
            store.connected(row.id, row.revision, &|| true);
        }
    }
    turn.told(wrote);
    Ok(wrote)
}

fn changed(
    change: Option<MembershipChange>,
    wrote: fn(WorkerId) -> MembershipWriteOutcome,
) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
    change
        .map(|change| wrote(change.id()))
        .ok_or(MembershipWriteRefused::Moved)
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
    let store = turn.store();
    changed(
        store.set_status(row.id, WorkerStatus::Draining, row.revision),
        MembershipWriteOutcome::Draining,
    )
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
/// the write a registration takes in place, and the store derives the
/// row's pair classes again there. That write has no guard of its own: the
/// door's turn holds the row as the guard read it.
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
            )
            | None => Err(MembershipWriteRefused::Moved),
        };
    }
    let patch = WorkerPatch {
        labels: Some(stated.labels),
        ..WorkerPatch::default()
    };
    changed(
        turn.patch(row.id, patch, row.revision),
        MembershipWriteOutcome::InPlace,
    )
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::Arc;

    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        model_card::ModelCard,
        settings::UnhealthyRemovalScope,
        worker::{HealthCheckConfig, WorkerType},
    };

    use super::{
        super::found, changed, drain, lay, lay_one, refresh, refresh_stated, stating, take,
        MembershipWriteOutcome, MembershipWriteRefused, MembershipWriter, Over, Precondition,
        Remover, RowRevision, Stated, Turn,
    };
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{
            demands::DemandFacts,
            view::{Role, WireModes, WorkerId},
        },
        store::{
            index::indexed_under,
            membership::{self, FactPastBound},
            pairing::PairingDescriptor,
            MembershipChange, MembershipTail, MemoryStore, Registrant, Signals, WorkerFacts,
            WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
        },
    };

    const URL: &str = "http://worker:8000";

    /// The reads and the writes of a turn over a store alone, which every
    /// turn of these cases is: the store's own table, where no holder
    /// states or decides a record.
    macro_rules! over_the_table {
        () => {
            fn upsert(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange> {
                Some(self.store().upsert_by(facts, maker))
            }

            fn patch(
                &self,
                id: WorkerId,
                patch: WorkerPatch,
                revision: u64,
            ) -> Option<MembershipChange> {
                self.store().patch(id, patch, revision)
            }

            fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
                self.store().remove(id, revision)
            }

            fn row(&self, id: WorkerId) -> Option<WorkerRow> {
                self.store().get(id)
            }

            fn row_under(&self, url: &str) -> Option<WorkerRow> {
                self.store().by_key(url)
            }

            fn rows_of(&self, engine: &str) -> Vec<WorkerRow> {
                let (_, rows) = self.store().all_rows();
                rows.into_iter()
                    .filter(|row| row.facts.engine == engine)
                    .collect()
            }
        };
    }

    /// A door over a store alone, with no election, no log and no holder:
    /// each call runs the door's rules over the store's own writes. The
    /// cases of the rules write through it, and so does a case elsewhere
    /// that needs a row written as a door writes one.
    pub(crate) struct StoreDoor {
        store: Arc<MemoryStore>,
        base: u64,
    }

    impl StoreDoor {
        pub(crate) fn new(store: Arc<MemoryStore>) -> Self {
            Self {
                store,
                base: membership::base(),
            }
        }
    }

    impl Turn for StoreDoor {
        fn store(&self) -> &MemoryStore {
            &self.store
        }

        over_the_table!();

        fn life(&self) -> Result<u64, MembershipWriteRefused> {
            Ok(membership::life(0, self.base, self.store.epoch()))
        }

        fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
            Ok(())
        }
    }

    impl MembershipWriter for StoreDoor {
        fn found(
            &self,
            maker: &Registrant,
            expect: Precondition,
            url: &str,
            state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
        ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
            lay_one(self, maker, expect, url, state)
        }

        fn found_rows(
            &self,
            maker: &Registrant,
            expect: Precondition,
            built: &[Arc<dyn Worker>],
            over: Option<Over<'_>>,
        ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
            stating(built, over, found::stated, |rows| {
                lay(self, maker, expect, rows)
            })
        }

        fn drained(
            &self,
            by: Remover<'_>,
            url: &str,
            seen: RowRevision,
        ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
            drain(self, by, url, seen)
        }

        fn gone(
            &self,
            by: Remover<'_>,
            url: &str,
            seen: RowRevision,
        ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
            take(self, by, url, seen)
        }

        fn refreshed(
            &self,
            seen: RowRevision,
            built: &Arc<dyn Worker>,
            _: &Arc<dyn Worker>,
        ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
            refresh(self, seen, built.as_ref(), found::stated)
        }
    }

    fn door() -> (Arc<MemoryStore>, StoreDoor) {
        let store = Arc::new(MemoryStore::default());
        let door = StoreDoor::new(Arc::clone(&store));
        (store, door)
    }

    fn stated() -> WorkerFacts {
        WorkerFacts {
            key: URL.to_owned(),
            engine: URL.to_owned(),
            incarnation: 17,
            wire_modes: WireModes::HTTP,
            dp_size: 1,
            ..WorkerFacts::default()
        }
    }

    /// A door as the cases reach it: the trait, and the registration's
    /// write over rows of stated facts, where a case states facts no built
    /// worker would.
    pub(crate) trait Rows: MembershipWriter {
        fn rows(
            &self,
            maker: &Registrant,
            expect: Precondition,
            rows: &[Stated<'_>],
        ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused>;
    }

    impl Rows for StoreDoor {
        fn rows(
            &self,
            maker: &Registrant,
            expect: Precondition,
            rows: &[Stated<'_>],
        ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
            lay(self, maker, expect, rows)
        }
    }

    pub(crate) trait FindOne: Rows {
        /// One row, upserted.
        fn found_one(
            &self,
            maker: &Registrant,
            url: &str,
            state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
        ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
            self.found_one_over(maker, Precondition::Any, url, state)
        }

        /// One row, laid over what its registration expects.
        fn found_one_over(
            &self,
            maker: &Registrant,
            expect: Precondition,
            url: &str,
            state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
        ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
            let answer = self.rows(maker, expect, &[Stated { url, state }])?;
            assert_eq!(answer.len(), 1);
            Ok(answer[0])
        }
    }

    impl<T: Rows + ?Sized> FindOne for T {}

    fn standing(row: &WorkerRow) -> RowRevision {
        RowRevision {
            row: row.id,
            revision: row.revision,
        }
    }

    const ABSENT: RowRevision = RowRevision {
        row: WorkerId(0),
        revision: 0,
    };

    fn local_door() -> (Arc<MemoryStore>, Box<dyn Rows>) {
        let (store, door) = door();
        (store, Box::new(door))
    }

    fn ranked(width: u16) -> Vec<WorkerFacts> {
        (0..width)
            .map(|rank| WorkerFacts {
                key: format!("{URL}@{rank}"),
                dp_rank: rank,
                dp_size: width,
                ..stated()
            })
            .collect()
    }

    fn find_rows(
        door: &dyn Rows,
        maker: &Registrant,
        facts: &[WorkerFacts],
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
        find_rows_over(door, maker, Precondition::Any, facts)
    }

    fn find_rows_over(
        door: &dyn Rows,
        maker: &Registrant,
        expect: Precondition,
        facts: &[WorkerFacts],
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
        let states: Vec<_> = facts
            .iter()
            .map(|facts| move |_: Option<&WorkerFacts>| facts.clone())
            .collect();
        let rows: Vec<_> = facts
            .iter()
            .zip(&states)
            .map(|(facts, state)| Stated {
                url: &facts.key,
                state,
            })
            .collect();
        door.rows(maker, expect, &rows)
    }

    type Open = fn() -> (Arc<MemoryStore>, Box<dyn Rows>);

    #[test]
    fn a_width_of_four_makes_four_rows_in_one_write() {
        check_four_rows(local_door).expect("four ranks registered");
    }

    fn check_four_rows(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let mut facts = ranked(4);
        facts.rotate_left(2);
        for row in &mut facts[..2] {
            row.health.probes = false;
        }
        let answer = find_rows(&*door, &Registrant::Control, &facts).map_err(|_| "registered")?;
        assert_eq!(answer.len(), 4);
        let mut lives = Vec::new();
        for (index, stated) in facts.iter().enumerate() {
            let row = store.by_key(&stated.key).ok_or("the registered rank")?;
            assert_eq!(answer[index], MembershipWriteOutcome::Registered(row.id));
            assert_eq!(row.facts.key, stated.key);
            assert_eq!(row.facts.engine, URL);
            assert_eq!((row.facts.dp_rank, row.facts.dp_size), (stated.dp_rank, 4));
            let status = if index < 2 {
                WorkerStatus::Ready
            } else {
                WorkerStatus::Pending
            };
            assert_eq!(row.status, status);
            lives.push(row.facts.incarnation);
        }
        let steps: Vec<_> = lives
            .windows(2)
            .map(|pair| pair[1].checked_sub(pair[0]))
            .collect();
        assert_eq!(steps, [Some(2), Some(2), Some(1)], "{lives:?}");
        Ok(())
    }

    #[test]
    fn a_registration_refused_at_one_rank_writes_no_rank() {
        check_bad_label(local_door);
    }

    fn check_bad_label(open: Open) {
        let (store, door) = open();
        let mut ranks = ranked(4);
        ranks[2].labels.insert("tier".to_owned(), "x".repeat(4097));
        let epoch = store.epoch();
        assert_eq!(
            find_rows(&*door, &Registrant::Control, &ranks),
            Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "labels",
                why: "a label's value is at most 4096 bytes",
            }))
        );
        assert_eq!(store.epoch(), epoch);
        assert!(store.all_rows().1.is_empty());
    }

    #[test]
    fn a_worker_that_comes_back_narrower_loses_the_ranks_it_no_longer_states() {
        check_narrower(local_door).expect("four ranks, then two");
    }

    fn check_narrower(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Registrant::Control;
        let four = ranked(4);
        find_rows(&*door, &maker, &four).map_err(|_| "registered four ranks")?;
        let old: Vec<_> = four
            .iter()
            .map(|facts| store.by_key(&facts.key).ok_or("the old rank"))
            .collect::<Result<_, _>>()?;
        let alias = WorkerFacts {
            key: format!("{URL}@03"),
            engine: format!("{URL}@03"),
            ..stated()
        };
        store.upsert(alias.clone());
        let epoch = store.epoch();
        let two = ranked(2);
        assert_eq!(
            find_rows(&*door, &maker, &two),
            Ok(vec![
                MembershipWriteOutcome::InPlace(old[0].id),
                MembershipWriteOutcome::InPlace(old[1].id),
                MembershipWriteOutcome::Removed(old[2].id),
                MembershipWriteOutcome::Removed(old[3].id),
            ])
        );
        let MembershipTail::Changes(changes) = store.changes_since(epoch) else {
            return Err("four changes after this epoch");
        };
        assert_eq!(
            changes
                .into_iter()
                .map(|(_, change)| change)
                .collect::<Vec<_>>(),
            [
                MembershipChange::Resized { id: old[0].id },
                MembershipChange::Resized { id: old[1].id },
                MembershipChange::Removed { id: old[2].id },
                MembershipChange::Removed { id: old[3].id },
            ]
        );
        for row in &old[..2] {
            let resized = store.get(row.id).ok_or("the resized rank")?;
            assert_eq!(resized.facts.dp_size, 2);
            assert_eq!(resized.facts.incarnation, row.facts.incarnation);
        }
        assert!(four[2..]
            .iter()
            .all(|facts| store.by_key(&facts.key).is_none()));
        assert_eq!(store.by_key(&alias.key).map(|row| row.facts), Some(alias));
        Ok(())
    }

    #[test]
    fn ranks_left_above_removed_ranks_are_the_engines_still() {
        check_partial_removal(local_door).expect("two ranks of four removed");
    }

    /// Ranks zero and one of four are removed one row at a time. The two
    /// left refuse an add, and leave with the next registration.
    fn check_partial_removal(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Registrant::Control;
        let four = ranked(4);
        find_rows(&*door, &maker, &four).map_err(|_| "registered four ranks")?;
        let old: Vec<_> = four
            .iter()
            .map(|facts| store.by_key(&facts.key).ok_or("the old rank"))
            .collect::<Result<_, _>>()?;
        for row in &old[..2] {
            door.gone(Remover::Registrant(&maker), &row.facts.key, standing(row))
                .map_err(|_| "a low rank removed")?;
        }
        let epoch = store.epoch();
        let two = ranked(2);
        assert_eq!(
            find_rows_over(&*door, &maker, Precondition::Absent, &two),
            Err(MembershipWriteRefused::Stands)
        );
        assert_eq!(store.epoch(), epoch);

        let answer = find_rows(&*door, &maker, &two).map_err(|_| "two ranks registered")?;
        assert!(matches!(
            answer[..2],
            [
                MembershipWriteOutcome::Registered(_),
                MembershipWriteOutcome::Registered(_)
            ]
        ));
        assert_eq!(
            answer[2..],
            [
                MembershipWriteOutcome::Removed(old[2].id),
                MembershipWriteOutcome::Removed(old[3].id)
            ]
        );
        assert!(four[2..]
            .iter()
            .all(|facts| store.by_key(&facts.key).is_none()));
        Ok(())
    }

    #[test]
    fn a_plain_worker_that_comes_back_ranked_loses_its_plain_row() {
        check_ranked_then_plain(local_door).expect("plain, ranked and plain again");
    }

    fn check_ranked_then_plain(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Registrant::Control;
        find_rows(&*door, &maker, &[stated()]).map_err(|_| "registered the plain row")?;
        let plain = store.by_key(URL).ok_or("the plain row")?;
        let four = ranked(4);
        let answer = find_rows(&*door, &maker, &four).map_err(|_| "registered four ranks")?;
        assert_eq!(answer.len(), 5);
        assert_eq!(answer[4], MembershipWriteOutcome::Removed(plain.id));
        assert_eq!(store.by_key(URL), None);
        let ids: Vec<_> = four
            .iter()
            .map(|facts| store.by_key(&facts.key).map(|row| row.id).ok_or("the rank"))
            .collect::<Result<_, _>>()?;
        let answer = find_rows(&*door, &maker, &[stated()]).map_err(|_| "plain again")?;
        let fresh = store.by_key(URL).ok_or("the new plain row")?;
        let expected: Vec<_> = std::iter::once(MembershipWriteOutcome::Registered(fresh.id))
            .chain(ids.into_iter().map(MembershipWriteOutcome::Removed))
            .collect();
        assert_eq!(answer, expected);
        assert!(four.iter().all(|facts| store.by_key(&facts.key).is_none()));
        assert_ne!(plain.id, fresh.id);
        Ok(())
    }

    #[test]
    fn one_row_under_the_engines_own_url_is_the_engine_at_any_width() {
        check_whole_row(local_door).expect("the engine stands as one row");
    }

    /// A worker that serves several engines over one connection states
    /// their count and no rank: one row under the engine's own URL, which
    /// is no rank of that width.
    fn check_whole_row(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Registrant::Control;
        let whole = WorkerFacts {
            dp_size: 4,
            ..stated()
        };
        let answer =
            find_rows(&*door, &maker, std::slice::from_ref(&whole)).map_err(|_| "registered")?;
        let row = store.by_key(URL).ok_or("the engine's one row")?;
        assert_eq!(answer, [MembershipWriteOutcome::Registered(row.id)]);
        assert_eq!((row.facts.dp_rank, row.facts.dp_size), (0, 4));
        // The same facts again write nothing, and another width is laid in
        // place.
        assert_eq!(
            find_rows(&*door, &maker, std::slice::from_ref(&whole)),
            Ok(vec![MembershipWriteOutcome::Unmoved(row.id)])
        );
        let wider = WorkerFacts {
            dp_size: 8,
            ..whole
        };
        assert_eq!(
            find_rows(&*door, &maker, &[wider]),
            Ok(vec![MembershipWriteOutcome::InPlace(row.id)])
        );
        assert_eq!(store.by_key(URL).map(|row| row.facts.dp_size), Some(8));
        // Four ranks retire it, as they retire a row of one.
        let answer = find_rows(&*door, &maker, &ranked(4)).map_err(|_| "four ranks")?;
        assert_eq!(answer.len(), 5);
        assert_eq!(answer[4], MembershipWriteOutcome::Removed(row.id));
        assert_eq!(store.by_key(URL), None);
        Ok(())
    }

    #[test]
    fn another_makers_row_of_the_engine_refuses_the_registration_whole() {
        check_other_maker(local_door).expect("the control row stands alone");
    }

    fn check_other_maker(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let provider = Registrant::provider("pool", "private/key").map_err(|_| "the provider")?;
        find_rows(&*door, &Registrant::Control, &[stated()])
            .map_err(|_| "the control registration")?;
        let before = store.all_rows();
        assert_eq!(
            find_rows(&*door, &provider, &ranked(4)),
            Err(MembershipWriteRefused::NotRegistrant(
                "the row was made by a control call: its maker alone changes it".to_owned()
            ))
        );
        assert_eq!(store.all_rows(), before);
        Ok(())
    }

    #[test]
    fn ranks_that_are_not_zero_to_the_width_each_once_are_refused() {
        check_invalid_ranks(local_door);
    }

    fn check_invalid_ranks(open: Open) {
        let (store, door) = open();
        let four = ranked(4);
        let mut duplicate = four.clone();
        duplicate[1] = four[0].clone();
        let mut past_width = four.clone();
        past_width[3].dp_rank = 4;
        past_width[3].key = format!("{URL}@4");
        let mut another_engine = four.clone();
        another_engine[1].engine = "http://other:8000".to_owned();
        // The door reads the engine from the first row: a first row that
        // states another, or every row stating another, is held to its key.
        let mut another_first = four.clone();
        another_first[0].engine = "http://other:8000".to_owned();
        let mut another_all = four.clone();
        for facts in &mut another_all {
            facts.engine = "http://other:8000".to_owned();
        }
        let mut wrong_key = four.clone();
        wrong_key[1].key = format!("{URL}@01");
        let mut wrong_width = four.clone();
        wrong_width[1].dp_size = 2;
        let mut wrong_plain = stated();
        wrong_plain.dp_rank = 1;
        let missing_rank = four[..3].to_vec();
        for ranks in [
            duplicate,
            past_width,
            another_engine,
            another_first,
            another_all,
            wrong_key,
            wrong_width,
            vec![wrong_plain],
            missing_rank,
        ] {
            assert!(
                matches!(
                    find_rows(&*door, &Registrant::Control, &ranks),
                    Err(MembershipWriteRefused::Past(FactPastBound {
                        fact: "dp_rank",
                        ..
                    }))
                ),
                "{ranks:?}"
            );
            assert_eq!(store.all_rows(), (0, Vec::new()));
        }
        // Facts under another key than their row's URL are refused by the
        // key's name, ahead of the group's shape.
        let row = Stated {
            url: "http://other:8000",
            state: &|_| stated(),
        };
        assert!(matches!(
            door.rows(&Registrant::Control, Precondition::Any, &[row]),
            Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "key",
                ..
            }))
        ));
        assert_eq!(store.all_rows(), (0, Vec::new()));
    }

    #[test]
    fn the_same_ranks_stated_again_write_nothing() {
        let (store, door) = door();
        check_unchanged(&store, &door).expect("the same four ranks stand");
    }

    fn check_unchanged(store: &MemoryStore, door: &dyn Rows) -> Result<(), &'static str> {
        let maker = Registrant::Control;
        let four = ranked(4);
        let first = find_rows(door, &maker, &four).map_err(|_| "registered four ranks")?;
        let epoch = store.epoch();
        let expected = first
            .into_iter()
            .map(|wrote| match wrote {
                MembershipWriteOutcome::Registered(id) => Ok(MembershipWriteOutcome::Unmoved(id)),
                _ => Err("a new rank must be registered"),
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(find_rows(door, &maker, &four), Ok(expected));
        assert_eq!(store.epoch(), epoch);
        Ok(())
    }

    #[test]
    fn the_write_of_one_row_holds_no_group_rule_and_takes_no_row() {
        check_one_row(local_door);
    }

    /// The first form of the write, as its callers hand it: a rank laid
    /// alone, and one rank's row changed beside the other.
    pub(crate) fn check_one_row(open: Open) {
        let (store, door) = open();
        let ranks = ranked(2);
        let one = |facts: &WorkerFacts, expect| {
            door.found(&Registrant::Control, expect, &facts.key, &|_| facts.clone())
        };
        let low = one(&ranks[0], Precondition::Absent);
        assert!(
            matches!(low, Ok(MembershipWriteOutcome::Registered(_))),
            "{low:?}"
        );
        // An add of the row again is refused before a fact is stated,
        // whoever made the row: one row has no rest to lay.
        let unstated = |_: Option<&WorkerFacts>| -> WorkerFacts { panic!("no fact is stated") };
        let pool = Registrant::provider("pool", "private/key").expect("a maker");
        for maker in [&Registrant::Control, &pool] {
            assert_eq!(
                door.found(maker, Precondition::Absent, &ranks[0].key, &unstated),
                Err(MembershipWriteRefused::Stands)
            );
        }
        // The rows write holds the engine to its width where one rank stands.
        let epoch = store.epoch();
        assert!(matches!(
            find_rows(&*door, &Registrant::Control, &ranks[1..]),
            Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "dp_rank",
                ..
            }))
        ));
        assert_eq!(store.epoch(), epoch);
        let high = one(&ranks[1], Precondition::Absent);
        assert!(
            matches!(high, Ok(MembershipWriteOutcome::Registered(_))),
            "{high:?}"
        );

        // One rank's row is changed where it stands, and the other stays.
        let other = store.by_key(&ranks[0].key).expect("the low rank");
        let seen = standing(&store.by_key(&ranks[1].key).expect("the high rank"));
        let resized = WorkerFacts {
            max_running: Some(8),
            ..ranks[1].clone()
        };
        let wrote = one(&resized, Precondition::AtRevision(seen));
        assert!(
            matches!(
                wrote,
                Ok(MembershipWriteOutcome::InPlace(_) | MembershipWriteOutcome::Another { .. })
            ),
            "{wrote:?}"
        );
        assert_eq!(store.by_key(&ranks[0].key), Some(other));
        assert_eq!(
            store.by_key(&ranks[1].key).map(|row| row.facts.max_running),
            Some(Some(8))
        );
    }

    #[test]
    fn an_add_tried_again_lays_the_ranks_its_first_attempt_left() {
        check_same_add(local_door).expect("the same add");
    }

    /// The rows a lost lead leaves, then the same add: the rank that stands
    /// answers unmoved under its id and the others are registered. With
    /// every rank standing the engine is whole, and the add is refused.
    fn check_same_add(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let three = ranked(3);
        let first = door
            .found(
                &Registrant::Control,
                Precondition::Absent,
                &three[0].key,
                &|_| three[0].clone(),
            )
            .map_err(|_| "the first rank alone")?;
        let add = || find_rows_over(&*door, &Registrant::Control, Precondition::Absent, &three);
        let wrote = add().map_err(|_| "the same add")?;
        let [MembershipWriteOutcome::Unmoved(stood), MembershipWriteOutcome::Registered(_), MembershipWriteOutcome::Registered(_)] =
            wrote.as_slice()
        else {
            return Err("one rank unmoved and two registered");
        };
        assert_eq!(first, MembershipWriteOutcome::Registered(*stood));
        assert_eq!(store.all_rows().1.len(), 3);
        let epoch = store.epoch();
        assert_eq!(add(), Err(MembershipWriteRefused::Stands));
        assert_eq!(store.epoch(), epoch);
        Ok(())
    }

    #[test]
    fn an_add_over_a_standing_rank_that_is_not_its_own_writes_nothing() {
        check_add_not_its_own(local_door).expect("each add refused");
    }

    /// An add lays the rest by what stands alone. A standing rank whose
    /// facts differ, a rank another maker wrote and a rank past the stated
    /// width each refuse it whole as a row that stands, and nothing is
    /// written.
    fn check_add_not_its_own(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let three = ranked(3);
        door.found(
            &Registrant::Control,
            Precondition::Absent,
            &three[2].key,
            &|_| three[2].clone(),
        )
        .map_err(|_| "the last rank alone")?;
        let before = store.all_rows();
        let refused = |maker: &Registrant, facts: &[WorkerFacts]| {
            let add = find_rows_over(&*door, maker, Precondition::Absent, facts);
            assert_eq!(add, Err(MembershipWriteRefused::Stands));
            assert_eq!(store.all_rows(), before);
        };

        // The add states a label the standing rank does not.
        let mut labelled = three.clone();
        for facts in &mut labelled {
            facts.labels.insert("tier".to_owned(), "gold".to_owned());
        }
        refused(&Registrant::Control, &labelled);
        // Another maker states the same rows.
        let provider = Registrant::provider("pool", "private/key").map_err(|_| "the provider")?;
        refused(&provider, &three);
        // The standing rank is past the width the add states.
        refused(&Registrant::Control, &ranked(2));
        Ok(())
    }

    #[test]
    fn no_row_stated_is_refused() {
        check_empty(local_door);
    }

    fn check_empty(open: Open) {
        let (store, door) = open();
        assert!(matches!(
            door.rows(&Registrant::Control, Precondition::Any, &[]),
            Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "rows",
                ..
            }))
        ));
        assert_eq!(store.all_rows(), (0, Vec::new()));
    }

    #[cfg(feature = "controller")]
    pub(crate) fn check_rank_rows(open: Open) -> Result<(), &'static str> {
        check_four_rows(open)?;
        check_bad_label(open);
        check_narrower(open)?;
        check_partial_removal(open)?;
        check_ranked_then_plain(open)?;
        check_whole_row(open)?;
        check_other_maker(open)?;
        check_invalid_ranks(open);
        check_same_add(open)?;
        check_add_not_its_own(open)?;
        let (store, door) = open();
        check_unchanged(&store, &*door)?;
        check_empty(open);
        Ok(())
    }

    #[test]
    fn a_registration_is_laid_over_what_it_expects() {
        check_expectations(local_door);
    }

    #[test]
    fn a_found_row_states_its_event_signal() {
        check_event_signal(local_door);
    }

    /// A generation whose prefill profile of `pd/*` reads the index, and no
    /// other profile of it.
    fn prefill_reads_the_index() -> AppliedSpec {
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "default": { "filter": ["healthy"], "pick": "round-robin" },
            "prefill": { "filter": ["healthy"], "score": ["prefix-affinity"], "pick": "best" },
            "decode": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
        let mut spec: PolicySpec = serde_json::from_str(text).expect("a well-formed document");
        spec.generation = 1;
        AppliedSpec::validate(&spec).expect("the document validates")
    }

    /// The door states a found row's event signal itself, whatever the
    /// facts state: what the generation in force reads for the row's first
    /// model in the phase of its role, and only for a gRPC worker. A write
    /// under a later generation states it again.
    pub(crate) fn check_event_signal(open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>)) {
        const PAIRED: &str = "pd/x";
        // The row's model, role and wire, what its facts state, and what the
        // door states.
        let rows = [
            (PAIRED, Role::Prefill, WireModes::GRPC, false, true),
            (PAIRED, Role::Decode, WireModes::GRPC, true, false),
            (PAIRED, Role::Single, WireModes::GRPC, true, false),
            (PAIRED, Role::Prefill, WireModes::HTTP, true, false),
            ("other", Role::Prefill, WireModes::GRPC, true, false),
        ];
        for (model, role, wire_modes, events, asked) in rows {
            let (store, door) = open();
            store
                .apply(prefill_reads_the_index())
                .expect("the first generation");
            let found = WorkerFacts {
                models: vec![model.to_owned()],
                role,
                wire_modes,
                signals: Signals {
                    events,
                    ..Signals::default()
                },
                ..stated()
            };
            door.found_one_over(&Registrant::Control, Precondition::Absent, URL, &|_| {
                found.clone()
            })
            .expect("registered");
            let row = store.by_key(URL).expect("the found row");
            assert_eq!(
                row.facts.signals.events, asked,
                "{model} {role:?} {wire_modes:?}"
            );
            assert_eq!(indexed_under(&row).is_some(), asked);

            // Every profile of the cache-aware preset reads the index.
            let mut later = PolicySpec::preset("cache_aware").expect("the preset");
            later.generation = 2;
            store
                .apply(AppliedSpec::validate(&later).expect("the preset validates"))
                .expect("the second generation");
            door.found_one(&Registrant::Control, URL, &|_| found.clone())
                .expect("laid again");
            let row = store.by_key(URL).expect("the row laid again");
            assert_eq!(
                row.facts.signals.events,
                wire_modes == WireModes::GRPC,
                "{model} {role:?} {wire_modes:?} under the later generation"
            );
        }
    }

    /// An add registers an absent URL. Where every row it states stands it
    /// is refused before a fact is stated, whoever made the rows. A
    /// replacement is laid over the row its asker saw by the rule every
    /// registration takes, and is refused where that row moved or left. An
    /// add of ranks over a standing rank that differs writes no rank.
    pub(crate) fn check_expectations(open: Open) {
        let (store, door) = open();
        let control = Registrant::Control;
        let pool = Registrant::provider("pool", "private/key").expect("a maker");
        let unstated = |_: Option<&WorkerFacts>| -> WorkerFacts { panic!("no fact is stated") };

        let added = door.found_one_over(&control, Precondition::Absent, URL, &|_| stated());
        let row = store.by_key(URL).expect("the added row");
        assert_eq!(added, Ok(MembershipWriteOutcome::Registered(row.id)));
        let epoch = store.epoch();
        for maker in [&control, &pool] {
            assert_eq!(
                door.found_one_over(maker, Precondition::Absent, URL, &unstated),
                Err(MembershipWriteRefused::Stands)
            );
        }

        let seen = Precondition::AtRevision(standing(&row));
        assert_eq!(
            door.found_one_over(&control, seen, URL, &|_| stated()),
            Ok(MembershipWriteOutcome::Unmoved(row.id))
        );
        assert_eq!(
            door.found_one_over(&pool, seen, URL, &unstated),
            Err(MembershipWriteRefused::NotRegistrant(
                "the row was made by a control call: its maker alone changes it".to_owned()
            ))
        );
        assert_eq!(store.epoch(), epoch);

        // New labels are laid in place, and the row's revision moves on.
        let relabelled = |_: Option<&WorkerFacts>| WorkerFacts {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..stated()
        };
        assert_eq!(
            door.found_one_over(&control, seen, URL, &relabelled),
            Ok(MembershipWriteOutcome::InPlace(row.id))
        );
        assert_eq!(
            door.found_one_over(&control, seen, URL, &unstated),
            Err(MembershipWriteRefused::Moved)
        );

        // Other models are the URL's next life, as under an upsert.
        let moved = store.by_key(URL).expect("the relabelled row");
        let other = |_: Option<&WorkerFacts>| WorkerFacts {
            models: vec!["other".to_owned()],
            ..stated()
        };
        let wrote = door.found_one_over(
            &control,
            Precondition::AtRevision(standing(&moved)),
            URL,
            &other,
        );
        let next = store.by_key(URL).expect("the next life");
        assert_eq!(
            wrote,
            Ok(MembershipWriteOutcome::Another {
                old: row.id,
                new: next.id
            })
        );

        // A draining row is leaving: its removal wins over a replacement,
        // which is told so. Another maker is told the row is not its own.
        let seen = standing(&next);
        assert_eq!(
            door.drained(Remover::Registrant(&control), URL, seen),
            Ok(MembershipWriteOutcome::Draining(next.id))
        );
        assert_eq!(
            door.found_one_over(&control, Precondition::AtRevision(seen), URL, &unstated),
            Err(MembershipWriteRefused::Draining)
        );
        assert!(matches!(
            door.found_one_over(&pool, Precondition::AtRevision(seen), URL, &unstated),
            Err(MembershipWriteRefused::NotRegistrant(_))
        ));
        assert_eq!(
            store.by_key(URL).map(|row| (row.id, row.status)),
            Some((next.id, WorkerStatus::Draining))
        );

        // The row its asker saw left.
        assert_eq!(
            door.gone(Remover::Registrant(&control), URL, seen),
            Ok(MembershipWriteOutcome::Removed(next.id))
        );
        assert_eq!(
            door.found_one_over(&control, Precondition::AtRevision(seen), URL, &unstated),
            Err(MembershipWriteRefused::Absent)
        );
        assert_eq!(store.by_key(URL), None);

        // The second of two ranks stands with another label: the add
        // writes neither.
        let (store, door) = open();
        let two = ranked(2);
        store.upsert(WorkerFacts {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..two[1].clone()
        });
        let epoch = store.epoch();
        assert_eq!(
            find_rows_over(&*door, &control, Precondition::Absent, &two),
            Err(MembershipWriteRefused::Stands)
        );
        assert_eq!(store.epoch(), epoch);

        // Two ranks are added in one write. The engine's row in its other
        // form refuses an add, and a replacement states one row.
        let (store, door) = open();
        let add =
            |facts: &[WorkerFacts]| find_rows_over(&*door, &control, Precondition::Absent, facts);
        let ranks = add(&two).expect("two ranks added");
        assert!(matches!(
            ranks[..],
            [
                MembershipWriteOutcome::Registered(_),
                MembershipWriteOutcome::Registered(_)
            ]
        ));
        let epoch = store.epoch();
        assert_eq!(add(&[stated()]), Err(MembershipWriteRefused::Stands));
        let rank = store.by_key(&two[0].key).expect("rank zero");
        assert!(matches!(
            find_rows_over(
                &*door,
                &control,
                Precondition::AtRevision(standing(&rank)),
                &two
            ),
            Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "rows",
                ..
            }))
        ));
        assert_eq!(store.epoch(), epoch);
        find_rows(&*door, &control, &[stated()]).expect("plain again");
        let epoch = store.epoch();
        assert_eq!(add(&two), Err(MembershipWriteRefused::Stands));
        assert_eq!(store.epoch(), epoch);
    }

    #[test]
    fn a_removal_takes_only_the_row_its_asker_saw() {
        check_guard(local_door).expect("both registrations stand in turn");
    }

    pub(crate) fn check_guard(
        open: impl FnOnce() -> (Arc<MemoryStore>, Box<dyn Rows>),
    ) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Registrant::Control;
        let by = Remover::Registrant(&maker);
        assert!(door.found_one(&maker, URL, &|_| stated()).is_ok());
        let first = store.by_key(URL).ok_or("the first row")?;
        assert!(door
            .found_one(&maker, URL, &|_| WorkerFacts {
                role: Role::Decode,
                ..stated()
            })
            .is_ok());
        let replacement = store.by_key(URL).ok_or("the replacement row")?;
        assert_ne!(first.id, replacement.id);
        assert_eq!(first.revision, replacement.revision);
        let epoch = store.epoch();
        let wrong_revision = RowRevision {
            revision: replacement.revision + 1,
            ..standing(&replacement)
        };
        for seen in [wrong_revision, standing(&first)] {
            assert_eq!(
                door.drained(by, URL, seen),
                Err(MembershipWriteRefused::Moved)
            );
            assert_eq!(door.gone(by, URL, seen), Err(MembershipWriteRefused::Moved));
            assert_eq!(
                door.gone(Remover::Provider("other"), URL, seen),
                Err(MembershipWriteRefused::Moved)
            );
        }
        assert_eq!(store.by_key(URL), Some(replacement));
        assert_eq!(store.epoch(), epoch);
        Ok(())
    }

    #[test]
    fn the_hosts_own_source_removes_its_rows_under_any_key() {
        check_provider(local_door).expect("the makers and their rows");
    }

    pub(crate) fn check_provider(
        open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>),
    ) -> Result<(), &'static str> {
        let own = Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        let another =
            Registrant::provider("other", "private/key").map_err(|_| "another source's maker")?;
        for (maker, refusal) in [
            (own, None),
            (
                Registrant::Control,
                Some("the row was made by a control call: its maker alone changes it"),
            ),
            (
                another,
                Some("the row is provider other's: its worker is changed at the provider's source"),
            ),
        ] {
            let (store, door) = open();
            assert!(door.found_one(&maker, URL, &|_| stated()).is_ok());
            let row = store.by_key(URL).ok_or("the registered row")?;
            let by = Remover::Provider("pool");
            let seen = standing(&row);
            if let Some(refusal) = refusal {
                let expected = Err(MembershipWriteRefused::NotRegistrant(refusal.to_owned()));
                assert_eq!(door.drained(by, URL, seen), expected);
                assert_eq!(door.gone(by, URL, seen), expected);
                assert_eq!(store.by_key(URL), Some(row));
            } else {
                assert_eq!(
                    door.drained(by, URL, seen),
                    Ok(MembershipWriteOutcome::Draining(row.id))
                );
                assert_eq!(
                    door.gone(by, URL, seen),
                    Ok(MembershipWriteOutcome::Removed(row.id))
                );
                assert_eq!(store.by_key(URL), None);
            }
        }
        Ok(())
    }

    #[test]
    fn health_removes_inside_its_scope_alone() {
        check_health_scope(local_door).expect("the makers and their rows");
    }

    pub(crate) fn check_health_scope(
        open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>),
    ) -> Result<(), &'static str> {
        let provider =
            Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        for (scope, control, discovered) in [
            (UnhealthyRemovalScope::Disabled, false, false),
            (UnhealthyRemovalScope::DiscoveryOwned, false, true),
            (UnhealthyRemovalScope::AllLocal, true, true),
        ] {
            for (maker, allowed) in [(&Registrant::Control, control), (&provider, discovered)] {
                let (store, door) = open();
                assert!(door.found_one(maker, URL, &|_| stated()).is_ok());
                let row = store.by_key(URL).ok_or("the registered row")?;
                let by = Remover::Health(scope);
                let seen = standing(&row);
                if allowed {
                    assert_eq!(
                        door.drained(by, URL, seen),
                        Ok(MembershipWriteOutcome::Draining(row.id))
                    );
                    assert_eq!(
                        door.gone(by, URL, seen),
                        Ok(MembershipWriteOutcome::Removed(row.id))
                    );
                    assert_eq!(store.by_key(URL), None);
                } else {
                    assert_eq!(
                        door.drained(by, URL, seen),
                        Err(MembershipWriteRefused::OutOfScope)
                    );
                    assert_eq!(
                        door.gone(by, URL, seen),
                        Err(MembershipWriteRefused::OutOfScope)
                    );
                    assert_eq!(store.by_key(URL), Some(row));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn a_removal_leaves_the_rows_maker_as_it_was() {
        check_maker(local_door).expect("the maker and its row");
    }

    pub(crate) fn check_maker(
        open: impl FnOnce() -> (Arc<MemoryStore>, Box<dyn Rows>),
    ) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker =
            Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        assert!(door.found_one(&maker, URL, &|_| stated()).is_ok());
        let row = store.by_key(URL).ok_or("the registered row")?;
        assert_eq!(
            door.drained(Remover::Provider("pool"), URL, standing(&row)),
            Ok(MembershipWriteOutcome::Draining(row.id))
        );
        assert_eq!(
            store.by_key(URL),
            Some(WorkerRow {
                status: WorkerStatus::Draining,
                ..row
            })
        );
        Ok(())
    }

    /// A worker a registration or a refresh built at [`URL`], under one
    /// label.
    fn built(tier: &str) -> BasicWorkerBuilder {
        <BasicWorkerBuilder>::new(URL)
            .model(ModelCard::new("m"))
            .label("tier", tier)
    }

    #[test]
    fn a_refresh_writes_its_workers_labels_and_no_other_fact() {
        check_refresh(local_door).expect("the makers and their rows");
    }

    /// A refresh writes the labels its rebuilt worker states over the row
    /// its asker saw, whoever made the row. The row keeps its maker and
    /// every other fact, and equal labels write no revision. A worker that
    /// states another fact is refused in the refresh's own variant, a label
    /// past its bound is refused, and a row that moved, drains or left
    /// takes no refresh. A refused refresh writes nothing.
    pub(crate) fn check_refresh(
        open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>),
    ) -> Result<(), &'static str> {
        let provider =
            Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        let worker = |built: BasicWorkerBuilder| -> Arc<dyn Worker> { Arc::new(built.build()) };
        for maker in [&Registrant::Control, &provider] {
            let (store, door) = open();
            let first = worker(built("gold"));
            assert!(door
                .found_built(maker, Precondition::Absent, &first, None)
                .is_ok());
            let row = store.by_key(URL).ok_or("the registered row")?;
            let seen = standing(&row);

            let equal = door.refreshed(seen, &worker(built("gold")), &first);
            assert_eq!(equal, Ok(MembershipWriteOutcome::Unmoved(row.id)));

            // Each states one fact more than a label: a model, a role, a
            // health threshold.
            let checks = HealthCheckConfig {
                failure_threshold: 9,
                ..HealthCheckConfig::default()
            };
            let strays = [
                built("silver").model(ModelCard::new("other")),
                built("silver").worker_type(WorkerType::Prefill),
                built("silver").health_config(checks),
            ];
            for stray in strays {
                let refused = door.refreshed(seen, &worker(stray), &first);
                assert_eq!(refused, Err(MembershipWriteRefused::NotLabelsAlone));
            }
            let wide = "x".repeat(WorkerFacts::LABEL_VALUE_BYTES + 1);
            let refused = door.refreshed(seen, &worker(built(&wide)), &first);
            assert!(
                matches!(
                    refused,
                    Err(MembershipWriteRefused::Past(FactPastBound {
                        fact: "labels",
                        ..
                    }))
                ),
                "{refused:?}"
            );
            assert_eq!(store.by_key(URL).as_ref(), Some(&row));

            let wrote = door.refreshed(seen, &worker(built("silver")), &first);
            assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
            let moved = store.by_key(URL).ok_or("the refreshed row")?;
            let mut facts = row.facts.clone();
            facts.labels.insert("tier".to_owned(), "silver".to_owned());
            assert_eq!(
                moved,
                WorkerRow {
                    facts,
                    revision: row.revision + 1,
                    ..row
                }
            );

            // The asker's row moved under it, then drains, then is gone.
            let late = worker(built("bronze"));
            assert_eq!(
                door.refreshed(seen, &late, &first),
                Err(MembershipWriteRefused::Moved)
            );
            let seen = standing(&moved);
            let by = Remover::Registrant(maker);
            assert_eq!(
                door.drained(by, URL, seen),
                Ok(MembershipWriteOutcome::Draining(moved.id))
            );
            assert_eq!(
                door.refreshed(seen, &late, &first),
                Err(MembershipWriteRefused::Draining)
            );
            assert_eq!(
                door.gone(by, URL, seen),
                Ok(MembershipWriteOutcome::Removed(moved.id))
            );
            assert_eq!(
                door.refreshed(seen, &late, &first),
                Err(MembershipWriteRefused::Absent)
            );
        }
        Ok(())
    }

    /// A refresh's scope is its worker's labels and the pairing descriptor
    /// they derive. A descriptor that moved is laid in place, with a label
    /// or by itself, and the row keeps its id, its maker and every other
    /// fact. One that differs from the row's by its blanks alone is the
    /// row's. One that reports nothing is none, stated or held: it
    /// withdraws a row's descriptor and writes nothing over a row with
    /// none. Another fact beside a moved descriptor is refused, and
    /// nothing is written.
    #[test]
    fn a_refresh_lays_the_pairing_its_labels_derive() {
        let (store, door) = door();
        let first: Arc<dyn Worker> = Arc::new(built("gold").build());
        let registered = door.found_built(&Registrant::Control, Precondition::Absent, &first, None);
        assert!(registered.is_ok(), "{registered:?}");
        let row = store.by_key(URL).expect("the registered row");
        let version = |version: &str| {
            Some(Arc::new(PairingDescriptor {
                version: Some(version.to_owned()),
                ..PairingDescriptor::default()
            }))
        };
        let refresh = |seen: &WorkerRow, pairing: Option<Arc<PairingDescriptor>>, tier: &str| {
            refresh_stated(&door, standing(seen), URL, &|standing| {
                let mut facts = standing.clone();
                facts.pairing.clone_from(&pairing);
                facts.labels.insert("tier".to_owned(), tier.to_owned());
                facts
            })
        };

        // The descriptor and a label move together.
        let wrote = refresh(&row, version("2"), "silver");
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
        let moved = store.by_key(URL).expect("the refreshed row");
        let mut facts = row.facts.clone();
        facts.labels.insert("tier".to_owned(), "silver".to_owned());
        facts.pairing = version("2");
        assert_eq!(
            moved,
            WorkerRow {
                facts,
                revision: row.revision + 1,
                ..row.clone()
            }
        );

        // The same version between blanks is the row's descriptor.
        let equal = refresh(&moved, version(" 2 "), "silver");
        assert_eq!(equal, Ok(MembershipWriteOutcome::Unmoved(row.id)));
        assert_eq!(store.by_key(URL).as_ref(), Some(&moved));

        // The descriptor moves by itself, then is withdrawn: by none, and
        // by one that reports nothing, which the row holds as none.
        let empty = || Some(Arc::new(PairingDescriptor::default()));
        let mut bare = moved.clone();
        for withdrawn in [None, version(" ")] {
            let wrote = refresh(&bare, version("3"), "silver");
            assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
            let again = store.by_key(URL).expect("the row of a moved descriptor");
            assert_eq!(again.facts.pairing, version("3"));
            assert_eq!(again.facts.labels, moved.facts.labels);
            let wrote = refresh(&again, withdrawn, "silver");
            assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
            bare = store
                .by_key(URL)
                .expect("the row of a withdrawn descriptor");
            assert_eq!(bare.facts.pairing, None);
        }

        // Over a row with none, one that reports nothing writes nothing.
        for nothing in [empty(), version(" ")] {
            let unmoved = refresh(&bare, nothing, "silver");
            assert_eq!(unmoved, Ok(MembershipWriteOutcome::Unmoved(row.id)));
            assert_eq!(store.by_key(URL).as_ref(), Some(&bare));
        }

        // Another fact beside a moved descriptor is outside the scope.
        let stray = refresh_stated(&door, standing(&bare), URL, &|standing| {
            let mut facts = standing.clone();
            facts.pairing = version("4");
            facts.priority += 1;
            facts
        });
        assert_eq!(stray, Err(MembershipWriteRefused::NotLabelsAlone));
        assert_eq!(store.by_key(URL).as_ref(), Some(&bare));

        // A row may hold one that reports nothing, as a registration states
        // it. It reads as a row with none.
        let laid = door.found(&Registrant::Control, Precondition::Any, URL, &|standing| {
            let mut facts = standing.expect("the standing row").clone();
            facts.pairing = empty();
            facts
        });
        assert_eq!(laid, Ok(MembershipWriteOutcome::InPlace(row.id)));
        let held = store.by_key(URL).expect("the row of the registration");
        assert_eq!(held.facts.pairing, empty());
        for nothing in [None, empty()] {
            let unmoved = refresh(&held, nothing, "silver");
            assert_eq!(unmoved, Ok(MembershipWriteOutcome::Unmoved(row.id)));
            assert_eq!(store.by_key(URL).as_ref(), Some(&held));
        }
    }

    #[test]
    fn a_found_worker_is_registered_under_its_maker() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        let wrote = door.found_one(&maker, URL, &|standing| {
            assert!(standing.is_none());
            stated()
        });
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(wrote, Ok(MembershipWriteOutcome::Registered(row.id)));
        assert_eq!(row.maker, maker);
        assert_eq!(
            row.facts,
            WorkerFacts {
                incarnation: membership::life(0, door.base, 0),
                ..stated()
            }
        );
        assert_eq!(
            (row.status, row.revision, store.epoch()),
            (WorkerStatus::Pending, 1, 1)
        );
    }

    #[test]
    fn equal_facts_write_nothing() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(
            door.found_one(&maker, URL, &|_| stated()),
            Ok(MembershipWriteOutcome::Unmoved(row.id))
        );
        let seen = standing(&row);
        assert_eq!(store.by_key(URL), Some(row));
        assert_eq!(store.epoch(), 1);

        assert!(matches!(
            door.gone(Remover::Registrant(&Registrant::Control), URL, seen),
            Err(MembershipWriteRefused::NotRegistrant(_))
        ));
        assert_eq!(store.epoch(), 1);
    }

    #[test]
    fn a_changed_label_is_written_in_place_and_a_changed_role_is_another_life() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        let old = store.by_key(URL).expect("the registered row");
        let labelled = WorkerFacts {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..stated()
        };
        assert_eq!(
            door.found_one(&maker, URL, &|_| labelled.clone()),
            Ok(MembershipWriteOutcome::InPlace(old.id))
        );
        let standing = store.by_key(URL).expect("the refreshed row");
        assert_eq!(standing.id, old.id);
        assert_eq!(standing.facts.incarnation, old.facts.incarnation);
        assert_eq!(standing.facts.labels, labelled.labels);
        assert_eq!(standing.revision, 2);

        let before = store.epoch();
        let wrote = door.found_one(&maker, URL, &|_| WorkerFacts {
            role: Role::Decode,
            ..labelled.clone()
        });
        let row = store.by_key(URL).expect("the replaced row");
        assert_eq!(
            wrote,
            Ok(MembershipWriteOutcome::Another {
                old: old.id,
                new: row.id
            })
        );
        assert_ne!(row.id, old.id);
        assert_eq!(store.get(old.id), None);
        assert_eq!(row.maker, maker);
        assert_eq!(
            row.facts.incarnation,
            membership::life(0, door.base, before)
        );
        assert_eq!((row.facts.role, row.revision), (Role::Decode, 1));
    }

    /// A refresh that changes the runtime a worker states starts another
    /// life. The row that stated the old number is gone when the write is
    /// answered. The new life is handed the old one's status: the worker's
    /// checks are on, and none runs between the two.
    #[test]
    fn a_changed_demand_fact_is_another_life() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        let stating = |runtime| WorkerFacts {
            demand_facts: DemandFacts {
                runtime,
                ..DemandFacts::default()
            },
            ..stated()
        };
        door.found_one(&maker, URL, &|_| stating(7))
            .expect("registered");
        // Its checks are on: a check's word makes the first life ready.
        let pending = store.by_key(URL).expect("the registered row");
        store
            .set_status(pending.id, WorkerStatus::Ready, pending.revision)
            .expect("ready");
        let old = store.by_key(URL).expect("the ready row");

        let wrote = door.found_one(&maker, URL, &|_| stating(9));
        let row = store.by_key(URL).expect("the replaced row");
        assert_eq!(
            wrote,
            Ok(MembershipWriteOutcome::Another {
                old: old.id,
                new: row.id
            })
        );
        assert_eq!(store.get(old.id), None);
        assert_eq!(
            row.facts.demand_facts,
            DemandFacts {
                runtime: 9,
                ..DemandFacts::default()
            }
        );
        assert_eq!(row.status, WorkerStatus::Ready, "handed on by the store");
    }

    #[test]
    fn another_makers_row_refuses_every_write_and_stands() {
        let caller = Registrant::provider("pool", "second").expect("a maker");
        for (maker, why) in [
            (
                Registrant::Control,
                "the row was made by a control call: its maker alone changes it",
            ),
            (
                Registrant::provider("pool", "first").expect("a maker"),
                "the row is provider pool's under another key: that key's caller changes it",
            ),
        ] {
            let (store, door) = door();
            door.found_one(&maker, URL, &|_| stated())
                .expect("registered");
            let row = store.by_key(URL).expect("the registered row");
            let expected = Err(MembershipWriteRefused::NotRegistrant(why.to_owned()));
            assert_eq!(
                door.found_one(&caller, URL, &|_| panic!("only the maker states facts")),
                expected
            );
            assert_eq!(
                door.drained(Remover::Registrant(&caller), URL, standing(&row)),
                expected
            );
            assert_eq!(
                door.gone(Remover::Registrant(&caller), URL, standing(&row)),
                expected
            );
            assert_eq!(store.by_key(URL), Some(row));
            assert_eq!(store.epoch(), 1);
        }
    }

    /// A life is asked a row at a time: a turn that may make no second life
    /// refuses there, and the row written before it stands.
    #[test]
    fn a_life_refused_between_two_rows_leaves_the_row_before() {
        struct OneLife<'a>(&'a MemoryStore, std::cell::Cell<bool>);

        impl Turn for OneLife<'_> {
            fn store(&self) -> &MemoryStore {
                self.0
            }

            over_the_table!();

            fn life(&self) -> Result<u64, MembershipWriteRefused> {
                if self.1.replace(true) {
                    return Err(MembershipWriteRefused::NotLeading(
                        "the term moved".to_owned(),
                    ));
                }
                Ok(7)
            }

            fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
                Ok(())
            }
        }

        let (store, _) = door();
        let two = ranked(2);
        let states: Vec<_> = two
            .iter()
            .map(|facts| move |_: Option<&WorkerFacts>| facts.clone())
            .collect();
        let rows: Vec<_> = two
            .iter()
            .zip(&states)
            .map(|(facts, state)| Stated {
                url: &facts.key,
                state,
            })
            .collect();
        let turn = OneLife(&store, std::cell::Cell::new(false));
        assert_eq!(
            lay(&turn, &Registrant::Control, Precondition::Any, &rows),
            Err(MembershipWriteRefused::NotLeading(
                "the term moved".to_owned()
            ))
        );
        assert!(store.by_key(&two[0].key).is_some());
        assert!(store.by_key(&two[1].key).is_none());
    }

    /// A turn whose term holds for a number of checks and is lost then.
    struct TermFor<'a>(&'a MemoryStore, std::cell::Cell<usize>);

    impl Turn for TermFor<'_> {
        fn store(&self) -> &MemoryStore {
            self.0
        }

        over_the_table!();

        fn check_term(&self) -> Result<(), MembershipWriteRefused> {
            let Some(left) = self.1.get().checked_sub(1) else {
                return Err(MembershipWriteRefused::NotLeading(
                    "the term moved".to_owned(),
                ));
            };
            self.1.set(left);
            Ok(())
        }

        fn life(&self) -> Result<u64, MembershipWriteRefused> {
            Ok(7)
        }

        fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
            Ok(())
        }
    }

    fn lay_under(
        turn: &TermFor<'_>,
        facts: &[WorkerFacts],
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
        lay_as(turn, Precondition::Any, facts)
    }

    fn lay_as(
        turn: &TermFor<'_>,
        expect: Precondition,
        facts: &[WorkerFacts],
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
        let states: Vec<_> = facts
            .iter()
            .map(|facts| move |_: Option<&WorkerFacts>| facts.clone())
            .collect();
        let rows: Vec<_> = facts
            .iter()
            .zip(&states)
            .map(|(facts, state)| Stated {
                url: &facts.key,
                state,
            })
            .collect();
        lay(turn, &Registrant::Control, expect, &rows)
    }

    /// The term is asked ahead of every write, whatever the write makes of
    /// its row. A label laid in place and a rank that leaves are refused
    /// behind a lost lead, and what was written before it stands.
    #[test]
    fn a_term_lost_between_two_rows_stops_a_write_in_place_and_a_removal() {
        let (store, door) = door();
        let three = ranked(3);
        find_rows(&door, &Registrant::Control, &three).expect("three ranks registered");
        let lost = Err(MembershipWriteRefused::NotLeading(
            "the term moved".to_owned(),
        ));
        let term = |checks| TermFor(&store, std::cell::Cell::new(checks));

        // Every rank's label changes. The term holds for the first row.
        let mut labelled = three.clone();
        for facts in &mut labelled {
            facts.labels.insert("tier".to_owned(), "gold".to_owned());
        }
        assert_eq!(lay_under(&term(1), &labelled), lost);
        let labels: Vec<_> = three
            .iter()
            .map(|facts| store.by_key(&facts.key).map(|row| row.facts.labels.len()))
            .collect();
        assert_eq!(labels, [Some(1), Some(0), Some(0)]);

        // The worker comes back two wide. The term holds for the two rows
        // that stay and is lost ahead of the third rank's removal.
        assert_eq!(lay_under(&term(2), &ranked(2)), lost);
        let widths: Vec<_> = three
            .iter()
            .map(|facts| store.by_key(&facts.key).map(|row| row.facts.dp_size))
            .collect();
        assert_eq!(widths, [Some(2), Some(2), Some(3)]);
    }

    /// A row laid alone asks the term ahead of its write.
    #[test]
    fn a_row_laid_alone_under_a_lost_term_writes_nothing() {
        let (store, _) = door();
        let turn = TermFor(&store, std::cell::Cell::new(0));
        assert_eq!(
            lay_one(&turn, &Registrant::Control, Precondition::Any, URL, &|_| {
                stated()
            }),
            Err(MembershipWriteRefused::NotLeading(
                "the term moved".to_owned()
            ))
        );
        assert_eq!(store.all_rows(), (0, Vec::new()));
    }

    /// A lead lost behind an add's first row leaves that row. The same add
    /// under a term that holds lays the rest, and the row that stands
    /// answers unmoved.
    #[test]
    fn an_add_that_lost_its_lead_lays_the_rest_when_it_is_tried_again() {
        let (store, _) = door();
        let three = ranked(3);
        let add = |checks| {
            let turn = TermFor(&store, std::cell::Cell::new(checks));
            lay_as(&turn, Precondition::Absent, &three)
        };
        assert_eq!(
            add(1),
            Err(MembershipWriteRefused::NotLeading(
                "the term moved".to_owned()
            ))
        );
        assert_eq!(store.all_rows().1.len(), 1);
        let wrote = add(3);
        assert!(
            matches!(
                wrote.as_deref(),
                Ok([
                    MembershipWriteOutcome::Unmoved(_),
                    MembershipWriteOutcome::Registered(_),
                    MembershipWriteOutcome::Registered(_)
                ])
            ),
            "{wrote:?}"
        );
        assert_eq!(store.all_rows().1.len(), 3);
    }

    #[test]
    fn a_fact_past_its_bound_writes_nothing() {
        struct Shut<'a>(&'a MemoryStore);

        impl Turn for Shut<'_> {
            fn store(&self) -> &MemoryStore {
                self.0
            }

            over_the_table!();

            fn life(&self) -> Result<u64, MembershipWriteRefused> {
                Ok(0)
            }

            fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
                Err(MembershipWriteRefused::Past(FactPastBound {
                    fact: "row",
                    why: "no room",
                }))
            }
        }

        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        let invalid = WorkerFacts {
            models: vec!["m".repeat(257)],
            ..stated()
        };
        let no = MembershipWriteRefused::Past(FactPastBound {
            fact: "models",
            why: "a model id is at most 256 bytes",
        });
        assert_eq!(
            door.found_one(&maker, URL, &|_| invalid.clone()),
            Err(no.clone())
        );
        assert_eq!(
            lay(
                &Shut(&store),
                &maker,
                Precondition::Any,
                &[Stated {
                    url: URL,
                    state: &|_| invalid.clone()
                }]
            ),
            Err(no.clone())
        );
        assert_eq!(
            lay(
                &Shut(&store),
                &maker,
                Precondition::Any,
                &[Stated {
                    url: URL,
                    state: &|_| stated()
                }]
            ),
            Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "row",
                why: "no room"
            }))
        );
        assert_eq!(store.epoch(), 0);
        assert_eq!(store.by_key(URL), None);

        store.upsert_by(invalid.clone(), maker.clone());
        let standing = store.by_key(URL);
        assert_eq!(door.found_one(&maker, URL, &|_| invalid.clone()), Err(no));
        assert_eq!(store.by_key(URL), standing);
        assert_eq!(store.epoch(), 1);
    }

    #[test]
    fn a_drain_marks_the_makers_row_once_and_a_removal_takes_it() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        let stale = row.revision - 1;
        assert_eq!(
            changed(
                store.set_status(row.id, WorkerStatus::Draining, stale),
                MembershipWriteOutcome::Draining
            ),
            Err(MembershipWriteRefused::Moved)
        );
        assert_eq!(
            changed(store.remove(row.id, stale), MembershipWriteOutcome::Removed),
            Err(MembershipWriteRefused::Moved)
        );
        assert_eq!(
            door.drained(Remover::Registrant(&maker), URL, standing(&row)),
            Ok(MembershipWriteOutcome::Draining(row.id))
        );
        assert_eq!(
            store.by_key(URL),
            Some(WorkerRow {
                status: WorkerStatus::Draining,
                ..row.clone()
            })
        );
        assert_eq!(store.epoch(), 2);
        assert_eq!(
            door.drained(Remover::Registrant(&maker), URL, standing(&row)),
            Ok(MembershipWriteOutcome::Unmoved(row.id))
        );
        assert_eq!(store.epoch(), 2);
        assert_eq!(
            door.gone(Remover::Registrant(&maker), URL, standing(&row)),
            Ok(MembershipWriteOutcome::Removed(row.id))
        );
        assert_eq!(store.epoch(), 3);
        assert_eq!(store.by_key(URL), None);
    }

    #[test]
    fn no_row_under_the_url_is_absent() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        assert_eq!(
            door.drained(Remover::Registrant(&maker), URL, ABSENT),
            Err(MembershipWriteRefused::Absent)
        );
        assert_eq!(
            door.gone(Remover::Registrant(&maker), URL, ABSENT),
            Err(MembershipWriteRefused::Absent)
        );
        assert_eq!(store.epoch(), 0);
    }

    #[test]
    fn a_worker_whose_checks_are_off_is_ready_at_once() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        assert_eq!(
            store.by_key(URL).map(|row| row.status),
            Some(WorkerStatus::Pending)
        );
        let wrote = door.found_one(&maker, URL, &|_| {
            let mut facts = stated();
            facts.health.probes = false;
            facts
        });
        let row = store.by_key(URL).expect("the row with its checks off");
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
        assert_eq!((row.status, row.revision), (WorkerStatus::Ready, 2));

        door.gone(Remover::Registrant(&maker), URL, standing(&row))
            .expect("removed");
        let wrote = door.found_one(&maker, URL, &|_| row.facts.clone());
        let fresh = store.by_key(URL).expect("the newly registered row");
        assert_eq!(wrote, Ok(MembershipWriteOutcome::Registered(fresh.id)));
        assert_eq!((fresh.status, fresh.revision), (WorkerStatus::Ready, 1));
    }

    #[test]
    fn the_facts_are_stated_over_the_row_that_stands() {
        let (store, door) = door();
        let maker = Registrant::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| WorkerFacts {
            zone: Some("east".to_owned()),
            ..stated()
        })
        .expect("registered");
        let old = store.by_key(URL).expect("the registered row");
        let wrote = door.found_one(&maker, URL, &|standing| {
            assert_eq!(standing, Some(&old.facts));
            WorkerFacts {
                zone: standing.and_then(|facts| facts.zone.clone()),
                priority: 5,
                ..stated()
            }
        });
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(old.id)));
        let row = store.by_key(URL).expect("the refreshed row");
        assert_eq!(row.facts.zone.as_deref(), Some("east"));
        assert_eq!(row.facts.priority, 5);

        // Facts under another key are refused in every build: the store
        // writes by the facts' key, and no maker was asked of that row.
        let mismatch = door.found_one(&maker, URL, &|_| WorkerFacts {
            key: "http://another".to_owned(),
            ..stated()
        });
        let refused = MembershipWriteRefused::Past(FactPastBound {
            fact: "key",
            why: "the facts name the URL of the call",
        });
        assert_eq!(mismatch, Err(refused));
        assert_eq!(store.by_key(URL), Some(row));
        assert_eq!(store.by_key("http://another"), None);
    }

    /// The cases in which a decision reads what the door wrote. The engine
    /// over worker lists decides over the lists a holder lends, once the
    /// holder has followed the door's writes.
    #[cfg(feature = "in-process")]
    mod decided {
        use std::{sync::Arc, time::Instant};

        use smg_worker::{
            identity::{basic::Worker, builder::BasicWorkerBuilder},
            model_card::ModelCard,
            registry::WorkerRegistry,
            worker::{RuntimeType, WorkerType},
        };

        use super::{door, stated, URL};
        use crate::{
            document::spec::{AppliedSpec, Pairing, PolicySpec},
            engine::{
                Constraints, DecideOutcome, Decided, Decision, Demands, PairLists, Phase,
                PolicyEngine, Request, RequestFacts, RequestMaker, Role, ShedReason, WorkerId,
                WorkerLists,
            },
            store::{MemoryStore, Registrant, WorkerFacts, WorkerStatus, WorkersStore},
            worker_host::{
                facts, pairing_descriptor, Members, MembershipWriteOutcome, MembershipWriter,
                Objects, OpenAdmission, Precondition,
            },
        };

        const MODEL: &str = "model-a";

        /// A request stated ahead, with the lists its host hands for it.
        struct Asks<'a>(Request<'a>, WorkerLists<'a>);

        impl RequestMaker for Asks<'_> {
            fn model(&self) -> &str {
                self.0.facts.model
            }

            fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
                self.0
            }

            fn workers(&self) -> WorkerLists<'_> {
                self.1
            }
        }

        /// A request for [`MODEL`] at `phase`.
        fn request(phase: Phase) -> Request<'static> {
            Request {
                id: b"request",
                facts: RequestFacts {
                    model: MODEL,
                    seed: 7,
                    phase,
                    ..RequestFacts::default()
                },
                constraints: Constraints::default(),
            }
        }

        /// A holder with a registry of its own, whose objects take their
        /// rows' status. It reads `store` where a case has it follow.
        fn holder() -> Objects<Members> {
            Objects::new(Arc::new(WorkerRegistry::new())).with_status_followed()
        }

        /// What `engine` decides for `asked` over `lists`: a whole decision.
        /// `Asks` states no request plain ([`RequestMaker::plain`]), so
        /// every request is decided whole and none is answered a place.
        fn decided(
            engine: &PolicyEngine,
            asked: Request<'_>,
            lists: WorkerLists<'_>,
        ) -> DecideOutcome {
            let asks = Asks(asked, lists);
            match engine.decide_at(&asks, Instant::now).expect("an answer") {
                Decided::Whole(decided, _) => decided,
                Decided::Alone(_) => panic!("a place for a request that is not plain"),
            }
        }

        /// A worker of a runtime that names its transfer backend by a label,
        /// at `url` in `role`, of `version`.
        fn partner(url: &str, role: WorkerType, version: &str) -> Arc<dyn Worker> {
            Arc::new(
                <BasicWorkerBuilder>::new(url)
                    .model(ModelCard::new(MODEL))
                    .worker_type(role)
                    .runtime_type(RuntimeType::Sglang)
                    .label("disaggregation_transfer_backend", "nixl")
                    .label("version", version)
                    .build(),
            )
        }

        /// A registration that states the pairing descriptor its worker
        /// derived lays it in the row, and the same registration again
        /// writes nothing. One that moves the version label moves the
        /// descriptor in place, and under the strict rule the next decision
        /// over the holder's lists reads it: the classes pair no prefill
        /// with the decode, the pair's request is shed, and it is placed
        /// once the prefill follows.
        #[test]
        fn a_pairing_a_registration_moves_is_read_by_the_next_decision() {
            const DECODE: &str = "http://decode:8000";
            let (store, door) = door();
            let mut strict =
                PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
            strict.generation = 1;
            strict.plans.pd_pair.pairing = Pairing::Strict;
            let applied = AppliedSpec::validate(&strict).expect("the document validates");
            let engine = PolicyEngine::new(applied);
            let prefill = partner(URL, WorkerType::Prefill, "1");
            let decode = partner(DECODE, WorkerType::Decode, "1");
            // A registration of a built worker, which states the worker's
            // facts over the standing row's and the descriptor it derived.
            let state = |built: &Arc<dyn Worker>, expect: Precondition| {
                door.found(&Registrant::Control, expect, built.url(), &|standing| {
                    WorkerFacts {
                        pairing: pairing_descriptor(built.as_ref()),
                        ..facts(built.as_ref(), standing)
                    }
                })
            };
            // Each row is laid, and a check's word makes it ready.
            let ready = |built: &Arc<dyn Worker>| {
                let wrote = state(built, Precondition::Absent);
                assert!(wrote.is_ok(), "{wrote:?}");
                let laid = store.by_key(built.url()).expect("the registered row");
                let _ = store.set_status(laid.id, WorkerStatus::Ready, laid.revision);
                store.by_key(built.url()).expect("the ready row")
            };
            let first = ready(&prefill);
            let row = ready(&decode);
            let said = row.facts.pairing.as_deref().expect("a descriptor");
            assert_eq!(said.transport.as_deref(), Some("nixl"));
            assert_eq!(said.version.as_deref(), Some("1"));
            assert_eq!(facts(decode.as_ref(), Some(&row.facts)), row.facts);
            let again = state(&decode, Precondition::Any);
            assert_eq!(again, Ok(MembershipWriteOutcome::Unmoved(row.id)));
            let holder = holder();
            // A pair's request, decided over the two lists the holder lends
            // once it has followed the store, under the classes it hands
            // beside them: whether the prefill and the decode pair, and the
            // pair the decision picked or what it was shed under.
            let placed = || {
                holder.follow::<MemoryStore>(&store);
                let registry = holder.registry();
                let generation = registry.model_generation(MODEL).expect("a generation");
                let lent = |role: Role| holder.lent(&generation, role);
                let classes = holder.pair_classes(&generation, Pairing::Strict);
                let lists = PairLists {
                    prefill: lent(Role::Prefill),
                    decode: lent(Role::Decode),
                    classes,
                    admission: &OpenAdmission,
                    prefill_places: None,
                    decode_places: None,
                    prefill_stalled: &[],
                    prefill_held: &[],
                    decode_stalled: &[],
                    decode_held: &[],
                };
                let asked = request(Phase::Prefill);
                let picked = match decided(&engine, asked, WorkerLists::Pair(lists)).decision {
                    Decision::Pick(pick) => {
                        let target = pick.targets.first().expect("a target");
                        Ok((target.id, target.pair.map(|pair| pair.decode_id)))
                    }
                    Decision::Shed(shed) => Err((shed.reason, shed.emptied_by)),
                    Decision::Queue(_) => panic!("a pair's request is not queued"),
                };
                (classes.pairs(0, 0), picked)
            };
            let the_pair: Result<(WorkerId, Option<WorkerId>), (ShedReason, &str)> =
                Ok((first.id, Some(row.id)));
            assert_eq!(placed(), (true, the_pair));

            // The decode comes back on another version: the classes pair
            // no prefill with it.
            let rebuilt = partner(DECODE, WorkerType::Decode, "2");
            let wrote = state(&rebuilt, Precondition::Any);
            assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
            let moved = store
                .by_key(DECODE)
                .expect("the row of the moved descriptor");
            let version = moved
                .facts
                .pairing
                .as_deref()
                .and_then(|said| said.version.as_deref());
            assert_eq!(version, Some("2"));
            assert_eq!(
                placed(),
                (false, Err((ShedReason::NoEndpoints, "partnered")))
            );

            // The prefill follows, and the same request is placed again.
            let followed = partner(URL, WorkerType::Prefill, "2");
            let wrote = state(&followed, Precondition::Any);
            assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(first.id)));
            assert_eq!(placed(), (true, the_pair));
        }

        /// A refresh that turns a worker's vision off starts another life,
        /// as one that changes a demand fact does: the door reads the bit
        /// as a hard demand. The row that took images is gone when the
        /// write is answered. Once the holder has followed the write, a
        /// request that needs vision is decided over nobody, and one that
        /// needs none is placed on the new life, which is handed the old
        /// one's status.
        #[test]
        fn a_changed_vision_is_another_life_and_the_new_one_takes_no_image() {
            let (store, door) = door();
            let maker = Registrant::provider("pool", "first").expect("a maker");
            let seeing = |vision| WorkerFacts {
                models: vec![MODEL.to_owned()],
                vision,
                ..stated()
            };
            door.found(&maker, Precondition::Any, URL, &|_| seeing(true))
                .expect("registered");
            // Its checks are on: a check's word makes the first life ready.
            let pending = store.by_key(URL).expect("the registered row");
            store
                .set_status(pending.id, WorkerStatus::Ready, pending.revision)
                .expect("ready");
            let old = store.by_key(URL).expect("the ready row");
            let mut document = PolicySpec::preset("random").expect("a preset of this build");
            document.generation = 1;
            let applied = AppliedSpec::validate(&document).expect("the document validates");
            let engine = PolicyEngine::new(applied);
            let holder = holder();
            // A request that states whether it needs vision, decided over
            // the list the holder lends once it has followed the store.
            let decide = |vision: bool| {
                holder.follow::<MemoryStore>(&store);
                let registry = holder.registry();
                let generation = registry.model_generation(MODEL).expect("a generation");
                let list = holder.lent(&generation, Role::Single);
                let mut asked = request(Phase::Single);
                asked.constraints.demands = Demands {
                    vision,
                    ..Demands::default()
                };
                decided(&engine, asked, WorkerLists::Single(list))
            };
            let placed = decide(true);
            let target = placed.decision.pick().expect("the worker takes images");
            assert_eq!(target.targets.first().map(|target| target.id), Some(old.id));

            let wrote = door.found(&maker, Precondition::Any, URL, &|_| seeing(false));
            let row = store.by_key(URL).expect("the replaced row");
            assert_eq!(
                wrote,
                Ok(MembershipWriteOutcome::Another {
                    old: old.id,
                    new: row.id
                })
            );
            assert_eq!(store.get(old.id), None);
            assert!(!row.facts.vision);
            assert_eq!(row.status, WorkerStatus::Ready, "handed on by the store");

            let refused = decide(true);
            assert!(
                refused.decision.pick().is_none() && refused.lease.is_none(),
                "the worker takes no image: {refused:?}"
            );
            let plain = decide(false);
            let target = plain.decision.pick().expect("the request needs no vision");
            assert_eq!(target.targets.first().map(|target| target.id), Some(row.id));
        }
    }
}
