//! Writes through a host's membership door.

use std::{mem, slice, sync::Arc};

use parking_lot::Mutex;
use smg_worker::{
    identity::{basic::Worker, builder::rank_url},
    settings::UnhealthyRemovalScope,
};

use super::{
    found::{facts, stated},
    kv_feed::asks_events,
    Standing,
};
use crate::{
    store::{
        membership::{self, NotMaker, Past},
        pairing::PairingDescriptor,
        HealthSink, Laid, Maker, MembershipChange, MemoryStore, WorkerFacts, WorkerPatch,
        WorkerRow, WorkerStatus, WorkersStore,
    },
    view::WorkerId,
};

/// What a write of the door made of the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrote {
    Registered(WorkerId),
    InPlace(WorkerId),
    Another { old: WorkerId, new: WorkerId },
    Unmoved(WorkerId),
    Draining(WorkerId),
    Removed(WorkerId),
}

impl Wrote {
    fn changed(&self) -> bool {
        !matches!(self, Self::Unmoved(_))
    }
}

/// A write the door refused. Nothing was written, with one exception: a
/// registration's rows refused as [`Refused::NotLeading`] behind their first
/// write keep the rows written before the lead was lost.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Refused {
    /// Another maker's row, named by its maker.
    NotMaker(String),
    /// A fact the door does not take, or the row past its size. A fact is
    /// refused past its bound, under a key that is not the call's URL, or,
    /// at the controller's door, when its log does not carry it to a standby
    /// or when it states a host-local wire and names no gateway.
    Past(Past),
    /// No row stands under the URL.
    Absent,
    /// The host does not lead now, with its reason.
    NotLeading(String),
    /// Another writer moved the row under the call.
    Moved,
    /// A row stands where an add expects none: under a URL it states, or
    /// one it would retire. The rows of the same add, tried again, are not
    /// such rows ([`Expect::Nothing`]).
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

impl Refused {
    pub(crate) const OUT_OF_SCOPE: &str = "the row's maker is outside the health removal scope";
    pub(crate) const NOT_LABELS_ALONE: &str =
        "a refresh writes a row's labels and the pairing they derive, and states no other fact";
}

/// Who asks to drain or remove a row.
#[derive(Clone, Copy, Debug)]
pub enum Remover<'a> {
    /// A caller acting as the row's whole maker.
    Maker(&'a Maker),
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
pub enum Expect {
    /// Nothing: an add. A row standing under a URL it states, or one the
    /// registration would retire, refuses it whole ([`Refused::Stands`]),
    /// whoever made the row.
    ///
    /// One standing is the add's own. A lead lost between two rows leaves
    /// the rows written before it, and the same add tried again lays the
    /// rest: every row that stands under a URL it states is its maker's
    /// and states equal facts, and a row is left to write. The standing
    /// rows answer unmoved and the others are registered. An engine that
    /// stands whole refuses the add, and so does a standing row that
    /// differs or is another maker's.
    Nothing,
    /// The row its asker saw, at the revision it saw: a replacement by id,
    /// which states that one row. No row refuses it ([`Refused::Absent`]),
    /// and so does another row or revision ([`Refused::Moved`]). A draining
    /// row refuses it too ([`Refused::Draining`]): its removal wins.
    Seen(Standing),
    /// Anything: an upsert.
    Anything,
}

/// The object a guarded write replaces, named by the write's verb: the
/// object its asker read and built from.
///
/// A kept object can change while its row's revision stands, so a door
/// that hands the built worker in refuses the write as moved where another
/// object stands for the row by then. A door that keeps no object has
/// nothing to hold it against, and still reads the verb.
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

/// The membership door a host hands its registration and removal steps.
pub trait Membership: Send + Sync + 'static {
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
        maker: &Maker,
        expect: Expect,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused>;

    /// Lays the rows one registration states of one engine, each with the
    /// worker its registration built, over what it expects to stand. A
    /// row's URL is its worker's, and its facts are the ones the worker
    /// states over the standing row ([`facts`]), by the
    /// write's verb. The engine is the one the rows' own facts state, so no
    /// caller names it another way.
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
    /// refuses there ([`Refused::NotLeading`]), and what was written before
    /// it stands. An upsert lays the rest, and so does the same add tried
    /// again ([`Expect::Nothing`]). A guarded write ([`Expect::Seen`])
    /// states the one row its asker saw and changes it where it stands, by
    /// that row's own rule.
    ///
    /// Each door says what it does with the objects. A door whose host
    /// keeps built objects hands them to the holder for the write's turn
    /// ([`Objects::handing`](super::Objects::handing)). A door over a door
    /// passes the call on. Any other door writes the facts alone, and its
    /// host's holder builds each object from its row.
    ///
    /// `over` names the object a guarded write replaces, by the write's
    /// verb ([`Over`]). A door that hands the objects in holds the write
    /// against that object. Every door reads the verb: a replacement by id
    /// states its thresholds whole, and any other write keeps the standing
    /// row's where a built worker states none.
    fn found_rows(
        &self,
        maker: &Maker,
        expect: Expect,
        built: &[Arc<dyn Worker>],
        over: Option<Over<'_>>,
    ) -> Result<Vec<Wrote>, Refused>;

    /// Lay the registration of one built worker:
    /// [`found_rows`](Self::found_rows) with one row.
    fn found_built(
        &self,
        maker: &Maker,
        expect: Expect,
        built: &Arc<dyn Worker>,
        over: Option<Over<'_>>,
    ) -> Result<Wrote, Refused> {
        let wrote = self.found_rows(maker, expect, slice::from_ref(built), over)?;
        wrote.first().copied().ok_or(Refused::Moved)
    }

    /// Mark the row the asker saw as draining. Its revision stays the same.
    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused>;

    /// Remove the row the asker saw under the registered URL.
    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused>;

    /// Write the labels of the worker the health loop's refresh rebuilt,
    /// and the pairing descriptor they derive, over the row the asker saw
    /// under the worker's URL. Both are the engine's word and nobody's
    /// registration: no maker is asked, a control call's row and a
    /// provider's are served alike, and the row keeps the maker it has.
    ///
    /// The facts `built` states over the standing row ([`facts`]) may
    /// differ from the row's in the labels and the descriptor alone. Any
    /// other difference is refused ([`Refused::NotLabelsAlone`]), so a
    /// refresh starts no life. No row, another row or revision, and a
    /// draining row refuse it too. The stated facts' room is tested as a
    /// registration's is: the controller's door, whose log carries no
    /// descriptor, refuses a refresh that states one as it refuses such a
    /// registration. Equal labels and an equal descriptor write nothing
    /// and answer [`Wrote::Unmoved`].
    ///
    /// Each door says what it does with the object, as for
    /// [`Self::found_built`]. A door whose host keeps built objects hands
    /// `built` to the holder in the write's turn, whether or not a label
    /// moved: a refresh can rebuild what no row states. A door over a door
    /// passes the call on. Any other door writes the row alone.
    ///
    /// `over` is the object the refresh rebuilt from. Equal labels and an
    /// equal descriptor move no revision, so the row's guard cannot tell an
    /// older rebuild from a newer one: a door that hands `built` in refuses
    /// the refresh as moved where another object stands for the row by
    /// then, or none. It asks ahead of the row's guard, so a worker that
    /// left reads as moved there. A door that keeps no object has nothing
    /// to hold it against.
    fn refreshed(
        &self,
        seen: Standing,
        built: &Arc<dyn Worker>,
        over: &Arc<dyn Worker>,
    ) -> Result<Wrote, Refused>;
}

/// What a door owns of a write, beside the rules every door shares.
pub(crate) trait Turn {
    fn store(&self) -> &MemoryStore;

    /// Check the term ahead of a write. A registration asks before each
    /// row it writes and each row it removes.
    fn check_term(&self) -> Result<(), Refused> {
        Ok(())
    }

    /// The number of the life a registration makes now, or a refusal where
    /// the turn may make none: under a term its log no longer leads.
    fn life(&self) -> Result<u64, Refused>;

    /// The door's own rules of the facts it is about to write, behind the
    /// store's measure.
    fn room(&self, facts: &WorkerFacts, maker: &Maker) -> Result<(), Refused>;

    /// Hear a row as its write left it, ahead of the connect signal of a
    /// row whose checks are off.
    fn laid(&self, _: &WorkerRow) {}

    /// Hear a row's write at its place in the turn.
    fn told(&self, _: Wrote) {}
}

fn made_by(row: &WorkerRow, maker: &Maker) -> Result<(), Refused> {
    row.made_by(maker)
        .map_err(|not| Refused::NotMaker(not.to_string()))
}

/// The row an asker saw, where it still stands under the URL at the
/// revision the asker saw.
fn seen_row(store: &MemoryStore, url: &str, seen: Standing) -> Result<WorkerRow, Refused> {
    let row = store.by_key(url).ok_or(Refused::Absent)?;
    if (row.id, row.revision) != (seen.row, seen.revision) {
        return Err(Refused::Moved);
    }
    Ok(row)
}

const RANKS: Refused = Refused::Past(Past {
    fact: "dp_rank",
    why: "a registration states every rank of one engine under its own URL",
});

/// The rows `built` states, handed to `write`: each worker's URL, and its
/// facts over the row that stands there, by the write's verb ([`stated`]).
pub(crate) fn stating<T>(
    built: &[Arc<dyn Worker>],
    over: Option<Over<'_>>,
    write: impl FnOnce(&[Stated<'_>]) -> T,
) -> T {
    let states: Vec<_> = built
        .iter()
        .map(|worker| move |standing: Option<&WorkerFacts>| stated(worker.as_ref(), standing, over))
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
pub(crate) type Ready = (WorkerFacts, Option<(WorkerId, Laid)>);

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
    store: &MemoryStore,
    expect: Expect,
    url: &str,
) -> Result<Option<WorkerRow>, Refused> {
    Ok(match expect {
        Expect::Seen(seen) => Some(seen_row(store, url, seen)?),
        Expect::Nothing | Expect::Anything => store.by_key(url),
    })
}

/// Check one stated row by its own rule, over the row its write expects to
/// stand under its URL ([`standing_under`]). Nothing is written. The row's
/// event signal is stated here, over the facts its callback states
/// ([`asks_events`]).
fn ready_row(
    turn: &impl Turn,
    maker: &Maker,
    expect: Expect,
    Stated { url, state }: &Stated<'_>,
    standing: Option<WorkerRow>,
) -> Result<Ready, Refused> {
    let store = turn.store();
    if let Some(row) = &standing {
        // An add of several rows asks no maker here. What it meets is
        // judged over its rows together, and its refusal is the same
        // whoever made the row ([`same_add`]).
        if expect != Expect::Nothing {
            made_by(row, maker)?;
        }
        // A draining row is leaving. Laid over, it would take the
        // URL's next life and its removal would leave the worker: the
        // removal wins over a replacement.
        if matches!(expect, Expect::Seen(_)) && row.status == WorkerStatus::Draining {
            return Err(Refused::Draining);
        }
    }
    let mut facts = state(standing.as_ref().map(|row| &row.facts));
    if facts.key != *url {
        return Err(Refused::Past(Past {
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
    facts.bounded(maker).map_err(Refused::Past)?;
    turn.room(&facts, maker)?;
    let laid = standing.map(|row| (row.id, row.laid(&facts)));
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
fn same_add(store: &MemoryStore, maker: &Maker, ready: &[Ready]) -> Result<(), Refused> {
    let mut writes = false;
    for (_, laid) in ready {
        let Some((id, laid)) = laid else {
            writes = true;
            continue;
        };
        let own = store.get(*id).is_some_and(|row| row.made_by(maker).is_ok());
        if !own || *laid != Laid::Unmoved {
            return Err(Refused::Stands);
        }
    }
    if writes {
        Ok(())
    } else {
        Err(Refused::Stands)
    }
}

/// Lay one row by its own rule alone: the write of [`Membership::found`].
/// No rule of the engine's other rows is held and no row leaves. One row
/// has no rest to lay, so an add of it is refused by any row that stands,
/// before a fact is stated and whoever made the row.
pub(crate) fn lay_one(
    turn: &impl Turn,
    maker: &Maker,
    expect: Expect,
    url: &str,
    state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
) -> Result<Wrote, Refused> {
    let standing = standing_under(turn.store(), expect, url)?;
    if expect == Expect::Nothing && standing.is_some() {
        return Err(Refused::Stands);
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
    maker: &Maker,
    expect: Expect,
    rows: &[Stated<'_>],
) -> Result<Vec<Wrote>, Refused> {
    if rows.is_empty() {
        return Err(Refused::Past(Past {
            fact: "rows",
            why: "a registration states at least one row",
        }));
    }
    // A guarded write states the one row its asker saw and changes it where
    // it stands, by that row's own rule: the group's rule is a
    // registration's.
    if matches!(expect, Expect::Seen(_)) {
        let [row] = rows else {
            return Err(Refused::Past(Past {
                fact: "rows",
                why: "a replacement states the one row its asker saw",
            }));
        };
        return lay_one(turn, maker, expect, row.url, row.state).map(|wrote| vec![wrote]);
    }
    let store = turn.store();
    let mut standing = Vec::with_capacity(rows.len());
    for row in rows {
        standing.push(standing_under(store, expect, row.url)?);
    }
    // An add with no row left to write is refused before a fact is stated,
    // whoever made the rows that stand.
    if expect == Expect::Nothing && standing.iter().all(Option::is_some) {
        return Err(Refused::Stands);
    }
    let mut ready = Vec::with_capacity(rows.len());
    for (row, standing) in rows.iter().zip(standing) {
        ready.push(ready_row(turn, maker, expect, row, standing)?);
    }
    lay_ready(turn, maker, expect == Expect::Nothing, ready)
}

/// The group's rule over the rows one registration states, then the rows
/// that leave and the writes. A door that states its rows by a rule of its
/// own enters here, behind that rule: `adds` says the registration is an
/// add, which meets the rows that stand by the add's rule ([`same_add`])
/// and retires none. Such a door hands an add's standing row in with what
/// laying it would make, and asks its maker of no row.
pub(crate) fn lay_ready(
    turn: &impl Turn,
    maker: &Maker,
    adds: bool,
    ready: Vec<Ready>,
) -> Result<Vec<Wrote>, Refused> {
    let store = turn.store();
    let Some((first, _)) = ready.first() else {
        return Err(Refused::Past(Past {
            fact: "rows",
            why: "a registration states at least one row",
        }));
    };
    if adds {
        same_add(store, maker, &ready)?;
    }
    // The engine is the one the first row states. The others are held to
    // it below, and the store is asked by it.
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
    let mut left: Vec<_> = store
        .of_engine(engine)
        .into_iter()
        .filter(|row| ready.iter().all(|(facts, _)| facts.key != row.facts.key))
        .collect();
    left.sort_by_key(|row| row.facts.dp_rank);
    for row in &left {
        // An add retires no row: the engine stands in another form.
        if adds {
            return Err(Refused::Stands);
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
        let wrote = changed(store.remove(row.id, row.revision), Wrote::Removed)?;
        turn.told(wrote);
        answer.push(wrote);
    }
    Ok(answer)
}

fn write_row(
    turn: &impl Turn,
    maker: &Maker,
    mut facts: WorkerFacts,
    laid: Option<(WorkerId, Laid)>,
) -> Result<Wrote, Refused> {
    match laid {
        None | Some((_, Laid::Another)) => facts.incarnation = turn.life()?,
        Some((id, Laid::Unmoved)) => {
            let wrote = Wrote::Unmoved(id);
            turn.told(wrote);
            return Ok(wrote);
        }
        Some((_, Laid::InPlace)) => {}
    }
    let store = turn.store();
    let probes = facts.health.probes;
    let change = store.upsert_by(facts, maker.clone());
    let wrote = match change {
        MembershipChange::Registered { id } => Wrote::Registered(id),
        MembershipChange::Relabelled { id } | MembershipChange::Resized { id } => {
            Wrote::InPlace(id)
        }
        MembershipChange::Replaced { old, new } => Wrote::Another { old, new },
        MembershipChange::Removed { .. } | MembershipChange::StatusChanged { .. } => {
            return Err(Refused::Moved);
        }
    };
    // The row as its write left it: a connect signal moves it next.
    if let Some(row) = store.get(change.id()) {
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
    wrote: fn(WorkerId) -> Wrote,
) -> Result<Wrote, Refused> {
    change
        .map(|change| wrote(change.id()))
        .ok_or(Refused::Moved)
}

fn removal_row(
    store: &MemoryStore,
    by: Remover<'_>,
    url: &str,
    seen: Standing,
) -> Result<WorkerRow, Refused> {
    let row = seen_row(store, url, seen)?;
    match by {
        Remover::Maker(maker) => made_by(&row, maker)?,
        Remover::Provider(name) => match &row.maker {
            Maker::Provider(provider) if provider.provider() == name => {}
            Maker::Provider(provider) => {
                return Err(Refused::NotMaker(
                    NotMaker::Provider {
                        provider: provider.provider(),
                    }
                    .to_string(),
                ));
            }
            Maker::Control => return Err(Refused::NotMaker(NotMaker::Control.to_string())),
        },
        Remover::Health(scope) => {
            let allowed = match scope {
                UnhealthyRemovalScope::Disabled => false,
                UnhealthyRemovalScope::DiscoveryOwned => matches!(row.maker, Maker::Provider(_)),
                UnhealthyRemovalScope::AllLocal => true,
            };
            if !allowed {
                return Err(Refused::OutOfScope);
            }
        }
    }
    Ok(row)
}

/// Mark the guarded row as draining while the door holds its turn.
pub(crate) fn drain(
    store: &MemoryStore,
    by: Remover<'_>,
    url: &str,
    seen: Standing,
) -> Result<Wrote, Refused> {
    let row = removal_row(store, by, url, seen)?;
    if row.status == WorkerStatus::Draining {
        return Ok(Wrote::Unmoved(row.id));
    }
    changed(
        store.set_status(row.id, WorkerStatus::Draining, row.revision),
        Wrote::Draining,
    )
}

/// Remove the guarded row while the door holds its turn.
pub(crate) fn take(
    store: &MemoryStore,
    by: Remover<'_>,
    url: &str,
    seen: Standing,
) -> Result<Wrote, Refused> {
    let row = removal_row(store, by, url, seen)?;
    changed(store.remove(row.id, row.revision), Wrote::Removed)
}

/// Write what a refresh's rebuilt worker states over the guarded row while
/// the door holds its turn: its labels, and the pairing descriptor they
/// derive. The row's other facts are the test of the refresh's scope and
/// are not written: the door states a row's event signal at its
/// registration, and a refresh leaves it as it stands.
pub(crate) fn refresh(
    turn: &impl Turn,
    seen: Standing,
    built: &dyn Worker,
) -> Result<Wrote, Refused> {
    refresh_stated(turn, seen, built.url(), &|standing| {
        facts(built, Some(standing))
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
    seen: Standing,
    url: &str,
    state: &dyn Fn(&WorkerFacts) -> WorkerFacts,
) -> Result<Wrote, Refused> {
    let store = turn.store();
    let row = seen_row(store, url, seen)?;
    // A draining row is leaving: its removal wins over a refresh, as over
    // a replacement.
    if row.status == WorkerStatus::Draining {
        return Err(Refused::Draining);
    }
    let mut stated = state(&row.facts);
    let labels = mem::replace(&mut stated.labels, row.facts.labels.clone());
    let mut pairing = mem::replace(&mut stated.pairing, row.facts.pairing.clone());
    if stated != row.facts {
        return Err(Refused::NotLabelsAlone);
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
        return Ok(Wrote::Unmoved(row.id));
    }
    stated.labels = labels;
    stated.pairing = pairing;
    stated.bounded(&row.maker).map_err(Refused::Past)?;
    turn.room(&stated, &row.maker)?;
    // The write, in place or by the patch, is a write of the turn: the term
    // is asked ahead of it.
    turn.check_term()?;
    if pairing_moved {
        return match store.upsert_by(stated, row.maker) {
            MembershipChange::Relabelled { id } | MembershipChange::Resized { id } => {
                Ok(Wrote::InPlace(id))
            }
            MembershipChange::Registered { .. }
            | MembershipChange::Replaced { .. }
            | MembershipChange::Removed { .. }
            | MembershipChange::StatusChanged { .. } => Err(Refused::Moved),
        };
    }
    let patch = WorkerPatch {
        labels: Some(stated.labels),
        ..WorkerPatch::default()
    };
    changed(store.patch(row.id, patch, row.revision), Wrote::InPlace)
}

/// The membership door over a store alone, for a host with no election or
/// log. A host shares one door for its store so its writes share a turn.
pub struct StoreDoor {
    store: Arc<MemoryStore>,
    base: u64,
    turn: Mutex<()>,
    told: Option<Box<dyn Fn() + Send + Sync>>,
}

impl StoreDoor {
    /// The one term such a host writes under.
    pub const TERM: u64 = 0;

    /// Open a door over the store, with a base for this run's lives.
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self {
            store,
            base: membership::base(),
            turn: Mutex::new(()),
            told: None,
        }
    }

    /// Call `told` once after a turn that changed any rows is released.
    #[must_use]
    pub fn telling(mut self, told: impl Fn() + Send + Sync + 'static) -> Self {
        self.told = Some(Box::new(told));
        self
    }

    fn call<T>(
        &self,
        write: impl FnOnce() -> Result<T, Refused>,
        changed: impl FnOnce(&T) -> bool,
    ) -> Result<T, Refused> {
        let turn = self.turn.lock();
        let answer = write();
        drop(turn);
        if answer.as_ref().is_ok_and(changed) {
            if let Some(told) = &self.told {
                told();
            }
        }
        answer
    }

    /// The registration's write over rows of stated facts, in the door's
    /// turn.
    fn rows(
        &self,
        maker: &Maker,
        expect: Expect,
        rows: &[Stated<'_>],
    ) -> Result<Vec<Wrote>, Refused> {
        self.call(
            || lay(self, maker, expect, rows),
            |answer| answer.iter().any(Wrote::changed),
        )
    }
}

impl Turn for StoreDoor {
    fn store(&self) -> &MemoryStore {
        &self.store
    }

    fn life(&self) -> Result<u64, Refused> {
        Ok(membership::life(Self::TERM, self.base, self.store.epoch()))
    }

    fn room(&self, _: &WorkerFacts, _: &Maker) -> Result<(), Refused> {
        Ok(())
    }
}

impl Membership for StoreDoor {
    fn found(
        &self,
        maker: &Maker,
        expect: Expect,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused> {
        self.call(|| lay_one(self, maker, expect, url, state), Wrote::changed)
    }

    // This door has no holder. The rows take the facts alone, by the
    // write's verb.
    fn found_rows(
        &self,
        maker: &Maker,
        expect: Expect,
        built: &[Arc<dyn Worker>],
        over: Option<Over<'_>>,
    ) -> Result<Vec<Wrote>, Refused> {
        stating(built, over, |rows| self.rows(maker, expect, rows))
    }

    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.call(|| drain(&self.store, by, url, seen), Wrote::changed)
    }

    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.call(|| take(&self.store, by, url, seen), Wrote::changed)
    }

    // This door has no holder. The row alone takes the refresh.
    fn refreshed(
        &self,
        seen: Standing,
        built: &Arc<dyn Worker>,
        _: &Arc<dyn Worker>,
    ) -> Result<Wrote, Refused> {
        self.call(|| refresh(self, seen, built.as_ref()), Wrote::changed)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc, Barrier, Weak,
        },
        thread,
        time::{Duration, Instant},
    };

    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        model_card::ModelCard,
        settings::UnhealthyRemovalScope,
        worker::{HealthCheckConfig, WorkerType},
    };

    use super::{
        changed, lay, lay_one, refresh_stated, Expect, Membership, Refused, Remover, Standing,
        Stated, StoreDoor, Turn, Wrote,
    };
    use crate::{
        demands::{DemandFacts, Demands},
        door::{
            tests::{request, GATEWAY, MODEL},
            Door,
        },
        spec::{AppliedSpec, PolicySpec},
        store::{
            index::indexed_under,
            membership::{self, Past},
            pairing::PairingDescriptor,
            Maker, MembershipChange, MembershipTail, MemoryStore, Signals, WorkerFacts, WorkerRow,
            WorkerStatus, WorkersStore,
        },
        view::{Role, WireModes, WorkerId},
    };

    const URL: &str = "http://worker:8000";

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
    pub(crate) trait Rows: Membership {
        fn rows(
            &self,
            maker: &Maker,
            expect: Expect,
            rows: &[Stated<'_>],
        ) -> Result<Vec<Wrote>, Refused>;
    }

    impl Rows for StoreDoor {
        fn rows(
            &self,
            maker: &Maker,
            expect: Expect,
            rows: &[Stated<'_>],
        ) -> Result<Vec<Wrote>, Refused> {
            StoreDoor::rows(self, maker, expect, rows)
        }
    }

    pub(crate) trait FindOne: Rows {
        /// One row, upserted.
        fn found_one(
            &self,
            maker: &Maker,
            url: &str,
            state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
        ) -> Result<Wrote, Refused> {
            self.found_one_over(maker, Expect::Anything, url, state)
        }

        /// One row, laid over what its registration expects.
        fn found_one_over(
            &self,
            maker: &Maker,
            expect: Expect,
            url: &str,
            state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
        ) -> Result<Wrote, Refused> {
            let answer = self.rows(maker, expect, &[Stated { url, state }])?;
            assert_eq!(answer.len(), 1);
            Ok(answer[0])
        }
    }

    impl<T: Rows + ?Sized> FindOne for T {}

    fn standing(row: &WorkerRow) -> Standing {
        Standing {
            row: row.id,
            revision: row.revision,
        }
    }

    const ABSENT: Standing = Standing {
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
        maker: &Maker,
        facts: &[WorkerFacts],
    ) -> Result<Vec<Wrote>, Refused> {
        find_rows_over(door, maker, Expect::Anything, facts)
    }

    fn find_rows_over(
        door: &dyn Rows,
        maker: &Maker,
        expect: Expect,
        facts: &[WorkerFacts],
    ) -> Result<Vec<Wrote>, Refused> {
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
        let answer = find_rows(&*door, &Maker::Control, &facts).map_err(|_| "registered")?;
        assert_eq!(answer.len(), 4);
        let mut lives = Vec::new();
        for (index, stated) in facts.iter().enumerate() {
            let row = store.by_key(&stated.key).ok_or("the registered rank")?;
            assert_eq!(answer[index], Wrote::Registered(row.id));
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
            find_rows(&*door, &Maker::Control, &ranks),
            Err(Refused::Past(Past {
                fact: "labels",
                why: "a label's value is at most 4096 bytes",
            }))
        );
        assert_eq!(store.epoch(), epoch);
        assert!(store.snapshot().1.is_empty());
    }

    #[test]
    fn a_worker_that_comes_back_narrower_loses_the_ranks_it_no_longer_states() {
        check_narrower(local_door).expect("four ranks, then two");
    }

    fn check_narrower(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Maker::Control;
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
                Wrote::InPlace(old[0].id),
                Wrote::InPlace(old[1].id),
                Wrote::Removed(old[2].id),
                Wrote::Removed(old[3].id),
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
        let maker = Maker::Control;
        let four = ranked(4);
        find_rows(&*door, &maker, &four).map_err(|_| "registered four ranks")?;
        let old: Vec<_> = four
            .iter()
            .map(|facts| store.by_key(&facts.key).ok_or("the old rank"))
            .collect::<Result<_, _>>()?;
        for row in &old[..2] {
            door.gone(Remover::Maker(&maker), &row.facts.key, standing(row))
                .map_err(|_| "a low rank removed")?;
        }
        let epoch = store.epoch();
        let two = ranked(2);
        assert_eq!(
            find_rows_over(&*door, &maker, Expect::Nothing, &two),
            Err(Refused::Stands)
        );
        assert_eq!(store.epoch(), epoch);

        let answer = find_rows(&*door, &maker, &two).map_err(|_| "two ranks registered")?;
        assert!(matches!(
            answer[..2],
            [Wrote::Registered(_), Wrote::Registered(_)]
        ));
        assert_eq!(
            answer[2..],
            [Wrote::Removed(old[2].id), Wrote::Removed(old[3].id)]
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
        let maker = Maker::Control;
        find_rows(&*door, &maker, &[stated()]).map_err(|_| "registered the plain row")?;
        let plain = store.by_key(URL).ok_or("the plain row")?;
        let four = ranked(4);
        let answer = find_rows(&*door, &maker, &four).map_err(|_| "registered four ranks")?;
        assert_eq!(answer.len(), 5);
        assert_eq!(answer[4], Wrote::Removed(plain.id));
        assert_eq!(store.by_key(URL), None);
        let ids: Vec<_> = four
            .iter()
            .map(|facts| store.by_key(&facts.key).map(|row| row.id).ok_or("the rank"))
            .collect::<Result<_, _>>()?;
        let answer = find_rows(&*door, &maker, &[stated()]).map_err(|_| "plain again")?;
        let fresh = store.by_key(URL).ok_or("the new plain row")?;
        let expected: Vec<_> = std::iter::once(Wrote::Registered(fresh.id))
            .chain(ids.into_iter().map(Wrote::Removed))
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
        let maker = Maker::Control;
        let whole = WorkerFacts {
            dp_size: 4,
            ..stated()
        };
        let answer =
            find_rows(&*door, &maker, std::slice::from_ref(&whole)).map_err(|_| "registered")?;
        let row = store.by_key(URL).ok_or("the engine's one row")?;
        assert_eq!(answer, [Wrote::Registered(row.id)]);
        assert_eq!((row.facts.dp_rank, row.facts.dp_size), (0, 4));
        // The same facts again write nothing, and another width is laid in
        // place.
        assert_eq!(
            find_rows(&*door, &maker, std::slice::from_ref(&whole)),
            Ok(vec![Wrote::Unmoved(row.id)])
        );
        let wider = WorkerFacts {
            dp_size: 8,
            ..whole
        };
        assert_eq!(
            find_rows(&*door, &maker, &[wider]),
            Ok(vec![Wrote::InPlace(row.id)])
        );
        assert_eq!(store.by_key(URL).map(|row| row.facts.dp_size), Some(8));
        // Four ranks retire it, as they retire a row of one.
        let answer = find_rows(&*door, &maker, &ranked(4)).map_err(|_| "four ranks")?;
        assert_eq!(answer.len(), 5);
        assert_eq!(answer[4], Wrote::Removed(row.id));
        assert_eq!(store.by_key(URL), None);
        Ok(())
    }

    #[test]
    fn another_makers_row_of_the_engine_refuses_the_registration_whole() {
        check_other_maker(local_door).expect("the control row stands alone");
    }

    fn check_other_maker(open: Open) -> Result<(), &'static str> {
        let (store, door) = open();
        let provider = Maker::provider("pool", "private/key").map_err(|_| "the provider")?;
        find_rows(&*door, &Maker::Control, &[stated()]).map_err(|_| "the control registration")?;
        let before = store.snapshot();
        assert_eq!(
            find_rows(&*door, &provider, &ranked(4)),
            Err(Refused::NotMaker(
                "the row was made by a control call: its maker alone changes it".to_owned()
            ))
        );
        assert_eq!(store.snapshot(), before);
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
                    find_rows(&*door, &Maker::Control, &ranks),
                    Err(Refused::Past(Past {
                        fact: "dp_rank",
                        ..
                    }))
                ),
                "{ranks:?}"
            );
            assert_eq!(store.snapshot(), (0, Vec::new()));
        }
        // Facts under another key than their row's URL are refused by the
        // key's name, ahead of the group's shape.
        let row = Stated {
            url: "http://other:8000",
            state: &|_| stated(),
        };
        assert!(matches!(
            door.rows(&Maker::Control, Expect::Anything, &[row]),
            Err(Refused::Past(Past { fact: "key", .. }))
        ));
        assert_eq!(store.snapshot(), (0, Vec::new()));
    }

    #[test]
    fn the_same_ranks_stated_again_write_nothing() {
        let heard = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&heard);
        let store = Arc::new(MemoryStore::default());
        let door = StoreDoor::new(Arc::clone(&store)).telling(move || {
            count.fetch_add(1, Ordering::Relaxed);
        });
        check_unchanged(&store, &door).expect("the same four ranks stand");
        assert_eq!(heard.load(Ordering::Relaxed), 1);
    }

    fn check_unchanged(store: &MemoryStore, door: &dyn Rows) -> Result<(), &'static str> {
        let maker = Maker::Control;
        let four = ranked(4);
        let first = find_rows(door, &maker, &four).map_err(|_| "registered four ranks")?;
        let epoch = store.epoch();
        let expected = first
            .into_iter()
            .map(|wrote| match wrote {
                Wrote::Registered(id) => Ok(Wrote::Unmoved(id)),
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
            door.found(&Maker::Control, expect, &facts.key, &|_| facts.clone())
        };
        let low = one(&ranks[0], Expect::Nothing);
        assert!(matches!(low, Ok(Wrote::Registered(_))), "{low:?}");
        // An add of the row again is refused before a fact is stated,
        // whoever made the row: one row has no rest to lay.
        let unstated = |_: Option<&WorkerFacts>| -> WorkerFacts { panic!("no fact is stated") };
        let pool = Maker::provider("pool", "private/key").expect("a maker");
        for maker in [&Maker::Control, &pool] {
            assert_eq!(
                door.found(maker, Expect::Nothing, &ranks[0].key, &unstated),
                Err(Refused::Stands)
            );
        }
        // The rows write holds the engine to its width where one rank stands.
        let epoch = store.epoch();
        assert!(matches!(
            find_rows(&*door, &Maker::Control, &ranks[1..]),
            Err(Refused::Past(Past {
                fact: "dp_rank",
                ..
            }))
        ));
        assert_eq!(store.epoch(), epoch);
        let high = one(&ranks[1], Expect::Nothing);
        assert!(matches!(high, Ok(Wrote::Registered(_))), "{high:?}");

        // One rank's row is changed where it stands, and the other stays.
        let other = store.by_key(&ranks[0].key).expect("the low rank");
        let seen = standing(&store.by_key(&ranks[1].key).expect("the high rank"));
        let resized = WorkerFacts {
            max_running: Some(8),
            ..ranks[1].clone()
        };
        let wrote = one(&resized, Expect::Seen(seen));
        assert!(
            matches!(wrote, Ok(Wrote::InPlace(_) | Wrote::Another { .. })),
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
            .found(&Maker::Control, Expect::Nothing, &three[0].key, &|_| {
                three[0].clone()
            })
            .map_err(|_| "the first rank alone")?;
        let add = || find_rows_over(&*door, &Maker::Control, Expect::Nothing, &three);
        let wrote = add().map_err(|_| "the same add")?;
        let [Wrote::Unmoved(stood), Wrote::Registered(_), Wrote::Registered(_)] = wrote.as_slice()
        else {
            return Err("one rank unmoved and two registered");
        };
        assert_eq!(first, Wrote::Registered(*stood));
        assert_eq!(store.snapshot().1.len(), 3);
        let epoch = store.epoch();
        assert_eq!(add(), Err(Refused::Stands));
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
        door.found(&Maker::Control, Expect::Nothing, &three[2].key, &|_| {
            three[2].clone()
        })
        .map_err(|_| "the last rank alone")?;
        let before = store.snapshot();
        let refused = |maker: &Maker, facts: &[WorkerFacts]| {
            let add = find_rows_over(&*door, maker, Expect::Nothing, facts);
            assert_eq!(add, Err(Refused::Stands));
            assert_eq!(store.snapshot(), before);
        };

        // The add states a label the standing rank does not.
        let mut labelled = three.clone();
        for facts in &mut labelled {
            facts.labels.insert("tier".to_owned(), "gold".to_owned());
        }
        refused(&Maker::Control, &labelled);
        // Another maker states the same rows.
        let provider = Maker::provider("pool", "private/key").map_err(|_| "the provider")?;
        refused(&provider, &three);
        // The standing rank is past the width the add states.
        refused(&Maker::Control, &ranked(2));
        Ok(())
    }

    #[test]
    fn no_row_stated_is_refused() {
        check_empty(local_door);
    }

    fn check_empty(open: Open) {
        let (store, door) = open();
        assert!(matches!(
            door.rows(&Maker::Control, Expect::Anything, &[]),
            Err(Refused::Past(Past { fact: "rows", .. }))
        ));
        assert_eq!(store.snapshot(), (0, Vec::new()));
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
            door.found_one_over(&Maker::Control, Expect::Nothing, URL, &|_| found.clone())
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
            door.found_one(&Maker::Control, URL, &|_| found.clone())
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
        let control = Maker::Control;
        let pool = Maker::provider("pool", "private/key").expect("a maker");
        let unstated = |_: Option<&WorkerFacts>| -> WorkerFacts { panic!("no fact is stated") };

        let added = door.found_one_over(&control, Expect::Nothing, URL, &|_| stated());
        let row = store.by_key(URL).expect("the added row");
        assert_eq!(added, Ok(Wrote::Registered(row.id)));
        let epoch = store.epoch();
        for maker in [&control, &pool] {
            assert_eq!(
                door.found_one_over(maker, Expect::Nothing, URL, &unstated),
                Err(Refused::Stands)
            );
        }

        let seen = Expect::Seen(standing(&row));
        assert_eq!(
            door.found_one_over(&control, seen, URL, &|_| stated()),
            Ok(Wrote::Unmoved(row.id))
        );
        assert_eq!(
            door.found_one_over(&pool, seen, URL, &unstated),
            Err(Refused::NotMaker(
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
            Ok(Wrote::InPlace(row.id))
        );
        assert_eq!(
            door.found_one_over(&control, seen, URL, &unstated),
            Err(Refused::Moved)
        );

        // Other models are the URL's next life, as under an upsert.
        let moved = store.by_key(URL).expect("the relabelled row");
        let other = |_: Option<&WorkerFacts>| WorkerFacts {
            models: vec!["other".to_owned()],
            ..stated()
        };
        let wrote = door.found_one_over(&control, Expect::Seen(standing(&moved)), URL, &other);
        let next = store.by_key(URL).expect("the next life");
        assert_eq!(
            wrote,
            Ok(Wrote::Another {
                old: row.id,
                new: next.id
            })
        );

        // A draining row is leaving: its removal wins over a replacement,
        // which is told so. Another maker is told the row is not its own.
        let seen = standing(&next);
        assert_eq!(
            door.drained(Remover::Maker(&control), URL, seen),
            Ok(Wrote::Draining(next.id))
        );
        assert_eq!(
            door.found_one_over(&control, Expect::Seen(seen), URL, &unstated),
            Err(Refused::Draining)
        );
        assert!(matches!(
            door.found_one_over(&pool, Expect::Seen(seen), URL, &unstated),
            Err(Refused::NotMaker(_))
        ));
        assert_eq!(
            store.by_key(URL).map(|row| (row.id, row.status)),
            Some((next.id, WorkerStatus::Draining))
        );

        // The row its asker saw left.
        assert_eq!(
            door.gone(Remover::Maker(&control), URL, seen),
            Ok(Wrote::Removed(next.id))
        );
        assert_eq!(
            door.found_one_over(&control, Expect::Seen(seen), URL, &unstated),
            Err(Refused::Absent)
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
            find_rows_over(&*door, &control, Expect::Nothing, &two),
            Err(Refused::Stands)
        );
        assert_eq!(store.epoch(), epoch);

        // Two ranks are added in one write. The engine's row in its other
        // form refuses an add, and a replacement states one row.
        let (store, door) = open();
        let add = |facts: &[WorkerFacts]| find_rows_over(&*door, &control, Expect::Nothing, facts);
        let ranks = add(&two).expect("two ranks added");
        assert!(matches!(
            ranks[..],
            [Wrote::Registered(_), Wrote::Registered(_)]
        ));
        let epoch = store.epoch();
        assert_eq!(add(&[stated()]), Err(Refused::Stands));
        let rank = store.by_key(&two[0].key).expect("rank zero");
        assert!(matches!(
            find_rows_over(&*door, &control, Expect::Seen(standing(&rank)), &two),
            Err(Refused::Past(Past { fact: "rows", .. }))
        ));
        assert_eq!(store.epoch(), epoch);
        find_rows(&*door, &control, &[stated()]).expect("plain again");
        let epoch = store.epoch();
        assert_eq!(add(&two), Err(Refused::Stands));
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
        let maker = Maker::Control;
        let by = Remover::Maker(&maker);
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
        let wrong_revision = Standing {
            revision: replacement.revision + 1,
            ..standing(&replacement)
        };
        for seen in [wrong_revision, standing(&first)] {
            assert_eq!(door.drained(by, URL, seen), Err(Refused::Moved));
            assert_eq!(door.gone(by, URL, seen), Err(Refused::Moved));
            assert_eq!(
                door.gone(Remover::Provider("other"), URL, seen),
                Err(Refused::Moved)
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
        let own = Maker::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        let another =
            Maker::provider("other", "private/key").map_err(|_| "another source's maker")?;
        for (maker, refusal) in [
            (own, None),
            (
                Maker::Control,
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
                let expected = Err(Refused::NotMaker(refusal.to_owned()));
                assert_eq!(door.drained(by, URL, seen), expected);
                assert_eq!(door.gone(by, URL, seen), expected);
                assert_eq!(store.by_key(URL), Some(row));
            } else {
                assert_eq!(door.drained(by, URL, seen), Ok(Wrote::Draining(row.id)));
                assert_eq!(door.gone(by, URL, seen), Ok(Wrote::Removed(row.id)));
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
        let provider = Maker::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        for (scope, control, discovered) in [
            (UnhealthyRemovalScope::Disabled, false, false),
            (UnhealthyRemovalScope::DiscoveryOwned, false, true),
            (UnhealthyRemovalScope::AllLocal, true, true),
        ] {
            for (maker, allowed) in [(&Maker::Control, control), (&provider, discovered)] {
                let (store, door) = open();
                assert!(door.found_one(maker, URL, &|_| stated()).is_ok());
                let row = store.by_key(URL).ok_or("the registered row")?;
                let by = Remover::Health(scope);
                let seen = standing(&row);
                if allowed {
                    assert_eq!(door.drained(by, URL, seen), Ok(Wrote::Draining(row.id)));
                    assert_eq!(door.gone(by, URL, seen), Ok(Wrote::Removed(row.id)));
                    assert_eq!(store.by_key(URL), None);
                } else {
                    assert_eq!(door.drained(by, URL, seen), Err(Refused::OutOfScope));
                    assert_eq!(door.gone(by, URL, seen), Err(Refused::OutOfScope));
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
        let maker = Maker::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        assert!(door.found_one(&maker, URL, &|_| stated()).is_ok());
        let row = store.by_key(URL).ok_or("the registered row")?;
        assert_eq!(
            door.drained(Remover::Provider("pool"), URL, standing(&row)),
            Ok(Wrote::Draining(row.id))
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
        let provider = Maker::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        let worker = |built: BasicWorkerBuilder| -> Arc<dyn Worker> { Arc::new(built.build()) };
        for maker in [&Maker::Control, &provider] {
            let (store, door) = open();
            let first = worker(built("gold"));
            assert!(door
                .found_built(maker, Expect::Nothing, &first, None)
                .is_ok());
            let row = store.by_key(URL).ok_or("the registered row")?;
            let seen = standing(&row);

            let equal = door.refreshed(seen, &worker(built("gold")), &first);
            assert_eq!(equal, Ok(Wrote::Unmoved(row.id)));

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
                assert_eq!(refused, Err(Refused::NotLabelsAlone));
            }
            let wide = "x".repeat(WorkerFacts::LABEL_VALUE_BYTES + 1);
            let refused = door.refreshed(seen, &worker(built(&wide)), &first);
            assert!(
                matches!(refused, Err(Refused::Past(Past { fact: "labels", .. }))),
                "{refused:?}"
            );
            assert_eq!(store.by_key(URL).as_ref(), Some(&row));

            let wrote = door.refreshed(seen, &worker(built("silver")), &first);
            assert_eq!(wrote, Ok(Wrote::InPlace(row.id)));
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
            assert_eq!(door.refreshed(seen, &late, &first), Err(Refused::Moved));
            let seen = standing(&moved);
            let by = Remover::Maker(maker);
            assert_eq!(door.drained(by, URL, seen), Ok(Wrote::Draining(moved.id)));
            assert_eq!(door.refreshed(seen, &late, &first), Err(Refused::Draining));
            assert_eq!(door.gone(by, URL, seen), Ok(Wrote::Removed(moved.id)));
            assert_eq!(door.refreshed(seen, &late, &first), Err(Refused::Absent));
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
        let registered = door.found_built(&Maker::Control, Expect::Nothing, &first, None);
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
        assert_eq!(wrote, Ok(Wrote::InPlace(row.id)));
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
        assert_eq!(equal, Ok(Wrote::Unmoved(row.id)));
        assert_eq!(store.by_key(URL).as_ref(), Some(&moved));

        // The descriptor moves by itself, then is withdrawn: by none, and
        // by one that reports nothing, which the row holds as none.
        let empty = || Some(Arc::new(PairingDescriptor::default()));
        let mut bare = moved.clone();
        for withdrawn in [None, version(" ")] {
            let wrote = refresh(&bare, version("3"), "silver");
            assert_eq!(wrote, Ok(Wrote::InPlace(row.id)));
            let again = store.by_key(URL).expect("the row of a moved descriptor");
            assert_eq!(again.facts.pairing, version("3"));
            assert_eq!(again.facts.labels, moved.facts.labels);
            let wrote = refresh(&again, withdrawn, "silver");
            assert_eq!(wrote, Ok(Wrote::InPlace(row.id)));
            bare = store
                .by_key(URL)
                .expect("the row of a withdrawn descriptor");
            assert_eq!(bare.facts.pairing, None);
        }

        // Over a row with none, one that reports nothing writes nothing.
        for nothing in [empty(), version(" ")] {
            let unmoved = refresh(&bare, nothing, "silver");
            assert_eq!(unmoved, Ok(Wrote::Unmoved(row.id)));
            assert_eq!(store.by_key(URL).as_ref(), Some(&bare));
        }

        // Another fact beside a moved descriptor is outside the scope.
        let stray = refresh_stated(&door, standing(&bare), URL, &|standing| {
            let mut facts = standing.clone();
            facts.pairing = version("4");
            facts.priority += 1;
            facts
        });
        assert_eq!(stray, Err(Refused::NotLabelsAlone));
        assert_eq!(store.by_key(URL).as_ref(), Some(&bare));

        // A row may hold one that reports nothing, as a registration states
        // it. It reads as a row with none.
        let laid = door.found(&Maker::Control, Expect::Anything, URL, &|standing| {
            let mut facts = standing.expect("the standing row").clone();
            facts.pairing = empty();
            facts
        });
        assert_eq!(laid, Ok(Wrote::InPlace(row.id)));
        let held = store.by_key(URL).expect("the row of the registration");
        assert_eq!(held.facts.pairing, empty());
        for nothing in [None, empty()] {
            let unmoved = refresh(&held, nothing, "silver");
            assert_eq!(unmoved, Ok(Wrote::Unmoved(row.id)));
            assert_eq!(store.by_key(URL).as_ref(), Some(&held));
        }
    }

    #[test]
    fn a_found_worker_is_registered_under_its_maker() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let wrote = door.found_one(&maker, URL, &|standing| {
            assert!(standing.is_none());
            stated()
        });
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(wrote, Ok(Wrote::Registered(row.id)));
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
    fn equal_facts_write_nothing_and_tell_nobody() {
        let store = Arc::new(MemoryStore::default());
        let count = Arc::new(AtomicUsize::new(0));
        let heard = Arc::clone(&count);
        let door = StoreDoor::new(Arc::clone(&store)).telling(move || {
            heard.fetch_add(1, Ordering::Relaxed);
        });
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(
            door.found_one(&maker, URL, &|_| stated()),
            Ok(Wrote::Unmoved(row.id))
        );
        let seen = standing(&row);
        assert_eq!(store.by_key(URL), Some(row));
        assert_eq!(store.epoch(), 1);
        assert_eq!(count.load(Ordering::Relaxed), 1);

        assert!(matches!(
            door.gone(Remover::Maker(&Maker::Control), URL, seen),
            Err(Refused::NotMaker(_))
        ));
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_changed_label_is_written_in_place_and_a_changed_role_is_another_life() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        let old = store.by_key(URL).expect("the registered row");
        let labelled = WorkerFacts {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..stated()
        };
        assert_eq!(
            door.found_one(&maker, URL, &|_| labelled.clone()),
            Ok(Wrote::InPlace(old.id))
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
            Ok(Wrote::Another {
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
    /// answered: with no publication awaited, a retry that demands the old
    /// runtime is decided over nobody, and one that demands the new runtime
    /// is placed on the new life. The new life is handed the old one's
    /// status: the worker's checks are on, and none runs between the two.
    #[test]
    fn a_changed_demand_fact_is_another_life_and_the_old_one_takes_no_retry() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let stating = |runtime| WorkerFacts {
            models: vec![MODEL.to_owned()],
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
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 1;
        let applied = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(applied).expect("above the unbound start");
        store.publish_now();
        let policy = Door::new(Arc::clone(&store));
        let decide = |id: &[u8], runtime: u32| {
            let mut retry = request(id, 7);
            retry.constraints.wire_modes = WireModes::HTTP;
            retry.constraints.demands = Demands {
                runtime,
                ..Demands::default()
            };
            let decided = policy.decide(GATEWAY, &retry, Instant::now());
            decided.expect("a decision")
        };
        let placed = decide(b"before", 7);
        let target = placed.decision.pick().expect("the worker states runtime 7");
        assert_eq!(target.targets.first().map(|target| target.id), Some(old.id));

        let wrote = door.found_one(&maker, URL, &|_| stating(9));
        let row = store.by_key(URL).expect("the replaced row");
        assert_eq!(
            wrote,
            Ok(Wrote::Another {
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

        let refused = decide(b"after", 7);
        assert!(
            refused.decision.pick().is_none() && refused.lease.is_none(),
            "the worker states runtime 9: {refused:?}"
        );
        let moved = decide(b"moved", 9);
        let target = moved.decision.pick().expect("the worker states runtime 9");
        assert_eq!(target.targets.first().map(|target| target.id), Some(row.id));
    }

    /// A refresh that turns a worker's vision off starts another life, as
    /// one that changes a demand fact does: the door reads the bit as a
    /// hard demand. The row that took images is gone when the write is
    /// answered: with no publication awaited, a request that needs vision
    /// is decided over nobody, and one that needs none is placed on the
    /// new life, which is handed the old one's status.
    #[test]
    fn a_changed_vision_is_another_life_and_the_new_one_takes_no_image() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let seeing = |vision| WorkerFacts {
            models: vec![MODEL.to_owned()],
            vision,
            ..stated()
        };
        door.found(&maker, Expect::Anything, URL, &|_| seeing(true))
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
        store.apply(applied).expect("above the unbound start");
        store.publish_now();
        let policy = Door::new(Arc::clone(&store));
        let decide = |id: &[u8], vision: bool| {
            let mut asked = request(id, 7);
            asked.constraints.wire_modes = WireModes::HTTP;
            asked.constraints.demands = Demands {
                vision,
                ..Demands::default()
            };
            let decided = policy.decide(GATEWAY, &asked, Instant::now());
            decided.expect("a decision")
        };
        let placed = decide(b"before", true);
        let target = placed.decision.pick().expect("the worker takes images");
        assert_eq!(target.targets.first().map(|target| target.id), Some(old.id));

        let wrote = door.found(&maker, Expect::Anything, URL, &|_| seeing(false));
        let row = store.by_key(URL).expect("the replaced row");
        assert_eq!(
            wrote,
            Ok(Wrote::Another {
                old: old.id,
                new: row.id
            })
        );
        assert_eq!(store.get(old.id), None);
        assert!(!row.facts.vision);
        assert_eq!(row.status, WorkerStatus::Ready, "handed on by the store");

        let refused = decide(b"after", true);
        assert!(
            refused.decision.pick().is_none() && refused.lease.is_none(),
            "the worker takes no image: {refused:?}"
        );
        let plain = decide(b"plain", false);
        let target = plain.decision.pick().expect("the request needs no vision");
        assert_eq!(target.targets.first().map(|target| target.id), Some(row.id));
    }

    #[test]
    fn another_makers_row_refuses_every_write_and_stands() {
        let caller = Maker::provider("pool", "second").expect("a maker");
        for (maker, why) in [
            (
                Maker::Control,
                "the row was made by a control call: its maker alone changes it",
            ),
            (
                Maker::provider("pool", "first").expect("a maker"),
                "the row is provider pool's under another key: that key's caller changes it",
            ),
        ] {
            let (store, door) = door();
            door.found_one(&maker, URL, &|_| stated())
                .expect("registered");
            let row = store.by_key(URL).expect("the registered row");
            let expected = Err(Refused::NotMaker(why.to_owned()));
            assert_eq!(
                door.found_one(&caller, URL, &|_| panic!("only the maker states facts")),
                expected
            );
            assert_eq!(
                door.drained(Remover::Maker(&caller), URL, standing(&row)),
                expected
            );
            assert_eq!(
                door.gone(Remover::Maker(&caller), URL, standing(&row)),
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

            fn life(&self) -> Result<u64, Refused> {
                if self.1.replace(true) {
                    return Err(Refused::NotLeading("the term moved".to_owned()));
                }
                Ok(7)
            }

            fn room(&self, _: &WorkerFacts, _: &Maker) -> Result<(), Refused> {
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
            lay(&turn, &Maker::Control, Expect::Anything, &rows),
            Err(Refused::NotLeading("the term moved".to_owned()))
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

        fn check_term(&self) -> Result<(), Refused> {
            let Some(left) = self.1.get().checked_sub(1) else {
                return Err(Refused::NotLeading("the term moved".to_owned()));
            };
            self.1.set(left);
            Ok(())
        }

        fn life(&self) -> Result<u64, Refused> {
            Ok(7)
        }

        fn room(&self, _: &WorkerFacts, _: &Maker) -> Result<(), Refused> {
            Ok(())
        }
    }

    fn lay_under(turn: &TermFor<'_>, facts: &[WorkerFacts]) -> Result<Vec<Wrote>, Refused> {
        lay_as(turn, Expect::Anything, facts)
    }

    fn lay_as(
        turn: &TermFor<'_>,
        expect: Expect,
        facts: &[WorkerFacts],
    ) -> Result<Vec<Wrote>, Refused> {
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
        lay(turn, &Maker::Control, expect, &rows)
    }

    /// The term is asked ahead of every write, whatever the write makes of
    /// its row. A label laid in place and a rank that leaves are refused
    /// behind a lost lead, and what was written before it stands.
    #[test]
    fn a_term_lost_between_two_rows_stops_a_write_in_place_and_a_removal() {
        let (store, door) = door();
        let three = ranked(3);
        find_rows(&door, &Maker::Control, &three).expect("three ranks registered");
        let lost = Err(Refused::NotLeading("the term moved".to_owned()));
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
            lay_one(&turn, &Maker::Control, Expect::Anything, URL, &|_| stated()),
            Err(Refused::NotLeading("the term moved".to_owned()))
        );
        assert_eq!(store.snapshot(), (0, Vec::new()));
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
            lay_as(&turn, Expect::Nothing, &three)
        };
        assert_eq!(
            add(1),
            Err(Refused::NotLeading("the term moved".to_owned()))
        );
        assert_eq!(store.snapshot().1.len(), 1);
        let wrote = add(3);
        assert!(
            matches!(
                wrote.as_deref(),
                Ok([
                    Wrote::Unmoved(_),
                    Wrote::Registered(_),
                    Wrote::Registered(_)
                ])
            ),
            "{wrote:?}"
        );
        assert_eq!(store.snapshot().1.len(), 3);
    }

    #[test]
    fn a_fact_past_its_bound_writes_nothing() {
        struct Shut<'a>(&'a MemoryStore);

        impl Turn for Shut<'_> {
            fn store(&self) -> &MemoryStore {
                self.0
            }

            fn life(&self) -> Result<u64, Refused> {
                Ok(0)
            }

            fn room(&self, _: &WorkerFacts, _: &Maker) -> Result<(), Refused> {
                Err(Refused::Past(Past {
                    fact: "row",
                    why: "no room",
                }))
            }
        }

        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let invalid = WorkerFacts {
            models: vec!["m".repeat(257)],
            ..stated()
        };
        let no = Refused::Past(Past {
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
                Expect::Anything,
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
                Expect::Anything,
                &[Stated {
                    url: URL,
                    state: &|_| stated()
                }]
            ),
            Err(Refused::Past(Past {
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
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        let stale = row.revision - 1;
        assert_eq!(
            changed(
                store.set_status(row.id, WorkerStatus::Draining, stale),
                Wrote::Draining
            ),
            Err(Refused::Moved)
        );
        assert_eq!(
            changed(store.remove(row.id, stale), Wrote::Removed),
            Err(Refused::Moved)
        );
        assert_eq!(
            door.drained(Remover::Maker(&maker), URL, standing(&row)),
            Ok(Wrote::Draining(row.id))
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
            door.drained(Remover::Maker(&maker), URL, standing(&row)),
            Ok(Wrote::Unmoved(row.id))
        );
        assert_eq!(store.epoch(), 2);
        assert_eq!(
            door.gone(Remover::Maker(&maker), URL, standing(&row)),
            Ok(Wrote::Removed(row.id))
        );
        assert_eq!(store.epoch(), 3);
        assert_eq!(store.by_key(URL), None);
    }

    #[test]
    fn no_row_under_the_url_is_absent() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        assert_eq!(
            door.drained(Remover::Maker(&maker), URL, ABSENT),
            Err(Refused::Absent)
        );
        assert_eq!(
            door.gone(Remover::Maker(&maker), URL, ABSENT),
            Err(Refused::Absent)
        );
        assert_eq!(store.epoch(), 0);
    }

    #[test]
    fn the_listener_is_told_once_a_write_with_the_turn_released() {
        let store = Arc::new(MemoryStore::default());
        let count = Arc::new(AtomicUsize::new(0));
        let heard = Arc::clone(&count);
        let maker = Maker::provider("pool", "first").expect("a maker");
        let door = Arc::new_cyclic(|door: &Weak<StoreDoor>| {
            let door = door.clone();
            StoreDoor::new(Arc::clone(&store)).telling(move || {
                let door = door.upgrade().expect("the calling door");
                assert_eq!(
                    door.gone(Remover::Maker(&Maker::Control), "http://absent", ABSENT),
                    Err(Refused::Absent)
                );
                heard.fetch_add(1, Ordering::Relaxed);
            })
        });
        let (done, finished) = mpsc::channel();
        let writer = thread::spawn(move || {
            let found = door.found_one(&maker, URL, &|_| stated());
            let seen = standing(&store.by_key(URL).expect("the registered row"));
            let by = Remover::Maker(&maker);
            let drained = door.drained(by, URL, seen);
            let gone = door.gone(by, URL, seen);
            done.send([found, drained, gone]).expect("the answers");
        });
        let [found, drained, gone] = finished
            .recv_timeout(Duration::from_secs(5))
            .expect("a listener can call the door");
        let Ok(Wrote::Registered(id)) = found else {
            panic!("the row was not registered: {found:?}")
        };
        assert_eq!(drained, Ok(Wrote::Draining(id)));
        assert_eq!(gone, Ok(Wrote::Removed(id)));
        assert_eq!(count.load(Ordering::Relaxed), 3);
        writer.join().expect("the writer finished");
    }

    #[test]
    fn a_worker_whose_checks_are_off_is_ready_at_once() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
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
        assert_eq!(wrote, Ok(Wrote::InPlace(row.id)));
        assert_eq!((row.status, row.revision), (WorkerStatus::Ready, 2));

        door.gone(Remover::Maker(&maker), URL, standing(&row))
            .expect("removed");
        let wrote = door.found_one(&maker, URL, &|_| row.facts.clone());
        let fresh = store.by_key(URL).expect("the newly registered row");
        assert_eq!(wrote, Ok(Wrote::Registered(fresh.id)));
        assert_eq!((fresh.status, fresh.revision), (WorkerStatus::Ready, 1));
    }

    #[test]
    fn calls_at_once_take_their_turn() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let ready = Barrier::new(4);
        let answers = thread::scope(|scope| {
            let callers = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        ready.wait();
                        door.found_one(&maker, URL, &|_| {
                            assert!(
                                door.turn.is_locked(),
                                "the facts are stated in the door's turn"
                            );
                            stated()
                        })
                    })
                })
                .collect::<Vec<_>>();
            callers
                .into_iter()
                .map(|caller| caller.join().expect("the call finished"))
                .collect::<Vec<_>>()
        });
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(
            answers
                .iter()
                .filter(|answer| **answer == Ok(Wrote::Registered(row.id)))
                .count(),
            1
        );
        assert_eq!(
            answers
                .iter()
                .filter(|answer| **answer == Ok(Wrote::Unmoved(row.id)))
                .count(),
            3
        );
        assert_eq!(store.epoch(), 1);
    }

    #[test]
    fn the_facts_are_stated_over_the_row_that_stands() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
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
        assert_eq!(wrote, Ok(Wrote::InPlace(old.id)));
        let row = store.by_key(URL).expect("the refreshed row");
        assert_eq!(row.facts.zone.as_deref(), Some("east"));
        assert_eq!(row.facts.priority, 5);

        // Facts under another key are refused in every build: the store
        // writes by the facts' key, and no maker was asked of that row.
        let mismatch = door.found_one(&maker, URL, &|_| WorkerFacts {
            key: "http://another".to_owned(),
            ..stated()
        });
        let refused = Refused::Past(Past {
            fact: "key",
            why: "the facts name the URL of the call",
        });
        assert_eq!(mismatch, Err(refused));
        assert_eq!(store.by_key(URL), Some(row));
        assert_eq!(store.by_key("http://another"), None);
    }
}
