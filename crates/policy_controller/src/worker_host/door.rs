//! Writes through a host's membership door.

use std::sync::Arc;

use parking_lot::Mutex;
use smg_worker::{identity::basic::Worker, settings::UnhealthyRemovalScope};

use super::{found::facts, kv_feed::asks_events, Standing};
use crate::{
    store::{
        membership::{self, NotMaker, Past},
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

/// A write the door refused. Nothing was written.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// A row stands under the URL of a registration that expects none.
    Stands,
    /// The row a replacement or a refresh names is draining: its removal
    /// takes the worker.
    Draining,
    /// The row's maker is outside the health removal scope.
    OutOfScope,
    /// A refresh's worker states another fact than the row's labels. A
    /// refresh writes the labels alone.
    NotLabelsAlone,
}

impl Refused {
    pub(crate) const OUT_OF_SCOPE: &str = "the row's maker is outside the health removal scope";
    pub(crate) const NOT_LABELS_ALONE: &str =
        "a refresh writes a row's labels and states no other fact";
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

/// What a registration expects to stand under its URL. Once that holds, its
/// facts are laid by one rule whatever it expects: an absent URL is
/// registered, and a standing row is left, changed in place or replaced as
/// its facts say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// Nothing: an add. A standing row refuses it ([`Refused::Stands`]),
    /// whoever made the row.
    Nothing,
    /// The row its asker saw, at the revision it saw: a replacement by id.
    /// No row refuses it ([`Refused::Absent`]), and so does another row or
    /// revision ([`Refused::Moved`]). A draining row refuses it too
    /// ([`Refused::Draining`]): its removal wins.
    Seen(Standing),
    /// Anything: an upsert.
    Anything,
}

/// The membership door a host hands its registration and removal steps.
pub trait Membership: Send + Sync + 'static {
    /// Lay a registration's facts under `url`, over what it expects to
    /// stand there. The callback states facts for `url` during the door's
    /// turn and must leave writes to the door. Facts whose key is not `url`
    /// are refused: the maker was asked of the row under `url`, and the
    /// store writes by the facts' key. The door states the row's event
    /// signal itself, by the generation in force, whatever the facts state.
    fn found(
        &self,
        maker: &Maker,
        expect: Expect,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<Wrote, Refused>;

    /// Lay the registration of a worker its asker built, under the
    /// worker's URL. The facts are the ones `built` states over the
    /// standing row ([`facts`]).
    ///
    /// Each door says what it does with the object. A door whose host
    /// keeps built objects hands `built` to the holder in the write's turn
    /// ([`Objects::handing`](super::Objects::handing)). A door over a door
    /// passes the call on. Any other door writes the facts alone, and its
    /// host's holder builds the object from the row.
    ///
    /// `over` is the object a guarded write replaces: the one its asker
    /// read and built from. A kept object can change while its row's
    /// revision stands, so a door that hands `built` in refuses the write
    /// as moved where another object stands for the row by then. A door
    /// that keeps no object has nothing to hold it against.
    fn found_built(
        &self,
        maker: &Maker,
        expect: Expect,
        built: &Arc<dyn Worker>,
        over: Option<&Arc<dyn Worker>>,
    ) -> Result<Wrote, Refused>;

    /// Mark the row the asker saw as draining. Its revision stays the same.
    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused>;

    /// Remove the row the asker saw under the registered URL.
    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused>;

    /// Write the labels of the worker the health loop's refresh rebuilt,
    /// over the row the asker saw under the worker's URL. The labels are
    /// the engine's word and nobody's registration: no maker is asked, a
    /// control call's row and a provider's are served alike, and the row
    /// keeps the maker it has.
    ///
    /// The facts `built` states over the standing row ([`facts`]) may
    /// differ from the row's in the labels alone. Any other difference is
    /// refused ([`Refused::NotLabelsAlone`]), so a refresh starts no life. No
    /// row, another row or revision, and a draining row refuse it too. The
    /// labels' room is tested as a registration's is. Equal labels write
    /// nothing and answer [`Wrote::Unmoved`].
    ///
    /// Each door says what it does with the object, as for
    /// [`Self::found_built`]. A door whose host keeps built objects hands
    /// `built` to the holder in the write's turn, whether or not a label
    /// moved: a refresh can rebuild what no row states. A door over a door
    /// passes the call on. Any other door writes the labels alone.
    ///
    /// `over` is the object the refresh rebuilt from. Equal labels move no
    /// revision, so the row's guard cannot tell an older rebuild from a
    /// newer one: a door that hands `built` in refuses the refresh as moved
    /// where another object stands for the row by then, or none. It asks
    /// ahead of the row's guard, so a worker that left reads as moved
    /// there. A door that keeps no object has nothing to hold it against.
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

    /// The number of the life a registration makes now.
    fn life(&self) -> Result<u64, Refused>;

    /// The door's own rules of the facts it is about to write, behind the
    /// store's measure.
    fn room(&self, facts: &WorkerFacts, maker: &Maker) -> Result<(), Refused>;
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

/// Lay a registration over what it expects to stand while the door holds
/// its turn. The row's event signal is stated here, over the facts the
/// callback states ([`asks_events`]).
pub(crate) fn lay(
    turn: &impl Turn,
    maker: &Maker,
    expect: Expect,
    url: &str,
    state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
) -> Result<Wrote, Refused> {
    let store = turn.store();
    let standing = match expect {
        Expect::Seen(seen) => Some(seen_row(store, url, seen)?),
        Expect::Nothing | Expect::Anything => store.by_key(url),
    };
    // An add changes no standing row, so it asks no maker and states no
    // fact: its refusal is the same whoever made the row.
    if expect == Expect::Nothing && standing.is_some() {
        return Err(Refused::Stands);
    }
    if let Some(row) = &standing {
        made_by(row, maker)?;
        // A draining row is leaving. Laid over, it would take the URL's
        // next life and its removal would leave the worker: the removal
        // wins over a replacement.
        if matches!(expect, Expect::Seen(_)) && row.status == WorkerStatus::Draining {
            return Err(Refused::Draining);
        }
    }

    let mut facts = state(standing.as_ref().map(|row| &row.facts));
    if facts.key != url {
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

    match &standing {
        None => facts.incarnation = turn.life()?,
        Some(row) => match row.laid(&facts) {
            Laid::Unmoved => return Ok(Wrote::Unmoved(row.id)),
            Laid::InPlace => {}
            Laid::Another => facts.incarnation = turn.life()?,
        },
    }
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
    if !probes {
        if let Some(row) = store.get(change.id()) {
            store.connected(row.id, row.revision, &|| true);
        }
    }
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

/// Write the labels a refresh's rebuilt worker states over the guarded row
/// while the door holds its turn. The row's other facts are the test of
/// the refresh's scope and are not written: the door states a row's event
/// signal at its registration, and a refresh leaves it as it stands.
pub(crate) fn refresh(
    turn: &impl Turn,
    seen: Standing,
    built: &dyn Worker,
) -> Result<Wrote, Refused> {
    let store = turn.store();
    let row = seen_row(store, built.url(), seen)?;
    // A draining row is leaving: its removal wins over a refresh, as over
    // a replacement.
    if row.status == WorkerStatus::Draining {
        return Err(Refused::Draining);
    }
    let mut stated = facts(built, Some(&row.facts));
    let labels = std::mem::replace(&mut stated.labels, row.facts.labels.clone());
    if stated != row.facts {
        return Err(Refused::NotLabelsAlone);
    }
    if labels == row.facts.labels {
        return Ok(Wrote::Unmoved(row.id));
    }
    stated.labels = labels;
    stated.bounded(&row.maker).map_err(Refused::Past)?;
    turn.room(&stated, &row.maker)?;
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

    /// Call `told` once per write, after the turn is released. A refusal
    /// or an unmoved row leaves the listener alone.
    #[must_use]
    pub fn telling(mut self, told: impl Fn() + Send + Sync + 'static) -> Self {
        self.told = Some(Box::new(told));
        self
    }

    fn call(&self, write: impl FnOnce() -> Result<Wrote, Refused>) -> Result<Wrote, Refused> {
        let turn = self.turn.lock();
        let answer = write();
        drop(turn);
        if !matches!(answer, Err(_) | Ok(Wrote::Unmoved(_))) {
            if let Some(told) = &self.told {
                told();
            }
        }
        answer
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
        self.call(|| lay(self, maker, expect, url, state))
    }

    // This door has no holder. The row takes the facts alone.
    fn found_built(
        &self,
        maker: &Maker,
        expect: Expect,
        built: &Arc<dyn Worker>,
        _: Option<&Arc<dyn Worker>>,
    ) -> Result<Wrote, Refused> {
        self.found(maker, expect, built.url(), &|standing| {
            facts(built.as_ref(), standing)
        })
    }

    fn drained(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.call(|| drain(&self.store, by, url, seen))
    }

    fn gone(&self, by: Remover<'_>, url: &str, seen: Standing) -> Result<Wrote, Refused> {
        self.call(|| take(&self.store, by, url, seen))
    }

    // This door has no holder. The row takes the labels alone.
    fn refreshed(
        &self,
        seen: Standing,
        built: &Arc<dyn Worker>,
        _: &Arc<dyn Worker>,
    ) -> Result<Wrote, Refused> {
        self.call(|| refresh(self, seen, built.as_ref()))
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
        changed, lay, Expect, Membership, Refused, Remover, Standing, StoreDoor, Turn, Wrote,
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
            Maker, MemoryStore, Signals, WorkerFacts, WorkerRow, WorkerStatus, WorkersStore,
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

    fn local_door() -> (Arc<MemoryStore>, Box<dyn Membership>) {
        let (store, door) = door();
        (store, Box::new(door))
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
    pub(crate) fn check_event_signal(open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Membership>)) {
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
            door.found(&Maker::Control, Expect::Nothing, URL, &|_| found.clone())
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
            door.found(&Maker::Control, Expect::Anything, URL, &|_| found.clone())
                .expect("laid again");
            let row = store.by_key(URL).expect("the row laid again");
            assert_eq!(
                row.facts.signals.events,
                wire_modes == WireModes::GRPC,
                "{model} {role:?} {wire_modes:?} under the later generation"
            );
        }
    }

    /// An add registers an absent URL, and a standing row refuses it before
    /// a fact is stated, whoever made the row. A replacement is laid over
    /// the row its asker saw by the rule every registration takes, and is
    /// refused where that row moved or left.
    pub(crate) fn check_expectations(
        open: impl FnOnce() -> (Arc<MemoryStore>, Box<dyn Membership>),
    ) {
        let (store, door) = open();
        let control = Maker::Control;
        let pool = Maker::provider("pool", "private/key").expect("a maker");
        let unstated = |_: Option<&WorkerFacts>| -> WorkerFacts { panic!("no fact is stated") };

        let added = door.found(&control, Expect::Nothing, URL, &|_| stated());
        let row = store.by_key(URL).expect("the added row");
        assert_eq!(added, Ok(Wrote::Registered(row.id)));
        let epoch = store.epoch();
        for maker in [&control, &pool] {
            assert_eq!(
                door.found(maker, Expect::Nothing, URL, &unstated),
                Err(Refused::Stands)
            );
        }

        let seen = Expect::Seen(standing(&row));
        assert_eq!(
            door.found(&control, seen, URL, &|_| stated()),
            Ok(Wrote::Unmoved(row.id))
        );
        assert_eq!(
            door.found(&pool, seen, URL, &unstated),
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
            door.found(&control, seen, URL, &relabelled),
            Ok(Wrote::InPlace(row.id))
        );
        assert_eq!(
            door.found(&control, seen, URL, &unstated),
            Err(Refused::Moved)
        );

        // Other models are the URL's next life, as under an upsert.
        let moved = store.by_key(URL).expect("the relabelled row");
        let other = |_: Option<&WorkerFacts>| WorkerFacts {
            models: vec!["other".to_owned()],
            ..stated()
        };
        let wrote = door.found(&control, Expect::Seen(standing(&moved)), URL, &other);
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
            door.found(&control, Expect::Seen(seen), URL, &unstated),
            Err(Refused::Draining)
        );
        assert!(matches!(
            door.found(&pool, Expect::Seen(seen), URL, &unstated),
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
            door.found(&control, Expect::Seen(seen), URL, &unstated),
            Err(Refused::Absent)
        );
        assert_eq!(store.by_key(URL), None);
    }

    #[test]
    fn a_removal_takes_only_the_row_its_asker_saw() {
        check_guard(local_door).expect("both registrations stand in turn");
    }

    pub(crate) fn check_guard(
        open: impl FnOnce() -> (Arc<MemoryStore>, Box<dyn Membership>),
    ) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Maker::Control;
        let by = Remover::Maker(&maker);
        assert!(door
            .found(&maker, Expect::Anything, URL, &|_| stated())
            .is_ok());
        let first = store.by_key(URL).ok_or("the first row")?;
        assert!(door
            .found(&maker, Expect::Anything, URL, &|_| WorkerFacts {
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
        open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Membership>),
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
            assert!(door
                .found(&maker, Expect::Anything, URL, &|_| stated())
                .is_ok());
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
        open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Membership>),
    ) -> Result<(), &'static str> {
        let provider = Maker::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        for (scope, control, discovered) in [
            (UnhealthyRemovalScope::Disabled, false, false),
            (UnhealthyRemovalScope::DiscoveryOwned, false, true),
            (UnhealthyRemovalScope::AllLocal, true, true),
        ] {
            for (maker, allowed) in [(&Maker::Control, control), (&provider, discovered)] {
                let (store, door) = open();
                assert!(door
                    .found(maker, Expect::Anything, URL, &|_| stated())
                    .is_ok());
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
        open: impl FnOnce() -> (Arc<MemoryStore>, Box<dyn Membership>),
    ) -> Result<(), &'static str> {
        let (store, door) = open();
        let maker = Maker::provider("pool", "private/key").map_err(|_| "the source's maker")?;
        assert!(door
            .found(&maker, Expect::Anything, URL, &|_| stated())
            .is_ok());
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
        open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Membership>),
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

    #[test]
    fn a_found_worker_is_registered_under_its_maker() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        let wrote = door.found(&maker, Expect::Anything, URL, &|standing| {
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
        door.found(&maker, Expect::Anything, URL, &|_| stated())
            .expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(
            door.found(&maker, Expect::Anything, URL, &|_| stated()),
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
        door.found(&maker, Expect::Anything, URL, &|_| stated())
            .expect("registered");
        let old = store.by_key(URL).expect("the registered row");
        let labelled = WorkerFacts {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..stated()
        };
        assert_eq!(
            door.found(&maker, Expect::Anything, URL, &|_| labelled.clone()),
            Ok(Wrote::InPlace(old.id))
        );
        let standing = store.by_key(URL).expect("the refreshed row");
        assert_eq!(standing.id, old.id);
        assert_eq!(standing.facts.incarnation, old.facts.incarnation);
        assert_eq!(standing.facts.labels, labelled.labels);
        assert_eq!(standing.revision, 2);

        let before = store.epoch();
        let wrote = door.found(&maker, Expect::Anything, URL, &|_| WorkerFacts {
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
            demand_facts: DemandFacts { runtime },
            ..stated()
        };
        door.found(&maker, Expect::Anything, URL, &|_| stating(7))
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
            retry.constraints.demands = Demands { runtime };
            let decided = policy.decide(GATEWAY, &retry, Instant::now());
            decided.expect("a decision")
        };
        let placed = decide(b"before", 7);
        let target = placed.decision.pick().expect("the worker states runtime 7");
        assert_eq!(target.targets.first().map(|target| target.id), Some(old.id));

        let wrote = door.found(&maker, Expect::Anything, URL, &|_| stating(9));
        let row = store.by_key(URL).expect("the replaced row");
        assert_eq!(
            wrote,
            Ok(Wrote::Another {
                old: old.id,
                new: row.id
            })
        );
        assert_eq!(store.get(old.id), None);
        assert_eq!(row.facts.demand_facts, DemandFacts { runtime: 9 });
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
            door.found(&maker, Expect::Anything, URL, &|_| stated())
                .expect("registered");
            let row = store.by_key(URL).expect("the registered row");
            let expected = Err(Refused::NotMaker(why.to_owned()));
            assert_eq!(
                door.found(&caller, Expect::Anything, URL, &|_| panic!(
                    "only the maker states facts"
                )),
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

    #[test]
    fn a_fact_past_its_bound_writes_nothing() {
        struct Shut<'a>(&'a MemoryStore);

        impl Turn for Shut<'_> {
            fn store(&self) -> &MemoryStore {
                self.0
            }

            fn life(&self) -> Result<u64, Refused> {
                Err(Refused::NotLeading("no life".to_owned()))
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
        let no = Err(Refused::Past(Past {
            fact: "models",
            why: "a model id is at most 256 bytes",
        }));
        assert_eq!(
            door.found(&maker, Expect::Anything, URL, &|_| invalid.clone()),
            no
        );
        assert_eq!(
            lay(&Shut(&store), &maker, Expect::Anything, URL, &|_| invalid
                .clone()),
            no
        );
        assert_eq!(
            lay(&Shut(&store), &maker, Expect::Anything, URL, &|_| stated()),
            Err(Refused::Past(Past {
                fact: "row",
                why: "no room"
            }))
        );
        assert_eq!(store.epoch(), 0);
        assert_eq!(store.by_key(URL), None);

        store.upsert_by(invalid.clone(), maker.clone());
        let standing = store.by_key(URL);
        assert_eq!(
            door.found(&maker, Expect::Anything, URL, &|_| invalid.clone()),
            no
        );
        assert_eq!(store.by_key(URL), standing);
        assert_eq!(store.epoch(), 1);
    }

    #[test]
    fn a_drain_marks_the_makers_row_once_and_a_removal_takes_it() {
        let (store, door) = door();
        let maker = Maker::provider("pool", "first").expect("a maker");
        door.found(&maker, Expect::Anything, URL, &|_| stated())
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
            let found = door.found(&maker, Expect::Anything, URL, &|_| stated());
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
        door.found(&maker, Expect::Anything, URL, &|_| stated())
            .expect("registered");
        assert_eq!(
            store.by_key(URL).map(|row| row.status),
            Some(WorkerStatus::Pending)
        );
        let wrote = door.found(&maker, Expect::Anything, URL, &|_| {
            let mut facts = stated();
            facts.health.probes = false;
            facts
        });
        let row = store.by_key(URL).expect("the row with its checks off");
        assert_eq!(wrote, Ok(Wrote::InPlace(row.id)));
        assert_eq!((row.status, row.revision), (WorkerStatus::Ready, 2));

        door.gone(Remover::Maker(&maker), URL, standing(&row))
            .expect("removed");
        let wrote = door.found(&maker, Expect::Anything, URL, &|_| row.facts.clone());
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
                        door.found(&maker, Expect::Anything, URL, &|_| {
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
        door.found(&maker, Expect::Anything, URL, &|_| WorkerFacts {
            zone: Some("east".to_owned()),
            ..stated()
        })
        .expect("registered");
        let old = store.by_key(URL).expect("the registered row");
        let wrote = door.found(&maker, Expect::Anything, URL, &|standing| {
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
        let mismatch = door.found(&maker, Expect::Anything, URL, &|_| WorkerFacts {
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
