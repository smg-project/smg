use std::{sync::Arc, time::Instant};

use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    settings::UnhealthyRemovalScope,
    worker::{HealthCheckConfig, WorkerType},
};

use super::{
    super::found, asks_events_under, awaits_connect, changed, drain, lay, lay_one, refresh,
    refresh_stated, stating, take, MembershipWriteOutcome, MembershipWriteRefused,
    MembershipWriter, Over, Precondition, Remover, RowRevision, Stated, Turn,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        demands::DemandFacts,
        view::{Role, WireModes, WorkerId},
    },
    service::replication::{Event, Log},
    store::{
        membership::{self, FactPastBound},
        pairing::PairingDescriptor,
        HealthPolicy, MembershipChange, MemoryStore, Position, Registrant, Signals, WorkerFacts,
        WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
    },
    worker_host::{kv_tables::indexed_under, Members, Objects},
};

const URL: &str = "http://worker:8000";

/// The reads and the writes of a turn over a store alone, which every
/// turn of these cases is: the store's own table, where no holder
/// states or decides a record. `$store` names the turn's field that
/// holds the store.
macro_rules! over_the_table {
    ($store:tt) => {
        fn asks_events(&self, facts: &WorkerFacts) -> bool {
            asks_events_under(facts, &self.$store)
        }

        fn upsert(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange> {
            Some(self.$store.upsert_by(facts, maker))
        }

        fn patch(
            &self,
            id: WorkerId,
            patch: WorkerPatch,
            revision: u64,
        ) -> Option<MembershipChange> {
            self.$store.patch(id, patch, revision)
        }

        fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
            self.$store.remove(id, revision)
        }

        fn connect(&self, id: WorkerId, revision: u64) {
            let standing = self.$store.get(id).map(|row| row.status);
            if !matches!(standing, Some(WorkerStatus::Draining | WorkerStatus::Ready)) {
                self.$store.set_status(id, WorkerStatus::Ready, revision);
            }
        }

        fn drain(&self, id: WorkerId, revision: u64) -> bool {
            self.$store
                .set_status(id, WorkerStatus::Draining, revision)
                .is_some()
        }

        fn row(&self, id: WorkerId) -> Option<WorkerRow> {
            self.$store.get(id)
        }

        fn row_under(&self, url: &str) -> Option<WorkerRow> {
            self.$store.by_key(url)
        }

        fn rows_of(&self, engine: &str) -> Vec<WorkerRow> {
            let (_, rows) = self.$store.all_rows();
            rows.into_iter()
                .filter(|row| row.facts.engine == engine)
                .collect()
        }
    };
}

mod expectations;
mod lives;
mod ranks;
mod refresh;
mod removal;
mod term;

#[cfg(feature = "controller")]
pub(crate) use self::{
    expectations::{check_event_signal, check_expectations},
    ranks::{check_one_row, check_rank_rows},
    refresh::check_refresh,
    removal::{check_guard, check_health_scope, check_maker, check_provider},
};

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
    over_the_table!(store);

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

/// A door over a store and the holder of its worker objects, with no
/// election and no log: each call runs the door's rules, reads a row
/// off the worker objects and writes through the holder, as a
/// controller's door does. So the object of a row it wrote stands when
/// it answers.
pub(crate) struct HolderDoor {
    store: Arc<MemoryStore>,
    objects: Arc<Objects<Members>>,
    base: u64,
}

impl HolderDoor {
    pub(crate) fn new(store: Arc<MemoryStore>, objects: Arc<Objects<Members>>) -> Self {
        Self {
            store,
            objects,
            base: membership::base(),
        }
    }
}

impl Turn for HolderDoor {
    fn asks_events(&self, facts: &WorkerFacts) -> bool {
        asks_events_under(facts, &self.store)
    }

    fn row(&self, id: WorkerId) -> Option<WorkerRow> {
        self.objects.read_row(id)
    }

    fn row_under(&self, url: &str) -> Option<WorkerRow> {
        self.objects.read_row_under(url)
    }

    fn rows_of(&self, engine: &str) -> Vec<WorkerRow> {
        self.objects.records_of(engine)
    }

    fn upsert(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange> {
        self.objects.upsert(&self.store, facts, maker)
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        self.objects.patch(&self.store, id, patch, revision)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        self.objects.remove(&self.store, id, revision)
    }

    fn connect(&self, id: WorkerId, revision: u64) {
        self.objects.connect_row(&self.store, id, revision);
    }

    fn drain(&self, id: WorkerId, revision: u64) -> bool {
        self.objects.drain_row(&self.store, id, revision)
    }

    fn life(&self) -> Result<u64, MembershipWriteRefused> {
        Ok(membership::life(0, self.base, self.store.epoch()))
    }

    fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
        Ok(())
    }
}

impl MembershipWriter for HolderDoor {
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

/// A row waits for its connect signal where its checks are off and it
/// stands in a status that signal moves: every one but ready and
/// draining. A checked row waits for none, whatever it stands in.
#[test]
fn a_row_awaits_its_connect_signal_with_its_checks_off_unless_ready_or_draining() {
    let row = |probes, status| WorkerRow {
        maker: Registrant::Control,
        id: WorkerId(1),
        facts: WorkerFacts {
            health: HealthPolicy {
                probes,
                ..HealthPolicy::default()
            },
            ..WorkerFacts::default()
        },
        status,
        revision: 1,
        admitted_at: Instant::now(),
    };
    for (status, awaits) in [
        (WorkerStatus::Pending, true),
        (WorkerStatus::NotReady, true),
        (WorkerStatus::Failed, true),
        (WorkerStatus::Ready, false),
        (WorkerStatus::Draining, false),
    ] {
        assert_eq!(awaits_connect(&row(false, status)), awaits, "{status:?}");
        assert!(!awaits_connect(&row(true, status)), "{status:?}, checked");
    }
}

#[test]
fn a_fact_past_its_bound_writes_nothing() {
    struct Shut<'a>(&'a MemoryStore);

    impl Turn for Shut<'_> {
        over_the_table!(0);

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
            &door,
            store.set_status(row.id, WorkerStatus::Draining, stale),
            MembershipWriteOutcome::Draining
        ),
        Err(MembershipWriteRefused::Moved)
    );
    assert_eq!(
        changed(
            &door,
            store.remove(row.id, stale),
            MembershipWriteOutcome::Removed
        ),
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
/// holder has laid the rows the door wrote.
#[cfg(feature = "in-process")]
mod decided {
    use std::{sync::Arc, time::Instant};

    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        model_card::ModelCard,
        pd::pairing::PdPairingMode,
        registry::WorkerRegistry,
        worker::{RuntimeType, WorkerSpec, WorkerType},
    };

    use super::{door, stated, URL};
    use crate::{
        document::spec::{AppliedSpec, Pairing, PolicySpec},
        engine::{
            Constraints, DecideOutcome, Decided, Decision, Demands, InProcess, PairLists, Phase,
            Request, RequestFacts, RequestMaker, Role, ShedReason, WorkerId, WorkerLists,
        },
        store::{Registrant, WorkerFacts, WorkerStatus, WorkersStore},
        worker_host::{
            facts, objects::tests::lay_standing, pairing_descriptor, Members,
            MembershipWriteOutcome, MembershipWriter, Objects, OpenAdmission, Precondition,
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

    /// A holder with a registry of its own. It lays the rows of a store
    /// where a case has it lay them, as a cut laid whole does, and builds
    /// each object in its row's status.
    fn holder() -> Objects<Members> {
        Objects::new(Arc::new(WorkerRegistry::new()))
    }

    /// What `engine` decides for `asked` over `lists`: a whole decision.
    /// `Asks` states no request plain ([`RequestMaker::plain`]), so
    /// every request is decided whole and none is answered a place.
    fn decided(engine: &InProcess, asked: Request<'_>, lists: WorkerLists<'_>) -> DecideOutcome {
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
        let engine = InProcess::new(applied);
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
        // once it has laid the store's rows, under the classes it hands
        // beside them: whether the prefill and the decode pair, and the
        // pair the decision picked or what it was shed under.
        let placed = || {
            lay_standing(&holder, &store);
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

    /// A pair's worker as a provider's pass finds it, at `url` in
    /// `role`: its operator's pairing protocol stands on its spec, where
    /// the Kubernetes provider writes a pod's label, and a label names
    /// its transfer backend.
    fn found_side(
        url: &str,
        role: WorkerType,
        protocol: Option<&str>,
        transport: &str,
    ) -> Arc<dyn Worker> {
        let mut spec = WorkerSpec::new(url);
        spec.worker_type = role;
        spec.runtime_type = RuntimeType::Sglang;
        spec.pairing_protocol = protocol.map(str::to_owned);
        Arc::new(
            <BasicWorkerBuilder>::from_spec(spec)
                .model(ModelCard::new(MODEL))
                .label("disaggregation_transfer_backend", transport)
                .build(),
        )
    }

    /// Two found workers pair as their own pairing facts do. A door that
    /// reads a built worker alone lays no descriptor in its row, and
    /// lays the protocol the worker's spec holds among the row's labels.
    /// The holder's classes read what the object built of each row
    /// offers. So under the lenient rule a prefill and a decode whose
    /// operators' protocols differ pair with nobody, and neither do two
    /// of unlike transports: a pair's request is shed. One protocol on
    /// both sides settles a pair of two transports, as the workers' own
    /// rule has it, and the request is placed on the two.
    #[test]
    fn two_found_workers_pair_as_their_own_pairing_facts_do() {
        const DECODE: &str = "http://decode:8000";
        let mut lenient =
            PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
        lenient.generation = 1;
        lenient.plans.pd_pair.pairing = Pairing::Lenient;
        let applied = AppliedSpec::validate(&lenient).expect("the document validates");
        let engine = InProcess::new(applied);
        let sides = [
            ((Some("blue"), "nixl"), (Some("green"), "nixl")),
            ((None, "nixl"), (None, "mooncake")),
            ((Some("blue"), "nixl"), (Some("blue"), "mooncake")),
            ((None, "nixl"), (None, "nixl")),
        ];
        let mut paired = Vec::new();
        for ((protocol, transport), (other_protocol, other_transport)) in sides {
            let (store, door) = door();
            let prefill = found_side(URL, WorkerType::Prefill, protocol, transport);
            let decode = found_side(DECODE, WorkerType::Decode, other_protocol, other_transport);
            let own = prefill
                .pd_pairing()
                .mismatch(decode.pd_pairing(), PdPairingMode::Lenient)
                .is_none();
            let mut rows = Vec::new();
            for built in [&prefill, &decode] {
                let maker = Registrant::Control;
                let wrote = door.found_built(&maker, Precondition::Absent, built, None);
                assert!(wrote.is_ok(), "{wrote:?}");
                let laid = store.by_key(built.url()).expect("the registered row");
                assert_eq!(laid.facts.pairing, None, "the row states no descriptor");
                let stated = laid.facts.labels.get("pairing_protocol");
                assert_eq!(stated.map(String::as_str), built.pd_pairing().explicit());
                let _ = store.set_status(laid.id, WorkerStatus::Ready, laid.revision);
                rows.push(laid.id);
            }
            let holder = holder();
            lay_standing(&holder, &store);
            let registry = holder.registry();
            let generation = registry.model_generation(MODEL).expect("a generation");
            let classes = holder.pair_classes(&generation, Pairing::Lenient);
            assert_eq!(classes.pairs(0, 0), own, "as the workers' own facts pair");
            let lists = PairLists {
                prefill: holder.lent(&generation, Role::Prefill),
                decode: holder.lent(&generation, Role::Decode),
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
            let placed = match decided(&engine, asked, WorkerLists::Pair(lists)).decision {
                Decision::Pick(pick) => {
                    let target = pick.targets.first().expect("a target");
                    Ok((target.id, target.pair.map(|pair| pair.decode_id)))
                }
                Decision::Shed(shed) => Err((shed.reason, shed.emptied_by)),
                Decision::Queue(_) => panic!("a pair's request is not queued"),
            };
            let the_pair = Ok((rows[0], Some(rows[1])));
            let shed = Err((ShedReason::NoEndpoints, "partnered"));
            assert_eq!(placed, if own { the_pair } else { shed });
            paired.push(own);
        }
        assert_eq!(paired, [false, false, true, true]);
    }

    /// A refresh that turns a worker's vision off starts another life,
    /// as one that changes a demand fact does: the door reads the bit
    /// as a hard demand. The row that took images is gone when the
    /// write is answered. Once the holder has laid the new row, a
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
        let engine = InProcess::new(applied);
        let holder = holder();
        // A request that states whether it needs vision, decided over
        // the list the holder lends once it has laid the store's rows.
        let decide = |vision: bool| {
            lay_standing(&holder, &store);
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
