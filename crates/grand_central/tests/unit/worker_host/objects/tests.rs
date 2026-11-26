use std::{cell::RefCell, collections::BTreeMap, sync::mpsc, thread, time::Duration};

use smg_worker::{
    health::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        port::HealthSink as _,
    },
    identity::builder::BasicWorkerBuilder,
    kv_events::port::{BatchFacts, BatchOutcome},
    registry::event::WorkerEvent,
    runtime::StallReason,
    worker::WorkerStatus as ObjectStatus,
};
use tokio::sync::broadcast::Receiver;

use super::*;
pub(crate) use super::{status::tests::stand, write::tests::lay_standing};
use crate::{
    store::{
        GatewayId, MembershipChange, MemoryStore, Overload, Registrant, WorkerFacts, WorkerPatch,
        WorkerStatus,
    },
    worker_host::{kv_tables, object, HealthAdapter, KeptCredential, Members, SoleLead},
};

const A: &str = "http://a:8000";
const B: &str = "http://b:8000";
const C: &str = "http://c:8000";
const D: &str = "http://d:8000";
const E: &str = "http://e:8000";
const F: &str = "http://f:8000";
const G: &str = "http://g:8000";
const KEY_A: &str = "key-a";
const KEY_B: &str = "key-b";
const KEY_C: &str = "key-c";

fn facts(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        key: key.to_string(),
        engine: key.to_string(),
        incarnation,
        models: vec!["m".to_string()],
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

fn rank_facts(key: &str, engine: &str, rank: u16) -> WorkerFacts {
    WorkerFacts {
        engine: engine.to_string(),
        dp_rank: rank,
        dp_size: 2,
        ..facts(key, 1)
    }
}

/// [`facts`] with one label.
fn labelled(key: &str, incarnation: u64, tier: &str) -> WorkerFacts {
    WorkerFacts {
        labels: BTreeMap::from([("tier".to_string(), tier.to_string())]),
        ..facts(key, incarnation)
    }
}

fn labels(tier: &str) -> WorkerPatch {
    WorkerPatch {
        labels: Some(BTreeMap::from([("tier".to_string(), tier.to_string())])),
        ..WorkerPatch::default()
    }
}

/// One write of a replica's store, under the name a failure says.
type Write<'a> = (&'a str, &'a dyn Fn(&Replica));

/// What a registry told since the last read of `events`, in its order.
fn told(events: &mut Receiver<WorkerEvent>) -> Vec<(&'static str, ObjectId)> {
    let mut told = Vec::new();
    while let Ok(event) = events.try_recv() {
        told.push(match event {
            WorkerEvent::Registered { worker_id, .. } => ("registered", worker_id),
            WorkerEvent::Replaced { worker_id, .. } => ("replaced", worker_id),
            WorkerEvent::Removed { worker_id, .. } => ("removed", worker_id),
            WorkerEvent::StatusChanged { worker_id, .. } => ("status", worker_id),
        });
    }
    told
}

/// One replica with its store, holder and keeper. Its registry tells
/// what happened to the objects.
struct Replica {
    store: Arc<MemoryStore>,
    objects: Arc<Objects<Members>>,
    credentials: Arc<Credentials>,
    events: Receiver<WorkerEvent>,
}

impl Replica {
    fn new() -> Self {
        let registry = Arc::new(WorkerRegistry::new());
        let events = registry.subscribe_events();
        let credentials = Arc::new(Credentials::new());
        let objects = Objects::new(registry).with_credentials(Arc::clone(&credentials));
        Self {
            store: Arc::new(MemoryStore::new()),
            objects: Arc::new(objects),
            credentials,
            events,
        }
    }

    /// Have the holder lay each row written around it, by the read
    /// of a refused write.
    fn lay_standing(&self) {
        lay_standing(&self.objects, &self.store);
    }

    /// What the registry told since the last call, in its order.
    fn told(&mut self) -> Vec<(&'static str, ObjectId)> {
        told(&mut self.events)
    }

    fn at(&self, url: &str) -> Arc<dyn Worker> {
        self.objects.registry().get_by_url(url).expect("an object")
    }

    /// A registration taken: its keep, its write through the holder, the
    /// token accepted with the row the write answered, and the address
    /// laid again where the acceptance says so. Return the row.
    fn register(&self, facts: WorkerFacts, credential: Option<&str>) -> WorkerId {
        let token = self.kept(&facts, credential);
        let row = self.write(facts.clone());
        if token.accepted(row) {
            self.lay_again(&facts);
        }
        row
    }

    /// A registration's write as a door over the holder makes it,
    /// decided over the object that stands. A write the store refuses
    /// is decided once more, over what its read left.
    fn write(&self, facts: WorkerFacts) -> WorkerId {
        let store = self.store.as_ref();
        let lay = || {
            self.objects
                .upsert(store, facts.clone(), Registrant::Control)
        };
        let change = lay().or_else(lay);
        change.expect("the store takes the second write").id()
    }

    /// A registration's keep, its answer left to the case.
    fn kept(&self, facts: &WorkerFacts, credential: Option<&str>) -> KeptCredential {
        self.credentials
            .keep(facts, credential.map(str::to_owned))
            .expect("no registration in flight at the address")
    }

    /// The holder lays again the objects at the address `facts` dial.
    fn lay_again(&self, facts: &WorkerFacts) -> usize {
        self.objects.lay_again(self.store.as_ref(), facts)
    }

    fn holds_credential(&self, url: &str, credential: Option<&str>) -> bool {
        self.at(url).api_key().map(String::as_str) == credential
    }

    fn id(&self, url: &str) -> ObjectId {
        self.objects.registry().get_id_by_url(url).expect("an id")
    }

    fn row(&self, url: &str) -> WorkerRow {
        self.store.by_key(url).expect("a row")
    }

    /// A patch through the holder.
    fn patch(&self, url: &str, patch: WorkerPatch) {
        let (store, row) = (self.store.as_ref(), self.row(url));
        let change = self.objects.patch(store, row.id, patch, row.revision);
        assert_eq!(change, Some(MembershipChange::Relabelled { id: row.id }));
    }

    /// A removal through the holder.
    fn remove(&self, url: &str) {
        let (store, row) = (self.store.as_ref(), self.row(url));
        let removed = self.objects.remove(store, row.id, row.revision);
        assert!(removed.is_some());
    }

    /// Another writer's patch: the store's own entry, around the holder.
    fn patch_around(&self, url: &str, patch: WorkerPatch) {
        let row = self.row(url);
        let change = self.store.patch(row.id, patch, row.revision);
        assert_eq!(change, Some(MembershipChange::Relabelled { id: row.id }));
    }

    /// Another writer's removal: the store's own entry, around the
    /// holder.
    fn remove_around(&self, url: &str) {
        let row = self.row(url);
        assert!(self.store.remove(row.id, row.revision).is_some());
    }

    fn tier(&self, url: &str) -> Option<String> {
        self.at(url).metadata().spec.labels.get("tier").cloned()
    }
}

/// The decision of a retry that demands `runtime`, made by `engine`
/// over the list a holder lends.
#[cfg(feature = "in-process")]
fn retry_demanding(
    engine: &crate::engine::InProcess,
    lent: LentList<'_>,
    id: &'static [u8],
    runtime: u32,
) -> crate::engine::Decision {
    use std::time::Instant;

    use crate::{
        document::spec::AppliedSpec,
        engine::{Constraints, Decided, Demands, Request, RequestFacts, RequestMaker, WorkerLists},
    };

    /// A request stated ahead, with the list its host hands for it.
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

    let asked = Request {
        id,
        facts: RequestFacts {
            model: "m",
            seed: 7,
            ..RequestFacts::default()
        },
        constraints: Constraints {
            demands: Demands {
                runtime,
                ..Demands::default()
            },
            ..Constraints::default()
        },
    };
    let asks = Asks(asked, WorkerLists::Single(lent));
    match engine.decide_at(&asks, Instant::now).expect("an answer") {
        Decided::Whole(decided, _) => decided.decision,
        Decided::Alone(_) => panic!("a place for a request that states a demand"),
    }
}

/// The holder lends what each stored worker states for a request's hard
/// demands, and a decision over its list reads it. A retry that demands
/// a runtime lands on the worker that states that runtime, whichever
/// request it is, and one that demands a runtime no worker states is
/// shed under the stage `wires`. The classic router holds a retry to
/// its first plan's runtime and transport by one test of each worker
/// (`model_gateway/src/routers/common/placement.rs:84-93`).
#[cfg(feature = "in-process")]
#[test]
fn a_stored_workers_demand_facts_are_lent_and_hold_a_retry_to_its_runtime() {
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{Decision, DemandFacts, InProcess, Role, ShedReason},
    };

    let replica = Replica::new();
    // A stored worker that states `runtime`, ready by its connect signal.
    let ready = |key: &str, runtime: u32| {
        let stated = WorkerFacts {
            demand_facts: DemandFacts {
                runtime,
                ..DemandFacts::default()
            },
            ..facts(key, 1)
        };
        let id = replica.register(stated, None);
        let row = replica.store.get(id).expect("the registered row");
        replica
            .objects
            .connect_row(&replica.store, id, row.revision);
        id
    };
    let (seven, nine) = (ready(A, 7), ready(B, 9));

    let registry = replica.objects.registry();
    let generation = registry
        .model_generation("m")
        .expect("the model's generation");
    let lent = replica.objects.lent(&generation, Role::Single);
    let mut stated: Vec<(Option<WorkerId>, DemandFacts)> = lent
        .facts
        .expect("the holder lends facts")
        .iter()
        .map(|facts| (facts.id, facts.demands))
        .collect();
    stated.sort_by_key(|(_, demands)| demands.runtime);
    let states = |runtime| DemandFacts {
        runtime,
        ..DemandFacts::default()
    };
    assert_eq!(
        stated,
        [(Some(seven), states(7)), (Some(nine), states(9))],
        "each worker's statement, with its row's id"
    );

    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = 1;
    let applied = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(applied);
    // A retry that demands `runtime`, decided over the list lent.
    let decide = |id: &'static [u8], runtime: u32| retry_demanding(&engine, lent, id, runtime);
    let landed = |id: &'static [u8], runtime: u32| match decide(id, runtime) {
        Decision::Pick(pick) => pick.targets.first().map(|target| target.id),
        other => panic!("a worker states runtime {runtime}: {other:?}"),
    };
    for id in [&b"first"[..], b"second", b"third", b"fourth"] {
        assert_eq!(landed(id, 9), Some(nine), "the runtime's one worker");
        assert_eq!(landed(id, 7), Some(seven), "the runtime's one worker");
    }
    match decide(b"none", 5) {
        Decision::Shed(shed) => {
            assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::NoEndpoints, "wires")
            );
        }
        other => panic!("no worker states runtime 5: {other:?}"),
    }
}

/// A registration that changes the runtime a stored worker states is
/// read by the decisions behind it. The door starts another life for
/// the worker, and the holder lends the new life's
/// statement under its id: a retry that demands the new runtime lands
/// on the new life, and one that demands the old runtime is shed under
/// the stage `wires`, as no worker states it any more.
#[cfg(feature = "in-process")]
#[test]
fn a_registration_that_changes_a_runtime_moves_the_retries_that_demand_it() {
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{Decision, DemandFacts, InProcess, Role, ShedReason},
        worker_host::{
            membership_writer::tests::HolderDoor, MembershipWriteOutcome, MembershipWriter,
            Precondition,
        },
    };

    let replica = Replica::new();
    let door = HolderDoor::new(Arc::clone(&replica.store), Arc::clone(&replica.objects));
    // The registration of the worker at `A` as one that states `runtime`.
    let registers = |runtime: u32| {
        let stating = |_: Option<&WorkerFacts>| WorkerFacts {
            demand_facts: DemandFacts {
                runtime,
                ..DemandFacts::default()
            },
            ..facts(A, 1)
        };
        door.found(&Registrant::Control, Precondition::Any, A, &stating)
    };
    // What the holder lends of the model's workers as they stand: each
    // worker's id and the runtime it states.
    let lends = |replica: &Replica| {
        let registry = replica.objects.registry();
        let generation = registry
            .model_generation("m")
            .expect("the model's generation");
        let lent = replica.objects.lent(&generation, Role::Single);
        let stated: Vec<(Option<WorkerId>, u32)> = lent
            .facts
            .expect("the holder lends facts")
            .iter()
            .map(|facts| (facts.id, facts.demands.runtime))
            .collect();
        (generation, stated)
    };
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = 1;
    let applied = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(applied);
    // Where a retry that demands `runtime` lands over the list the
    // holder lends now, or the stage it is shed under.
    let retry = |replica: &Replica, runtime: u32| {
        let (generation, _) = lends(replica);
        let lent = replica.objects.lent(&generation, Role::Single);
        match retry_demanding(&engine, lent, b"again", runtime) {
            Decision::Pick(pick) => Ok(pick.targets.first().map(|target| target.id)),
            Decision::Shed(shed) => Err((shed.reason, shed.emptied_by)),
            other @ Decision::Queue(_) => panic!("neither a pick nor a shed: {other:?}"),
        }
    };
    let unstated = Err((ShedReason::NoEndpoints, "wires"));

    let wrote = registers(7);
    let Ok(MembershipWriteOutcome::Registered(first)) = wrote else {
        panic!("the first registration is a join: {wrote:?}");
    };
    // Its checks are on: its connect signal makes the first life ready.
    let row = replica.store.get(first).expect("the registered row");
    replica
        .objects
        .connect_row(&replica.store, first, row.revision);
    assert_eq!(lends(&replica).1, [(Some(first), 7)]);
    assert_eq!(retry(&replica, 7), Ok(Some(first)));
    assert_eq!(retry(&replica, 9), unstated);

    // The same worker registers again and states another runtime: the
    // rule hands the first life's status on to the next.
    let wrote = registers(9);
    let Ok(MembershipWriteOutcome::Another { old, new }) = wrote else {
        panic!("a changed runtime is another life: {wrote:?}");
    };
    assert_eq!(old, first);
    assert_eq!(lends(&replica).1, [(Some(new), 9)], "under the new id");
    assert_eq!(retry(&replica, 9), Ok(Some(new)));
    assert_eq!(retry(&replica, 7), unstated);
}

/// One side alone is no pair: the holder makes no member and does not run
/// the rule, and every worker lent takes class `0`, which is what the rule
/// answers over one side. In a debug build the rule refuses to read a
/// descriptor that stands between blanks, and no write of the store
/// leaves one so. So the case lays such a row by the holder's own step:
/// nobody reads it.
#[cfg(feature = "in-process")]
#[test]
fn one_side_alone_makes_no_member_and_every_worker_lent_pairs_with_nobody() {
    use crate::{document::spec::Pairing, engine::WireModes, store::pairing::PairingDescriptor};

    for role in [Role::Prefill, Role::Decode] {
        let replica = Replica::new();
        for url in [A, B] {
            let side = WorkerFacts {
                role,
                ..facts(url, 1)
            };
            replica.write(side);
        }
        let mut row = replica.row(A);
        let between_blanks = PairingDescriptor {
            transport: Some(" a ".to_string()),
            ..PairingDescriptor::default()
        };
        row.facts.pairing = Some(Arc::new(between_blanks));
        replica.objects.restate(&row);

        let registry = replica.objects.registry();
        let generation = registry
            .model_generation("m")
            .expect("the model's generation");
        // The side as the rule can read it: two members that report
        // nothing.
        let members = [A, B].map(|url| {
            let id = replica.row(url).id;
            Member::new(id, WireModes::default(), "", None)
        });
        for mode in [Pairing::Off, Pairing::Lenient, Pairing::Strict] {
            let derived = match role {
                Role::Prefill => PairClasses::derive(&members, &[], mode).by_place([true; 2], []),
                _ => PairClasses::derive(&[], &members, mode).by_place([], [true; 2]),
            };
            let classes = replica.objects.pair_classes(&generation, mode);
            assert_eq!(*classes, derived, "{role:?} alone under {mode:?}");
            let lent = classes.prefills().iter().chain(classes.decodes());
            assert_eq!(lent.collect::<Vec<_>>(), [&0; 2], "{role:?} alone");
            assert!(classes.table().is_empty());
        }
    }
}

/// A host that is no gateway asks for a host-local row's object with
/// its checks off, whatever the row's policy says. The object of a row
/// any gateway dials keeps its checks, and so does the row's own once
/// its reach turns. A holder that was not asked builds each as its
/// policy says.
#[test]
fn a_host_local_rows_object_is_unchecked_where_the_host_asked() {
    let checked = |replica: &Replica, url| {
        !replica
            .at(url)
            .metadata()
            .health_config
            .disable_health_check
    };
    let local = WorkerFacts {
        reachable_from: Reachability::Gateway(GatewayId(1)),
        ..facts(A, 1)
    };
    let mut outside = Replica::new();
    let registry = Arc::clone(outside.objects.registry());
    outside.objects = Arc::new(Objects::new(registry).with_host_local_unchecked());
    for (replica, asked) in [(Replica::new(), false), (outside, true)] {
        replica.write(local.clone());
        replica.write(facts(B, 1));
        assert_eq!(checked(&replica, A), !asked, "the host-local row");
        assert!(checked(&replica, B), "a row any gateway dials");

        replica.write(facts(A, 1));
        assert!(checked(&replica, A), "the row once any gateway dials it");
    }
}

/// A host whose own load monitor judges overload asks for each object
/// built with its row's thresholds, each over the host's default for
/// its signal. A row whose thresholds move is laid again, and its
/// object states the new ones. A holder that was not asked builds an
/// object that states none, so its host's monitor writes no verdict.
#[test]
fn an_object_states_its_rows_thresholds_where_the_host_judges_overload() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let none = OverloadThresholds::default();
    let stated = |replica: &Replica, url| replica.at(url).metadata().overload;
    let mut judging = Replica::new();
    let registry = Arc::clone(judging.objects.registry());
    judging.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    for (replica, asked) in [(Replica::new(), false), (judging, true)] {
        replica.write(facts(A, 1));
        replica.write(WorkerFacts {
            overload: Overload {
                kv_usage_permille: Some(650),
                waiting_requests: None,
            },
            ..facts(B, 1)
        });
        let own = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.65),
        };
        assert_eq!(stated(&replica, A), if asked { defaults } else { none });
        assert_eq!(stated(&replica, B), if asked { own } else { none });

        // The row's thresholds move: its object is laid again.
        let moved = Overload {
            kv_usage_permille: None,
            waiting_requests: Some(3),
        };
        replica.patch(
            B,
            WorkerPatch {
                overload: Some(moved),
                ..WorkerPatch::default()
            },
        );
        let own = OverloadThresholds {
            waiting_requests: Some(3),
            token_usage: Some(0.8),
        };
        assert_eq!(stated(&replica, B), if asked { own } else { none });
    }
}

/// The overload switch stated on `replica`'s holder, as a replica that
/// leads states it: how many objects were laid again.
fn switch(replica: &Replica, on: bool) -> usize {
    let store = replica.store.as_ref();
    replica.objects.state_overload_switch(store, on, &|| true)
}

/// A host states its document's overload switch on the holder. Off, the
/// object of an engine's row states its row's own thresholds alone, and
/// on again the host's default under them. A moved word lays again the
/// objects it changes and no other: a row that names both of its own
/// and an external row keep their objects. The same word again lays
/// nothing. A row laid again keeps its id,
/// its status and its count in flight, and loses its overload verdict
/// alone. A row that comes under a word is built by it, once.
#[test]
fn the_overload_switch_lays_again_the_rows_whose_default_it_changes() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let set = |waiting_requests, token_usage| OverloadThresholds {
        waiting_requests,
        token_usage,
    };
    let own = |kv_usage_permille, waiting_requests| Overload {
        kv_usage_permille,
        waiting_requests,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    // A names none, B a usage alone, C both, and D is external.
    replica.write(facts(A, 1));
    replica.write(WorkerFacts {
        overload: own(Some(650), None),
        ..facts(B, 1)
    });
    replica.write(WorkerFacts {
        overload: own(Some(650), Some(3)),
        ..facts(C, 1)
    });
    replica.write(WorkerFacts {
        engine_kind: "external".to_string(),
        ..facts(D, 1)
    });
    let urls = [A, B, C, D];
    let stated = |replica: &Replica, url| replica.at(url).metadata().overload;
    // Every standing object runs: a request in flight, ready, vetoed.
    let before = urls.map(|url| replica.at(url));
    let ids = urls.map(|url| replica.id(url));
    for worker in &before {
        worker.increment_load();
        worker.set_status(ObjectStatus::Ready);
        assert!(replica
            .objects
            .registry()
            .set_worker_overloaded(worker, true));
    }
    replica.told();

    // On is the word a holder starts under: stated again, it lays none.
    assert_eq!(switch(&replica, true), 0);
    assert!(replica.told().is_empty());
    // Off: the two rows that took a default are laid again.
    assert_eq!(switch(&replica, false), 2);
    let mut replaced = replica.told();
    replaced.retain(|told| told.0 == "replaced");
    replaced.sort_by(|one, other| one.1.as_str().cmp(other.1.as_str()));
    let mut laid = vec![("replaced", ids[0].clone()), ("replaced", ids[1].clone())];
    laid.sort_by(|one, other| one.1.as_str().cmp(other.1.as_str()));
    assert_eq!(replaced, laid);
    assert_eq!(stated(&replica, A), set(None, None));
    assert_eq!(stated(&replica, B), set(None, Some(0.65)));
    assert_eq!(stated(&replica, C), set(Some(3), Some(0.65)));
    assert_eq!(stated(&replica, D), set(None, None));
    for (at, url) in urls.into_iter().enumerate() {
        let (after, again) = (replica.at(url), at < 2);
        assert_eq!(!Arc::ptr_eq(&before[at], &after), again, "{url}");
        assert_eq!(replica.id(url), ids[at], "{url}: the same id");
        let running = (after.status(), after.load());
        assert_eq!(
            running,
            (ObjectStatus::Ready, 1),
            "{url}: its running state"
        );
        assert_eq!(after.is_overloaded(), !again, "{url}: the verdict alone");
    }

    // The same word again lays nothing.
    assert_eq!(switch(&replica, false), 0);
    assert!(replica.told().is_empty());
    // A row that comes under the word is built by it, once.
    replica.write(facts(E, 1));
    assert_eq!(replica.told(), [("registered", replica.id(E))]);
    assert_eq!(stated(&replica, E), set(None, None));

    // On again: each row that takes a default is laid again.
    assert_eq!(switch(&replica, true), 3);
    let replaced = replica
        .told()
        .into_iter()
        .filter(|told| told.0 == "replaced");
    assert_eq!(replaced.count(), 3);
    assert_eq!(stated(&replica, A), defaults);
    assert_eq!(stated(&replica, B), set(Some(8), Some(0.65)));
    assert_eq!(stated(&replica, C), set(Some(3), Some(0.65)));
    assert_eq!(stated(&replica, D), set(None, None));
    assert_eq!(stated(&replica, E), defaults);
}

/// A host that states the word ahead of its first write builds each
/// object once, by that word: nothing stands to lay again. A holder
/// that was handed no thresholds keeps the word and lays nothing: its
/// objects state none under either word.
#[test]
fn the_overload_switch_stated_ahead_of_the_first_write_builds_each_object_once() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let none = OverloadThresholds::default();
    let mut judging = Replica::new();
    let registry = Arc::clone(judging.objects.registry());
    judging.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    for (mut replica, asked) in [(Replica::new(), false), (judging, true)] {
        assert_eq!(switch(&replica, false), 0);
        assert!(replica.told().is_empty(), "nothing is built by the word");
        replica.write(facts(A, 1));
        replica.write(facts(B, 1));
        let built: Vec<_> = replica.told().into_iter().map(|told| told.0).collect();
        assert_eq!(built, ["registered", "registered"], "each object once");
        for url in [A, B] {
            assert_eq!(replica.at(url).metadata().overload, none, "{url}");
        }
        // The word back on changes a holder with thresholds alone.
        let laid = switch(&replica, true);
        assert_eq!(laid, if asked { 2 } else { 0 });
        let stands = if asked { defaults } else { none };
        for url in [A, B] {
            assert_eq!(replica.at(url).metadata().overload, stands, "{url}");
        }
    }
}

/// The switch lays again a row as the holder holds it, and reads no row
/// of the store. A row that left and came back under its id, neither
/// read by the holder yet, is another life. A moved word lays the first
/// life's object again under the word, and nothing of the second life
/// takes its place: the second takes no status and no count in flight
/// of the first. The holder's next read of the row, behind a write the
/// store refuses, ends the first life's object and builds the second's
/// anew, under the word.
#[test]
fn the_overload_switch_lays_no_row_the_holder_has_not_read() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let none = OverloadThresholds::default();
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    // The first life stands at its second revision: a write decided
    // over it is refused once another life stands at its first.
    let first = replica.write(facts(A, 1));
    replica.patch(A, labels("iron"));
    let (old_id, before) = (replica.id(A), replica.at(A));
    before.increment_load();
    before.set_status(ObjectStatus::Ready);
    replica.told();

    // The second life stands in the store alone.
    replica.remove_around(A);
    assert_eq!(replica.store.upsert(labelled(A, 1, "gold")).id(), first);
    assert_eq!(switch(&replica, false), 1);
    let replaced = replica
        .told()
        .into_iter()
        .filter(|told| told.0 == "replaced");
    assert_eq!(replaced.collect::<Vec<_>>(), [("replaced", old_id.clone())]);
    // The first life's object under the word: its status and its count,
    // and no label of the second life.
    let laid = replica.at(A);
    assert_eq!(
        replica.tier(A).as_deref(),
        Some("iron"),
        "the first life's row"
    );
    assert_eq!(laid.metadata().overload, none);
    assert_eq!((laid.status(), laid.load()), (ObjectStatus::Ready, 1));

    let store = replica.store.as_ref();
    let refused = replica.objects.patch(store, first, labels("silver"), 2);
    assert_eq!(refused, None, "decided over a life that left");
    let new_id = replica.id(A);
    assert_ne!(old_id, new_id);
    assert_eq!(
        replica.told(),
        [("removed", old_id), ("registered", new_id)]
    );
    let second = replica.at(A);
    assert_eq!(replica.tier(A).as_deref(), Some("gold"));
    assert_eq!(second.metadata().overload, none, "built under the word");
    assert_eq!(second.load(), 0, "nothing of the first life runs on");
}

/// A report that lands inside the registry's replacement leaves no
/// verdict on an object the switch laid with no threshold. A writer
/// judges as a load monitor does: it writes its verdict through the
/// object the registry holds, where that object states a threshold. It
/// runs against the switch's moves, off and on. The registry releases
/// a verdict ahead of its swap, so the writer's verdict lands between
/// the two in some round, and the object laid with no threshold would
/// stand vetoed with no report left to judge it.
#[test]
fn the_overload_switch_leaves_no_verdict_a_report_set_inside_a_replacement() {
    /// Stops the writer where the case ends, a failed case too.
    struct Stop<'a>(&'a AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Relaxed);
        }
    }
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects =
        Arc::new(Objects::new(Arc::clone(&registry)).with_overload_thresholds(defaults));
    replica.write(facts(A, 1));
    let stopped = AtomicBool::new(false);
    thread::scope(|scope| {
        let _stop = Stop(&stopped);
        scope.spawn(|| {
            while !stopped.load(Relaxed) {
                let judged = registry
                    .get_by_url(A)
                    .filter(|object| object.metadata().overload.is_enabled());
                if let Some(object) = judged {
                    registry.set_worker_overloaded(&object, true);
                }
            }
        });
        for round in 0..400 {
            assert_eq!(switch(&replica, false), 1);
            let laid = replica.at(A);
            assert!(!laid.metadata().overload.is_enabled());
            assert!(!laid.is_overloaded(), "round {round}");
            assert_eq!(switch(&replica, true), 1);
        }
    });
}

/// The overload switch settles a status with the table behind each
/// lay. A health driver moved the standing object, and its telling
/// finds that object replaced and tells nothing: the object laid again
/// took the moved status, and the table holds its own. On a replica
/// that leads the table is told the object's status by the time the
/// call returns. One that does not lead writes no row, and its object
/// takes the status its row holds.
#[test]
fn the_overload_switch_settles_a_status_that_moved_ahead_of_its_lay() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    for leads in [true, false] {
        let mut replica = Replica::new();
        let registry = Arc::clone(replica.objects.registry());
        replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
        let row = replica.write(facts(A, 1));
        let before = replica.store.get(row).expect("the row");
        assert_eq!(before.status, WorkerStatus::Pending);
        // The driver's move, with no word to the table.
        replica.at(A).set_status(ObjectStatus::Ready);
        let epoch = replica.store.epoch();
        let store = replica.store.as_ref();
        let laid = replica
            .objects
            .state_overload_switch(store, false, &|| leads);
        assert_eq!(laid, 1);
        let after = replica.store.get(row).expect("the row");
        let object = replica.at(A).status();
        if leads {
            assert_eq!(after.status, WorkerStatus::Ready, "the table is told");
            assert_eq!(object, ObjectStatus::Ready);
        } else {
            assert_eq!(replica.store.epoch(), epoch, "no row is written");
            assert_eq!(after.status, WorkerStatus::Pending);
            assert_eq!(object, ObjectStatus::Pending, "its row's status");
        }
    }
}

/// The overload switch holds the holder's turn over its lays, so every
/// build reads one word. Its caller's word of the lead is asked behind
/// a lay, and finds the turn held. Behind the call the turn is free.
#[test]
fn the_overload_switch_holds_the_holders_turn_over_its_lays() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    replica.write(facts(A, 1));
    let (objects, store) = (&replica.objects, replica.store.as_ref());
    let turns = RefCell::new(Vec::new());
    let leads = || {
        turns.borrow_mut().push(objects.turn.is_locked());
        true
    };
    assert_eq!(objects.state_overload_switch(store, false, &leads), 1);
    let turns = turns.into_inner();
    assert!(!turns.is_empty(), "the lay asks for the lead");
    assert!(turns.iter().all(|&held| held), "inside the turn");
    assert!(!objects.turn.is_locked(), "the turn is let go");
}

/// The overload switch takes the holder's turn ahead of its word. While
/// another holds the turn the call waits there and the word stands, as
/// a read under that turn builds every object by one word.
#[test]
fn the_overload_switch_takes_the_holders_turn_ahead_of_its_word() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    replica.write(facts(A, 1));
    let (objects, store) = (&replica.objects, replica.store.as_ref());
    let (stated, state) = mpsc::channel();
    let turn = objects.turn.lock();
    thread::scope(|scope| {
        let call = scope.spawn(move || {
            let laid = objects.state_overload_switch(store, false, &|| true);
            let _ = stated.send(());
            laid
        });
        assert_eq!(
            state.recv_timeout(Duration::from_millis(200)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        assert!(objects.defaults_on.load(Relaxed), "the word waits");
        drop(turn);
        assert_eq!(call.join().expect("the call ends"), 1);
    });
    assert!(!objects.defaults_on.load(Relaxed), "the word moved");
}

/// A default that states one signal alone changes the rows that take
/// that signal, and no other: a row that names it keeps its object
/// under either word, whatever it leaves out of the other signal.
#[test]
fn the_overload_switch_leaves_a_row_that_takes_no_signal_of_the_default() {
    let count = Overload {
        kv_usage_permille: None,
        waiting_requests: Some(3),
    };
    let usage = Overload {
        kv_usage_permille: Some(650),
        waiting_requests: None,
    };
    let stating = |waiting_requests, token_usage| OverloadThresholds {
        waiting_requests,
        token_usage,
    };
    // Each default with the row that takes it and the row that names it.
    for (defaults, takes, names) in [
        (stating(Some(8), None), usage, count),
        (stating(None, Some(0.8)), count, usage),
    ] {
        let mut replica = Replica::new();
        let registry = Arc::clone(replica.objects.registry());
        replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
        replica.write(WorkerFacts {
            overload: takes,
            ..facts(A, 1)
        });
        replica.write(WorkerFacts {
            overload: names,
            ..facts(B, 1)
        });
        let (took, named) = (replica.at(A), replica.at(B));
        let stated = named.metadata().overload;
        for on in [false, true] {
            let laid = switch(&replica, on);
            assert_eq!(laid, 1, "{defaults:?} switched to {on}");
            assert!(!Arc::ptr_eq(&took, &replica.at(A)), "{defaults:?}");
            assert!(Arc::ptr_eq(&named, &replica.at(B)), "{defaults:?}");
            assert_eq!(replica.at(B).metadata().overload, stated);
        }
    }
}

/// A row that states the external runtime takes no default: its object
/// states the thresholds its row names and no other, a queue alone or a
/// usage alone, and in any spelling of the runtime's name. An engine's
/// row takes the default, one that names its runtime too. A gateway
/// registers an engine's worker with its default thresholds (its
/// registration workflow's `overload_defaults`) and an external worker
/// with none (`build_external_worker`), so until that worker's first
/// update its monitor judges it by nothing.
#[test]
fn an_external_rows_object_states_no_default_threshold() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    let of_kind = |url, kind: &str| WorkerFacts {
        engine_kind: kind.to_string(),
        ..facts(url, 1)
    };
    replica.write(of_kind(A, "external"));
    replica.write(WorkerFacts {
        overload: Overload {
            kv_usage_permille: None,
            waiting_requests: Some(3),
        },
        ..of_kind(B, "external")
    });
    replica.write(facts(C, 1));
    replica.write(WorkerFacts {
        overload: Overload {
            kv_usage_permille: Some(500),
            waiting_requests: None,
        },
        ..of_kind(D, "external")
    });
    replica.write(of_kind(E, "vllm"));
    replica.write(of_kind(F, "EXTERNAL"));

    let stated = |url| replica.at(url).metadata().overload;
    let queue = OverloadThresholds {
        waiting_requests: Some(3),
        token_usage: None,
    };
    let usage = OverloadThresholds {
        waiting_requests: None,
        token_usage: Some(0.5),
    };
    let none = OverloadThresholds::default();
    assert_eq!(stated(A), none);
    assert_eq!(stated(B), queue, "an external row states its own alone");
    assert_eq!(stated(D), usage, "and no default queue beside its usage");
    assert_eq!(stated(F), none, "in any spelling of the name");
    assert_eq!(stated(C), defaults, "an engine's row takes the default");
    assert_eq!(stated(E), defaults, "one that names its runtime too");
}

/// The first act: a row written through the holder.
#[test]
fn a_row_that_arrives_has_its_object_built() {
    let mut replica = Replica::new();
    assert!(replica.objects.is_empty());
    let first = replica.write(facts(A, 1));
    let second = replica.write(facts(B, 1));
    let registry = Arc::clone(replica.objects.registry());
    assert_eq!((replica.objects.len(), registry.len()), (2, 2));
    assert_eq!(registry.get_by_model("m").len(), 2);
    for (url, row) in [(A, first), (B, second)] {
        let id = replica.id(url);
        assert_eq!(replica.at(url).url(), url);
        assert_eq!(replica.objects.object(row), Some(id.clone()));
        let revision = replica.row(url).revision;
        assert_eq!(
            replica.objects.standing(&id),
            Some(RowRevision { row, revision })
        );
    }
    let told = [("registered", replica.id(A)), ("registered", replica.id(B))];
    assert_eq!(replica.told(), told);
}

#[test]
fn a_row_laid_behind_a_kept_credential_builds_an_object_that_holds_it() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    // A row that names no engine dials its key: the keeper works the
    // address out as the holder does.
    let keyed = WorkerFacts {
        engine: String::new(),
        ..facts(B, 1)
    };
    replica.register(keyed, Some(KEY_B));
    // A row no registration here stated takes none.
    replica.write(facts(C, 1));
    assert!(replica.holds_credential(A, Some(KEY_A)));
    assert!(replica.holds_credential(B, Some(KEY_B)));
    assert!(replica.holds_credential(C, None));
}

#[test]
fn a_holder_handed_no_keeper_builds_objects_without_a_credential() {
    let registry = Arc::new(WorkerRegistry::<Members>::new());
    let objects = Objects::new(Arc::clone(&registry));
    let store = MemoryStore::new();
    objects.upsert(&store, facts(A, 1), Registrant::Control);
    let laid = registry.get_by_url(A).expect("the row's object");
    assert!(laid.api_key().is_none());
    assert_eq!(objects.lay_again(&store, &facts(A, 1)), 0);
}

/// [`facts`] of a row whose signals ask for cache events.
fn asking(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        signals: crate::store::Signals {
            events: true,
            ..crate::store::Signals::default()
        },
        ..facts(key, incarnation)
    }
}

/// Whether each of `rows` holds a lane at rank 0 of `tables`.
fn lanes(tables: &KvTables, rows: &[WorkerId]) -> Vec<bool> {
    rows.iter().map(|row| tables.holds_lane(*row, 0)).collect()
}

/// One batch of `sequence` that stores a chain of blocks from the root,
/// one for each of `contents`, written through `row`'s lane at rank 0
/// as the body of an admitted batch writes it. A block is named by its
/// batch and its content, so no two blocks of a case read alike.
pub(crate) fn stored_chain(
    tables: &KvTables,
    row: WorkerId,
    sequence: u64,
    contents: &[u64],
) -> BatchOutcome {
    let blocks: Vec<_> = contents
        .iter()
        .map(|&content| kv_tables::StoredBlock {
            seq_hash: kv_tables::SequenceHash(1_000 * sequence + content),
            content_hash: kv_index::ContentHash(content),
        })
        .collect();
    let facts = BatchFacts {
        sequence,
        ..BatchFacts::default()
    };
    tables.admit_through(row, 0, facts, &mut |lane| {
        let mut map = std::mem::take(lane.map());
        let placed = lane.stored(&blocks, None, &mut map);
        *lane.map() = map;
        placed.expect("stored");
    })
}

#[test]
fn a_holder_handed_tables_keeps_a_lane_for_each_row_that_asks_for_events() {
    let tables = Arc::new(KvTables::default());
    let objects = Objects::new(Arc::new(WorkerRegistry::<Members>::new()))
        .with_kv_tables(Arc::clone(&tables));
    let store = MemoryStore::new();
    let write = |facts: WorkerFacts| {
        let change = objects.upsert(&store, facts, Registrant::Control);
        change.expect("the store takes the write").id()
    };
    let (a, b, quiet) = (write(asking(A, 1)), write(asking(B, 1)), write(facts(C, 1)));
    assert_eq!(lanes(&tables, &[a, b, quiet]), [true, true, false]);
    // A row stated again in place: one that stops asking gives its lane
    // up, one that asks takes one. No object is laid anew for either.
    write(facts(A, 1));
    write(asking(C, 1));
    assert_eq!(lanes(&tables, &[a, b, quiet]), [false, true, true]);
    // A replaced row's lane leaves with it, blocks and all, and its
    // next life under the same URL starts with an empty lane, unheard.
    assert_eq!(stored_chain(&tables, b, 1, &[1]), BatchOutcome::Applied);
    assert_eq!(tables.indexed_blocks(b, 0), 1);
    let again = write(asking(B, 2));
    assert_eq!(lanes(&tables, &[b, again]), [false, true]);
    assert_eq!(tables.indexed_blocks(again, 0), 0);
    assert_eq!(tables.cursor(again, 0), None, "unheard");
    // A removed row's lane leaves with it.
    let left = store.by_key(C).expect("the row");
    let removed = objects.remove(&store, left.id, left.revision);
    assert!(removed.is_some());
    assert_eq!(lanes(&tables, &[quiet, again]), [false, true]);
    // A row that stays keeps what was heard for it through a later
    // write that changes nothing of it.
    assert_eq!(stored_chain(&tables, again, 1, &[1]), BatchOutcome::Applied);
    write(asking(D, 1));
    assert_eq!(tables.indexed_blocks(again, 0), 1);
}

#[test]
fn a_holder_handed_no_tables_makes_no_lane() {
    let objects = Objects::new(Arc::new(WorkerRegistry::<Members>::new()));
    let store = MemoryStore::new();
    objects.upsert(&store, asking(A, 1), Registrant::Control);
    assert!(objects.kv_tables().is_none());
    assert!(
        objects.registry().get_by_url(A).is_some(),
        "the object is laid all the same"
    );
}

#[test]
fn each_rank_row_of_an_engine_holds_the_engines_credential() {
    let replica = Replica::new();
    replica.register(rank_facts(A, C, 0), Some(KEY_C));
    replica.register(rank_facts(B, C, 1), Some(KEY_C));
    assert!(replica.holds_credential(A, Some(KEY_C)));
    assert!(replica.holds_credential(B, Some(KEY_C)));
}

#[test]
fn a_rank_row_another_host_registered_takes_no_credential_until_it_is_stated_here() {
    let replica = Replica::new();
    replica.register(rank_facts(A, C, 0), Some(KEY_C));
    // The engine's other rank came through another host: no keep here.
    let other = replica.write(rank_facts(B, C, 1));
    assert!(replica.holds_credential(A, Some(KEY_C)));
    assert!(replica.holds_credential(B, None));

    // It is stated here too. A door writes nothing over equal facts,
    // and the acceptance says the address is to be laid again.
    let stated = replica.kept(&rank_facts(B, C, 1), Some(KEY_C));
    assert!(stated.accepted(other));
    assert_eq!(replica.lay_again(&rank_facts(B, C, 1)), 1);
    assert!(replica.holds_credential(B, Some(KEY_C)));
}

#[test]
fn a_registration_refreshed_behind_a_new_credential_is_laid_with_it() {
    let mut replica = Replica::new();
    let registered = facts(A, 1);
    replica.register(registered.clone(), Some(KEY_A));
    assert!(replica.holds_credential(A, Some(KEY_A)));
    let id = replica.id(A);
    replica.told();

    // The row is written again behind another credential, and the
    // holder's write builds its object while the registration is in
    // flight.
    let refreshed = replica.kept(&registered, Some(KEY_B));
    let row = replica.write(registered.clone());
    assert_eq!(replica.told(), [("replaced", id)]);
    assert!(replica.holds_credential(A, Some(KEY_B)));

    // The object already holds what its row takes.
    assert!(refreshed.accepted(row));
    assert_eq!(replica.lay_again(&registered), 0);
}

#[test]
fn a_removal_forgets_nothing_while_another_row_dials_the_address() {
    let replica = Replica::new();
    replica.register(rank_facts(A, C, 0), Some(KEY_C));
    replica.register(rank_facts(B, C, 1), Some(KEY_C));

    replica.remove(A);
    assert_eq!(replica.credentials.len(), 1);
    // The other rank's row is written again: it is bound still.
    replica.write(rank_facts(B, C, 1));
    assert!(replica.holds_credential(B, Some(KEY_C)));
    replica.remove(B);
    assert!(replica.credentials.is_empty());
}

#[test]
fn a_removal_keeps_what_a_registration_in_flight_stated() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    let next = replica.kept(&facts(A, 2), Some(KEY_B));
    replica.remove(A);
    // The removed row is forgotten. What the flight stated stays.
    assert_eq!(replica.credentials.len(), 1);
    let row = replica.write(facts(A, 2));
    if next.accepted(row) {
        replica.lay_again(&facts(A, 2));
    }
    assert!(replica.holds_credential(A, Some(KEY_B)));
}

#[test]
fn a_removal_after_a_registration_that_moved_nothing_forgets_the_credential() {
    // On either answer of the registration that moved nothing.
    for put_back in [true, false] {
        let replica = Replica::new();
        let row = replica.register(facts(A, 1), Some(KEY_A));

        // The same registration again: a door writes nothing over
        // equal facts, and nothing is to be laid again.
        let again = replica.kept(&facts(A, 1), Some(KEY_A));
        if put_back {
            assert!(!again.put_back());
        } else {
            assert!(!again.accepted(row));
        }

        replica.remove(A);
        assert!(replica.credentials.is_empty());
    }
}

#[test]
fn a_registration_refused_at_an_address_that_stands_leaves_the_standing_workers_credential() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));

    let before = replica.at(A);
    assert!(!replica.kept(&facts(A, 1), Some(KEY_B)).put_back());
    assert!(Arc::ptr_eq(&before, &replica.at(A)));

    replica.write(facts(A, 1));
    assert!(!Arc::ptr_eq(&before, &replica.at(A)));
    assert!(replica.holds_credential(A, Some(KEY_A)));
}

#[test]
fn a_registration_that_moved_no_row_behind_a_new_credential_is_laid_again_with_it() {
    let mut replica = Replica::new();
    let row = replica.register(facts(A, 1), Some(KEY_A));
    let id = replica.id(A);
    let standing = replica.objects.standing(&id);
    replica.told();

    // The same worker registered with another credential: a door
    // writes nothing over equal facts, so no row moves and no object
    // is built. The acceptance says the address is to be laid again.
    let again = replica.kept(&facts(A, 1), Some(KEY_B));
    assert!(again.accepted(row));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    // The holder lays the object again: one replacement under its id,
    // at the revision its row stands at.
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, Some(KEY_B)));
    assert_eq!(replica.told(), [("replaced", id.clone())]);
    assert_eq!(replica.objects.standing(&id), standing);

    // An object that holds what its row takes is not laid again.
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    assert!(replica.told().is_empty());

    // A registration that states no credential changes it too, and
    // the object is laid again without one.
    assert!(replica.kept(&facts(A, 1), None).accepted(row));
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, None));
}

#[test]
fn an_object_built_with_a_refused_credential_is_laid_again_with_what_stood() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));

    // The standing row is written again while a registration is in
    // flight: its object is built with the stated credential.
    let refused = replica.kept(&facts(A, 1), Some(KEY_B));
    replica.write(facts(A, 1));
    assert!(replica.holds_credential(A, Some(KEY_B)));

    // The put-back says so, and the holder lays the object again.
    assert!(refused.put_back());
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, Some(KEY_A)));
}

#[test]
fn a_row_built_while_a_first_registration_is_in_flight_loses_its_credential_at_a_put_back() {
    let replica = Replica::new();
    // The row came through another host while a registration here
    // stated a credential for its address and was then refused.
    let refused = replica.kept(&facts(A, 1), Some(KEY_A));
    replica.write(facts(A, 1));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    assert!(refused.put_back());
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, None));
    assert!(replica.credentials.is_empty());
}

#[test]
fn a_registration_refused_behind_a_rotation_not_yet_laid_leaves_the_rotated_credential() {
    let replica = Replica::new();
    let row = replica.register(facts(A, 1), Some(KEY_A));

    // The credential alone changes. It is accepted, and the holder
    // has not laid the address again yet.
    assert!(replica.kept(&facts(A, 1), Some(KEY_B)).accepted(row));
    // The next registration states the first credential again. The
    // holder lays the address meanwhile: the object already holds
    // what is stated, and the read marks the flight.
    let refused = replica.kept(&facts(A, 1), Some(KEY_A));
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    // Refused: the rotated credential stands again, and the put-back
    // says an object holds the refused one.
    if refused.put_back() {
        replica.lay_again(&facts(A, 1));
    }
    assert!(replica.holds_credential(A, Some(KEY_B)));
}

/// One registration that keeps its credential at a time: its write
/// runs inside the holder's keeping turn, and the turn is let go
/// behind it.
#[test]
fn a_keeping_write_runs_in_a_turn_of_its_own() {
    let replica = Replica::new();
    let objects = &replica.objects;
    let held = objects.keeping(A, None, || objects.keeping.is_locked());
    assert_eq!(held, Ok(true), "the write runs inside the turn");
    assert!(!objects.keeping.is_locked(), "let go behind the write");
}

/// A registration that keeps, whose door answers and names no row the
/// registration stands on, is put back. The keep stands ahead of the
/// write and inside the turn. Nothing is kept behind it, the door's
/// answer comes back whole, and the turn is let go.
#[test]
fn a_keeping_registration_whose_answer_names_no_row_is_put_back() {
    let replica = Replica::new();
    let (objects, credentials) = (&replica.objects, &replica.credentials);
    let stated = facts(A, 1);
    let wrote = objects.register_keeping(
        replica.store.as_ref(),
        Some((&stated, Some(KEY_A.to_owned()))),
        (A, None),
        || {
            assert!(
                objects.keeping.is_locked(),
                "the write runs inside the turn"
            );
            assert_eq!(credentials.len(), 1, "kept ahead of the write");
            Ok(Vec::new())
        },
    );
    assert_eq!(wrote, Ok(Ok(Vec::new())));
    assert!(credentials.is_empty(), "put back");
    assert!(!objects.keeping.is_locked(), "let go behind the answer");
}

#[test]
fn laying_again_reaches_each_rank_row_that_dials_the_address_and_no_other() {
    let replica = Replica::new();
    let first = replica.register(rank_facts(A, C, 0), Some(KEY_C));
    replica.register(rank_facts(B, C, 1), Some(KEY_C));
    replica.register(facts(D, 1), Some(KEY_A));
    let other = replica.at(D);

    // One rank's registration states another credential: it is the
    // engine's, for each rank row bound there.
    let rotated = replica.kept(&rank_facts(A, C, 0), Some(KEY_B));
    assert!(rotated.accepted(first));
    assert_eq!(replica.lay_again(&rank_facts(A, C, 0)), 2);
    assert!(replica.holds_credential(A, Some(KEY_B)));
    assert!(replica.holds_credential(B, Some(KEY_B)));
    assert!(Arc::ptr_eq(&other, &replica.at(D)));

    // An address no row dials lays nothing.
    assert_eq!(replica.lay_again(&facts(E, 1)), 0);
}

/// Laying again reads no store. A row a registration wrote around the
/// holder has no object for the call to lay, and the keeper forgets
/// it: its object, laid at its key's next refused write, takes no
/// credential. A row
/// written through the holder is laid again from the row its object
/// states, not from the store's.
#[test]
fn laying_again_reads_the_objects_and_no_store() {
    let replica = Replica::new();
    let token = replica.kept(&facts(A, 1), Some(KEY_A));
    let row = replica.store.upsert(facts(A, 1)).id();
    assert!(token.accepted(row));
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    assert!(replica.objects.is_empty(), "the call laid nothing");
    assert!(replica.credentials.is_empty(), "a row with no object");
    replica.lay_standing();
    assert!(replica.holds_credential(A, None));

    // Through the holder, then a rotation, then the store's row moved
    // around the holder: the call lays the row the object states.
    let row = replica.register(facts(B, 1), Some(KEY_A));
    assert!(replica.kept(&facts(B, 1), Some(KEY_B)).accepted(row));
    let late = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    replica.patch_around(B, late);
    assert_eq!(replica.lay_again(&facts(B, 1)), 1);
    assert!(replica.holds_credential(B, Some(KEY_B)));
    let standing = replica.objects.standing(&replica.id(B));
    assert_eq!(standing.map(|standing| standing.revision), Some(1));
}

/// A removal that lands ahead of its registration's answer finds no
/// bound row to forget. The answer then binds a row that left. It
/// says the address is to be laid again, and that call finds no object
/// for the row and has the keeper forget it.
#[test]
fn laying_again_forgets_a_row_bound_after_it_left() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    let token = replica.kept(&facts(A, 1), Some(KEY_A));
    let row = replica.write(facts(A, 1));
    let removed = replica.objects.remove(store, row, 1);
    assert!(removed.is_some(), "the holder's own removal");
    assert!(token.accepted(row), "a row not bound before");
    assert_eq!(replica.credentials.len(), 1, "bound behind its leaving");
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    assert!(replica.credentials.is_empty());
}

/// A write through the holder that takes a row out has the keeper
/// forget that row in the same call: a removal its
/// row, and a key's next life the old life's row.
#[test]
fn a_write_that_takes_a_row_out_forgets_it() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    let row = replica.register(facts(A, 1), Some(KEY_A));
    replica.register(facts(B, 1), Some(KEY_B));
    assert_eq!(replica.credentials.len(), 2);
    assert!(replica.objects.remove(store, row, 1).is_some());
    assert_eq!(replica.credentials.len(), 1, "the removed row's alone");
    // The other key's next life, stated with no keep.
    let next = replica
        .objects
        .upsert(store, facts(B, 2), Registrant::Control);
    assert!(matches!(next, Some(MembershipChange::Replaced { .. })));
    assert!(replica.credentials.is_empty(), "the old life's row");
    assert!(replica.holds_credential(B, None));
}

#[test]
fn a_registration_with_no_credential_clears_one_a_removed_worker_left() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));
    replica.remove(A);

    replica.register(facts(A, 1), None);
    assert!(replica.holds_credential(A, None));
}

/// A refused write's read and a cut laid whole forget, each what it
/// read. A registration decided over a row that left around the holder
/// is refused: its read ends the key's object, and the keeper drops
/// that row and asks about no other. So a row a registration bound and
/// the holder has not read yet keeps its credential, and its object is
/// built with it. A cut that drops a row ends its object, and the
/// keeper drops every row that left. A credential goes with its
/// address's last row.
#[test]
fn a_refused_write_and_a_cut_laid_whole_forget_the_rows_that_left() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    replica.register(facts(A, 1), Some(KEY_A));
    // A second registration, bound and not read by the holder yet.
    let token = replica.kept(&facts(B, 1), Some(KEY_B));
    assert!(token.accepted(store.upsert(facts(B, 1)).id()));
    // Another writer's removal, then the holder's own write of the key.
    replica.remove_around(A);
    let labelled = WorkerFacts {
        labels: BTreeMap::from([("tier".to_owned(), "gold".to_owned())]),
        ..facts(A, 1)
    };
    let refused = replica.objects.upsert(store, labelled, Registrant::Control);
    assert_eq!(refused, None, "decided over a row that left");
    assert_eq!(replica.credentials.len(), 1, "the row that left alone");
    replica.lay_standing();
    assert!(replica.holds_credential(B, Some(KEY_B)), "the unread row's");

    // A cut with no row.
    let cut = MemoryStore::new().checkpoint();
    assert!(replica.objects.restore_from(store, cut));
    assert!(replica.objects.is_empty(), "the cut dropped the row");
    assert!(replica.credentials.is_empty(), "behind the cut's read");
}

/// A refused write names the row it was decided over to the keeper,
/// whatever it decided: a change in place, a new width, a key's next
/// life, a removal. That row left around the holder, and the keeper
/// forgets it.
#[test]
fn a_refused_write_of_each_kind_forgets_the_row_it_was_decided_over() {
    type Refused = fn(&Replica, WorkerId) -> Option<MembershipChange>;
    fn lay(replica: &Replica, facts: WorkerFacts) -> Option<MembershipChange> {
        let store = replica.store.as_ref();
        replica.objects.upsert(store, facts, Registrant::Control)
    }
    let writes: [(&str, Refused); 4] = [
        ("a change in place", |replica, _| {
            lay(replica, labelled(A, 1, "gold"))
        }),
        ("a new width", |replica, _| {
            let wider = WorkerFacts {
                dp_size: 2,
                ..facts(A, 1)
            };
            lay(replica, wider)
        }),
        ("a key's next life", |replica, _| lay(replica, facts(A, 2))),
        ("a removal", |replica, row| {
            replica.objects.remove(replica.store.as_ref(), row, 1)
        }),
    ];
    for (name, write) in writes {
        let replica = Replica::new();
        let row = replica.register(facts(A, 1), Some(KEY_A));
        // Another writer's removal, around the holder.
        replica.remove_around(A);
        assert_eq!(write(&replica, row), None, "{name}: over a row that left");
        assert!(replica.credentials.is_empty(), "{name}");
    }
}

/// A write refused over a row another writer moved in place reads the
/// row and lays it. The row stands, so the keeper keeps what is bound
/// to it: the forgetting behind that read drops a row that left alone.
/// A refused removal keeps it too: its object is out by then, and the
/// keeper forgets only behind the store's word.
#[test]
fn a_refused_write_over_a_row_moved_in_place_keeps_its_credential() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    let row = replica.register(facts(A, 1), Some(KEY_A));
    // Another writer's patch, around the holder.
    let late = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    replica.patch_around(A, late);
    let refused = replica.objects.patch(store, row, labels("gold"), 1);
    assert_eq!(refused, None, "decided over a row that moved");
    assert_eq!(replica.credentials.len(), 1, "the row stands");
    assert!(replica.holds_credential(A, Some(KEY_A)));
    // What is bound still: an object laid later takes it.
    replica.patch_around(A, labels("silver"));
    let refused = replica.objects.patch(store, row, labels("gold"), 2);
    assert_eq!(refused, None, "moved once more");
    assert!(replica.holds_credential(A, Some(KEY_A)), "laid with it");
    // A removal decided over the object that read left, refused too.
    replica.patch_around(A, labels("bronze"));
    let refused = replica.objects.remove(store, row, 3);
    assert_eq!(refused, None, "the removal's row moved");
    assert_eq!(replica.credentials.len(), 1, "the row stands");
    assert!(
        replica.holds_credential(A, Some(KEY_A)),
        "laid anew with it"
    );
}

#[test]
fn the_row_is_answered_only_for_the_object_the_entry_was_written_for() {
    let replica = Replica::new();
    let row = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let id = replica.id(A);
    let laid = replica.at(A);
    let standing = RowRevision {
        row,
        revision: replica.row(A).revision,
    };
    assert_eq!(replica.objects.standing_of(&id, &laid), Some(standing));
    assert_eq!(replica.objects.standing_of(&replica.id(B), &laid), None);
    assert_eq!(
        replica.objects.standing_of(&id, &object(&replica.row(A))),
        None
    );

    replica.patch(A, labels("gold"));
    assert_eq!(replica.objects.standing_of(&id, &laid), None);
    assert_eq!(
        replica.objects.standing_of(&id, &replica.at(A)),
        Some(RowRevision {
            row,
            revision: replica.row(A).revision,
        })
    );
}

/// The second act.
#[test]
fn a_row_that_goes_has_its_object_dropped() {
    let mut replica = Replica::new();
    let row = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let id = replica.id(A);
    replica.told();
    replica.remove(A);
    let registry = replica.objects.registry();
    assert!(registry.get_by_url(A).is_none());
    assert_eq!((replica.objects.len(), registry.len()), (1, 1));
    assert!(!replica.objects.is_empty());
    assert_eq!(replica.objects.object(row), None);
    assert_eq!(replica.objects.standing(&id), None);
    assert_eq!(replica.told(), [("removed", id)]);
}

/// The third act, at each write of a row's facts: a patch, a refresh of
/// the registration, a new width. The id stands, the object is another,
/// and what ran on the old one runs on the new: never a removal and an
/// add.
#[test]
fn a_row_whose_facts_change_has_its_object_replaced_under_the_same_id() {
    let mut replica = Replica::new();
    let row = replica.write(facts(A, 1));
    let id = replica.id(A);
    replica.told();
    let wider = WorkerFacts {
        dp_size: 2,
        ..facts(A, 1)
    };
    let writes: [Write<'_>; 3] = [
        ("a patch", &|replica| replica.patch(A, labels("gold"))),
        ("a refresh", &|replica| {
            let (store, objects) = (replica.store.as_ref(), &replica.objects);
            let change = objects.upsert(store, facts(A, 1), Registrant::Control);
            assert_eq!(change, Some(MembershipChange::Relabelled { id: row }));
        }),
        ("a new width", &|replica| {
            let (store, objects) = (replica.store.as_ref(), &replica.objects);
            let change = objects.upsert(store, wider.clone(), Registrant::Control);
            assert_eq!(change, Some(MembershipChange::Resized { id: row }));
        }),
    ];
    let shown: [(Option<&str>, Option<usize>); 3] =
        [(Some("gold"), None), (None, None), (None, Some(2))];
    for (turn, ((name, write), shown)) in writes.into_iter().zip(shown).enumerate() {
        let before = replica.at(A);
        before.increment_load();
        write(&replica);
        let after = replica.at(A);
        assert!(!Arc::ptr_eq(&before, &after), "{name}: another object");
        assert_eq!(replica.id(A), id, "{name}: the same id");
        assert_eq!(after.load(), turn + 1, "{name}: the running state");
        assert_eq!(
            (replica.tier(A).as_deref(), after.dp_size()),
            shown,
            "{name}"
        );
        let revision = replica.row(A).revision;
        assert_eq!(
            replica.objects.standing(&id),
            Some(RowRevision { row, revision }),
            "{name}"
        );
        assert_eq!(replica.told(), [("replaced", id.clone())], "{name}");
    }
}

/// The holder's copy of the status a table holds, as a told status's
/// lay and a cut's call it.
#[test]
fn a_status_copied_from_the_table_makes_a_worker_available_only_while_ready() {
    let mut replica = Replica::new();
    replica.write(facts(A, 1));
    let row = replica.row(A);
    let copy = |replica: &Replica| replica.objects.copy_status(replica.store.as_ref(), row.id);
    let (id, worker) = (replica.id(A), replica.at(A));
    assert_eq!(worker.status(), ObjectStatus::Pending);
    assert!(!worker.is_available());
    assert_eq!(replica.told(), [("registered", id.clone())]);

    for (row_status, object_status, available) in [
        (WorkerStatus::Ready, ObjectStatus::Ready, true),
        (WorkerStatus::NotReady, ObjectStatus::NotReady, false),
    ] {
        assert!(replica
            .store
            .set_status(row.id, row_status, row.revision)
            .is_some());
        copy(&replica);
        assert!(Arc::ptr_eq(&worker, &replica.at(A)));
        assert_eq!(worker.status(), object_status);
        assert_eq!(worker.is_available(), available);
        // The registry's subscribers hear each change once.
        assert_eq!(replica.told(), [("status", id.clone())]);
    }

    for _ in 0..CircuitBreakerConfig::default().failure_threshold {
        worker.record_circuit_breaker_outcome(false);
    }
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    assert!(replica
        .objects
        .registry()
        .set_worker_overloaded(&worker, true));
    assert!(worker.set_stall(Some(StallReason::Unreachable)));
    assert!(replica
        .store
        .set_status(row.id, WorkerStatus::Ready, row.revision)
        .is_some());
    replica.told();
    copy(&replica);
    assert_eq!(worker.status(), ObjectStatus::Ready);
    assert_eq!(replica.told(), [("status", id)]);
    // A promotion comes back as any promotion does: the open breaker
    // goes on trial. The host's own vetoes stand.
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
    assert!(worker.is_overloaded());
    assert_eq!(worker.stall_reason(), Some(StallReason::Unreachable));
    assert!(!worker.is_available());
}

#[test]
fn a_status_at_a_new_row_revision_waits_for_the_new_object() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    replica.write(facts(A, 1));
    let row = replica.row(A);
    let before = replica.at(A);
    let moved = store.set_status(row.id, WorkerStatus::Ready, row.revision);
    assert!(moved.is_some(), "the status moved");
    replica.patch_around(A, labels("gold"));

    replica.objects.copy_status(store, row.id);
    assert_eq!(before.status(), ObjectStatus::Pending);
    replica.lay_standing();
    let after = replica.at(A);
    assert!(!Arc::ptr_eq(&before, &after));
    assert_eq!(after.status(), ObjectStatus::Pending, "a lay copies none");
    replica.objects.copy_status(store, row.id);
    assert_eq!(after.status(), ObjectStatus::Ready);
}

#[test]
fn a_status_change_leaves_an_object_the_holder_did_not_lay_alone() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    replica.write(facts(A, 1));
    let row = replica.row(A);
    assert!(replica
        .store
        .set_status(row.id, WorkerStatus::Ready, row.revision)
        .is_some());
    replica.objects.copy_status(store, row.id);
    let registry = replica.objects.registry();
    let foreign = object(&row);
    assert!(registry.replace(&replica.id(A), Arc::clone(&foreign)));
    assert_eq!(foreign.status(), ObjectStatus::Ready);

    assert!(replica
        .store
        .set_status(row.id, WorkerStatus::NotReady, row.revision)
        .is_some());
    replica.objects.copy_status(store, row.id);
    assert_eq!(foreign.status(), ObjectStatus::Ready);
}

#[test]
fn a_cut_laid_whole_lays_the_status_of_both_kept_and_replaced_objects() {
    let mut replica = Replica::new();
    replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let (a, b) = (replica.at(A), replica.at(B));
    let (id_a, id_b) = (replica.id(A), replica.id(B));
    assert!(!a.is_available() && !b.is_available());
    replica.told();

    let other = MemoryStore::new();
    other.upsert(facts(A, 1));
    other.upsert(labelled(B, 1, "gold"));
    for url in [A, B] {
        let row = other.by_key(url).expect("a row in the snapshot");
        assert!(other
            .set_status(row.id, WorkerStatus::Ready, row.revision)
            .is_some());
    }
    let cut = other.checkpoint();
    assert!(replica.objects.restore_from(replica.store.as_ref(), cut));

    assert!(Arc::ptr_eq(&a, &replica.at(A)));
    assert!(a.is_available());
    assert!(!Arc::ptr_eq(&b, &replica.at(B)));
    assert!(replica.at(B).is_available());
    // Each object's change is told, kept or laid again.
    let told = replica.told();
    for id in [&id_a, &id_b] {
        assert!(told.contains(&("status", id.clone())), "{told:?}");
    }

    let row = other.by_key(A).expect("a row in the snapshot");
    assert!(other
        .set_status(row.id, WorkerStatus::NotReady, row.revision)
        .is_some());
    let cut = other.checkpoint();
    assert!(replica.objects.restore_from(replica.store.as_ref(), cut));
    assert!(Arc::ptr_eq(&a, &replica.at(A)));
    assert!(!a.is_available());
    assert_eq!(replica.told(), [("status", id_a)]);
}

#[test]
fn a_new_incarnation_of_a_key_drops_the_old_object_and_builds_one_under_a_new_id() {
    let mut replica = Replica::new();
    let old = replica.write(facts(A, 1));
    let (old_id, before) = (replica.id(A), replica.at(A));
    before.increment_load();
    replica.told();
    let new = replica.write(facts(A, 2));
    assert_ne!(old, new);
    let new_id = replica.id(A);
    assert_ne!(old_id, new_id);
    assert_eq!(replica.at(A).load(), 0, "nothing of the old object runs on");
    assert_eq!(replica.objects.len(), 1);
    assert_eq!(replica.objects.object(old), None);
    assert_eq!(replica.objects.standing(&old_id), None);
    assert_eq!(replica.objects.object(new), Some(new_id.clone()));
    let row = replica
        .objects
        .standing(&new_id)
        .map(|standing| standing.row);
    assert_eq!(row, Some(new));
    assert_eq!(
        replica.told(),
        [("removed", old_id), ("registered", new_id)]
    );
}

/// A cut laid whole reads every row once: one write of each kind went
/// around the holder ahead of it. The objects of
/// the rows that went leave first, then each row that does not stand as
/// its object was built is laid again.
/// A row only a status write touched keeps the very object, and takes
/// the cut's status. A
/// registration refreshed with equal facts moved its revision and is
/// laid again. So is the row that left and came
/// back under its id and its revision with a new label: its facts moved.
#[test]
fn a_cut_laid_whole_drops_the_rows_that_went_and_lays_the_moved_rows_again() {
    let mut replica = Replica::new();
    let gone = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let replaced = replica.write(facts(C, 1));
    replica.write(facts(E, 1));
    let back = replica.write(facts(F, 1));
    replica.write(facts(G, 1));
    let (a, b, c) = (replica.id(A), replica.id(B), replica.id(C));
    let (e, f, g) = (replica.id(E), replica.id(F), replica.id(G));
    let stood = replica.at(E);
    replica.told();

    replica.remove_around(A);
    replica.patch_around(B, labels("gold"));
    let next = replica.store.upsert(facts(C, 2)).id();
    let arrived = replica.store.upsert(facts(D, 1)).id();
    replica.remove_around(F);
    assert_eq!(replica.store.upsert(labelled(F, 1, "gold")).id(), back);
    assert_eq!(replica.row(F).revision, 1, "the first revision again");
    // A status write moves neither E's facts nor its revision.
    let unmoved = replica.row(E);
    assert!(replica
        .store
        .set_status(unmoved.id, WorkerStatus::NotReady, unmoved.revision)
        .is_some());
    assert_eq!(replica.row(E).revision, unmoved.revision);
    // A registration refreshed with equal facts moves G's revision alone.
    let refreshed = replica.row(G);
    replica.store.upsert(facts(G, 1));
    assert_eq!(replica.row(G).facts, refreshed.facts);
    assert_ne!(replica.row(G).revision, refreshed.revision);
    let cut = replica.store.checkpoint();
    assert!(replica.objects.restore_from(&replica.store, cut));

    let registry = Arc::clone(replica.objects.registry());
    assert_eq!((replica.objects.len(), registry.len()), (6, 6));
    assert!(registry.get_by_url(A).is_none());
    assert_eq!(replica.objects.object(gone), None);
    assert_eq!(replica.objects.object(replaced), None);
    assert_eq!(replica.id(B), b);
    assert_eq!(replica.tier(B).as_deref(), Some("gold"));
    assert_eq!((replica.id(E), replica.id(F)), (e.clone(), f.clone()));
    assert_eq!(replica.tier(F).as_deref(), Some("gold"));
    // E stands as its object was built: the very object, and its entry
    // names the row's revision still. A cut's status is laid on it.
    assert!(Arc::ptr_eq(&stood, &replica.at(E)));
    assert_eq!(stood.status(), ObjectStatus::NotReady);
    let standing = replica.objects.standing(&e).expect("E's row");
    assert_eq!(standing.revision, replica.row(E).revision);
    // G's revision moved: an object in its place, built at that revision.
    assert_eq!(replica.id(G), g);
    let standing = replica.objects.standing(&g).expect("G's row");
    assert_eq!(standing.revision, replica.row(G).revision);
    let (c_next, d) = (replica.id(C), replica.id(D));
    assert_ne!(c_next, c);
    assert_eq!(replica.objects.object(next), Some(c_next.clone()));
    assert_eq!(replica.objects.object(arrived), Some(d.clone()));
    // The cut's one status that moved, E's, is told behind the lays.
    let mut told = replica.told();
    assert_eq!(told.pop(), Some(("status", e)));
    let left: HashSet<_> = told.iter().take(2).cloned().collect();
    assert_eq!(left, HashSet::from([("removed", a), ("removed", c)]));
    let laid: HashSet<_> = told.iter().skip(2).cloned().collect();
    let each = [
        ("replaced", b),
        ("registered", c_next),
        ("registered", d),
        ("replaced", f),
        ("replaced", g),
    ];
    assert_eq!((told.len(), laid), (7, HashSet::from(each)));
}

/// A store that runs a callback after one row read.
#[derive(Default)]
struct Late {
    store: Arc<MemoryStore>,
    after_get: Mutex<Option<AfterGet>>,
}

type AfterGet = Box<dyn FnOnce(&MemoryStore) + Send>;

impl WorkersStore for Late {
    fn upsert_by(&self, facts: WorkerFacts, maker: Registrant) -> MembershipChange {
        self.store.upsert_by(facts, maker)
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        self.store.patch(id, patch, revision)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        self.store.remove(id, revision)
    }

    fn set_status(
        &self,
        id: WorkerId,
        status: WorkerStatus,
        revision: u64,
    ) -> Option<MembershipChange> {
        self.store.set_status(id, status, revision)
    }

    fn get(&self, id: WorkerId) -> Option<WorkerRow> {
        let row = self.store.get(id);
        if let Some(after_get) = self.after_get.lock().take() {
            after_get(&self.store);
        }
        row
    }

    fn by_key(&self, key: &str) -> Option<WorkerRow> {
        self.store.by_key(key)
    }

    fn epoch(&self) -> u64 {
        self.store.epoch()
    }

    fn all_rows(&self) -> (u64, Vec<WorkerRow>) {
        self.store.all_rows()
    }
}

#[test]
fn an_adapter_write_waits_for_the_holders_status_read() {
    let registry = Arc::new(WorkerRegistry::<Members>::new());
    let objects = Arc::new(Objects::new(Arc::clone(&registry)));
    let store = Late::default();
    let row = objects.upsert(&store.store, facts(A, 1), Registrant::Control);
    let row = row.expect("the store takes the write").id();
    let id = objects.object(row).expect("the row's object");
    let worker = registry.get(&id).expect("a registered worker");
    let revision = store.get(row).expect("the worker's row").revision;
    assert!(store
        .set_status(row, WorkerStatus::Ready, revision)
        .is_some());
    objects.copy_status(&store, row);
    assert_eq!(worker.status(), ObjectStatus::Ready);

    let adapter = HealthAdapter::new(
        Arc::clone(&objects),
        Arc::clone(&store.store),
        Arc::new(SoleLead),
        SoleLead::TERM,
    );
    assert!(store
        .set_status(row, WorkerStatus::NotReady, revision)
        .is_some());
    let (reading, read) = mpsc::channel();
    let (attempting, attempt) = mpsc::channel();
    let (written, write) = mpsc::channel();
    *store.after_get.lock() = Some(Box::new(move |_| {
        reading.send(()).expect("the adapter waits for the read");
        attempt
            .recv_timeout(Duration::from_secs(5))
            .expect("the adapter starts its write");
        assert_eq!(
            write.recv_timeout(Duration::from_secs(1)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
    }));
    thread::scope(|scope| {
        let connected = scope.spawn(move || {
            read.recv_timeout(Duration::from_secs(5))
                .expect("the holder reads the row");
            attempting.send(()).expect("the holder waits for the write");
            let answer = adapter.connected(&id, worker.revision());
            let _ = written.send(());
            assert_eq!(answer, Some((ObjectStatus::NotReady, ObjectStatus::Ready)));
            assert_eq!(worker.status(), ObjectStatus::Ready);
        });
        objects.copy_status(&store, row);
        connected.join().expect("the adapter finished");
    });
    assert_eq!(
        store.get(row).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
    assert_eq!(
        registry.get_by_url(A).expect("the worker").status(),
        ObjectStatus::Ready
    );
}

/// The holder is the registry's one writer.
#[test]
fn an_object_another_writer_left_under_a_rows_url_gives_way_to_the_rows() {
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    let foreign: Arc<dyn Worker> = Arc::new(<BasicWorkerBuilder>::new(A).build());
    let id = registry.register(Arc::clone(&foreign)).expect("a free URL");
    let row = replica.write(facts(A, 1));
    assert!(!Arc::ptr_eq(&foreign, &replica.at(A)));
    assert_eq!(replica.id(A), id);
    assert_eq!(replica.objects.object(row), Some(id.clone()));
    let standing = replica.objects.standing(&id).map(|standing| standing.row);
    assert_eq!(standing, Some(row));
    assert_eq!(
        replica.told(),
        [("registered", id.clone()), ("replaced", id)]
    );
    assert_eq!(format!("{:?}", replica.objects), "Objects { held: 1, .. }");
}

/// Another store's cut can hold under one id and one revision a row of
/// other facts: the row reads as the one held, and its facts tell it
/// apart.
#[test]
fn a_store_that_took_another_historys_rows_has_its_moved_rows_laid_again() {
    let mut replica = Replica::new();
    let row = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    replica.write(facts(D, 1));
    let (a, b) = (replica.id(A), replica.id(B));
    let stood = replica.at(D);
    replica.told();

    let other = MemoryStore::new();
    other.upsert(WorkerFacts {
        labels: BTreeMap::from([("tier".to_string(), "gold".to_string())]),
        ..facts(A, 1)
    });
    other.upsert(facts(C, 1));
    other.upsert(facts(D, 1));
    let cut = other.checkpoint();
    let theirs = other.by_key(A).expect("a row");
    assert_eq!((theirs.id, theirs.revision), (row, replica.row(A).revision));
    assert!(replica.objects.restore_from(&replica.store, cut));

    assert_eq!(replica.id(A), a);
    assert_eq!(replica.tier(A).as_deref(), Some("gold"));
    assert!(replica.objects.registry().get_by_url(B).is_none());
    assert_eq!(replica.objects.len(), 3);
    // D states the same facts in the other history: the very object.
    assert!(Arc::ptr_eq(&stood, &replica.at(D)));
    let c = replica.id(C);
    assert_eq!(
        replica.told(),
        [("removed", b), ("replaced", a), ("registered", c)]
    );
}

thread_local! {
    /// The holder and the row a case watches.
    static WATCHED: RefCell<Option<(Arc<Objects<Seen>>, WorkerId)>> =
        const { RefCell::new(None) };
    /// What each publish read of them.
    static READ: RefCell<Vec<Option<u64>>> = const { RefCell::new(Vec::new()) };
}

/// A generation that reads, where the registry publishes it inside its
/// own write, the revision the holder says the watched row's object was
/// built from.
#[derive(Debug)]
struct Seen(Arc<[Arc<dyn Worker>]>);

impl Generation for Seen {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        WATCHED.with_borrow(|watched| {
            if let Some((objects, row)) = watched {
                let built = objects
                    .object(*row)
                    .and_then(|id| objects.standing(&id))
                    .map(|standing| standing.revision);
                READ.with_borrow_mut(|read| read.push(built));
            }
        });
        Self(members)
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.0
    }
}

/// An object's entry is written before the registry tells of the
/// object, at a build and at a replacement: what the registry publishes
/// inside its write already reads the row's revision of that write.
#[test]
fn an_objects_entry_stands_before_the_registry_tells_of_it() {
    let store = MemoryStore::new();
    let objects = Arc::new(Objects::<Seen>::new(Arc::new(WorkerRegistry::new())));
    let (row, built) = (crate::store::identity(A, 1), 1);
    WATCHED.set(Some((Arc::clone(&objects), row)));
    let written = objects.upsert(&store, facts(A, 1), Registrant::Control);
    assert_eq!(written, Some(MembershipChange::Registered { id: row }));
    let at_build = READ.take();
    assert!(!at_build.is_empty(), "a registration publishes");
    assert!(
        at_build.iter().all(|read| *read == Some(built)),
        "{at_build:?}"
    );

    assert!(objects.patch(&store, row, labels("gold"), built).is_some());
    let replaced = store.get(row).expect("a row").revision;
    assert_ne!(replaced, built);
    let at_replacement = READ.take();
    assert!(!at_replacement.is_empty(), "a replacement publishes");
    assert!(
        at_replacement.iter().all(|read| *read == Some(replaced)),
        "{at_replacement:?}"
    );
    WATCHED.set(None);
}

/// An engine's ranks are answered by their rows, asked of its rank 0
/// row. A rank no row holds is answered by none, a row above rank 0 is
/// answered none, and a row that is its own engine has its one rank.
#[test]
fn an_engines_ranks_are_answered_by_their_rows() {
    let replica = Replica::new();
    let lower = replica.write(rank_facts(A, C, 0));
    let upper = replica.write(rank_facts(B, C, 1));
    let alone = replica.write(facts(D, 1));
    let objects = &replica.objects;
    assert_eq!(objects.rank_row(lower, 0), Some(lower));
    assert_eq!(objects.rank_row(lower, 1), Some(upper));
    assert_eq!(objects.rank_row(lower, 2), None);
    for rank in 0..3 {
        assert_eq!(objects.rank_row(upper, rank), None);
    }
    assert_eq!(objects.rank_row(alone, 0), Some(alone));
    assert_eq!(objects.rank_row(alone, 1), None);
}

#[test]
fn a_rank_row_that_leaves_is_answered_no_longer() {
    let replica = Replica::new();
    let lower = replica.write(rank_facts(A, C, 0));
    let upper = replica.write(rank_facts(B, C, 1));
    assert_eq!(replica.objects.rank_row(lower, 1), Some(upper));

    replica.remove(B);
    assert_eq!(replica.objects.rank_row(lower, 1), None);
    assert_eq!(replica.objects.rank_row(upper, 1), None);
    assert_eq!(replica.objects.rank_row(lower, 0), Some(lower));
}

/// A row stands at the rank its facts state and at no other, and at
/// none once it left: what a late dispatch report is held against.
#[test]
fn a_row_stands_at_its_own_rank_until_it_leaves() {
    let replica = Replica::new();
    let lower = replica.write(rank_facts(A, C, 0));
    let upper = replica.write(rank_facts(B, C, 1));
    let objects = &replica.objects;
    assert!(objects.lists_rank(lower, 0));
    assert!(objects.lists_rank(upper, 1));
    assert!(!objects.lists_rank(lower, 1), "another row's rank");
    assert!(!objects.lists_rank(upper, 0), "another row's rank");

    replica.remove(B);
    assert!(!objects.lists_rank(upper, 1), "a row that left");
    assert!(objects.lists_rank(lower, 0));
}

/// A rank's key that comes back as a new incarnation is another row, and
/// the rank is answered by it. A row whose facts move to another rank is
/// answered under its new rank alone.
#[test]
fn a_rank_row_laid_again_is_answered_by_its_new_row() {
    let replica = Replica::new();
    let lower = replica.write(rank_facts(A, C, 0));
    let upper = replica.write(rank_facts(B, C, 1));

    let again = replica.write(WorkerFacts {
        incarnation: 2,
        ..rank_facts(B, C, 1)
    });
    assert_ne!(again, upper);
    assert_eq!(replica.objects.rank_row(lower, 1), Some(again));

    let moved = replica.write(WorkerFacts {
        incarnation: 2,
        dp_rank: 2,
        dp_size: 3,
        ..rank_facts(B, C, 1)
    });
    assert_eq!(moved, again);
    assert_eq!(replica.objects.rank_row(lower, 1), None);
    assert_eq!(replica.objects.rank_row(lower, 2), Some(again));
}
