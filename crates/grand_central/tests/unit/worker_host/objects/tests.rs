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

mod credentials;
mod event_lanes;
mod host_options;
mod reconciliation;

const A: &str = "http://a:8000";
const B: &str = "http://b:8000";
const C: &str = "http://c:8000";
const D: &str = "http://d:8000";
const E: &str = "http://e:8000";
const F: &str = "http://f:8000";
const G: &str = "http://g:8000";

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

    /// Have the holder lay each row written around it, by its pass
    /// over a cut's rows.
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
    /// decided over the object that stands.
    fn write(&self, facts: WorkerFacts) -> WorkerId {
        let store = self.store.as_ref();
        let change = self.objects.upsert(store, facts, Registrant::Control);
        change.expect("the store takes the write").id()
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
        engine::{
            Constraints, Decided, Demands, LendsLists, Request, RequestFacts, RequestMaker,
            WorkerLists,
        },
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
    }

    impl LendsLists for Asks<'_> {
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

/// The rows of an engine's ranks are answered whole, asked of its rank 0
/// row: its own and every other rank's. A row above rank 0 is answered
/// none, a row that is its own engine itself alone, and a rank's row
/// that left is in no answer.
#[test]
fn an_engines_rows_are_answered_whole_to_its_rank_0_row() {
    let replica = Replica::new();
    let lower = replica.write(rank_facts(A, C, 0));
    let upper = replica.write(rank_facts(B, C, 1));
    let alone = replica.write(facts(D, 1));
    let objects = &replica.objects;
    assert_eq!(objects.rank_rows(lower), [lower, upper]);
    assert!(objects.rank_rows(upper).is_empty());
    assert_eq!(objects.rank_rows(alone), [alone]);

    replica.remove(B);
    assert_eq!(objects.rank_rows(lower), [lower]);
    assert!(objects.rank_rows(upper).is_empty());
}

/// An engine's address that ends in a slash is the address its ranks
/// stand under, as its rows state it. A rank above 0 is answered by its
/// row, and by none once that row left.
#[test]
fn an_engines_ranks_stand_under_an_address_that_ends_in_a_slash() {
    const SLASHED: &str = "http://e:8000/";
    let replica = Replica::new();
    let lower = replica.write(rank_facts(A, SLASHED, 0));
    let upper = replica.write(rank_facts(B, SLASHED, 1));
    assert_eq!(replica.objects.rank_row(lower, 1), Some(upper));

    replica.remove(B);
    assert_eq!(replica.objects.rank_row(lower, 1), None, "the rank left");
    assert_eq!(replica.objects.rank_row(lower, 0), Some(lower));
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

/// A row that states a rank beside a width of one stands at the rank it
/// states. Its object holds no rank, so the answer is the row's own.
#[test]
fn a_row_of_width_one_stands_at_the_rank_it_states() {
    let replica = Replica::new();
    let row = replica.write(WorkerFacts {
        dp_rank: 3,
        dp_size: 1,
        ..facts(A, 1)
    });
    assert!(replica.objects.lists_rank(row, 3), "the rank it states");
    assert!(!replica.objects.lists_rank(row, 0), "its object's rank");
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
