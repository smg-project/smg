//! The cases of a generation's lists by role, which stand where the build
//! holds the engine over worker lists.
#![cfg(feature = "in-process")]

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::Barrier,
    thread,
    time::Instant,
};

use openai_protocol::model_card::ModelCard;
use smg_worker::{registry::WorkerRegistry, test_support::ready_worker};

use super::*;
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        Constraints, LentList, PolicyEngine, Request, RequestFacts, RequestMaker, WorkerId,
        WorkerLists,
    },
    store::{MemoryStore, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::Objects,
};

const MODEL: &str = "m";
const SINGLE_0: &str = "http://single-0:8000";
const SINGLE_1: &str = "http://single-1:8000";
const PREFILL: &str = "http://prefill-0:8000";
const DECODE: &str = "http://decode-0:8000";
const ENCODE: &str = "http://encode-0:8000";
const STRAY: &str = "http://stray-0:8000";

/// A holder with the store it follows.
struct Replica {
    store: MemoryStore,
    objects: Objects<Members>,
}

impl Replica {
    fn new() -> Self {
        Self {
            store: MemoryStore::new(),
            objects: Objects::new(Arc::new(WorkerRegistry::new())),
        }
    }

    /// Write the row at `url`, of `role` and with its image bit, and have
    /// the holder follow: the row's id.
    fn write(&self, url: &str, role: Role, vision: bool) -> WorkerId {
        let facts = WorkerFacts {
            key: url.to_string(),
            engine: url.to_string(),
            incarnation: 1,
            role,
            models: vec![MODEL.to_string()],
            dp_size: 1,
            vision,
            ..WorkerFacts::default()
        };
        let row = self.store.upsert(facts).id();
        self.objects.follow(&self.store);
        row
    }

    /// Remove the row at `url` and have the holder follow.
    fn remove(&self, url: &str) {
        let row = self.store.by_key(url).expect("a row");
        assert!(self.store.remove(row.id, row.revision).is_some());
        self.objects.follow(&self.store);
    }

    /// A worker another writer registers: the holder lays no row for it.
    fn register_stray(&self) {
        let stray = ready_worker(STRAY, MODEL);
        assert!(self.objects.registry().register(stray).is_some());
    }

    /// The model's generation, as the registry holds it.
    fn generation(&self) -> Arc<Members> {
        let registry = self.objects.registry();
        registry
            .model_generation(MODEL)
            .expect("the model's generation")
    }

    /// The address of each of `generation`'s workers of `role`, as the
    /// holder lends them, with the image bit lent with each.
    fn lent<'a>(&self, generation: &'a Members, role: Role) -> Vec<(&'a str, bool)> {
        let list = self.objects.lent(generation, role);
        let facts = list.facts.expect("the holder lends facts");
        assert_eq!(facts.len(), list.workers.len(), "one fact a worker");
        let urls = list.workers.iter().map(|worker| worker.url());
        urls.zip(facts.iter().map(|fact| fact.vision)).collect()
    }
}

fn engine() -> PolicyEngine {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 1;
    PolicyEngine::new(AppliedSpec::validate(&document).expect("the preset validates"))
}

/// A single request with the list its host hands for it.
struct Asks<'a>(Request<'a>, LentList<'a>);

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.0.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.0
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.1)
    }
}

/// The holder lends each role's workers of a generation in the registry's
/// order, with a cell of its own and with the image bit its rows state of
/// each. The engine's records of the list carry the bit.
#[test]
fn each_role_is_lent_in_the_registrys_order_with_its_rows_facts() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, false);
    replica.write(PREFILL, Role::Prefill, false);
    replica.write(SINGLE_1, Role::Single, true);
    replica.write(DECODE, Role::Decode, true);
    replica.write(ENCODE, Role::Encode, true);
    let generation = replica.generation();

    let singles = [(SINGLE_0, false), (SINGLE_1, true)];
    assert_eq!(replica.lent(&generation, Role::Single), singles);
    assert_eq!(replica.lent(&generation, Role::Prefill), [(PREFILL, false)]);
    assert_eq!(replica.lent(&generation, Role::Decode), [(DECODE, true)]);
    assert_eq!(replica.lent(&generation, Role::Encode), [(ENCODE, true)]);
    assert_eq!(replica.objects.unstated(), 0);

    let single = replica.objects.lent(&generation, Role::Single);
    let again = replica.objects.lent(&generation, Role::Single);
    assert!(Arc::ptr_eq(single.workers, again.workers), "cut once");
    assert!(ptr::eq(single.cell, again.cell), "one cell a role");
    let prefill = replica.objects.lent(&generation, Role::Prefill);
    assert!(!ptr::eq(single.cell, prefill.cell));

    let engine = engine();
    let records = engine
        .members_of(single)
        .records(engine.spec(), engine.clock());
    let take_images: Vec<bool> = records.fixed().iter().map(|row| row.vision).collect();
    assert_eq!(take_images, [false, true]);
}

/// The holder lends each worker with its row's id, and a decision over the
/// list names its targets by those ids and by their places in the list lent:
/// a retry that names a row is passed over that row's worker. No row's id
/// is zero, which would state none.
#[test]
fn a_decision_over_a_lent_list_names_its_targets_by_their_rows_ids() {
    let mut replica = Replica::new();
    replica.objects = replica.objects.with_status_followed();
    // A worker with no row stands first among the members and in no list.
    replica.register_stray();
    let first = replica.write(SINGLE_0, Role::Single, false);
    let second = replica.write(SINGLE_1, Role::Single, false);
    assert!(first != WorkerId(0) && second != WorkerId(0));
    // A decision takes a worker whose row is ready.
    for url in [SINGLE_0, SINGLE_1] {
        let row = replica.store.by_key(url).expect("a row");
        let ready = WorkerStatus::Ready;
        assert!(replica
            .store
            .set_status(row.id, ready, row.revision)
            .is_some());
    }
    replica.objects.follow(&replica.store);
    let generation = replica.generation();
    let list = replica.objects.lent(&generation, Role::Single);
    let facts = list.facts.expect("the holder lends facts");
    let stated: Vec<Option<WorkerId>> = facts.iter().map(|fact| fact.id).collect();
    assert_eq!(stated, [Some(first), Some(second)]);

    let engine = engine();
    let tried = [first];
    let retry = Request {
        id: b"request",
        facts: RequestFacts {
            model: MODEL,
            seed: 7,
            tried: &tried,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    let asks = Asks(retry, list);
    for _ in 0..4 {
        let answered = engine.decide_at(&asks, Instant::now).expect("an answer");
        let (whole, _) = answered.whole().expect("a whole decision");
        let pick = whole.decision.pick().expect("a pick");
        let targets: Vec<(WorkerId, u32)> = pick
            .targets
            .iter()
            .map(|target| (target.id, target.position))
            .collect();
        assert_eq!(targets, [(second, 1)], "the tried row's worker is passed");
    }
}

/// A join publishes another generation: its lists hold the new worker with
/// its row's facts, beside cells of their own. The generation before stands
/// as it was for who holds it.
#[test]
fn a_join_publishes_a_generation_that_lends_the_new_worker() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, false);
    let before = replica.generation();
    assert_eq!(replica.lent(&before, Role::Single), [(SINGLE_0, false)]);

    replica.write(SINGLE_1, Role::Single, true);
    let after = replica.generation();
    assert!(!Arc::ptr_eq(&before, &after), "another generation");
    let joined = [(SINGLE_0, false), (SINGLE_1, true)];
    assert_eq!(replica.lent(&after, Role::Single), joined);
    assert_eq!(replica.lent(&before, Role::Single), [(SINGLE_0, false)]);
    let cell = |generation: &Members| {
        let list = replica.objects.lent(generation, Role::Single);
        ptr::from_ref(list.cell)
    };
    assert_ne!(cell(&before), cell(&after), "a cell of its own");
    assert_eq!(replica.objects.unstated(), 0);
}

/// A leave publishes another generation, whose lists hold the rest.
#[test]
fn a_leave_publishes_a_generation_that_lends_the_rest() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, false);
    replica.write(SINGLE_1, Role::Single, true);
    replica.remove(SINGLE_0);
    let rest = replica.generation();
    assert_eq!(replica.lent(&rest, Role::Single), [(SINGLE_1, true)]);
    assert_eq!(replica.objects.unstated(), 0);
}

/// A row whose facts change has its object replaced, which publishes
/// another generation: its lists state the row's new facts. The generation
/// before keeps what its first ask read.
#[test]
fn a_replacement_publishes_a_generation_with_the_rows_new_facts() {
    let replica = Replica::new();
    let row = replica.write(SINGLE_0, Role::Single, false);
    let before = replica.generation();
    assert_eq!(replica.lent(&before, Role::Single), [(SINGLE_0, false)]);

    assert_eq!(replica.write(SINGLE_0, Role::Single, true), row);
    let after = replica.generation();
    assert!(!Arc::ptr_eq(&before, &after), "another generation");
    assert_eq!(replica.lent(&after, Role::Single), [(SINGLE_0, true)]);
    assert_eq!(replica.lent(&before, Role::Single), [(SINGLE_0, false)]);
    assert_eq!(replica.objects.unstated(), 0);
}

/// A generation another writer of the registry published, over the same
/// members, is lent as any other: the holder reads its rows at the first
/// ask, and no ask waits for a write of the holder's.
#[test]
fn a_generation_another_writer_published_is_lent_with_the_rows_facts() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, true);
    let before = replica.generation();
    let worker = Arc::clone(&before.members()[0]);
    let cards = vec![ModelCard::new(MODEL), ModelCard::new("another")];
    assert!(replica.objects.registry().refresh_models(&worker, cards));
    let after = replica.generation();
    assert!(!Arc::ptr_eq(&before, &after), "another generation");
    assert_eq!(replica.lent(&after, Role::Single), [(SINGLE_0, true)]);
    assert_eq!(replica.objects.unstated(), 0);
}

/// A worker another writer registered has no row with the holder: it is
/// left out of the lists and counted, so every worker lent states its
/// row's id.
#[test]
fn a_member_with_no_row_is_left_out_and_counted() {
    let replica = Replica::new();
    replica.register_stray();
    let row = replica.write(SINGLE_0, Role::Single, true);
    let generation = replica.generation();
    assert_eq!(generation.members().len(), 2);
    assert_eq!(replica.lent(&generation, Role::Single), [(SINGLE_0, true)]);
    assert_eq!(replica.objects.unstated(), 1);
    let list = replica.objects.lent(&generation, Role::Single);
    let facts = list.facts.expect("the holder lends facts");
    let ids: Vec<Option<WorkerId>> = facts.iter().map(|fact| fact.id).collect();
    assert_eq!(ids, [Some(row)]);
}

/// A generation first asked once its worker's row left reads no row for
/// that worker: it is lent without it, and the worker is counted. The
/// registry holds the next generation by then.
#[test]
fn a_generation_first_asked_once_its_worker_left_is_lent_without_it() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, true);
    replica.write(SINGLE_1, Role::Single, true);
    let before = replica.generation();
    replica.remove(SINGLE_0);
    assert_eq!(before.members().len(), 2);
    assert_eq!(replica.lent(&before, Role::Single), [(SINGLE_1, true)]);
    assert_eq!(replica.objects.unstated(), 1);
    let rest = replica.generation();
    assert_eq!(replica.lent(&rest, Role::Single), [(SINGLE_1, true)]);
    assert_eq!(replica.objects.unstated(), 1);
}

/// A generation first asked once its worker's object was replaced reads no
/// row's object for that worker: it is lent without it, and never with the
/// facts of the object that took its place. The worker is counted.
#[test]
fn a_generation_first_asked_once_its_worker_was_replaced_is_lent_without_it() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, false);
    let before = replica.generation();
    replica.write(SINGLE_0, Role::Single, true);
    assert_eq!(before.members().len(), 1);
    assert!(replica.lent(&before, Role::Single).is_empty());
    assert_eq!(replica.objects.unstated(), 1);
    let after = replica.generation();
    assert_eq!(replica.lent(&after, Role::Single), [(SINGLE_0, true)]);
    assert_eq!(replica.objects.unstated(), 1);
}

/// Asks of one generation that meet at its first ask cut its lists once:
/// each is lent the same list, and the holder's rows are read once.
#[test]
fn asks_that_meet_at_a_generations_first_ask_cut_its_lists_once() {
    let replica = Replica::new();
    replica.register_stray();
    replica.write(SINGLE_0, Role::Single, true);
    let generation = replica.generation();
    let start = Barrier::new(8);
    let ask = || {
        start.wait();
        Arc::clone(replica.objects.lent(&generation, Role::Single).workers)
    };
    let lists: Vec<_> = thread::scope(|scope| {
        let asks: Vec<_> = (0..8).map(|_| scope.spawn(ask)).collect();
        let lent = asks.into_iter().map(|ask| ask.join().expect("an ask"));
        lent.collect()
    });
    assert!(lists.iter().all(|list| Arc::ptr_eq(list, &lists[0])));
    // The worker with no row is left out, and counted once for each read
    // of the rows.
    assert_eq!(replica.objects.unstated(), 1, "one read of the rows");
}

/// A generation is cut with one answer a member. Answers of another count
/// are refused where the lists are cut, the generation stays uncut, and its
/// next ask cuts it.
#[test]
fn facts_of_another_count_than_the_members_are_refused() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, true);
    let generation = replica.generation();
    let short = catch_unwind(AssertUnwindSafe(|| {
        let _ = generation.lent(Role::Single, |_| Box::default());
    }));
    assert!(short.is_err(), "no answer for the one member");
    assert_eq!(replica.lent(&generation, Role::Single), [(SINGLE_0, true)]);
}

/// A model's last worker leaves: the registry holds no generation of the
/// model. A generation of no member lends empty lists.
#[test]
fn a_generation_of_no_member_lends_empty_lists() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, false);
    replica.remove(SINGLE_0);
    let registry = replica.objects.registry();
    assert!(registry.model_generation(MODEL).is_none());
    let empty = Members::of(Arc::new([]));
    assert!(replica.lent(&empty, Role::Single).is_empty());
    assert!(replica.lent(&empty, Role::Decode).is_empty());
    assert_eq!(replica.objects.unstated(), 0);
}

/// The cases of an encode request's columns, in a file of their own.
mod encode;
/// The cases of a pair's classes, in a file of their own.
mod pairs;
