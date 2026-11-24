//! The cases of the fill of a model's cells ahead of its requests: under an
//! engine handed its program's lists, the follow asks the lists of a word's
//! models and the engine fills their cells, off the path of a request, as
//! the engine's task runs it.

use std::{
    mem,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use parking_lot::Mutex;
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    registry::{Generation, WorkerRegistry},
    test_support::{no_health_check, ready_worker},
    worker::WorkerStatus,
};

use super::{handed, id, in_id_order, started, told, url, Follow, MODEL};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision_engine::{DecisionEngine, DecisionRefused, Request, RequestMaker, WorkerLists},
        list_cell::{LentList, ListCell, ModelLists},
        policy_engine::{
            single::tests::{hearing, request, Heard},
            PolicyEngine,
        },
        view::FixedFacts,
    },
    store::GatewayId,
};

/// A second model, of workers registered under it.
const OTHER: &str = "model-b";

/// A program's value of one generation of a model: the list its registry
/// published and a list the program derived from it, the published one
/// less its first worker, each with the cell the program keeps beside it.
#[derive(Debug)]
pub(in crate::engine::policy_engine) struct Celled {
    all: Arc<[Arc<dyn Worker>]>,
    cell: ListCell,
    rest: Arc<[Arc<dyn Worker>]>,
    rest_cell: ListCell,
}

impl Generation for Celled {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        let rest = members.iter().skip(1).cloned().collect();
        Self {
            all: members,
            cell: ListCell::default(),
            rest,
            rest_cell: ListCell::default(),
        }
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.all
    }
}

/// A program whose lists live in the worker crate's registry. It keeps the
/// models it was asked for, in the order asked. `short` makes it lend each
/// list with two facts, whatever the list's count. `derived` makes it hand
/// a model's derived list behind its published one. `panics` makes its own
/// code panic where it is asked for the model.
#[derive(Debug)]
pub(in crate::engine::policy_engine) struct Program {
    pub(in crate::engine::policy_engine) registry: Arc<WorkerRegistry<Celled>>,
    asked: Mutex<Vec<String>>,
    short: AtomicBool,
    derived: AtomicBool,
    panics: AtomicBool,
    two: [FixedFacts; 2],
}

impl Program {
    /// A program in whose registry the workers at `url(0)` to
    /// `url(workers - 1)` stand.
    pub(in crate::engine::policy_engine) fn of(workers: usize) -> Arc<Self> {
        let program = Self {
            registry: Arc::new(WorkerRegistry::new()),
            asked: Mutex::default(),
            short: AtomicBool::new(false),
            derived: AtomicBool::new(false),
            panics: AtomicBool::new(false),
            two: [FixedFacts::default(); 2],
        };
        for at in 0..workers {
            program.register(at);
        }
        Arc::new(program)
    }

    /// Registers a worker of the model at `url(at)`, which holds none.
    pub(in crate::engine::policy_engine) fn register(&self, at: usize) {
        self.register_under(at, MODEL);
    }

    /// Registers a worker of `model` at `url(at)`, which holds none.
    fn register_under(&self, at: usize, model: &str) {
        let registered = self.registry.register(ready_worker(&url(at), model));
        registered.expect("a free address");
    }

    /// Registers one worker of both models at `url(at)`, which holds none.
    fn register_both(&self, at: usize) {
        let worker = <BasicWorkerBuilder>::new(url(at))
            .models(vec![ModelCard::new(MODEL), ModelCard::new(OTHER)])
            .health_config(no_health_check())
            .build();
        worker.set_status(WorkerStatus::Ready);
        let registered = self.registry.register(Arc::new(worker));
        registered.expect("a free address");
    }

    /// Removes the worker at `url(at)`.
    fn remove(&self, at: usize) {
        let removed = self.registry.remove_by_url(&url(at));
        removed.expect("the worker was removed");
    }

    /// Replaces the object of the worker at `url(at)` under its id, by one
    /// that serves `model`.
    fn replace(&self, at: usize, model: &str) {
        let standing = self.registry.get_id_by_url(&url(at));
        let standing = standing.expect("a worker there");
        let next = ready_worker(&url(at), model);
        assert!(self.registry.replace(&standing, next));
    }

    /// The model's generation as the registry holds it now.
    pub(in crate::engine::policy_engine) fn published(&self) -> Arc<Celled> {
        let published = self.registry.model_generation(MODEL);
        published.expect("the model has workers")
    }

    /// `lists`' published list, lent with its cell: with no fact, or with
    /// two where the program lends short.
    fn lent<'a>(&'a self, lists: &'a Celled) -> LentList<'a> {
        let short = self.short.load(Ordering::Relaxed);
        LentList {
            workers: &lists.all,
            cell: &lists.cell,
            facts: short.then_some(&self.two[..]),
        }
    }

    /// `lists`' derived list, lent as the published one is.
    fn lent_rest<'a>(&'a self, lists: &'a Celled) -> LentList<'a> {
        LentList {
            workers: &lists.rest,
            cell: &lists.rest_cell,
            ..self.lent(lists)
        }
    }

    /// The models the program was asked for since the last call.
    pub(in crate::engine::policy_engine) fn asked(&self) -> Vec<String> {
        mem::take(&mut *self.asked.lock())
    }
}

impl ModelLists for Program {
    fn lists_of(&self, model: &str, hand: &mut dyn FnMut(LentList<'_>)) {
        self.asked.lock().push(model.to_owned());
        let panics = self.panics.load(Ordering::Relaxed) && model == MODEL;
        assert!(!panics, "the program's own code panics for the model");
        // The generation is taken out of the registry first: no guard of
        // the registry stands while a cell is filled.
        let Some(lists) = self.registry.model_generation(model) else {
            return;
        };
        hand(self.lent(&lists));
        if self.derived.load(Ordering::Relaxed) {
            hand(self.lent_rest(&lists));
        }
    }
}

/// A request that lends one list, as a program lends its model's.
struct Lends<'a>(Request<'a>, LentList<'a>);

impl RequestMaker for Lends<'_> {
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

/// An engine that keeps what its stages are told, handed `program`'s
/// registry and its lists, and the follow its task starts at an upkeep.
fn filling(program: &Arc<Program>) -> (PolicyEngine, Heard, Option<Follow>) {
    let (engine, heard) = hearing(false);
    engine.set_registry(handed(&program.registry));
    engine.set_lists(Arc::clone(program) as Arc<dyn ModelLists>);
    let follow = started(&engine);
    assert!(follow.is_some(), "a registry was handed");
    (engine, heard, follow)
}

/// Whether a request over `lists`' list, lent as the program lends it, is
/// answered at both of the engine's entries for a single request.
pub(in crate::engine::policy_engine) fn answered(
    engine: &PolicyEngine,
    program: &Program,
    lists: &Celled,
) -> [bool; 2] {
    let maker = Lends(request(), program.lent(lists));
    let now = Instant::now();
    let leased = engine.decide_under(GatewayId::LOCAL, &maker, now);
    let bare = engine.decide_without_lease(&maker, now);
    for refused in [leased.as_ref().err(), bare.as_ref().err()] {
        let faulted = matches!(refused, None | Some(DecisionRefused::Faulted));
        assert!(faulted, "an answer, or a fault");
    }
    [leased.is_ok(), bare.is_ok()]
}

/// The follow's start reads the registry whole and fills the cell of each
/// list that stands, and a word of a registration fills the cell of the
/// model's newer list: the program is asked for the word's model once, and
/// the engine makes the list's value there. A request over a list filled
/// so makes no value: it reads the cell. Lists handed a second time are
/// ignored: the program handed first is the one asked.
#[test]
fn the_start_and_each_word_of_a_registration_fill_a_models_cell_ahead_of_its_requests() {
    let program = Program::of(2);
    let (engine, _heard, mut follow) = filling(&program);
    let later = Program::of(0);
    engine.set_lists(Arc::clone(&later) as Arc<dyn ModelLists>);
    // Two workers of one model stand: the model is asked for once.
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 1);
    let standing = program.published();
    assert_eq!(answered(&engine, &program, &standing), [true; 2]);
    assert_eq!(engine.values_made(), 1, "the requests read the cell");

    // A registration publishes the model's next list, with an empty cell.
    program.register(2);
    let newer = program.published();
    assert!(!Arc::ptr_eq(&standing, &newer));
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 2, "the word filled the new cell");
    assert_eq!(answered(&engine, &program, &newer), [true; 2]);
    assert_eq!(engine.values_made(), 2, "the requests read the cell");
    assert_eq!(engine.upkeep_faults(), 0);
    assert!(later.asked().is_empty(), "the first hand-over stands");
}

/// A removal and a replacement publish the model's list again, and each
/// word fills the new list's cell. A replacement's two objects serve one
/// model: it is asked for once. Where the new object serves another model,
/// both models' lists were published again, and both are asked. The
/// removal of a model's last worker leaves no list: the program is asked
/// and hands nothing. A model that no word names is not asked again.
#[test]
fn a_removal_and_a_replacement_fill_the_cell_of_the_list_they_publish() {
    let program = Program::of(3);
    let (engine, _heard, mut follow) = filling(&program);
    assert_eq!((program.asked().len(), engine.values_made()), (1, 1));

    program.remove(0);
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 2);

    program.replace(1, MODEL);
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 3);
    let lists = program.published();
    assert_eq!(answered(&engine, &program, &lists), [true; 2]);
    assert_eq!(engine.values_made(), 3, "the requests read the cell");

    // The worker at `url(2)` serves the other model from here on: the
    // model's list lost it, and the other model's list came with it.
    program.replace(2, OTHER);
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL, OTHER]);
    assert_eq!(engine.values_made(), 5, "a value a model");

    program.remove(1);
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL], "the word's model alone");
    assert_eq!(engine.values_made(), 5, "no list stands: none was made");
    assert!(program.registry.model_generation(MODEL).is_none());
}

/// A worker that serves two models stands in a list of each: its word asks
/// both models' lists.
#[test]
fn a_word_of_a_worker_of_two_models_asks_both_models_lists() {
    let program = Program::of(1);
    let (engine, _heard, mut follow) = filling(&program);
    assert_eq!((program.asked().len(), engine.values_made()), (1, 1));

    program.register_both(1);
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL, OTHER]);
    assert_eq!(engine.values_made(), 3, "a value a model");
}

/// Words that stand ready together are heard first, and each of their
/// models' lists is asked once, behind the last of them, whatever the
/// order the models came in: a list published and replaced inside the run
/// is never filled, and a second model's list is filled beside the
/// first's.
#[test]
fn words_that_stand_ready_together_ask_each_models_lists_once() {
    let program = Program::of(1);
    let (engine, heard, mut follow) = filling(&program);
    assert_eq!((program.asked().len(), engine.values_made()), (1, 1));
    drop(told(&heard));

    // The model, the other model, the model again.
    program.register(1);
    program.register_under(2, OTHER);
    program.register(3);
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL, OTHER], "an ask a model");
    assert_eq!(engine.values_made(), 3, "each model's last list was filled");
    // Each word was told all the same.
    let joined: Vec<_> = (1..4).map(|at| ("joined", id(at))).collect();
    assert_eq!(told(&heard), joined);
}

/// The task's upkeep pays what the follow owes with words still standing:
/// a model that a word heard owes is asked there, and asked again behind
/// the last word, where the ask reads the cell the upkeep filled.
#[test]
fn the_upkeep_pays_what_is_owed_with_words_still_standing() {
    let program = Program::of(1);
    let (engine, _heard, mut follow) = filling(&program);
    assert_eq!((program.asked().len(), engine.values_made()), (1, 1));

    program.register(1);
    program.register(2);
    // The first of two words is heard: one still stands, so nothing is
    // asked behind it.
    let first = follow.as_mut().and_then(Follow::ready).expect("a word");
    engine.heard(follow.as_mut(), first);
    assert!(program.asked().is_empty(), "a word still stands");
    assert_eq!(engine.values_made(), 1);

    // The upkeep pays: the list that stands holds both workers.
    engine.follow_registry(&mut follow);
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 2);

    // The last word asks the model again, and makes no second value.
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 2, "the ask read the cell");
}

/// An engine handed a registry and no lists asks for no model: a word owes
/// no fill, and each list's first request fills its cell.
#[test]
fn an_engine_handed_no_lists_fills_no_cell_ahead() {
    let program = Program::of(2);
    let (engine, _heard) = hearing(false);
    engine.set_registry(handed(&program.registry));
    let mut follow = started(&engine).expect("a registry was handed");
    program.register(2);
    // The word is told with no payment behind it: it owes nothing.
    let word = follow.ready().expect("a word");
    assert!(engine.tells(&mut follow, word));
    assert!(follow.owed.is_empty());
    assert!(program.asked().is_empty());
    assert_eq!(engine.values_made(), 0);
    let lists = program.published();
    assert_eq!(answered(&engine, &program, &lists), [true; 2]);
    assert_eq!(engine.values_made(), 1, "the first request filled the cell");
}

/// Lists a program hands behind the task's first step are taken all the
/// same: at its next upkeep the follow owes every list that stands a
/// fill, once, and from then on each word owes its own.
#[test]
fn lists_handed_behind_the_follows_start_fill_the_standing_lists_at_the_next_upkeep() {
    let program = Program::of(2);
    let (engine, _heard) = hearing(false);
    engine.set_registry(handed(&program.registry));
    let mut follow = started(&engine);
    assert!(program.asked().is_empty());

    engine.set_lists(Arc::clone(&program) as Arc<dyn ModelLists>);
    engine.follow_registry(&mut follow);
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 1);
    // Owed once: the next upkeep asks for nothing.
    engine.follow_registry(&mut follow);
    assert!(program.asked().is_empty());

    program.register(2);
    engine.follow_so_far(&mut follow);
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 2);
}

/// A list lent with facts of another count than its workers is refused
/// where its value is made. The fill ahead holds the panic and counts it
/// with the task's faults, and it costs that list's fill alone: the
/// program's code goes on, and the list it hands next is filled. The
/// stages are told each join, and the follow owes no whole read. The
/// refused cell takes no value, so a decision through either entry over
/// the list lent so is answered a fault and counted with the engine's. The
/// list lent right is filled at the next word and decided.
#[test]
fn a_fill_refused_for_its_facts_is_held_and_counted_and_costs_the_fill_alone() {
    let program = Program::of(3);
    program.short.store(true, Ordering::Relaxed);
    program.derived.store(true, Ordering::Relaxed);
    let (engine, heard, mut follow) = filling(&program);
    // Two facts: refused over the three workers published, and right over
    // the two of the derived list, which is handed behind it.
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!((engine.upkeep_faults(), engine.values_made()), (1, 1));
    let joined = [("joined", id(0)), ("joined", id(1)), ("joined", id(2))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(joined.to_vec()));
    assert!(!follow.as_ref().expect("a follow").unread);

    let lists = program.published();
    assert_eq!(answered(&engine, &program, &lists), [false; 2]);
    assert_eq!(engine.faults(), 2, "one fault a call");
    assert_eq!(engine.values_made(), 1, "the refused cell took no value");

    // A word's fills are refused too, the four workers' and the derived
    // three's, and the word is told.
    program.register(3);
    engine.follow_so_far(&mut follow);
    assert_eq!((engine.upkeep_faults(), engine.values_made()), (3, 1));
    assert_eq!(told(&heard), [("joined", id(3))]);
    assert!(!follow.as_ref().expect("a follow").unread);

    // Lent right, the lists that stand are filled behind the next word.
    program.short.store(false, Ordering::Relaxed);
    program.register(4);
    engine.follow_so_far(&mut follow);
    assert_eq!((engine.upkeep_faults(), engine.values_made()), (3, 3));
    let lists = program.published();
    assert_eq!(answered(&engine, &program, &lists), [true; 2]);
    assert_eq!((engine.faults(), engine.values_made()), (2, 3));
}

/// A panic of the program's own code where it is asked for a model's lists
/// is held and counted with the task's faults, and costs that model's
/// fills alone: the other model owed with it is asked and filled, and the
/// stages are told each join.
#[test]
fn a_panic_of_the_programs_own_code_costs_its_models_fills_alone() {
    let program = Program::of(1);
    program.register_under(1, OTHER);
    program.panics.store(true, Ordering::Relaxed);
    let (engine, heard, follow) = filling(&program);
    assert_eq!(program.asked(), [MODEL, OTHER]);
    assert_eq!((engine.upkeep_faults(), engine.values_made()), (1, 1));
    let joined = [("joined", id(0)), ("joined", id(1))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(joined.to_vec()));
    assert!(!follow.as_ref().expect("a follow").unread);
}
