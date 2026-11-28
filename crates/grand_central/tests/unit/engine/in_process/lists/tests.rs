use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::Barrier,
    thread,
};

use smg_worker::{
    registry::{Generation, WorkerRegistry},
    test_support::{ready_worker, worker_with_model_aliases},
    worker::WorkerType,
};

use super::*;
use crate::document::spec::{AppliedSpec, PolicySpec};
// Named by the cases that run under debug assertions alone.
#[cfg(debug_assertions)]
use crate::engine::view::WorkerId;

impl PublishedList {
    /// Whether the cell holds a value: whether a decision or the
    /// engine's task asked it. A case's question: a program asks the
    /// engine for no value of a list.
    pub(crate) fn filled(&self) -> bool {
        self.cell.0.get().is_some()
    }
}

const MODEL: &str = "model-a";

/// The workers the test program says take images.
const TAKE_IMAGES: [&str; 2] = ["http://worker-1:8000", "http://worker-2:8000"];

/// What the test program knows of a worker beside its object.
fn stated(worker: &Arc<dyn Worker>) -> FixedFacts {
    FixedFacts {
        vision: TAKE_IMAGES.contains(&worker.url()),
        ..FixedFacts::default()
    }
}

/// A list of the program's with what the program knows of its workers.
type Stated = (Arc<[Arc<dyn Worker>]>, Box<[FixedFacts]>);

fn stated_list(workers: Arc<[Arc<dyn Worker>]>) -> Stated {
    let facts = workers.iter().map(stated).collect();
    (workers, facts)
}

/// A program's value of one generation: the model's shared list, its
/// decode workers as a list built at its first use, and beside each its
/// facts and its cell. The registry makes the value under its guard,
/// where no member is read: the shared list's facts are made at its
/// first lending.
#[derive(Debug)]
struct Lists {
    all: Arc<[Arc<dyn Worker>]>,
    all_facts: OnceLock<Box<[FixedFacts]>>,
    all_cell: ListCell,
    decode: OnceLock<Stated>,
    decode_cell: ListCell,
}

impl Generation for Lists {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            all: members,
            all_facts: OnceLock::new(),
            all_cell: ListCell::default(),
            decode: OnceLock::new(),
            decode_cell: ListCell::default(),
        }
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.all
    }
}

impl Lists {
    fn all(&self) -> LentList<'_> {
        let facts = || self.all.iter().map(stated).collect();
        LentList {
            workers: &self.all,
            cell: &self.all_cell,
            facts: Some(self.all_facts.get_or_init(facts)),
        }
    }

    /// The decode workers: a list built here, at its first use.
    fn decode(&self) -> LentList<'_> {
        let (workers, facts) = self.decode.get_or_init(|| {
            let all = self.all.iter();
            let decode = all.filter(|worker| *worker.worker_type() == WorkerType::Decode);
            stated_list(decode.cloned().collect())
        });
        LentList {
            workers,
            cell: &self.decode_cell,
            facts: Some(facts),
        }
    }
}

/// A program whose lists live in the worker crate's registry.
#[derive(Debug)]
struct Program {
    registry: WorkerRegistry<Lists>,
}

impl Program {
    fn of(workers: &[Arc<dyn Worker>]) -> Self {
        let registry = WorkerRegistry::new();
        for worker in workers {
            let registered = registry.register(Arc::clone(worker));
            assert!(registered.is_some(), "a new worker registers");
        }
        Self { registry }
    }

    /// The model's generation as the registry holds it now.
    fn published(&self) -> Arc<Lists> {
        let published = self.registry.model_generation(MODEL);
        published.expect("the model has workers")
    }
}

impl ModelLists for Program {
    fn lists_of(&self, model: &str, hand: &mut dyn FnMut(LentList<'_>)) {
        let Some(lists) = self.registry.model_generation(model) else {
            return;
        };
        hand(lists.all());
        if lists.decode.get().is_some() {
            hand(lists.decode());
        }
    }
}

fn engine() -> InProcess {
    let document = PolicySpec::preset("round_robin").expect("a built-in preset");
    InProcess::new(AppliedSpec::validate(&document).expect("the preset validates"))
}

fn regular(at: usize) -> Arc<dyn Worker> {
    ready_worker(&format!("http://worker-{at}:8000"), MODEL)
}

/// The value a cell took first, where it holds one.
fn first_of(cell: &ListCell) -> Option<&PoolMembers> {
    cell.0.get().map(|made| &made.members)
}

/// Whether each worker takes images, as the list's records under
/// `engine` say it, by the worker's place.
fn take_images(engine: &InProcess, members: &PoolMembers) -> Vec<bool> {
    let records = members.records(engine.standing_spec(), engine.clock());
    records.fixed().iter().map(|row| row.vision).collect()
}

/// A list's cell is empty when the list is published. The first ask
/// fills it with the engine's value of that list, and every later ask
/// reads the same value. Where several threads ask an empty cell at
/// once, one value is made. The cell is two words beside the list.
#[test]
fn a_cell_is_filled_once_beside_its_own_list() {
    let (engine, program) = (engine(), Program::of(&[regular(0), regular(1), regular(2)]));
    let lists = program.published();
    assert!(first_of(&lists.all_cell).is_none(), "empty at the publish");
    assert!(size_of::<ListCell>() <= 2 * size_of::<usize>());

    let together = Barrier::new(4);
    let asked: Vec<&PoolMembers> = thread::scope(|scope| {
        let ask = || {
            together.wait();
            engine.members_of(lists.all())
        };
        let asks: Vec<_> = (0..4).map(|_| scope.spawn(ask)).collect();
        let joined = asks.into_iter().map(|ask| ask.join());
        joined.map(|ask| ask.expect("an ask ends")).collect()
    });
    let first = asked[0];
    assert_eq!(engine.values_made(), 1, "one ask fills, the others wait");
    assert!(Arc::ptr_eq(first.workers(), &lists.all), "over the list");
    assert!(asked.iter().all(|members| ptr::eq(*members, first)));
    assert!(ptr::eq(engine.members_of(lists.all()), first));
    let filled = first_of(&lists.all_cell).expect("a filled cell");
    assert!(ptr::eq(filled, first), "the cell's value");
}

/// A membership change publishes another list, and its cell is empty:
/// the engine's value of the old list is not carried over. That value
/// stands for as long as a reader holds the old generation, and falls
/// with it: nothing of the engine holds the old list.
#[test]
fn a_membership_change_publishes_a_list_with_an_empty_cell() {
    let (engine, program) = (engine(), Program::of(&[regular(0), regular(1)]));
    let old = program.published();
    let old_list = Arc::downgrade(&old.all);
    assert_eq!(engine.members_of(old.all()).workers().len(), 2);

    let registered = program.registry.register(regular(2));
    assert!(registered.is_some(), "a new worker registers");
    let new = program.published();
    assert_eq!(new.all.len(), 3);
    assert!(
        first_of(&new.all_cell).is_none(),
        "the new list's cell is empty"
    );
    assert!(first_of(&old.all_cell).is_some(), "the old value stands");
    let members = engine.members_of(new.all());
    assert!(Arc::ptr_eq(members.workers(), &new.all));

    drop(old);
    assert!(old_list.upgrade().is_none(), "the old list fell whole");
}

/// A program hands each list of a model that it has published so far,
/// each with its own cell and its own facts: the model's shared list at
/// once, and a list it builds at its first use from then on. A model it
/// holds no worker of is handed nothing. The value asked of each handed
/// list fills that list's own cell.
#[test]
fn each_list_of_a_model_is_handed_with_its_own_cell() {
    let decode = worker_with_model_aliases("http://decode-0:8000", MODEL, &[], WorkerType::Decode);
    let (engine, program) = (engine(), Program::of(&[regular(0), regular(1), decode]));
    let lists = program.published();
    let handed = |model: &str| {
        let mut seen = Vec::new();
        program.lists_of(model, &mut |lent| {
            let facts = lent.facts.expect("the program states facts").len();
            assert_eq!(facts, lent.workers.len(), "one fact a worker");
            seen.push((Arc::clone(lent.workers), ptr::from_ref(lent.cell)));
        });
        seen
    };
    assert!(handed("another-model").is_empty());
    assert_eq!(handed(MODEL).len(), 1, "the decode list is not built yet");

    assert_eq!(lists.decode().workers.len(), 1);
    let both = handed(MODEL);
    assert_eq!(both.len(), 2);
    assert!(Arc::ptr_eq(&both[0].0, &lists.all));
    assert!(ptr::eq(both[0].1, &lists.all_cell));
    assert!(Arc::ptr_eq(&both[1].0, lists.decode().workers));
    assert!(ptr::eq(both[1].1, &lists.decode_cell));

    program.lists_of(MODEL, &mut |lent| {
        let members = engine.members_of(lent);
        assert!(Arc::ptr_eq(members.workers(), lent.workers));
    });
    let filled = |cell| Arc::clone(first_of(cell).expect("a filled cell").workers());
    assert!(Arc::ptr_eq(&filled(&lists.all_cell), &lists.all));
    assert!(Arc::ptr_eq(
        &filled(&lists.decode_cell),
        lists.decode().workers
    ));
}

/// The facts lent with a list go into the list's records where its cell
/// is filled, each worker's at its place: the workers the program names
/// take images and no other does. One list lent beside two cells keeps
/// each cell's own facts, and a list lent with none says of no worker
/// that it takes images. The facts are read at the fill alone: a list
/// lent again with none, beside its filled cell, answers as the cell
/// was filled.
#[test]
fn the_facts_lent_with_a_list_are_in_its_records_by_place() {
    let (engine, (list, facts)) = (engine(), stated_list((0..5).map(regular).collect()));
    let named = [false, true, true, false, false];
    let stated: Vec<bool> = facts.iter().map(|fact| fact.vision).collect();
    assert_eq!(stated, named, "the program names two workers of five");

    let (told_cell, silent_cell) = (ListCell::default(), ListCell::default());
    let told = LentList {
        workers: &list,
        cell: &told_cell,
        facts: Some(&facts),
    };
    let silent = LentList {
        cell: &silent_cell,
        facts: None,
        ..told
    };
    assert_eq!(take_images(&engine, engine.members_of(told)), named);
    assert_eq!(take_images(&engine, engine.members_of(silent)), [false; 5]);
    let again = engine.members_of(told);
    assert_eq!(take_images(&engine, again), named, "each cell its own");
    let unsaid = LentList {
        facts: None,
        ..told
    };
    let filled = engine.members_of(unsaid);
    assert_eq!(
        take_images(&engine, filled),
        named,
        "read at the fill alone"
    );
}

/// A cell asked by a second engine holds a value for each. The first
/// engine's value stands where it stood and is the one that engine
/// reads, and the second engine reads its own. Both are over the list
/// and of the facts lent with it. No two engines share a number, and
/// the cell is still two words.
#[test]
fn a_cell_holds_a_value_for_each_engine_that_asks_its_list() {
    let (one, two) = (engine(), engine());
    assert_ne!(one.applied(), two.applied(), "a number of its own");
    let (list, facts) = stated_list((0..5).map(regular).collect());
    let cell = ListCell::default();
    let lent = LentList {
        workers: &list,
        cell: &cell,
        facts: Some(&facts),
    };
    let (ones, twos) = (one.members_of(lent), two.members_of(lent));
    assert!(!ptr::eq(ones, twos), "a value for each engine");
    assert!(ptr::eq(first_of(&cell).expect("a filled cell"), ones));
    assert!(ptr::eq(one.members_of(lent), ones), "where it stood");
    assert!(ptr::eq(two.members_of(lent), twos));
    assert!(Arc::ptr_eq(ones.workers(), &list) && Arc::ptr_eq(twos.workers(), &list));
    assert_eq!(take_images(&one, ones), take_images(&two, twos));
    assert_eq!(take_images(&two, twos), [false, true, true, false, false]);
    assert!(size_of::<ListCell>() <= 2 * size_of::<usize>());
}

/// Engines that ask one empty cell at once each get one value: two
/// askers of one engine read the same value, and no two engines share
/// one.
#[test]
fn engines_that_ask_an_empty_cell_at_once_each_get_one_value() {
    let engines = [engine(), engine(), engine()];
    let (list, _) = stated_list((0..3).map(regular).collect());
    let cell = ListCell::default();
    let lent = LentList {
        workers: &list,
        cell: &cell,
        facts: None,
    };
    let together = Barrier::new(6);
    let asked: Vec<&PoolMembers> = thread::scope(|scope| {
        let ask = |engine: &'_ InProcess| {
            together.wait();
            engine.members_of(lent)
        };
        let twice = engines.iter().chain(&engines);
        let asks = twice.map(|engine| scope.spawn(move || ask(engine)));
        let asks: Vec<_> = asks.collect();
        let joined = asks.into_iter().map(|ask| ask.join());
        joined.map(|ask| ask.expect("an ask ends")).collect()
    });
    for (at, engine) in engines.iter().enumerate() {
        assert!(ptr::eq(asked[at], asked[at + 3]), "one value an engine");
        assert!(ptr::eq(engine.members_of(lent), asked[at]));
        assert_eq!(engine.values_made(), 1, "made once by its engine");
    }
    assert!(!ptr::eq(asked[0], asked[1]) && !ptr::eq(asked[1], asked[2]));
    assert!(!ptr::eq(asked[0], asked[2]), "no two engines share one");
}

/// A cell that many documents asked holds as many values, and they fall
/// one behind the other: a long chain falls on a small stack.
#[test]
fn a_cell_of_many_values_falls_on_a_small_stack() {
    let (engine, list): (_, Arc<[Arc<dyn Worker>]>) = (engine(), Arc::new([]));
    let mut cell = ListCell::default();
    for under in 0..20_000 {
        // A value with no records made: the fall reads none.
        let made = Box::new(Made {
            under,
            members: engine.pool_members(&list),
            next: cell,
        });
        cell = ListCell(OnceLock::from(made));
    }
    let small = thread::Builder::new().stack_size(64 * 1024);
    let fall = small.spawn(move || drop(cell)).expect("a thread starts");
    fall.join().expect("the values fall");
}

/// What the engine says of an ask that it refuses.
fn refused(engine: &InProcess, lent: LentList<'_>) -> &'static str {
    let asked = catch_unwind(AssertUnwindSafe(|| {
        let _ = engine.members_of(lent);
    }));
    let said = asked.expect_err("the ask is refused");
    said.downcast_ref::<&str>().copied().unwrap_or_default()
}

/// A list lent with fewer facts than its workers, with an empty slice of
/// them or with more, is refused where its value is made, in every
/// build, and its cell stays empty: the same list lent right is filled,
/// of its facts.
#[test]
fn facts_of_another_count_than_the_list_are_refused() {
    let (engine, (list, facts)) = (engine(), stated_list((0..3).map(regular).collect()));
    let more: Vec<FixedFacts> = facts.iter().chain(&facts[..1]).copied().collect();
    let cell = ListCell::default();
    let lent = |facts| LentList {
        workers: &list,
        cell: &cell,
        facts: Some(facts),
    };
    for wrong in [&facts[..0], &facts[..2], &more[..]] {
        assert!(refused(&engine, lent(wrong)).contains("one fact a worker"));
        assert!(first_of(&cell).is_none(), "the cell stays empty");
    }
    let members = engine.members_of(lent(&facts[..]));
    assert_eq!(take_images(&engine, members), [false, true, true]);
}

/// A cell lent with another list than the one that filled it is
/// refused at the ask, in a build with debug assertions: to the engine
/// whose value fills it, and to a second engine, whose own value would
/// stand behind that one and be of the other list. The cell takes no
/// value of the other list.
#[cfg(debug_assertions)]
#[test]
fn a_cell_lent_with_another_list_is_refused() {
    let (engine, second) = (engine(), engine());
    let (list, _) = stated_list((0..2).map(regular).collect());
    let (other, _) = stated_list((2..4).map(regular).collect());
    let cell = ListCell::default();
    let lent = LentList {
        workers: &list,
        cell: &cell,
        facts: None,
    };
    assert!(Arc::ptr_eq(engine.members_of(lent).workers(), &list));
    let wrong = LentList {
        workers: &other,
        ..lent
    };
    for asker in [&engine, &second] {
        assert!(refused(asker, wrong).contains("the list it stands beside"));
    }
    assert_eq!(second.values_made(), 0, "no value of the other list");
    assert!(Arc::ptr_eq(second.members_of(lent).workers(), &list));
}

/// The values a cell holds fall with it, those made behind the first
/// too: each holds its list, and the list falls where the cell does.
#[test]
fn the_values_behind_the_first_fall_with_their_cell() {
    let engines = [engine(), engine(), engine()];
    let list: Arc<[Arc<dyn Worker>]> = (0..2).map(regular).collect();
    let cell = ListCell::default();
    for engine in &engines {
        let lent = LentList {
            workers: &list,
            cell: &cell,
            facts: None,
        };
        assert!(Arc::ptr_eq(engine.members_of(lent).workers(), &list));
    }
    let fallen = Arc::downgrade(&list);
    drop(list);
    assert!(fallen.upgrade().is_some(), "the cell's values hold it");
    drop(cell);
    assert!(fallen.upgrade().is_none(), "every value fell");
}

/// A cell is formatted as how many values it holds and the number each
/// was made under, in the order they were made. No record of a worker
/// is printed, of a list of any size.
#[test]
fn a_cell_is_formatted_as_the_numbers_of_its_values() {
    let (one, two) = (engine(), engine());
    let list = PublishedList::new((0..4).map(regular).collect());
    let shown = |list: &PublishedList| format!("{:?}", list.lent().cell);
    assert_eq!(shown(&list), "ListCell { values: 0, under: [] }");
    let _ = (one.members_of(list.lent()), two.members_of(list.lent()));
    let (first, second) = (one.applied(), two.applied());
    let numbers = format!("ListCell {{ values: 2, under: [{first}, {second}] }}");
    assert_eq!(shown(&list), numbers);
}

/// A published list lends its list, its cell and its facts together,
/// the same three at every lending. One built with nothing stated lends
/// no fact. One built with facts asked its workers once each, in the
/// list's order. The facts are in the list's records by place, whether
/// a function stated them or the program made them in a pass of its
/// own.
#[test]
fn a_published_list_lends_its_cell_and_its_facts_with_it() {
    let engine = engine();
    let workers: Arc<[Arc<dyn Worker>]> = (0..5).map(regular).collect();
    let silent = PublishedList::new(Arc::clone(&workers));
    assert!(silent.lent().facts.is_none() && !silent.filled());
    assert!(Arc::ptr_eq(silent.workers(), &workers));
    let mut asked = Vec::new();
    let told = PublishedList::with_facts(Arc::clone(&workers), |worker| {
        asked.push(worker.url().to_owned());
        stated(worker)
    });
    let urls = workers.iter().map(|worker| worker.url().to_owned());
    assert_eq!(asked, urls.collect::<Vec<_>>(), "once a worker, in order");
    let facts = workers.iter().map(stated).collect();
    let made = PublishedList::stating(Arc::clone(&workers), facts);
    let named = [false, true, true, false, false];
    for list in [&told, &made] {
        let (first, again) = (list.lent(), list.lent());
        assert!(ptr::eq(first.cell, again.cell), "one cell");
        assert!(Arc::ptr_eq(first.workers, &workers));
        let at = |lent: LentList<'_>| lent.facts.map(<[FixedFacts]>::as_ptr);
        assert!(at(first).is_some() && at(first) == at(again));
        let members = engine.members_of(first);
        assert_eq!(take_images(&engine, members), named);
        assert!(list.filled() && ptr::eq(engine.members_of(again), members));
    }
    let unsaid = engine.members_of(silent.lent());
    assert_eq!(take_images(&engine, unsaid), [false; 5]);
}

/// Facts of another count than the list are refused where the list is
/// published with them, in every build.
#[test]
#[should_panic(expected = "one fact a worker")]
fn a_list_published_with_facts_of_another_count_is_refused() {
    let workers: Arc<[Arc<dyn Worker>]> = (0..3).map(regular).collect();
    let facts = workers.iter().take(2).map(stated).collect();
    let _ = PublishedList::stating(workers, facts);
}

/// A list's facts that state `ids` of its workers, in the list's order.
#[cfg(debug_assertions)]
fn with_ids(ids: &[u64]) -> Vec<FixedFacts> {
    let stated = |id: &u64| FixedFacts {
        id: Some(WorkerId(*id)),
        ..FixedFacts::default()
    };
    ids.iter().map(stated).collect()
}

/// What the engine says of each of `wrong`, facts lent with a list of
/// three workers, each beside a cell of its own: every one is refused
/// with those words, and its cell stays empty.
#[cfg(debug_assertions)]
fn refused_ids(wrong: &[Vec<FixedFacts>]) {
    let (engine, list): (_, Arc<[Arc<dyn Worker>]>) = (engine(), (0..3).map(regular).collect());
    for facts in wrong {
        let cell = ListCell::default();
        let lent = LentList {
            workers: &list,
            cell: &cell,
            facts: Some(facts.as_slice()),
        };
        let said = refused(&engine, lent);
        assert!(said.contains("for every worker or for none"), "{said}");
        assert!(first_of(&cell).is_none(), "the cell stays empty");
    }
}

/// A program states an id for every worker of a list or for none. A
/// list lent with an id for some of its workers is refused where its
/// value is made, in a build with debug assertions, and a zero states
/// none. A list of ids for every worker is filled, and its records
/// carry them by place. A list of zeros alone states none for every
/// worker: it is filled too, and its records carry no zero.
#[cfg(debug_assertions)]
#[test]
fn ids_stated_for_some_workers_of_a_list_are_refused() {
    let mut absent = with_ids(&[7, 5, 9]);
    absent[1].id = None;
    refused_ids(&[absent, with_ids(&[7, 0, 9])]);

    let engine = engine();
    let ids_of = |stated: &[u64]| {
        let workers = (0..3).map(regular).collect();
        let list = PublishedList::stating(workers, with_ids(stated).into());
        let members = engine.members_of(list.lent());
        let records = members.records(engine.standing_spec(), engine.clock());
        records.ids().to_vec()
    };
    assert_eq!(ids_of(&[7, 5, 9]), [WorkerId(7), WorkerId(5), WorkerId(9)]);
    assert!(ids_of(&[0, 0, 0]).iter().all(|id| *id != WorkerId(0)));
}

/// No id is stated twice in a list: a list lent with one id for two of
/// its workers is refused where its value is made, in a build with
/// debug assertions.
#[cfg(debug_assertions)]
#[test]
fn an_id_stated_twice_in_a_list_is_refused() {
    refused_ids(&[with_ids(&[7, 5, 7]), with_ids(&[4, 4, 4])]);
}
